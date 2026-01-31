# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import gc
from importlib.metadata import version
from pathlib import Path
from typing import TypeVar

import torch
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import load_dataset, load_from_disk
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from optuna import Trial
from requests.exceptions import ConnectionError, Timeout
from requests import HTTPError
from rich.console import Console

from .config import DatasetSpecification, Settings
from .exceptions import DatasetConfigError, DatasetError, NetworkTimeoutError
from .logging import get_logger

logger = get_logger(__name__)

print = Console(highlight=False).print


class BatchSizeError(Exception):
    """Raised when batch size is too large for available GPU memory."""

    pass


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage.

    Returns:
        dict with keys: total_gb, used_gb, free_gb, utilization_pct
        Returns zeros if no GPU is available.
    """
    if not torch.cuda.is_available():
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "utilization_pct": 0}

    try:
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        free = total - reserved

        return {
            "total_gb": total / 1e9,
            "used_gb": allocated / 1e9,
            "free_gb": free / 1e9,
            "utilization_pct": (allocated / total) * 100 if total > 0 else 0,
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "utilization_pct": 0}


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def _parse_split_count(split: str) -> int | None:
    """Parse sample count from split string like 'train[:200]'.

    Args:
        split: Split specification (e.g., "train[:200]", "train[400:600]")

    Returns:
        Sample count if parseable, None if no count specified (e.g., "train")

    Examples:
        "train[:200]" -> 200
        "train[400:600]" -> 200 (600-400)
        "train[100:150]" -> 50
        "train" -> None
    """
    import re

    # Match patterns like [M:N] or [:N]
    match = re.search(r"\[(\d*):(\d+)\]", split)
    if match:
        start_str, end_str = match.groups()
        start = int(start_str) if start_str else 0
        end = int(end_str)
        return end - start

    # No slice notation - no count specified
    return None


def load_prompts(specification: DatasetSpecification) -> list[str]:
    """Load prompts from dataset, using streaming for large datasets like C4.

    For C4 dataset, uses streaming to avoid downloading 30-50GB of shards
    for small splits like train[:200]. Other datasets use standard download.

    Args:
        specification: Dataset configuration

    Returns:
        List of prompt strings

    Raises:
        ValueError: If C4 dataset is missing required config or sample count
        RuntimeError: If streaming fails due to network or other issues
    """
    logger.debug(
        "Loading prompts",
        dataset=specification.dataset,
        config=specification.config,
        split=specification.split,
        column=specification.column,
    )

    # Check if dataset is a local path
    dataset_path = Path(specification.dataset)
    if dataset_path.exists() and dataset_path.is_dir():
        # Local dataset saved with save_to_disk()
        logger.debug("Loading from local disk", path=str(dataset_path))
        dataset_dict = load_from_disk(specification.dataset)
        dataset = dataset_dict[specification.split]
        return list(dataset[specification.column])

    # HuggingFace Hub dataset
    # Detect C4 dataset (massive, benefits from streaming)
    is_c4 = "c4" in specification.dataset.lower()

    if is_c4:
        # C4 is ~800GB - use streaming to avoid downloading shards
        # Parse split to extract sample count (e.g., "train[:200]" -> 200)
        sample_count = _parse_split_count(specification.split)

        # FAIL LOUDLY: C4 requires explicit sample count
        if sample_count is None:
            raise ValueError(
                f"C4 dataset requires explicit sample count in split. "
                f"Got: '{specification.split}'. "
                f"Expected format: 'train[:N]' or 'train[M:N]'"
            )

        # FAIL LOUDLY: C4 requires config parameter
        if not specification.config:
            raise ValueError(
                "C4 dataset requires config parameter (e.g., 'en'). "
                "Use --unhelpfulness-prompts.config en"
            )

        # Extract base split name (e.g., "train[:200]" -> "train")
        # Streaming datasets don't support slice notation in split parameter
        import re

        base_split = re.sub(r"\[.*\]", "", specification.split)

        # Load as streaming dataset
        try:
            dataset = load_dataset(
                specification.dataset,
                specification.config,
                split=base_split,  # Use base split without slice notation
                streaming=True,  # KEY: Stream instead of download
            )
        except (ConnectionError, Timeout) as e:
            logger.error(
                "Network error streaming dataset",
                dataset=specification.dataset,
                config=specification.config,
                error=str(e),
            )
            print(f"[red]Network Error: Cannot stream dataset[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check your internet connection")
            print("  2. Try again in a few minutes")
            print("  3. Check HuggingFace Hub status: https://status.huggingface.co")
            raise NetworkTimeoutError(
                f"Network timeout while streaming dataset '{specification.dataset}'. "
                f"Check internet connection and try again."
            ) from e
        except ValueError as e:
            # Config parameter error
            error_msg = str(e).lower()
            if "config" in error_msg or "variant" in error_msg:
                print(f"[red]Dataset Config Error: {specification.dataset}[/]")
                print("[yellow]Solutions:[/]")
                print(f"  1. Add config parameter: --unhelpfulness-prompts.config en")
                print(f"  2. List available configs: from datasets import get_dataset_config_names")
                print(f"     get_dataset_config_names('{specification.dataset}')")
                raise DatasetConfigError(
                    f"Dataset '{specification.dataset}' requires a config parameter. "
                    f"Try: --unhelpfulness-prompts.config en"
                ) from e
            raise
        except DatasetNotFoundError as e:
            print(f"[red]Dataset Not Found: {specification.dataset}[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check dataset name spelling")
            print("  2. Search HuggingFace Hub: https://huggingface.co/datasets")
            raise DatasetError(
                f"Dataset '{specification.dataset}' not found. "
                f"Search: https://huggingface.co/datasets"
            ) from e
        except KeyboardInterrupt:
            raise

        # Take N samples from stream and materialize to list
        # This downloads only the data needed, not full shards
        prompts = []
        try:
            for i, example in enumerate(dataset):
                if i >= sample_count:
                    break
                prompts.append(example[specification.column])
        except (ConnectionError, Timeout) as e:
            print(f"[red]Network Error: Stream interrupted[/]")
            print(f"[yellow]Got {len(prompts)}/{sample_count} examples before failure[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check internet connection")
            print("  2. Try again - streaming will resume")
            raise NetworkTimeoutError(
                f"Network error while streaming examples (got {len(prompts)}/{sample_count}). "
                f"Check internet connection."
            ) from e
        except KeyError as e:
            print(f"[red]Column Not Found: {specification.column}[/]")
            print(f"[yellow]Dataset: {specification.dataset}[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check column name spelling")
            print(f"  2. List available columns: dataset.column_names")
            raise DatasetConfigError(
                f"Column '{specification.column}' not found in dataset '{specification.dataset}'. "
                f"Check column name or list available columns."
            ) from e
        except KeyboardInterrupt:
            raise

        # FAIL LOUDLY: Verify we got expected count
        if len(prompts) < sample_count:
            raise ValueError(
                f"C4 stream exhausted early: got {len(prompts)}, expected {sample_count}"
            )

        return prompts
    else:
        # Other datasets: use existing download behavior
        try:
            if specification.config:
                dataset = load_dataset(
                    specification.dataset, specification.config, split=specification.split
                )
            else:
                dataset = load_dataset(specification.dataset, split=specification.split)
        except (ConnectionError, Timeout) as e:
            print(f"[red]Network Error: Cannot download dataset[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check your internet connection")
            print("  2. Try again in a few minutes")
            print("  3. Check HuggingFace Hub status: https://status.huggingface.co")
            raise NetworkTimeoutError(
                f"Network timeout while downloading dataset '{specification.dataset}'. "
                f"Check internet connection and try again."
            ) from e
        except HTTPError as e:
            # Check for authentication or not found
            if e.response.status_code == 401:
                print(f"[red]Authentication Required: {specification.dataset}[/]")
                print("[yellow]Run: huggingface-cli login[/]")
                raise DatasetError(
                    f"Dataset '{specification.dataset}' requires authentication. "
                    f"Run 'huggingface-cli login' first."
                ) from e
            elif e.response.status_code == 404:
                print(f"[red]Dataset Not Found: {specification.dataset}[/]")
                print("[yellow]Search HuggingFace Hub: https://huggingface.co/datasets[/]")
                raise DatasetError(
                    f"Dataset '{specification.dataset}' not found. "
                    f"Check name or search: https://huggingface.co/datasets"
                ) from e
            raise
        except DatasetNotFoundError as e:
            print(f"[red]Dataset Config Not Found: {specification.config}[/]")
            print(f"[yellow]Dataset: {specification.dataset}[/]")
            print("[yellow]Solutions:[/]")
            print("  1. Check config name spelling")
            print(f"  2. List available configs: from datasets import get_dataset_config_names")
            print(f"     get_dataset_config_names('{specification.dataset}')")
            raise DatasetConfigError(
                f"Config '{specification.config}' not found for dataset '{specification.dataset}'. "
                f"Use get_dataset_config_names() to list available configs."
            ) from e
        except ValueError as e:
            # Check if split format error
            error_msg = str(e).lower()
            if "split" in error_msg:
                print(f"[red]Invalid Split Format: {specification.split}[/]")
                print("[yellow]Solutions:[/]")
                print("  1. Use valid split format: 'train', 'test', 'validation'")
                print("  2. For slices: 'train[:100]' or 'train[10:20]'")
                print("  3. For percentages: 'train[:10%]'")
                raise DatasetConfigError(
                    f"Invalid split format: '{specification.split}'. "
                    f"Use format like 'train', 'train[:100]', or 'train[:10%]'"
                ) from e
            elif "config" in error_msg:
                print(f"[red]Dataset Config Required: {specification.dataset}[/]")
                print("[yellow]Add config parameter (e.g., --unhelpfulness-prompts.config en)[/]")
                raise DatasetConfigError(
                    f"Dataset '{specification.dataset}' requires a config parameter. "
                    f"Try adding: --unhelpfulness-prompts.config <config_name>"
                ) from e
            raise
        except OSError as e:
            # Check if disk space error
            error_msg = str(e).lower()
            if "disk" in error_msg or "space" in error_msg or "storage" in error_msg:
                print(f"[red]Insufficient Disk Space[/]")
                print("[yellow]Solutions:[/]")
                print("  1. Free up disk space")
                print("  2. Clear HuggingFace cache: rm -rf ~/.cache/huggingface/datasets")
                print("  3. Use smaller dataset split (e.g., train[:100] instead of train[:1000])")
                raise DatasetError(
                    f"Insufficient disk space to download dataset '{specification.dataset}'. "
                    f"Free up space or use smaller split."
                ) from e
            raise
        except KeyboardInterrupt:
            raise

        try:
            return list(dataset[specification.column])
        except KeyError as e:
            print(f"[red]Column Not Found: {specification.column}[/]")
            print(f"[yellow]Dataset: {specification.dataset}[/]")
            print("[yellow]Available columns:[/]")
            print(f"  {list(dataset.column_names)}")
            raise DatasetConfigError(
                f"Column '{specification.column}' not found in dataset '{specification.dataset}'. "
                f"Available columns: {list(dataset.column_names)}"
            ) from e


T = TypeVar("T")


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def empty_cache():
    # Collecting garbage is not an idempotent operation, and to avoid OOM errors,
    # gc.collect() has to be called both before and after emptying the backend cache.
    # See https://github.com/p-e-w/heretic/pull/17 for details.
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()
    elif is_sdaa_available():
        torch.sdaa.empty_cache()
    elif is_musa_available():
        torch.musa.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()


def get_trial_parameters(trial: Trial) -> dict[str, str]:
    params = {}

    direction_index = trial.user_attrs["direction_index"]
    params["direction_index"] = (
        "per layer" if (direction_index is None) else f"{direction_index:.2f}"
    )

    for component, parameters in trial.user_attrs["parameters"].items():
        # parameters is already a dict (serialized for SQLite storage)
        for name, value in parameters.items():
            params[f"{component}.{name}"] = f"{value:.2f}"

    return params


def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[str],
) -> str:
    model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"

    return f"""# This is a decensored version of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version("heretic-llm")}

## Abliteration parameters

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in get_trial_parameters(trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
| **KL divergence** | {trial.user_attrs["kl_divergence"]:.2f} | 0 *(by definition)* |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(bad_prompts)} | {base_refusals}/{
        len(bad_prompts)
    } |

-----

"""
