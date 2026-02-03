# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""Configuration verification and logging utilities.

This module provides functions to verify that configuration files are loaded
correctly and to log effective settings at startup, preventing silent failures
where TOML config is ignored and defaults are used instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .logging import get_logger
from .utils import print

if TYPE_CHECKING:
    from .config import Settings

logger = get_logger(__name__)

# Expected config file name
CONFIG_FILE_NAME = "config.toml"


def log_config_status() -> bool:
    """Check and log whether config.toml was found.

    This function has two purposes:
    1. Side effect: Print user-facing messages about config status
    2. Return value: Used by caller to conditionally run verify_config_was_parsed()

    Returns:
        True if config.toml exists in current directory, False otherwise.
        The return value is intentionally used by main.py to gate further verification.
    """
    config_path = Path.cwd() / CONFIG_FILE_NAME

    if config_path.exists():
        print(f"[green]Configuration loaded from:[/] {config_path}")
        logger.info(f"Config file loaded from {config_path}")
        return True
    else:
        print(f"[yellow]Warning: No {CONFIG_FILE_NAME} found in {Path.cwd()}[/yellow]")
        print(
            "[yellow]Using default settings. Copy config.default.toml to config.toml to customize.[/yellow]"
        )
        logger.warning(
            f"Config file not found at {config_path}, using defaults (cwd={Path.cwd()})"
        )
        return False


def log_effective_settings(settings: "Settings") -> None:
    """Log key effective settings to help users verify configuration.

    This prints critical settings that are commonly misconfigured or have
    important behavioral implications.

    Args:
        settings: The Settings instance with loaded configuration
    """
    print()
    print("[bold]Effective configuration:[/]")

    # Core settings - use str() to handle mock objects in tests
    print(f"  * Model: [bold]{str(settings.model)}[/]")
    print(f"  * Trials: {str(settings.n_trials)}")
    print(f"  * Study: {str(settings.study_name)}")

    # Feature flags - these are the ones that caused silent failures
    features = []
    if settings.use_mpoa:
        features.append("MPOA")
    if settings.use_concept_cones:
        features.append("Concept Cones")
    if settings.use_caa:
        features.append("CAA")
    if settings.use_activation_calibration:
        features.append("Activation Calibration")
    if settings.use_circuit_ablation:
        features.append("Circuit Ablation")
    if settings.use_sacred_directions:
        features.append("Sacred Directions")
    if settings.ensemble_probe_pca:
        features.append("Ensemble Probe+PCA")
    if settings.orthogonalize_directions:
        features.append("Helpfulness Orthog.")
    if settings.use_warm_start_params:
        features.append("Warm-Start")

    if features:
        print(f"  * Enabled features: {', '.join(features)}")
    else:
        print("  * Enabled features: [dim]None (basic ablation only)[/dim]")

    # Log disabled features that are commonly expected to be enabled
    disabled = []
    if not settings.use_mpoa:
        disabled.append("MPOA")
    if not settings.use_concept_cones:
        disabled.append("Concept Cones")
    if not settings.use_caa:
        disabled.append("CAA")

    if disabled:
        print(f"  * Disabled features: {', '.join(disabled)}")

    # Critical parameters that affect behavior
    if settings.use_activation_calibration:
        try:
            percentile = float(settings.activation_target_percentile)
            print(f"  * Activation calibration: percentile={percentile:.2f}")
        except (TypeError, ValueError):
            print(
                f"  * Activation calibration: percentile={settings.activation_target_percentile}"
            )

    if settings.use_pca_extraction:
        print(
            f"  * Direction extraction: PCA with {settings.n_refusal_directions} directions"
        )
        if settings.use_eigenvalue_weights:
            print(
                f"  * Direction weights: [bold]eigenvalue-computed[/bold] ({str(settings.eigenvalue_weight_method)})"
            )
        else:
            try:
                weights_str = ", ".join(
                    f"{float(w):.2f}" for w in settings.direction_weights
                )
                print(
                    f"  * Direction weights: [bold]user-specified[/bold] [{weights_str}]"
                )
            except (TypeError, ValueError):
                print(
                    f"  * Direction weights: [bold]user-specified[/bold] {settings.direction_weights}"
                )

    print()

    # Log to structured logger for debugging
    logger.info(
        f"Effective settings: model={settings.model}, n_trials={settings.n_trials}, "
        f"use_mpoa={settings.use_mpoa}, use_concept_cones={settings.use_concept_cones}, "
        f"use_caa={settings.use_caa}, use_activation_calibration={settings.use_activation_calibration}, "
        f"activation_target_percentile={settings.activation_target_percentile}, "
        f"use_eigenvalue_weights={settings.use_eigenvalue_weights}, "
        f"direction_weights={settings.direction_weights}, use_sacred_directions={settings.use_sacred_directions}"
    )


def verify_config_was_parsed(settings: "Settings") -> list[str]:
    """Verify that config.toml values were actually loaded, not just defaults.

    Compares key settings against their Field defaults to detect cases where
    the TOML file exists but wasn't properly parsed.

    Args:
        settings: The Settings instance to verify

    Returns:
        List of warning messages for settings that appear to use defaults
        despite a config file being present
    """
    warnings = []

    # Check if config.toml exists
    config_path = Path.cwd() / CONFIG_FILE_NAME
    if not config_path.exists():
        # No config file, so defaults are expected
        return warnings

    # Try to read the TOML file and check if key values were overridden
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            # Can't verify without TOML parser
            logger.debug(
                "TOML parser unavailable (tomllib/tomli not installed), skipping config verification"
            )
            return warnings

    try:
        with open(config_path, "rb") as f:
            toml_config = tomllib.load(f)
    except Exception as e:
        warnings.append(f"Config file exists but failed to parse: {e}. Using defaults.")
        return warnings

    # Check for settings that are in TOML but might not have been loaded
    # These are the ones that caused silent failures
    # Import defaults from Settings.model_fields to avoid drift between hardcoded values here
    # and actual Field defaults in config.py
    from .config import Settings

    fields = Settings.model_fields
    critical_settings = [
        ("use_mpoa", settings.use_mpoa, fields["use_mpoa"].default),
        (
            "use_concept_cones",
            settings.use_concept_cones,
            fields["use_concept_cones"].default,
        ),
        ("use_caa", settings.use_caa, fields["use_caa"].default),
        (
            "activation_target_percentile",
            settings.activation_target_percentile,
            fields["activation_target_percentile"].default,
        ),
    ]

    for setting_name, actual_value, default_value in critical_settings:
        # Check if setting is in TOML file
        if setting_name in toml_config:
            toml_value = toml_config[setting_name]
            # If TOML has a different value than default, but actual matches default,
            # the TOML wasn't loaded properly
            if toml_value != default_value and actual_value == default_value:
                warnings.append(
                    f"'{setting_name}' is {toml_value} in config.toml but "
                    f"using default {default_value}. CLI may have overridden."
                )

    return warnings


def print_config_summary(settings: "Settings") -> None:
    """Print a detailed configuration summary for --show-config flag.

    Args:
        settings: The Settings instance to display
    """
    print()
    print("[bold cyan]Bruno Configuration Summary[/bold cyan]")
    print("=" * 50)
    print()

    # Core settings
    print("[bold]Core Settings:[/bold]")
    print(f"  model: {settings.model}")
    print(f"  n_trials: {settings.n_trials}")
    print(f"  study_name: {settings.study_name}")
    print(f"  storage: {settings.storage}")
    print(f"  batch_size: {settings.batch_size} (0 = auto)")
    print(f"  max_batch_size: {settings.max_batch_size}")
    print()

    # Feature flags
    print("[bold]Feature Flags:[/bold]")
    print(f"  use_mpoa: {settings.use_mpoa}")
    print(f"  use_concept_cones: {settings.use_concept_cones}")
    print(f"  use_caa: {settings.use_caa}")
    print(f"  use_activation_calibration: {settings.use_activation_calibration}")
    print(f"  use_circuit_ablation: {settings.use_circuit_ablation}")
    print(f"  use_sacred_directions: {settings.use_sacred_directions}")
    print(f"  use_neural_refusal_detection: {settings.use_neural_refusal_detection}")
    print(f"  ensemble_probe_pca: {settings.ensemble_probe_pca}")
    print(f"  orthogonalize_directions: {settings.orthogonalize_directions}")
    print(f"  use_warm_start_params: {settings.use_warm_start_params}")
    print()

    # Direction extraction
    print("[bold]Direction Extraction:[/bold]")
    print(f"  use_pca_extraction: {settings.use_pca_extraction}")
    print(f"  n_refusal_directions: {settings.n_refusal_directions}")
    print(f"  use_eigenvalue_weights: {settings.use_eigenvalue_weights}")
    if settings.use_eigenvalue_weights:
        print(f"    eigenvalue_weight_method: {settings.eigenvalue_weight_method}")
    else:
        print(f"    direction_weights: {settings.direction_weights}")
    print()

    # Calibration
    if settings.use_activation_calibration:
        print("[bold]Activation Calibration:[/bold]")
        print(
            f"  activation_target_percentile: {settings.activation_target_percentile}"
        )
        print(
            f"  activation_calibration_layer_frac: {settings.activation_calibration_layer_frac}"
        )
        print(
            f"  activation_calibration_min_factor: {settings.activation_calibration_min_factor}"
        )
        print(
            f"  activation_calibration_max_factor: {settings.activation_calibration_max_factor}"
        )
        print()

    # MPOA settings
    if settings.use_mpoa:
        print("[bold]MPOA Settings:[/bold]")
        print(f"  mpoa_norm_mode: {settings.mpoa_norm_mode}")
        print(f"  mpoa_min_scale: {settings.mpoa_min_scale}")
        print(f"  mpoa_max_scale: {settings.mpoa_max_scale}")
        print()

    # Datasets
    print("[bold]Datasets:[/bold]")
    print(
        f"  good_prompts: {settings.good_prompts.dataset} [{settings.good_prompts.split}]"
    )
    print(
        f"  bad_prompts: {settings.bad_prompts.dataset} [{settings.bad_prompts.split}]"
    )
    if settings.orthogonalize_directions:
        print(f"  unhelpfulness_prompts: {settings.unhelpfulness_prompts.dataset}")
        if settings.unhelpfulness_prompts.config:
            print(f"    config: {settings.unhelpfulness_prompts.config}")
    print()

    print("=" * 50)
    print()
