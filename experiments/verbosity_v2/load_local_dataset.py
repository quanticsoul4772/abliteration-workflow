#!/usr/bin/env python3
"""Convert local JSON prompts to HuggingFace Dataset format.

This script creates a local HuggingFace dataset from JSON prompt files
so they can be used with heretic's standard dataset loading.

Usage:
    python load_local_dataset.py

This will create datasets in:
    - experiments/verbosity_v2/concise_dataset/
    - experiments/verbosity_v2/verbose_dataset/
"""

import json
from pathlib import Path

from datasets import Dataset


def create_dataset(json_path: str, output_dir: str):
    """Create a HuggingFace dataset from a JSON file of prompts."""
    with open(json_path, encoding="utf-8") as f:
        prompts = json.load(f)

    # Create dataset with 'text' column (matching heretic's expected format)
    dataset = Dataset.from_dict({"text": prompts})

    # Split into train/test
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)

    # Save to disk
    split_dataset.save_to_disk(output_dir)

    print(f"Created dataset at {output_dir}")
    print(f"  Train: {len(split_dataset['train'])} prompts")
    print(f"  Test: {len(split_dataset['test'])} prompts")

    return split_dataset


def main():
    base_dir = Path(__file__).parent

    print("=== Verbosity v2 Dataset Creation ===")
    print(
        "\nThis experiment uses the SAME questions with different verbosity instructions"
    )
    print("to isolate padding behavior from question complexity.\n")

    # Create concise prompts dataset (base questions - expect direct answers)
    print("Creating concise prompts dataset (base questions)...")
    create_dataset(
        base_dir / "concise_prompts.json",
        base_dir / "concise_dataset",
    )

    # Create verbose prompts dataset (same questions + "explain in detail")
    print("\nCreating verbose prompts dataset (with 'explain in detail')...")
    create_dataset(
        base_dir / "verbose_prompts.json",
        base_dir / "verbose_dataset",
    )

    print("\n" + "=" * 50)
    print("Datasets created successfully!")
    print("=" * 50)
    print("\nPrompt comparison (first 3):")

    with open(base_dir / "concise_prompts.json", encoding="utf-8") as f:
        concise = json.load(f)
    with open(base_dir / "verbose_prompts.json", encoding="utf-8") as f:
        verbose = json.load(f)

    for i in range(3):
        print(f"\n  Concise: {concise[i]}")
        print(f"  Verbose: {verbose[i]}")

    print("\n" + "=" * 50)
    print("Next steps:")
    print(
        "  1. Copy config: cp experiments/verbosity_v2/config.verbosity_v2.toml config.toml"
    )
    print(
        "  2. Run heretic: heretic --model Qwen/Qwen2.5-7B-Instruct --auto-select true"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
