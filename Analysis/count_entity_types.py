from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent  # repository root

DATASET_PATHS = {
    "gold": ROOT / "Annotations/Train/gold_quality/json_format/train_gold.json",
    "silver": ROOT / "Annotations/Train/silver_quality/json_format/train_silver.json",
    "silver2025": ROOT / "Annotations/Train/silver_quality/json_format/train_silver_2025.json",
    "bronze": ROOT / "Annotations/Train/bronze_quality/json_format/train_bronze.json",
    "dev": ROOT / "Annotations/Dev/json_format/dev.json",
    "augmented": ROOT / "SyntheticData" / "augmented_qwen14b_1200examples_1prompt_inline_filtered_metadata.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count how many entities of each type appear in the GutBrainIE annotation datasets."
    )
    for name in DATASET_PATHS:
        parser.add_argument(f"--{name}", action="store_true", help=f"Include the {name} dataset")
    parser.add_argument("--all", action="store_true", help="Include every dataset")
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also show combined entity counts summed across all selected datasets",
    )
    return parser.parse_args()


def count_entity_types(data: dict) -> Counter:
    counts: Counter = Counter()
    for document in data.values():
        for entity in document.get("entities", []):
            label = entity.get("label")
            if isinstance(label, str):
                counts[label] += 1
    return counts


def main() -> None:
    args = parse_args()

    selected = [name for name in DATASET_PATHS if args.all or getattr(args, name)]
    if not selected:
        print("No dataset selected. Pass one or more of: " + ", ".join(f"--{name}" for name in DATASET_PATHS) + ", or --all")
        return

    combined_counts: Counter = Counter()

    for name in selected:
        path = DATASET_PATHS[name]
        print(f"=== {name} ({path}) ===")

        if not path.exists():
            print("  File not found, skipping.\n")
            continue

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        counts = count_entity_types(data)
        total = sum(counts.values())
        combined_counts.update(counts)

        if not counts:
            print("  No entities found.\n")
            continue

        for label, count in counts.most_common():
            print(f"  {label}: {count}")
        print(f"  TOTAL: {total}\n")

    if args.combined:
        print(f"=== combined ({', '.join(selected)}) ===")
        if not combined_counts:
            print("  No entities found.\n")
            return

        for label, count in combined_counts.most_common():
            print(f"  {label}: {count}")
        print(f"  TOTAL: {sum(combined_counts.values())}")


if __name__ == "__main__":
    main()
