from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an annotations JSON file down to a single entity type. "
            "Examples with no entities of that type are dropped, and "
            "relations/mention_level_relations/concept_level_relations are removed."
        )
    )
    parser.add_argument("input_file", type=Path, help="Annotations JSON file to filter (e.g. train_gold.json)")
    parser.add_argument("entity_type", help='Entity label to keep, e.g. "gene" (case-insensitive)')
    parser.add_argument(
        "output_file",
        nargs="?",
        type=Path,
        default=None,
        help="Output JSON file (default: <input_stem>_<entity_type>.json next to the input file)",
    )
    return parser.parse_args()


def filter_by_entity_type(data: dict, entity_type: str) -> dict:
    target = entity_type.lower()
    filtered: dict = {}

    for document_id, document in data.items():
        entities = document.get("entities", [])
        if not isinstance(entities, list):
            continue

        matching_entities = [
            entity for entity in entities
            if isinstance(entity.get("label"), str) and entity["label"].lower() == target
        ]
        if not matching_entities:
            continue

        filtered[document_id] = {
            "metadata": document.get("metadata", {}),
            "entities": matching_entities,
        }

    return filtered


def main() -> None:
    args = parse_args()
    output_file = args.output_file or args.input_file.with_name(
        f"{args.input_file.stem}_{args.entity_type.lower()}.json"
    )

    with args.input_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("The input JSON must contain an object of document records")

    filtered = filter_by_entity_type(data, args.entity_type)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(filtered, file, indent=2, ensure_ascii=False)

    print(f"Original examples: {len(data)}")
    print(f"Examples with '{args.entity_type}' entities: {len(filtered)}")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
