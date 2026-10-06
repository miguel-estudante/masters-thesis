#!/usr/bin/env python
# coding: utf-8

"""Convert train_gold.json into an abstract-only entity dataset.

The script keeps the article id as the top-level key and writes, for each
example, only the abstract text and the entities whose location is 'abstract'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repository root


DEFAULT_INPUT = ROOT / "Annotations/Train/gold_quality/json_format/train_gold.json"
DEFAULT_OUTPUT = ROOT / "Annotations/Train/gold_quality/json_format/train_gold_entities_only.json"


def convert_train_gold(input_path: Path, output_path: Path) -> None:
	with input_path.open("r", encoding="utf-8") as file:
		annotations = json.load(file)

	converted = {}
	for article_id, article_data in annotations.items():
		metadata = article_data.get("metadata", {})
		abstract = metadata.get("abstract", "")
		abstract_entities = [
			entity
			for entity in article_data.get("entities", [])
			if entity.get("location") == "abstract"
		]

		converted[article_id] = {
			"abstract": abstract,
			"entities": abstract_entities,
		}

	with output_path.open("w", encoding="utf-8") as file:
		json.dump(converted, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Create a JSON file that keeps only the abstract and abstract entities from train_gold.json."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=DEFAULT_INPUT,
		help=f"Input annotations file (default: {DEFAULT_INPUT.relative_to(ROOT).as_posix()})",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT,
		help=f"Output JSON file (default: {DEFAULT_OUTPUT.relative_to(ROOT).as_posix()})",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	convert_train_gold(args.input, args.output)
	print(f"Wrote {args.output}")


if __name__ == "__main__":
	main()
