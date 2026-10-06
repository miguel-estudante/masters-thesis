"""Find abstracts that contain inline entity markup."""

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, TypedDict


ENTITY_LABELS = {
    "anatomical_location",
    "animal",
    "bacteria",
    "biomedical_technique",
    "chemical",
    "ddf",
    "dietary_supplement",
    "drug",
    "food",
    "gene",
    "human",
    "microbiome",
    "statistical_technique",
}

ENTITY_TAG_PATTERN = re.compile(r"<\s*/?\s*([A-Za-z][A-Za-z0-9_-]*)\s*>")
THINK_PATTERN = re.compile(r"<\s*think\b", re.IGNORECASE)


class InlineEntity(TypedDict):
    start_idx: int
    end_idx: int
    location: str
    text_span: str
    label: str


def normalized_label(label: str) -> str:
    """Normalize tag spelling to the dataset's entity-label format."""
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def contains_inline_entity_tag(text: str) -> bool:
    """Return whether text contains at least one recognized entity tag."""
    return any(
        normalized_label(match.group(1)) in ENTITY_LABELS
        for match in ENTITY_TAG_PATTERN.finditer(text)
    )


def contains_thinking(text: str) -> bool:
    return bool(THINK_PATTERN.search(text))


def remove_inline_entity_tags(text: str) -> tuple[str, list[InlineEntity]]:
    """Remove entity tags and return entities indexed in the cleaned text."""
    cleaned_parts: list[str] = []
    inline_entities: list[InlineEntity] = []
    open_entities: list[tuple[str, int, str]] = []
    cursor = 0

    for match in ENTITY_TAG_PATTERN.finditer(text):
        raw_tag = text[match.start():match.end()]
        cleaned_parts.append(text[cursor:match.start()])
        tag = normalized_label(match.group(1))
        is_closing = raw_tag.lstrip().startswith("</")

        if tag in ENTITY_LABELS:
            if is_closing:
                for stack_index in range(len(open_entities) - 1, -1, -1):
                    open_tag, start_idx, label = open_entities[stack_index]
                    if open_tag == tag:
                        open_entities.pop(stack_index)
                        cleaned_text = "".join(cleaned_parts)
                        text_span = cleaned_text[start_idx:]
                        if text_span:
                            inline_entities.append(
                                {
                                    "start_idx": start_idx,
                                    "end_idx": len(cleaned_text),
                                    "location": "abstract",
                                    "text_span": text_span,
                                    "label": label,
                                }
                            )
                        break
            else:
                open_entities.append((tag, sum(len(part) for part in cleaned_parts), tag))
        else:
            cleaned_parts.append(raw_tag)

        cursor = match.end()

    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts), inline_entities


def update_entity_offsets(entities: list[dict[str, Any]], abstract: str) -> None:
    for entity in entities:
        text_span = entity.get("text_span")
        if not isinstance(text_span, str) or not text_span:
            continue

        text_span = ENTITY_TAG_PATTERN.sub(
            lambda match: match.group(0)
            if normalized_label(match.group(1)) not in ENTITY_LABELS
            else "",
            text_span,
        )
        entity["text_span"] = text_span
        start_idx = abstract.find(text_span)
        if start_idx >= 0:
            entity["start_idx"] = start_idx
            entity["end_idx"] = start_idx + len(text_span)


def get_abstract(example: dict[str, Any]) -> Any:
    metadata = example.get("metadata")
    return metadata.get("abstract") if isinstance(metadata, dict) else None


def repair_record(example: dict[str, Any]) -> bool:
    """Clean one record, returning whether an abstract was changed."""
    abstract = get_abstract(example)
    if not isinstance(abstract, str) or contains_thinking(abstract):
        return False

    cleaned_abstract, inline_entities = remove_inline_entity_tags(abstract)
    if not inline_entities and cleaned_abstract == abstract:
        return False

    existing_entities = example.get("entities")
    if not isinstance(existing_entities, list):
        existing_entities = []

    existing_keys = {
        (
            entity.get("start_idx"),
            entity.get("end_idx"),
            normalized_label(str(entity.get("label", ""))),
        )
        for entity in existing_entities
        if isinstance(entity, dict)
    }
    for entity in inline_entities:
        key = (entity["start_idx"], entity["end_idx"], entity["label"])
        if key not in existing_keys:
            existing_entities.append(entity)
            existing_keys.add(key)

    example["metadata"]["abstract"] = cleaned_abstract
    update_entity_offsets(existing_entities, cleaned_abstract)
    example["entities"] = existing_entities
    return True


def find_affected_ids(data: dict[str, Any]) -> list[str]:
    """Return IDs whose abstract contains inline entity markup."""
    affected_ids = []
    for example_id, example in data.items():
        if not isinstance(example, dict):
            continue
        abstract = get_abstract(example)
        if isinstance(abstract, str) and contains_inline_entity_tag(abstract):
            affected_ids.append(str(example_id))
    return affected_ids


def find_thinking_ids(data: dict[str, Any]) -> list[str]:
    return [
        str(example_id)
        for example_id, example in data.items()
        if isinstance(example, dict)
        and isinstance(get_abstract(example), str)
        and contains_thinking(get_abstract(example))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find examples whose abstract contains inline entity tags."
    )
    parser.add_argument("json_file", type=Path, help="Path to the dataset JSON file")
    parser.add_argument(
        "--repair-output",
        type=Path,
        help="Write a repaired copy, leaving abstracts containing <think> unchanged",
    )
    parser.add_argument(
        "--filter-output",
        type=Path,
        help="Write a copy excluding abstracts with entity tags or <think> content",
    )
    args = parser.parse_args()

    try:
        with args.json_file.open("r", encoding="utf-8") as input_file:
            data = json.load(input_file)
    except OSError as error:
        parser.error(f"could not open {args.json_file}: {error}")
    except json.JSONDecodeError as error:
        parser.error(f"invalid JSON in {args.json_file}: {error}")

    if not isinstance(data, dict):
        parser.error("expected the JSON root to be an object keyed by example ID")

    affected_ids = find_affected_ids(data)
    thinking_ids = find_thinking_ids(data)

    print(f"Total examples with inline entity tags: {len(affected_ids)}")
    print("Example IDs:")
    for example_id in affected_ids:
        print(example_id)

    print(f"\nTotal abstracts with thinking: {len(thinking_ids)}")
    print("Thinking example IDs:")
    for example_id in thinking_ids:
        print(example_id)

    if args.repair_output is not None:
        repaired_data = copy.deepcopy(data)
        repaired_ids = []
        for example_id, example in repaired_data.items():
            if isinstance(example, dict) and repair_record(example):
                repaired_ids.append(str(example_id))

        with args.repair_output.open("w", encoding="utf-8") as output_file:
            json.dump(repaired_data, output_file, ensure_ascii=False, indent=2)

        print(f"\nRepaired examples written: {len(repaired_ids)}")
        print(f"Repair output: {args.repair_output}")

    if args.filter_output is not None:
        excluded_ids = set(affected_ids) | set(thinking_ids)
        filtered_data = {
            example_id: example
            for example_id, example in data.items()
            if str(example_id) not in excluded_ids
        }
        with args.filter_output.open("w", encoding="utf-8") as output_file:
            json.dump(filtered_data, output_file, ensure_ascii=False, indent=2)

        print(f"\nExcluded examples: {len(excluded_ids)}")
        print(f"Filtered examples written: {len(filtered_data)}")
        print(f"Filter output: {args.filter_output}")


if __name__ == "__main__":
    main()
