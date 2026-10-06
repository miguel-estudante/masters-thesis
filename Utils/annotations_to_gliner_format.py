#!/usr/bin/env python
"""Convert annotations to the format used by GLiNER for fine-tuning.

Usage:
    python Utils/annotations_to_gliner_format.py <input.json> [output.json]

Without an output path, the result is written to Train/data/<input name>.json.
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repository root


def tokenize_text_with_positions(text):
    # Split text into tokens, preserving punctuation (except for hyphens and underscores)
    tokens = []
    token_spans = []  # list of (start_char_index, end_char_index) for each token
    pattern = re.compile(r"\w+|[.,!?;:\'\"()\[\]{}<>]|[\s]+|\S")
    for match in pattern.finditer(text):
        token = match.group()
        if token.isspace():
            continue  # Skip whitespace tokens
        start_pos = match.start()
        if re.match(r"\w+-\w+", token) or re.match(r"\w+_\w+", token):
            # Keep hyphenated or underscored words intact
            tokens.append(token)
            token_spans.append((start_pos, match.end()))
        else:
            # Split contractions (e.g., "don't" -> "don", "'", "t")
            contraction_match = re.match(r"(\w+)(')(\w+)", token)
            if contraction_match:
                groups = contraction_match.groups()
                for group in groups:
                    end_pos = start_pos + len(group)
                    tokens.append(group)
                    token_spans.append((start_pos, end_pos))
                    start_pos = end_pos
            else:
                tokens.append(token)
                token_spans.append((start_pos, match.end()))
    return tokens, token_spans

def process_annotations(data):
    output_data = []

    for doc_id, doc_data in data.items():
        overall_tokenized_text = []
        overall_ner = []
        token_offset = 0

        fields = ["title", "abstract"]

        for field in fields:
            text = doc_data["metadata"].get(field, "")
            tokens, token_spans = tokenize_text_with_positions(text)

            # Collect entities for this field
            field_entities = []
            for entity in doc_data.get("entities", []):
                mention_location = entity.get("location", "")
                if mention_location == field:
                    field_entities.append(entity)

            # Map entities from character indices to token indices
            for entity in field_entities:
                entity_start_char = entity["start_idx"]
                entity_end_char = entity["end_idx"] + 1  # Adjusting end index to be exclusive
                entity_label = entity["label"]

                entity_start_token_index = None
                entity_end_token_index = None

                for i, (token_start_char, token_end_char) in enumerate(token_spans):
                    if token_end_char <= entity_start_char:
                        continue  # Token is before the entity
                    if token_start_char >= entity_end_char:
                        break  # Token is after the entity
                    # Token overlaps with entity
                    if entity_start_token_index is None:
                        entity_start_token_index = i
                    entity_end_token_index = i  # Update to the last overlapping token

                if entity_start_token_index is not None and entity_end_token_index is not None:
                    overall_ner.append([
                        entity_start_token_index + token_offset,
                        entity_end_token_index + token_offset,
                        entity_label.lower()
                    ])
                else:
                    print(f"Warning: Could not find tokens for entity in doc {doc_id}, field {field}")

            # Append tokens to the overall tokenized text
            overall_tokenized_text.extend(tokens)
            token_offset += len(tokens)

        # Sort the word positions by the start index
        overall_ner.sort(key=lambda x: x[0])

        # Create the output dictionary for this document
        output_doc = {
            "tokenized_text": overall_tokenized_text,
            "ner": overall_ner
        }

        output_data.append(output_doc)

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Convert annotations to the GLiNER fine-tuning format.")
    parser.add_argument("input", type=Path, help="Annotations JSON file")
    parser.add_argument("output", type=Path, nargs="?", help="Output file (default: Train/data/<input name>.json)")
    args = parser.parse_args()

    output = args.output or ROOT / "Train/data" / args.input.name

    with open(args.input, 'r', encoding='utf-8') as file:
        data = json.load(file)

    with open(output, 'w', encoding='utf-8') as file:
        json.dump(process_annotations(data), file)

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
