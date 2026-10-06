from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent  # repository root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create entity-span frequency graphs from synthetic_data.json."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=ROOT / "SyntheticData" / "synthetic_data_deepseek.json",
        help="JSON file to analyze (default: SyntheticData/synthetic_data_deepseek.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Results" / "synthetic_entity_graphs_deepseek",
        help="Directory for generated graphs (default: Results/synthetic_entity_graphs_deepseek)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only graph the N most frequent spans per entity type.",
    )
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=ROOT / "Annotations/Train/gold_quality/json_format/train_gold.json",
        help="Gold-standard annotations used to validate spans (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="pdf",
        help="Output file format for the graphs (default: pdf, a vector format ideal for a thesis)",
    )
    return parser.parse_args()


def safe_filename(label: str) -> str:
    filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip().lower())
    return filename.strip("_") or "unknown"


def collect_gold_spans(data: dict) -> dict[str, set[str]]:
    """Map lowercased text span -> set of lowercased labels it is annotated with in gold."""
    gold: dict[str, set[str]] = defaultdict(set)

    for document in data.values():
        entities = document.get("entities", [])
        if not isinstance(entities, list):
            continue

        for entity in entities:
            label = entity.get("label")
            text_span = entity.get("text_span")
            if not isinstance(label, str) or not isinstance(text_span, str):
                continue
            gold[text_span.lower()].add(label.lower())

    return dict(gold)


def collect_span_counts(data: dict) -> dict[str, Counter[str]]:
    """Map label -> Counter of text spans, aggregated case-insensitively.

    The most frequent original casing for each span is kept as the display form.
    """
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    display_forms: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))

    for document_id, document in data.items():
        entities = document.get("entities", [])
        if not isinstance(entities, list):
            print(f"Skipping invalid entities list in {document_id}")
            continue

        for entity in entities:
            label = entity.get("label")
            text_span = entity.get("text_span")
            if not isinstance(label, str) or not isinstance(text_span, str):
                print(f"Skipping invalid entity in {document_id}: {entity}")
                continue
            span_key = text_span.lower()
            counts[label][span_key] += 1
            display_forms[label][span_key][text_span] += 1

    renamed_counts: dict[str, Counter[str]] = {}
    for label, span_counter in counts.items():
        renamed = Counter()
        for span_key, frequency in span_counter.items():
            display_span = display_forms[label][span_key].most_common(1)[0][0]
            renamed[display_span] = frequency
        renamed_counts[label] = renamed

    return renamed_counts


def create_graph(
    label: str,
    span_counts: Counter[str],
    output_dir: Path,
    top: int | None,
    gold_spans: dict[str, set[str]],
    file_format: str,
) -> None:
    ordered_spans = span_counts.most_common(top)
    ordered_spans.reverse()
    spans = [span for span, _ in ordered_spans]
    frequencies = [frequency for _, frequency in ordered_spans]

    label_key = label.lower()

    def bar_color(span: str) -> str:
        gold_labels = gold_spans.get(span.lower())
        if not gold_labels:
            return "#D9D9D9"  # gray: span not present in gold at all
        if label_key in gold_labels:
            return "#2E75B6"  # blue: span/label pair matches gold
        return "#E69F00"  # orange: span is present in gold but under a different label

    bar_colors = [bar_color(span) for span in spans]

    figure_height = max(4.0, 0.38 * len(spans) + 1.5)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    bars = axis.barh(spans, frequencies, color=bar_colors)
    axis.set_title(f"{label} entity spans")
    axis.set_xlabel("Occurrences")
    axis.set_ylabel("Text span")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor="#2E75B6", label="Matches gold label"),
        Patch(facecolor="#E69F00", label="In gold under a different label"),
        Patch(facecolor="#D9D9D9", label="Not present in gold"),
    ]
    axis.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    for bar, frequency in zip(bars, frequencies):
        axis.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(frequency),
            va="center",
        )

    axis.set_xlim(0, max(frequencies) * 1.15)
    figure.tight_layout()
    figure.savefig(output_dir / f"span_graph_{safe_filename(label)}.{file_format}", dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.top is not None and args.top < 1:
        raise ValueError("--top must be at least 1")

    with args.input_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("The input JSON must contain an object of document records")

    with args.gold_file.open("r", encoding="utf-8") as file:
        gold_data = json.load(file)

    if not isinstance(gold_data, dict):
        raise ValueError("The gold JSON must contain an object of document records")

    gold_spans = collect_gold_spans(gold_data)

    span_counts = collect_span_counts(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for label, counts in sorted(span_counts.items()):
        create_graph(label, counts, args.output_dir, args.top, gold_spans, args.format)
        print(f"Created {label}: {len(counts)} unique spans, {sum(counts.values())} occurrences")

    print(f"Saved {len(span_counts)} graph(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
