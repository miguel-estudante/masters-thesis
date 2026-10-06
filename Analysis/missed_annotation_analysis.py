"""
Missed-annotation deep dive.

Checks how often a "known" real entity string (from Gold/Silver/Silver-2025)
appears in the synthetic abstracts without being tagged at all, as a lower-bound
estimate of false-negative supervision introduced by incomplete LLM annotation.

Requires: pip install flashtext
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

from flashtext import KeywordProcessor

# ---- File paths (relative to the repository root) ----
ROOT = Path(__file__).resolve().parent.parent  # repository root
GOLD_PATH = ROOT / "Annotations/Train/gold_quality/json_format/train_gold.json"
SILVER_PATH = ROOT / "Annotations/Train/silver_quality/json_format/train_silver.json"
SILVER_2025_PATH = ROOT / "Annotations/Train/silver_quality/json_format/train_silver_2025.json"
# Bronze is intentionally excluded -- see build_vocab() docstring.
SYNTHETIC_PATH = ROOT / "SyntheticData" / "augmented_qwen14b_1200examples_1prompt_inline_filtered_metadata.json"

MIN_SPAN_LENGTH = 3   # drop spans shorter than this
MIN_FREQUENCY = 2      # span must appear at least this many times in the reference tiers


def is_unambiguous(span: str) -> bool:
    """
    Heuristic filter to exclude common, context-dependent English words
    (e.g. "gut", "controls", "model") that were tagged as an entity only
    incidentally somewhere in a large corpus, while keeping genuinely
    specific technical spans.

    A span passes if it is:
      - multi-word or hyphenated (e.g. "Lactobacillus rhamnosus", "gut-brain axis"), OR
      - contains a digit (e.g. "TLR4", "16S rRNA"), OR
      - contains a run of 2+ uppercase letters, i.e. acronym-like (e.g. "BDNF", "PCA")
    """
    if " " in span or "-" in span:
        return True
    if re.search(r"\d", span):
        return True
    if re.search(r"[A-Z]{2,}", span):
        return True
    return False


def build_vocab(paths):
    """
    Build a per-label vocabulary of "known real entity strings" from the
    given files (list of paths to Gold/Silver/Silver-2025 -- NOT Bronze).

    Bronze is excluded deliberately: it is a weakly/automatically labeled
    tier, and inspection showed it contains annotation noise (e.g. "the"
    and "and" tagged as DDF) that would otherwise inflate false positives
    in this analysis. Gold/Silver/Silver-2025 are the more reliably
    curated tiers.

    A span is kept only if it appears at least MIN_FREQUENCY times across
    these files (to filter one-off annotation noise) and passes
    is_unambiguous() (to filter generic, context-dependent single words).
    """
    span_counts = defaultdict(Counter)

    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for _, entry in data.items():
            for e in entry.get("entities", []):
                span = e.get("text_span", "").strip()
                label = e.get("label", "").strip().lower()
                if len(span) >= MIN_SPAN_LENGTH:
                    span_counts[label][span] += 1

    vocab_by_label = {}
    for label, counter in span_counts.items():
        vocab_by_label[label] = {
            span for span, count in counter.items()
            if count >= MIN_FREQUENCY and is_unambiguous(span)
        }
    return vocab_by_label


def build_processors(vocab_by_label):
    """One flashtext KeywordProcessor per label, for fast multi-pattern matching."""
    processors = {}
    for label, spans in vocab_by_label.items():
        kp = KeywordProcessor(case_sensitive=True)
        for span in spans:
            kp.add_keyword(span)
        processors[label] = kp
    return processors


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and a_end > b_start


def analyze(synthetic_path, processors):
    """
    For every synthetic abstract, find occurrences of known vocabulary
    (per label) and check whether each occurrence is covered by ANY tagged
    entity in that abstract (regardless of the tag's own label).

    Returns per-label stats dict and a dict of example missed spans for
    manual spot-checking.
    """
    with open(synthetic_path, encoding="utf-8") as f:
        synth = json.load(f)

    stats = defaultdict(lambda: {"total": 0, "covered_any": 0, "covered_same": 0, "missed": 0})
    missed_examples = defaultdict(list)

    for doc_id, entry in synth.items():
        abstract = entry["metadata"]["abstract"]
        tagged_spans = [
            (e["start_idx"], e["end_idx"], e["label"].strip().lower())
            for e in entry["entities"]
        ]

        for label, kp in processors.items():
            for keyword, start, end in kp.extract_keywords(abstract, span_info=True):
                s = stats[label]
                s["total"] += 1

                covered_any = False
                covered_same = False
                for (ts, te, tl) in tagged_spans:
                    if overlaps(start, end, ts, te):
                        covered_any = True
                        if tl == label:
                            covered_same = True

                if covered_any:
                    s["covered_any"] += 1
                if covered_same:
                    s["covered_same"] += 1
                if not covered_any:
                    s["missed"] += 1
                    if len(missed_examples[label]) < 10:
                        missed_examples[label].append((doc_id, keyword))

    return dict(stats), dict(missed_examples)


def print_report(stats):
    """
    Reports two distinct failure modes per label:
      - Miss rate: known vocabulary occurrences with NO tag covering them at all
        (false negatives / missed annotations).
      - Mislabel rate: known vocabulary occurrences that WERE tagged, but under
        a different label than the vocabulary's own type (cross-type confusion,
        e.g. a known "gene" span tagged as "chemical" instead).
    These are independent: a occurrence is either missed, mislabeled, or
    correctly labeled (covered_same), and the three are mutually exclusive.
    """
    overall = {"total": 0, "covered_any": 0, "covered_same": 0, "missed": 0}
    header = (f'{"Label":22}{"Total":>8}{"CoveredAny":>12}{"CoveredSame":>13}'
              f'{"Missed":>8}{"MissRate":>10}{"Mislabeled":>12}{"MislabelRate":>14}')
    print(header)
    for label in sorted(stats, key=lambda l: -stats[l]["total"]):
        s = stats[label]
        mislabeled = s["covered_any"] - s["covered_same"]
        miss_rate = s["missed"] / s["total"] * 100 if s["total"] else 0
        mislabel_rate = mislabeled / s["total"] * 100 if s["total"] else 0
        print(f'{label:22}{s["total"]:8d}{s["covered_any"]:12d}{s["covered_same"]:13d}'
              f'{s["missed"]:8d}{miss_rate:9.1f}%{mislabeled:12d}{mislabel_rate:13.1f}%')
        for k in overall:
            overall[k] += s[k]
 
    print()
    overall_mislabeled = overall["covered_any"] - overall["covered_same"]
    overall_miss_rate = overall["missed"] / overall["total"] * 100 if overall["total"] else 0
    overall_mislabel_rate = overall_mislabeled / overall["total"] * 100 if overall["total"] else 0
    print(f'OVERALL: total={overall["total"]}, covered_any={overall["covered_any"]}, '
          f'covered_same={overall["covered_same"]}, missed={overall["missed"]}, '
          f'mislabeled={overall_mislabeled}, '
          f'miss_rate={overall_miss_rate:.1f}%, mislabel_rate={overall_mislabel_rate:.1f}%')


if __name__ == "__main__":
    vocab_by_label = build_vocab([GOLD_PATH, SILVER_PATH, SILVER_2025_PATH])
    for label, spans in sorted(vocab_by_label.items(), key=lambda x: -len(x[1])):
        print(f"{label}: {len(spans)} unambiguous spans")
    print()

    processors = build_processors(vocab_by_label)
    stats, missed_examples = analyze(SYNTHETIC_PATH, processors)
    print_report(stats)

    print("\nSample missed spans per label (for manual spot-checking):")
    for label, examples in missed_examples.items():
        print(f"  {label}: {examples[:5]}")
