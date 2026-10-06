import json
import re
from typing import List, Dict, Tuple, Any
import random
from pathlib import Path
import copy
import nlpaug
import nlpaug.augmenter.word as naw
import numpy as np
import nltk
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

nltk.download('averaged_perceptron_tagger_eng')

# Configuration: Word2Vec embeddings augmenter for entity augmentation only
AUGMENTATION_MODE = "strategy"
SELECTED_STRATEGY = "embeddings"

STRATEGY_CONFIGS = {
    "embeddings": {
        "aug_p": 0.5,
        "top_k": 20,
        "aug_min": 1,
        "aug_max": None,
    },
}

SEED = 50
NUM_VARIATIONS = 10
SAMPLE_VARIANTS_TO_PRINT = 0
# Percentage of entities to attempt to augment per record (0.0-1.0)
AUGMENT_ENTITY_PCT = 0.15
LOCAL_WORD2VEC_MODEL = "cc.en.300.word2vec.bin"
TOTAL_ENTITIES_PROCESSED = 0
ACTUALLY_AUGMENTED_ENTITIES = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(nlpaug, "set_seed"):
        nlpaug.set_seed(seed)


set_seed(SEED)

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent

# Input / output mirroring the original script naming pattern
def augmentation_slug() -> str:
    return f"entities_only_{SELECTED_STRATEGY}"

PATH_DATA = PROJECT_DIR / "Annotations" / "Train" / "augmented_datasets" / "train_gold_augmented_strategy_synonym_p15.json"
OUTPUT_PATH = PROJECT_DIR / "Annotations" / "Train" / "augmented_datasets" / f"train_gold_augmented_synonym15_{augmentation_slug()}.json"

with open(PATH_DATA, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize augmenter once at module level
config = STRATEGY_CONFIGS.get(SELECTED_STRATEGY)
if config is None:
    raise ValueError(f"Unsupported strategy: {SELECTED_STRATEGY}")

# Load Word2Vec augmenter
local_model_path = SCRIPT_DIR / LOCAL_WORD2VEC_MODEL
if local_model_path.exists():
    print(f"Using local Word2Vec model: {local_model_path}")
    word2vec_model_path = local_model_path
else:
    raise FileNotFoundError(f"Local Word2Vec model not found at {local_model_path}")

augmenter = naw.WordEmbsAug(
    model_type="word2vec",
    model_path=str(word2vec_model_path),
    action="substitute",
    aug_p=float(config.get("aug_p")),
    top_k=int(config.get("top_k")),
    aug_min=int(config.get("aug_min")),
    aug_max=config.get("aug_max"),
)


def augment_text_entities_only(text: str, entities: List[Dict[str, Any]], location_name: str) -> Tuple[str, str, str, Dict[int, Tuple[int, int]]]:
    """Augment a subset of entities in `text` for `location_name`.
    We pick entities independently with probability `AUGMENT_ENTITY_PCT` and send
    only the entity text to the contextual augmenter. After replacement we
    update following entity spans immediately.

    Returns: (final_augmented_text, original_text, last_augmented_masked, updated_spans_by_local_idx)
    """
    if not entities:
        return text, text, text, {}

    located_entities = [e for e in entities if e.get('location') == location_name]
    if not located_entities:
        return text, text, text, {}

    # Work on a mutable string builder and update offsets as we replace entities.
    augmented_text = text
    updated_spans: Dict[int, Tuple[int, int]] = {}

    # Sort by original start index
    indexed = sorted(
        enumerate([e for e in entities if e.get('location') == location_name]),
        key=lambda x: x[1]['start_idx'],
    )

    global TOTAL_ENTITIES_PROCESSED, ACTUALLY_AUGMENTED_ENTITIES
    offset = 0
    for local_idx, entity in indexed:
        TOTAL_ENTITIES_PROCESSED += 1
        orig_start = entity['start_idx']
        orig_end = entity['end_idx']
        start = orig_start + offset
        end = orig_end + offset

        if start < 0 or end >= len(augmented_text) or start > end:
            # invalid span; skip
            updated_spans[local_idx] = (orig_start, orig_end)
            continue

        entity_text = augmented_text[start:end+1]

        # decide whether to augment this entity
        if random.random() <= AUGMENT_ENTITY_PCT:
            # keep seed determinism per-variation; the caller sets global seed
            try:
                replacement = augmenter.augment(entity_text)
                # augmenter may return list when n>1; normalize
                if isinstance(replacement, list):
                    replacement = replacement[0]
                if not isinstance(replacement, str):
                    replacement = str(replacement)
                if replacement != entity_text:
                    ACTUALLY_AUGMENTED_ENTITIES += 1
            except Exception:
                replacement = entity_text
        else:
            replacement = entity_text

        # Apply replacement
        augmented_text = augmented_text[:start] + replacement + augmented_text[end+1:]

        new_start = start
        new_end = start + len(replacement) - 1
        updated_spans[local_idx] = (new_start, new_end)

        # update offset for subsequent entities
        offset += len(replacement) - (end - start + 1)

    return augmented_text, text, augmented_text, updated_spans


def augment_record(record: Dict[str, Any]) -> Dict[str, Any]:
    augmented_record = copy.deepcopy(record)

    title_text = record.get("metadata", {}).get("title", "")
    abstract_text = record.get("metadata", {}).get("abstract", "")

    augmented_title, _, _, title_spans = augment_text_entities_only(
        title_text,
        record.get("entities", []),
        "title",
    )
    augmented_abstract, _, _, abstract_spans = augment_text_entities_only(
        abstract_text,
        record.get("entities", []),
        "abstract",
    )

    if "metadata" in augmented_record:
        augmented_record["metadata"]["title"] = augmented_title
        augmented_record["metadata"]["abstract"] = augmented_abstract

    title_counter = 0
    abstract_counter = 0
    for entity in augmented_record.get("entities", []):
        location = entity.get("location")
        if location == "title":
            if title_counter in title_spans:
                new_start, new_end = title_spans[title_counter]
                entity["start_idx"] = new_start
                entity["end_idx"] = new_end
                entity["text_span"] = augmented_record["metadata"]["title"][new_start:new_end + 1]
            title_counter += 1
        elif location == "abstract":
            if abstract_counter in abstract_spans:
                new_start, new_end = abstract_spans[abstract_counter]
                entity["start_idx"] = new_start
                entity["end_idx"] = new_end
                entity["text_span"] = augmented_record["metadata"]["abstract"][new_start:new_end + 1]
            abstract_counter += 1

    return augmented_record


def build_augmented_variants(record: Dict[str, Any], num_variations: int, progress_bar: Any = None) -> List[Dict[str, Any]]:
    variants = []
    for variation_idx in range(num_variations):
        set_seed(SEED + variation_idx)
        variants.append(augment_record(record))
        if progress_bar is not None:
            progress_bar.update(1)
    return variants


def print_variant_preview(record_key: str, original_record: Dict[str, Any], variants: List[Dict[str, Any]], sample_count: int) -> None:
    if sample_count <= 0:
        return

    print(f"\nPreview for {record_key}:")
    for idx, variant in enumerate(variants[:sample_count], start=1):
        title = variant.get("metadata", {}).get("title", "")
        abstract = variant.get("metadata", {}).get("abstract", "")
        print(f"  Variant {idx}:")
        print(f"    title: {title[:200]}")
        print(f"    abstract: {abstract[:500]}")


augmented_data = {}
validation_errors = []
first_record_key = next(iter(data), None)

total_steps = len(data) * NUM_VARIATIONS
with tqdm(total=total_steps, desc="Total augmentation progress", leave=True) as progress_bar:
    for key, record in data.items():
        augmented_variants = build_augmented_variants(record, NUM_VARIATIONS, progress_bar)
        if first_record_key is not None and key == first_record_key:
            print_variant_preview(key, record, augmented_variants, SAMPLE_VARIANTS_TO_PRINT)
        for variation_idx, augmented_record in enumerate(augmented_variants, start=1):
            variant_key = f"{key}__aug{variation_idx}"
            augmented_data[variant_key] = augmented_record

with open(OUTPUT_PATH, 'w', encoding='utf-8') as output_file:
    json.dump(augmented_data, output_file, ensure_ascii=False, indent=2)

print("\nWrote entity-only augmented dataset to:\n", OUTPUT_PATH)
if TOTAL_ENTITIES_PROCESSED > 0:
    augmented_pct = (ACTUALLY_AUGMENTED_ENTITIES / TOTAL_ENTITIES_PROCESSED) * 100
    print(
        f"Actually augmented entities: {ACTUALLY_AUGMENTED_ENTITIES}/{TOTAL_ENTITIES_PROCESSED} "
        f"({augmented_pct:.2f}%)"
    )
else:
    print("Actually augmented entities: 0/0 (0.00%)")
