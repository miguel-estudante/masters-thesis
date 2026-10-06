import json
import re
from typing import List, Dict, Tuple, Any
import random
from pathlib import Path
import copy
import nlpaug
import nlpaug.augmenter.word as naw
import gensim.downloader as gensim_api
from nlpaug.flow import Sequential
from nlpaug.util.text.tokenizer import Tokenizer
import numpy as np
import nltk
try:
	from tqdm.auto import tqdm
except ImportError:
	def tqdm(iterable, **kwargs):
		return iterable

nltk.download('averaged_perceptron_tagger_eng')

# Choose one mode:
# - "pipeline": use a named nlpaug Sequential flow from AUGMENTATION_PIPELINES
# - "strategy": use one augmenter from STRATEGY_CONFIGS
AUGMENTATION_MODE = "strategy"

SELECTED_PIPELINE = "aggressive_mix"
SELECTED_STRATEGY = "synonym"

STRATEGY_CONFIGS = {
	"synonym": {"aug_p": 0.15},
	"random_swap": {"aug_p": 0.1},
	"random_delete": {"aug_p": 0.05},
}

AUGMENTATION_PIPELINES = {
	"light_mix": ["synonym", "random_swap"],
	"aggressive_mix": ["synonym", "random_swap", "random_delete"],
}

SEED = 50
NUM_VARIATIONS = 1
SAMPLE_VARIANTS_TO_PRINT = 0
FORMAT_TAG_PATTERN = re.compile(r"<\s*/?\s*i\s*>", re.IGNORECASE)
PLACEHOLDER_TOKEN_PATTERN = re.compile(r"__ENTITY_[A-Z]+_\d+__|__FMT_TAG_\d+__")


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	if hasattr(nlpaug, "set_seed"):
		nlpaug.set_seed(seed)


set_seed(SEED)

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent

def augmentation_slug() -> str:
	if AUGMENTATION_MODE == "pipeline":
		return f"pipeline_{SELECTED_PIPELINE}"
	if AUGMENTATION_MODE == "strategy":
		return f"strategy_{SELECTED_STRATEGY}_p{int(STRATEGY_CONFIGS[SELECTED_STRATEGY]['aug_p'] * 100)}"
	raise ValueError(f"Unsupported AUGMENTATION_MODE: {AUGMENTATION_MODE}")

PATH_DATA = PROJECT_DIR / "Annotations" / "Train" / "silver_quality" / "json_format" / "train_silver.json" 
OUTPUT_PATH = PROJECT_DIR / "Annotations" / "Train" / "augmented_datasets" / f"train_gold_augmented_{augmentation_slug()}.json"

with open(PATH_DATA, 'r', encoding='utf-8') as file:
	data = json.load(file)


def placeholder_aware_tokenizer(text: str) -> List[str]:
	parts = re.split(r"(__ENTITY_[A-Z]+_\d+__|__FMT_TAG_\d+__)", text)
	tokens: List[str] = []
	for part in parts:
		if not part:
			continue
		if PLACEHOLDER_TOKEN_PATTERN.fullmatch(part):
			tokens.append(part)
			continue

		sub_tokens = re.split(r"(\W)", part)
		tokens.extend([token for token in sub_tokens if len(token.strip()) > 0])

	return tokens


def build_strategy_augmenter(strategy_name: str, protected_tokens: List[str]):
	config = STRATEGY_CONFIGS.get(strategy_name)
	if config is None:
		raise ValueError(f"Unsupported strategy: {strategy_name}")

	aug_p = float(config.get("aug_p"))
	protected_tokens_case_robust = sorted(set(protected_tokens + [token.lower() for token in protected_tokens]))
	placeholder_regex = r"(?i)^__(?:entity_(?:title|abstract)_\d+|fmt_tag_\d+)__$"

	# Protect placeholders so only non-entity text gets augmented.
	if strategy_name == "synonym":
		return naw.SynonymAug(
			aug_src="wordnet",
			aug_p=aug_p,
			stopwords=protected_tokens_case_robust,
			stopwords_regex=placeholder_regex,
			tokenizer=placeholder_aware_tokenizer,
			reverse_tokenizer=Tokenizer.reverse_tokenizer,
			aug_max=None,
			aug_min=0,
		)

	if strategy_name == "random_swap":
		return naw.RandomWordAug(
			action="swap",
			aug_p=aug_p,
			stopwords=protected_tokens_case_robust,
			stopwords_regex=placeholder_regex,
			tokenizer=placeholder_aware_tokenizer,
			reverse_tokenizer=Tokenizer.reverse_tokenizer,
			aug_max=None,
			aug_min=0,
		)

	if strategy_name == "random_delete":
		return naw.RandomWordAug(
			action="delete",
			aug_p=aug_p,
			stopwords=protected_tokens_case_robust,
			stopwords_regex=placeholder_regex,
			tokenizer=placeholder_aware_tokenizer,
			reverse_tokenizer=Tokenizer.reverse_tokenizer,
			aug_max=None,
			aug_min=0,
		)

	raise ValueError(f"Unsupported strategy: {strategy_name}")


def build_pipeline_augmenter(pipeline_name: str, protected_tokens: List[str]):
	strategy_names = AUGMENTATION_PIPELINES.get(pipeline_name)
	if strategy_names is None:
		raise ValueError(f"Unsupported pipeline: {pipeline_name}")

	augmenters = [build_strategy_augmenter(name, protected_tokens) for name in strategy_names]
	if not augmenters:
		raise ValueError(f"Pipeline is empty: {pipeline_name}")

	return Sequential(augmenters)


def build_augmenter(protected_tokens: List[str]):
	if AUGMENTATION_MODE == "strategy":
		return build_strategy_augmenter(SELECTED_STRATEGY, protected_tokens)

	if AUGMENTATION_MODE == "pipeline":
		return build_pipeline_augmenter(SELECTED_PIPELINE, protected_tokens)

	raise ValueError(f"Unsupported AUGMENTATION_MODE: {AUGMENTATION_MODE}")


def protect_formatting_tags(text: str) -> Tuple[str, Dict[str, str]]:
	mapping = {}

	def _replacer(match: re.Match[str]) -> str:
		placeholder = f"__FMT_TAG_{len(mapping)}__"
		mapping[placeholder] = match.group(0)
		return placeholder

	masked_text = FORMAT_TAG_PATTERN.sub(_replacer, text)
	return masked_text, mapping


def restore_placeholders(augmented_text: str, mapping: Dict[str, str]) -> str:
	restored_text = augmented_text
	for placeholder in sorted(mapping.keys(), key=len, reverse=True):
		restored_text = restored_text.replace(placeholder, mapping[placeholder])
	return restored_text


def normalize_placeholder_casing(text: str, placeholders: List[str]) -> str:
	normalized_text = text
	for placeholder in sorted(set(placeholders), key=len, reverse=True):
		normalized_text = re.sub(re.escape(placeholder), placeholder, normalized_text, flags=re.IGNORECASE)
	return normalized_text


def normalize_augmented_output(augmented_output: Any) -> str:
	"""Unwrap nlpaug flow outputs that may return nested lists."""
	current = augmented_output
	while isinstance(current, list):
		if not current:
			raise ValueError("Augmenter returned an empty list output")
		current = current[0]

	if not isinstance(current, str):
		raise ValueError(f"Augmenter returned unexpected type: {type(current).__name__}")

	return current


def compute_updated_entity_spans(masked_text: str, mapping: Dict[str, str], placeholder_to_idx: Dict[str, int]) -> Dict[int, Tuple[int, int]]:
	placeholder_positions = []
	for placeholder, entity_text in mapping.items():
		start = masked_text.find(placeholder)
		if start == -1:
			raise ValueError(f"Placeholder not found after augmentation: {placeholder}")
		placeholder_positions.append((start, placeholder, entity_text))

	placeholder_positions.sort(key=lambda x: x[0])

	updated_spans = {}
	running_delta = 0
	for masked_start, placeholder, entity_text in placeholder_positions:
		local_idx = placeholder_to_idx[placeholder]
		start_idx = masked_start + running_delta
		end_idx = start_idx + len(entity_text) - 1
		updated_spans[local_idx] = (start_idx, end_idx)
		running_delta += len(entity_text) - len(placeholder)

	return updated_spans


def augment_text_and_entities(
	text: str,
	entities: List[Dict[str, Any]],
	location_name: str,
) -> Tuple[str, str, str, Dict[int, Tuple[int, int]]]:
	modified_text = text

	located_entities = [
		{"local_idx": local_idx, "full_idx": full_idx, "entity": entity}
		for local_idx, (full_idx, entity) in enumerate(
			[(i, x) for i, x in enumerate(entities) if x.get('location') == location_name]
		)
	]

	if not located_entities:
		return text, text, text, {}

	# Overlapping/nested annotations (e.g. "vitamin D" and "D") break the
	# placeholder replacement logic. Skip augmentation for this section to keep
	# entity offsets consistent and avoid crashes.
	sorted_by_start = sorted(
		located_entities,
		key=lambda item: (item["entity"]["start_idx"], item["entity"]["end_idx"]),
	)
	prev_end = -1
	for item in sorted_by_start:
		start = item["entity"].get("start_idx", -1)
		end = item["entity"].get("end_idx", -1)
		if start <= prev_end:
			return text, text, text, {}
		prev_end = end

	placeholder_map = {}
	placeholder_to_local_idx = {}

	indexed_entities = list(located_entities)
	indexed_entities.sort(key=lambda x: x["entity"]['start_idx'], reverse=True)

	for item in indexed_entities:
		local_idx = item["local_idx"]
		entity = item["entity"]
		start = entity['start_idx']
		end = entity['end_idx']
		entity_text = text[start:end + 1]

		placeholder = f"__ENTITY_{location_name.upper()}_{local_idx}__"
		modified_text = modified_text[:start] + placeholder + modified_text[end + 1:]
		placeholder_map[placeholder] = entity_text
		placeholder_to_local_idx[placeholder] = local_idx

	masked_with_format_placeholders, format_placeholder_map = protect_formatting_tags(modified_text)

	protected_tokens = list(placeholder_map.keys()) + list(format_placeholder_map.keys())
	augmenter = build_augmenter(protected_tokens)

	augmented_masked = normalize_augmented_output(augmenter.augment(masked_with_format_placeholders))

	augmented_masked_with_format_restored = restore_placeholders(augmented_masked, format_placeholder_map)
	augmented_masked_with_format_restored = normalize_placeholder_casing(
		augmented_masked_with_format_restored,
		list(placeholder_map.keys()),
	)

	updated_spans_by_local_idx = compute_updated_entity_spans(
		augmented_masked_with_format_restored,
		placeholder_map,
		placeholder_to_local_idx,
	)

	augmented_text = restore_placeholders(augmented_masked_with_format_restored, placeholder_map)
	return augmented_text, modified_text, augmented_masked_with_format_restored, updated_spans_by_local_idx


def augment_record(record: Dict[str, Any]) -> Dict[str, Any]:
	augmented_record = copy.deepcopy(record)

	title_text = record.get("metadata", {}).get("title", "")
	abstract_text = record.get("metadata", {}).get("abstract", "")

	augmented_title, _, _, title_spans = augment_text_and_entities(
		title_text,
		record.get("entities", []),
		"title",
	)
	augmented_abstract, _, _, abstract_spans = augment_text_and_entities(
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
			title_counter += 1
		elif location == "abstract":
			if abstract_counter in abstract_spans:
				new_start, new_end = abstract_spans[abstract_counter]
				entity["start_idx"] = new_start
				entity["end_idx"] = new_end
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

	unique_outputs = {
		(
			variant.get("metadata", {}).get("title", ""),
			variant.get("metadata", {}).get("abstract", ""),
		)
		for variant in variants[:sample_count]
	}
	print(f"  unique previews: {len(unique_outputs)} / {min(sample_count, len(variants))}")


def validate_updated_entity_offsets(original_record: Dict[str, Any], augmented_record: Dict[str, Any]) -> List[str]:
	errors = []
	original_title = original_record.get("metadata", {}).get("title", "")
	original_abstract = original_record.get("metadata", {}).get("abstract", "")
	augmented_title = augmented_record.get("metadata", {}).get("title", "")
	augmented_abstract = augmented_record.get("metadata", {}).get("abstract", "")

	original_entities = original_record.get("entities", [])
	augmented_entities = augmented_record.get("entities", [])

	if len(original_entities) != len(augmented_entities):
		errors.append(
			f"Entity count mismatch: original={len(original_entities)} augmented={len(augmented_entities)}"
		)
		return errors

	for idx, (orig_entity, aug_entity) in enumerate(zip(original_entities, augmented_entities)):
		location = orig_entity.get("location")
		if location == "title":
			orig_text = original_title
			aug_text = augmented_title
		elif location == "abstract":
			orig_text = original_abstract
			aug_text = augmented_abstract
		else:
			continue

		orig_start = orig_entity.get("start_idx")
		orig_end = orig_entity.get("end_idx")
		aug_start = aug_entity.get("start_idx")
		aug_end = aug_entity.get("end_idx")

		if any(v is None for v in [orig_start, orig_end, aug_start, aug_end]):
			errors.append(f"Entity {idx} missing span indices for location={location}")
			continue

		if orig_start < 0 or orig_end >= len(orig_text) or orig_start > orig_end:
			errors.append(f"Entity {idx} invalid original span ({orig_start}, {orig_end}) location={location}")
			continue

		if aug_start < 0 or aug_end >= len(aug_text) or aug_start > aug_end:
			errors.append(f"Entity {idx} invalid augmented span ({aug_start}, {aug_end}) location={location}")
			continue

		expected_entity_text = orig_text[orig_start:orig_end + 1]
		observed_entity_text = aug_text[aug_start:aug_end + 1]

		if expected_entity_text != observed_entity_text:
			errors.append(
				f"Entity {idx} mismatch location={location} expected={expected_entity_text!r} observed={observed_entity_text!r}"
			)

	return errors


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
			record_errors = validate_updated_entity_offsets(record, augmented_record)
			for error in record_errors:
				validation_errors.append(f"{variant_key}: {error}")

with open(OUTPUT_PATH, 'w', encoding='utf-8') as output_file:
	json.dump(augmented_data, output_file, ensure_ascii=False, indent=2)

print(f"\nValidation mismatches: {len(validation_errors)}")
if validation_errors:
	print("Showing up to 10 mismatches:")
	for error in validation_errors[:10]:
		print("-", error)

print("\nUpdated JSON written to:\n", OUTPUT_PATH)

