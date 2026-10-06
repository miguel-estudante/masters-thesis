from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import random
import re
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent  # repository root

def convert_json_to_tagged(entry, allowed_types=None, field="abstract"):
    """
    entry: one document's dict, e.g. data["38860943"]
    allowed_types: set of `label` strings to keep (e.g. {"food", "dietary supplement",
                    "statistical technique", "gene", "biomedical technique"})
                    if None, all types are tagged
    field: "abstract" or "title" — which text field to tag
    """
    text = entry[field]
    entities = [e for e in entry["entities"] if e["location"] == field]

    if allowed_types is not None:
        entities = [e for e in entities if e["label"] in allowed_types]

    # sort descending by start_idx so inserting tags doesn't shift earlier offsets
    entities_sorted = sorted(entities, key=lambda e: e["start_idx"], reverse=True)

    tagged_text = text
    for e in entities_sorted:
        start, end = e["start_idx"], e["end_idx"] + 1
        tag = e["label"].strip().lower().replace(" ", "_")
        span = tagged_text[start:end]
        # sanity check — should always match text_span, guards against offset drift
        assert span == e["text_span"], f"Mismatch: {span!r} != {e['text_span']!r}"
        tagged_text = (
            tagged_text[:start]
            + f"<{tag}>{span}</{tag}>"
            + tagged_text[end:]
        )

    return tagged_text


PATH_GOLD_INLINE = ROOT / "Annotations/Train/gold_quality/json_format/train_gold_entities_only_inline.json"

with open(PATH_GOLD_INLINE, 'r', encoding='utf-8') as file:
    examples = json.load(file)

PRIORITY_TYPES = {"gene", "food", "dietary_supplement", "statistical_technique", "biomedical_technique"}
ALLOWED_ENTITIES = ["gene","food","dietary_supplement","statistical_technique","biomedical_technique"]


def contains_priority_type(tagged_text, priority_types):
    tags_found = set(re.findall(r'<(\w+)>', tagged_text))
    return bool(tags_found & priority_types)

# assuming `all_tagged_abstracts` is a list of strings loaded from your file
all_abstracts = []
for abstract in examples:
    all_abstracts.append(examples[abstract])

priority_pool = [a for a in all_abstracts if contains_priority_type(a, PRIORITY_TYPES)]

def sample_fewshot(k=3):
    chosen = random.sample(priority_pool, k=min(k, len(priority_pool)))
    return "\n\n---\n\n".join(chosen)


TAG_PATTERN = re.compile(r'<(\w+)>(.*?)</\1>')

# reverse the underscore normalization used for generation tags
# back to the label format your real dataset uses (spaces)
TAG_TO_LABEL = {
    "disease": "DDF",
    "ddf": "DDF",
    "anatomical_location": "anatomical location",
    "animal": "animal",
    "bacteria": "bacteria",
    "biomedical_technique": "biomedical technique",
    "chemical": "chemical",
    "dietary_supplement": "dietary supplement",
    "drug": "drug",
    "food": "food",
    "gene": "gene",
    "human": "human",
    "microbiome": "microbiome",
    "statistical_technique": "statistical technique",
}

def convert_tagged_to_json(tagged_text, doc_id=None, topic_seed=None, forced_types=None):
    entities = []
    clean_text = ""
    last_end = 0

    for match in TAG_PATTERN.finditer(tagged_text):
        # append untagged text before this entity, unchanged
        clean_text += tagged_text[last_end:match.start()]
        start_idx = len(clean_text)

        tag = match.group(1)
        entity_text = match.group(2)
        clean_text += entity_text
        end_idx = len(clean_text)

        label = TAG_TO_LABEL.get(tag)
        if label is None:
            # unknown tag — model invented something outside your 13 types
            print(f"WARNING: unknown tag '{tag}' in doc {doc_id}, skipping")
            last_end = match.end()
            continue

        entities.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "location": "abstract",
            "text_span": entity_text,
            "label": label
        })
        last_end = match.end()

    clean_text += tagged_text[last_end:]

    # sanity check — guards against any offset drift from the regex logic
    for e in entities:
        span = clean_text[e["start_idx"]:e["end_idx"]]
        assert span == e["text_span"], f"Offset mismatch: {span!r} != {e['text_span']!r}"

    return doc_id,{
        "metadata": {
            "title": "",
            "author": None,
            "journal": None,
            "year": None,
            "annotator": "llm_synthetic",
            "abstract": clean_text,
            "forced_types": forced_types
        },
        "entities": entities,
    }

def build_user_prompt(forced_types):
    return f"""
Write a single research abstract (150-220 words).

The abstract must include at least one instance of each of these entity
types: {forced_types}. Other entity types may appear naturally if relevant
to the topic, but do not force types that don't fit.

Write it as continuous prose (no section headers like Background/Methods/
Results). Vary sentence structure and phrasing — avoid generic
scientific-abstract boilerplate openings like "This study investigates..."
or "Recent research has shown...". Start the abstract directly with a
substantive scientific claim or finding relevant to the topic.
"""

def build_system_prompt(abstract_examples):
    return f"""
You are a biomedical researcher writing abstract excerpts for scientific
literature on the gut-brain axis. You write in a natural, technical,
scientific tone consistent with published research abstracts.

You must tag every entity mention in the text using inline XML-style tags.
Only use these exact tag names: <disease>, <anatomical_location>, <animal>,
<bacteria>, <biomedical_technique>, <chemical>, <dietary_supplement>,
<drug>, <food>, <gene>, <human>, <microbiome>, <statistical_technique>.

Rules:
- Tag every entity mention, not just the first occurrence.
- Use the exact tag names given above, lowercase, with underscores.
- Do not tag common words that are not domain entities.
- Do not add any commentary, headers, or explanation — output only the
  tagged abstract text.

EXAMPLES:
{abstract_examples}
"""

# DeepSeek exposes an OpenAI-compatible API — swap in the API key/model via env vars.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

NUMBER_GENERATIONS = 1200
MAX_WORKERS = 16


def generate_one(label):
    """One API call + parse. Runs in a worker thread, so it touches no shared
    state — the caller assigns doc ids and accumulates the token counters."""
    abstract_examples = sample_fewshot(k=3)
    forced_types = random.sample(ALLOWED_ENTITIES, random.choice([2,3]))

    system_prompt = build_system_prompt(abstract_examples)
    user_prompt = build_user_prompt(forced_types)

    temperature = random.uniform(0.75, 1.05)
    top_p = random.uniform(0.9, 0.95)

    try:
        response_abstract = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            top_p=top_p
        )
    except Exception as e:
        return {"status": "api_error", "label": label, "error": str(e)}

    # DeepSeek reports context-cache usage as extra fields on `usage`.
    usage = response_abstract.usage
    cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    cache_miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0

    # Reasoning models return their thinking in a separate `reasoning_content`
    # field, so `content` is already clean; the strip is a fallback for models
    # that inline it as <think> blocks instead.
    sample_abstract = response_abstract.choices[0].message.content
    content_abstract = re.sub(r"<think>.*?</think>\s*", "", sample_abstract, flags=re.DOTALL)

    try:
        _, entry = convert_tagged_to_json(content_abstract, doc_id=label, forced_types=forced_types)
    except (AssertionError, ValueError) as e:
        return {"status": "rejected", "label": label, "error": str(e),
                "cache_hit": cache_hit, "cache_miss": cache_miss, "raw": sample_abstract}

    return {"status": "ok", "label": label, "entry": entry,
            "cache_hit": cache_hit, "cache_miss": cache_miss, "raw": sample_abstract}


results = {}
attempt = 0
total_cache_hit_tokens = 0
total_cache_miss_tokens = 0
sample_abstract = None

with tqdm(total=NUMBER_GENERATIONS) as progress:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Each wave submits only what is still missing, so rejections and API
        # errors get topped up without ever overshooting NUMBER_GENERATIONS.
        while len(results) < NUMBER_GENERATIONS:
            wave = [executor.submit(generate_one, attempt + n)
                    for n in range(NUMBER_GENERATIONS - len(results))]
            attempt += len(wave)

            for future in as_completed(wave):
                outcome = future.result()
                i = outcome["label"]

                if outcome["status"] == "api_error":
                    progress.write(f"[{i}] API ERROR: {outcome['error']}")
                    continue

                total_cache_hit_tokens += outcome["cache_hit"]
                total_cache_miss_tokens += outcome["cache_miss"]
                sample_abstract = outcome["raw"]

                if outcome["status"] == "rejected":
                    progress.write(f"[{i}] REJECTED: {outcome['error']}")
                    continue

                if len(results) >= NUMBER_GENERATIONS:
                    continue

                entry = outcome["entry"]
                results[f"synthetic_{len(results)}"] = entry
                progress.write(f"[{i}] OK - {len(entry['entities'])} entities "
                               f"- cache hit/miss: {outcome['cache_hit']}/{outcome['cache_miss']}")
                progress.update(1)

with open(ROOT / "SyntheticData" / "synthetic_data_deepseek.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


total_prompt_tokens = total_cache_hit_tokens + total_cache_miss_tokens
hit_rate = 100 * total_cache_hit_tokens / total_prompt_tokens if total_prompt_tokens else 0
print(f"\nCACHE: {total_cache_hit_tokens} hit / {total_cache_miss_tokens} miss "
      f"prompt tokens ({hit_rate:.1f}% hit rate over {attempt} calls)")

print("\nLLM RESPONSE\n")
print(sample_abstract)
