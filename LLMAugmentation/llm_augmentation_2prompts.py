# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "llama-cpp-python",
#     "tqdm",
#     "huggingface-hub",
# ]
# ///
from llama_cpp import Llama
import json
from pathlib import Path
import random
import re
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent  # repository root


def find_offsets(text, entities):
    annotations = []
    matches_by_text = {}
    next_match_index = {}

    valid_entities = []
    for entity in entities:
        if isinstance(entity, dict) and "text" in entity and entity["text"]:
            valid_entities.append(entity)
        else:
            print(f"Skipping malformed entity: {entity}")

    for entity in valid_entities:
        entity_text = entity["text"]

        if entity_text not in matches_by_text:
            starts = [m.start() for m in re.finditer(re.escape(entity_text), text)]
            matches_by_text[entity_text] = starts
            next_match_index[entity_text] = 0

        current_index = next_match_index[entity_text]
        starts = matches_by_text[entity_text]

        if current_index >= len(starts):
            print(f"Missing: {entity['text']}")
            continue

        start = starts[current_index]
        next_match_index[entity_text] = current_index + 1

        annotations.append({
            **entity,
            "start": start,
            "end": start + len(entity_text)
        })

    annotations.sort(key=lambda item: item["start"])
    sample = {"abstract":text,"entities":annotations}

    return sample

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen3-8B-GGUF",
    filename="Qwen3-8B-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=True,
    flash_attn=True
)

NEUROLOGICAL_DISORDERS = [
    "Alzheimer's disease",
    "Amyotrophic lateral sclerosis",
    "Autism spectrum disorder",
    "Epilepsy",
    "Generalized anxiety disorder",
    "Huntington's disease",
    "Major depressive disorder",
    "Migraine",
    "Multiple sclerosis",
    "Parkinson's disease",
    "Schizophrenia",
    "Traumatic brain injury",
]

GI_DISORDERS = [
    "Crohn's disease",
    "Ulcerative colitis",
    "Irritable bowel syndrome",
    "Functional dyspepsia",
    "Celiac disease",
    "Small intestinal bacterial overgrowth",
    "Gastroesophageal reflux disease",
    "Clostridioides difficile infection",
]

HOSTS = [
    "C57BL/6 mice",
    "BALB/c mice",
    "Sprague-Dawley rats",
    "Wistar rats",
    "Zebrafish",
    "Rhesus macaques",
    "Domestic pigs",
    "Human participants",
]

INTERVENTIONS = [
    "high-fiber diet",
    "Mediterranean diet",
    "ketogenic diet",
    "fecal microbiota transplantation",
    "broad-spectrum antibiotics",
    "inulin supplementation",
    "resistant starch supplementation",
    "omega-3 fatty acid supplementation",
    "polyphenol-rich diet",
    "probiotic supplementation",
    "synbiotic supplementation",
    "intermittent fasting",
]

BACTERIA = [
    "Akkermansia muciniphila",
    "Bacteroides fragilis",
    "Bacteroides vulgatus",
    "Bifidobacterium adolescentis",
    "Bifidobacterium longum",
    "Blautia obeum",
    "Clostridium butyricum",
    "Collinsella aerofaciens",
    "Coprococcus comes",
    "Escherichia coli",
    "Faecalibacterium prausnitzii",
    "Lactobacillus plantarum",
    "Parabacteroides distasonis",
    "Prevotella copri",
    "Roseburia intestinalis",
    "Ruminococcus gnavus",
]

BIOMARKERS = [
    "brain-derived neurotrophic factor",
    "C-reactive protein",
    "interleukin-1β",
    "interleukin-6",
    "interleukin-10",
    "tumor necrosis factor-α",
    "lipopolysaccharide",
    "cortisol",
    "serotonin",
    "dopamine",
    "γ-aminobutyric acid",
    "acetate",
    "propionate",
    "butyrate",
    "indole-3-propionic acid",
]

TECHNIQUES = [
    "16S rRNA gene sequencing",
    "shotgun metagenomic sequencing",
    "metatranscriptomic sequencing",
    "liquid chromatography-mass spectrometry",
    "gas chromatography-mass spectrometry",
    "enzyme-linked immunosorbent assay",
    "quantitative PCR",
    "RNA sequencing",
    "Western blotting",
    "flow cytometry",
]

STATISTICAL_METHODS = [
    "Mann-Whitney U test",
    "Student's t-test",
    "Wilcoxon signed-rank test",
    "Kruskal-Wallis test",
    "PERMANOVA",
    "Spearman correlation",
    "Pearson correlation",
    "principal coordinates analysis",
    "principal component analysis",
    "false discovery rate correction",
]


PROMPT_ABSTRACT = """
You are an expert biomedical researcher.

Write one fictional biomedical research abstract.

The abstract must be between 180 and 250 words.

Requirements:

- The study must be scientifically plausible.
- The study must be fictional.
- The abstract must read like a publication abstract.
- Additional bacteria, biomarkers and techniques may be included if appropriate.
- Do not include a title.
- Do not include citations.
- Do not include section headings.

Return ONLY the abstract.
"""
def build_user_abstract(study):
    return f"""

    Generate an abstract.

    The study MUST use the following study specification.

    Neurological disorder:
    {study["neurological_disorder"]}

    Gastrointestinal disorder:
    {study["gastrointestinal_disorder"]}

    Host:
    {study["host"]}

    Primary bacterial taxon:
    {study["bacterium"]}

    Intervention:
    {study["intervention"]}

    Primary biomarker:
    {study["biomarker"]}

    Primary laboratory technique:
    {study["technique"]}

    Primary statistical analysis:
    {study["statistics"]}
    """

PROMPT_NER = """
You are an expert biomedical named entity annotator.

Your task is to annotate the provided abstract.

Annotate EVERY occurrence of every entity that belongs to one of the following categories.

Allowed labels:

- anatomical location
- animal
- biomedical technique
- bacteria
- chemical
- dietary supplement
- ddf -  This means disease, disorder or finding
- drug
- food
- gene
- human
- microbiome
- statistical technique

Rules:

- Annotate every occurrence, even if the entity is repeated.
- Do not invent entities.
- Every entity text must appear exactly in the abstract.
- If an entity does not fit one of the allowed labels, ignore it.
- Never create new labels.
- Never output explanations.
- Never output markdown.
- Use exactly the allowed label strings.

Return ONLY a JSON object.

Schema:

{
    "entities":[
        {
            "label":"chemical",
            "text":"serotonin"
        }
    ]
}

"""


def build_study_spec():
    return {
        "neurological_disorder": random.choice(NEUROLOGICAL_DISORDERS),
        "gastrointestinal_disorder": random.choice(GI_DISORDERS),
        "host": random.choice(HOSTS),
        "intervention": random.choice(INTERVENTIONS),
        "bacterium": random.choice(BACTERIA),
        "biomarker": random.choice(BIOMARKERS),
        "technique": random.choice(TECHNIQUES),
        "statistics": random.choice(STATISTICAL_METHODS),
    }


def generate_sample():
    study = build_study_spec()

    response_abstract = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": PROMPT_ABSTRACT},
            {"role": "user", "content": build_user_abstract(study)}
        ],
        temperature=0.8,
    )
    sample_abstract = response_abstract["choices"][0]["message"]["content"]

    content_abstract = re.sub(r"<think>.*?</think>\s*", "", sample_abstract, flags=re.DOTALL)

    response_NER = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": PROMPT_NER},
            {"role": "user", "content": f"""/no_think

Abstract:
{content_abstract}
Annotate the abstract."""}
        ],
        temperature=0.2,
    )

    raw_response_NER = response_NER["choices"][0]["message"]["content"]
    content_entities = re.sub(r"<think>.*?</think>\s*", "", raw_response_NER, flags=re.DOTALL)

    json_match = re.search(r"\{.*\}", content_entities, re.DOTALL)
    if json_match:
        content_entities = json_match.group(0)

    content_entities = content_entities.replace("'", '"')
    content_entities = re.sub(r",\s*\}", "}", content_entities)
    content_entities = re.sub(r",\s*\]", "]", content_entities)

    try:
        parsed = json.loads(content_entities)
    except json.JSONDecodeError:
        print(f"Failed to parse NER response:\n{content_entities}")
        return generate_sample()

    if not isinstance(parsed.get("entities"), list):
        print(f"Missing or invalid 'entities' in NER response:\n{content_entities}")
        return generate_sample()

    annotations = find_offsets(content_abstract, parsed["entities"])

    return {
        "metadata": {
            "title": "",
            "author": "",
            "journal": "",
            "year": "",
            "abstract": annotations["abstract"],
            "annotator": ""
        },
        "entities": annotations["entities"]
    }


output = {}

for sample_number in tqdm(range(1, 51), total=50, desc="Generating samples"):
    output[f"sample{sample_number}"] = generate_sample()

with open(ROOT / "SyntheticData" / "sample.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
