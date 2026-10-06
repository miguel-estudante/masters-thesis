# masters-thesis

Code used for my Master's thesis in Computer Science: Extraction of Named Entities From Biomedical Documents Using Augmented Data. Code is adapted from the baseline code provided by the organizers, accessible [here](https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline).

## Requirements

This project expects the packages listed in `requirements.txt`.

Install them from the repository root with:

```bash
pip install -r requirements.txt
```

The word2vec model used for entity augmentation is available from this Google Drive link: <https://drive.google.com/drive/folders/19f9wYugmEUZZCX-j1W1V9MCr8-UpJQxk?usp=sharing>. Place it in the `Utils` folder.

The training and inference script uses PyTorch with CUDA 12.6 packages, so a GPU-enabled environment is recommended.

## Project Layout

- `Train` contains the main training and prediction script.
- `Eval` contains submission validation utilities.
- `Articles` and `Annotations` contain the dataset files used by the script.
- `Predictions` is where prediction output is written.
- `Utils` contains additional scripts that are necessary for the project.
- `LLMAugmentation` contains the LLM-based synthetic data generation scripts, their pre- and post-processing scripts, and the prompts (`prompts/`).
- `SyntheticData` contains the generated synthetic abstracts (`pilots/` holds the small exploratory runs).
- `Analysis` contains scripts that analyse the datasets and the synthetic data.
- `Results` contains per-entity results and the span-frequency graphs of the synthetic data.

All scripts resolve paths from the repository root, so they can be run from any directory.

## Running the Code

The main entry point is `Train/gliner_interface.py`, which fine-tunes NuNerZero (`numind/NuNerZero`, downloaded from Hugging Face) and generates predictions:

```bash
python Train/gliner_interface.py
```

The script takes no command-line arguments. It is configured through variables at the top of the file:

- `THRESHOLD` sets the confidence threshold used in evaluation and prediction (default `0.6`).
- `finetune_model` enables fine-tuning.
- `generate_predictions` enables prediction with the fine-tuned model in `Train/outputs`. With `finetune_model = False`, this model must already exist from a previous run.
- `PATH_ARTICLES` selects the articles to predict on: `Articles/json_format/articles_dev.json` (default) or `Articles/json_format/articles_test.json`.
- `train_data` sets the combination of training sets.

### Output Files

- Prediction JSON is written to `Predictions/predicted_entities.json`.
- Fine-tuning checkpoints are saved under `Train/logs`.
- The best fine-tuned model is copied to `Train/outputs/NuNerZero_finetuned_T60.0` (for `THRESHOLD = 0.6`).

## Converting Annotations to the GLiNER Format

`Train/gliner_interface.py` reads its training data from `Train/data`, in the GLiNER format. `Utils/annotations_to_gliner_format.py` converts an annotations file into this format:

```bash
python Utils/annotations_to_gliner_format.py <input.json> [output.json]
```

Without an output path, the result is written to `Train/data/<input name>.json`.

The synonym and embedding-augmented files are converted the same way from the datasets in `Annotations/Train/augmented_datasets/augmented_datasets.zip`, after unzipping them into that folder.

## Evaluating Predictions

The NER evaluator expects predictions in evaluation format. To evaluate a run:

1. Convert `Predictions/predicted_entities.json` to `Predictions/predicted_entities_eval_format.json` using `Utils/NER_predictions_to_evaluation_format.py`.
2. Run `Eval/evaluate.py` after confirming `PREDICTIONS_PATH_NER` points to the converted file.

The evaluator computes micro and macro precision, recall, and F1 against the dev set.

## Data Augmentation

The data augmentation scripts are available in the `Utils` folder. The augmented datasets are stored in the `Annotations/Train/augmented_datasets/augmented_datasets.zip` file. Unzipping it is only needed to re-convert the augmented datasets to the GLiNER format. This file contains the augmented datasets in the same format as the provided base datasets.

If the user wants to generate these datasets again, they will have to manually choose from which dataset to augment and what strategies to use.

## LLM-Based Data Augmentation

Synthetic abstracts with inline entity tags are generated from few-shot examples of the gold training set.

1. `python LLMAugmentation/convert_annotations_to_entities_only.py` keeps only the abstracts and their entities from the gold set (`train_gold_entities_only.json`). The inline-tagged version used as few-shot examples is `train_gold_entities_only_inline.json`.
2. Generate synthetic abstracts with one of:
   - `LLMAugmentation/llm_augmentation_1prompt.py`: Qwen3-14B locally through llama.cpp (writes `SyntheticData/synthetic_data.json`).
   - `LLMAugmentation/llm_augmentation_1prompt_deepseek.py`: DeepSeek API (writes `SyntheticData/synthetic_data_deepseek.json`). Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY` first.
   - `LLMAugmentation/llm_augmentation_2prompts.py`: earlier two-prompt variant (generate, then annotate) used in the pilot runs.
3. Post-process the Qwen output: `python LLMAugmentation/find_inline_entity_tags.py <file> --filter-output <filtered_file>` removes abstracts with leftover tags or `<think>` content.
4. Convert the result to the GLiNER format (see [Converting Annotations to the GLiNER Format](#converting-annotations-to-the-gliner-format)).

`LLMAugmentation/filter_entities_by_type.py <file> <entity_type>` optionally keeps a single entity type in a dataset.

The `Analysis` scripts (`count_entity_types.py`, `analyze_synthetic_data.py`, `missed_annotation_analysis.py`) produce the statistics and graphs in `Results`.
