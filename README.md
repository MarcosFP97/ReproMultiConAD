# Reproducing MultiConAD: Assessing the Robustness of Speech-Based Cognitive Impairment Detection

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="NLP" src="https://img.shields.io/badge/NLP-Clinical%20Language-5A4FCF?style=for-the-badge">
  <img alt="Transformers" src="https://img.shields.io/badge/HuggingFace-Transformers-FFCC00?style=for-the-badge&logo=huggingface&logoColor=black">
  <img alt="LLMs" src="https://img.shields.io/badge/LLMs-Gemini%20%7C%20Mistral-111827?style=for-the-badge">
  <img alt="HPC" src="https://img.shields.io/badge/HPC-SLURM-ED1C24?style=for-the-badge">
</p>


Repository accompanying the reproducibility paper **Reproducing MultiConAD: Assessing the Robustness of Speech-Based Cognitive Impairment Detection**. The project combines classical models (TF-IDF), dense representations (E5), Transformer models (BERT), synthetic data augmentation with LLMs, and cross-dataset experiments, structured into five experimental phases.

> This repository is intended for research purposes. It is not a clinical tool or a medical diagnostic system.

---

## Starting Point: MultiConAD

The project builds upon the pipeline introduced in **MultiConAD: A Unified Multilingual Conversational Dataset for Early Alzheimer's Detection** [[arXiv:2502.19208]](https://arxiv.org/abs/2502.19208), which provides the underlying infrastructure for normalizing conversational datasets on dementia, harmonizing clinical and demographic metadata, and evaluating models in both monolingual and multilingual settings.

This paper restricts the study to **English and Spanish**, incorporating Transformer-based models, dataset-specific analyses, CHAT markers as special tokens, controlled synthetic data augmentation, and cross-dataset transfer.

---

## Datasets

| Dataset             | Language | Task                  | Classes           |
| ------------------- | -------- | --------------------- | ----------------- |
| Pitt (DementiaBank) | English  | Cookie Theft          | HC, MCI, Dementia |
| Delaware            | English  | Various               | HC, MCI, Dementia |
| Lu                  | English  | Various               | HC, Dementia      |
| Taukadial           | English  | Picture description   | HC, Dementia      |
| VAS                 | English  | Various               | HC, MCI           |
| WLS                 | English  | Various               | HC, Dementia      |
| Ivanova             | Spanish  | Reading (Don Quixote) | HC, MCI, Dementia |
| PerLA               | Spanish  | Various               | Dementia          |



The original datasets are not distributed in this repository. Access to the DementiaBank data is restricted and must be requested through DementiaBank/TalkBank. Researchers should follow the [official data access instructions](https://talkbank.org/dementia/access/) to **request access** and comply with the applicable data-use requirements.

All scripts required to extract the data (including parsing and normalization to JSONL) are located under the `Extracting_data/` directory. The scripts required for subsequent text preprocessing can be found under the `Preprocessing_text/` directory.

After running these scripts, the data will be stored in JSONL format following the `NormalizedDataPoint` schema:

```json
{
  "PID": "...",
  "Dataset": "Pitt",
  "Diagnosis": "Dementia",
  "Age": 75,
  "Gender": "F",
  "MMSE": 21,
  "Text_participant": "PAR: ..."
}
```

**Note** that the Taukadial dataset provides audio recordings rather than transcribed text. Therefore, the recordings must first be transcribed using the scripts provided under `Audio_transcription/`.

Finally, the `scripts/datasets_creation/ directory contains an example of how to run these processing steps sequentially.

---

## Project Structure

```
ReproMultiConAD/
├── audio_transcription/       # Automatic audio transcription (necessary for working with Taukadial)
├── data_augmentation/         # Synthetic data generation with Gemini and Mistral
├── experiments/               # Classifiers: TF-IDF, E5, BERT
├── extracting_data/           # Parsing and normalization of CHAT files
├── preprocessing_text/        # Text cleaning and conversion of CHAT markers
├── scripts/
│   ├── datasets_creation/     # Parsing, cleaning, and JSONL creation
├── environment.yml            # Conda environment dependencies
```

---

## Pipeline

```
CHAT files (.cha)
    │
    ▼
extracting_data/          ← Parsing and normalization to JSONL
    │
    ▼
preprocessing_text/       ← Text cleaning + CHAT markers → special tokens
    │
    ▼
┌───────────────────────────────────────────────┐
│               EXPERIMENTS                     │
│  TF-IDF baselines → E5 embeddings → BERT      │
└───────────────────────────────────────────────┘
    │
    ▼
data_augmentation/        ← Synthetic generation (Gemini / Mistral)
    │
    ▼
experiments/BERT_balanced.py   ← Training with real + synthetic data + cross-dataset
```

---

## Experiments

| # | Experiment | Main script |
|---|---|---|
| 1 | Baselines TF-IDF (SVM, RF, NB, DT, LR) | `experiments/TF_IDF_classifier.py` |
| 2 | Dense embeddings E5                    | `experiments/e5_larg_classifier.py` |
| 3 | TF-IDF por dataset individual (balanceado/no balanceado) | `experiments/TF_IDF_single_classifier.py` |
| 4 | BERT fine-tuning | `experiments/BERT_classification.py` |
| 5 | BERT with CHAT tokens (`[PAUSE]`, `[REP]`, `[REF]`) | `experiments/BERT_tokenizer.py` |
| 6 | BERT cross-dataset and synthetic | `experiments/BERT_beyond.py` |
| 7 | Synthetic generation (0–100 % real data) | `data_augmentation/syn_gen_*.py` |
| 8 | Cross-dataset analysis | `experiments/cross_task_analysis.py` |

---

### Instalación

```bash
git clone URL_TO_REPO
cd ReproMultiConAD
conda env create -f environment.yml
conda activate repro_multi
```

## Running Locally

Examples of how to run the main experiments reported in the paper:

```bash
# Reproducibility of MultiConAD
python experiments/TF_IDF_classifier.py --test_language en --task binary --translated no
python experiments/e5_larg_classifier.py --test_language en --task multiclass

# Performance across individual corpora
python experiments/TF_IDF_single_classifier.py --dataset pitt --task binary

# BERT-based models
python experiments/BERT_classification.py --language en --task binary 
python experiments/BERT_tokenizer.py --language en --task binary --mode ref

# Cross-dataset transfer
python experiments/BERT_beyond.py --mode cross --train-dataset pitt --test-dataset wls --task binary

# Synthetic data
python experiments/BERT_beyond.py --mode synthetic --real-percentage 80 --task binary

# Synthetic generation
python data_augmentation/syn_gen_gemini.py --dataset pitt --real-percentage 20
python data_augmentation/syn_gen_sintetica_mistral.py --dataset pitt --real-percentage 20
```
---

## Train/test splits

For the reproduction experiments, we followed the official experimental setup provided by MultiConAD. For the individual experiments, we used the following training and test splits for each corpus. The table also reports the proportion of the majority class in the test set, providing an indication of class imbalance across corpora.

| Corpus | Training Size | Test Size | Majority Class Proportion |
|---|---:|---:|---:|
| Delaware | 255 | 64 | 65.6% |
| Ivanova | 285 | 72 | 54.2% |
| Lu | 40 | 11 | 54.5% |
| Pitt | 390 | 108 | 54.6% |
| Taukadial | 186 | 60 | 50.0% |
| VAS | 80 | 20 | 65.0% |
| WLS | 1,095 | 273 | 81.0% |

WLS is the most imbalanced corpus, with the majority class accounting for 81.0% of the test set.

## Hyperparameters

For the classical machine learning classifiers, hyperparameters were selected using 5-fold cross-validation on the training set. We evaluated the following predefined parameter grids:

| Classifier | Hyperparameters |
|---|---|
| Decision Tree | `max_depth`: [10, 20, 30] |
| Random Forest | `n_estimators`: [50, 100, 200] |
| SVM | `C`: [0.1, 1, 10], `kernel`: [`linear`, `rbf`] |
| Logistic Regression | `C`: [0.1, 1, 10] |

The best hyperparameter configuration for each classifier was selected based on the cross-validation results.

For BERT-based models, we used a fixed configuration across experiments to ensure consistent comparisons:

| Hyperparameter | Value |
|---|---:|
| Maximum sequence length | 256 |
| Batch size | 16 |
| Learning rate | 5e-5 |
| Number of epochs | 3 |


### CHAT Markers as Special Tokens

| Phenomenon | Original notation | Token |
|---|---|---|
| Pauses | `(.)`, `(..)`, `(1.2)` | `[PAUSE]` |
| Repetitions | `[/]` | `[REP]` |
| Reformulations | `[//]` | `[REF]` |

The `--mode` argument controls which tokens are enabled: `pause`, `rep`, `ref`, `all`, or `none` (no markers, controlled reference). SHAP analysis saves the HTML files and aggregated Excel files to `results/BERT_tokenizer/`. For Spanish, only `[REP]` is analyzed, as the Ivanova dataset does not provide useful coverage of pauses or reformulations.

---

### Generation with Gemini

Create a `.env` file in the repository root:

```
GEMINI_API_KEY=your_key_here
```

### Generation with Mistral (Ollama)

```
ollama pull mistral-small3.2

ollama serve   # must be running during execution
```

**Prompts** can be found in `Data_augmentation/prompt_system.py` 

#### Reference to Our Study

To appear
