"""
BERT_balanced.py — Fine-tuning BERT with class-weighted loss for cognitive
impairment detection from conversational transcripts.

Supports three experimental modes:
  - individual : trains and evaluates on the same dataset (internal 80/20 split).
  - synthetic   : trains with different proportions of real and synthetic data.
  - cross_task  : trains on train+test from one dataset and evaluates on external datasets.
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer, TrainingArguments
from torch import nn
import torch.nn.functional as F

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

DATA_ROOT = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl"
MODEL_ROOT = "/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/BERT_Models"
RESULTS_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/results/BERT_synthetic_analysis"
SPANISH_DATASETS = {"ivanova", "perla"}

parser = argparse.ArgumentParser(
    description="Balanced BERT training for individual, cross-dataset, or synthetic experiments."
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["individual", "synthetic", "cross_task"],
    default="individual",
    help="Experimental design: individual, synthetic, or cross_task.",
)
parser.add_argument(
    "--train-dataset",
    type=str,
    required=True,
    help="Dataset used for training (e.g., ivanova, pitt, taukadial).",
)
parser.add_argument(
    "--cross-test-datasets",
    nargs="*",
    default=[],
    help=(
        "External test datasets for mode=cross_task "
        "(e.g., --cross-test-datasets wls taukadial)."
    ),
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    choices=["binary", "multiclass"],
    help="Task type.",
)
parser.add_argument(
    "--binary-task",
    type=str,
    choices=["hc_dementia", "hc_mci", "disease_status"],
    default=None,
    help=(
        "Explicit definition of the binary task. "
        "hc_dementia keeps HC/Dementia; hc_mci keeps HC/MCI; "
        "disease_status maps HC=NoDisease and MCI/Dementia=Disease. "
        "If not specified for task=binary, hc_dementia is used for compatibility."
    ),
)
parser.add_argument(
    "--train-source",
    type=str,
    choices=["real", "synthetic", "augmented"],
    default="augmented",
    help="Training source in mode=synthetic: real, synthetic, or augmented.",
)
parser.add_argument(
    "--real-percentage",
    type=int,
    default=None,
    help="Percentage of real data used in mode=synthetic.",
)
parser.add_argument(
    "--synthetic-percentage",
    type=int,
    default=None,
    help="Percentage of synthetic data used in mode=synthetic.",
)
parser.add_argument(
    "--synthetic-source",
    type=str,
    choices=["mistral", "gemini"],
    default="mistral",
    help="LLM used for synthetic files in mode=synthetic.",
)
args = parser.parse_args()

mode = args.mode
train_dataset = args.train_dataset.strip().lower()
test_dataset = train_dataset
cross_test_datasets = []
for dataset_name in args.cross_test_datasets:
    dataset_name = dataset_name.strip().lower()
    if dataset_name and dataset_name not in cross_test_datasets:
        cross_test_datasets.append(dataset_name)
task = args.task
train_source = args.train_source
binary_task = args.binary_task

if mode == "cross_task":
    if task != "binary":
        parser.error("--mode cross_task requires --task binary.")
    if binary_task is not None and binary_task != "disease_status":
        parser.error("--mode cross_task requires --binary-task disease_status.")
    binary_task = "disease_status"
elif task == "binary":
    binary_task = binary_task or "hc_dementia"
else:
    if binary_task is not None:
        parser.error("--binary-task can only be used with --task binary.")
    binary_task = None

BINARY_TASKS = {
    "hc_dementia": {
        "keep_labels": ["HC", "Dementia"],
        "label_mapping": None,
    },
    "hc_mci": {
        "keep_labels": ["HC", "MCI"],
        "label_mapping": None,
    },
    "disease_status": {
        "keep_labels": ["HC", "MCI", "Dementia"],
        "label_mapping": {
            "HC": "NoDisease",
            "MCI": "Disease",
            "Dementia": "Disease",
        },
    },
}

BINARY_TASK_CONFIG = BINARY_TASKS[binary_task] if binary_task else {}
KEEP_LABEL_VALUES = BINARY_TASK_CONFIG.get("keep_labels")
LABEL_MAPPING = BINARY_TASK_CONFIG.get("label_mapping")
EXPECTED_LABEL_VALUES = sorted(set(LABEL_MAPPING.values())) if LABEL_MAPPING else KEEP_LABEL_VALUES
EXPERIMENT_TASK = task if binary_task is None else f"{task}_{binary_task}"

if mode == "cross_task" and not cross_test_datasets:
    parser.error("--mode cross_task requires at least one dataset in --cross-test-datasets.")

if mode != "cross_task" and cross_test_datasets:
    parser.error("--cross-test-datasets can only be used with --mode cross_task.")

if mode != "synthetic":
    if args.real_percentage is not None or args.synthetic_percentage is not None:
        parser.error("--real-percentage and --synthetic-percentage can only be used with --mode synthetic.")

if args.real_percentage is not None and not 0 <= args.real_percentage <= 100:
    parser.error("--real-percentage must be in the range 0..100.")

if args.synthetic_percentage is not None and not 0 <= args.synthetic_percentage <= 100:
    parser.error("--synthetic-percentage must be in the range 0..100.")


def real_train_path(dataset_name: str, percentage: int) -> str:
    if percentage == 100:
        return os.path.join(DATA_ROOT, "individual_sets", f"train_{dataset_name}.jsonl")
    return os.path.join(DATA_ROOT, "synthetic_data", "real", f"train_{dataset_name}_real{percentage}.jsonl")


def synthetic_train_path(dataset_name: str, percentage: int, source: str) -> str:
    return os.path.join(DATA_ROOT, "synthetic_data", "synthetic", f"train_{dataset_name}_synthetic{percentage}_{source}.jsonl")

if mode == "individual":
    TRAIN_PATHS = [os.path.join(DATA_ROOT, "individual_sets", f"train_{train_dataset}.jsonl")]
    TEST_PATH = os.path.join(DATA_ROOT, "individual_sets", f"test_{test_dataset}.jsonl")
    EXPERIMENT_ID = f"bert_balanced_individual_{EXPERIMENT_TASK}_{train_dataset}"
elif mode == "cross_task":
    TRAIN_PATHS = [
        os.path.join(DATA_ROOT, "individual_sets", f"train_{train_dataset}.jsonl"),
        os.path.join(DATA_ROOT, "individual_sets", f"test_{train_dataset}.jsonl"),
    ]
    TEST_PATH = None
    test_dataset = None
    EXPERIMENT_ID = f"bert_balanced_cross_task_{EXPERIMENT_TASK}_{train_dataset}_full"
else:
    if train_source == "real":
        if args.real_percentage is None:
            parser.error("--real-percentage is required with --train-source real.")
        if args.synthetic_percentage is not None:
            parser.error("--synthetic-percentage cannot be used with --train-source real.")
        real_percentage = args.real_percentage
        synthetic_percentage = None
        TRAIN_PATHS = [real_train_path(train_dataset, real_percentage)]
        EXPERIMENT_ID = f"bert_balanced_real_{EXPERIMENT_TASK}_{train_dataset}_real{real_percentage}"

    elif train_source == "synthetic":
        if args.synthetic_percentage is None:
            parser.error("--synthetic-percentage is required with --train-source synthetic.")
        if args.real_percentage is not None:
            parser.error("--real-percentage cannot be used with --train-source synthetic.")
        real_percentage = None
        synthetic_percentage = args.synthetic_percentage
        TRAIN_PATHS = [synthetic_train_path(train_dataset, synthetic_percentage, args.synthetic_source)]
        EXPERIMENT_ID = f"bert_balanced_synthetic_{args.synthetic_source}_{EXPERIMENT_TASK}_{train_dataset}_synthetic{synthetic_percentage}"

    else:
        if args.real_percentage is None or args.synthetic_percentage is None:
            parser.error("--real-percentage and --synthetic-percentage are required with --train-source augmented.")
        real_percentage = args.real_percentage
        synthetic_percentage = args.synthetic_percentage
        if real_percentage + synthetic_percentage != 100:
            parser.error("In --train-source augmented, real-percentage + synthetic-percentage must sum to 100.")

        TRAIN_PATHS = []
        if real_percentage > 0:
            TRAIN_PATHS.append(real_train_path(train_dataset, real_percentage))
        if synthetic_percentage > 0:
            TRAIN_PATHS.append(synthetic_train_path(train_dataset, synthetic_percentage, args.synthetic_source))

        EXPERIMENT_ID = (
            f"bert_balanced_augmented_{args.synthetic_source}_{EXPERIMENT_TASK}_{train_dataset}"
            f"_real{real_percentage}_synthetic{synthetic_percentage}"
        )

    TEST_PATH = os.path.join(DATA_ROOT, "individual_sets", f"test_{test_dataset}.jsonl")

OUTPUT_DIR = os.path.join(MODEL_ROOT, EXPERIMENT_ID)

TEXT_COL  = "Text_interviewer_participant"
LABEL_COL = "Diagnosis"

if train_dataset in SPANISH_DATASETS:
    MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
else:
    MODEL_NAME = "bert-base-uncased"
    
# MAX_LEN=256 covers the ~90th percentile of lengths in the clinical corpora used
# without the memory/computation cost of 512 tokens; LR=5e-5 and EPOCHS=3 follow the
# original BERT fine-tuning recommendations (Devlin et al., 2019).
MAX_LEN    = 256
BATCH_SIZE = 16
LR         = 5e-5
EPOCHS     = 3

VERBOSE = True


class ClassificationDataset(Dataset):
    """
    Representation of each example:
      - Text (string)
      - Label (int)
      - tokenizer(...) produces:
          input_ids:      tensor [max_len]
          attention_mask: tensor [max_len]
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # HuggingFace Trainer expects the key "labels" (not "label")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def load_and_prepare_df(paths, text_col, label_col, keep_label_values=None, label_mapping=None):
    if isinstance(paths, (list, tuple)):
        frames = [pd.read_json(path, lines=True) for path in paths]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_json(paths, lines=True)

    df = df[[label_col, text_col]].copy()

    if keep_label_values is not None:
        before = len(df)
        df = df[df[label_col].isin(keep_label_values)].copy()
        after = len(df)
        if VERBOSE:
            print(f"[DATA] Keep labels {keep_label_values}: {before} -> {after} rows")

    if label_mapping is not None:
        df[label_col] = df[label_col].map(label_mapping)

    df[text_col] = df[text_col].astype(str)

    return df

def validate_label_coverage(df, label_col, expected_labels, split_name, path_description):
    """Fails explicitly if an expected class is missing from the split.

    Prevents silently trivial evaluations (e.g., cross-dataset settings with
    incompatible classes that end up at zero after filtering).
    """
    if not expected_labels:
        return

    present_labels = set(df[label_col].dropna().unique())
    missing_labels = [label for label in expected_labels if label not in present_labels]

    if missing_labels:
        distribution = df[label_col].value_counts().to_dict()
        raise ValueError(
            f"After filtering {expected_labels}, {split_name} does not contain all classes. "
            f"Missing: {missing_labels}. Current distribution: {distribution}. "
            f"Source: {path_description}"
        )

def encode_labels_fit(df, label_col):
    """
    Fits the LabelEncoder on train data and creates a df['label'] column with integer values.
    """
    le = LabelEncoder()
    df = df.copy()
    df["label"] = le.fit_transform(df[label_col])

    if VERBOSE:
        print("\n[LABEL ENCODING] classes_ (order -> id):")
        for i, c in enumerate(le.classes_):
            print(f"  {i} -> {c}")

        print("\n[LABEL ENCODING] distribution in train_df (by name):")
        print(df[label_col].value_counts())

        print("\n[LABEL ENCODING] distribution in train_df (by id):")
        print(df["label"].value_counts().sort_index())

    return df, le

def encode_labels_transform(df, label_col, label_encoder: LabelEncoder):
    """
    Uses the train LabelEncoder to transform the test labels.
    """
    df = df.copy()

    unseen = set(df[label_col].unique()) - set(label_encoder.classes_)
    if unseen:
        raise ValueError(
            f"Labels in TEST that do not exist in TRAIN: {unseen}. "
            "Make sure train contains all classes."
        )

    df["label"] = label_encoder.transform(df[label_col])
    return df

def split_train_val(df, text_col, label_encoded_col="label", test_size=0.2, random_state=42):
    """
    Splits the train set into train/val to monitor learning during fine-tuning.
    stratify preserves class proportions when possible.
    """
    labels = df[label_encoded_col].values
    class_counts = pd.Series(labels).value_counts()
    n_classes = int(class_counts.shape[0])

    if isinstance(test_size, float):
        val_size_est = int(np.ceil(len(df) * test_size))
    else:
        val_size_est = int(test_size)

    can_stratify = (class_counts.min() >= 2) and (val_size_est >= n_classes)
    stratify_labels = labels if can_stratify else None

    if VERBOSE and not can_stratify:
        print(
            "[SPLIT][WARN] Stratification cannot be applied safely "
            f"(min_class_count={int(class_counts.min())}, n_classes={n_classes}, "
            f"val_size_est={val_size_est}). A split without stratification will be used."
        )

    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df[text_col].values,
            labels,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=random_state,
        )
    except ValueError as e:
        if stratify_labels is not None:
            if VERBOSE:
                print(f"[SPLIT][WARN] Stratified split failed ({e}). Retrying without stratify.")
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                df[text_col].values,
                labels,
                test_size=test_size,
                stratify=None,
                random_state=random_state,
            )
        else:
            raise

    if VERBOSE:
        print(f"\n[SPLIT] Train size: {len(train_texts)} | Val size: {len(val_texts)}")
        print("[SPLIT] Class distribution (train):", dict(zip(*np.unique(train_labels, return_counts=True))))
        print("[SPLIT] Class distribution (val):  ", dict(zip(*np.unique(val_labels, return_counts=True))))

    return train_texts, val_texts, train_labels, val_labels

def build_loader(texts, labels, tokenizer, max_len=128, batch_size=16, shuffle=False):
    dataset = ClassificationDataset(texts, labels, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

class CustomTrainer(Trainer):
    """Extends HuggingFace Trainer to apply class-weighted loss.

    In imbalanced clinical datasets (HC >> MCI or Dementia), standard CrossEntropyLoss
    tends to optimize the majority class. Weights inversely proportional to class
    frequency correct this bias.
    """
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None and self.model is not None:
            self.class_weights = self.class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(model_name, num_labels, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    return model

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro_f1}

def describe_token_lengths(df, tokenizer, text_col, max_len):
    lengths = df[text_col].apply(lambda x: len(tokenizer.tokenize(str(x))))
    print("\n[TOKENS] Token length stats (WordPiece tokens):")
    print(lengths.describe())

    trunc_pct = (lengths > max_len).mean() * 100
    print(f"[TOKENS] % samples that will be truncated at MAX_LEN={max_len}: {trunc_pct:.2f}%")

    sample_text = str(df[text_col].iloc[0])
    tokens = tokenizer.tokenize(sample_text)[:50]
    print("\n[TOKENS] Example tokenization (first 50 tokens of first sample):")
    print(tokens)

    ids = tokenizer.convert_tokens_to_ids(tokens)
    print("\n[TOKENS] Corresponding token ids (first 50):")
    print(ids)

def inspect_one_batch(loader):
    batch = next(iter(loader))
    print("\n[BATCH] Example batch shapes:")
    print("  input_ids:     ", tuple(batch["input_ids"].shape))      # [B, max_len]
    print("  attention_mask:", tuple(batch["attention_mask"].shape)) # [B, max_len]
    print("  labels:        ", tuple(batch["labels"].shape))         # [B]


def compute_truncation_pct(df: pd.DataFrame, tokenizer, text_col: str, max_len: int) -> float:
    lengths = df[text_col].apply(lambda x: len(tokenizer.tokenize(str(x))))
    return float((lengths > max_len).mean() * 100.0)

def build_experiment_row(*, y_true: list, y_pred: list, class_names: list, exp_meta: dict,) -> pd.DataFrame:
    """
    Returns a 1-row DataFrame with:
    - experiment metadata
    - accuracy, macro/weighted precision/recall/f1
    - per-class metrics (precision/recall/f1/support)
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    row = dict(exp_meta)

    row["Accuracy"] = report.get("accuracy", np.nan)

    row["Macro_precision"] = report["macro avg"]["precision"]
    row["Macro_recall"]    = report["macro avg"]["recall"]
    row["Macro_f1"]        = report["macro avg"]["f1-score"]

    row["Weighted_precision"] = report["weighted avg"]["precision"]
    row["Weighted_recall"]    = report["weighted avg"]["recall"]
    row["Weighted_f1"]        = report["weighted avg"]["f1-score"]

    for cls in class_names:
        row[f"{cls}_precision"] = report[cls]["precision"]
        row[f"{cls}_recall"]    = report[cls]["recall"]
        row[f"{cls}_f1"]        = report[cls]["f1-score"]
        row[f"{cls}_support"]   = report[cls]["support"]

    return pd.DataFrame([row])

def save_results_excel(
    out_path: str,
    *,
    summary_row_df: pd.DataFrame,
    y_true: list,
    y_pred: list,
    class_names: list,
    predictions_df: Optional[pd.DataFrame] = None,
    baseline_y_pred: Optional[list] = None,
):
    """
    Excel:
      - summary: 1 row with metrics + metadata
      - confusion_matrix: matrix with labels
      - baseline_confusion_matrix: majority-baseline confusion matrix if applicable
      - predictions: row-by-row predictions for cross-task analysis
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_row_df.to_excel(writer, index=False, sheet_name="summary")
        cm_df.to_excel(writer, sheet_name="confusion_matrix")
        if baseline_y_pred is not None:
            baseline_cm = confusion_matrix(y_true, baseline_y_pred, labels=list(range(len(class_names))))
            baseline_cm_df = pd.DataFrame(
                baseline_cm,
                index=[f"true_{c}" for c in class_names],
                columns=[f"pred_{c}" for c in class_names],
            )
            baseline_cm_df.to_excel(writer, sheet_name="baseline_confusion_matrix")
        if predictions_df is not None:
            predictions_df.to_excel(writer, index=False, sheet_name="predictions")

    print(f"[SAVE] Excel results saved to: {out_path}")

def main():
    print("========== BERT TEXT CLASSIFICATION PIPELINE ==========")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[STEP 1] Loading train/test data (jsonl) and selecting needed columns...")
    print("[DATA] Train paths:")
    for path in TRAIN_PATHS:
        print(f"  - {path}")
    if TEST_PATH is not None:
        print(f"[DATA] Test path: {TEST_PATH}")
    else:
        print("[DATA] Test path: none (cross_task trains with full source dataset)")
    train_df = load_and_prepare_df(TRAIN_PATHS, TEXT_COL, LABEL_COL, KEEP_LABEL_VALUES, LABEL_MAPPING)
    test_df = None
    if TEST_PATH is not None:
        test_df = load_and_prepare_df(TEST_PATH, TEXT_COL, LABEL_COL, KEEP_LABEL_VALUES, LABEL_MAPPING)
    validate_label_coverage(train_df, LABEL_COL, EXPECTED_LABEL_VALUES, "TRAIN", " | ".join(TRAIN_PATHS))
    if test_df is not None:
        validate_label_coverage(test_df, LABEL_COL, EXPECTED_LABEL_VALUES, "TEST", TEST_PATH)
    print(f"[DATA] Train rows: {len(train_df)} | Test rows: {len(test_df) if test_df is not None else 0}")

    print("\n[STEP 2] Label encoding using TRAIN only...")
    train_df, label_encoder = encode_labels_fit(train_df, label_col=LABEL_COL)
    if test_df is not None:
        test_df = encode_labels_transform(test_df, label_col=LABEL_COL, label_encoder=label_encoder)
    
    print("\n[STEP 3] Splitting TRAIN into train/val...")
    train_texts, val_texts, train_labels, val_labels = split_train_val(train_df, text_col=TEXT_COL)

    print("\n[STEP 4] Loading tokenizer and describing tokenization...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if VERBOSE:
        describe_token_lengths(train_df, tokenizer, TEXT_COL, MAX_LEN)

    print("\n[STEP 5] Calculating class weights for imbalanced data...")
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float)
    if VERBOSE:
        print(f"[WEIGHTS] Class weights: {class_weights_tensor}")

    include_majority_baseline = mode == "cross_task"
    majority_baseline = None
    if include_majority_baseline:
        majority_baseline = DummyClassifier(strategy="most_frequent")
        majority_baseline.fit(np.zeros((len(train_df), 1)), train_df["label"].values)
        majority_label_id = int(majority_baseline.predict([[0]])[0])
        majority_label = label_encoder.inverse_transform([majority_label_id])[0]
        print(f"[BASELINE] DummyClassifier(strategy='most_frequent') majority label: {majority_label}")

    print("\n[STEP 6] Building Datasets for Trainer...")
    train_hf_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_hf_dataset   = ClassificationDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    print("\n[STEP 7] Building model and CustomTrainer...")
    device = get_device()
    model = build_model(MODEL_NAME, num_labels=len(label_encoder.classes_), device=device)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1", # To avoid collapsing to the majority class in individual models
        greater_is_better=True,
        save_total_limit=1,
        logging_dir='./logs',
        logging_strategy="epoch",
        disable_tqdm=True,
        report_to="none",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=val_hf_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor
    )

    print("\n[STEP 8] Training with CustomTrainer...")
    trainer.train()

    trunc_pct = compute_truncation_pct(train_df, tokenizer, TEXT_COL, MAX_LEN)

    def evaluate_and_save(eval_df, eval_dataset_name, eval_test_path, eval_experiment_id, eval_mode):
        print(f"\n[STEP 9] Predicting on TEST set: {eval_dataset_name}...")
        eval_hf_dataset = ClassificationDataset(
            eval_df[TEXT_COL].values,
            eval_df["label"].values,
            tokenizer,
            MAX_LEN,
        )
        predictions_output = trainer.predict(eval_hf_dataset)

        y_true = predictions_output.label_ids
        logits = predictions_output.predictions
        y_probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        y_pred = np.argmax(y_probs, axis=1)
        baseline_y_pred = None
        if majority_baseline is not None:
            baseline_y_pred = majority_baseline.predict(np.zeros((len(y_true), 1)))

        test_acc = (y_true == y_pred).mean()
        print(f"[TEST:{eval_dataset_name}] Accuracy: {test_acc:.4f}")

        id2label = {i: c for i, c in enumerate(label_encoder.classes_)}
        test_texts = eval_df[TEXT_COL].values.tolist()

        rows = []
        for text, yt, yp, prob_vec in zip(test_texts, y_true, y_pred, y_probs):
            true_name = id2label[yt]
            pred_name = id2label[yp]
            confidence = float(max(prob_vec))
            prob_dict = {id2label[i]: float(prob_vec[i]) for i in range(len(prob_vec))}

            rows.append({
                "text": text,
                "true_id": yt,
                "pred_id": yp,
                "true_label": true_name,
                "pred_label": pred_name,
                "confidence": confidence,
                **{f"prob_{k}": v for k, v in prob_dict.items()}
            })

        pred_df = pd.DataFrame(rows)
        pred_df["correct"] = pred_df["true_id"] == pred_df["pred_id"]
        pred_df["dataset"] = eval_dataset_name
        pred_df["split"] = eval_mode
        pred_df["Train_dataset"] = train_dataset
        pred_df["Test_dataset"] = eval_dataset_name
        pred_df["Train_source"] = train_source if mode == "synthetic" else ""
        pred_df["Binary_task"] = binary_task or ""
        pred_df["Evaluation_experiment_id"] = eval_experiment_id
        pred_df["Training_experiment_id"] = EXPERIMENT_ID
        if baseline_y_pred is not None:
            pred_df["baseline_pred_id"] = baseline_y_pred
            pred_df["baseline_pred_label"] = [id2label[int(idx)] for idx in baseline_y_pred]
            pred_df["baseline_correct"] = pred_df["true_id"] == pred_df["baseline_pred_id"]

        os.makedirs(RESULTS_DIR, exist_ok=True)

        exp_meta = {
            "Estimator": "BERT_balanced",
            "Model": MODEL_NAME,
            "Max_len": MAX_LEN,
            "Batch_size": BATCH_SIZE,
            "LR": LR,
            "Epochs": EPOCHS,
            "Mode": eval_mode,
            "Train_source": train_source if mode == "synthetic" else "",
            "Train_dataset": train_dataset,
            "Test_dataset": eval_dataset_name,
            "Real_percentage": args.real_percentage if mode == "synthetic" else "",
            "Synthetic_percentage": args.synthetic_percentage if mode == "synthetic" else "",
            "Synthetic_source": args.synthetic_source if mode == "synthetic" else "",
            "Binary_task": binary_task or "",
            "Kept_label_values": " | ".join(KEEP_LABEL_VALUES) if KEEP_LABEL_VALUES else "",
            "Label_mapping": str(LABEL_MAPPING or ""),
            "Train_rows": len(train_df),
            "Test_rows": len(eval_df),
            "Train_trunc_pct": trunc_pct,
            "Train_path": " | ".join(TRAIN_PATHS),
            "Test_path": eval_test_path,
            "Training_experiment_id": EXPERIMENT_ID,
            "Evaluation_experiment_id": eval_experiment_id,
            "Output_dir": OUTPUT_DIR,
        }

        summary_row = build_experiment_row(
            y_true=y_true,
            y_pred=y_pred,
            class_names=list(label_encoder.classes_),
            exp_meta=exp_meta
        )
        if baseline_y_pred is not None:
            baseline_meta = dict(exp_meta)
            baseline_meta.update({
                "Estimator": "Dummy_most_frequent",
                "Model": "DummyClassifier(strategy=most_frequent)",
                "Output_dir": "",
            })
            baseline_summary_row = build_experiment_row(
                y_true=y_true,
                y_pred=baseline_y_pred,
                class_names=list(label_encoder.classes_),
                exp_meta=baseline_meta
            )
            summary_row = pd.concat([summary_row, baseline_summary_row], ignore_index=True)

        out_xlsx = os.path.join(RESULTS_DIR, f"{eval_experiment_id}.xlsx")
        save_results_excel(
            out_xlsx,
            summary_row_df=summary_row,
            y_true=y_true,
            y_pred=y_pred,
            class_names=list(label_encoder.classes_),
            predictions_df=pred_df,
            baseline_y_pred=baseline_y_pred,
        )

        n_total = len(pred_df)
        n_wrong = int((~pred_df["correct"]).sum())
        print(f"\n[ERRORS:{eval_dataset_name}] Wrong predictions: {n_wrong}/{n_total} ({(n_wrong/n_total)*100:.2f}%)")

        wrong_df = pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False)

        print(f"\n[ERRORS:{eval_dataset_name}] Top 20 most confident WRONG examples:")
        for _, row in wrong_df.head(20).iterrows():
            txt = row["text"]
            if len(txt) > 300:
                txt = txt[:300] + "..."
            print(f"- true={row['true_label']} | pred={row['pred_label']} | conf={row['confidence']:.3f} | {txt}")

        print(f"\n[TEST:{eval_dataset_name}] Classification report:")
        print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))
        print(f"[TEST:{eval_dataset_name}] Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        if baseline_y_pred is not None:
            baseline_acc = accuracy_score(y_true, baseline_y_pred)
            baseline_macro_f1 = f1_score(y_true, baseline_y_pred, average="macro", zero_division=0)
            print(f"[BASELINE:{eval_dataset_name}] Accuracy: {baseline_acc:.4f} | Macro-F1: {baseline_macro_f1:.4f}")
            print(f"[BASELINE:{eval_dataset_name}] Confusion matrix:")
            print(confusion_matrix(y_true, baseline_y_pred))

    if test_df is not None:
        evaluate_and_save(
            test_df,
            test_dataset,
            TEST_PATH,
            EXPERIMENT_ID,
            mode,
        )

    for cross_test_dataset in cross_test_datasets:
        cross_test_path = os.path.join(DATA_ROOT, "individual_sets", f"test_{cross_test_dataset}.jsonl")
        print(f"\n[CROSS] Loading extra test dataset: {cross_test_dataset}")
        cross_test_df = load_and_prepare_df(cross_test_path, TEXT_COL, LABEL_COL, KEEP_LABEL_VALUES, LABEL_MAPPING)
        validate_label_coverage(cross_test_df, LABEL_COL, EXPECTED_LABEL_VALUES, "CROSS TEST", cross_test_path)
        cross_test_df = encode_labels_transform(cross_test_df, label_col=LABEL_COL, label_encoder=label_encoder)
        cross_experiment_id = f"bert_balanced_cross_task_{EXPERIMENT_TASK}_train-{train_dataset}_full_test-{cross_test_dataset}"
        evaluate_and_save(
            cross_test_df,
            cross_test_dataset,
            cross_test_path,
            cross_experiment_id,
            "cross_task",
        )


    print("\n[STEP 10] Saving model artifacts...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    torch.save(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pth"))

    if VERBOSE:
        print(f"\n[SAVE] Model + tokenizer + label_encoder saved to:\n  {OUTPUT_DIR}")

    print("\n========== DONE ==========")

if __name__ == "__main__":

    main()
