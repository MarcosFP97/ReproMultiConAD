"""
Fine-tuning BERT estándar para clasificación binaria (HC/Dementia) o multiclase (HC/MCI/Dementia)
sobre transcripciones conversacionales en inglés o español.

Basado en el loop de entrenamiento manual con DataLoaders (no usa HuggingFace Trainer).
Para la versión con pérdida ponderada y Trainer, ver BERT_balanced.py.
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Entrenamiento de BERT con conjuntos de datos en español/inglés")
parser.add_argument("--language", type=str, required=True, help="Idioma (en o spa)")
parser.add_argument("--task", type=str, required=True, help="Tipo de clasificación (binary o multiclass)")
args = parser.parse_args()

language = args.language
task = args.task

MODEL_NAME = "bert-base-uncased"
TRAIN_PATH = f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/train_{language}_e5.jsonl"
TEST_PATH  = f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/test_{language}_e5.jsonl"
OUTPUT_DIR = f"/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/BERT_Models/bert_{language}_{task}_len256"

TEXT_COL  = "Text_interviewer_participant"
LABEL_COL = "Diagnosis"

MAX_LEN    = 256
BATCH_SIZE = 16
LR         = 5e-5
EPOCHS     = 3

if task == "binary" :
    DROP_LABEL_VALUE = "MCI"
elif task == "multiclass" :
    DROP_LABEL_VALUE = None

VERBOSE = True


class ClassificationDataset(Dataset):
    """
    Representación de cada ejemplo:
      - Texto (string)
      - Etiqueta (int)
      - tokenizer(...) produce:
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

        # encoding["input_ids"] tiene shape [1, max_len] -> quitamos la dimensión 0
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

def load_and_prepare_df(path, text_col, label_col, drop_label_value=None):
    df = pd.read_json(path, lines=True)

    df = df[[label_col, text_col]].copy()

    if drop_label_value is not None:
        before = len(df)
        df = df[df[label_col] != drop_label_value].copy()
        after = len(df)
        if VERBOSE:
            print(f"[DATA] Filter '{drop_label_value}': {before} -> {after} rows")

    df[text_col] = df[text_col].astype(str)

    return df

def encode_labels_fit(df, label_col):
    """
    Ajusta LabelEncoder con train y crea columna df['label'] con ints.
    """
    le = LabelEncoder()
    df = df.copy()
    df["label"] = le.fit_transform(df[label_col])

    if VERBOSE:
        print("\n[LABEL ENCODING] classes_ (orden -> id):")
        for i, c in enumerate(le.classes_):
            print(f"  {i} -> {c}")

        print("\n[LABEL ENCODING] distribución en train_df (por nombre):")
        print(df[label_col].value_counts())

        print("\n[LABEL ENCODING] distribución en train_df (por id):")
        print(df["label"].value_counts().sort_index())

    return df, le

def encode_labels_transform(df, label_col, label_encoder: LabelEncoder):
    """
    Usa el LabelEncoder del train para transformar etiquetas del test.
    """
    df = df.copy()

    unseen = set(df[label_col].unique()) - set(label_encoder.classes_)
    if unseen:
        raise ValueError(
            f"Etiquetas en TEST que no existen en TRAIN: {unseen}. "
            "Asegúrate de que train tenga todas las clases."
        )

    df["label"] = label_encoder.transform(df[label_col])
    return df

def split_train_val(df, text_col, label_encoded_col="label", test_size=0.2, random_state=42):
    """
    Divide el train en train/val para controlar el aprendizaje durante el fine-tuning.
    stratify mantiene proporciones de clase.
    """
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df[text_col].values,
        df[label_encoded_col].values,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_encoded_col].values
    )

    if VERBOSE:
        print(f"\n[SPLIT] Train size: {len(train_texts)} | Val size: {len(val_texts)}")
        import numpy as np
        print("[SPLIT] Class distribution (train):", dict(zip(*np.unique(train_labels, return_counts=True))))
        print("[SPLIT] Class distribution (val):  ", dict(zip(*np.unique(val_labels, return_counts=True))))

    return train_texts, val_texts, train_labels, val_labels

def build_loader(texts, labels, tokenizer, max_len=128, batch_size=16, shuffle=False):
    dataset = ClassificationDataset(texts, labels, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(model_name, num_labels, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    return model

def build_optimizer(model, lr=5e-5):
    return AdamW(model.parameters(), lr=lr)

def train_one_epoch(model, train_loader, optimizer, device, epoch_idx, epochs_total):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, leave=True)

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)             # [B, max_len]
        attention_mask = batch["attention_mask"].to(device)   # [B, max_len]
        labels = batch["label"].to(device)                    # [B]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_description(f"Epoch {epoch_idx+1}/{epochs_total}")
        loop.set_postfix(loss=loss.item())

    return total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return (correct / total) if total > 0 else 0.0

def predict_labels(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return y_true, y_pred

def predict_with_probs(model, data_loader, device):
    model.eval()
    all_true = []
    all_pred = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)  # [B, num_labels]
            preds = torch.argmax(probs, dim=1)

            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return all_true, all_pred, all_probs

def save_artifacts(model, tokenizer, label_encoder, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(label_encoder, os.path.join(out_dir, "label_encoder.pth"))

    if VERBOSE:
        print(f"\n[SAVE] Model + tokenizer + label_encoder saved to:\n  {out_dir}")


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
    print("  labels:        ", tuple(batch["label"].shape))          # [B] 

def compute_truncation_pct(df: pd.DataFrame, tokenizer, text_col: str, max_len: int) -> float:
    lengths = df[text_col].apply(lambda x: len(tokenizer.tokenize(str(x))))
    return float((lengths > max_len).mean() * 100.0)

def build_experiment_row(*,y_true: list,y_pred: list,class_names: list,exp_meta: dict,) -> pd.DataFrame:
    """
    Devuelve un DF de 1 fila con:
    - metadatos del experimento
    - accuracy, macro/weighted precision/recall/f1
    - métricas por clase (precision/recall/f1/support)
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

def save_results_excel(out_path: str,*,summary_row_df: pd.DataFrame,y_true: list,y_pred: list,class_names: list):
    """
    Excel:
      - summary: 1 fila con métricas + metadatos
      - confusion_matrix: matriz con labels
      - top_errors: (opcional) errores más confiados
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

    print(f"[SAVE] Excel results saved to: {out_path}")

def main():
    print("========== BERT TEXT CLASSIFICATION PIPELINE ==========")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[STEP 1] Loading train/test data (jsonl) and selecting needed columns...")
    train_df = load_and_prepare_df(TRAIN_PATH, TEXT_COL, LABEL_COL, DROP_LABEL_VALUE)
    test_df  = load_and_prepare_df(TEST_PATH,  TEXT_COL, LABEL_COL, DROP_LABEL_VALUE)
    print(f"[DATA] Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    print("\n[STEP 2] Label encoding using TRAIN only...")
    train_df, label_encoder = encode_labels_fit(train_df, label_col=LABEL_COL)
    test_df = encode_labels_transform(test_df, label_col=LABEL_COL, label_encoder=label_encoder)
    
    print("\n[STEP 3] Splitting TRAIN into train/val...")
    train_texts, val_texts, train_labels, val_labels = split_train_val(train_df, text_col=TEXT_COL)

    print("\n[STEP 4] Loading tokenizer and describing tokenization...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if VERBOSE:
        describe_token_lengths(train_df, tokenizer, TEXT_COL, MAX_LEN)

    print("\n[STEP 5] Building DataLoaders...")
    train_loader = build_loader(train_texts, train_labels, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = build_loader(val_texts, val_labels, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = build_loader(test_df[TEXT_COL].values, test_df["label"].values, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE, shuffle=False)

    if VERBOSE:
        inspect_one_batch(train_loader)

    print("\n[STEP 6] Building model and optimizer...")
    device = get_device()
    print(f"[DEVICE] Using: {device}")
    if device.type == "cuda":
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(MODEL_NAME, num_labels=len(label_encoder.classes_), device=device)
    optimizer = build_optimizer(model, lr=LR)

    print("\n[STEP 7] Training...")
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, EPOCHS)
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"[EPOCH {epoch+1}] Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    print("\n[STEP 8] Final evaluation on TEST...")
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"[TEST] Accuracy: {test_acc:.4f}")

    y_true, y_pred, y_probs = predict_with_probs(model, test_loader, device)

    id2label = {i: c for i, c in enumerate(label_encoder.classes_)}

    # reconstruimos los textos en el mismo orden que el test_loader
    test_texts = test_df[TEXT_COL].values.tolist()

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
    
    results_dir = "/mnt/beegfs/groups/irgroup/sara_tfg/results/"
    os.makedirs(results_dir, exist_ok=True)

    trunc_pct = compute_truncation_pct(train_df, tokenizer, TEXT_COL, MAX_LEN)

    exp_meta = {
        "Model": MODEL_NAME,
        "Max_len": MAX_LEN,
        "Batch_size": BATCH_SIZE,
        "LR": LR,
        "Epochs": EPOCHS,
        "Drop_label_value": str(DROP_LABEL_VALUE),
        "Train_rows": len(train_df),
        "Test_rows": len(test_df),
        "Train_trunc_pct": trunc_pct,
        "Train_path": TRAIN_PATH,
        "Test_path": TEST_PATH,
        "Output_dir": OUTPUT_DIR,
    }

    summary_row = build_experiment_row(
        y_true=y_true,
        y_pred=y_pred,
        class_names=list(label_encoder.classes_),
        exp_meta=exp_meta
    )

    out_xlsx = os.path.join(results_dir, f"BERT_{language}_{task}.xlsx")

    save_results_excel(
        out_xlsx,
        summary_row_df=summary_row,
        y_true=y_true,
        y_pred=y_pred,
        class_names=list(label_encoder.classes_)
    )

    n_total = len(pred_df)
    n_wrong = int((~pred_df["correct"]).sum())
    print(f"\n[ERRORS] Wrong predictions: {n_wrong}/{n_total} ({(n_wrong/n_total)*100:.2f}%)")

    # muestra los 20 fallos más “seguros” (alta confianza pero equivocado)
    wrong_df = pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False)

    print("\n[ERRORS] Top 20 most confident WRONG examples:")
    for i, row in wrong_df.head(20).iterrows():
        txt = row["text"]
        if len(txt) > 300:
            txt = txt[:300] + "..."
        print(f"- true={row['true_label']} | pred={row['pred_label']} | conf={row['confidence']:.3f} | {txt}")

    print("\n[TEST] Classification report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    print("[TEST] Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


    print("\n[STEP 9] Saving model artifacts...")
    save_artifacts(model, tokenizer, label_encoder, out_dir=OUTPUT_DIR)

    print("\n========== DONE ==========")

if __name__ == "__main__":

    main()
