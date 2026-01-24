import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm


# ============================================================
# CONFIG (cambia aquí lo que necesites)
# ============================================================

TRAIN_PATH = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/train_english_e5.jsonl"
TEST_PATH  = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/test_english_e5.jsonl"
OUTPUT_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_Models/bert_patient_classifier_len256"

TEXT_COL  = "Text_interviewer_participant"
LABEL_COL = "Diagnosis"

MODEL_NAME = "bert-base-uncased"
#MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LEN    = 256
BATCH_SIZE = 16
LR         = 5e-5
EPOCHS     = 3

DROP_LABEL_VALUE = "MCI"   # quitamos MCI para binario, como ya estabas haciendo
VERBOSE = True             # ponlo en False si quieres menos prints


# ============================================================
# 1) DATASET
# ============================================================

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
            add_special_tokens=True,     # añade [CLS] al inicio y [SEP] al final
            max_length=self.max_len,     # longitud fija
            padding="max_length",        # pad hasta max_len
            truncation=True,             # trunca si es más largo
            return_tensors="pt",         # devuelve tensores torch
        )

        # encoding["input_ids"] tiene shape [1, max_len] -> quitamos la dimensión 0
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# 2) DATA PREP
# ============================================================
def load_and_prepare_df(path, text_col, label_col, drop_label_value=None):
    df = pd.read_json(path, lines=True)

    # Nos quedamos SOLO con lo que necesitamos
    df = df[[label_col, text_col]].copy()

    # Filtrado (ej: quitar MCI)
    if drop_label_value is not None:
        before = len(df)
        df = df[df[label_col] != drop_label_value].copy()
        after = len(df)
        if VERBOSE:
            print(f"[DATA] Filter '{drop_label_value}': {before} -> {after} rows")

    # Aseguramos que texto sea string y sin NaN
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
        # Distribución por clase en train/val
        import numpy as np
        print("[SPLIT] Class distribution (train):", dict(zip(*np.unique(train_labels, return_counts=True))))
        print("[SPLIT] Class distribution (val):  ", dict(zip(*np.unique(val_labels, return_counts=True))))

    return train_texts, val_texts, train_labels, val_labels


def build_loader(texts, labels, tokenizer, max_len=128, batch_size=16, shuffle=False):
    dataset = ClassificationDataset(texts, labels, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


# ============================================================
# 3) MODEL SETUP
# ============================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name, num_labels, device):
    #model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    return model


def build_optimizer(model, lr=5e-5):
    return AdamW(model.parameters(), lr=lr)


# ============================================================
# 4) TRAIN / EVAL
# ============================================================
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
    all_probs = []  # lista de vectores [num_labels]

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


# ============================================================
# 5) SAVE
# ============================================================
def save_artifacts(model, tokenizer, label_encoder, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(label_encoder, os.path.join(out_dir, "label_encoder.pth"))

    if VERBOSE:
        print(f"\n[SAVE] Model + tokenizer + label_encoder saved to:\n  {out_dir}")


# ============================================================
# 6) EXTRAS: inspección de tokenización / longitudes
# ============================================================
def describe_token_lengths(df, tokenizer, text_col, max_len):
    lengths = df[text_col].apply(lambda x: len(tokenizer.tokenize(str(x))))
    print("\n[TOKENS] Token length stats (WordPiece tokens):")
    print(lengths.describe())

    trunc_pct = (lengths > max_len).mean() * 100
    print(f"[TOKENS] % samples that will be truncated at MAX_LEN={max_len}: {trunc_pct:.2f}%")

    # Ejemplo concreto de tokenización
    sample_text = str(df[text_col].iloc[0])
    tokens = tokenizer.tokenize(sample_text)[:50]
    print("\n[TOKENS] Example tokenization (first 50 tokens of first sample):")
    print(tokens)

    # Ejemplo de ids
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print("\n[TOKENS] Corresponding token ids (first 50):")
    print(ids)


def inspect_one_batch(loader):
    batch = next(iter(loader))
    print("\n[BATCH] Example batch shapes:")
    print("  input_ids:     ", tuple(batch["input_ids"].shape))      # [B, max_len]
    print("  attention_mask:", tuple(batch["attention_mask"].shape)) # [B, max_len]
    print("  labels:        ", tuple(batch["label"].shape))          # [B] 


# ============================================================
# MAIN
# ============================================================
def main():
    print("========== BERT TEXT CLASSIFICATION PIPELINE ==========")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Cargar datos
    print("\n[STEP 1] Loading train/test data (jsonl) and selecting needed columns...")
    train_df = load_and_prepare_df(TRAIN_PATH, TEXT_COL, LABEL_COL, drop_label_value=DROP_LABEL_VALUE)
    test_df  = load_and_prepare_df(TEST_PATH,  TEXT_COL, LABEL_COL, drop_label_value=DROP_LABEL_VALUE)
    print(f"[DATA] Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    # 2) Label encoding
    print("\n[STEP 2] Label encoding using TRAIN only...")
    train_df, label_encoder = encode_labels_fit(train_df, label_col=LABEL_COL)
    test_df = encode_labels_transform(test_df, label_col=LABEL_COL, label_encoder=label_encoder)

    # 3) Split train/val
    print("\n[STEP 3] Splitting TRAIN into train/val...")
    train_texts, val_texts, train_labels, val_labels = split_train_val(train_df, text_col=TEXT_COL)

    # 4) Tokenizer
    print("\n[STEP 4] Loading tokenizer and describing tokenization...")
    #tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if VERBOSE:
        describe_token_lengths(train_df, tokenizer, TEXT_COL, MAX_LEN)

    # 5) Loaders
    print("\n[STEP 5] Building DataLoaders...")
    train_loader = build_loader(train_texts, train_labels, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = build_loader(val_texts, val_labels, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = build_loader(test_df[TEXT_COL].values, test_df["label"].values, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE, shuffle=False)

    if VERBOSE:
        inspect_one_batch(train_loader)

    # 6) Modelo + optimizer
    print("\n[STEP 6] Building model and optimizer...")
    device = get_device()
    print(f"[DEVICE] Using: {device}")
    if device.type == "cuda":
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(MODEL_NAME, num_labels=len(label_encoder.classes_), device=device)
    optimizer = build_optimizer(model, lr=LR)

    # 7) Entrenar
    print("\n[STEP 7] Training...")
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, EPOCHS)
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"[EPOCH {epoch+1}] Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 8) Evaluación en TEST
    print("\n[STEP 8] Final evaluation on TEST...")
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"[TEST] Accuracy: {test_acc:.4f}")

    y_true, y_pred, y_probs = predict_with_probs(model, test_loader, device)

    id2label = {i: c for i, c in enumerate(label_encoder.classes_)}
    label2id = {c: i for i, c in enumerate(label_encoder.classes_)}

    # reconstruimos los textos en el mismo orden que el test_loader
    test_texts = test_df[TEXT_COL].values.tolist()

    rows = []
    for text, yt, yp, prob_vec in zip(test_texts, y_true, y_pred, y_probs):
        true_name = id2label[yt]
        pred_name = id2label[yp]
        confidence = float(max(prob_vec))

        # prob de Dementia / HC 
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

    # imprime resumen rápido
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

    # guarda a CSV para inspección
    out_csv = os.path.join(OUTPUT_DIR, "test_predictions_with_errors.csv")
    pred_df.to_csv(out_csv, index=False)
    print(f"\n[SAVE] Saved per-example predictions to: {out_csv}")

    # report + matriz 
    print("\n[TEST] Classification report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    print("[TEST] Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


    # 9) Guardar modelo
    print("\n[STEP 9] Saving model artifacts...")
    save_artifacts(model, tokenizer, label_encoder, out_dir=OUTPUT_DIR)

    print("\n========== DONE ==========")

if __name__ == "__main__":

    main()
