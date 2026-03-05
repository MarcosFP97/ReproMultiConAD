import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# CLASE TRAINER. Intentaremos hacer un BERT balanceado
from sklearn.utils.class_weight import compute_class_weight
from transformers import Trainer, TrainingArguments
from torch import nn
import torch.nn.functional as F

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

#TRAIN_PATH = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/train_english_e5.jsonl"
#TEST_PATH  = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/test_english_e5.jsonl"
#OUTPUT_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_Models/bert_english_patient_classifier_len256"

#TRAIN_PATH = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/ivanova_augmented.jsonl"

# Configuración de los argumentos de entrada
parser = argparse.ArgumentParser(description="Entrenamiento de BERT con slices de datos")
parser.add_argument("--dataset", type=str, required=True, help="Nombre del dataset (ej: ivanova, pitt)")
parser.add_argument("--percentage", type=int, required=True, help="Porcentaje del slice (ej: 20, 40, 60, 80)")
args = parser.parse_args()

# Asignamos los argumentos a variables para usarlas en la config
dataset = args.dataset
percentage = args.percentage

TRAIN_PATH = f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/slices/train_{dataset}_{percentage}_synthetic.jsonl"
TEST_PATH  = f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets/test_{dataset}.jsonl"
OUTPUT_DIR = f"/mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_Models/bert_{dataset}_patient_classifier_len256"

TEXT_COL  = "Text_interviewer_participant"
LABEL_COL = "Diagnosis"

if dataset == "ivanova":
    MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
else:
    MODEL_NAME = "bert-base-uncased"
    
MAX_LEN    = 256
BATCH_SIZE = 16
LR         = 5e-5
EPOCHS     = 3

DROP_LABEL_VALUE = "MCI"   # quitamos MCI para binario
VERBOSE = True             # False si queremos menos prints


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
        # Cambia "label" por "labels" (Trainer)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long), # <--- AQUÍ
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

# Clase que deriva de trainer
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs): # Quitamos *args
        super().__init__(**kwargs)                    # Quitamos *args
        # Guardamos los pesos y los mandamos al dispositivo (CPU/GPU) correcto
        self.class_weights = class_weights
        if self.class_weights is not None and self.model is not None:
            self.class_weights = self.class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extraemos las etiquetas
        labels = inputs.pop("labels")
        # Pasamos los inputs al modelo
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Calculamos la pérdida (Loss) usando nuestros pesos personalizados
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
    #model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    return model

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# ============================================================
# 4) EXTRAS: inspección de tokenización / longitudes
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
# 5) GUARDADO A EXCEL
# ============================================================
def compute_truncation_pct(df: pd.DataFrame, tokenizer, text_col: str, max_len: int) -> float:
    lengths = df[text_col].apply(lambda x: len(tokenizer.tokenize(str(x))))
    return float((lengths > max_len).mean() * 100.0)

def build_experiment_row(*, y_true: list, y_pred: list, class_names: list, exp_meta: dict,) -> pd.DataFrame:
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

    # global
    row["Accuracy"] = report.get("accuracy", np.nan)

    # macro
    row["Macro_precision"] = report["macro avg"]["precision"]
    row["Macro_recall"]    = report["macro avg"]["recall"]
    row["Macro_f1"]        = report["macro avg"]["f1-score"]

    # weighted
    row["Weighted_precision"] = report["weighted avg"]["precision"]
    row["Weighted_recall"]    = report["weighted avg"]["recall"]
    row["Weighted_f1"]        = report["weighted avg"]["f1-score"]

    # por clase
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

    # confusion matrix
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

# ============================================================
# MAIN
# ============================================================
def main():
    print("========== BERT TEXT CLASSIFICATION PIPELINE ==========")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Cargar datos
    print("\n[STEP 1] Loading train/test data (jsonl) and selecting needed columns...")
    train_df = load_and_prepare_df(TRAIN_PATH, TEXT_COL, LABEL_COL, DROP_LABEL_VALUE)
    test_df  = load_and_prepare_df(TEST_PATH,  TEXT_COL, LABEL_COL, DROP_LABEL_VALUE)
    print(f"[DATA] Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    
    # Aplica la limpieza a tus dos DataFrames ANTES de pasarlos al tokenizador de BERT
    print("[PREPROCESO] Limpiando códigos de tiempo del dataset Real...")
    train_df['Text_interviewer_participant'] = train_df['Text_interviewer_participant'].apply(limpiar_texto_chat)
    test_df['Text_interviewer_participant'] = test_df['Text_interviewer_participant'].apply(limpiar_texto_chat)

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

    # =========================================================
    # NUEVO FLUJO CON HUGGING FACE TRAINER
    # =========================================================

    # 5) Calcular pesos de clase para balancear
    print("\n[STEP 5] Calculating class weights for imbalanced data...")
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights_array, dtype=torch.float)
    if VERBOSE:
        print(f"[WEIGHTS] Class weights: {class_weights_tensor}")

    # 6) Datasets de Hugging Face (ya no usamos DataLoaders manuales)
    print("\n[STEP 6] Building Datasets for Trainer...")
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset   = ClassificationDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_dataset  = ClassificationDataset(test_df[TEXT_COL].values, test_df["label"].values, tokenizer, MAX_LEN)

    # 7) Modelo y Configuración del Trainer
    print("\n[STEP 7] Building model and CustomTrainer...")
    device = get_device()
    model = build_model(MODEL_NAME, num_labels=len(label_encoder.classes_), device=device)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        evaluation_strategy="epoch", # Evalúa al final de cada epoch
        save_strategy="epoch",
        load_best_model_at_end=True, # Se queda con el mejor modelo según validación
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics, # <--- AÑADIDO
        class_weights=class_weights_tensor
    )

    # 8) Entrenar
    print("\n[STEP 8] Training with CustomTrainer...")
    trainer.train()

    # 9) Predicciones en TEST para tu Excel
    print("\n[STEP 9] Predicting on TEST set...")
    predictions_output = trainer.predict(test_dataset)
    
    # Extraer y_true, y_pred, y_probs para que no se rompa tu código de Excel
    y_true = predictions_output.label_ids
    logits = predictions_output.predictions
    # Convertimos logits a probabilidades usando Softmax
    y_probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    y_pred = np.argmax(y_probs, axis=1)

    # test_acc para mostrarlo en pantalla
    test_acc = (y_true == y_pred).mean()
    print(f"[TEST] Accuracy: {test_acc:.4f}")
    
    # === A partir de aquí tu código sigue igual ===
    id2label = {i: c for i, c in enumerate(label_encoder.classes_)}

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
    
    # =========================
    # EXCEL: resumen del experimento
    # =========================
    results_dir = "/mnt/beegfs/groups/irgroup/sara_tfg/results/"
    os.makedirs(results_dir, exist_ok=True)

    # cosas útiles para guardar
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

    ################## Excel ######################
    task_name = "binary" if DROP_LABEL_VALUE is not None else "multiclass"
    out_xlsx = os.path.join(results_dir, f"BERT_Synthetic_{dataset}_{percentage}_{task_name}.xlsx")

    save_results_excel(
        out_xlsx,
        summary_row_df=summary_row,
        y_true=y_true,
        y_pred=y_pred,
        class_names=list(label_encoder.classes_)
    )

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

    # report + matriz 
    print("\n[TEST] Classification report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    print("[TEST] Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


    # 9) Guardar modelo (ACTUALIZADO PARA TRAINER)
    print("\n[STEP 10] Saving model artifacts...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR) # Esto guarda el mejor modelo, configs y args
    tokenizer.save_pretrained(OUTPUT_DIR)
    torch.save(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pth"))

    if VERBOSE:
        print(f"\n[SAVE] Model + tokenizer + label_encoder saved to:\n  {OUTPUT_DIR}")

    print("\n========== DONE ==========")

if __name__ == "__main__":

    main()
