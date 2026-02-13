import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
TFIDF = False  # True -> quita puntuación/dígitos (TF-IDF). False -> deja más “natural” (mejor para e5)
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TEXT_LEN = 60  # mínimo de caracteres (después de limpiar)

# === Elige UN SOLO dataset (un único .jsonl) ===

DATASET_NAME = "Pitt"  # para nombrar los ficheros de salida
INPUT_JSONL = f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/{DATASET_NAME}.jsonl"

OUTPUT_DIRECTORY = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


# =========================
# HELPERS
# =========================
def remove_zh_language_rows(df):
    # Si no existe la columna, no hace nada
    if "Languages" not in df.columns:
        return df
    return df[df["Languages"] != "zh"]


def clean_diagnosis(df):
    if "Diagnosis" not in df.columns:
        raise ValueError("No existe la columna 'Diagnosis' en el JSONL.")

    diagnoses_to_remove = ['Vascular', 'Memory', 'Aphasia', "Pick's", 'Other']
    df = df[~df['Diagnosis'].isin(diagnoses_to_remove)]

    df = df[df['Diagnosis'].notna() & (df['Diagnosis'] != '')]

    df['Diagnosis'] = df['Diagnosis'].replace({
        'Control': 'HC',
        'Conrol': 'HC',
        'NC': 'HC',
        'H': 'HC',
        'AD': 'Dementia',
        'DM': 'Dementia',
        'PossibleAD': 'Dementia',
        'ProbableAD': 'Dementia',
        'Probable': 'Dementia',
        'potential dementia': 'Dementia',
        'D': 'Dementia',
        "Alzheimer's": 'Dementia'
    })

    return df


def clean_gender(df):
    if "Gender" not in df.columns:
        return df

    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()

    df["Gender"] = df["Gender"].replace({
        "m": "M",
        "male": "M",
        "f": "F",
        "female": "F",
        "w": "F",
        "nan": "U",
        "none": "U",
        "": "U"
    })

    df.loc[~df["Gender"].isin(["M", "F"]), "Gender"] = "U"
    return df


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Limpiezas comunes
    text = re.sub(r'\b[A-Z]{3}\b', '', text)
    text = re.sub(r'xxx', '', text)
    text = re.sub(r'<[^>]*>', '', text)

    if TFIDF:
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.replace('PAR', '')
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\\x[0-9A-Za-z_]+\\x', '', text)
        text = re.sub(r'\b\w+:\s*', '', text)
        text = text.replace('\n', ' ')
        text = text.replace('→', '')
        text = text.replace('(', '').replace(')', '')
        text = re.sub(r'[\\+^"/„]', '', text)
        text = re.sub(r"[_']", '', text)
        text = text.replace('\t', ' ')
        text = re.sub(r'\[.*?\]', '', text)
        text = text.replace('&=laughs', '')
        text = text.replace('&=nods', '')
        text = text.replace('&=coughs', '')
        text = text.replace('&=snaps:tongue', '')
        text = text.replace('<', '').replace('>', '')
        text = text.replace('*', '').replace('&', '')
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([.,!?;:])\s+\1', r'\1', text)
        text = re.sub(r'(\.\s*){2,}', '.', text)

        if '.' in text:
            text = text.rsplit('.', 1)[0] + '.'

    return text


def remove_short_transcripts(df, text_col, min_length=60):
    return df[df[text_col].astype(str).str.len() > min_length]


def pick_text_columns(df):
    """
    Usa estas columnas si existen. Si alguna no existe, la crea vacía.
    """
    if "Text_interviewer_participant" not in df.columns:
        df["Text_interviewer_participant"] = ""
    return df


# =========================
# LOAD
# =========================
df = pd.read_json(INPUT_JSONL, lines=True)

# Asegurar columnas texto
df = pick_text_columns(df)

# =========================
# CLEANING
# =========================
df = remove_zh_language_rows(df)
df = clean_diagnosis(df)
df = clean_gender(df)

print("Distribución Diagnosis (antes de filtrar por longitud):")
print(df["Diagnosis"].value_counts(dropna=False))

# Preprocesado de texto
df["Text_interviewer_participant"] = df["Text_interviewer_participant"].apply(preprocess_text)

# Filtrar por longitud (elige la que uses para modelar; aquí usamos interviewer+participant)
df = remove_short_transcripts(df, "Text_interviewer_participant", min_length=MIN_TEXT_LEN)

print("\nDistribución Diagnosis (después de filtrar por longitud):")
print(df["Diagnosis"].value_counts(dropna=False))

# =========================
# TRAIN/TEST SPLIT (estratificado)
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["Diagnosis"],
    random_state=RANDOM_STATE
)

print(f"\nSplit realizado: train={len(train_df)} | test={len(test_df)}")

# =========================
# SAVE
# =========================
train_out = os.path.join(OUTPUT_DIRECTORY, f"train_{DATASET_NAME.lower()}.jsonl")
#test_out = os.path.join(OUTPUT_DIRECTORY, f"test_{DATASET_NAME.lower()}_english.jsonl")

train_df.to_json(train_out, orient="records", lines=True, force_ascii=False)
#test_df.to_json(test_out, orient="records", lines=True, force_ascii=False)

print("\nGuardado:")
print(" -", train_out)
#print(" -", test_out)
