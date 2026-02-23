import re
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
INPUT_FILE = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Ivanova.jsonl"
OUTPUT_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets"

TRAIN_OUT = f"{OUTPUT_DIR}/train_ivanova.jsonl"
TEST_OUT  = f"{OUTPUT_DIR}/test_ivanova.jsonl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF = False

# LÍMITES SOLO PARA IVANOVA
MIN_WORDS = 40
MAX_WORDS = 100


# =========================
# TEXT CLEANING
# =========================
def preprocess_text(text: str) -> str:
    text = str(text)

    text = re.sub(r'\b[A-Z]{3}\b', '', text)
    text = re.sub(r'xxx', '', text)
    text = re.sub(r'<[^>]*>', '', text)

    # OJO: esto elimina puntuación/dígitos (más tipo TF-IDF). Para e5 a veces conviene dejar puntuación.
    if TFIDF :
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        text = text.replace('PAR', '')
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\\x[0-9A-Za-z_]+\\x', '', text)
        text = re.sub(r'\b\w+:\s*', '', text)

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

        # Normalizaciones “extra”
        text = re.sub(r'([.,!?;:])\s+\1', r'\1', text)
        text = re.sub(r'(\.\s*){2,}', '.', text)
        if '.' in text:
            text = text.rsplit('.', 1)[0] + '.'

    return text

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

def trim_by_word_limits(text: str, min_words: int, max_words: int):
    words = str(text).split()
    if len(words) < min_words:
        return None
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text


# =========================
# PIPELINE SOLO IVANOVA
# =========================
df = pd.read_json(INPUT_FILE, lines=True)

# Normaliza etiquetas de diagnóstico (si aplica)
df = df.copy()
df["Diagnosis"] = df["Diagnosis"].replace({"DTA": "Dementia", "AD": "Dementia"})
df = clean_gender(df)

# Filtra diagnosis vacías/unknown
df = df[df["Diagnosis"].notnull()].copy()
df = df[df["Diagnosis"].astype(str).str.strip().ne("")].copy()
df = df[df["Diagnosis"].ne("Unknown")].copy()

# Limpieza de texto
df["Text_interviewer_participant"] = df["Text_interviewer_participant"].apply(preprocess_text)

# Recorte por longitud
df["Text_interviewer_participant"] = df["Text_interviewer_participant"].apply(
    lambda t: trim_by_word_limits(t, MIN_WORDS, MAX_WORDS)
)

# Quita los que quedaron en None (muy cortos)
df = df[df["Text_interviewer_participant"].notnull()].copy()

# Columna de longitud (por si la quieres inspeccionar)
df["length"] = df["Text_interviewer_participant"].apply(lambda x: len(str(x).split()))

print("Distribución de diagnósticos (post-procesado):")
print(df["Diagnosis"].value_counts())

# Split estratificado
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["Diagnosis"],
    random_state=RANDOM_STATE
)

# Guarda JSONL
train_df.to_json(TRAIN_OUT, orient="records", lines=True, force_ascii=False)
test_df.to_json(TEST_OUT, orient="records", lines=True, force_ascii=False)

print(f"\nGuardado:\n- {TRAIN_OUT}\n- {TEST_OUT}")
