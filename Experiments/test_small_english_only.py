import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# ------------------------------------------------------------------
# PATH
# ------------------------------------------------------------------
path_to_data_folder = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/"


# ------------------------------------------------------------------
# LOAD ONLY ENGLISH DATA
# ------------------------------------------------------------------
train_en = pd.read_json(
    path_to_data_folder + "train_english.jsonl",
    lines=True
)

test_en = pd.read_json(
    path_to_data_folder + "test_english.jsonl",
    lines=True
)


# ------------------------------------------------------------------
# VERY SMALL SUBSET (SAFE FOR CPU)
# ------------------------------------------------------------------
train_en = train_en.head(20)
test_en = test_en.head(10)


# ------------------------------------------------------------------
# BINARY TASK (OPTIONAL, MATCHES YOUR PIPELINE)
# ------------------------------------------------------------------
train_en['Diagnosis'] = train_en['Diagnosis'].replace('AD', 'Dementia')
test_en['Diagnosis'] = test_en['Diagnosis'].replace('AD', 'Dementia')

train_en = train_en[train_en['Diagnosis'] != 'MCI']
test_en = test_en[test_en['Diagnosis'] != 'MCI']


# ------------------------------------------------------------------
# EMBEDDINGS (CPU SAFE)
# ------------------------------------------------------------------
def extract_embeddings(df, text_column, label_column):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-MiniLM-L6-v2"
    ).to(device)

    texts = ["passage: " + t for t in df[text_column].tolist()]
    labels = df[label_column].tolist()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        device=device
    )

    return np.vstack(embeddings), np.array(labels)


print("➡️  Extrayendo embeddings train")
X_train, y_train = extract_embeddings(
    train_en,
    'Text_interviewer_participant',
    'Diagnosis'
)

print("➡️  Extrayendo embeddings test")
X_test, y_test = extract_embeddings(
    test_en,
    'Text_interviewer_participant',
    'Diagnosis'
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ------------------------------------------------------------------
# CLASSIFIER (MINIMAL BUT REAL)
# ------------------------------------------------------------------
clf = LogisticRegression(max_iter=1000, random_state=42)

param_grid = {'C': [1]}

grid = GridSearchCV(
    clf,
    param_grid,
    cv=2,
    scoring='accuracy'
)

print("➡️  Entrenando clasificador")
grid.fit(X_train, y_train)

print("➡️  Mejor parámetro:", grid.best_params_)


# ------------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------------
y_pred = grid.predict(X_test)

print("\n📊 CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\n✅ PRUEBA PEQUEÑA EN INGLÉS COMPLETADA")
