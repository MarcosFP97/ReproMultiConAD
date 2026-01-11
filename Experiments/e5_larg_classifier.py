import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_language', required=True)
parser.add_argument('--task', required=True)
parser.add_argument('--translated', required=True)

args_slurm = parser.parse_args()

path_to_data_folder = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/"
train_en = pd.read_json(path_to_data_folder + "train_english_e5.jsonl", lines=True)
test_en = pd.read_json(path_to_data_folder + "test_english_e5.jsonl", lines=True)

#train_spa = pd.read_json(path_to_data_folder + "train_spanish.jsonl", lines=True)
#test_spa=pd.read_json(path_to_data_folder + "test_spanish.jsonl", lines=True)
train_spa = pd.read_json(path_to_data_folder + "train_spanish_e5.jsonl", lines=True)
test_spa=pd.read_json(path_to_data_folder + "test_spanish_e5.jsonl", lines=True)

#----------------------------------- de momento estos no -----------------------------------------
#train_gr = pd.read_json(path_to_data_folder+"/translated_train_gr.jsonl", lines=True)
#train_cha = pd.read_json(path_to_data_folder + "/translated_train_cha.jsonl", lines=True)
#test_gr= pd.read_json(path_to_data_folder + "/translated_test_gr.jsonl", lines=True)
#test_cha= pd.read_json(path_to_data_folder + "/translated_test_cha.jsonl", lines=True)
#------------------------------------------------------------------------------------------------

# Multi-lingual training and testing
#train_dfs = [train_en, train_gr, train_cha, train_spa]
#test_dfs = {
#    'en': test_en,
#    'gr': test_gr,
#    'cha': test_cha,
#    'spa': test_spa
#}

# Mono-lingual training and testing
train_dfs = [train_en]
test_dfs = {
    'en': test_en
}

def _get_confidence(estimator, X):
    """
    Devuelve un score de confianza por ejemplo.
    - Si hay predict_proba: max(probabilidades)
    - Si hay decision_function:
        * binario: |score|
        * multiclase: margen top1-top2
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return proba.max(axis=1)
    elif hasattr(estimator, "decision_function"):
        dec = estimator.decision_function(X)
        if dec.ndim == 1:
            return np.abs(dec)
        else:
            top2 = np.partition(dec, -2, axis=1)[:, -2:]
            return top2[:, 1] - top2[:, 0]
    return None


# Add a column for translated text for English dataset
if args_slurm.translated== "yes":
    train_en['translated'] = train_en['Text_interviewer_participant']
    test_en['translated'] = test_en['Text_interviewer_participant']

def extract_embeddings(df, text_column, label_column):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('intfloat/multilingual-e5-large').to(device)
    texts = ["passage: " + text for text in df[text_column].tolist()]
    labels = df[label_column].tolist()
    embeddings = model.encode(texts, normalize_embeddings=True,device=device)
    return np.vstack(embeddings), np.array(labels)

def classify_language_dataset_e5(train_dfs, test_dfs, test_language,random_state=42,task=None,translated=None):
    # Combine the train sets from all languages
    train_combined = pd.concat(train_dfs, ignore_index=True)
    if any(df.equals(train_en) for df in train_dfs):
        train_combined['Diagnosis'] = train_combined['Diagnosis'].replace('AD', 'Dementia')
    if task== "binary":
        train_combined = train_combined[train_combined['Diagnosis'] != 'MCI']
    # Extract embeddings and labels for the combined train set
    if translated == "yes":  
        X_train, y_train = extract_embeddings(train_combined, 'translated', 'Diagnosis')
    else:
        X_train, y_train = extract_embeddings(train_combined, 'Text_interviewer_participant', 'Diagnosis')
    
    
    # Select the appropriate test set based on the test_language argument

    test_df = test_dfs[test_language]
    if any(df.equals(train_en) for df in train_dfs):
         test_df['Diagnosis'] = test_df['Diagnosis'].replace('AD', 'Dementia')
    if task == "binary":
        test_df = test_df[test_df['Diagnosis'] != 'MCI']
    
    if translated == "yes":
        X_test_text = test_df["translated"].astype(str)
        X_test, y_test = extract_embeddings(test_df, 'translated', 'Diagnosis')
    else:
        X_test_text = test_df["Text_interviewer_participant"].astype(str)
        X_test, y_test = extract_embeddings(test_df, 'Text_interviewer_participant', 'Diagnosis')
    
    # Para guardar los resultados en un excel
    results = []
    
    # Define classifiers and their hyperparameters for grid search
    classifiers = {
        'Decision Tree': (DecisionTreeClassifier(random_state=random_state), {'max_depth': [10, 20, 30]}),
        'Random Forest': (RandomForestClassifier(random_state=random_state), {'n_estimators': [50, 100, 200]}),
        'SVM': (SVC(random_state=random_state), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
        'Logistic Regression': (LogisticRegression(random_state=random_state), {'C': [0.1, 1, 10]})
    }
    
    # Perform grid search and classification
    for name, (clf, params) in classifiers.items():
        grid_search = GridSearchCV(clf, params, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)
        
        best_model = grid_search.best_estimator_
        conf = _get_confidence(best_model, X_test)

        # DataFrame de evaluación
        eval_df = test_df.copy()
        eval_df["_text"] = X_test_text.values
        eval_df["y_true"] = pd.Series(y_test).astype(str).values
        eval_df["y_pred"] = pd.Series(y_pred).astype(str).values
        eval_df["correct"] = (eval_df["y_true"] == eval_df["y_pred"])

        if conf is None:
            eval_df["conf"] = np.nan
        else:
            eval_df["conf"] = conf

        # snippet corto para que no explote el log
        eval_df["text_snip"] = (
            eval_df["_text"]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace("\t", " ", regex=False)
            .str.slice(0, 220)
        )

        # ---- Resumen de aciertos/fallos
        n_total = len(eval_df)
        n_ok = int(eval_df["correct"].sum())
        n_bad = n_total - n_ok
        print("--------------------------------------------------")
        print(f"[{name}] Resumen ejemplos")
        print(f"Total: {n_total} | Aciertos: {n_ok} | Fallos: {n_bad}")
        print("Aciertos por clase (y_true):")
        print(eval_df.loc[eval_df["correct"], "y_true"].value_counts())
        print("Fallos por clase (y_true):")
        print(eval_df.loc[~eval_df["correct"], "y_true"].value_counts())

        # ---- Matriz de confusión
        labels = list(grid_search.classes_)
        cm = confusion_matrix(eval_df["y_true"], eval_df["y_pred"], labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels],
            columns=[f"pred_{l}" for l in labels]
        )
        print("\nMatriz de confusión:")
        print(cm_df)

        # Confusiones más frecuentes (true != pred)
        confusions = []
        for i, tl in enumerate(labels):
            for j, pl in enumerate(labels):
                if i != j and cm[i, j] > 0:
                    confusions.append((cm[i, j], tl, pl))
        confusions.sort(reverse=True, key=lambda x: x[0])
        if confusions:
            print("\nTop confusiones (count, true -> pred):")
            for c, tl, pl in confusions[:10]:
                print(f"  {c:>4}  {tl} -> {pl}")

        # ---- Columnas útiles para imprimir (si existen IDs en tu dataset)
        id_candidates = ["ID", "Participant_ID", "participant_id", "Interview_ID", "File", "file"]
        id_cols = [c for c in id_candidates if c in eval_df.columns]
        show_cols = id_cols + ["conf", "y_true", "y_pred", "text_snip"]

        def _print_block(df, title, n=20):
            print(f"\n{title}")
            if len(df) == 0:
                print("  (vacío)")
                return
            n = min(n, len(df))
            print(df[show_cols].head(n).to_string(index=False))

        # ---- 20 MEJORES: aciertos con más confianza
        best_correct = eval_df[eval_df["correct"]].copy()
        if best_correct["conf"].notna().any():
            best_correct = best_correct.sort_values("conf", ascending=False)
        _print_block(best_correct, "20 MEJORES (aciertos más seguros):", n=20)

        # ---- 20 PEORES: fallos con más confianza
        worst_wrong = eval_df[~eval_df["correct"]].copy()
        if worst_wrong["conf"].notna().any():
            worst_wrong = worst_wrong.sort_values("conf", ascending=False)
        _print_block(worst_wrong, "20 PEORES (fallos con más confianza):", n=20)

        report = classification_report(y_test, y_pred, output_dict=True)

        metrics_dict = {
            "Classifier": name,
            "Best Params": str(grid_search.best_params_),
            "Accuracy": report["accuracy"],

            "Dementia_precision": report["Dementia"]["precision"],
            "Dementia_recall": report["Dementia"]["recall"],
            "Dementia_f1": report["Dementia"]["f1-score"],
            "Dementia_support": report["Dementia"]["support"],

            "HC_precision": report["HC"]["precision"],
            "HC_recall": report["HC"]["recall"],
            "HC_f1": report["HC"]["f1-score"],
            "HC_support": report["HC"]["support"],

            "Macro_precision": report["macro avg"]["precision"],
            "Macro_recall": report["macro avg"]["recall"],
            "Macro_f1": report["macro avg"]["f1-score"],

            "Weighted_precision": report["weighted avg"]["precision"],
            "Weighted_recall": report["weighted avg"]["recall"],
            "Weighted_f1": report["weighted avg"]["f1-score"],

            "Test Language": test_language,
            "Task": task,
            "Translated": translated,
            "Representation": "e5-large"
        }

        # Convertimos a formato vertical
        metrics_df = pd.DataFrame(
            list(metrics_dict.items()),
            columns=["Metric", "Value"]
        )

        # Añadimos una columna para identificar el clasificador
        metrics_df["Classifier"] = name

        results.append(metrics_df)

        print(f"Classifier: {name}")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Test Set Language: {test_language}")
        print(classification_report(y_test, y_pred))
        print("\n")

    # Guardamos en el excel
    results_path = f"/mnt/beegfs/groups/irgroup/sara_tfg/results/E5_{test_language}_results.xlsx"

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_excel(results_path, index=False)

    print(f"Resultados guardados en: {results_path}")

    print("test dataset: ", test_language)
    for df in train_dfs:
        df_name = [name for name, value in globals().items() if value is df][0]
        print(f"DataFrame in training set: {df_name}")
    print(task)
    print("e5_large")
    print("Translation status: ",translated)


# --- log a TXT (solo archivo) ---
log_dir = "/mnt/beegfs/groups/irgroup/sara_tfg/logs/"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(
    log_dir,
    f"E5_{args_slurm.test_language}.txt"
)

sys.stdout = open(log_path, "w", encoding="utf-8")
sys.stderr = sys.stdout  # opcional: también guarda errores

classify_language_dataset_e5(train_dfs, test_dfs, args_slurm.test_language, task=args_slurm.task,translated=args_slurm.translated)

sys.stdout.close()