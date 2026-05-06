"""
Clasificador TF-IDF sobre la pipeline global de datos combinados por idioma.

Lee train_{language}_e5.jsonl y test_{language}_e5.jsonl, aplica TfidfVectorizer
y ejecuta GridSearchCV sobre Decision Tree, Random Forest, Naive Bayes, SVM y
Logistic Regression. Soporta clasificación binaria (sin MCI) o multiclase.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import argparse

LABEL = 'Text_interviewer_participant'

parser = argparse.ArgumentParser()
parser.add_argument('--test_language', required=True)
parser.add_argument('--task', required=True)
parser.add_argument('--translated', required=True, help="'yes' para usar texto traducido al inglés")

args_slurm = parser.parse_args()

path_to_data_folder = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/"
train_en = pd.read_json(path_to_data_folder + "train_en.jsonl", lines=True)
test_en = pd.read_json(path_to_data_folder + "test_en.jsonl", lines=True)

train_spa = pd.read_json(path_to_data_folder + "train_spa.jsonl", lines=True)
test_spa=pd.read_json(path_to_data_folder + "test_spa.jsonl", lines=True)

train_by_lang = {
    "en": train_en,
    "spa": train_spa
}

test_by_lang = {
    "en": test_en,
    "spa": test_spa
}

train_dfs = [train_by_lang[args_slurm.test_language]]
test_dfs = {
    args_slurm.test_language: test_by_lang[args_slurm.test_language]
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
        conf = proba.max(axis=1)
        return conf
    elif hasattr(estimator, "decision_function"):
        dec = estimator.decision_function(X)
        if dec.ndim == 1:          # binario
            conf = np.abs(dec)
        else:                      # multiclase
            # margen entre las 2 clases con mayor score
            top2 = np.partition(dec, -2, axis=1)[:, -2:]
            conf = top2[:, 1] - top2[:, 0]
        return conf
    else:
        return None


if args_slurm.translated== "yes":
    train_en['translated'] = train_en[LABEL]
    test_en['translated'] = test_en[LABEL]

def classify_language_dataset_TFIDF(train_dfs, test_dfs, test_language, random_state=42,task=None,translated=None):

    train_combined = pd.concat(train_dfs, ignore_index=True)
    if any(df.equals(train_en) for df in train_dfs):
        train_combined['Diagnosis'] = train_combined['Diagnosis'].replace('AD', 'Dementia')

    if task == "binary":
        train_combined = train_combined[train_combined['Diagnosis'] != 'MCI']

    if translated == "yes":
         X_train = train_combined['translated']
    else:
        X_train = train_combined[LABEL]
    y_train = train_combined['Diagnosis']

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)

    test_df = test_dfs[test_language]
    if any(df.equals(train_en) for df in train_dfs):
         test_df['Diagnosis'] = test_df['Diagnosis'].replace('AD', 'Dementia')
    if task == "binary":
        test_df = test_df[test_df['Diagnosis'] != 'MCI']

    if translated == "yes":
        X_test = test_df['translated']
    else:
        X_test = test_df[LABEL]

    y_test = test_df['Diagnosis']

    X_test_tfidf = tfidf.transform(X_test)

    results = []

    classifiers = {
        'Decision Tree': (DecisionTreeClassifier(random_state=random_state), {'max_depth': [10, 20, 30]}),
        'Random Forest': (RandomForestClassifier(random_state=random_state), {'n_estimators': [50, 100, 200]}),
        'Naive Bayes': (MultinomialNB(), {'alpha': [0.5, 1.0, 1.5]}),
        'SVM': (SVC(random_state=random_state), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
        'Logistic Regression': (LogisticRegression(random_state=random_state), {'C': [0.1, 1, 10]})
    }

    for name, (clf, params) in classifiers.items():
        grid_search = GridSearchCV(clf, params, cv=5, scoring='accuracy')
        grid_search.fit(X_train_tfidf, y_train)

        y_pred = grid_search.predict(X_test_tfidf)
        
        best_model = grid_search.best_estimator_
        conf = _get_confidence(best_model, X_test_tfidf)

        eval_df = test_df.copy()
        eval_df["_text"] = X_test.astype(str).values
        eval_df["y_true"] = y_test.astype(str).values
        eval_df["y_pred"] = pd.Series(y_pred).astype(str).values
        eval_df["correct"] = (eval_df["y_true"] == eval_df["y_pred"])
        
        # Algunos JSONL de la pipeline global guardan el origen con distinta capitalización.
        if "Dataset" not in eval_df.columns:
            for alt in ["dataset", "DATASET", "Corpus", "corpus"]:
                if alt in eval_df.columns:
                    eval_df["Dataset"] = eval_df[alt]
                    break
            else:
                eval_df["Dataset"] = "Unknown"

        print("--------------------------------------------------")
        print(f"[{name}] Resumen ejemplos")
        wrong_df = eval_df[~eval_df["correct"]].copy()

        wrong_by_dataset = wrong_df["Dataset"].value_counts(dropna=False)
        total_by_dataset = eval_df["Dataset"].value_counts(dropna=False)

        fail_rate = (wrong_by_dataset / total_by_dataset).fillna(0).sort_values(ascending=False)

        print("\nFallos por Dataset (conteo):")
        print(wrong_by_dataset.to_string())

        print("\nFallos por Dataset (tasa = fallos/total en test):")
        fail_table = pd.DataFrame({
            "wrong": wrong_by_dataset,
            "total": total_by_dataset,
            "fail_rate": fail_rate
        }).fillna(0).sort_values("wrong", ascending=False)

        print(fail_table.to_string())


        if conf is None:
            eval_df["conf"] = np.nan
        else:
            eval_df["conf"] = conf

        eval_df["text_snip"] = (
            eval_df["_text"]
            .str.replace("\n", " ", regex=False)
            .str.replace("\t", " ", regex=False)
            .str.slice(0, 220)
        )

        n_total = len(eval_df)
        n_ok = int(eval_df["correct"].sum())
        n_bad = n_total - n_ok
        print(f"Total: {n_total} | Aciertos: {n_ok} | Fallos: {n_bad}")
        print("Aciertos por clase (y_true):")
        print(eval_df.loc[eval_df["correct"], "y_true"].value_counts())
        print("Fallos por clase (y_true):")
        print(eval_df.loc[~eval_df["correct"], "y_true"].value_counts())

        labels = list(grid_search.classes_)
        cm = confusion_matrix(eval_df["y_true"], eval_df["y_pred"], labels=labels)
        cm_df = pd.DataFrame(cm,
                            index=[f"true_{l}" for l in labels],
                            columns=[f"pred_{l}" for l in labels])
        print("\nMatriz de confusión:")
        print(cm_df)

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

        best_correct = eval_df[eval_df["correct"]].copy()
        if best_correct["conf"].notna().any():
            best_correct = best_correct.sort_values("conf", ascending=False)
        _print_block(best_correct, "20 MEJORES (aciertos más seguros):", n=20)

        worst_wrong = eval_df[~eval_df["correct"]].copy()
        if worst_wrong["conf"].notna().any():
            worst_wrong = worst_wrong.sort_values("conf", ascending=False)
        _print_block(worst_wrong, "20 PEORES (fallos con más confianza):", n=20)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics_dict = {
            "Classifier": name,
            "Best Params": str(grid_search.best_params_),
            "Accuracy": report["accuracy"],
            "Test Language": test_language,
            "Task": task,
            "Translated": translated,
            "Representation": "TF-IDF"
        }

        class_labels = list(grid_search.classes_)

        for label in class_labels:
            if label in report and isinstance(report[label], dict):
                metrics_dict[f"{label}_precision"] = report[label]["precision"]
                metrics_dict[f"{label}_recall"] = report[label]["recall"]
                metrics_dict[f"{label}_f1"] = report[label]["f1-score"]
                metrics_dict[f"{label}_support"] = report[label]["support"]
            else:
                metrics_dict[f"{label}_precision"] = np.nan
                metrics_dict[f"{label}_recall"] = np.nan
                metrics_dict[f"{label}_f1"] = np.nan
                metrics_dict[f"{label}_support"] = 0

        metrics_dict["Macro_precision"] = report["macro avg"]["precision"]
        metrics_dict["Macro_recall"] = report["macro avg"]["recall"]
        metrics_dict["Macro_f1"] = report["macro avg"]["f1-score"]

        metrics_dict["Weighted_precision"] = report["weighted avg"]["precision"]
        metrics_dict["Weighted_recall"] = report["weighted avg"]["recall"]
        metrics_dict["Weighted_f1"] = report["weighted avg"]["f1-score"]

        results.append(metrics_dict)

        print(f"Classifier: {name}")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Test Set Language: {test_language}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\n")

    results_path = f"/mnt/beegfs/groups/irgroup/sara_tfg/results/TFIDF_{test_language}_{task}.xlsx"
    final_df = pd.DataFrame(results)
    final_df.to_excel(results_path, index=False)
    print(f"Resultados guardados en: {results_path}")

    print("test dataset: ", test_language)
    for df in train_dfs:
        df_name = [name for name, value in globals().items() if value is df][0]
        print(f"DataFrame in training set: {df_name}")
    print(task)
    print("TF-IDF")
    print("Translation status: ",translated)

log_dir = "/mnt/beegfs/groups/irgroup/sara_tfg/logs/"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(
    log_dir,
    f"TFIDF_{args_slurm.test_language}_{args_slurm.task}.log"
)

sys.stdout = open(log_path, "w", encoding="utf-8")
sys.stderr = sys.stdout

print(f"Logging en: {log_path}\n")

classify_language_dataset_TFIDF(train_dfs, test_dfs, args_slurm.test_language, task=args_slurm.task,translated=args_slurm.translated)

sys.stdout.close()
