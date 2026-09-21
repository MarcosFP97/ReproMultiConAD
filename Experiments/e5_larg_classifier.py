"""
Classifier based on multilingual-E5-large embeddings with classical ML models
(global pipeline).

Encodes transcripts using intfloat/multilingual-e5-large and applies GridSearchCV
over Decision Tree, Random Forest, SVM, and Logistic Regression. Supports
monolingual training in English or Spanish, as well as binary classification
(excluding MCI) and multiclass classification.
"""

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

args_slurm = parser.parse_args()

path_to_data_folder = "./jsonl/"
train_en = pd.read_json(path_to_data_folder + "train_en_e5.jsonl", lines=True)
test_en = pd.read_json(path_to_data_folder + "test_en_e5.jsonl", lines=True)

train_spa = pd.read_json(path_to_data_folder + "train_spa_e5.jsonl", lines=True)
test_spa=pd.read_json(path_to_data_folder + "test_spa_e5.jsonl", lines=True)

train_by_lang = {
    "en": train_en,
    "spa": train_spa
}

test_by_lang = {
    "en": test_en,
    "spa": test_spa
}

# Mono-lingual training and testing
train_dfs = [train_by_lang[args_slurm.test_language]]
test_dfs = {
    args_slurm.test_language: test_by_lang[args_slurm.test_language]
}

def _get_confidence(estimator, X):
    """
    Returns a confidence score for each example.

    - If predict_proba is available: max(probabilities)

    - If decision_function is available:

        * binary: |score|

        * multiclass: top1-top2 margin
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


def extract_embeddings(df, text_column, label_column):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('intfloat/multilingual-e5-large').to(device)
    # The "passage: " prefix is required for E5; without it, the model produces degraded embeddings.
    texts = ["passage: " + text for text in df[text_column].tolist()]
    labels = df[label_column].tolist()
    embeddings = model.encode(texts, normalize_embeddings=True,device=device)
    return np.vstack(embeddings), np.array(labels)

def classify_language_dataset_e5(train_dfs, test_dfs, test_language, random_state=42, task=None):
    # Combine the train sets from all languages
    train_combined = pd.concat(train_dfs, ignore_index=True)
    if any(df.equals(train_en) for df in train_dfs):
        train_combined['Diagnosis'] = train_combined['Diagnosis'].replace('AD', 'Dementia')
    if task== "binary":
        train_combined = train_combined[train_combined['Diagnosis'] != 'MCI']
    # Extract embeddings and labels for the combined train set
    X_train, y_train = extract_embeddings(train_combined, 'Text_interviewer_participant', 'Diagnosis')
    
    
    # Select the appropriate test set based on the test_language argument

    test_df = test_dfs[test_language]
    if any(df.equals(train_en) for df in train_dfs):
         test_df['Diagnosis'] = test_df['Diagnosis'].replace('AD', 'Dementia')
    if task == "binary":
        test_df = test_df[test_df['Diagnosis'] != 'MCI']
    
    X_test_text = test_df["Text_interviewer_participant"].astype(str)
    X_test, y_test = extract_embeddings(test_df, 'Text_interviewer_participant', 'Diagnosis')
    
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

        eval_df["text_snip"] = (
            eval_df["_text"]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace("\t", " ", regex=False)
            .str.slice(0, 220)
        )

        n_total = len(eval_df)
        n_ok = int(eval_df["correct"].sum())
        n_bad = n_total - n_ok
        print("--------------------------------------------------")
        print(f"[{name}] Example summary")
        print(f"Total: {n_total} | Correct: {n_ok} | Incorrect: {n_bad}")
        print("Correct predictions by class (y_true):")
        print(eval_df.loc[eval_df["correct"], "y_true"].value_counts())
        print("Incorrect predictions by class (y_true):")
        print(eval_df.loc[~eval_df["correct"], "y_true"].value_counts())

        desired_order = ["HC", "MCI", "Dementia"]

        labels_order = [c for c in desired_order if c in grid_search.classes_]

        cm = confusion_matrix(
            eval_df["y_true"],
            eval_df["y_pred"],
            labels=labels_order
        )

        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels_order],
            columns=[f"pred_{l}" for l in labels_order]
        )

        print("\nConfusion Matrix (HC-MCI-Dementia):")
        print(cm_df)

        confusions = []
        for i, tl in enumerate(labels_order):
            for j, pl in enumerate(labels_order):
                if i != j and cm[i, j] > 0:
                    confusions.append((cm[i, j], tl, pl))
        confusions.sort(reverse=True, key=lambda x: x[0])
        if confusions:
            for c, tl, pl in confusions[:10]:
                print(f"  {c:>4}  {tl} -> {pl}")

        id_candidates = ["ID", "Participant_ID", "participant_id", "Interview_ID", "File", "file"]
        id_cols = [c for c in id_candidates if c in eval_df.columns]
        show_cols = id_cols + ["conf", "y_true", "y_pred", "text_snip"]

        def _print_block(df, title, n=20):
            print(f"\n{title}")
            if len(df) == 0:
                print("  (empty)")
                return
            n = min(n, len(df))
            print(df[show_cols].head(n).to_string(index=False))

        best_correct = eval_df[eval_df["correct"]].copy()
        if best_correct["conf"].notna().any():
            best_correct = best_correct.sort_values("conf", ascending=False)
        _print_block(best_correct, "20 best (more confidence):", n=20)

        worst_wrong = eval_df[~eval_df["correct"]].copy()
        if worst_wrong["conf"].notna().any():
            worst_wrong = worst_wrong.sort_values("conf", ascending=False)
        _print_block(worst_wrong, "20 worst (fails with more confidence):", n=20)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics_dict = {
            "Classifier": name,
            "Best Params": str(grid_search.best_params_),
            "Accuracy": report["accuracy"],
            "Test Language": test_language,
            "Task": task,
            "Representation": "e5-large"
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

        # ---- Macro avg ----
        metrics_dict["Macro_precision"] = report["macro avg"]["precision"]
        metrics_dict["Macro_recall"] = report["macro avg"]["recall"]
        metrics_dict["Macro_f1"] = report["macro avg"]["f1-score"]

        # ---- Weighted avg ----
        metrics_dict["Weighted_precision"] = report["weighted avg"]["precision"]
        metrics_dict["Weighted_recall"] = report["weighted avg"]["recall"]
        metrics_dict["Weighted_f1"] = report["weighted avg"]["f1-score"]

        # -> 1 fila
        results.append(metrics_dict)

        # ----------------------------------------------

        print(f"Classifier: {name}")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Test Set Language: {test_language}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\n")

    results_path = f"./E5_{test_language}_{task}.xlsx"

    final_df = pd.DataFrame(results)
    final_df.to_excel(results_path, index=False)

    print(f"Results saved in: {results_path}")
    # -----------------------------------------------------

    print("test dataset: ", test_language)
    for df in train_dfs:
        df_name = [name for name, value in globals().items() if value is df][0]
        print(f"DataFrame in training set: {df_name}")
    print(task)
    print("e5_large")


classify_language_dataset_e5(train_dfs, test_dfs, args_slurm.test_language, task=args_slurm.task)
