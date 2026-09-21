"""
TF-IDF classifier for individual datasets (individual pipeline).

Binary classification: HC vs. Impaired (MCI + Dementia grouped), consistent
with the cross-dataset experiment.

Multiclass classification: HC / MCI / Dementia (only for datasets containing
all three classes: Pitt and Ivanova).
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import argparse

LABEL = "Text_interviewer_participant"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="e.g.: Pitt, Lu, Baycrest, Delaware, taukadial, ivanova")
parser.add_argument("--task", default="binary", choices=["binary", "multiclass"])
parser.add_argument(
    "--data-dir",
    default="./individual_sets/TFIDF"
)
parser.add_argument(
    "--results-dir",
    default="./results"
)
parser.add_argument(
    "--log-dir",
    default="./logs",
)
args = parser.parse_args()

DATASET = args.dataset.strip()
DATASET_LOWER = DATASET.lower()
TASK = args.task
balance_tag = "balanced" 
class_weight = "balanced" 

data_dir = args.data_dir
train_path = os.path.join(data_dir, f"train_{DATASET_LOWER}.jsonl")
test_path = os.path.join(data_dir, f"test_{DATASET_LOWER}.jsonl")

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError(
        f"Files not found:\n  {train_path}\n  {test_path}\n"
    )

train_df = pd.read_json(train_path, lines=True)
test_df  = pd.read_json(test_path, lines=True)

train_df["Diagnosis"] = train_df["Diagnosis"].replace("AD", "Dementia")
test_df["Diagnosis"]  = test_df["Diagnosis"].replace("AD", "Dementia")

if TASK == "binary":
    train_df["Diagnosis"] = train_df["Diagnosis"].replace({"MCI": "Enfermo", "Dementia": "Enfermo"})
    test_df["Diagnosis"]  = test_df["Diagnosis"].replace({"MCI": "Enfermo", "Dementia": "Enfermo"})
    keep = {"HC", "Enfermo"}
    positive_label = "Enfermo"
else:
    keep = {"HC", "MCI", "Dementia"}
    positive_label = "Dementia"  

train_df = train_df[train_df["Diagnosis"].isin(keep)].copy()
test_df  = test_df[test_df["Diagnosis"].isin(keep)].copy()

def _get_confidence(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return proba.max(axis=1)
    elif hasattr(estimator, "decision_function"):
        dec = estimator.decision_function(X)
        return np.abs(dec) if dec.ndim == 1 else (np.partition(dec, -2, axis=1)[:, -1] - np.partition(dec, -2, axis=1)[:, -2])
    return None


def run_tfidf_binary(train_df, test_df, random_state=42):
    X_train_text = train_df[LABEL].astype(str)
    y_train = train_df["Diagnosis"].astype(str)

    X_test_text = test_df[LABEL].astype(str)
    y_test = test_df["Diagnosis"].astype(str)

    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    classifiers = {
        "Decision Tree": (DecisionTreeClassifier(random_state=random_state, class_weight=class_weight), {"max_depth": [10, 20, 30]}),
        "Random Forest": (RandomForestClassifier(random_state=random_state, class_weight=class_weight), {"n_estimators": [50, 100, 200]}),
        "Naive Bayes": (MultinomialNB(), {"alpha": [0.5, 1.0, 1.5]}),
        "SVM": (SVC(random_state=random_state, class_weight=class_weight), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
        "Logistic Regression": (LogisticRegression(random_state=random_state, max_iter=2000, class_weight=class_weight), {"C": [0.1, 1, 10]}),
    }

    results = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, (clf, params) in classifiers.items():
        grid_search = GridSearchCV(clf, params, cv=cv, scoring="f1_macro")
        grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)

        best_model = grid_search.best_estimator_
        conf = _get_confidence(best_model, X_test)

        eval_df = test_df.copy()
        eval_df["_text"] = X_test_text.values
        eval_df["y_true"] = y_test.values
        eval_df["y_pred"] = pd.Series(y_pred).astype(str).values
        eval_df["correct"] = (eval_df["y_true"] == eval_df["y_pred"])
        eval_df["conf"] = np.nan if conf is None else conf

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
        print(f"[{name}] Dataset={DATASET} | Binary: HC vs {positive_label}")
        print(f"Total: {n_total} | Aciertos: {n_ok} | Fallos: {n_bad}")
        print("Hits per class(y_true):")
        print(eval_df.loc[eval_df["correct"], "y_true"].value_counts())
        print("Fails per class (y_true):")
        print(eval_df.loc[~eval_df["correct"], "y_true"].value_counts())

        labels_order = ["HC", positive_label]
        cm = confusion_matrix(eval_df["y_true"], eval_df["y_pred"], labels=labels_order)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels_order],
            columns=[f"pred_{l}" for l in labels_order],
        )
        print("\n Confusion matrix (HC first):")
        print(cm_df)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics_dict = {
            "Dataset": DATASET,
            "Binary_Positive": positive_label,
            "Classifier": name,
            "Best Params": str(grid_search.best_params_),
            "Accuracy": report["accuracy"],
            "HC_precision": report["HC"]["precision"] if "HC" in report else np.nan,
            "HC_recall": report["HC"]["recall"] if "HC" in report else np.nan,
            "HC_f1": report["HC"]["f1-score"] if "HC" in report else np.nan,
            "HC_support": report["HC"]["support"] if "HC" in report else 0,
            f"{positive_label}_precision": report[positive_label]["precision"] if positive_label in report else np.nan,
            f"{positive_label}_recall": report[positive_label]["recall"] if positive_label in report else np.nan,
            f"{positive_label}_f1": report[positive_label]["f1-score"] if positive_label in report else np.nan,
            f"{positive_label}_support": report[positive_label]["support"] if positive_label in report else 0,
            "Macro_precision": report["macro avg"]["precision"],
            "Macro_recall": report["macro avg"]["recall"],
            "Macro_f1": report["macro avg"]["f1-score"],
            "Weighted_precision": report["weighted avg"]["precision"],
            "Weighted_recall": report["weighted avg"]["recall"],
            "Weighted_f1": report["weighted avg"]["f1-score"],
            "Representation": "TF-IDF",
        }

        results.append(metrics_dict)

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\n")

    return pd.DataFrame(results)

def run_tfidf_multiclass(train_df, test_df, random_state=42):
    X_train_text = train_df[LABEL].astype(str)
    y_train = train_df["Diagnosis"].astype(str)
    X_test_text = test_df[LABEL].astype(str)
    y_test = test_df["Diagnosis"].astype(str)

    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    classifiers = {
        "Decision Tree": (DecisionTreeClassifier(random_state=random_state, class_weight=class_weight), {"max_depth": [10, 20, 30]}),
        "Random Forest": (RandomForestClassifier(random_state=random_state, class_weight=class_weight), {"n_estimators": [50, 100, 200]}),
        "Naive Bayes": (MultinomialNB(), {"alpha": [0.5, 1.0, 1.5]}),
        "SVM": (SVC(random_state=random_state, class_weight=class_weight), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
        "Logistic Regression": (LogisticRegression(random_state=random_state, max_iter=2000, class_weight=class_weight), {"C": [0.1, 1, 10]}),
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, (clf, params) in classifiers.items():
        grid_search = GridSearchCV(clf, params, cv=cv, scoring="f1_macro")
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)

        eval_df = test_df.copy()
        eval_df["y_true"] = y_test.values
        eval_df["y_pred"] = pd.Series(y_pred).astype(str).values
        eval_df["correct"] = (eval_df["y_true"] == eval_df["y_pred"])

        n_total = len(eval_df)
        n_ok = int(eval_df["correct"].sum())
        n_bad = n_total - n_ok

        print("--------------------------------------------------")
        print(f"[{name}] Dataset={DATASET} | Multiclass: HC / MCI / Dementia")
        print(f"Total: {n_total} | Aciertos: {n_ok} | Fallos: {n_bad}")

        labels_present = sorted(y_test.unique())
        cm = confusion_matrix(eval_df["y_true"], eval_df["y_pred"], labels=labels_present)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels_present],
            columns=[f"pred_{l}" for l in labels_present],
        )
        print("\nConfusion matrix:")
        print(cm_df)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics_dict = {
            "Dataset": DATASET,
            "Task": "multiclass",
            "Classifier": name,
            "Best Params": str(grid_search.best_params_),
            "Accuracy": report["accuracy"],
            "Macro_precision": report["macro avg"]["precision"],
            "Macro_recall": report["macro avg"]["recall"],
            "Macro_f1": report["macro avg"]["f1-score"],
            "Weighted_precision": report["weighted avg"]["precision"],
            "Weighted_recall": report["weighted avg"]["recall"],
            "Weighted_f1": report["weighted avg"]["f1-score"],
            "Representation": "TF-IDF",
        }
        for cls in ["HC", "MCI", "Dementia"]:
            if cls in report:
                metrics_dict[f"{cls}_precision"] = report[cls]["precision"]
                metrics_dict[f"{cls}_recall"]    = report[cls]["recall"]
                metrics_dict[f"{cls}_f1"]        = report[cls]["f1-score"]
                metrics_dict[f"{cls}_support"]   = report[cls]["support"]
            else:
                metrics_dict[f"{cls}_precision"] = np.nan
                metrics_dict[f"{cls}_recall"]    = np.nan
                metrics_dict[f"{cls}_f1"]        = np.nan
                metrics_dict[f"{cls}_support"]   = 0

        results.append(metrics_dict)

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\n")

    return pd.DataFrame(results)


log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, f"TFIDF_{DATASET}_{balance_tag}_{TASK}.log")

sys.stdout = open(log_path, "w", encoding="utf-8")
sys.stderr = sys.stdout

print(f"Logging en: {log_path}")
task_desc = f"Binary: HC vs {positive_label}" if TASK == "binary" else "Multiclass: HC / MCI / Dementia"
print(f"Dataset: {DATASET} | {task_desc}")
print(f"Train: {train_path}")
print(f"Test : {test_path}\n")

if TASK == "binary":
    final_df = run_tfidf_binary(train_df, test_df)
else:
    final_df = run_tfidf_multiclass(train_df, test_df)

results_dir = args.results_dir
os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, f"TFIDF_{DATASET}_{balance_tag}_{TASK}.xlsx")
final_df.to_excel(results_path, index=False)

print(f"\nResults saved in: {results_path}")

sys.stdout.close()
