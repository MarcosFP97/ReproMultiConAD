"""
Clasificador TF-IDF binario por dataset individual (pipeline individual).

Para cada dataset entrena HC vs Dementia o HC vs MCI (Delaware y Taukadial),
con GridSearchCV sobre Decision Tree, Random Forest, Naive Bayes, SVM y Logistic Regression.
Incluye visualizaciones 2D/3D de la frontera SVM proyectada con TruncatedSVD.
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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import argparse

LABEL = "Text_interviewer_participant"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Nombre del dataset (ej: Pitt, Lu, Baycrest, Delaware, taukadial, ivanova)")
parser.add_argument("--balanced", action="store_true", help="Usar class_weight='balanced' en los clasificadores")
parser.add_argument("--task", default="binary", choices=["binary", "multiclass"], help="Tarea de clasificación")
args = parser.parse_args()

DATASET = args.dataset.strip()
DATASET_LOWER = DATASET.lower()
BALANCED = args.balanced
TASK = args.task
balance_tag = "balanced" if BALANCED else "unbalanced"
class_weight = "balanced" if BALANCED else None

data_dir = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets/TFIDF"
train_path = os.path.join(data_dir, f"train_{DATASET_LOWER}.jsonl")
test_path = os.path.join(data_dir, f"test_{DATASET_LOWER}.jsonl")

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError(
        f"No encuentro los ficheros:\n  {train_path}\n  {test_path}\n"
        "Revisa el nombre del dataset y cómo lo guardaste."
    )

train_df = pd.read_json(train_path, lines=True)
test_df  = pd.read_json(test_path, lines=True)

mci_hc_datasets = {"delaware", "taukadial"}

train_df["Diagnosis"] = train_df["Diagnosis"].replace("AD", "Dementia")
test_df["Diagnosis"]  = test_df["Diagnosis"].replace("AD", "Dementia")

if TASK == "binary":
    if DATASET_LOWER in mci_hc_datasets:
        keep = {"HC", "MCI"}
        positive_label = "MCI"
    else:
        keep = {"HC", "Dementia"}
        positive_label = "Dementia"
else:
    keep = {"HC", "MCI", "Dementia"}
    positive_label = "Dementia"  # no usado en multiclass

train_df = train_df[train_df["Diagnosis"].isin(keep)].copy()
test_df  = test_df[test_df["Diagnosis"].isin(keep)].copy()

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError(
        f"Tras filtrar clases {keep}, te quedaste con train={len(train_df)} / test={len(test_df)}.\n"
        "Revisa que ese dataset realmente tenga esas clases."
    )

def _get_confidence(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return proba.max(axis=1)
    elif hasattr(estimator, "decision_function"):
        dec = estimator.decision_function(X)
        return np.abs(dec) if dec.ndim == 1 else (np.partition(dec, -2, axis=1)[:, -1] - np.partition(dec, -2, axis=1)[:, -2])
    return None

def plot_svm_frontier_2d(X_train_tfidf, y_train, X_test_tfidf, y_test, best_params, dataset_name, positive_label, out_path=None, random_state=42):
    """
    Visualiza una frontera SVM en 2D:
    - Proyecta TF-IDF a 2D con TruncatedSVD
    - Escala (StandardScaler)
    - Entrena SVM con best_params (kernel/C)
    - Dibuja frontera (nivel 0) y márgenes (niveles ±1)
    """
    svd = TruncatedSVD(n_components=2, random_state=random_state)
    Xtr_2d = svd.fit_transform(X_train_tfidf)
    Xte_2d = svd.transform(X_test_tfidf)

    # SVC se beneficia del escalado porque opera en espacio euclídeo, no en el TF-IDF original.
    svm_kwargs = {
        "kernel": best_params.get("kernel", "linear"),
        "C": best_params.get("C", 1.0),
    }

    clf2d = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**svm_kwargs)),
    ])
    clf2d.fit(Xtr_2d, y_train)

    x_min, x_max = Xtr_2d[:, 0].min() - 0.8, Xtr_2d[:, 0].max() + 0.8
    y_min, y_max = Xtr_2d[:, 1].min() - 0.8, Xtr_2d[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf2d.decision_function(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    # frontera 0 y márgenes ±1
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"])

    classes = ["HC", positive_label]
    markers = {"HC": "o", positive_label: "s"}

    for cls in classes:
        tr_idx = (y_train == cls)
        te_idx = (y_test == cls)

        plt.scatter(Xtr_2d[tr_idx, 0], Xtr_2d[tr_idx, 1],
                    marker=markers[cls], alpha=0.30, label=f"train {cls}")
        plt.scatter(Xte_2d[te_idx, 0], Xte_2d[te_idx, 1],
                    marker=markers[cls], alpha=0.95, label=f"test {cls}")

    plt.title(f"{dataset_name} | HC vs {positive_label} | SVM({svm_kwargs['kernel']}, C={svm_kwargs['C']}) en SVD-2D")
    plt.xlabel("SVD comp. 1")
    plt.ylabel("SVD comp. 2")
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()

def plot_svm_svd3d_scatter(X_train_tfidf, y_train, X_test_tfidf, y_test,
                          best_params, dataset_name, positive_label,
                          out_path=None, random_state=42):
    """
    Visualiza HC vs positive_label en 3D:
    - Proyecta TF-IDF a 3D con TruncatedSVD
    - Escala (StandardScaler)
    - (Opcional) entrena SVM 3D solo para coherencia, pero NO dibuja hiperplano
    - Dibuja scatter 3D train/test por clase
    """
    svd = TruncatedSVD(n_components=3, random_state=random_state)
    Xtr_3d = svd.fit_transform(X_train_tfidf)
    Xte_3d = svd.transform(X_test_tfidf)

    # El SVM 3D no se dibuja, pero se entrena para mantener coherencia con los parámetros del grid.
    svm_kwargs = {
        "kernel": best_params.get("kernel", "linear"),
        "C": best_params.get("C", 1.0),
    }
    clf3d = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**svm_kwargs)),
    ])
    clf3d.fit(Xtr_3d, y_train)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    classes = ["HC", positive_label]
    markers = {"HC": "o", positive_label: "s"}

    for cls in classes:
        tr_idx = (y_train == cls)
        te_idx = (y_test == cls)

        ax.scatter(Xtr_3d[tr_idx, 0], Xtr_3d[tr_idx, 1], Xtr_3d[tr_idx, 2],
                   marker=markers[cls], alpha=0.25, label=f"train {cls}")
        ax.scatter(Xte_3d[te_idx, 0], Xte_3d[te_idx, 1], Xte_3d[te_idx, 2],
                   marker=markers[cls], alpha=0.95, label=f"test {cls}")

    ax.set_title(f"{dataset_name} | HC vs {positive_label} | SVM({svm_kwargs['kernel']}, C={svm_kwargs['C']}) en SVD-3D")
    ax.set_xlabel("SVD comp. 1")
    ax.set_ylabel("SVD comp. 2")
    ax.set_zlabel("SVD comp. 3")
    ax.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()

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
        # subo max_iter para evitar warnings de convergencia
        "Logistic Regression": (LogisticRegression(random_state=random_state, max_iter=2000, class_weight=class_weight), {"C": [0.1, 1, 10]}),
    }

    results = []
    
    # StratifiedKFold preserva la proporción de clases en cada fold; crítico para los datasets clínicos desbalanceados.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, (clf, params) in classifiers.items():
        grid_search = GridSearchCV(clf, params, cv=cv, scoring="f1_macro")
        grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)

        best_model = grid_search.best_estimator_
        conf = _get_confidence(best_model, X_test)
        
        if name == "SVM":
            fig_dir = "/mnt/beegfs/groups/irgroup/sara_tfg/results/figs/"
            os.makedirs(fig_dir, exist_ok=True)

            fig_path = os.path.join(
                fig_dir,
                f"SVM_{balance_tag}_frontier_{DATASET}_HC_vs_{positive_label}.png"
            )

            plot_svm_frontier_2d(
                X_train, y_train.values,
                X_test,  y_test.values,
                best_params=grid_search.best_params_,
                dataset_name=DATASET,
                positive_label=positive_label,
                out_path=fig_path,
                random_state=random_state
            )
            print(f"[SVM] Figura guardada en: {fig_path}")

            fig_path_3d = os.path.join(fig_dir, f"SVM_{balance_tag}_scatter3D_{DATASET}_HC_vs_{positive_label}.png")

            plot_svm_svd3d_scatter(
                X_train, y_train.values,
                X_test,  y_test.values,
                best_params=grid_search.best_params_,
                dataset_name=DATASET,
                positive_label=positive_label,
                out_path=fig_path_3d,
                random_state=random_state
            )
            print(f"[SVM] Figura 3D guardada en: {fig_path_3d}")


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
        print("Aciertos por clase (y_true):")
        print(eval_df.loc[eval_df["correct"], "y_true"].value_counts())
        print("Fallos por clase (y_true):")
        print(eval_df.loc[~eval_df["correct"], "y_true"].value_counts())

        labels_order = ["HC", positive_label]
        cm = confusion_matrix(eval_df["y_true"], eval_df["y_pred"], labels=labels_order)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels_order],
            columns=[f"pred_{l}" for l in labels_order],
        )
        print("\nMatriz de confusión (HC primero):")
        print(cm_df)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics_dict = {
            "Dataset": DATASET,
            "Binary_Positive": positive_label,
            "Balanced": BALANCED,
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
        print("\nMatriz de confusión:")
        print(cm_df)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics_dict = {
            "Dataset": DATASET,
            "Task": "multiclass",
            "Balanced": BALANCED,
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


log_dir = "/mnt/beegfs/groups/irgroup/sara_tfg/logs/"
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, f"TFIDF_{DATASET}_{balance_tag}_{TASK}.log")

sys.stdout = open(log_path, "w", encoding="utf-8")
sys.stderr = sys.stdout

print(f"Logging en: {log_path}")
task_desc = f"Binary: HC vs {positive_label}" if TASK == "binary" else "Multiclass: HC / MCI / Dementia"
print(f"Dataset: {DATASET} | {task_desc} | Balanced: {BALANCED}")
print(f"Train: {train_path}")
print(f"Test : {test_path}\n")

if TASK == "binary":
    final_df = run_tfidf_binary(train_df, test_df)
else:
    final_df = run_tfidf_multiclass(train_df, test_df)

results_dir = "/mnt/beegfs/groups/irgroup/sara_tfg/results/"
os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, f"TFIDF_{DATASET}_{balance_tag}_{TASK}.xlsx")
final_df.to_excel(results_path, index=False)

print(f"\nResultados guardados en: {results_path}")

sys.stdout.close()
