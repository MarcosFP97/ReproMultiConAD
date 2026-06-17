#!/usr/bin/env python3
"""
Visualización de resultados del experimento 06 (TF-IDF por dataset individual).

Lee todos los archivos TFIDF_{dataset}_{balance}_{task}.xlsx del directorio
de resultados y genera figuras separadas por idioma en results_dir/figs_06/:

  EN/heatmap_binary.png         — F1 macro por dataset × clasificador (binario, inglés)
  EN/heatmap_multiclass.png     — ídem multiclase
  EN/binary_vs_multiclass.png   — mejor F1 binario vs multiclase por dataset
  EN/balanced_vs_unbalanced.png — efecto del class_weight
  ES/heatmap_binary.png         — ídem para español
  ...
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Rutas ─────────────────────────────────────────────────────────────────────
_HPC   = Path("/mnt/beegfs/groups/irgroup/sara_tfg/results")
_LOCAL = Path("/Users/saracastrolopez/Desktop/clases/tfg/results")
RESULTS_DIR = _HPC if _HPC.exists() else _LOCAL
FIGS_DIR = RESULTS_DIR / "figs_06"

# ── Constantes de presentación ────────────────────────────────────────────────
CLASSIFIER_SHORT = {
    "Decision Tree":     "DT",
    "Random Forest":     "RF",
    "Naive Bayes":       "NB",
    "SVM":               "SVM",
    "Logistic Regression": "LR",
}
CLASSIFIER_ORDER = ["DT", "RF", "NB", "SVM", "LR"]

DATASET_LANG = {
    "pitt": "EN", "delaware": "EN", "lu": "EN",
    "taukadial": "EN", "vas": "EN", "wls": "EN",
    "ivanova": "ES",
}


# ── Carga ─────────────────────────────────────────────────────────────────────
def load_results(results_dir: Path) -> pd.DataFrame:
    dfs = []
    pattern = re.compile(r"TFIDF_(.+)_(balanced|unbalanced)_(binary|multiclass)\.xlsx")

    for path in sorted(results_dir.glob("TFIDF_*_*.xlsx")):
        m = pattern.match(path.name)
        if not m:
            continue
        dataset, balance, task = m.groups()

        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"[WARN] {path.name}: {e}")
            continue

        df["Task"]    = task
        df["Balance"] = balance
        df["Dataset"] = dataset.lower()
        df["Lang"]    = df["Dataset"].map(DATASET_LANG).fillna("?")
        df["Clf"]     = df["Classifier"].map(CLASSIFIER_SHORT).fillna(df["Classifier"])
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No se encontraron archivos TFIDF_*.xlsx en {results_dir}")

    return pd.concat(dfs, ignore_index=True)


# ── Figura 1 & 2: Heatmap dataset × clasificador ──────────────────────────────
def plot_heatmap(df: pd.DataFrame, task: str, figs_dir: Path, lang_label: str) -> None:
    sub = df[df["Task"] == task]
    if sub.empty:
        print(f"  [SKIP] Sin datos para heatmap {task}")
        return

    pivot = (
        sub.groupby(["Dataset", "Clf"])["Macro_f1"]
        .max()
        .unstack("Clf")
        .reindex(columns=CLASSIFIER_ORDER)
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(7, max(3, len(pivot) * 0.65 + 1)))
    sns.heatmap(
        pivot,
        annot=True, fmt=".2f",
        cmap="YlOrRd",
        vmin=0.3, vmax=1.0,
        linewidths=0.5,
        ax=ax,
    )
    title_task = "Binario (HC vs Dementia/MCI)" if task == "binary" else "Multiclase (HC / MCI / Dementia)"
    ax.set_title(f"F1 macro — TF-IDF individual ({lang_label})\n{title_task}", fontsize=11, pad=10)
    ax.set_xlabel("Clasificador", fontsize=10)
    ax.set_ylabel("Dataset", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    out = figs_dir / f"heatmap_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out.relative_to(FIGS_DIR.parent)}")


# ── Figura 3: Binario vs multiclase ───────────────────────────────────────────
def plot_binary_vs_multiclass(df: pd.DataFrame, figs_dir: Path, lang_label: str) -> None:
    best = (
        df.groupby(["Dataset", "Task"])["Macro_f1"]
        .max()
        .reset_index()
    )
    if best.empty:
        return

    order = (
        best[best["Task"] == "binary"]
        .sort_values("Macro_f1", ascending=False)["Dataset"]
        .tolist()
    )
    for d in best["Dataset"].unique():
        if d not in order:
            order.append(d)

    fig, ax = plt.subplots(figsize=(max(5, len(order) * 1.2 + 2), 4.5))
    sns.barplot(
        data=best, x="Dataset", y="Macro_f1", hue="Task",
        order=order,
        palette={"binary": "#4C72B0", "multiclass": "#DD8452"},
        ax=ax,
    )
    ax.set_title(f"Mejor F1 macro por dataset ({lang_label}) — binario vs multiclase", fontsize=11, pad=10)
    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("F1 macro", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.legend(title="Tarea", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = figs_dir / "binary_vs_multiclass.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out.relative_to(FIGS_DIR.parent)}")


# ── Figura 4: Balanced vs unbalanced ─────────────────────────────────────────
def plot_balanced_vs_unbalanced(df: pd.DataFrame, figs_dir: Path, lang_label: str) -> None:
    from matplotlib.patches import Patch

    best = (
        df.groupby(["Dataset", "Task", "Balance"])["Macro_f1"]
        .max()
        .reset_index()
    )
    pivot = best.pivot_table(index=["Dataset", "Task"], columns="Balance", values="Macro_f1").reset_index()
    if "balanced" not in pivot.columns or "unbalanced" not in pivot.columns:
        print(f"  [SKIP] Sin suficientes datos para balanced vs unbalanced ({lang_label})")
        return

    pivot["delta"] = pivot["balanced"] - pivot["unbalanced"]
    pivot = pivot.sort_values("delta", ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.5 + 1)))
    colors = pivot["Task"].map({"binary": "#4C72B0", "multiclass": "#DD8452"})
    ax.barh(
        y=pivot["Dataset"] + " (" + pivot["Task"] + ")",
        width=pivot["delta"],
        color=colors,
        alpha=0.85,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Efecto del class_weight='balanced' ({lang_label})\n(F1 macro balanced − unbalanced)",
                 fontsize=11, pad=10)
    ax.set_xlabel("Δ F1 macro", fontsize=10)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)

    legend_elements = [
        Patch(facecolor="#4C72B0", alpha=0.85, label="binary"),
        Patch(facecolor="#DD8452", alpha=0.85, label="multiclass"),
    ]
    ax.legend(handles=legend_elements, title="Tarea", fontsize=9)

    plt.tight_layout()
    out = figs_dir / "balanced_vs_unbalanced.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out.relative_to(FIGS_DIR.parent)}")


def generate_lang(df: pd.DataFrame, lang: str, lang_label: str) -> None:
    sub = df[df["Lang"] == lang]
    if sub.empty:
        print(f"  [SKIP] Sin datos para {lang_label}")
        return

    figs_dir = FIGS_DIR / lang
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {lang_label} ({sub['Dataset'].nunique()} datasets) ---")
    plot_heatmap(sub, "binary", figs_dir, lang_label)
    plot_heatmap(sub, "multiclass", figs_dir, lang_label)
    plot_binary_vs_multiclass(sub, figs_dir, lang_label)
    plot_balanced_vs_unbalanced(sub, figs_dir, lang_label)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    print(f"Leyendo resultados desde: {RESULTS_DIR}")
    df = load_results(RESULTS_DIR)
    print(f"  {len(df)} filas cargadas ({df['Dataset'].nunique()} datasets, "
          f"{df['Task'].nunique()} tareas)")

    generate_lang(df, "EN", "Inglés")
    generate_lang(df, "ES", "Español")

    print(f"\nDone. Figuras en: {FIGS_DIR}")
