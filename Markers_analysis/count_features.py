#!/usr/bin/env python3
"""
Unified CHAT marker analysis across all ConvoCognition corpora.

Generates 4 figures saved to markers_analysis/figs/:
  - español_rep_rate.png
  - english_pause_rate.png
  - english_rep_rate.png
  - english_ref_rate.png

Each figure shows a boxplot of the feature distribution per diagnosis class.
Individual data points are overlaid as a strip plot colored by corpus, so
corpus-level effects are immediately visible (e.g. PerLA has no pauses,
Taukadial has no CHAT markers at all since it was auto-transcribed).
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Regex patterns ─────────────────────────────────────────────────────────────
RE_SHORT    = re.compile(r"\(\s*\.\s*\)")
RE_MEDIUM   = re.compile(r"\(\s*\.\.\s*\)")
RE_LONG     = re.compile(r"\(\s*\.\.\.\s*\)")
RE_TIMED    = re.compile(r"\(\s*\d+(?:\.\d+)?\s*\)")
RE_REP      = re.compile(r"\[\s*/\s*\]")
RE_REF      = re.compile(r"\[\s*//\s*\]")
RE_PAUSE_ALL = re.compile(r"\(\s*\.+\s*\)|\(\s*\d+(?:\.\d+)?\s*\)")

# ── Dataset configuration ──────────────────────────────────────────────────────
_HPC_BASE   = Path("/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection")
_LOCAL_BASE = Path.cwd() / "results_cha_collection"
BASE = _HPC_BASE if _HPC_BASE.exists() else _LOCAL_BASE

TEXT_FIELD    = "Text_interviewer_participant"
LABEL_FIELD   = "Diagnosis"
FILE_ID_FIELD = "File_ID"
DIAG_ORDER    = ["HC", "MCI", "Dementia"]

# Diagnosis normalization — mirrors preprocess_global_spanish.py / prepare_light_df
DIAG_MAP = {
    "AD":               "Dementia",
    "DM":               "Dementia",
    "DTA":              "Dementia",
    "D":                "Dementia",
    "PossibleAD":       "Dementia",
    "ProbableAD":       "Dementia",
    "Probable":         "Dementia",
    "potential dementia": "Dementia",
    "Alzheimer's":      "Dementia",
    "Control":          "HC",
    "Conrol":           "HC",
    "NC":               "HC",
    "H":                "HC",
}

# Labels excluded from analysis (same as clean_diagnosis in preprocessing)
DIAG_EXCLUDE = {"Vascular", "Memory", "Aphasia", "Pick's", "Other", "Unknown"}
FIGS_DIR      = Path(__file__).parent / "figs"

FEATURES: dict[str, tuple[str, str]] = {
    "pause_rate": ("Tasa de Pausas",          "Pausas / palabra"),
    "rep_rate":   ("Tasa de Repeticiones",    "Repeticiones / palabra"),
    "ref_rate":   ("Tasa de Reformulaciones", "Reformulaciones / palabra"),
}

FEATURES_BY_LANGUAGE = {
    "Español": ("rep_rate",),
    "English": ("pause_rate", "rep_rate", "ref_rate"),
}

# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_features(text: str) -> dict:
    if not text:
        return dict(n_pause=0, n_rep=0, n_ref=0, n_words=0,
                    pause_rate=0.0, rep_rate=0.0, ref_rate=0.0)

    n_pause = (len(RE_SHORT.findall(text))
               + len(RE_MEDIUM.findall(text))
               + len(RE_LONG.findall(text))
               + len(RE_TIMED.findall(text)))
    n_rep = len(RE_REP.findall(text))
    n_ref = len(RE_REF.findall(text))

    clean = RE_REP.sub(" ", RE_REF.sub(" ", RE_PAUSE_ALL.sub(" ", text)))
    n_words = len(clean.split())

    return dict(
        n_pause=n_pause, n_rep=n_rep, n_ref=n_ref, n_words=n_words,
        pause_rate=n_pause / n_words if n_words > 0 else 0.0,
        rep_rate=n_rep   / n_words if n_words > 0 else 0.0,
        ref_rate=n_ref   / n_words if n_words > 0 else 0.0,
    )


def load_dataset(name: str, paths: "Path | list[Path]") -> list[dict]:
    if isinstance(paths, Path):
        paths = [paths]

    rows = []
    for path in paths:
        if not path.exists():
            print(f"  [WARN] Not found: {path}")
            continue

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                diagnosis = obj.get(LABEL_FIELD)
                diagnosis = DIAG_MAP.get(diagnosis, diagnosis)
                if diagnosis in DIAG_EXCLUDE or diagnosis not in DIAG_ORDER:
                    continue

                text = obj.get(TEXT_FIELD) or ""
                feats = extract_features(text)

                if feats["n_words"] == 0:
                    continue

                rows.append({
                    "file_id":   obj.get(FILE_ID_FIELD, ""),
                    "diagnosis": diagnosis,
                    "dataset":   name,
                    **feats,
                })
    return rows


def build_dataframe(datasets: dict) -> pd.DataFrame:
    rows = []
    for name, paths in datasets.items():
        print(f"  Loading {name}...")
        batch = load_dataset(name, paths)
        print(f"    → {len(batch)} transcripciones válidas")
        rows.extend(batch)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["diagnosis"] = pd.Categorical(
        df["diagnosis"], categories=DIAG_ORDER, ordered=True
    )
    return df


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_feature(
    df: pd.DataFrame,
    feature: str,
    feature_label: str,
    ylabel: str,
    lang_label: str,
    out_path: Path,
    palette: dict,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Boxplot — shows class-level distribution in neutral gray
    sns.boxplot(
        x="diagnosis", y=feature, data=df,
        order=DIAG_ORDER,
        color="whitesmoke",
        showfliers=False,
        width=0.5,
        linewidth=1.3,
        ax=ax,
    )

    # Strip plot — individual points colored by corpus
    sns.stripplot(
        x="diagnosis", y=feature, data=df,
        order=DIAG_ORDER,
        hue="dataset",
        palette=palette,
        alpha=0.65,
        jitter=True,
        size=4,
        dodge=False,
        ax=ax,
    )

    ax.set_title(f"{lang_label} — {feature_label}", fontsize=13, pad=10)
    ax.set_xlabel("Diagnóstico", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.get_legend().remove()

    # Custom legend outside the plot
    handles = [
        mpatches.Patch(color=palette[d], label=d)
        for d in sorted(palette)
        if d in df["dataset"].values
    ]
    ax.legend(
        handles=handles,
        title="Corpus",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=True,
        fontsize=9,
        title_fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out_path.name}")


def generate_plots(
    df: pd.DataFrame,
    lang_label: str,
    datasets: dict,
    figs_dir: Path,
) -> None:
    dataset_names = list(datasets.keys())
    colors = sns.color_palette("tab10", n_colors=len(dataset_names))
    palette = dict(zip(dataset_names, colors))

    lang_slug = lang_label.lower().replace(" ", "_")

    for feature in FEATURES_BY_LANGUAGE[lang_label]:
        feature_label, ylabel = FEATURES[feature]
        out_path = figs_dir / f"{lang_slug}_{feature}.png"
        plot_feature(df, feature, feature_label, ylabel, lang_label, out_path, palette)


def print_stats(df: pd.DataFrame, lang_label: str) -> None:
    print(f"\n{'─'*50}")
    print(f"  {lang_label} — Estadísticas por diagnóstico")
    print(f"{'─'*50}")
    cols = ["pause_rate", "rep_rate", "ref_rate"]
    print(df.groupby("diagnosis", observed=True)[cols].mean().round(5).to_string())
    print()
    print("  Transcripciones por corpus y diagnóstico:")
    print(df.groupby(["dataset", "diagnosis"], observed=True).size().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera diagramas de caja de las marcas CHAT por diagnóstico."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=BASE,
        help="Carpeta con los JSONL originales de cada corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGS_DIR,
        help="Carpeta de salida para las cuatro figuras.",
    )
    return parser.parse_args()


def build_dataset_paths(data_root: Path) -> tuple[dict, dict]:
    spanish = {
        "Ivanova": data_root / "Ivanova.jsonl",
        "PerLA": data_root / "PerLA.jsonl",
    }
    english = {
        "Baycrest": data_root / "Baycrest.jsonl",
        "Delaware": data_root / "Delaware.jsonl",
        "Kempler": data_root / "Kempler.jsonl",
        "Lu": data_root / "Lu.jsonl",
        "Pitt": data_root / "Pitt.jsonl",
        "Taukadial": [
            data_root / "taukadial_English_train.jsonl",
            data_root / "taukadial_English_test.jsonl",
        ],
        "VAS": data_root / "VAS.jsonl",
        "WLS": data_root / "WLS.jsonl",
    }
    return spanish, english


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)
    spanish_datasets, english_datasets = build_dataset_paths(args.data_root)

    print("=== Español ===")
    df_spa = build_dataframe(spanish_datasets)
    if not df_spa.empty:
        print_stats(df_spa, "Español")
        generate_plots(df_spa, "Español", spanish_datasets, args.output_dir)
    else:
        print("  [ERROR] No se cargaron datos en español.")

    print("\n=== English ===")
    df_en = build_dataframe(english_datasets)
    if not df_en.empty:
        print_stats(df_en, "English")
        generate_plots(df_en, "English", english_datasets, args.output_dir)
    else:
        print("  [ERROR] No data loaded for English.")

    print("\nDone. Figures saved to:", args.output_dir)
