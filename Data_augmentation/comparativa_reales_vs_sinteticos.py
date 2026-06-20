"""
Curvas de rendimiento para los experimentos reales y sinteticos de la Fase 5.

Lee la hoja ``summary`` de los Excel generados por BERT_balanced.py y compara,
para cada corpus y tarea, el entrenamiento solo con datos reales frente a las
mezclas con datos sinteticos de Gemini y Mistral.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RESULTS_DIR = Path(
    "/mnt/beegfs/groups/irgroup/sara_tfg/results/BERT_synthetic_analysis"
)
DATASETS = ("ivanova", "pitt")
TASKS = {
    "binary": "binary_hc_dementia",
    "multiclass": "multiclass",
}
SOURCES = ("gemini", "mistral")
METRICS = {
    "Accuracy": "Accuracy",
    "Macro-F1": "Macro_f1",
}
REAL_PERCENTAGES = (20, 40, 60, 80, 100)
AUGMENTED_REAL_PERCENTAGES = (20, 40, 60, 80)
SOURCE_STYLES = {
    "gemini": ("#D55E00", "s"),
    "mistral": ("#0072B2", "^"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera las curvas de resultados de la Fase 5."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Carpeta que contiene los Excel de BERT_synthetic_analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta de salida. Por defecto, <results-dir>/plots.",
    )
    return parser.parse_args()


def real_result_path(results_dir, dataset, task_name, real_percentage):
    return results_dir / (
        f"bert_balanced_real_{task_name}_{dataset}_real{real_percentage}.xlsx"
    )


def synthetic_result_path(results_dir, dataset, task_name, source):
    return results_dir / (
        f"bert_balanced_synthetic_{source}_{task_name}_{dataset}_synthetic100.xlsx"
    )


def augmented_result_path(
    results_dir, dataset, task_name, source, real_percentage
):
    synthetic_percentage = 100 - real_percentage
    return results_dir / (
        f"bert_balanced_augmented_{source}_{task_name}_{dataset}"
        f"_real{real_percentage}_synthetic{synthetic_percentage}.xlsx"
    )


def get_metric(path, metric_name):
    try:
        summary = pd.read_excel(path, sheet_name="summary")
        return float(summary.loc[0, metric_name])
    except (FileNotFoundError, KeyError, ValueError, IndexError) as exc:
        print(f"[WARN] No se pudo leer {path.name}: {exc}")
        return float("nan")


def get_majority_baseline(path):
    """Devuelve accuracy y macro-F1 al predecir siempre la clase mayoritaria."""
    try:
        summary = pd.read_excel(path, sheet_name="summary")
        support_columns = [
            column for column in summary.columns if column.endswith("_support")
        ]
        supports = [
            float(summary.loc[0, column])
            for column in support_columns
            if pd.notna(summary.loc[0, column])
        ]
        total = sum(supports)
        majority_support = max(supports)
        majority_fraction = majority_support / total
        majority_f1 = 2 * majority_fraction / (1 + majority_fraction)
        return {
            "Accuracy": majority_fraction,
            "Macro_f1": majority_f1 / len(supports),
        }
    except (FileNotFoundError, ValueError, IndexError) as exc:
        print(f"[WARN] No se pudo calcular el baseline de {path.name}: {exc}")
        return {"Accuracy": float("nan"), "Macro_f1": float("nan")}


def get_series(results_dir, dataset, task_name, metric_column):
    real_values = [
        get_metric(
            real_result_path(results_dir, dataset, task_name, percentage),
            metric_column,
        )
        for percentage in REAL_PERCENTAGES
    ]
    source_values = {}
    for source in SOURCES:
        values = [
            get_metric(
                synthetic_result_path(results_dir, dataset, task_name, source),
                metric_column,
            )
        ]
        values.extend(
            get_metric(
                augmented_result_path(
                    results_dir,
                    dataset,
                    task_name,
                    source,
                    percentage,
                ),
                metric_column,
            )
            for percentage in AUGMENTED_REAL_PERCENTAGES
        )
        source_values[source] = values
    return real_values, source_values


def draw_metric_panel(
    ax,
    results_dir,
    dataset,
    task,
    task_name,
    metric_label,
    metric_column,
    show_legend=True,
):
    real_values, source_values = get_series(
        results_dir, dataset, task_name, metric_column
    )
    ax.plot(
        REAL_PERCENTAGES,
        real_values,
        marker="o",
        linewidth=2,
        color="#333333",
        label="Solo datos reales",
    )

    x_values = (0,) + AUGMENTED_REAL_PERCENTAGES
    for source in SOURCES:
        color, marker = SOURCE_STYLES[source]
        ax.plot(
            x_values,
            source_values[source],
            marker=marker,
            linestyle="--",
            linewidth=2,
            color=color,
            label=f"Real + {source.capitalize()}",
        )

    baseline = get_majority_baseline(
        real_result_path(results_dir, dataset, task_name, 100)
    )[metric_column]
    ax.axhline(
        baseline,
        color="#777777",
        linestyle=":",
        linewidth=2,
        label="Clasificador mayoritario",
    )
    ax.text(
        99,
        baseline + 0.012,
        f"Trivial: {baseline:.3f}",
        color="#666666",
        fontsize=9,
        ha="right",
        va="bottom",
    )

    task_label = "Binaria" if task == "binary" else "Multiclase"
    ax.set_title(f"{dataset.capitalize()} - {task_label}")
    ax.set_xlabel("Porcentaje de datos reales")
    ax.set_ylabel(metric_label)
    ax.set_xticks((0, 20, 40, 60, 80, 100))
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    if show_legend:
        ax.legend()


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        for task, task_name in TASKS.items():
            for metric_label, metric_column in METRICS.items():
                fig, ax = plt.subplots(figsize=(9, 5.5))
                draw_metric_panel(
                    ax,
                    results_dir,
                    dataset,
                    task,
                    task_name,
                    metric_label,
                    metric_column,
                )
                fig.tight_layout()

                metric_slug = metric_column.lower()
                output_path = output_dir / (
                    f"fase5_{dataset}_{task}_{metric_slug}_real_vs_synthetic.png"
                )
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"[SAVE] {output_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for ax, (dataset, task) in zip(
        axes.flat,
        (
            ("pitt", "binary"),
            ("ivanova", "binary"),
            ("pitt", "multiclass"),
            ("ivanova", "multiclass"),
        ),
    ):
        draw_metric_panel(
            ax,
            results_dir,
            dataset,
            task,
            TASKS[task],
            "Macro-F1",
            "Macro_f1",
            show_legend=False,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Fase 5: rendimiento frente al clasificador mayoritario",
        fontsize=17,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    summary_path = output_dir / "fase5_macro_f1_baseline_mayoritario.png"
    fig.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
