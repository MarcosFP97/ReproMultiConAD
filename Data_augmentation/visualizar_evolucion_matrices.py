"""
Evolucion de matrices de confusion en los experimentos de la Fase 5.

Genera una figura por corpus, tarea y generador. La fila superior contiene los
modelos entrenados solo con datos reales y la inferior los modelos entrenados
con datos sinteticos o con mezclas de datos reales y sinteticos.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS_DIR = Path(
    "/mnt/beegfs/groups/irgroup/sara_tfg/results/BERT_synthetic_analysis"
)
DATASETS = ("ivanova", "pitt")
TASKS = {
    "binary": "binary_hc_dementia",
    "multiclass": "multiclass",
}
SOURCES = ("gemini", "mistral")
REAL_PERCENTAGES = (20, 40, 60, 80, 100)
AUGMENTED_REAL_PERCENTAGES = (20, 40, 60, 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera las matrices de confusion de la Fase 5."
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


def load_confusion_matrix(path):
    try:
        return pd.read_excel(path, sheet_name="confusion_matrix", index_col=0)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[WARN] No se pudo leer {path.name}: {exc}")
        return None


def draw_matrix(ax, matrix, title, color_map):
    ax.set_title(title, fontsize=11)
    if matrix is None:
        ax.text(0.5, 0.5, "Resultado no encontrado", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=color_map,
        cbar=False,
        ax=ax,
        square=True,
        annot_kws={"size": 10},
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        for task, task_name in TASKS.items():
            for source in SOURCES:
                fig, axes = plt.subplots(2, 6, figsize=(22, 8))
                fig.suptitle(
                    f"{dataset.capitalize()} - {task.capitalize()} - "
                    f"{source.capitalize()}",
                    fontsize=17,
                )

                axes[0, 0].axis("off")
                for column, percentage in enumerate(REAL_PERCENTAGES, start=1):
                    matrix = load_confusion_matrix(
                        real_result_path(
                            results_dir, dataset, task_name, percentage
                        )
                    )
                    draw_matrix(
                        axes[0, column],
                        matrix,
                        f"Solo real: {percentage}%",
                        "Blues",
                    )

                synthetic_matrix = load_confusion_matrix(
                    synthetic_result_path(results_dir, dataset, task_name, source)
                )
                draw_matrix(
                    axes[1, 0],
                    synthetic_matrix,
                    "Solo sintético",
                    "Oranges",
                )

                for column, percentage in enumerate(
                    AUGMENTED_REAL_PERCENTAGES, start=1
                ):
                    matrix = load_confusion_matrix(
                        augmented_result_path(
                            results_dir,
                            dataset,
                            task_name,
                            source,
                            percentage,
                        )
                    )
                    draw_matrix(
                        axes[1, column],
                        matrix,
                        f"Real {percentage}% + sintético {100 - percentage}%",
                        "Oranges",
                    )

                axes[1, 5].axis("off")
                fig.tight_layout(rect=(0, 0, 1, 0.94))
                output_path = output_dir / (
                    f"fase5_matrices_{dataset}_{task}_{source}.png"
                )
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"[SAVE] {output_path}")


if __name__ == "__main__":
    main()
