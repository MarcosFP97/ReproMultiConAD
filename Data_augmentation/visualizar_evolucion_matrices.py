"""
Grids de evolución de matrices de confusión: solo-real vs. aumentado con Gemini.

Renderiza una cuadrícula 2×6 de heatmaps por (dataset, mode): fila superior = baseline
(real 20–100%), fila inferior = aumentado (base 0–80%). El diseño de 6 columnas alinea
ambas filas en el mismo eje de porcentajes para facilitar la comparación visual.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/results/"
DATASETS = ["ivanova", "pitt"]
MODES = ["binary", "multiclass"]

X_BASELINE = [20, 40, 60, 80, 100]
X_SYNTH = [0, 20, 40, 60, 80]


def load_cm(path):
    try:
        df = pd.read_excel(path, sheet_name="confusion_matrix", index_col=0)
        return df
    except Exception:
        return None


for dataset in DATASETS:
    for mode in MODES:
        print(f"\n--- Procesando matrices para: {dataset.upper()} ({mode}) ---")

        # 6 columnas para que ambas filas compartan el mismo eje de porcentajes (0,20,40,60,80,100).
        fig, axes = plt.subplots(2, 6, figsize=(26, 10))
        fig.suptitle(
            f"Evolución de la Confusión: {dataset.upper()} ({mode.capitalize()})\n"
            f"(Fila Superior: Baseline | Fila Inferior: Augmented)",
            fontsize=20, fontweight='bold', y=0.98
        )

        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")

        # --- FILA 1: BASELINE ---
        # Columnas 1..5 para que 20,40,60,80,100 queden alineados con el 0,20,40,60,80 de augmented.
        for i, p in enumerate(X_BASELINE):
            col = i + 1
            ax = axes[0, col]

            path = os.path.join(RESULTS_DIR, f"balancedBERT_{dataset}_{p}_{mode}.xlsx")
            cm = load_cm(path)

            ax.axis("on")
            if cm is not None:
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    ax=ax, cbar=False, annot_kws={"size": 12}
                )
                ax.set_title(f"Real {p}%", fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f"No encontrado:\n{p}% {mode}",
                        ha='center', va='center')
                ax.set_title(f"Real {p}%", fontsize=14, fontweight='bold')

            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")

        # --- FILA 2: AUGMENTED ---
        # Columnas 0..4 para los porcentajes base 0, 20, 40, 60, 80.
        for i, p in enumerate(X_SYNTH):
            col = i
            ax = axes[1, col]

            path = os.path.join(RESULTS_DIR, f"balancedBERT_{dataset}_{p}_{mode}_GEMINI.xlsx")
            cm = load_cm(path)

            ax.axis("on")
            if cm is not None:
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Reds',
                    ax=ax, cbar=False, annot_kws={"size": 12}
                )
                ax.set_title(f"Augmented (Base {p}%)", fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f"No encontrado:\nSynth {p}% {mode}",
                        ha='center', va='center')
                ax.set_title(f"Augmented (Base {p}%)", fontsize=14, fontweight='bold')

            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        output_name = f"evolucion_matrices_{dataset}_{mode}_GEMINI.png"
        plt.savefig(os.path.join(RESULTS_DIR, output_name), dpi=300)
        plt.close()

        print(f"[ÉXITO] Gráfica guardada: {output_name}")

print("\n¡Proceso finalizado! Se han generado 4 imágenes (2 para binario y 2 para multiclass).")