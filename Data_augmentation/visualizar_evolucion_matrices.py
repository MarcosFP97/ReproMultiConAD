import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# CONFIGURACIÓN
# ============================================================
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

# ============================================================
# BUCLE PRINCIPAL (DATASET -> MODO)
# ============================================================

for dataset in DATASETS:
    for mode in MODES:
        print(f"\n--- Procesando matrices para: {dataset.upper()} ({mode}) ---")
        
        # Ahora usamos 6 columnas para poder alinear por porcentaje
        fig, axes = plt.subplots(2, 6, figsize=(26, 10))
        fig.suptitle(
            f"Evolución de la Confusión: {dataset.upper()} ({mode.capitalize()})\n"
            f"(Fila Superior: Baseline | Fila Inferior: Augmented)",
            fontsize=20, fontweight='bold', y=0.98
        )

        # Ocultamos todos los ejes al principio
        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")

        # --- FILA 1: BASELINE ---
        # La colocamos en columnas 1..5 para que 20,40,60,80 queden alineados con augmented
        for i, p in enumerate(X_BASELINE):
            col = i + 1   # desplazamiento a la derecha
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
        # La colocamos en columnas 0..4
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