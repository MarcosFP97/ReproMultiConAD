"""
Gráficas de línea: baseline solo-real vs. rendimiento aumentado con Gemini.

Lee Accuracy y MacroF1 de la hoja 'summary' de cada Excel de resultados de BERT_balanced,
dibuja dos curvas por combinación (dataset, mode, métrica) y guarda un PNG por combinación
en RESULTS_DIR.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/results/"
DATASETS = ["ivanova", "pitt"]
MODES = ["binary", "multiclass"]
METRICS_MAP = {
    "Accuracy": "Accuracy",
    "MacroF1": "Macro_f1"
}

X_BASELINE = [20, 40, 60, 80, 100]
X_SYNTHETIC = [0, 20, 40, 60, 80]


def get_metric_from_excel(path, metric_name):
    try:
        df = pd.read_excel(path, sheet_name="summary")
        return df[metric_name].iloc[0]
    except Exception:
        print(f"Advertencia: No se pudo leer {os.path.basename(path)}. Saltando...")
        return None


for ds in DATASETS:
    for mode in MODES:
        for label, col_name in METRICS_MAP.items():
            plt.figure(figsize=(10, 6))

            # 1. Baseline: modelos entrenados solo con datos reales al 20–100%.
            base_y = []
            for p in X_BASELINE:
                file_path = os.path.join(RESULTS_DIR, f"balancedBERT_{ds}_{p}_{mode}.xlsx")
                base_y.append(get_metric_from_excel(file_path, col_name))

            plt.plot(X_BASELINE, base_y, marker='o', linestyle='-', linewidth=2,
                     label=f'Solo Reales (Baseline - {mode})', color='#2c3e50')

            # 2. Aumentado: fracción real + complemento sintético Gemini al 0–80%.
            synth_y = []
            for p in X_SYNTHETIC:
                file_path = os.path.join(RESULTS_DIR, f"balancedBERT_{ds}_{p}_{mode}_GEMINI.xlsx")
                synth_y.append(get_metric_from_excel(file_path, col_name))

            plt.plot(X_SYNTHETIC, synth_y, marker='s', linestyle='--', linewidth=2,
                     label=f'Reales + Sintéticos (Augmented - {mode})', color='#e74c3c')

            plt.title(f"Impacto Aumento Datos con Gemini - {ds.upper()} ({mode.capitalize()}) - {label}", fontsize=14, fontweight='bold')
            plt.xlabel("Porcentaje de Datos Reales utilizados (%)", fontsize=12)
            plt.ylabel(label, fontsize=12)
            plt.xticks([0, 20, 40, 60, 80, 100])
            plt.ylim(0, 1)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()

            output_name = f"plot_{ds}_{mode}_{label}_GEMINI.png"
            plt.savefig(os.path.join(RESULTS_DIR, output_name))
            print(f"[SAVE] Gráfica guardada: {output_name}")
            plt.close()

print("\n¡Gráficas generadas para Binary y Multiclass!")