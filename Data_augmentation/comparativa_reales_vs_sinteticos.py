import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURACIÓN
# ============================================================
RESULTS_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/results/"
DATASETS = ["ivanova", "pitt"]
MODES = ["binary", "multiclass"]  # <--- Soporte para ambos modos
METRICS_MAP = {
    "Accuracy": "Accuracy",
    "MacroF1": "Macro_f1"
}

# Definimos los puntos X exactos para cada línea
X_BASELINE = [20, 40, 60, 80, 100]
X_SYNTHETIC = [0, 20, 40, 60, 80]

def get_metric_from_excel(path, metric_name):
    try:
        df = pd.read_excel(path, sheet_name="summary")
        return df[metric_name].iloc[0]
    except Exception as e:
        # Imprimimos el path para saber exactamente qué archivo falta
        print(f"Advertencia: No se pudo leer {os.path.basename(path)}. Saltando...")
        return None

# ============================================================
# GENERACIÓN DE GRÁFICAS
# ============================================================

for ds in DATASETS:
    for mode in MODES: # <--- Nuevo bucle por modo
        for label, col_name in METRICS_MAP.items():
            plt.figure(figsize=(10, 6))
            
            # 1. Recopilar y pintar Baseline (Solo Reales: 20-100)
            base_y = []
            for p in X_BASELINE:
                # El nombre del archivo ahora incluye la variable {mode}
                file_path = os.path.join(RESULTS_DIR, f"balancedBERT_{ds}_{p}_{mode}.xlsx")
                base_y.append(get_metric_from_excel(file_path, col_name))
            
            plt.plot(X_BASELINE, base_y, marker='o', linestyle='-', linewidth=2, 
                     label=f'Solo Reales (Baseline - {mode})', color='#2c3e50')

            # 2. Recopilar y pintar Mixto (Reales + Sintéticos: 0-80)
            synth_y = []
            for p in X_SYNTHETIC:
                # El nombre del archivo ahora incluye la variable {mode}
                file_path = os.path.join(RESULTS_DIR, f"balancedBERT_{ds}_{p}_{mode}_GEMINI.xlsx")
                synth_y.append(get_metric_from_excel(file_path, col_name))
            
            plt.plot(X_SYNTHETIC, synth_y, marker='s', linestyle='--', linewidth=2, 
                     label=f'Reales + Sintéticos (Augmented - {mode})', color='#e74c3c')

            # Estética de la gráfica
            plt.title(f"Impacto Aumento Datos con Gemini - {ds.upper()} ({mode.capitalize()}) - {label}", fontsize=14, fontweight='bold')
            plt.xlabel("Porcentaje de Datos Reales utilizados (%)", fontsize=12)
            plt.ylabel(label, fontsize=12)
            plt.xticks([0, 20, 40, 60, 80, 100])
            plt.ylim(0, 1)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            
            # Guardar incluyendo el modo en el nombre del archivo
            output_name = f"plot_{ds}_{mode}_{label}_GEMINI.png"
            plt.savefig(os.path.join(RESULTS_DIR, output_name))
            print(f"[SAVE] Gráfica guardada: {output_name}")
            plt.close()

print("\n¡Gráficas generadas para Binary y Multiclass!")