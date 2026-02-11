import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any

# --- 1. DEFINICIÓN DE REGEX Y FUNCIONES ---

RE_SHORT = re.compile(r"\(\s*\.\s*\)")          # (.)
RE_MEDIUM = re.compile(r"\(\s*\.\.\s*\)")       # (..)
RE_LONG = re.compile(r"\(\s*\.\.\.\s*\)")       # (...)
RE_TIMED = re.compile(r"\(\s*\d+(?:\.\d+)?\s*\)") # (3.5)

def count_pauses(text: str) -> Dict[str, int]:
    """Cuenta los tipos de pausas en el texto."""
    if not text:
        return {k: 0 for k in ["n_pause_short", "n_pause_medium", "n_pause_long", "n_pause_timed", "n_pause_total"]}

    n_short = len(RE_SHORT.findall(text))
    n_medium = len(RE_MEDIUM.findall(text))
    n_long = len(RE_LONG.findall(text))
    n_timed = len(RE_TIMED.findall(text))
    
    return {
        "n_pause_short": n_short,
        "n_pause_medium": n_medium,
        "n_pause_long": n_long,
        "n_pause_timed": n_timed,
        "n_pause_total": n_short + n_medium + n_long + n_timed,
    }

def count_words_clean(text: str) -> int:
    """Cuenta palabras reales ignorando las marcas de pausa."""
    if not text:
        return 0
    
    clean_text = text
    clean_text = RE_SHORT.sub(" ", clean_text)
    clean_text = RE_MEDIUM.sub(" ", clean_text)
    clean_text = RE_LONG.sub(" ", clean_text)
    clean_text = RE_TIMED.sub(" ", clean_text)
    
    words = clean_text.split()
    return len(words)

def generate_pauses_jsonl(input_path, output_path, text_field, label_field, file_id_field):
    print(f"Procesando {input_path}...")
    
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line: continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # --- FILTRO DE EXCLUSIÓN TAUKADIAL ---
            # Verificamos si hay un campo "dataset" o "corpus"
            dataset_val = str(obj.get("dataset", obj.get("corpus", ""))).upper()
            
            # También verificamos el ID por si acaso el dataset no está explícito
            # (En los logs anteriores vimos IDs como 'taukdial-xxx')
            current_id = str(obj.get(file_id_field, obj.get("id", ""))).lower()

            if "TAUKADIAL" in dataset_val or "taukdial" in current_id:
                # Si es Taukadial, saltamos al siguiente ciclo del bucle sin guardar nada
                continue 
            # -------------------------------------

            text = obj.get(text_field) or ""
            diagnosis = obj.get(label_field)
            file_id = obj.get(file_id_field, obj.get("id", idx))

            # Cálculos
            counts = count_pauses(text)
            n_words = count_words_clean(text)
            
            # Tasa (evitando división por cero)
            pause_rate = (counts["n_pause_total"] / n_words) if n_words > 0 else 0.0

            out_obj = {
                "file_id": file_id,
                "diagnosis": diagnosis,
                "n_words": n_words,
                "pause_rate": pause_rate,
                **counts, 
            }
            
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            
    print(f"Finalizado. Guardado en {output_path} (Sin Taukadial)")

# --- 2. EJECUCIÓN DEL ANÁLISIS ---

# 1. Generar el fichero con todas las métricas (Filtrando Taukadial)
generate_pauses_jsonl(
    input_path="/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Ivanova.jsonl",
    output_path="pauses_analysis_filtered.jsonl",
    text_field="Text_interviewer_participant", 
    label_field="Diagnosis",       
    file_id_field="File_ID",       
)

# 2. Cargar datos
data = []
with open('pauses_analysis_filtered.jsonl', 'r') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except:
            continue

if data:
    df = pd.DataFrame(data)
    
    print(f"Total de muestras analizadas: {len(df)}")
    
    # ---------------------------------------------------------
    # GRÁFICO 1: PAUSAS TOTALES (SIN NORMALIZAR)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='diagnosis', y='n_pause_total', data=df, showfliers=False)
    sns.stripplot(x='diagnosis', y='n_pause_total', data=df, color='black', alpha=0.3, jitter=True)
    
    plt.title('IVANOVA - Distribución de Pausas Totales (Sin Taukadial)')
    plt.ylabel('Número de Pausas')
    plt.xlabel('Diagnóstico')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.savefig('analisis_IVANOVA_pausas_TOTAL_filtered.png')
    plt.close()
    print("--- Gráfico 1 Generado: analisis_IVANOVA_pausas_TOTAL_filtered.png ---")

    # ---------------------------------------------------------
    # GRÁFICO 2: PAUSAS NORMALIZADAS (TASA)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='diagnosis', y='pause_rate', data=df, showfliers=False)
    sns.stripplot(x='diagnosis', y='pause_rate', data=df, color='black', alpha=0.3, jitter=True)
    
    plt.title('IVANOVA - Tasa de Pausas Normalizada (Sin Taukadial)')
    plt.ylabel('Pausas / Palabra')
    plt.xlabel('Diagnóstico')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.savefig('analisis_IVANOVA_pausas_NORMALIZADO_filtered.png')
    plt.close()
    print("--- Gráfico 2 Generado: analisis_IVANOVA_pausas_NORMALIZADO_filtered.png ---")
    
    # ---------------------------------------------------------
    # ESTADÍSTICAS COMPARATIVAS
    # ---------------------------------------------------------
    print("\n========= SPA - ESTADÍSTICAS (SIN TAUKADIAL) =========")
    print("\n1. Media de Pausas Totales (Brutas):")
    print(df.groupby('diagnosis')['n_pause_total'].mean())

    print("\n2. Media de Tasa de Pausas (Normalizadas):")
    print(df.groupby('diagnosis')['pause_rate'].mean())
    
    print("\n3. Media de longitud del discurso (Nº palabras):")
    print(df.groupby('diagnosis')['n_words'].mean())

else:
    print("Error: No se encontraron datos tras el filtrado.")