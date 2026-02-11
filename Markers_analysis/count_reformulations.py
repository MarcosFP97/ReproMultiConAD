import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any

# --- 1. DEFINICIÓN DE REGEX ---

# Patrones para Repeticiones y Reformulaciones
RE_REP = re.compile(r"\[\s*/\s*\]")   # [/] Repetición
RE_REF = re.compile(r"\[\s*//\s*\]")  # [//] Reformulación

# Patrón de pausas (SOLO para borrarlas al contar palabras, no las analizamos)
RE_PAUSE_CLEAN = re.compile(r"\(\s*\.*\s*\)|\(\s*\d+(?:\.\d+)?\s*\)")

def count_disfluencies(text: str) -> Dict[str, int]:
    """Cuenta exclusivamente repeticiones y reformulaciones."""
    if not text:
        return {"n_rep": 0, "n_ref": 0}

    n_rep = len(RE_REP.findall(text))
    n_ref = len(RE_REF.findall(text))
    
    return {
        "n_rep": n_rep,
        "n_ref": n_ref
    }

def count_words_clean(text: str) -> int:
    """Cuenta palabras reales, limpiando todo tipo de marcas (pausas y disfluencias)."""
    if not text:
        return 0
    
    clean_text = text
    # 1. Borramos las marcas que estamos analizando
    clean_text = RE_REP.sub(" ", clean_text)
    clean_text = RE_REF.sub(" ", clean_text)
    
    # 2. Borramos las pausas para que no cuenten como palabras
    clean_text = RE_PAUSE_CLEAN.sub(" ", clean_text)
    
    # 3. Contamos palabras reales
    words = clean_text.split()
    return len(words)

def generate_disfluency_jsonl(input_path, output_path, text_field, label_field, file_id_field):
    print(f"Procesando {input_path}...")
    
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line: continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # --- FILTRO TAUKADIAL ---
            dataset_val = str(obj.get("dataset", obj.get("corpus", ""))).upper()
            current_id = str(obj.get(file_id_field, obj.get("id", ""))).lower()
            
            if "TAUKADIAL" in dataset_val or "taukdial" in current_id:
                continue 
            # ------------------------

            text = obj.get(text_field) or ""
            diagnosis = obj.get(label_field)
            file_id = obj.get(file_id_field, obj.get("id", idx))

            # Cálculos
            counts = count_disfluencies(text)
            n_words = count_words_clean(text)
            
            # Tasas (Rate)
            rep_rate = (counts["n_rep"] / n_words) if n_words > 0 else 0.0
            ref_rate = (counts["n_ref"] / n_words) if n_words > 0 else 0.0

            out_obj = {
                "file_id": file_id,
                "diagnosis": diagnosis,
                "n_words": n_words,
                "rep_rate": rep_rate,
                "ref_rate": ref_rate,
                **counts,
            }
            
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            
    print(f"Finalizado. Guardado en {output_path}")

# --- 2. EJECUCIÓN DEL ANÁLISIS ---

generate_disfluency_jsonl(
    input_path="/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Ivanova.jsonl",
    output_path="disfluency_analysis.jsonl",
    text_field="Text_interviewer_participant", 
    label_field="Diagnosis",       
    file_id_field="File_ID",       
)

# --- 3. GENERACIÓN DE GRÁFICOS ---
data = []
with open('disfluency_analysis.jsonl', 'r') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except:
            continue

if data:
    df = pd.DataFrame(data)
    print(f"Total muestras analizadas: {len(df)}")
    
    # Ajuste de estilo para que se vea bien
    sns.set_style("whitegrid")
    
    # ------------------------------------------------
    # GRÁFICO 1: Tasa de Repeticiones [/]
    # ------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='diagnosis', y='rep_rate', data=df, showfliers=False)
    sns.stripplot(x='diagnosis', y='rep_rate', data=df, color='black', alpha=0.3, jitter=True)
    
    plt.title('IVANOVA - Tasa de Repeticiones [/] (Normalizada)')
    plt.ylabel('Repeticiones / Palabra')
    plt.xlabel('Diagnóstico')
    
    plt.savefig('analisis_IVANOVA_REPETICIONES.png')
    plt.close()
    print("Gráfico generado: analisis_IVANOVA_REPETICIONES.png")
    
    # ------------------------------------------------
    # GRÁFICO 2: Tasa de Reformulaciones [//]
    # ------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='diagnosis', y='ref_rate', data=df, showfliers=False)
    sns.stripplot(x='diagnosis', y='ref_rate', data=df, color='black', alpha=0.3, jitter=True)
    
    plt.title('IVANOVA - Tasa de Reformulaciones [//] (Normalizada)')
    plt.ylabel('Reformulaciones / Palabra')
    plt.xlabel('Diagnóstico')
    
    plt.savefig('analisis_IVANOVA_REFORMULACIONES.png')
    plt.close()
    print("Gráfico generado: analisis_IVANOVA_REFORMULACIONES.png")

    # ------------------------------------------------
    # ESTADÍSTICAS
    # ------------------------------------------------
    print("\n=== MEDIA DE TASAS POR GRUPO ===")
    print(df.groupby('diagnosis')[['rep_rate', 'ref_rate']].mean())

else:
    print("Error: No se encontraron datos.")