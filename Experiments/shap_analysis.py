import os
import argparse
import torch
import pandas as pd
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ==========================================
# CONFIGURACIÓN DINÁMICA
# ==========================================

parser = argparse.ArgumentParser(description="Análisis BERT SHAP - Truncado Robusto")
parser.add_argument("--marker", type=str, required=True)
parser.add_argument("--language", type=str, required=True)
parser.add_argument("--task", type=str, required=True, choices=["binary", "multiclass"])
args = parser.parse_args()

language = args.language
task = args.task
marker = args.marker

# Límite real del modelo
MAX_LEN = 256

MODEL_DIR = f"/mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_Models/bert_{language}_{task}_{marker}_len256"
TEST_PATH = f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/markers_collections/test_{language}_e5_markers_{marker}.jsonl"
OUTPUT_HTML = f"/mnt/beegfs/groups/irgroup/sara_tfg/results/shap_{language}_{task}_{marker}.html"

def main():
    print(f"========== INICIANDO SHAP: {task.upper()} | {language.upper()} | {marker.upper()} ==========")
    
    if not os.path.exists(MODEL_DIR):
        print(f"!!! ERROR: El modelo {MODEL_DIR} no existe.")
        return

    # 1. Cargar el dataset de test
    df = pd.read_json(TEST_PATH, lines=True)
    
    # 2. Cargar modelo y tokenizador
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    
    # Mapeo de etiquetas
    if task == "binary":
        id2label = {0: "Dementia", 1: "HC"}
    else:
        id2label = {0: "Dementia", 1: "HC", 2: "MCI"}
    
    model.config.id2label = id2label
    model.config.label2id = {v: k for k, v in id2label.items()}
    
    device = 0 if torch.cuda.is_available() else -1
    
    # IMPORTANTE: El pipeline DEBE tener return_all_scores=True para que SHAP no de ceros
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, 
                    truncation=True, max_length=MAX_LEN, top_k=None)

    # 3. Selección de Muestras (Ivanova + Marcadores)
    interes = ["Dementia", "MCI"] if task == "multiclass" else ["Dementia"]
    df_filtrado = df[df["Diagnosis"].isin(interes)].copy()

    if language == "spa" and 'Dataset' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['Dataset'] == 'Ivanova']

    df_filtrado['has_marker'] = df_filtrado['Text_interviewer_participant'].str.contains(r'\[', na=False)
    df_con_marcadores = df_filtrado[df_filtrado['has_marker'] == True]
    
    # Cogemos 10 ejemplos
    if len(df_con_marcadores) >= 10:
        df_samples = df_con_marcadores.sample(10, random_state=42)
    else:
        df_samples = df_filtrado.head(10)

    textos_raw = df_samples['Text_interviewer_participant'].tolist()
    real_labels = df_samples['Diagnosis'].tolist()

    # 4. TRUNCADO EXACTO USANDO EL TOKENIZADOR
    textos_prueba = []
    print(f"Truncando textos exactamente a {MAX_LEN} tokens...")

    for t in textos_raw:
        # 1. Tokenizamos sin truncar primero para obtener los IDs
        tokens_info = tokenizer(
            t, 
            truncation=True, 
            max_length=MAX_LEN, 
            add_special_tokens=True,
            return_offsets_mapping=True  # Esto nos dice qué caracteres corresponden a qué tokens
        )
        
        # 2. Buscamos hasta qué caracter llega el token número 256
        # El offset_mapping es una lista de tuplas (inicio, fin) de cada token
        offsets = tokens_info['offset_mapping']
        if len(offsets) > 0:
            ultimo_caracter = offsets[-1][1] # El final del último token permitido
            texto_truncado = t[:ultimo_caracter]
        else:
            texto_truncado = t
            
        textos_prueba.append(texto_truncado)

    print("Truncado completado. Marcadores preservados.")

    # 5. Predicciones para el encabezado
    preds_raw = pipe(textos_prueba)
    
    custom_names = []
    for i in range(len(textos_prueba)):
        # Buscamos la etiqueta con mayor score
        p = sorted(preds_raw[i], key=lambda x: x['score'], reverse=True)[0]
        label_pred = p['label']
        score = p['score'] * 100
        status = "✅" if label_pred == real_labels[i] else "❌"
        custom_names.append(f"{status} Caso {i+1} [Real: {real_labels[i]} | Pred: {label_pred} ({score:.1f}%)]")

    # 6. SHAP: Usar el explainer directamente con el pipe corregido
    explainer = shap.Explainer(pipe)
    print("Calculando SHAP (ten paciencia, son 10 casos)...")
    shap_values = explainer(textos_prueba)

    # 7. Guardar HTML
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    html_plot = shap.plots.text(shap_values, display=False)
    
    header_info = f"<div style='font-family: sans-serif; padding: 20px; background: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>"
    header_info += f"<h2>Análisis SHAP - {language.upper()} {task.upper()} ({marker})</h2>"
    for name in custom_names:
        color = "#27ae60" if "✅" in name else "#e74c3c"
        header_info += f"<p style='color: {color};'><b>{name}</b></p>"
    header_info += "</div>"

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(header_info + html_plot)
        
    print(f"========== ANÁLISIS COMPLETADO: {OUTPUT_HTML} ==========")

if __name__ == "__main__":
    main()