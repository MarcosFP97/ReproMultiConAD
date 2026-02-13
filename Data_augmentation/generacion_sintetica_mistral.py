import pandas as pd
import numpy as np
import json
import re
from ollama import chat

BASIC=False

# --- CONFIGURACIÓN DE RUTAS ---
INPUT_PATH = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets/train_pitt.jsonl"
OUTPUT_PATH = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/pitt_sintetico.jsonl"

# --- CARGAR DATOS QUE NOS INTERESAN DEL PITT
def cargar_datos(ruta):
    df = pd.read_json(ruta, lines=True)
    df['MMSE'] = pd.to_numeric(df['MMSE'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    return df.dropna(subset=['MMSE', 'Age', 'Text_interviewer_participant', 'Diagnosis'])

# --- BUSCAR VECINOS PARA LA GENERACIÓN DE TEXTO SINTÉTICO ---
def buscar_vecinos_knn(target, df_real, k=3):
    # Filtrado inicial por diagnóstico y género
    df_filtrado = df_real[
        (df_real['Diagnosis'] == target['Diagnosis']) & 
        (df_real['Gender'] == target['Gender'])
    ].copy()
    
    if len(df_filtrado) < k:
        df_filtrado = df_real[df_real['Diagnosis'] == target['Diagnosis']].copy()

    # Normalización Min-Max para el cálculo de distancias
    for col in ['Age', 'MMSE']:
        c_min, c_max = df_filtrado[col].min(), df_filtrado[col].max()
        df_filtrado[f'{col}_n'] = (df_filtrado[col] - c_min) / (c_max - c_min)
        target[f'{col}_n'] = (target[col] - c_min) / (c_max - c_min)

    # Cálculo de Distancia Euclídea
    df_filtrado['dist'] = np.sqrt(
        (df_filtrado['Age_n'] - target['Age_n'])**2 + 
        (df_filtrado['MMSE_n'] - target['MMSE_n'])**2
    )
    
    return df_filtrado.sort_values('dist').head(k)

def generar_chat_pitt_dialogo(target, vecinos):
    # 1. Construimos el contexto (Paper Method)
    selected_transcripts = ""
    for i, (_, r) in enumerate(vecinos.iterrows(), 1):
        selected_transcripts += f"--- Neighbor {i} (Diagnosis: {r.Diagnosis}, MMSE: {r.MMSE}) ---\n{r.Text_interviewer_participant}\n\n"
    
    # Inicializamos la lista de mensajes para Ollama
    messages = []

    if BASIC:
        # --- PROMPT MINIMALISTA (PAPER) ---
        prompt_usuario = f"""
        Based on a list of transcripts collected from other subjects of similar characteristics:
        
        {selected_transcripts}
        
        Generate a new speech transcript for a patient with these characteristics:
        Diagnosis: {target['Diagnosis']}, Age: {target['Age']}, MMSE: {target['MMSE']}.
        """
        # En el modo BASIC, solo enviamos el mensaje del usuario
        messages.append({'role': 'user', 'content': prompt_usuario})
        
    else:
        # --- PROMPT INTELIGENTE (VERSIÓN MÁS "SMART") ---
        prompt_sistema = (
            "You are an expert data generator for the Pitt Corpus (DementiaBank). "
            "Your task is to generate a synthetic transcript that strictly follows the CHAT format codes. "
            "You MUST output a dialogue between *INV: and *PAR:."
        )
        
        prompt_usuario = f"""
        Based on the following transcripts collected from subjects with similar characteristics (Diagnosis: {target['Diagnosis']}, Age: {target['Age']}, MMSE: {target['MMSE']}):

        {selected_transcripts}

        TASK: Generate a NEW speech transcript for a patient with this profile.

        STRICT RULES:
        1. Start directly with the first speaker tag (e.g., *INV:).
        2. MIMIC the broken speech patterns found in the neighbors (do not correct grammar).
        3. YOU MUST INCLUDE CHAT CODES if the neighbors have them:
        - Pauses: (.) or (..)
        - Repetitions: [/] (e.g., "the [/] the cookie")
        - Revisions: [//] (e.g., "girl [//] boy")
        - Fillers: &-uh, &-um
        4. Maintain the 'Cookie Theft' context (jar, stool, water, sink).
        """
        # En el modo SMART, añadimos sistema y usuario para dar más contexto
        messages.append({'role': 'system', 'content': prompt_sistema})
        messages.append({'role': 'user', 'content': prompt_usuario})

    try:
        # Ahora pasamos la lista 'messages' ya construida
        response = chat(model='mistral-small', messages=messages)
        return response.message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

'''
# --- EJECUCIÓN (MODO PRINT PARA VERIFICAR) ---
print(f"--- INICIANDO PROCESO MEJORADO ---")
df_real = cargar_datos(INPUT_PATH)

perfiles_nuevos = [
    {'Age': 70, 'MMSE': 30, 'Diagnosis': 'HC', 'Gender': 'female'},
    {'Age': 75, 'MMSE': 22, 'Diagnosis': 'MCI', 'Gender': 'male'},
    {'Age': 82, 'MMSE': 8, 'Diagnosis': 'Dementia', 'Gender': 'female'}
]

for i, perfil in enumerate(perfiles_nuevos, 1):
    print(f"\n{'='*60}")
    print(f"TEST {i}: {perfil['Diagnosis']} | MMSE: {perfil['MMSE']} | Age: {perfil['Age']} | Gender: {perfil['Gender']}")
    print(f"{'='*60}")
    
    vecinos = buscar_vecinos_knn(perfil, df_real)
    print(f"Vecinos usados: {[v for v in vecinos['MMSE']]}") # Ver qué MMSE tienen los vecinos

    texto_sintetico = generar_chat_pitt_dialogo(perfil, vecinos)
    
    if texto_sintetico:
        print(f"\n--- GENERACIÓN ---")
        print(texto_sintetico)
        print(f"------------------")
'''

# --- 1. NUEVA FUNCIÓN: ANÁLISIS ESTADÍSTICO ---
def analizar_estadisticas(df):
    """
    Calcula estadísticas descriptivas para Edad y MMSE agrupadas por Diagnóstico.
    Devuelve un DataFrame con índices jerárquicos (Diagnosis -> Metrica).
    """
    # Agrupamos por diagnóstico y pedimos 'describe' de las columnas de interés
    stats = df.groupby('Diagnosis')[['Age', 'MMSE']].describe()
    
    print("\n" + "="*60)
    print("ESTUDIO ESTADÍSTICO DEL DATASET PITT (DementiaBank)")
    print("="*60)
    
    # Formateamos para que se lea bien en consola
    pd.options.display.float_format = '{:,.2f}'.format
    print(stats)
    print("="*60 + "\n")
    
    return stats

# --- 2. GENERADOR DE PERFILES BASADO EN ESTADÍSTICA ---
def generar_targets_estadisticos(stats, diagnosis, n_samples=10):
    """
    Genera perfiles sintéticos muestreando de una distribución Normal 
    basada en la media y desviación estándar del grupo real.
    """
    if diagnosis not in stats.index:
        print(f"Error: El diagnóstico '{diagnosis}' no existe en el dataset.")
        return []

    # Extraemos las métricas del grupo
    age_mean = stats.loc[diagnosis, ('Age', 'mean')]
    age_std  = stats.loc[diagnosis, ('Age', 'std')]
    age_min  = stats.loc[diagnosis, ('Age', 'min')]
    age_max  = stats.loc[diagnosis, ('Age', 'max')]

    mmse_mean = stats.loc[diagnosis, ('MMSE', 'mean')]
    mmse_std  = stats.loc[diagnosis, ('MMSE', 'std')]
    mmse_min  = stats.loc[diagnosis, ('MMSE', 'min')]
    mmse_max  = stats.loc[diagnosis, ('MMSE', 'max')]

    perfiles = []
    
    for _ in range(n_samples):
        # Generamos Edad aleatoria (distribución normal)
        # np.random.normal(media, desviacion)
        gen_age = np.random.normal(age_mean, age_std)
        # Cliping: Aseguramos que no salga un valor imposible (ej. 150 años)
        # Usamos los min/max reales del dataset como límites lógicos
        gen_age = max(age_min, min(age_max, gen_age))

        # Generamos MMSE aleatorio (distribución normal)
        gen_mmse = np.random.normal(mmse_mean, mmse_std)
        # Cliping: MMSE debe estar entre 0 y 30
        # También respetamos los límites reales vistos en los datos
        gen_mmse = max(0, min(30, gen_mmse)) 

        # Crear perfil
        perfiles.append({
            'Age': int(round(gen_age)),
            'MMSE': int(round(gen_mmse)),
            'Diagnosis': diagnosis,
            # Asignamos género aleatorio 50/50 o basado en proporción real si quisieras hilar más fino
            'Gender': np.random.choice(['female', 'male']) 
        })
        
    return perfiles

df_real = cargar_datos(INPUT_PATH)

# 2. ESTUDIO ESTADÍSTICO PREVIO
# Esto imprimirá la tabla con count, mean, std, min, 25%, 50%, 75%, max
stats_df = analizar_estadisticas(df_real)

print(f"\n--- PROCESO FINALIZADO ---")
