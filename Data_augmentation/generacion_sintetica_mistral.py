import argparse
import sys
import math
import logging
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prompt_system import (
    PROMPT_REGISTRY,
    get_prompt_spec,
    validate_generated_text,
    self_check_prompt_specs,
)

from ollama_backend import generar_dialogo_paciente_prompt, resolve_ollama_num_predict


# Configurar logs de transformers para que no sean molestos
logging.getLogger("transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument(
    '--real-percentage',
    dest='real_percentage',
    type=int,
    help="Porcentaje de datos reales usados como base (ej: 0, 20, 40, 60, 80, 100)",
)
parser.add_argument('--slice', dest='real_percentage', type=int, help=argparse.SUPPRESS)
parser.add_argument('--self_check', action='store_true')
parser.add_argument('--augmented', action='store_true', help=argparse.SUPPRESS)
args_slurm = parser.parse_args()
dataset = args_slurm.dataset.lower()
real_pct = args_slurm.real_percentage

if real_pct is None:
    parser.error("--real-percentage es obligatorio")

if real_pct < 0 or real_pct > 100:
    parser.error("--real-percentage debe estar en el rango 0..100")

synthetic_pct = 100 - real_pct
input_real_pct = 100 if real_pct == 0 else real_pct

# --- Semántica ---
# real=0   -> zero-shot: se genera synthetic100 sin usar ejemplos reales como contexto.
# real<100 -> low-resource: se genera el complemento synthetic(100-real).
# real=100 -> full-real: no hay síntesis.
is_zero_shot = (real_pct == 0)

# Configuracion basica
if input_real_pct == 100:
    INPUT_PATH = Path(f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets/train_{dataset}.jsonl")
else:
    INPUT_PATH = Path(f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/slices/train_{dataset}_real{input_real_pct}.jsonl")
OUTPUT_PATH = Path(f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/slices/train_{dataset}_synthetic{synthetic_pct}_mistral.jsonl")
MODEL_NAME = "mistral-small3.2"

BASIC = False
SAVE = True

K_NEIGHBORS = 3
RANDOM_SEED = 42

TOKENIZER: Any = None

# =========== TOKENIZER PARA TENER EN CUENTA LA VENTANA DE CONTEXTO DEL LLM =========== #
def get_tokenizer():
    """Carga diferida del tokenizador para evitar dependencias en el self-check."""
    global TOKENIZER
    if TOKENIZER is None:
        print("Cargando tokenizador de Mistral...")
        try:
            from transformers import AutoTokenizer
            # Usamos el tokenizer de Mistral-7B-Instruct-v0.2 que comparte vocabulario con Mistral Small y es menos pesado
            TOKENIZER = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        except Exception as e:
            sys.exit(f"Error cargando tokenizer (asegúrate de tener internet o el modelo en caché): {e}")
    return TOKENIZER

def contar_tokens_reales(texto: str) -> int:
    """Cuenta tokens EXACTOS usando el tokenizer de Mistral."""
    if not texto:
        return 0
    tokenizer = get_tokenizer()
    # encode devuelve los IDs, su longitud es el número de tokens
    return len(tokenizer.encode(texto, add_special_tokens=False))

# =========== CARGAR DATOS Y ANALIZAR ESTADÍSTICAS DEL DATASET =========== #
def cargar_datos(ruta: Path) -> tuple[pd.DataFrame, dict]:
    """Carga el JSONL y deja solo las columnas necesarias para el pipeline."""
    if not ruta.exists():
        sys.exit(f"No se encontró el archivo de entrada JSONL: {ruta}")

    try:
        df = pd.read_json(ruta, lines=True)
    except ValueError as e:
        sys.exit(f"JSONL mal formado o vacío en '{ruta}': {e}")
    except Exception as e:
        sys.exit(f"No se pudo leer el JSONL de entrada '{ruta}': {e}")

    if df.empty:
        sys.exit(f"El archivo de entrada está vacío: {ruta}")

    required_cols = {"Text_interviewer_participant", "Diagnosis", "Age", "MMSE", "Gender"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        sys.exit(f"Faltan columnas obligatorias en '{ruta}': {missing_cols}")

    # Normalización básica
    df["Age"] = pd.to_numeric(df.get("Age"), errors="coerce")
    df["MMSE"] = pd.to_numeric(df.get("MMSE"), errors="coerce")
    df["Gender"] = df.get("Gender").astype(str).str.strip().str.upper()

    # Conteo RAW: antes de eliminar por Age/MMSE
    conteo_diagnosticos_raw = (
        df.dropna(subset=["Diagnosis"])["Diagnosis"].value_counts().to_dict()
    )

    # Filtrado para el pipeline
    df_filtrado = df.dropna(
        subset=["Text_interviewer_participant", "Diagnosis", "Age", "MMSE"]
    ).copy()

    # debug para ver cuánto se pierde
    print(f"[INFO] Filas raw: {len(df)} | Filas pipeline: {len(df_filtrado)}")

    return df_filtrado, conteo_diagnosticos_raw

def analizar_estadisticas(df: pd.DataFrame, conteo_diagnosticos: dict) -> pd.DataFrame:
    """
    Imprime stats y devuelve el DataFrame con describe() para las variables numéricas.
    """
    stats = df.groupby("Diagnosis")[["Age", "MMSE"]].describe()
    df2 = df.copy()
    df2["Gender"] = df2["Gender"].astype("string").str.strip().str.upper()

    # --- Porcentaje GLOBAL ---
    gender_pct = df2["Gender"].value_counts(normalize=True)
    female_pct = gender_pct.get("F", 0.0) * 100
    male_pct = gender_pct.get("M", 0.0) * 100

    # --- Porcentaje POR DIAGNOSTICO ---
    counts = (
        df2.groupby(["Diagnosis", "Gender"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["F", "M"], fill_value=0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100  # cada fila suma 100

    print("\n" + "=" * 70)
    print("ESTADISTICA DESCRIPTIVA")
    print("=" * 70)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 170,
        "display.float_format", lambda x: f"{x:,.2f}",
    ):
        print(stats)

    print(f"\nGender en dataset -> F: {female_pct:.2f}% | M: {male_pct:.2f}%")

    print("\nGender por diagnostico (porcentaje):")
    for diag in pct.index:
        f = pct.loc[diag, "F"]
        m = pct.loc[diag, "M"]
        n = int(counts.loc[diag].sum())
        print(f"  - {diag}: F {f:.2f}% | M {m:.2f}%  (n={n})")

    # --- CÁLCULO DE N_SAMPLES DINÁMICO ---
    # Obtenemos cuántas filas hay por cada diagnóstico
    print("\nObjetivos de Generación (basado en input RAW):")
    for diag, count in conteo_diagnosticos.items():
        print(f"  -> Diagnóstico: {diag:<10} | Cantidad a generar: {count}")
        
    print("=" * 70 + "\n")
    return stats

def calcular_num_ctx_ollama(df: pd.DataFrame, output_tokens_budget: int, zero_shot: bool = False) -> int:
    """Calcula el num_ctx recomendado para Ollama a partir del percentil 95 del dataset."""
    print("Calculando tokens exactos para todo el dataset (puede tardar unos segundos)...")
    df["real_tokens"] = df["Text_interviewer_participant"].apply(contar_tokens_reales)

    avg_tokens = df["real_tokens"].mean()
    max_tokens = df["real_tokens"].max()
    p95_tokens = df["real_tokens"].quantile(0.95)

    prompt_budget = 800
    response_budget = int(output_tokens_budget)
    neighbor_budget = 0 if zero_shot else (p95_tokens * K_NEIGHBORS)
    estimated_need = neighbor_budget + prompt_budget + response_budget

    recommended_num_ctx = int(math.ceil(estimated_need / 1024.0)) * 1024

    min_ctx = 2048 if zero_shot else 4096
    if recommended_num_ctx < min_ctx:
        recommended_num_ctx = min_ctx

    print("\n--- PRESUPUESTO DE CONTEXTO OLLAMA ---")
    print(f"Media tokens/transcripción: {avg_tokens:.0f}")
    print(f"Máximo tokens/transcripción: {max_tokens:.0f}")
    print(f"Percentil 95 tokens: {p95_tokens:.0f}")
    print(f"Presupuesto vecinos en prompt: {neighbor_budget:.0f}")
    print(f"Presupuesto de salida aplicado (num_predict): {response_budget}")
    print(f"NUM_CTX RECOMENDADO PARA OLLAMA ({'zero-shot' if zero_shot else 'few-shot'}): {recommended_num_ctx} tokens")
    print("=" * 70 + "\n")

    return recommended_num_ctx

# =========== LÓGICA DEL PROGRAMA : GENERACIÓN DEL TARGET A GENERAR, BÚSQUEDA DE SUS VECINOS Y GENERACIÓN DEL DIÁLOGO SINTÉTICO =========== #
def generar_targets(df: pd.DataFrame, stats: pd.DataFrame, diagnosis_objetivo: str, n_samples: int, seed: int,) -> list[dict]:
    """
    Genera perfiles sintéticos SOLO para un diagnóstico:
    - Age: bootstrap (sample real del diagnóstico).
    - MMSE: bootstrap (sample real del diagnóstico).
    - Gender: por proporción real del diagnóstico.
    """
    rng = np.random.default_rng(seed)

    df2 = df.copy()
    df2["Gender"] = df2["Gender"].astype("string").str.strip().str.upper()

    # Nos quedamos solo con el diagnóstico objetivo
    g = df2[df2["Diagnosis"] == diagnosis_objetivo]
    if g.empty:
        return []

    # --- Age bootstrap ---
    age_values = g["Age"].dropna().to_numpy()
    age_min = float(stats.loc[diagnosis_objetivo, ("Age", "min")]) 
    age_max = float(stats.loc[diagnosis_objetivo, ("Age", "max")])

    # --- MMSE bootstrap ---
    mmse_values = g["MMSE"].dropna().to_numpy()

    # --- Gender por proporción del diagnóstico ---
    gender_probs = g["Gender"].value_counts(normalize=True)
    p_f = float(gender_probs.get("F", 0.0))
    p_m = float(gender_probs.get("M", 0.0))
    if (p_f + p_m) == 0:
        p_f, p_m = 0.5, 0.5
    else:
        s = p_f + p_m
        p_f, p_m = p_f / s, p_m / s

    targets = []
    for _ in range(n_samples):
        gen_age = float(rng.choice(age_values))
        gen_age = float(np.clip(gen_age, age_min, age_max))

        gen_mmse = float(rng.choice(mmse_values))
        gen_mmse = float(np.clip(gen_mmse, 0, 30))

        targets.append(
            {
                "Diagnosis": diagnosis_objetivo,
                "Age": int(round(gen_age)),
                "MMSE": int(round(gen_mmse)),
                "Gender": rng.choice(["F", "M"], p=[p_f, p_m]),
            }
        )

    return targets

def buscar_vecinos_knn(target: dict, df_real: pd.DataFrame, k: int = 3) -> pd.DataFrame | None:
    """
    Busca vecinos priorizando Diagnosis+Gender y, si no hay suficientes,
    completa con el resto del mismo Diagnosis (otros géneros).
    Incluye min-max seguro para evitar division por cero.
    """
    # 1) Preferimos vecinos con mismo diagnóstico y mismo género
    df_same = df_real[
        (df_real["Diagnosis"] == target["Diagnosis"]) & (df_real["Gender"] == target["Gender"])
    ].copy()

    # 2) Si no alcanza k, completamos con el mismo diagnóstico (sin importar género)
    if len(df_same) < k:
        df_diag = df_real[df_real["Diagnosis"] == target["Diagnosis"]].copy()
        df_other = df_diag[df_diag["Gender"] != target["Gender"]]
        df_filtrado = pd.concat([df_same, df_other], ignore_index=True)
    else:
        df_filtrado = df_same

    if df_filtrado.empty:
        return None

    # Normalización Min-Max para el cálculo de distancias
    norm_target = {}
    for col in ["Age", "MMSE"]:
        # Para la Normalización tenemos en cuenta los valores en el dataset real, para no restringir tantos valores
        c_min = df_real[col].min()
        c_max = df_real[col].max()

        # Si max=min, toda la columna es constante; fijamos 0.5 para todos.
        if c_max == c_min:
            df_filtrado[f"{col}_n"] = 0.5
            norm_target[col] = 0.5
        else:
            df_filtrado[f"{col}_n"] = (df_filtrado[col] - c_min) / (c_max - c_min)
            norm_target[col] = (target[col] - c_min) / (c_max - c_min)

    # Cálculo de Distancia Euclídea
    df_filtrado["dist"] = np.sqrt(
        (df_filtrado["Age_n"] - norm_target["Age"]) ** 2
        + (df_filtrado["MMSE_n"] - norm_target["MMSE"]) ** 2
    )

    return df_filtrado.sort_values("dist").head(k)

def generar_dialogo_paciente(dataset_name: str,target: dict,vecinos: pd.DataFrame,num_ctx: int,zero_shot: bool = False,) -> str | None:
    """Wrapper de compatibilidad: delega en el módulo de prompt system."""
    tokenizer = get_tokenizer()
    return generar_dialogo_paciente_prompt(
        dataset_name=dataset_name,
        target=target,
        vecinos=vecinos,
        num_ctx=num_ctx,
        basic=BASIC,
        model_name=MODEL_NAME,
        tokenizer=tokenizer,
        zero_shot=zero_shot,
    )

# =========== PROGRAMA PRINCIPAL =========== #
def main() -> None:
    print("Cargando datos...")
    # conteo_raw es algo tipo: {'Dementia': 204, 'HC': 194, 'MCI': 34}
    df_real, conteo_raw = cargar_datos(INPUT_PATH)
    
    # Coger el prompt necesario para el dataset
    prompt_spec = get_prompt_spec(dataset)
    
    # Calcular el output budget
    ollama_output_budget = resolve_ollama_num_predict(prompt_spec)
    
    if is_zero_shot and prompt_spec.zero_shot_user_template is None:
        sys.exit(f"El dataset '{dataset}' no define plantillas zero-shot en prompt_system.py")
        
    print(f"[INFO] PromptSpec activo: dataset='{dataset}' -> spec='{next((k for k, v in PROMPT_REGISTRY.items() if v == prompt_spec), 'default')}'")
    print(f"[INFO] Modo de generación: {'zero-shot' if is_zero_shot else 'few-shot'}")
    print(f"[INFO] Presupuesto de salida Ollama (num_predict): {ollama_output_budget}")
    
    if real_pct == 0:
        print("\n[ZERO-SHOT MODE] 0% real. No se usan datos reales como contexto de prompting.")
        for diag, count_real in conteo_raw.items():
            muestras_a_generar = count_real
            conteo_raw[diag] = muestras_a_generar
            print(
                f"{diag} tiene {count_real} reales de referencia. "
                f"Generando {muestras_a_generar} sintéticas para synthetic100."
            )
    elif real_pct < 100:
        print(f"\n[LOW-RESOURCE MODE] real{real_pct}. Se generará synthetic{synthetic_pct} por clase.")
        for diag, count_real in conteo_raw.items():
            muestras_a_generar = int(count_real * (synthetic_pct / real_pct))
            conteo_raw[diag] = muestras_a_generar
            print(
                f"real{real_pct}. {diag} tiene {count_real} reales. "
                f"Generando synthetic{synthetic_pct}: {muestras_a_generar} sintéticas."
            )
    else:
        print("\n[FULL-REAL MODE] real100. No se generan datos sintéticos.")
        return
            
    # Calculamos stats 
    stats = analizar_estadisticas(df_real, conteo_raw)
    recommended_num_ctx = calcular_num_ctx_ollama(
        df_real,
        output_tokens_budget=ollama_output_budget,
        zero_shot=is_zero_shot,
    )
    
    writer = None
    if SAVE:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        writer = OUTPUT_PATH.open("w", encoding="utf-8")
    
    for diag_objetivo, n_objetivo in conteo_raw.items():
        
        if n_objetivo <= 0:
            print(f"\n>>> SALTANDO DIAGNÓSTICO: {diag_objetivo} | Ya tiene el máximo de muestras.")
            continue
        
        print(f"\n>>> PROCESANDO DIAGNÓSTICO: {diag_objetivo} | META: {n_objetivo} muestras")
        samples_needed = n_objetivo 
        
        # Contador global para este diagnóstico para variar la seed si hace falta
        attempt_counter = 0
        
        while samples_needed > 0:

            print(f"Generando batch para {samples_needed} muestras faltantes...")
            
            # Pasamos samples_needed
            current_seed = RANDOM_SEED + attempt_counter + samples_needed
            targets = generar_targets(df_real, stats, diagnosis_objetivo=diag_objetivo, 
                                    n_samples=samples_needed, seed=current_seed)

            batch_ok = 0
            batch_bad = 0

            for target in targets:
                if is_zero_shot:
                    vecinos = pd.DataFrame()
                else:
                    vecinos = buscar_vecinos_knn(target, df_real, k=K_NEIGHBORS)

                    # --- VALIDACIÓN VECINOS ---
                    if vecinos is None or vecinos.empty:
                        batch_bad += 1
                        print(f"Descartado (sin vecinos): {target}")
                        continue

                generated_text = generar_dialogo_paciente(
                    dataset,
                    target,
                    vecinos,
                    recommended_num_ctx,
                    zero_shot=is_zero_shot,
                )

                # --- VALIDACIÓN DATASET-AWARE ---
                if not validate_generated_text(generated_text, prompt_spec):
                    batch_bad += 1
                    print(f"Descartado (texto inválido para dataset='{dataset}'): {target}")
                    continue

                # --- SI LLEGA AQUÍ, ES VÁLIDO ---
                neighbors_meta = (
                    []
                    if is_zero_shot
                    else vecinos[["Age", "MMSE", "Gender"]].to_dict("records")
                )
                sample = {
                    "Diagnosis": target["Diagnosis"],
                    "Age": target["Age"],
                    "MMSE": target["MMSE"],
                    "Gender": target["Gender"],
                    "Text_interviewer_participant": generated_text,
                    "Neighbors_meta": neighbors_meta
                }

                if SAVE:
                    writer.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    writer.flush() # Asegura que se guarde en disco inmediatamente
                else:
                    print(json.dumps(sample, ensure_ascii=False)[:400] + "...")

                batch_ok += 1

            print(f"Batch finalizado. Guardadas: {batch_ok} | Descartadas: {batch_bad}")
            
            # Actualizamos el while: solo pedimos las que fallaron
            samples_needed = batch_bad 
            attempt_counter += 1
            
            # SEGURIDAD: Si llevamos más de 10 intentos extra y no avanza, paramos este diagnóstico
            if attempt_counter > 10:
                print(f"ABORTANDO {diag_objetivo}: Demasiados intentos fallidos ({attempt_counter}).")
                break

    # --- CERRAMOS EL ARCHIVO FUERA DEL WHILE (Importante) ---
    if writer is not None:
        writer.close()
        
    print("Proceso finalizado.") 
    
if __name__ == "__main__":
    if args_slurm.self_check:
        self_check_prompt_specs()
    else:
        main()
