"""
Pipeline de generación de transcripciones sintéticas mediante la API de Gemini.

Tres modos controlados por --real-percentage:
  0    → zero-shot: genera synthetic100 sin usar ejemplos reales como contexto.
  1-99 → low-resource: complementa la fracción real con datos sintéticos hasta llegar a paridad.
  100  → full-real: no genera nada y termina.

La escritura es incremental (flush tras cada muestra) para no perder muestras válidas
si el job SLURM es interrumpido.
"""

import argparse
import sys
import math
import logging
import json
import os
import PIL.Image
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from prompt_system import (
    PROMPT_REGISTRY,
    get_prompt_spec,
    validate_generated_text,
    self_check_prompt_specs,
)
from gemini_backend import generar_dialogo_paciente_prompt, load_cookie_theft_image_inline


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
parser.add_argument(
    '--cookie_image',
    type=Path,
    default=Path("/mnt/beegfs/groups/irgroup/sara_tfg/assets/cookie-theft-picture.ppm"),
    help="Ruta local de la imagen Cookie Theft Picture",
)
args_slurm = parser.parse_args()
dataset = args_slurm.dataset.lower()
real_pct = args_slurm.real_percentage

load_dotenv()

if real_pct is None:
    parser.error("--real-percentage es obligatorio")

if real_pct < 0 or real_pct > 100:
    parser.error("--real-percentage debe estar en el rango 0..100")

synthetic_pct = 100 - real_pct
input_real_pct = 100 if real_pct == 0 else real_pct

# real=0   → zero-shot (se genera synthetic100 sin usar ejemplos reales como contexto).
# real<100 → low-resource (se complementa la fracción real con datos sintéticos).
# real=100 → full-real (no hay síntesis).
is_zero_shot = (real_pct == 0)

if input_real_pct == 100:
    INPUT_PATH = Path(f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets/train_{dataset}.jsonl")
else:
    INPUT_PATH = Path(f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/real/train_{dataset}_real{input_real_pct}.jsonl")
OUTPUT_PATH = Path(f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/synthetic/train_{dataset}_synthetic{synthetic_pct}_gemini.jsonl")
MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COOKIE_THEFT_IMAGE_PATH = args_slurm.cookie_image

BASIC = False
SAVE = True

K_NEIGHBORS = 3
RANDOM_SEED = 42

def contar_tokens_reales(texto: str) -> int:
    """Estimación aproximada: ~1 token cada 4 caracteres."""
    if not texto:
        return 0
    return max(1, math.ceil(len(texto) / 4.0))


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

    df["Age"] = pd.to_numeric(df.get("Age"), errors="coerce")
    df["MMSE"] = pd.to_numeric(df.get("MMSE"), errors="coerce")
    df["Gender"] = df.get("Gender").astype(str).str.strip().str.upper()

    # Conteo antes del filtrado, para que los objetivos de generación reflejen el dataset completo.
    conteo_diagnosticos_raw = (
        df.dropna(subset=["Diagnosis"])["Diagnosis"].value_counts().to_dict()
    )

    df_filtrado = df.dropna(
        subset=["Text_interviewer_participant", "Diagnosis", "Age", "MMSE"]
    ).copy()

    print(f"[INFO] Filas raw: {len(df)} | Filas pipeline: {len(df_filtrado)}")

    return df_filtrado, conteo_diagnosticos_raw

def analizar_estadisticas(df: pd.DataFrame, conteo_diagnosticos: dict) -> pd.DataFrame:
    """
    Imprime stats y devuelve el DataFrame con describe() para las variables numéricas.
    """
    stats = df.groupby("Diagnosis")[["Age", "MMSE"]].describe()
    df2 = df.copy()
    df2["Gender"] = df2["Gender"].astype("string").str.strip().str.upper()

    gender_pct = df2["Gender"].value_counts(normalize=True)
    female_pct = gender_pct.get("F", 0.0) * 100
    male_pct = gender_pct.get("M", 0.0) * 100

    counts = (
        df2.groupby(["Diagnosis", "Gender"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["F", "M"], fill_value=0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

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

    print("\nObjetivos de Generación (basado en input RAW):")
    for diag, count in conteo_diagnosticos.items():
        print(f"  -> Diagnóstico: {diag:<10} | Cantidad a generar: {count}")
        
    print("=" * 70 + "\n")
    return stats

def calcular_presupuesto_salida_gemini(df: pd.DataFrame, has_image_input: bool = True) -> int:
    """Calcula el presupuesto de salida de Gemini a partir del percentil 95 del dataset real."""
    print("Calculando tokens estimados para todo el dataset (puede tardar unos segundos)...")
    df["real_tokens"] = df["Text_interviewer_participant"].apply(contar_tokens_reales)

    avg_tokens = df["real_tokens"].mean()
    max_tokens = df["real_tokens"].max()
    p95_tokens = df["real_tokens"].quantile(0.95)

    output_multiplier = 3.0 if has_image_input else 1.5
    output_budget = max(int(math.ceil(p95_tokens * output_multiplier)), 4096)

    print("\n--- PRESUPUESTO DE SALIDA GEMINI ---")
    print(f"Media tokens/transcripción: {avg_tokens:.0f}")
    print(f"Máximo tokens/transcripción: {max_tokens:.0f}")
    print(f"Percentil 95 tokens: {p95_tokens:.0f}")
    print(f"Presupuesto dinámico de salida: {output_budget}")
    print("=" * 70 + "\n")

    return output_budget

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

    g = df2[df2["Diagnosis"] == diagnosis_objetivo]
    if g.empty:
        return []

    # Age y MMSE: bootstrap sobre valores reales del diagnóstico.
    age_values = g["Age"].dropna().to_numpy()
    age_min = float(stats.loc[diagnosis_objetivo, ("Age", "min")])
    age_max = float(stats.loc[diagnosis_objetivo, ("Age", "max")])

    mmse_values = g["MMSE"].dropna().to_numpy()

    # Género: samplear según proporción real del diagnóstico, no del dataset completo.
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

    norm_target = {}
    for col in ["Age", "MMSE"]:
        # Min-max calculado sobre el dataset real completo, no solo df_filtrado,
        # para que la escala de normalización sea consistente entre vecindarios.
        c_min = df_real[col].min()
        c_max = df_real[col].max()

        # Si toda la columna es constante, asignamos 0.5 para evitar división por cero.
        if c_max == c_min:
            df_filtrado[f"{col}_n"] = 0.5
            norm_target[col] = 0.5
        else:
            df_filtrado[f"{col}_n"] = (df_filtrado[col] - c_min) / (c_max - c_min)
            norm_target[col] = (target[col] - c_min) / (c_max - c_min)

    df_filtrado["dist"] = np.sqrt(
        (df_filtrado["Age_n"] - norm_target["Age"]) ** 2
        + (df_filtrado["MMSE_n"] - norm_target["MMSE"]) ** 2
    )

    return df_filtrado.sort_values("dist").head(k)

def generar_dialogo_paciente(dataset_name: str, target: dict, vecinos: pd.DataFrame, max_output_tokens: int, zero_shot: bool = False) -> str | None:
    """Delega la generación en el backend de Gemini."""
    return generar_dialogo_paciente_prompt(
        dataset_name=dataset_name,
        target=target,
        vecinos=vecinos,
        basic=BASIC,
        model_name=MODEL_NAME,
        api_key=GEMINI_API_KEY,
        cookie_theft_image_path=COOKIE_THEFT_IMAGE_PATH,
        max_output_tokens=max_output_tokens,
        token_counter=contar_tokens_reales,
        zero_shot=zero_shot,
    )

def main() -> None:
    print("Cargando datos...")
    if not GEMINI_API_KEY:
        sys.exit(
            "No se encontró la API key de Gemini. "
            "Define GEMINI_API_KEY en el entorno o en el archivo .env."
        )

    if not INPUT_PATH.exists():
        sys.exit(f"No se encontró el archivo de entrada JSONL: {INPUT_PATH}")

    df_real, conteo_raw = cargar_datos(INPUT_PATH)  # ej: {'Dementia': 204, 'HC': 194, 'MCI': 34}

    prompt_spec = get_prompt_spec(dataset)
    use_cookie_theft_image = prompt_spec.uses_cookie_theft_image
    if use_cookie_theft_image:
        load_cookie_theft_image_inline(COOKIE_THEFT_IMAGE_PATH)
    if is_zero_shot and prompt_spec.zero_shot_user_template is None:
        sys.exit(f"El dataset '{dataset}' no define plantillas zero-shot en prompt_system.py")
    print(f"[INFO] PromptSpec activo: dataset='{dataset}' -> spec='{next((k for k, v in PROMPT_REGISTRY.items() if v == prompt_spec), 'default')}'")
    print(f"[INFO] Modalidad Gemini: {'multimodal con imagen Cookie Theft' if use_cookie_theft_image else 'solo texto'}")
    print(f"[INFO] Modo de generación: {'zero-shot' if is_zero_shot else 'few-shot'}")
    
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
    stats = analizar_estadisticas(df_real, conteo_raw)
    output_budget = calcular_presupuesto_salida_gemini(
        df_real,
        has_image_input=use_cookie_theft_image,
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

        # Varía la seed en cada reintento para evitar generar exactamente los mismos targets.
        attempt_counter = 0

        while samples_needed > 0:

            print(f"Generando batch para {samples_needed} muestras faltantes...")

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

                    if vecinos is None or vecinos.empty:
                        batch_bad += 1
                        print(f"Descartado (sin vecinos): {target}")
                        continue

                generated_text = generar_dialogo_paciente(
                    dataset,
                    target,
                    vecinos,
                    output_budget,
                    zero_shot=is_zero_shot,
                )
                
                print(generated_text)

                if not validate_generated_text(generated_text, prompt_spec):
                    batch_bad += 1
                    print(f"Descartado (texto inválido para dataset='{dataset}'): {target}")
                    continue

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
                    writer.flush()  # Escritura inmediata: si el job muere, no se pierden muestras ya válidas.
                else:
                    print(json.dumps(sample, ensure_ascii=False)[:400] + "...")

                batch_ok += 1

            print(f"Batch finalizado. Guardadas: {batch_ok} | Descartadas: {batch_bad}")
            
            # Solo reintentamos las muestras que fallaron la validación.
            samples_needed = batch_bad
            attempt_counter += 1

            # Límite de seguridad: evita bucles infinitos si el modelo sigue fallando.
            if attempt_counter > 10:
                print(f"ABORTANDO {diag_objetivo}: Demasiados intentos fallidos ({attempt_counter}).")
                break

    if writer is not None:
        writer.close()
        
    print("Proceso finalizado.") 
    
if __name__ == "__main__":
    if args_slurm.self_check:
        self_check_prompt_specs()
    else:
        main()
