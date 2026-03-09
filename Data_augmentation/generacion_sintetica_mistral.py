import argparse
import sys
import math
import logging
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from prompt_system import (
    PROMPT_REGISTRY,
    get_prompt_spec,
    validate_generated_text,
    self_check_prompt_specs,
    generar_dialogo_paciente_prompt
)


# Configurar logs de transformers para que no sean molestos
logging.getLogger("transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--slice', type=int, required=True, help="Porcentaje de datos reales usados (ej: 20, 40, 60, 80)")
parser.add_argument('--self_check', action='store_true')
parser.add_argument('--augmented', action='store_true', help="Genera solo las muestras faltantes para igualar a la clase mayoritaria")
args_slurm = parser.parse_args()
dataset = args_slurm.dataset.lower()
slice_pct = args_slurm.slice

if slice_pct <= 0 or slice_pct > 100:
    parser.error("--slice debe estar en el rango 1..100")

# --- NUEVO: Lógica Zero-Shot Automática ---
is_zero_shot = (slice_pct == 100)

# Configuracion basica
INPUT_PATH = Path(f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/slices/train_{dataset}_{slice_pct}.jsonl")
OUTPUT_PATH = Path(f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/slices/train_{dataset}_{slice_pct}_synthetic.jsonl")
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
            # Si no tienes internet en el nodo de cómputo, asegúrate de tener esto en caché o descárgalo localmente
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
    df = pd.read_json(ruta, lines=True)

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

def analizar_estadisticas(df: pd.DataFrame,conteo_diagnosticos: dict,zero_shot: bool = False,) -> tuple[pd.DataFrame, int]:
    """
    Imprime stats y devuelve:
    1. DataFrame con describe() (stats numéricas).
    2. Dict con conteo por diagnóstico (ej: {'Dementia': 204, 'HC': 194}).
    3. Int donde se ha calculado el contexto exacto necesario.
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
        
    # --- ANÁLISIS DE TOKENS EXACTO ---    
    print("Calculando tokens exactos para todo el dataset (puede tardar unos segundos)...")
    df["real_tokens"] = df["Text_interviewer_participant"].apply(contar_tokens_reales)
    
    avg_tokens = df["real_tokens"].mean()
    max_tokens = df["real_tokens"].max()
    p95_tokens = df["real_tokens"].quantile(0.95) 
    
    # Cálculo dinámico del contexto:
    # - Few-shot: reserva para K vecinos en prompt.
    # - Zero-shot: no reserva tokens para vecinos.
    # - En ambos casos mantenemos un colchón fijo para instrucciones + respuesta.
    prompt_budget = 800
    response_budget = 1000
    neighbor_budget = 0 if zero_shot else (p95_tokens * K_NEIGHBORS)
    estimated_need = neighbor_budget + prompt_budget + response_budget
    
    # Redondeo al bloque de 1024 superior
    recommended_ctx = int(math.ceil(estimated_need / 1024.0)) * 1024
    
    # Mínimo de seguridad
    min_ctx = 2048 if zero_shot else 4096
    if recommended_ctx < min_ctx:
        recommended_ctx = min_ctx

    print("\n--- ANÁLISIS DE TOKENS (EXACTO - MISTRAL) ---")
    print(f"Media tokens/transcripción: {avg_tokens:.0f}")
    print(f"Máximo tokens/transcripción: {max_tokens:.0f}")
    print(f"Percentil 95 tokens: {p95_tokens:.0f}")
    print(f"Presupuesto vecinos en prompt: {neighbor_budget:.0f}")
    print(f"Reserva fija prompt+respuesta: {prompt_budget + response_budget}")
    print(f"CONTEXTO RECOMENDADO OLLAMA ({'zero-shot' if zero_shot else 'few-shot'}): {recommended_ctx} tokens")
        
    print("=" * 70 + "\n")
    return stats, recommended_ctx

def plot_distribuciones_por_diagnostico(df: pd.DataFrame) -> None:
    """
    Dibuja histogramas de Age y MMSE para cada diagnóstico del dataset.
    - Age: bins "normales"
    - MMSE: bins por entero (0-30) para ver bien la discreción y el ceiling effect
    """
    diagnosticos = sorted(df["Diagnosis"].dropna().unique())

    for diag in diagnosticos:
        g = df[df["Diagnosis"] == diag]

        # --- Age ---
        plt.figure()
        plt.hist(g["Age"].dropna(), bins=15)
        plt.title(f"Distribución de Age - {diag} (n={len(g)})")
        plt.xlabel("Age")
        plt.ylabel("Frecuencia")
        plt.savefig(Path.cwd() / f"hist_age_{diag}.png", dpi=200)

        # --- MMSE ---
        plt.figure()
        bins = np.arange(-0.5, 30.5 + 1, 1)  # barras centradas en enteros
        plt.hist(g["MMSE"].dropna(), bins=bins)
        plt.title(f"Distribución de MMSE - {diag} (n={len(g)})")
        plt.xlabel("MMSE")
        plt.ylabel("Frecuencia")
        plt.savefig(Path.cwd() / f"hist_mmse_{diag}.png", dpi=200)

def plot_distribuciones_objetivo(df: pd.DataFrame, diagnosis_objetivo: str) -> None:
    """
    Dibuja histogramas solo para el diagnóstico objetivo (más rápido si no quieres todo).
    """
    g = df[df["Diagnosis"] == diagnosis_objetivo]
    if g.empty:
        print(f"[WARN] No hay datos para Diagnosis='{diagnosis_objetivo}'. No se pueden plotear distribuciones.")
        return

    plt.figure()
    plt.hist(g["Age"].dropna(), bins=15)
    plt.title(f"Distribución de Age - {diagnosis_objetivo} (n={len(g)})")
    plt.xlabel("Age")
    plt.ylabel("Frecuencia")
    plt.savefig(Path.cwd() / f"dist_age_{diagnosis_objetivo}.png", dpi=200)

    plt.figure()
    bins = np.arange(-0.5, 30.5 + 1, 1)
    plt.hist(g["MMSE"].dropna(), bins=bins)
    plt.title(f"Distribución de MMSE - {diagnosis_objetivo} (n={len(g)})")
    plt.xlabel("MMSE")
    plt.ylabel("Frecuencia")
    plt.savefig(Path.cwd() / f"dist_mmse_{diagnosis_objetivo}.png", dpi=200)

def plot_relacion_age_mmse(df: pd.DataFrame, out_dir: Path | None = None) -> None:
    """
    Plots para ver relación Age vs MMSE:
    - Global: scatter + tendencia lineal
    - Por diagnóstico: scatter (alpha) + tendencia lineal
    Guarda PNGs si out_dir se especifica (si no, usa cwd).
    """
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Asegurar numéricos
    dfp = df.copy()
    dfp["Age"] = pd.to_numeric(dfp["Age"], errors="coerce")
    dfp["MMSE"] = pd.to_numeric(dfp["MMSE"], errors="coerce")
    dfp = dfp.dropna(subset=["Age", "MMSE", "Diagnosis"])

    # --- Plot GLOBAL ---
    plt.figure()
    x = dfp["Age"].to_numpy()
    y = dfp["MMSE"].to_numpy()

    plt.scatter(x, y, alpha=0.35, s=18)
    # tendencia lineal simple
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        plt.plot(xs, m * xs + b, linewidth=2)

    plt.title(f"Relación Age vs MMSE - GLOBAL (n={len(dfp)})")
    plt.xlabel("Age")
    plt.ylabel("MMSE")
    plt.ylim(-0.5, 30.5)
    plt.savefig(out_dir / "rel_age_mmse_GLOBAL.png", dpi=200, bbox_inches="tight")

    # --- Plot POR DIAGNÓSTICO ---
    diagnosticos = sorted(dfp["Diagnosis"].unique())
    for diag in diagnosticos:
        g = dfp[dfp["Diagnosis"] == diag]
        if g.empty:
            continue

        plt.figure()
        x = g["Age"].to_numpy()
        y = g["MMSE"].to_numpy()

        plt.scatter(x, y, alpha=0.4, s=22)

        # tendencia lineal (opcional)
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 200)
            plt.plot(xs, m * xs + b, linewidth=2)

        plt.title(f"Relación Age vs MMSE - {diag} (n={len(g)})")
        plt.xlabel("Age")
        plt.ylabel("MMSE")
        plt.ylim(-0.5, 30.5)
        plt.savefig(out_dir / f"rel_age_mmse_{diag}.png", dpi=200, bbox_inches="tight")

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

def generar_dialogo_paciente(dataset_name: str,target: dict,vecinos: pd.DataFrame,ctx_size: int,zero_shot: bool = False,) -> str | None:
    """Wrapper de compatibilidad: delega en el módulo de prompt system."""
    tokenizer = get_tokenizer()
    return generar_dialogo_paciente_prompt(
        dataset_name=dataset_name,
        target=target,
        vecinos=vecinos,
        ctx_size=ctx_size,
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
    prompt_spec = get_prompt_spec(dataset)
    if is_zero_shot and prompt_spec.zero_shot_user_template is None:
        sys.exit(f"El dataset '{dataset}' no define plantillas zero-shot en prompt_system.py")
    print(f"[INFO] PromptSpec activo: dataset='{dataset}' -> spec='{next((k for k, v in PROMPT_REGISTRY.items() if v == prompt_spec), 'default')}'")
    print(f"[INFO] Modo de generación: {'zero-shot' if is_zero_shot else 'few-shot'}")
    
    if args_slurm.slice < 100:
        y_pct = 100 - args_slurm.slice
        print(f"\n[LOW-RESOURCE MODE] Slice del {args_slurm.slice}%. Se generará el {y_pct}% faltante por clase.")
        for diag, count_real in conteo_raw.items():
            muestras_a_generar = int(count_real * (y_pct / args_slurm.slice))
            conteo_raw[diag] = muestras_a_generar
            print(
                f"Slice del {args_slurm.slice}%. {diag} tiene {count_real} reales. "
                f"Generando el {y_pct}% restante: {muestras_a_generar} sintéticas."
            )
    else:
        # Generar el mismo número de sintéticos si el slice es 100
        print("\n[FULL SYNTHETIC MODE] Slice del 100%. Se generará una cantidad idéntica de datos sintéticos.")
        for diag, count_real in conteo_raw.items():
            muestras_a_generar = count_real  # 1 sintético por cada 1 real
            conteo_raw[diag] = muestras_a_generar
            print(
                f"Slice del 100%. {diag} tiene {count_real} reales. "
                f"Generando {muestras_a_generar} sintéticas equivalentes."
            )
    
    # Plot de distribuciones 
    # plot_distribuciones_por_diagnostico(df_real)
    # plot_relacion_age_mmse(df_real)
    # o solo el diagnóstico objetivo:
    # plot_distribuciones_objetivo(df_real, DIAG_OBJETIVO)

    # Calculamos stats 
    stats, recommended_ctx = analizar_estadisticas(df_real,conteo_raw,zero_shot=is_zero_shot,)
    
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
                    recommended_ctx,
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
