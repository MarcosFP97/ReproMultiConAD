import argparse
import sys
import math
import logging
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from ollama import chat

from transformers import AutoTokenizer

# Configurar logs de transformers para que no sean molestos
logging.getLogger("transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
args_slurm = parser.parse_args()
dataset = args_slurm.dataset.lower()

# Configuracion basica
INPUT_PATH = Path(f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets/train_{dataset}.jsonl")
OUTPUT_PATH = Path(f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/{dataset}_synthetic.jsonl")
MODEL_NAME = "mistral-small"

BASIC = False
SAVE = True

K_NEIGHBORS = 3
RANDOM_SEED = 42

# --- CARGA DEL TOKENIZADOR (Global) ---
print("Cargando tokenizador de Mistral...")
try:
    # Usamos el tokenizer de Mistral-7B-Instruct-v0.2 que comparte vocabulario con Mistral Small y es menos pesado
    # Si no tienes internet en el nodo de cómputo, asegúrate de tener esto en caché o descárgalo localmente
    TOKENIZER = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
except Exception as e:
    sys.exit(f"Error cargando tokenizer (asegúrate de tener internet o el modelo en caché): {e}")

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

    # Filtrado para el pipeline (lo que ya hacías)
    df_filtrado = df.dropna(
        subset=["Text_interviewer_participant", "Diagnosis", "Age", "MMSE"]
    ).copy()

    # (opcional) debug para ver cuánto se pierde
    print(f"[INFO] Filas raw: {len(df)} | Filas pipeline: {len(df_filtrado)}")

    return df_filtrado, conteo_diagnosticos_raw

def contar_tokens_reales(texto: str) -> int:
    """Cuenta tokens EXACTOS usando el tokenizer de Mistral."""
    if not texto:
        return 0
    # encode devuelve los IDs, su longitud es el número de tokens
    return len(TOKENIZER.encode(texto, add_special_tokens=False))

def analizar_estadisticas(df: pd.DataFrame, conteo_diagnosticos: dict) -> tuple[pd.DataFrame, int]:
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
    # (Tokens por vecino * 3) + 
    # 800 (Prompt sistema + instrucciones) + 
    # 1000 (Reserva para respuesta generada)
    estimated_need = (p95_tokens * K_NEIGHBORS) + 1800
    
    # Redondeo al bloque de 1024 superior
    recommended_ctx = int(math.ceil(estimated_need / 1024.0)) * 1024
    
    # Mínimo de seguridad
    if recommended_ctx < 4096:
        recommended_ctx = 4096

    print("\n--- ANÁLISIS DE TOKENS (EXACTO - MISTRAL) ---")
    print(f"Media tokens/transcripción: {avg_tokens:.0f}")
    print(f"Máximo tokens/transcripción: {max_tokens:.0f}")
    print(f"Percentil 95 tokens: {p95_tokens:.0f}")
    print(f"CONTEXTO RECOMENDADO OLLAMA: {recommended_ctx} tokens")
        
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
        c_min = df_filtrado[col].min()
        c_max = df_filtrado[col].max()

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

def validate_chat(texto: str | None) -> bool:
    """
    Validacion minima CHAT:
    - Debe empezar con *INV: o *PAR:
    - Debe tener al menos una linea *INV: y una *PAR:
    """
    if not texto:
        return False

    lines = [l.strip() for l in texto.strip().splitlines() if l.strip()]
    if not lines:
        return False

    if not re.match(r"^\*(INV|PAR):", lines[0]):
        return False
    
    # Verificar códigos de tiempo prohibidos
    if "\x15" in texto or re.search(r"\x15\d+_\d+\x15", texto):
        return False
    
    # Rechazar code / markdown / notebooks
    if "```" in texto:
        return False
    if re.search(r"\.ipynb\b|%matplotlib\b|\bimport\b|\bdef\b|\bclass\b|\btensorflow\b", texto):
        return False
    if re.search(r"^\+{3,}", texto, flags=re.MULTILINE):
        return False

    has_inv = any(l.startswith("*INV:") for l in lines)
    has_par = any(l.startswith("*PAR:") for l in lines)
    return has_inv and has_par

def generar_dialogo_paciente(target: dict, vecinos: pd.DataFrame, ctx_size: int) -> str | None:
    """Construye prompt BASIC/SMART, llama a Ollama con el contexto dinámico y devuelve el texto generado."""
    bloques = []
    
    # Calcular límite de caracteres seguro por vecino basado en el contexto disponible
    # Aproximación inversa: 1 token ~ 3-4 chars. Restamos prompt y dividimos por 3.
    # Es solo un truncado de emergencia extrema.
    chars_avail_per_neighbor = int(((ctx_size - 2000) / 3) * 3.5)
    
    for i, r in enumerate(vecinos.itertuples(index=False), 1):
        text_safe = str(r.Text_interviewer_participant)#[:chars_avail_per_neighbor]
        bloques.append(
            f"--- Neighbor {i} | Diagnosis: {r.Diagnosis} | Age: {r.Age} | MMSE: {r.MMSE} | Gender: {r.Gender} ---\n"
            f"{text_safe}"
        )
    selected_transcripts = "\n\n".join(bloques)

    if BASIC:
        prompt_user = f"""
            Based on these similar transcripts:

            {selected_transcripts}

            Generate a new Pitt CHAT transcript for:
            Diagnosis: {target['Diagnosis']}
            Age: {target['Age']}
            MMSE: {target['MMSE']}
            Gender: {target['Gender']}
            """.strip()
        messages = [{"role": "user", "content": prompt_user}]
    else:
        prompt_system = (
            "You generate synthetic Pitt Corpus dialogues in CHAT format. "
            "Output only turns from interviewer (*INV:) and participant (*PAR:)."
        )
        prompt_user = f"""
            Use these neighbors as style anchors:

            {selected_transcripts}

            TARGET:
            Diagnosis: {target['Diagnosis']}
            Age: {target['Age']}
            MMSE: {target['MMSE']}
            Gender: {target['Gender']}

            Rules:
            1. Include both speakers.
            2. MIMIC the broken speech patterns found in the neighbors (do not correct grammar).
            3. YOU MUST INCLUDE CHAT CODES if the neighbors have them. Examples :
            - Pauses: (.) or (..)
            - Repetitions: [/] (e.g., "the [/] the cookie")
            - Revisions: [//] (e.g., "girl [//] boy")
            - Fillers: &-uh, &-um
            4. Keep Cookie Theft context.
            5. CRITICAL: DO NOT include time alignment bullets (e.g., \x15123_456\x15). Since this is synthetic text without audio, time codes are invalid.
            """.strip()
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]

    # AQUI SE AÑADE LA TEMPERATURA
    options={
        'temperature': 1.0,  # Aumenta la creatividad/variedad
        'repeat_penalty': 1.1, # Opcional: Ayuda extra si sigue repitiendo bucles
        'num_ctx': ctx_size  # CONTEXTO EXACTO
    }
    
    # Imprimir prints de tokens
    system_txt = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_txt   = next((m["content"] for m in messages if m["role"] == "user"), "")

    sys_tok  = len(TOKENIZER.encode(system_txt, add_special_tokens=False))
    usr_tok  = len(TOKENIZER.encode(user_txt, add_special_tokens=False))
    both_txt = system_txt + "\n" + user_txt
    tot_tok  = len(TOKENIZER.encode(both_txt, add_special_tokens=False))

    print(f"[TOKENS] system={sys_tok} | user={usr_tok} | total={tot_tok} | ctx={ctx_size}")
    
    response = chat(model=MODEL_NAME, messages=messages, options=options)
    return response.message.content.strip()

def main() -> None:
    print("Cargando datos...")
    # conteo_raw es algo tipo: {'Dementia': 204, 'HC': 194, 'MCI': 34}
    df_real, conteo_raw = cargar_datos(INPUT_PATH)
    
    # Plot de distribuciones 
    # plot_distribuciones_por_diagnostico(df_real)
    # plot_relacion_age_mmse(df_real)
    # o solo el diagnóstico objetivo:
    # plot_distribuciones_objetivo(df_real, DIAG_OBJETIVO)

    # Calculamos stats 
    stats, recommended_ctx = analizar_estadisticas(df_real, conteo_raw)
    
    writer = None
    if SAVE:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        writer = OUTPUT_PATH.open("a", encoding="utf-8")
    
    for diag_objetivo, n_objetivo in conteo_raw.items():
        
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
                vecinos = buscar_vecinos_knn(target, df_real, k=K_NEIGHBORS)
                
                # --- VALIDACIÓN VECINOS ---
                if vecinos is None or vecinos.empty:
                    batch_bad += 1
                    print(f"Descartado (sin vecinos): {target}")
                    continue

                generated_text = generar_dialogo_paciente(target, vecinos, recommended_ctx)
                
                # --- VALIDACIÓN CHAT ---
                if not validate_chat(generated_text):
                    batch_bad += 1
                    print(f"Descartado (CHAT invalido): {target}")
                    continue

                # --- SI LLEGA AQUÍ, ES VÁLIDO ---
                neighbors_meta = vecinos[["Age", "MMSE", "Gender"]].to_dict("records")
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
    main()
