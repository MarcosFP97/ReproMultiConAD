"""
Synthetic transcription generation pipeline using Mistral (local Ollama).

It follows the same three modes as the Gemini backend (zero-shot / low-resource / full-real),
but uses a local model via Ollama to avoid API costs and operate in environments
without internet access from the compute node. The context budget (num_ctx)
is computed dynamically from the 95th percentile of the real dataset.
"""

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

from ollama_backend import generate_dialog_pacient_prompt, resolve_ollama_num_predict


logging.getLogger("transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument(
    '--real-percentage',
    dest='real_percentage',
    type=int,
    help="Percentage of real data used as the base (e.g., 0, 20, 40, 60, 80, 100)",
)
parser.add_argument('--slice', dest='real_percentage', type=int, help=argparse.SUPPRESS)
parser.add_argument('--self_check', action='store_true')
parser.add_argument('--augmented', action='store_true', help=argparse.SUPPRESS)
args_slurm = parser.parse_args()
dataset = args_slurm.dataset.lower()
real_pct = args_slurm.real_percentage

if real_pct is None:
    parser.error("--real-percentage is required")

if real_pct < 0 or real_pct > 100:
    parser.error("--real-percentage must be in the range 0..100")

synthetic_pct = 100 - real_pct
input_real_pct = 100 if real_pct == 0 else real_pct

# real=0   → zero-shot (generates synthetic100 without using real examples as context).
# real<100 → low-resource (complements the real fraction with synthetic data).
# real=100 → full-real (no synthesis occurs).
is_zero_shot = (real_pct == 0)

if input_real_pct == 100:
    INPUT_PATH = Path(f"./jsonl/individual_sets/train_{dataset}.jsonl")
else:
    INPUT_PATH = Path(f"./jsonl/synthetic_data/real/train_{dataset}_real{input_real_pct}.jsonl")
OUTPUT_PATH = Path(f"./jsonl/synthetic_data/synthetic/train_{dataset}_synthetic{synthetic_pct}_mistral.jsonl")
MODEL_NAME = "mistral-small3.2"

BASIC = False
SAVE = True

K_NEIGHBORS = 3
RANDOM_SEED = 42

TOKENIZER: Any = None


def get_tokenizer():
    """Lazily loads the tokenizer to avoid blocking the self-check."""
    global TOKENIZER
    if TOKENIZER is None:
        print("Loading Mistral tokenizer...")
        try:
            from transformers import AutoTokenizer
            # Mistral-7B-Instruct-v0.2 shares vocabulary with Mistral Small and is lighter in cache.
            TOKENIZER = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        except Exception as e:
            sys.exit(f"Error loading tokenizer (make sure you have internet or the model cached): {e}")
    return TOKENIZER


def contar_tokens_reales(texto: str) -> int:
    """Counts exact tokens using the Mistral tokenizer."""
    if not texto:
        return 0
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(texto, add_special_tokens=False))


def cargar_datos(path: Path) -> tuple[pd.DataFrame, dict]:
    """Loads the JSONL and keeps only the columns needed by the pipeline."""
    if not path.exists():
        sys.exit(f"Input JSONL file not found: {path}")

    try:
        df = pd.read_json(path, lines=True)
    except ValueError as e:
        sys.exit(f"Malformed or empty JSONL in '{path}': {e}")
    except Exception as e:
        sys.exit(f"Could not read the input JSONL '{path}': {e}")

    if df.empty:
        sys.exit(f"The input file is empty: {path}")

    required_cols = {"Text_interviewer_participant", "Diagnosis", "Age", "MMSE", "Gender"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        sys.exit(f"Missing required columns in '{path}': {missing_cols}")

    df["Age"] = pd.to_numeric(df.get("Age"), errors="coerce")
    df["MMSE"] = pd.to_numeric(df.get("MMSE"), errors="coerce")
    df["Gender"] = df.get("Gender").astype(str).str.strip().str.upper()

    # Count before filtering so the generation targets reflect the full dataset.
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
    Prints stats and returns a DataFrame with describe() for the numeric variables.
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
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 170,
        "display.float_format", lambda x: f"{x:,.2f}",
    ):
        print(stats)

    print(f"\nGender in dataset -> F: {female_pct:.2f}% | M: {male_pct:.2f}%")

    print("\nGender by diagnosis (percentage):")
    for diag in pct.index:
        f = pct.loc[diag, "F"]
        m = pct.loc[diag, "M"]
        n = int(counts.loc[diag].sum())
        print(f"  - {diag}: F {f:.2f}% | M {m:.2f}%  (n={n})")

    print("\nGeneration targets (based on raw input):")
    for diag, count in conteo_diagnosticos.items():
        print(f"  -> Diagnosis: {diag:<10} | Quantity to generate: {count}")
        
    print("=" * 70 + "\n")
    return stats

def calcular_num_ctx_ollama(df: pd.DataFrame, output_tokens_budget: int, zero_shot: bool = False) -> int:
    """Calculates the recommended num_ctx for Ollama from the dataset's 95th percentile."""
    print("Calculating exact tokens for the full dataset (may take a few seconds)...")
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

    print("\n--- OLLAMA CONTEXT BUDGET ---")
    print(f"Average tokens/transcript: {avg_tokens:.0f}")
    print(f"Maximum tokens/transcript: {max_tokens:.0f}")
    print(f"95th percentile tokens: {p95_tokens:.0f}")
    print(f"Neighbor budget in prompt: {neighbor_budget:.0f}")
    print(f"Output budget applied (num_predict): {response_budget}")
    print(f"RECOMMENDED NUM_CTX FOR OLLAMA ({'zero-shot' if zero_shot else 'few-shot'}): {recommended_num_ctx} tokens")
    print("=" * 70 + "\n")

    return recommended_num_ctx

def generar_targets(df: pd.DataFrame, stats: pd.DataFrame, diagnosis_objetivo: str, n_samples: int, seed: int,) -> list[dict]:
    """
    Generates synthetic profiles for a single diagnosis only:
    - Age: bootstrap (sample from the diagnosis's real values).
    - MMSE: bootstrap (sample from the diagnosis's real values).
    - Gender: according to the diagnosis' real proportion.
    """
    rng = np.random.default_rng(seed)

    df2 = df.copy()
    df2["Gender"] = df2["Gender"].astype("string").str.strip().str.upper()

    g = df2[df2["Diagnosis"] == diagnosis_objetivo]
    if g.empty:
        return []

    # Age and MMSE: bootstrap from the diagnosis's real values.
    age_values = g["Age"].dropna().to_numpy()
    age_min = float(stats.loc[diagnosis_objetivo, ("Age", "min")])
    age_max = float(stats.loc[diagnosis_objetivo, ("Age", "max")])

    mmse_values = g["MMSE"].dropna().to_numpy()

    # Gender: sample according to the diagnosis's real proportion, not the full dataset.
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
    Finds neighbors by prioritizing Diagnosis+Gender, and if there are not enough,
    completes them with the rest of the same Diagnosis (other genders).
    Includes safe min-max handling to avoid division by zero.
    """
    # 1) Prefer neighbors with the same diagnosis and gender
    df_same = df_real[
        (df_real["Diagnosis"] == target["Diagnosis"]) & (df_real["Gender"] == target["Gender"])
    ].copy()

    # 2) If k is not reached, complete with the same diagnosis regardless of gender
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
        # Min-max computed over the full real dataset, not just df_filtrado,
        # so the normalization scale remains consistent across neighborhoods.
        c_min = df_real[col].min()
        c_max = df_real[col].max()

        # If the whole column is constant, assign 0.5 to avoid division by zero.
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

def generar_dialogo_paciente(dataset_name: str, target: dict, vecinos: pd.DataFrame, num_ctx: int, zero_shot: bool = False) -> str | None:
    """Delegates generation to the Ollama backend."""
    tokenizer = get_tokenizer()
    return generate_dialog_pacient_prompt(
        dataset_name=dataset_name,
        target=target,
        vecinos=vecinos,
        num_ctx=num_ctx,
        basic=BASIC,
        model_name=MODEL_NAME,
        tokenizer=tokenizer,
        zero_shot=zero_shot,
    )

def main() -> None:
    print("Loading data...")
    df_real, conteo_raw = cargar_datos(INPUT_PATH)  # example: {'Dementia': 204, 'HC': 194, 'MCI': 34}

    prompt_spec = get_prompt_spec(dataset)
    ollama_output_budget = resolve_ollama_num_predict(prompt_spec)
    
    if is_zero_shot and prompt_spec.zero_shot_user_template is None:
        sys.exit(f"The dataset '{dataset}' does not define zero-shot templates in prompt_system.py")
        
    print(f"[INFO] Active PromptSpec: dataset='{dataset}' -> spec='{next((k for k, v in PROMPT_REGISTRY.items() if v == prompt_spec), 'default')}'")
    print(f"[INFO] Generation mode: {'zero-shot' if is_zero_shot else 'few-shot'}")
    print(f"[INFO] Ollama output budget (num_predict): {ollama_output_budget}")
    
    if real_pct == 0:
        print("\n[ZERO-SHOT MODE] 0% real. Real data is not used as prompting context.")
        for diag, count_real in conteo_raw.items():
            muestras_a_generar = count_real
            conteo_raw[diag] = muestras_a_generar
            print(
                f"{diag} has {count_real} real reference samples. "
                f"Generating {muestras_a_generar} synthetic samples for synthetic100."
            )
    elif real_pct < 100:
        print(f"\n[LOW-RESOURCE MODE] real{real_pct}. synthetic{synthetic_pct} will be generated per class.")
        for diag, count_real in conteo_raw.items():
            muestras_a_generar = int(count_real * (synthetic_pct / real_pct))
            conteo_raw[diag] = muestras_a_generar
            print(
                f"real{real_pct}. {diag} has {count_real} real samples. "
                f"Generating synthetic{synthetic_pct}: {muestras_a_generar} synthetic samples."
            )
    else:
        print("\n[FULL-REAL MODE] real100. No synthetic data is generated.")
        return
            
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
            print(f"\n>>> SKIPPING DIAGNOSIS: {diag_objetivo} | It already has the maximum number of samples.")
            continue
        
        print(f"\n>>> PROCESSING DIAGNOSIS: {diag_objetivo} | TARGET: {n_objetivo} samples")
        samples_needed = n_objetivo

        # Varies the seed on each retry to avoid generating exactly the same targets.
        attempt_counter = 0

        while samples_needed > 0:

            print(f"Generating batch for {samples_needed} missing samples...")

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
                        print(f"Discarded (no neighbors): {target}")
                        continue

                generated_text = generar_dialogo_paciente(
                    dataset,
                    target,
                    vecinos,
                    recommended_num_ctx,
                    zero_shot=is_zero_shot,
                )

                if not validate_generated_text(generated_text, prompt_spec):
                    batch_bad += 1
                    print(f"Discarded (invalid text for dataset='{dataset}'): {target}")
                    print("[INVALID OUTPUT PREVIEW]")
                    print((generated_text or "<None>")[:500])
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
                    writer.flush()  # Immediate write: if the job dies, valid samples already written are not lost.
                else:
                    print(json.dumps(sample, ensure_ascii=False)[:400] + "...")

                batch_ok += 1

            print(f"Batch finished. Saved: {batch_ok} | Discarded: {batch_bad}")
            
            # We retry only the samples that failed validation.
            samples_needed = batch_bad
            attempt_counter += 1

            # Safety limit: prevents infinite loops if the model keeps failing.
            if attempt_counter > 10:
                print(f"ABORTING {diag_objetivo}: too many failed attempts ({attempt_counter}).")
                break

    if writer is not None:
        writer.close()
        
    print("Process finished.") 
    
if __name__ == "__main__":
    if args_slurm.self_check:
        self_check_prompt_specs()
    else:
        main()
