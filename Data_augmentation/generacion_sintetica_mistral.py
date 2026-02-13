import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from ollama import chat


# Configuracion basica
INPUT_PATH = Path("/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets/train_pitt.jsonl")
OUTPUT_PATH = Path("/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/pitt_sintetico.jsonl")
MODEL_NAME = "mistral-small"

BASIC = False
SAVE = True
SHOW_STATS = True

K_NEIGHBORS = 3
N_PER_DIAG = 10
RANDOM_SEED = 42


def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Carga el JSONL y deja solo las columnas necesarias para el pipeline."""
    df = pd.read_json(ruta, lines=True)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["MMSE"] = pd.to_numeric(df["MMSE"], errors="coerce")
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()

    # Para kNN y stats necesitamos Age/MMSE validos.
    return df.dropna(subset=["Text_interviewer_participant", "Diagnosis", "Age", "MMSE"]).copy()


def analizar_estadisticas(df: pd.DataFrame) -> pd.DataFrame:
    """Imprime describe por diagnostico usando option_context (sin tocar opciones globales)."""
    stats = df.groupby("Diagnosis")[["Age", "MMSE"]].describe()
    gender_pct = df["Gender"].value_counts(normalize=True)
    female_pct = gender_pct.get("female", 0.0) * 100
    male_pct = gender_pct.get("male", 0.0) * 100

    print("\n" + "=" * 70)
    print("ESTADISTICA DESCRIPTIVA")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 170, "display.float_format", lambda x: f"{x:,.2f}"):
        print(stats)
    print(f"\nGender en dataset -> female: {female_pct:.2f}% | male: {male_pct:.2f}%")
    print("=" * 70 + "\n")
    return stats


def generar_targets(df: pd.DataFrame, n_per_diag: int, seed: int) -> list[dict]:
    """
    Genera perfiles sinteticos por diagnostico:
    - Age: Normal + clipping a min/max del grupo.
    - MMSE: bootstrap del grupo (mejor que Normal cuando MMSE esta muy concentrado, p.ej. HC).
    - Gender: 50/50.
    """
    rng = np.random.default_rng(seed)
    targets = []

    for diagnosis, g in df.groupby("Diagnosis"):
        age_mean = g["Age"].mean()
        age_std = g["Age"].std()
        age_min = g["Age"].min()
        age_max = g["Age"].max()
        mmse_values = g["MMSE"].dropna().to_numpy()

        # Si std=0 o NaN (grupo muy pequeno/constante), usamos 1e-6 para evitar problemas.
        if pd.isna(age_std) or age_std == 0:
            age_std = 1e-6

        for _ in range(n_per_diag):
            gen_age = rng.normal(age_mean, age_std)
            gen_age = float(np.clip(gen_age, age_min, age_max))

            # Bootstrap MMSE: sample real del mismo diagnostico.
            gen_mmse = float(rng.choice(mmse_values))
            gen_mmse = float(np.clip(gen_mmse, 0, 30))

            targets.append(
                {
                    "Diagnosis": diagnosis,
                    "Age": int(round(gen_age)),
                    "MMSE": int(round(gen_mmse)),
                    "Gender": rng.choice(["female", "male"]),
                }
            )

    return targets


def buscar_vecinos_knn(target: dict, df_real: pd.DataFrame, k: int = 3) -> pd.DataFrame | None:
    """
    Busca vecinos por Diagnosis+Gender y hace fallback a Diagnosis.
    Incluye min-max seguro para evitar division por cero.
    """
    # Filtrado inicial por diagnóstico y género
    df_filtrado = df_real[
        (df_real["Diagnosis"] == target["Diagnosis"]) & (df_real["Gender"] == target["Gender"])
    ].copy()

    if len(df_filtrado) < k:
        df_filtrado = df_real[df_real["Diagnosis"] == target["Diagnosis"]].copy()

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

    has_inv = any(l.startswith("*INV:") for l in lines)
    has_par = any(l.startswith("*PAR:") for l in lines)
    return has_inv and has_par


def generar_chat_pitt_dialogo(target: dict, vecinos: pd.DataFrame) -> str | None:
    """Construye prompt BASIC/SMART, llama a Ollama y devuelve el texto generado."""
    bloques = []
    for i, r in enumerate(vecinos.itertuples(index=False), 1):
        # Header con metadatos clinicos para guiar mejor al modelo.
        bloques.append(
            f"--- Neighbor {i} | Diagnosis: {r.Diagnosis} | Age: {r.Age} | MMSE: {r.MMSE} | Gender: {r.Gender} ---\n"
            f"{r.Text_interviewer_participant}"
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
            "Output only turns from *INV: and *PAR:."
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
            """.strip()
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]

    response = chat(model=MODEL_NAME, messages=messages)
    return response.message.content.strip()


def main() -> None:
    print("Cargando datos...")
    df_real = cargar_datos(INPUT_PATH)

    if SHOW_STATS:
        analizar_estadisticas(df_real)

    print("Generando targets...")
    targets = generar_targets(df_real, n_per_diag=N_PER_DIAG, seed=RANDOM_SEED)

    writer = None
    if SAVE:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        writer = OUTPUT_PATH.open("w", encoding="utf-8")

    total_ok = 0
    total_bad = 0

    for target in targets:
        vecinos = buscar_vecinos_knn(target, df_real, k=K_NEIGHBORS)
        if vecinos is None or vecinos.empty:
            total_bad += 1
            print(f"Descartado (sin vecinos): {target}")
            continue

        generated_text = generar_chat_pitt_dialogo(target, vecinos)
        if not validate_chat(generated_text):
            total_bad += 1
            print(f"Descartado (CHAT invalido): {target}")
            continue

        neighbors_meta = vecinos[["Age", "MMSE", "Gender"]].to_dict("records")
        sample = {
            "Diagnosis": target["Diagnosis"],
            "Age": target["Age"],
            "MMSE": target["MMSE"],
            "Gender": target["Gender"],
            "neighbors_meta": neighbors_meta,
            "generated_text": generated_text,
        }

        if SAVE:
            writer.write(json.dumps(sample, ensure_ascii=False) + "\n")
        else:
            print(json.dumps(sample, ensure_ascii=False)[:400] + "...")

        total_ok += 1

    if writer is not None:
        writer.close()

    print(f"Proceso finalizado. Guardadas/validas: {total_ok} | Descartadas: {total_bad}")


if __name__ == "__main__":
    main()
