"""
Genera subconjuntos reales estratificados y anidados para los experimentos de bajo recurso.

Por cada porcentaje en PERCENTAGES escribe un train_{dataset}_real{pct}.jsonl con las
filas de menor percentil por grupo diagnóstico. Al ser anidados (real20 ⊆ real40 ⊆ real60 ⊆ real80),
se puede medir el efecto de añadir más datos reales manteniendo el balance de clases constante.
"""

import argparse
from pathlib import Path

import pandas as pd


DATASET = "pitt"
INPUT_TEMPLATE = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets/train_{dataset}.jsonl"
OUTPUT_DIR = Path("/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/synthetic_data/real")
PERCENTAGES = (20, 40, 60, 80)
RANDOM_STATE = 42


def shuffle_and_assign_percentile(group: pd.DataFrame) -> pd.DataFrame:
    # Asigna un rango percentil [0, 1] dentro de cada grupo diagnóstico para que
    # el filtrado por umbral produzca subconjuntos anidados y balanceados por clase.
    shuffled = group.sample(frac=1.0, random_state=RANDOM_STATE).copy()
    n_rows = len(shuffled)

    if n_rows == 1:
        shuffled["_percentile"] = 0.0
    else:
        shuffled["_percentile"] = [i / (n_rows - 1) for i in range(n_rows)]

    return shuffled


def main(dataset: str = DATASET) -> None:
    input_path = Path(INPUT_TEMPLATE.format(dataset=dataset))
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el dataset de entrada: {input_path}")

    df = pd.read_json(input_path, lines=True)
    if "Diagnosis" not in df.columns:
        raise ValueError("La columna 'Diagnosis' no existe en el dataset.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_with_percentile = (
        df.groupby("Diagnosis", group_keys=False, dropna=False)
        .apply(shuffle_and_assign_percentile)
        .reset_index(drop=True)
    )

    for percentage in PERCENTAGES:
        threshold = percentage / 100.0
        subset = df_with_percentile[df_with_percentile["_percentile"] <= threshold].drop(
            columns="_percentile"
        )

        output_path = OUTPUT_DIR / f"train_{dataset}_real{percentage}.jsonl"
        subset.to_json(output_path, orient="records", lines=True, force_ascii=False)

        print(f"\nArchivo guardado: {output_path}")
        print(f"Muestras: {len(subset)}")
        print("Diagnosis value_counts():")
        print(subset["Diagnosis"].value_counts(dropna=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera subconjuntos reales estratificados anidados desde train_{dataset}.jsonl"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help=f"Nombre del dataset (por defecto: {DATASET})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(dataset=args.dataset)
