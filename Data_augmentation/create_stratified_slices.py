"""
Generates nested stratified real subsets for low-resource experiments.

For each percentage in PERCENTAGES, it writes a train_{dataset}_real{pct}.jsonl with the
rows from the lower percentile within each diagnostic group. Because the subsets are nested
(real20 ⊆ real40 ⊆ real60 ⊆ real80), it is possible to measure the effect of adding more
real data while keeping class balance constant.
"""

import argparse
from pathlib import Path

import pandas as pd


DATASET = "pitt"
INPUT_TEMPLATE = "./jsonl/individual_sets/train_{dataset}.jsonl"
OUTPUT_DIR = Path("./jsonl/synthetic_data/real")
PERCENTAGES = (20, 40, 60, 80)
RANDOM_STATE = 42


def shuffle_and_assign_percentile(group: pd.DataFrame) -> pd.DataFrame:
    # Assigns a percentile range [0, 1] within each diagnostic group so that
    # threshold filtering produces nested, class-balanced subsets.
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
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    df = pd.read_json(input_path, lines=True)
    if "Diagnosis" not in df.columns:
        raise ValueError("The column 'Diagnosis' does not exist in the dataset.")

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

        print(f"\nSaved file: {output_path}")
        print(f"Samples: {len(subset)}")
        print("Diagnosis value_counts():")
        print(subset["Diagnosis"].value_counts(dropna=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generates nested stratified real subsets from train_{dataset}.jsonl"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help=f"Dataset name (default: {DATASET})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(dataset=args.dataset)
