from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

DATA_ROOT = Path("./jsonl")
RESULTS_DIRNAME = "results_cha_collection"
OUTPUT_DIRNAME = "individual_sets"
DEFAULT_DATASETS = ("pitt", "delaware", "lu", "taukadial", "vas", "wls")
ALL_DATASETS = DEFAULT_DATASETS + ("baycrest", "kempler")
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TEXT_LEN = 60
TEXT_FIELD = "Text_interviewer_participant"

DATASET_INPUTS = {
    "baycrest": {"raw": "Baycrest.jsonl"},
    "delaware": {"raw": "Delaware.jsonl"},
    "kempler": {"raw": "Kempler.jsonl"},
    "lu": {"raw": "Lu.jsonl"},
    "pitt": {"raw": "Pitt.jsonl"},
    "vas": {"raw": "VAS.jsonl"},
    "wls": {"raw": "WLS.jsonl"},
    "taukadial": {
        "train": "taukadial_English_train.jsonl",
        "test": "taukadial_English_test.jsonl",
    },
}


def remove_zh_language_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "Languages" not in df.columns:
        return df
    return df[df["Languages"] != "zh"]


def clean_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    if "Diagnosis" not in df.columns:
        raise ValueError("No existe la columna 'Diagnosis' en el JSONL.")

    diagnoses_to_remove = ["Vascular", "Memory", "Aphasia", "Pick's", "Other"]
    df = df[~df["Diagnosis"].isin(diagnoses_to_remove)]
    df = df[df["Diagnosis"].notna() & (df["Diagnosis"] != "")].copy()

    df["Diagnosis"] = df["Diagnosis"].replace(
        {
            "Control": "HC",
            "Conrol": "HC",
            "NC": "HC",
            "H": "HC",
            "AD": "Dementia",
            "DM": "Dementia",
            "PossibleAD": "Dementia",
            "ProbableAD": "Dementia",
            "Probable": "Dementia",
            "potential dementia": "Dementia",
            "D": "Dementia",
            "Alzheimer's": "Dementia",
        }
    )
    return df


def clean_gender(df: pd.DataFrame) -> pd.DataFrame:
    if "Gender" not in df.columns:
        return df

    df = df.copy()
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
    df["Gender"] = df["Gender"].replace(
        {
            "m": "M",
            "male": "M",
            "f": "F",
            "female": "F",
            "w": "F",
            "nan": "U",
            "none": "U",
            "": "U",
        }
    )
    df.loc[~df["Gender"].isin(["M", "F"]), "Gender"] = "U"
    return df


def preprocess_text(text, tfidf: bool = False) -> str:
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\b[A-Z]{3}\b", "", text)
    text = re.sub(r"xxx", "", text)
    text = re.sub(r"<[^>]*>", "", text)

    if tfidf:
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.replace("PAR", "")
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        text = re.sub(r"\\x[0-9A-Za-z_]+\\x", "", text)
        text = re.sub(r"\b\w+:\s*", "", text)
        text = text.replace("→", "")
        text = text.replace("(", "").replace(")", "")
        text = re.sub(r"[\\+^\"/„]", "", text)
        text = re.sub(r"[_']", "", text)
        text = text.replace("\t", " ")
        text = re.sub(r"\[.*?\]", "", text)
        for marker in ("&=laughs", "&=nods", "&=coughs", "&=snaps:tongue"):
            text = text.replace(marker, "")
        text = text.replace("<", "").replace(">", "")
        text = text.replace("*", "").replace("&", "")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"([.,!?;:])\s+\1", r"\1", text)
        text = re.sub(r"(\.\s*){2,}", ".", text)
        if "." in text:
            text = text.rsplit(".", 1)[0] + "."

    return text


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Text_interviewer_participant" not in df.columns:
        df["Text_interviewer_participant"] = ""

    df = remove_zh_language_rows(df)
    df = clean_diagnosis(df)
    df = clean_gender(df)
    df[TEXT_FIELD] = df[TEXT_FIELD].apply(preprocess_text)
    df = df[df[TEXT_FIELD].astype(str).str.len() > MIN_TEXT_LEN].copy()
    return df


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_counts = df["Diagnosis"].value_counts()
    n_classes = len(label_counts)
    test_count = math.ceil(len(df) * TEST_SIZE)
    train_count = len(df) - test_count
    can_stratify = label_counts.min() >= 2 and test_count >= n_classes and train_count >= n_classes

    return train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["Diagnosis"] if can_stratify else None,
        random_state=RANDOM_STATE,
    )


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el JSONL de entrada: {path}")
    return pd.read_json(path, lines=True)


def make_tfidf_df(df: pd.DataFrame) -> pd.DataFrame:
    tfidf_df = df.copy()
    tfidf_df[TEXT_FIELD] = tfidf_df[TEXT_FIELD].apply(lambda text: preprocess_text(text, tfidf=True))
    return tfidf_df


def write_pair(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path, dataset: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / f"train_{dataset}.jsonl"
    test_out = output_dir / f"test_{dataset}.jsonl"
    train_df.to_json(train_out, orient="records", lines=True, force_ascii=False)
    test_df.to_json(test_out, orient="records", lines=True, force_ascii=False)

def write_tfidf_pair(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path, dataset: str) -> None:
    write_pair(make_tfidf_df(train_df), make_tfidf_df(test_df), output_dir / "TFIDF", dataset)


def process_dataset(dataset: str, data_root: Path, output_dir: Path) -> None:
    config = DATASET_INPUTS[dataset]
    results_dir = data_root / RESULTS_DIRNAME

    if dataset == "taukadial":
        train_df = prepare_df(read_jsonl(results_dir / config["train"]))
        test_df = prepare_df(read_jsonl(results_dir / config["test"]))
    else:
        df = prepare_df(read_jsonl(results_dir / config["raw"]))
        train_df, test_df = stratified_split(df)

    write_pair(train_df, test_df, output_dir, dataset)
    write_tfidf_pair(train_df, test_df, output_dir, dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cleaned individual English dataset splits.")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all",) + ALL_DATASETS,
        help="Dataset to process. 'all' processes the default experimental set.",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Base jsonl directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / OUTPUT_DIRNAME,
        help="Directory where train/test individual JSONL files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global pd, train_test_split
    import pandas as pd
    from sklearn.model_selection import train_test_split

    datasets = DEFAULT_DATASETS if args.dataset == "all" else (args.dataset,)
    output_dir = args.output_dir
    if args.output_dir == DATA_ROOT / OUTPUT_DIRNAME and args.data_root != DATA_ROOT:
        output_dir = args.data_root / OUTPUT_DIRNAME

    for dataset in datasets:
        print(f"=== Preprocessing individual English dataset: {dataset} ===")
        process_dataset(dataset, args.data_root, output_dir)


if __name__ == "__main__":
    main()
