from __future__ import annotations

import argparse
from pathlib import Path
import re

DATA_ROOT = Path("./jsonl")
RESULTS_DIRNAME = "results_cha_collection"
OUTPUT_DIRNAME = "individual_sets"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TEXT_FIELD = "Text_interviewer_participant"
MIN_WORDS = 40
MAX_WORDS = 100

DATASET_INPUTS = {
    "ivanova": "Ivanova.jsonl",
}


def preprocess_text(text: str, tfidf: bool = False) -> str:
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


def trim_by_word_limits(text: str, min_words: int, max_words: int):
    words = str(text).split()
    if len(words) < min_words:
        return None
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Diagnosis"] = df["Diagnosis"].replace({"DTA": "Dementia", "AD": "Dementia"})
    df = clean_gender(df)
    df = df[df["Diagnosis"].notnull()].copy()
    df = df[df["Diagnosis"].astype(str).str.strip().ne("")].copy()
    df = df[df["Diagnosis"].ne("Unknown")].copy()
    df[TEXT_FIELD] = df[TEXT_FIELD].apply(preprocess_text)
    df[TEXT_FIELD] = df[TEXT_FIELD].apply(
        lambda text: trim_by_word_limits(text, MIN_WORDS, MAX_WORDS)
    )
    df = df[df[TEXT_FIELD].notnull()].copy()
    df["length"] = df[TEXT_FIELD].apply(lambda text: len(str(text).split()))
    return df


def make_tfidf_df(df: pd.DataFrame) -> pd.DataFrame:
    tfidf_df = df.copy()
    tfidf_df[TEXT_FIELD] = tfidf_df[TEXT_FIELD].apply(lambda text: preprocess_text(text, tfidf=True))
    return tfidf_df


def process_dataset(dataset: str, data_root: Path, output_dir: Path) -> None:
    input_path = data_root / RESULTS_DIRNAME / DATASET_INPUTS[dataset]
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el JSONL de entrada: {input_path}")

    df = prepare_df(pd.read_json(input_path, lines=True))
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["Diagnosis"],
        random_state=RANDOM_STATE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / f"train_{dataset}.jsonl"
    test_out = output_dir / f"test_{dataset}.jsonl"
    train_df.to_json(train_out, orient="records", lines=True, force_ascii=False)
    test_df.to_json(test_out, orient="records", lines=True, force_ascii=False)

    tfidf_output_dir = output_dir / "TFIDF"
    tfidf_output_dir.mkdir(parents=True, exist_ok=True)
    tfidf_train_out = tfidf_output_dir / f"train_{dataset}.jsonl"
    tfidf_test_out = tfidf_output_dir / f"test_{dataset}.jsonl"
    make_tfidf_df(train_df).to_json(tfidf_train_out, orient="records", lines=True, force_ascii=False)
    make_tfidf_df(test_df).to_json(tfidf_test_out, orient="records", lines=True, force_ascii=False)
    print(" -", tfidf_train_out)
    print(" -", tfidf_test_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cleaned individual Spanish dataset splits.")
    parser.add_argument("--dataset", default="all", choices=("all", "ivanova"), help="Dataset to process.")
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

    datasets = tuple(DATASET_INPUTS) if args.dataset == "all" else (args.dataset,)
    output_dir = args.output_dir
    if args.output_dir == DATA_ROOT / OUTPUT_DIRNAME and args.data_root != DATA_ROOT:
        output_dir = args.data_root / OUTPUT_DIRNAME

    for dataset in datasets:
        print(f"=== Preprocessing individual Spanish dataset: {dataset} ===")
        process_dataset(dataset, args.data_root, output_dir)


if __name__ == "__main__":
    main()
