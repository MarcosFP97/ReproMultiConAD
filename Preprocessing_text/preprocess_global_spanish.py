from __future__ import annotations

import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from extracting_data.collection import JSONLCombiner


RANDOM_STATE = 42
TEST_SIZE = 0.2
TEXT_FIELD = "Text_interviewer_participant"

input_files = [
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Ivanova.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/PerLA.jsonl",
]
output_directory = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl"
output_filename = "combined_jsonl_spanish.jsonl"


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


def process_transcripts(df: pd.DataFrame, word_limits: dict[str, tuple[int, int]]) -> pd.DataFrame:
    processed_data = []
    for _, row in df.iterrows():
        dataset_type = row["Dataset"]
        text = str(row[TEXT_FIELD])
        if dataset_type not in word_limits:
            continue

        min_words, max_words = word_limits[dataset_type]
        words = text.split()
        if len(words) < min_words:
            continue
        if len(words) > max_words:
            row[TEXT_FIELD] = " ".join(words[:max_words])
        processed_data.append(row)

    return pd.DataFrame(processed_data)


def prepare_light_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Diagnosis"] = df["Diagnosis"].replace({"DTA": "Dementia", "AD": "Dementia"})
    df = df[df["Diagnosis"].notnull() & (df["Diagnosis"].str.strip() != "") & (df["Diagnosis"] != "Unknown")].copy()
    df[TEXT_FIELD] = df[TEXT_FIELD].apply(preprocess_text)
    df["length"] = df[TEXT_FIELD].apply(lambda text: len(str(text).split()))

    print(df["Diagnosis"].value_counts())

    word_limits = {
        "Ivanova": (40, 100),
        "PerLA": (250, 1500),
    }
    processed_df = process_transcripts(df, word_limits)
    print(processed_df["Diagnosis"].value_counts())
    return processed_df


def make_tfidf_df(df: pd.DataFrame) -> pd.DataFrame:
    tfidf_df = df.copy()
    tfidf_df[TEXT_FIELD] = tfidf_df[TEXT_FIELD].apply(lambda text: preprocess_text(text, tfidf=True))
    tfidf_df["length"] = tfidf_df[TEXT_FIELD].apply(lambda text: len(str(text).split()))
    return tfidf_df


def main() -> None:
    os.makedirs(output_directory, exist_ok=True)
    combiner = JSONLCombiner(input_files, output_directory, output_filename)
    combiner.combine()

    spanish_df = pd.read_json(os.path.join(output_directory, output_filename), lines=True)
    light_df = prepare_light_df(spanish_df)

    train_light, test_light = train_test_split(
        light_df,
        test_size=TEST_SIZE,
        stratify=light_df["Diagnosis"],
        random_state=RANDOM_STATE,
    )

    train_tfidf = make_tfidf_df(train_light)
    test_tfidf = make_tfidf_df(test_light)

    train_tfidf.to_json(os.path.join(output_directory, "train_spa.jsonl"), orient="records", lines=True, force_ascii=False)
    test_tfidf.to_json(os.path.join(output_directory, "test_spa.jsonl"), orient="records", lines=True, force_ascii=False)
    train_light.to_json(os.path.join(output_directory, "train_spa_e5.jsonl"), orient="records", lines=True, force_ascii=False)
    test_light.to_json(os.path.join(output_directory, "test_spa_e5.jsonl"), orient="records", lines=True, force_ascii=False)

    print("\nGuardado global SPA:")
    print(" -", os.path.join(output_directory, "train_spa.jsonl"), "[TFIDF fuerte]")
    print(" -", os.path.join(output_directory, "test_spa.jsonl"), "[TFIDF fuerte]")
    print(" -", os.path.join(output_directory, "train_spa_e5.jsonl"), "[light]")
    print(" -", os.path.join(output_directory, "test_spa_e5.jsonl"), "[light]")


if __name__ == "__main__":
    main()
