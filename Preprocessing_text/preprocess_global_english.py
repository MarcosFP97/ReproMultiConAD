from __future__ import annotations

import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from extracting_data.collection import JSONLCombiner


RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TEXT_LEN = 60
TEXT_FIELD = "Text_interviewer_participant"
PARTICIPANT_FIELD = "Text_participant"

input_files = [
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Pitt.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Lu.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Baycrest.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/VAS.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Kempler.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/WLS.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/Delaware.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/taukadial_English_train.jsonl",
    "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection/taukadial_English_test.jsonl",
]

output_directory = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl"
output_filename = "combined_jsonl_english.jsonl"


def remove_zh_language_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "Languages" not in df.columns:
        return df
    return df[df["Languages"] != "zh"]


def clean_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    diagnoses_to_remove = ["Vascular", "Memory", "Aphasia", "Pick's", "Other"]
    df = df[~df["Diagnosis"].isin(diagnoses_to_remove)].copy()
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
    text = re.sub(r"\x15[0-9_]+\x15", " ", text)

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


def ensure_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TEXT_FIELD not in df.columns:
        df[TEXT_FIELD] = ""
    if PARTICIPANT_FIELD not in df.columns:
        df[PARTICIPANT_FIELD] = ""
    return df


def prepare_light_df(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_text_columns(df)
    df = remove_zh_language_rows(df)
    df = clean_diagnosis(df)
    df = clean_gender(df)
    print(df["Diagnosis"].value_counts())

    df[TEXT_FIELD] = df[TEXT_FIELD].apply(preprocess_text)
    df[PARTICIPANT_FIELD] = df[PARTICIPANT_FIELD].apply(preprocess_text)
    df["Text_length"] = df[TEXT_FIELD].apply(len)
    df = df[df["Text_length"] > MIN_TEXT_LEN].copy()
    return df


def make_tfidf_df(df: pd.DataFrame) -> pd.DataFrame:
    tfidf_df = df.copy()
    tfidf_df[TEXT_FIELD] = tfidf_df[TEXT_FIELD].apply(lambda text: preprocess_text(text, tfidf=True))
    tfidf_df[PARTICIPANT_FIELD] = tfidf_df[PARTICIPANT_FIELD].apply(lambda text: preprocess_text(text, tfidf=True))
    tfidf_df["Text_length"] = tfidf_df[TEXT_FIELD].apply(len)
    return tfidf_df


def main() -> None:
    os.makedirs(output_directory, exist_ok=True)
    combiner = JSONLCombiner(input_files, output_directory, output_filename)
    combiner.combine()

    english_df = pd.read_json(os.path.join(output_directory, output_filename), lines=True)
    light_df = prepare_light_df(english_df)

    train_light, test_light = train_test_split(
        light_df,
        test_size=TEST_SIZE,
        stratify=light_df["Diagnosis"],
        random_state=RANDOM_STATE,
    )

    train_tfidf = make_tfidf_df(train_light)
    test_tfidf = make_tfidf_df(test_light)

    train_tfidf.to_json(os.path.join(output_directory, "train_en.jsonl"), orient="records", lines=True, force_ascii=False)
    test_tfidf.to_json(os.path.join(output_directory, "test_en.jsonl"), orient="records", lines=True, force_ascii=False)
    train_light.to_json(os.path.join(output_directory, "train_en_e5.jsonl"), orient="records", lines=True, force_ascii=False)
    test_light.to_json(os.path.join(output_directory, "test_en_e5.jsonl"), orient="records", lines=True, force_ascii=False)

    print("\nGuardado global EN:")
    print(" -", os.path.join(output_directory, "train_en.jsonl"), "[TFIDF fuerte]")
    print(" -", os.path.join(output_directory, "test_en.jsonl"), "[TFIDF fuerte]")
    print(" -", os.path.join(output_directory, "train_en_e5.jsonl"), "[light]")
    print(" -", os.path.join(output_directory, "test_en_e5.jsonl"), "[light]")


if __name__ == "__main__":
    main()
