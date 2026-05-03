# Converting the transcribed audio files to normalized data points
import argparse
import os
from typing import Iterator
from .collection import Collection, RawDataPoint, NormalizedDataPoint
import json
from dataclasses import asdict


class ASRCollection(Collection):
    def __init__(self, file_path: str, metadata: dict):
        self.file_path = file_path
        self.metadata = metadata


    def __iter__(self) -> Iterator[RawDataPoint]:
        return self.parse_cha_file(self.file_path)


    def parse_cha_file(self, file_path: str) -> Iterator[RawDataPoint]:
        """
        Parses a JSON ASR file and enriches it with TAUKADIAL metadata.
        """

        with open(file_path, 'r', encoding='utf-8') as file:
            data_point = json.load(file)

            for item in data_point:
                file_id = item.get("file_name", "Unknown")

                # --- base info (igual que antes) ---
                info = {
                    "age": "Unknown",
                    "gender": "Unknown",
                    "PID": file_id,
                    "Languages": item.get("language", "Unknown"),
                    "Participants": [],
                    "File_ID": file_id,
                    "Media": "Audio",
                    "Education": "Unknown",
                    "Modality": "Speech",
                    "Task": ["Connected Speech"],
                    "Dataset": "TAUKADIAL",
                    "Diagnosis": "Unknown",
                    "MMSE": "Unknown",
                    "Continents": "Unknown",
                    "Countries": "Unknown",
                    "Duration": "Unknown",
                    "Location": "Unknown",
                    "Date": "Unknown",
                    "Transcriber": "ASR",
                    "Moca": "Unknown",
                    "Setting": "Unknown",
                    "Comment": "Unknown",
                    "text_participant": item.get("transcription", []),
                    "text_interviewer": [],
                    "text_interviewer_participant": item.get("transcription", []),
                }

                # --- ENRICHER TAUKADIAL ---
                if file_id in self.metadata:
                    meta = self.metadata[file_id]

                    info["age"] = meta.get("Age", "Unknown")
                    info["gender"] = meta.get("Gender", "Unknown")
                    info["MMSE"] = meta.get("MMSE", "Unknown")
                    info["Diagnosis"] = meta.get("Diagnosis", "Unknown")

                yield info


    def normalize_datapoint(self, raw_datapoint: RawDataPoint) -> NormalizedDataPoint:
        """
        Normalize a raw data point into a standardized format.
        """
        return NormalizedDataPoint(
            PID=raw_datapoint["PID"],
            Languages=raw_datapoint["Languages"],
            MMSE=raw_datapoint["MMSE"],
            Diagnosis=raw_datapoint["Diagnosis"],
            Participants=raw_datapoint["Participants"],
            Dataset=raw_datapoint["Dataset"],
            Modality=raw_datapoint["Media"],
            Task=raw_datapoint["Task"],
            File_ID=raw_datapoint["File_ID"],
            Media=raw_datapoint["Media"],
            Age=raw_datapoint["age"],
            Gender=raw_datapoint["gender"],
            Education=raw_datapoint["Education"],
            Source="CHA Dataset",
            Continents=raw_datapoint["Continents"],
            Countries=raw_datapoint["Countries"],
            Duration=raw_datapoint["Duration"],
            Location=raw_datapoint["Location"],
            Date=raw_datapoint["Date"],
            Transcriber=raw_datapoint["Transcriber"],
            Moca=raw_datapoint["Moca"],
            Setting=raw_datapoint["Setting"],
            Comment=raw_datapoint["Comment"],
            Text_interviewer_participant=raw_datapoint["text_interviewer_participant"],
            Text_participant=raw_datapoint["text_participant"],
            Text_interviewer=raw_datapoint["text_interviewer"]
        )

DEFAULT_OUTPUT_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection"


def build_metadata_loader(split: str):
    if split == "train":
        from metadata_integration.Loaders.taukadial_train_loader import TAUKADIALTrainLoader

        return TAUKADIALTrainLoader()
    if split == "test":
        from metadata_integration.Loaders.taukadial_test_loader import TAUKADIALTestLoader

        return TAUKADIALTestLoader()
    raise ValueError(f"Unsupported TAUKADIAL split: {split}")


def write_asr_collection(input_json: str, split: str, output_dir: str, output_name: str | None = None) -> str:
    metadata_loader = build_metadata_loader(split)
    taukadial_metadata = metadata_loader.load_metadata()
    collection = ASRCollection(input_json, taukadial_metadata)

    os.makedirs(output_dir, exist_ok=True)
    output_file_name = output_name or f"taukadial_English_{split}.jsonl"
    output_file_path = os.path.join(output_dir, output_file_name)

    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for raw_datapoint in collection:
            normalized_datapoint = collection.normalize_datapoint(raw_datapoint)
            normalized_dict = asdict(normalized_datapoint)
            json.dump(normalized_dict, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(f"ASR JSONL saved to: {output_file_path}")
    return output_file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize TAUKADIAL ASR transcriptions to JSONL.")
    parser.add_argument("--input-json", required=True, help="ASR transcription JSON generated by audio_transcription.")
    parser.add_argument("--split", required=True, choices=["train", "test"], help="TAUKADIAL split metadata to use.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for normalized JSONL output.")
    parser.add_argument("--output-name", default=None, help="Optional output filename.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_asr_collection(
        input_json=args.input_json,
        split=args.split,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )
