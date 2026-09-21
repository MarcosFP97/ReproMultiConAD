"""
Batch ASR transcription of audio datasets using OpenAI Whisper.

Iterates over a directory of audio files, transcribes each file using the
specified Whisper model, optionally filters the results by detected language,
and writes all transcriptions to a single JSON file. Designed to run as a
one-off SLURM job on clinical corpora, prior to JSONL preprocessing.
"""

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper.")
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files.")
    parser.add_argument("--output-path", required=True, help="Path where transcriptions JSON will be written.")
    parser.add_argument("--model", default="large-v3", help="Whisper model name.")
    parser.add_argument(
        "--language-filter",
        default="en",
        help="Keep only transcriptions detected in this language. Use 'all' to keep every language.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".wav"],
        help="Audio extensions to process.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search audio files recursively under --audio-dir.",
    )
    return parser.parse_args()


def iter_audio_files(audio_dir: Path, extensions: set[str], recursive: bool):
    pattern = "**/*" if recursive else "*"
    for path in sorted(audio_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def main() -> None:
    args = parse_args()
    import whisper
    from tqdm import tqdm

    audio_dir = Path(args.audio_dir)
    output_path = Path(args.output_path)
    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}

    if not audio_dir.is_dir():
        raise NotADirectoryError(f"Audio directory does not exist: {audio_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(args.model)
    results = []

    for audio_path in tqdm(list(iter_audio_files(audio_dir, extensions, args.recursive))):
        try:
            result = model.transcribe(str(audio_path))
            detected_language = result["language"]

            if args.language_filter != "all" and detected_language != args.language_filter:
                continue

            results.append(
                {
                    "file_name": os.path.splitext(audio_path.name)[0],
                    "transcription": result["text"],
                    "language": detected_language,
                }
            )
        except Exception as exc:
            print(f"Error processing file {audio_path}: {exc}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Transcriptions saved to {output_path}")


if __name__ == "__main__":
    main()
