from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable


# ============================================================
# CONFIG FIJA
# ============================================================
DATA_ROOT = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl"

TEXT_FIELD = "Text_interviewer_participant"

# Campos a eliminar (si existen)
DROP_FIELDS = ["Text_participant", "Text_interviewer"]

# Corpus excluidos del análisis de marcadores por carecer de anotación CHAT válida:
# - TAUKADIAL: transcripción automática (ASR), 0 marcadores en 506 transcripciones
# - PerLA: solo clase Dementia, sin pausas ni reformulaciones anotadas
EXCLUDE_DATASETS: dict[str, set[str]] = {
    "en":  {"TAUKADIAL"},
    "spa": {"PerLA"},
}

# Si un marcador NO está activo en el modo elegido:
# - True: lo elimina (recomendado para aislar señal)
# - False: lo deja como estaba
DROP_INACTIVE_MARKERS = True


# =========================
# TOKENS DE SALIDA
# =========================
PAUSE_TOKEN = "[PAUSE]"
REP_TOKEN   = "[REP]"
REF_TOKEN   = "[REF]"


# =========================
# REGEX DE MARCAS
# =========================
RE_PAUSE_ANY = re.compile(r"\(\s*(?:\.\s*|\.\.\s*|\.\.\.\s*|\d+(?:\.\d+)?)\s*\)")
# acepta "/" o "\/" dentro de los corchetes
RE_REP = re.compile(r"\[\s*(?:/|\\/)\s*\]")
# acepta "//" o "\/\/" dentro de los corchetes
RE_REF = re.compile(r"\[\s*(?:(?:/|\\/)\s*(?:/|\\/))\s*\]")


def preprocess_text(text: Any, mode: str) -> str:
    """
    mode: "none" | "pause" | "rep" | "ref" | "all"
    - Sustituye SOLO el/los marcadores activados por su token.
    - Si DROP_INACTIVE_MARKERS=True, elimina los otros marcadores.
    - "none" elimina los tres marcadores sin añadir tokens especiales.
    """
    t = "" if text is None else str(text)

    use_pause = mode in ("pause", "all")
    use_rep   = mode in ("rep", "all")
    use_ref   = mode in ("ref", "all")

    # REP
    if use_rep:
        t = RE_REP.sub(f" {REP_TOKEN} ", t)
    elif DROP_INACTIVE_MARKERS:
        t = RE_REP.sub(" ", t)

    # REF
    if use_ref:
        t = RE_REF.sub(f" {REF_TOKEN} ", t)
    elif DROP_INACTIVE_MARKERS:
        t = RE_REF.sub(" ", t)

    # PAUSE
    if use_pause:
        t = RE_PAUSE_ANY.sub(f" {PAUSE_TOKEN} ", t)
    elif DROP_INACTIVE_MARKERS:
        t = RE_PAUSE_ANY.sub(" ", t)

    # normaliza espacios
    return re.sub(r"\s+", " ", t).strip()


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def preprocess_file(input_path: str, output_path: str, mode: str, language: str) -> None:
    excluded = EXCLUDE_DATASETS.get(language, set())

    def _rows():
        skipped = 0
        for obj in iter_jsonl(input_path):
            if obj.get("Dataset") in excluded:
                skipped += 1
                continue

            obj[TEXT_FIELD] = preprocess_text(obj.get(TEXT_FIELD, ""), mode=mode)

            for k in DROP_FIELDS:
                obj.pop(k, None)

            yield obj

        if skipped:
            print(f"  Excluidos por corpus sin anotación CHAT: {skipped} registros ({excluded})")

    write_jsonl(output_path, _rows())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess markers in JSONL with a given mode.")
    parser.add_argument(
        "--language",
        required=True,
        choices=["en", "spa", "english", "spanish"],
        help="Language split to process.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["none", "pause", "rep", "ref", "all"],
        help="Which marker type to keep: none | pause | rep | ref | all",
    )
    parser.add_argument(
        "--data-root",
        default=DATA_ROOT,
        help="Base directory containing train/test JSONL files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    language = args.language
    mode = args.mode
    input_prefix_by_language = {
        "en": "en",
        "english": "en",
        "spa": "spa",
        "spanish": "spa",
    }
    output_prefix_by_language = {
        "en": "en",
        "english": "en",
        "spa": "spa",
        "spanish": "spa",
    }
    input_prefix = input_prefix_by_language[language]
    output_prefix = output_prefix_by_language[language]
    train_in = os.path.join(args.data_root, f"train_{input_prefix}_e5.jsonl")
    test_in = os.path.join(args.data_root, f"test_{input_prefix}_e5.jsonl")

    # Salidas por modo (para no pisar archivos)
    output_dir = os.path.join(args.data_root, "markers_collections")
    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, f"train_{output_prefix}_e5_markers_{mode}.jsonl")
    test_out = os.path.join(output_dir, f"test_{output_prefix}_e5_markers_{mode}.jsonl")

    print("=== Preprocessing markers + dropping fields ===")
    print("Language:", language)
    print("Mode:", mode)
    print("Train in :", train_in)
    print("Train out:", train_out)
    print("Test  in :", test_in)
    print("Test  out:", test_out)
    print("Overwriting field:", TEXT_FIELD)
    print("Dropping fields:", DROP_FIELDS)
    print("Drop inactive markers:", DROP_INACTIVE_MARKERS)
    print("Tokens:", PAUSE_TOKEN, REP_TOKEN, REF_TOKEN)

    preprocess_file(train_in, train_out, mode=mode, language=input_prefix)
    preprocess_file(test_in, test_out, mode=mode, language=input_prefix)

    print("Done")


if __name__ == "__main__":
    main()
