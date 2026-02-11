from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable


# ============================================================
# CONFIG FIJA
# ============================================================
TRAIN_IN  = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/train_spanish_e5.jsonl"
TEST_IN   = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/test_spanish_e5.jsonl"

TEXT_FIELD = "Text_interviewer_participant"

# Campos a eliminar (si existen)
DROP_FIELDS = ["Text_participant", "Text_interviewer"]

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
    mode: "pause" | "rep" | "ref" | "all"
    - Sustituye SOLO el/los marcadores activados por su token.
    - Si DROP_INACTIVE_MARKERS=True, elimina los otros marcadores.
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


def preprocess_file(input_path: str, output_path: str, mode: str) -> None:
    def _rows():
        for obj in iter_jsonl(input_path):
            # 1) sobrescribe el texto principal ya preprocesado
            obj[TEXT_FIELD] = preprocess_text(obj.get(TEXT_FIELD, ""), mode=mode)

            # 2) elimina campos que no quieres
            for k in DROP_FIELDS:
                obj.pop(k, None)

            yield obj

    write_jsonl(output_path, _rows())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess markers in JSONL with a given mode.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pause", "rep", "ref", "all"],
        help="Which marker type to keep: pause | rep | ref | all",
    )
    args = parser.parse_args()
    mode = args.mode

    # Salidas por modo (para no pisar archivos)
    train_out = f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/train_spanish_e5_markers_{mode}.jsonl"
    test_out  = f"/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/test_spanish_e5_markers_{mode}.jsonl"

    print("=== Preprocessing markers + dropping fields ===")
    print("Mode:", mode)
    print("Train in :", TRAIN_IN)
    print("Train out:", train_out)
    print("Test  in :", TEST_IN)
    print("Test  out:", test_out)
    print("Overwriting field:", TEXT_FIELD)
    print("Dropping fields:", DROP_FIELDS)
    print("Drop inactive markers:", DROP_INACTIVE_MARKERS)
    print("Tokens:", PAUSE_TOKEN, REP_TOKEN, REF_TOKEN)

    preprocess_file(TRAIN_IN, train_out, mode=mode)
    preprocess_file(TEST_IN, test_out, mode=mode)

    print("Done")


if __name__ == "__main__":
    main()
