"""
Análisis de interpretabilidad SHAP sobre modelos BERT fine-tuned con marcadores CHAT.

Genera un fichero HTML por combinación (language, task, marker) con visualizaciones
de atribución de tokens usando shap.PartitionExplainer. Diseñado para ejecutarse como
SLURM Job Array: cada job analiza la misma combinación con el dataset completo y extrae
una muestra estratificada priorizando ejemplos con marcadores CHAT.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from html import escape
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MAX_LEN = 256
TEXT_COL = "Text_interviewer_participant"
LABEL_COL = "Diagnosis"
TASK_LABELS = {
    "binary": ["Dementia", "HC"],
    "multiclass": ["Dementia", "HC", "MCI"],
}

DEFAULT_ROOT = Path("/mnt/beegfs/groups/irgroup/sara_tfg")
DEFAULT_MODEL_ROOT = DEFAULT_ROOT / "MultiConAD" / "Experiments" / "BERT_Models"
DEFAULT_TEST_ROOT = DEFAULT_ROOT / "jsonl" / "markers_collections"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisis SHAP para BERT con truncado robusto.")
    parser.add_argument("--marker", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["binary", "multiclass"])
    parser.add_argument("--sample-size", type=int, default=10, help="Numero de ejemplos para SHAP.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestreo reproducible.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--slurm-task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
        help="Indice del Job Array (0-based).",
    )
    parser.add_argument(
        "--slurm-task-count",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")),
        help="Numero total de jobs del array.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_dataset(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_json(path, lines=True)
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar el dataset en: {path}") from exc

    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas obligatorias en el dataset: {sorted(missing)}")

    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df[LABEL_COL] = df[LABEL_COL].fillna("").astype(str)
    return df


def load_model_and_tokenizer(model_dir: Path) -> tuple[Any, Any]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar modelo/tokenizer desde: {model_dir}") from exc

    if not tokenizer.is_fast:
        raise RuntimeError(
            "El tokenizer cargado no es fast tokenizer; se requiere para offset_mapping exacto."
        )
    return tokenizer, model


def resolve_id2label(model: Any, task: str) -> dict[int, str]:
    expected_num_labels = int(getattr(model.config, "num_labels", len(TASK_LABELS[task])))
    task_defaults = TASK_LABELS[task]
    fallback = {
        idx: task_defaults[idx] if idx < len(task_defaults) else f"LABEL_{idx}"
        for idx in range(expected_num_labels)
    }

    raw = getattr(model.config, "id2label", {})
    parsed: dict[int, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                parsed[int(key)] = str(value)
            except (ValueError, TypeError):
                continue

    complete_cfg = set(parsed.keys()) == set(range(expected_num_labels))
    generic_cfg = complete_cfg and all(label.upper().startswith("LABEL_") for label in parsed.values())

    if complete_cfg and not generic_cfg:
        id2label = parsed
    else:
        id2label = fallback

    model.config.id2label = id2label
    model.config.label2id = {label: idx for idx, label in id2label.items()}
    return id2label


def filter_by_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    # Se mantiene el enfoque original: centrarse en clases clinicamente relevantes.
    if task == "binary":
        focus_labels = {"Dementia", "HC"}
    else:
        focus_labels = {"Dementia", "MCI", "HC"}

    filtered = df[df[LABEL_COL].isin(focus_labels)].copy()
    if filtered.empty:
        logging.warning(
            "No hay muestras para etiquetas objetivo %s; se usan todas las filas disponibles.",
            sorted(focus_labels),
        )
        return df.copy()

    logging.info(
        "Filtrado por tarea '%s': %d -> %d filas.",
        task,
        len(df),
        len(filtered),
    )
    return filtered


def split_for_slurm(df: pd.DataFrame, task_id: int, task_count: int, seed: int) -> pd.DataFrame:
    """
    Anulamos la división por chunks para que cada Job (combinación)
    tenga acceso al dataset completo y pueda extraer sus 10 muestras.
    """
    logging.info("Dataset completo disponible para esta combinación (%d filas).", len(df))
    return df


def _sample_part(df: pd.DataFrame, take: int, seed: int) -> pd.DataFrame:
    if take <= 0 or df.empty:
        return df.head(0).copy()
    if len(df) <= take:
        return df.copy()
    return df.sample(n=take, random_state=seed)


def select_samples(df: pd.DataFrame, language: str, sample_size: int, seed: int) -> pd.DataFrame:
    work = df.copy()
    work["has_chat_marker"] = work[TEXT_COL].str.contains(r"\[[^\]]+\]", regex=True, na=False)
    lang = language.strip().lower()

    selected_parts: list[pd.DataFrame] = []
    selected_index: set[int] = set()

    def consume_group(name: str, group: pd.DataFrame, seed_offset: int) -> None:
        remaining = sample_size - sum(len(p) for p in selected_parts)
        if remaining <= 0:
            return
        group = group[~group.index.isin(selected_index)]
        taken = _sample_part(group, remaining, seed + seed_offset)
        if not taken.empty:
            selected_parts.append(taken)
            selected_index.update(taken.index.to_list())
            logging.info("Seleccion '%s': %d muestras.", name, len(taken))

    if lang == "spa" and "Dataset" in work.columns:
        dataset_norm = work["Dataset"].fillna("").astype(str).str.strip().str.casefold()
        ivanova_mask = dataset_norm.eq("ivanova")
        marker_mask = work["has_chat_marker"]

        consume_group("Ivanova + CHAT", work[ivanova_mask & marker_mask], 0)
        consume_group("CHAT resto datasets", work[~ivanova_mask & marker_mask], 1)
        consume_group("Ivanova sin CHAT", work[ivanova_mask & ~marker_mask], 2)
        consume_group("Resto", work[~ivanova_mask & ~marker_mask], 3)
    else:
        marker_mask = work["has_chat_marker"]
        consume_group("CHAT", work[marker_mask], 0)
        consume_group("Resto", work[~marker_mask], 1)

    if selected_parts:
        result = pd.concat(selected_parts, axis=0).copy()
    else:
        result = work.head(0).copy()

    if len(result) < sample_size:
        logging.warning(
            "No hay suficientes filas para completar %d muestras. Seleccionadas: %d.",
            sample_size,
            len(result),
        )
    return result.head(sample_size)


def truncate_with_offset_mapping(text: str, tokenizer: Any, max_len: int = MAX_LEN) -> str:
    if not text:
        return text

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if not offsets or len(offsets) <= max_len:
        return text

    # Se toma exactamente la ventana de 256 tokens de entrada del modelo.
    first_window = offsets[:max_len]
    valid_ends = [end for start, end in first_window if end > start]
    if not valid_ends:
        return text

    cut_char = max(valid_ends)
    truncated = text[:cut_char]

    # Evita dejar marcadores CHAT abiertos (ej. "[REP" sin corchete de cierre).
    if truncated.count("[") > truncated.count("]"):
        last_open = truncated.rfind("[")
        if last_open != -1 and truncated.find("]", last_open) == -1:
            truncated = truncated[:last_open].rstrip()

    # Evita cortar disfluencias CHAT tipo "&-eh" por la mitad.
    if cut_char < len(text):
        tail_start = truncated.rfind(" ")
        tail_token = truncated[tail_start + 1 :] if tail_start != -1 else truncated
        next_char = text[cut_char]
        if tail_token.startswith("&-") and not next_char.isspace():
            truncated = truncated[: tail_start + 1].rstrip() if tail_start != -1 else ""

    return truncated if truncated else text[:cut_char]


def truncate_batch(texts: Sequence[str], tokenizer: Any, max_len: int = MAX_LEN) -> list[str]:
    truncated = []
    changed = 0
    for text in texts:
        out = truncate_with_offset_mapping(str(text), tokenizer=tokenizer, max_len=max_len)
        if out != text:
            changed += 1
        truncated.append(out)
    logging.info(
        "Truncado quirurgico completado: %d/%d textos ajustados a %d tokens.",
        changed,
        len(texts),
        max_len,
    )
    return truncated


def build_predict_proba_fn(text_pipe: Any, ordered_labels: list[str]):
    label_to_pos = {label: idx for idx, label in enumerate(ordered_labels)}

    def predict_proba(texts: Sequence[str] | str) -> np.ndarray:
        batch = [texts] if isinstance(texts, str) else list(texts)
        batch = [str(item) for item in batch]

        raw_outputs = text_pipe(batch)
        if raw_outputs and isinstance(raw_outputs[0], dict):
            raw_outputs = [raw_outputs]

        scores = np.zeros((len(batch), len(ordered_labels)), dtype=np.float64)
        for row_idx, row in enumerate(raw_outputs):
            for item in row:
                label = str(item.get("label"))
                pos = label_to_pos.get(label)
                if pos is not None:
                    scores[row_idx, pos] = float(item.get("score", 0.0))

        row_sums = scores.sum(axis=1, keepdims=True)
        valid_rows = row_sums.squeeze(axis=1) > 0.0
        scores[valid_rows] = scores[valid_rows] / row_sums[valid_rows]
        invalid_rows = ~valid_rows
        if np.any(invalid_rows):
            logging.warning(
                "Pipeline devolvio filas sin scores validos; se aplica distribucion uniforme en %d casos.",
                int(np.sum(invalid_rows)),
            )
            scores[invalid_rows] = 1.0 / max(len(ordered_labels), 1)
        return scores

    return predict_proba


def build_header_html(
    *,
    language: str,
    task: str,
    marker: str,
    rows: list[dict[str, Any]],
    job_id: int,
    job_count: int,
) -> str:
    css = """
<style>
    body {
        font-family: Arial, sans-serif;
        margin: 24px;
        background: #ffffff;
    }
    .summary-card {
        border: 1px solid #d8dee8;
        border-radius: 10px;
        background: #f7f9fc;
        padding: 18px;
        margin-bottom: 16px;
    }
    .summary-meta {
        color: #2f3c4d;
        font-size: 14px;
        margin: 2px 0 12px 0;
    }
    .case-row {
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 14px;
        font-weight: 600;
    }
    .case-row.ok {
        background: #e7f7ed;
        color: #0f7a35;
        border-left: 4px solid #1f9d4d;
    }
    .case-row.fail {
        background: #fdecec;
        color: #b42318;
        border-left: 4px solid #d92d20;
    }
</style>
"""

    body = [
        "<div class='summary-card'>",
        f"<h2>Analisis SHAP - {escape(language.upper())} {escape(task.upper())} ({escape(marker)})</h2>",
        (
            "<p class='summary-meta'>"
            f"Muestras: {len(rows)} | Max tokens: {MAX_LEN} | Job array: {job_id}/{job_count - 1}"
            "</p>"
        ),
    ]

    for item in rows:
        cls = "ok" if item["is_correct"] else "fail"
        badge = "OK" if item["is_correct"] else "FAIL"
        body.append(
            "<div class='case-row {cls}'>[{badge}] Caso {idx} | Real: {real} | Pred: {pred} ({score:.1f}%)</div>".format(
                cls=cls,
                badge=badge,
                idx=item["case_id"],
                real=escape(item["real_label"]),
                pred=escape(item["pred_label"]),
                score=item["pred_score"] * 100.0,
            )
        )

    body.append("</div>")
    return css + "\n".join(body)


def save_html(path: Path, header_html: str, shap_html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><title>SHAP Analysis</title></head><body>"
        + header_html
        + shap_html
        + "</body></html>"
    )
    path.write_text(document, encoding="utf-8")


def main() -> int:
    configure_logging()
    args = parse_args()
    logging.info(
        "Inicio SHAP | task=%s | language=%s | marker=%s | sample_size=%d",
        args.task,
        args.language,
        args.marker,
        args.sample_size,
    )

    model_dir = args.model_root / f"bert_{args.language}_{args.task}_{args.marker}_len256"
    test_path = args.test_root / f"test_{args.language}_e5_markers_{args.marker}.jsonl"

    output_html = args.output_root / f"shap_{args.language}_{args.task}_{args.marker}.html"

    if not model_dir.exists():
        logging.error("No existe el directorio de modelo: %s", model_dir)
        return 1
    if not test_path.exists():
        logging.error("No existe el dataset de test: %s", test_path)
        return 1

    try:
        df = load_dataset(test_path)
    except Exception:
        logging.exception("Error cargando dataset.")
        return 1

    try:
        tokenizer, model = load_model_and_tokenizer(model_dir)
    except Exception:
        logging.exception("Error cargando modelo/tokenizer.")
        return 1

    id2label = resolve_id2label(model, args.task)
    ordered_labels = [id2label[idx] for idx in sorted(id2label.keys())]
    logging.info("Etiquetas finales: %s", ordered_labels)

    df = filter_by_task(df, args.task)
    df = split_for_slurm(df, args.slurm_task_id, args.slurm_task_count, args.seed)
    df_samples = select_samples(df, args.language, args.sample_size, args.seed)
    if df_samples.empty:
        logging.error("No hay muestras disponibles para calcular SHAP.")
        return 1

    raw_texts = df_samples[TEXT_COL].tolist()
    true_labels = df_samples[LABEL_COL].tolist()
    texts_for_shap = truncate_batch(raw_texts, tokenizer, max_len=MAX_LEN)

    device = 0 if torch.cuda.is_available() else -1
    logging.info("Dispositivo pipeline: %s", "cuda:0" if device == 0 else "cpu")

    text_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=MAX_LEN,
        top_k=None,
        function_to_apply="softmax",
    )
    predict_proba = build_predict_proba_fn(text_pipe, ordered_labels)

    logging.info("Computando predicciones para cabecera HTML.")
    probs = predict_proba(texts_for_shap)
    pred_ids = probs.argmax(axis=1)
    pred_scores = probs.max(axis=1)
    pred_labels = [ordered_labels[idx] for idx in pred_ids]

    rows = []
    for idx, (real, pred, score) in enumerate(zip(true_labels, pred_labels, pred_scores), start=1):
        rows.append(
            {
                "case_id": idx,
                "real_label": str(real),
                "pred_label": str(pred),
                "pred_score": float(score),
                "is_correct": str(real).strip().casefold() == str(pred).strip().casefold(),
            }
        )

    logging.info("Inicializando SHAP PartitionExplainer.")
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(
        predict_proba,
        masker=masker,
        output_names=ordered_labels,
        algorithm="partition",
    )

    logging.info("Calculando SHAP para %d muestras.", len(texts_for_shap))
    shap_values = explainer(texts_for_shap)
    shap_html_obj = shap.plots.text(shap_values, display=False)
    shap_html = shap_html_obj.data if hasattr(shap_html_obj, "data") else str(shap_html_obj)

    header_html = build_header_html(
        language=args.language,
        task=args.task,
        marker=args.marker,
        rows=rows,
        job_id=args.slurm_task_id,
        job_count=max(args.slurm_task_count, 1),
    )
    save_html(output_html, header_html, shap_html)

    logging.info("Analisis completado. Salida HTML: %s", output_html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
