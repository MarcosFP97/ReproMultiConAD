"""
Análisis de interpretabilidad SHAP sobre modelos BERT fine-tuned con marcadores CHAT.

Genera por combinación (language, task, marker):
  - un HTML con visualizaciones de atribución usando shap.PartitionExplainer;
  - un Excel con SHAP agregado por token, ablación sobre el test completo e
    intervalos de confianza bootstrap.

Diseñado para ejecutarse como SLURM Job Array: cada job analiza una combinación,
usa el test completo para la ablación y extrae una muestra estratificada para SHAP.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
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

# Regex para detectar el token CHAT relevante según el modo de marcadores
MARKER_REGEX = {
    "pause": r"\[PAUSE\]",
    "rep":   r"\[REP\]",
    "ref":   r"\[REF\]",
    "all":   r"\[PAUSE\]|\[REP\]|\[REF\]",
}
MARKER_TOKENS = {
    "pause": ["[PAUSE]"],
    "rep": ["[REP]"],
    "ref": ["[REF]"],
    "all": ["[PAUSE]", "[REP]", "[REF]"],
}

DEFAULT_ROOT = Path("/mnt/beegfs/groups/irgroup/sara_tfg")
DEFAULT_MODEL_ROOT = DEFAULT_ROOT / "ConvoCognition" / "Experiments" / "BERT_Models"
DEFAULT_TEST_ROOT = DEFAULT_ROOT / "jsonl" / "markers_collections"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "results" / "BERT_tokenizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisis SHAP para BERT con truncado robusto.")
    parser.add_argument("--marker", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["binary", "multiclass"])
    parser.add_argument("--sample-size", type=int, default=30, help="Numero de ejemplos para SHAP.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestreo reproducible.")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
        help="Iteraciones bootstrap para intervalos de confianza de las metricas agregadas.",
    )
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


def select_samples(df: pd.DataFrame, language: str, sample_size: int, seed: int, marker: str = "all") -> pd.DataFrame:
    work = df.copy()
    regex = MARKER_REGEX.get(marker, MARKER_REGEX["all"])
    work["has_chat_marker"] = work[TEXT_COL].str.contains(regex, regex=True, na=False)
    labels = [label for label in TASK_LABELS["multiclass"] if label in set(work[LABEL_COL])]
    if not labels:
        return work.head(0).copy()

    target_size = min(sample_size, len(work))
    base_quota, remainder = divmod(target_size, len(labels))
    selected_parts: list[pd.DataFrame] = []
    selected_index: set[int] = set()

    # Reparte el presupuesto entre clases. Dentro de cada clase prioriza
    # transcripciones que contienen el marcador analizado.
    for label_idx, label in enumerate(labels):
        quota = base_quota + (1 if label_idx < remainder else 0)
        label_rows = work[work[LABEL_COL] == label]
        with_marker = label_rows[label_rows["has_chat_marker"]]
        without_marker = label_rows[~label_rows["has_chat_marker"]]

        marker_take = min(quota, len(with_marker))
        selected_marker = _sample_part(with_marker, marker_take, seed + label_idx * 10)
        remaining = quota - len(selected_marker)
        selected_plain = _sample_part(without_marker, remaining, seed + label_idx * 10 + 1)

        selected = pd.concat([selected_marker, selected_plain], axis=0)
        if not selected.empty:
            selected_parts.append(selected)
            selected_index.update(selected.index.to_list())
        logging.info(
            "Seleccion clase '%s': %d/%d muestras (%d con marcador).",
            label,
            len(selected),
            quota,
            len(selected_marker),
        )

    result = pd.concat(selected_parts, axis=0) if selected_parts else work.head(0).copy()

    # Completa el presupuesto si alguna clase no tenia suficientes ejemplos.
    remaining = target_size - len(result)
    if remaining > 0:
        pool = work[~work.index.isin(selected_index)].copy()
        marker_pool = pool[pool["has_chat_marker"]]
        extra_marker = _sample_part(marker_pool, min(remaining, len(marker_pool)), seed + 100)
        remaining -= len(extra_marker)
        plain_pool = pool[
            ~pool.index.isin(extra_marker.index) & ~pool["has_chat_marker"]
        ]
        extra_plain = _sample_part(plain_pool, remaining, seed + 101)
        extra = pd.concat([extra_marker, extra_plain], axis=0)
        result = pd.concat([result, extra], axis=0)

    return result.sample(frac=1, random_state=seed).head(target_size)


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


def remove_marker_token(text: str, marker_token: str) -> str:
    ablated = re.sub(re.escape(marker_token), " ", str(text))
    return re.sub(r"\s+", " ", ablated).strip()


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    iterations: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1 or iterations <= 0:
        value = float(clean[0])
        return value, value

    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for idx in range(iterations):
        means[idx] = rng.choice(clean, size=clean.size, replace=True).mean()

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def summarize_metric(
    values: Sequence[float],
    *,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {
            "Mean": float("nan"),
            "Median": float("nan"),
            "Mean_abs": float("nan"),
            "Positive_pct": float("nan"),
            "CI95_low": float("nan"),
            "CI95_high": float("nan"),
        }

    ci_low, ci_high = bootstrap_mean_ci(
        clean,
        seed=seed,
        iterations=bootstrap_iterations,
    )
    return {
        "Mean": float(clean.mean()),
        "Median": float(np.median(clean)),
        "Mean_abs": float(np.abs(clean).mean()),
        "Positive_pct": float((clean > 0).mean() * 100.0),
        "CI95_low": ci_low,
        "CI95_high": ci_high,
    }


def build_ablation_metrics(
    *,
    df: pd.DataFrame,
    tokenizer: Any,
    predict_proba: Any,
    ordered_labels: list[str],
    marker_tokens: list[str],
    seed: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mide el efecto local controlado de retirar cada marcador del mismo texto.

    Delta_probability > 0 significa que la presencia del token aumenta la
    probabilidad de esa clase respecto al texto ablatido.
    """
    sample_rows: list[dict[str, Any]] = []

    for marker_token in marker_tokens:
        token_mask = df[TEXT_COL].str.contains(re.escape(marker_token), regex=True, na=False)
        token_df = df[token_mask].copy()
        if token_df.empty:
            logging.warning("No hay muestras con %s para la ablacion.", marker_token)
            continue

        original_texts = truncate_batch(token_df[TEXT_COL].tolist(), tokenizer, max_len=MAX_LEN)
        visible_positions = [
            idx for idx, text in enumerate(original_texts) if marker_token in str(text)
        ]
        if not visible_positions:
            logging.warning(
                "Las muestras con %s lo contienen fuera de la ventana de %d tokens.",
                marker_token,
                MAX_LEN,
            )
            continue
        token_df = token_df.iloc[visible_positions].copy()
        original_texts = [original_texts[idx] for idx in visible_positions]
        ablated_texts = [
            truncate_with_offset_mapping(
                remove_marker_token(text, marker_token),
                tokenizer,
                max_len=MAX_LEN,
            )
            for text in original_texts
        ]
        original_probs = predict_proba(original_texts)
        ablated_probs = predict_proba(ablated_texts)
        original_pred = original_probs.argmax(axis=1)
        ablated_pred = ablated_probs.argmax(axis=1)

        for row_pos, (_, source_row) in enumerate(token_df.iterrows()):
            true_label = str(source_row[LABEL_COL])
            occurrence_count = str(original_texts[row_pos]).count(marker_token)
            for class_idx, class_label in enumerate(ordered_labels):
                delta = float(original_probs[row_pos, class_idx] - ablated_probs[row_pos, class_idx])
                sample_rows.append(
                    {
                        "Marker_token": marker_token,
                        "Sample_index": source_row.name,
                        "True_label": true_label,
                        "Class": class_label,
                        "Occurrence_count": occurrence_count,
                        "Probability_with_token": float(original_probs[row_pos, class_idx]),
                        "Probability_without_token": float(ablated_probs[row_pos, class_idx]),
                        "Delta_probability": delta,
                        "Is_true_class": class_label == true_label,
                        "Prediction_with_token": ordered_labels[int(original_pred[row_pos])],
                        "Prediction_without_token": ordered_labels[int(ablated_pred[row_pos])],
                        "Prediction_changed": bool(original_pred[row_pos] != ablated_pred[row_pos]),
                    }
                )

    samples_df = pd.DataFrame(sample_rows)
    if samples_df.empty:
        return samples_df, pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    scopes = [("Overall", samples_df)]
    scopes.extend(
        (f"True_{label}", samples_df[samples_df["True_label"] == label])
        for label in ordered_labels
    )
    for scope_idx, (scope, scope_df) in enumerate(scopes):
        for marker_token in marker_tokens:
            marker_df = scope_df[scope_df["Marker_token"] == marker_token]
            for class_idx, class_label in enumerate(ordered_labels):
                group = marker_df[marker_df["Class"] == class_label]
                if group.empty:
                    continue
                stats = summarize_metric(
                    group["Delta_probability"],
                    seed=seed + scope_idx * 100 + class_idx,
                    bootstrap_iterations=bootstrap_iterations,
                )
                summary_rows.append(
                    {
                        "Scope": scope,
                        "Marker_token": marker_token,
                        "Class": class_label,
                        "Samples_with_token": int(group["Sample_index"].nunique()),
                        "Total_occurrences": int(
                            group.drop_duplicates("Sample_index")["Occurrence_count"].sum()
                        ),
                        "Mean_delta_probability": stats["Mean"],
                        "Median_delta_probability": stats["Median"],
                        "Mean_abs_delta_probability": stats["Mean_abs"],
                        "Positive_delta_pct": stats["Positive_pct"],
                        "Mean_delta_CI95_low": stats["CI95_low"],
                        "Mean_delta_CI95_high": stats["CI95_high"],
                        "Prediction_changed_pct": float(
                            group.drop_duplicates("Sample_index")["Prediction_changed"].mean() * 100.0
                        ),
                    }
                )

    return samples_df, pd.DataFrame(summary_rows)


def normalize_shap_token(token: Any) -> str:
    normalized = str(token).strip()
    for prefix in ("Ġ", "▁"):
        normalized = normalized.removeprefix(prefix)
    return normalized


def sample_shap_matrix(shap_values: Any, sample_idx: int, num_classes: int) -> np.ndarray:
    matrix = np.asarray(shap_values.values[sample_idx], dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if matrix.ndim != 2:
        raise RuntimeError(f"Shape SHAP inesperado para muestra {sample_idx}: {matrix.shape}")
    if matrix.shape[1] == num_classes:
        return matrix
    if matrix.shape[0] == num_classes:
        return matrix.T
    raise RuntimeError(
        f"No se pudo identificar el eje de clases en SHAP: {matrix.shape}, clases={num_classes}"
    )


def build_shap_token_metrics(
    *,
    shap_values: Any,
    true_labels: list[str],
    ordered_labels: list[str],
    marker_tokens: list[str],
    seed: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict[str, Any]] = []

    for sample_idx, true_label in enumerate(true_labels):
        tokens = [normalize_shap_token(token) for token in shap_values.data[sample_idx]]
        matrix = sample_shap_matrix(shap_values, sample_idx, len(ordered_labels))
        usable = min(len(tokens), matrix.shape[0])
        tokens = tokens[:usable]
        matrix = matrix[:usable]

        for marker_token in marker_tokens:
            positions = [idx for idx, token in enumerate(tokens) if token == marker_token]
            if not positions:
                continue
            token_values = matrix[positions, :]
            summed_values = token_values.sum(axis=0)
            mean_values = token_values.mean(axis=0)
            for class_idx, class_label in enumerate(ordered_labels):
                other_values = np.delete(summed_values, class_idx)
                margin = float(summed_values[class_idx] - np.max(other_values)) if other_values.size else float(summed_values[class_idx])
                sample_rows.append(
                    {
                        "Marker_token": marker_token,
                        "Sample_id": sample_idx + 1,
                        "True_label": str(true_label),
                        "Class": class_label,
                        "Occurrence_count": len(positions),
                        "SHAP_sum": float(summed_values[class_idx]),
                        "SHAP_mean_per_occurrence": float(mean_values[class_idx]),
                        "SHAP_class_margin": margin,
                        "Is_true_class": class_label == str(true_label),
                    }
                )

    samples_df = pd.DataFrame(sample_rows)
    if samples_df.empty:
        return samples_df, pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    scopes = [("Overall", samples_df)]
    scopes.extend(
        (f"True_{label}", samples_df[samples_df["True_label"] == label])
        for label in ordered_labels
    )
    for scope_idx, (scope, scope_df) in enumerate(scopes):
        for marker_token in marker_tokens:
            marker_df = scope_df[scope_df["Marker_token"] == marker_token]
            for class_idx, class_label in enumerate(ordered_labels):
                group = marker_df[marker_df["Class"] == class_label]
                if group.empty:
                    continue
                shap_stats = summarize_metric(
                    group["SHAP_sum"],
                    seed=seed + 1000 + scope_idx * 100 + class_idx,
                    bootstrap_iterations=bootstrap_iterations,
                )
                margin_stats = summarize_metric(
                    group["SHAP_class_margin"],
                    seed=seed + 2000 + scope_idx * 100 + class_idx,
                    bootstrap_iterations=bootstrap_iterations,
                )
                summary_rows.append(
                    {
                        "Scope": scope,
                        "Marker_token": marker_token,
                        "Class": class_label,
                        "Samples_with_token": int(group["Sample_id"].nunique()),
                        "Total_occurrences": int(
                            group.drop_duplicates("Sample_id")["Occurrence_count"].sum()
                        ),
                        "Mean_SHAP_sum": shap_stats["Mean"],
                        "Median_SHAP_sum": shap_stats["Median"],
                        "Mean_abs_SHAP_sum": shap_stats["Mean_abs"],
                        "Positive_SHAP_pct": shap_stats["Positive_pct"],
                        "Mean_SHAP_CI95_low": shap_stats["CI95_low"],
                        "Mean_SHAP_CI95_high": shap_stats["CI95_high"],
                        "Mean_class_margin": margin_stats["Mean"],
                        "Positive_class_margin_pct": margin_stats["Positive_pct"],
                    }
                )

    return samples_df, pd.DataFrame(summary_rows)


def save_metrics_excel(
    path: Path,
    *,
    run_summary: pd.DataFrame,
    ablation_samples: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    shap_samples: pd.DataFrame,
    shap_summary: pd.DataFrame,
    diagnostic_summary: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        run_summary.to_excel(writer, index=False, sheet_name="run_summary")
        ablation_summary.to_excel(writer, index=False, sheet_name="ablation_summary")
        ablation_samples.to_excel(writer, index=False, sheet_name="ablation_samples")
        shap_summary.to_excel(writer, index=False, sheet_name="shap_summary")
        shap_samples.to_excel(writer, index=False, sheet_name="shap_samples")
        diagnostic_summary.to_excel(writer, index=False, sheet_name="diagnostic_effect")


def build_diagnostic_effect_summary(
    *,
    ablation_summary: pd.DataFrame,
    shap_summary: pd.DataFrame,
    ordered_labels: list[str],
) -> pd.DataFrame:
    """
    Resume si el token refuerza la clase verdadera para cada diagnostico.

    Esta es la tabla mas directa para afirmaciones como: "[REP] aumenta de media
    la probabilidad de MCI en muestras cuyo diagnostico real es MCI".
    """
    rows: list[dict[str, Any]] = []
    marker_tokens = sorted(
        set(ablation_summary.get("Marker_token", pd.Series(dtype=str)).dropna())
        | set(shap_summary.get("Marker_token", pd.Series(dtype=str)).dropna())
    )

    def select_summary(
        summary: pd.DataFrame,
        *,
        scope: str,
        marker_token: str,
        label: str,
    ) -> pd.DataFrame:
        required = {"Scope", "Marker_token", "Class"}
        if summary.empty or not required.issubset(summary.columns):
            return summary.head(0)
        return summary[
            (summary["Scope"] == scope)
            & (summary["Marker_token"] == marker_token)
            & (summary["Class"] == label)
        ]

    for marker_token in marker_tokens:
        for label in ordered_labels:
            scope = f"True_{label}"
            ablation = select_summary(
                ablation_summary,
                scope=scope,
                marker_token=marker_token,
                label=label,
            )
            shap_group = select_summary(
                shap_summary,
                scope=scope,
                marker_token=marker_token,
                label=label,
            )
            if ablation.empty and shap_group.empty:
                continue

            row: dict[str, Any] = {
                "Marker_token": marker_token,
                "Diagnosis": label,
            }
            if not ablation.empty:
                item = ablation.iloc[0]
                row.update(
                    {
                        "Ablation_samples": item["Samples_with_token"],
                        "Mean_delta_true_probability": item["Mean_delta_probability"],
                        "Delta_true_probability_CI95_low": item["Mean_delta_CI95_low"],
                        "Delta_true_probability_CI95_high": item["Mean_delta_CI95_high"],
                        "Positive_delta_true_pct": item["Positive_delta_pct"],
                    }
                )
            if not shap_group.empty:
                item = shap_group.iloc[0]
                row.update(
                    {
                        "SHAP_samples": item["Samples_with_token"],
                        "Mean_SHAP_true_class": item["Mean_SHAP_sum"],
                        "SHAP_true_class_CI95_low": item["Mean_SHAP_CI95_low"],
                        "SHAP_true_class_CI95_high": item["Mean_SHAP_CI95_high"],
                        "Positive_SHAP_true_pct": item["Positive_SHAP_pct"],
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


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
    output_metrics = (
        args.output_root / f"shap_metrics_{args.language}_{args.task}_{args.marker}.xlsx"
    )

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
    df_samples = select_samples(df, args.language, args.sample_size, args.seed, marker=args.marker)
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
    marker_tokens = MARKER_TOKENS[args.marker]

    logging.info(
        "Calculando ablacion sobre el test completo para: %s.",
        ", ".join(marker_tokens),
    )
    ablation_samples, ablation_summary = build_ablation_metrics(
        df=df,
        tokenizer=tokenizer,
        predict_proba=predict_proba,
        ordered_labels=ordered_labels,
        marker_tokens=marker_tokens,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )

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
    shap_samples, shap_summary = build_shap_token_metrics(
        shap_values=shap_values,
        true_labels=[str(label) for label in true_labels],
        ordered_labels=ordered_labels,
        marker_tokens=marker_tokens,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    diagnostic_summary = build_diagnostic_effect_summary(
        ablation_summary=ablation_summary,
        shap_summary=shap_summary,
        ordered_labels=ordered_labels,
    )
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
    run_summary = pd.DataFrame(
        [
            {
                "Language": args.language,
                "Task": args.task,
                "Marker_mode": args.marker,
                "Marker_tokens": "|".join(marker_tokens),
                "SHAP_sample_size": len(texts_for_shap),
                "Test_rows": len(df),
                "Seed": args.seed,
                "Bootstrap_iterations": args.bootstrap_iterations,
                "Model_dir": str(model_dir),
                "Test_path": str(test_path),
                "HTML_path": str(output_html),
            }
        ]
    )
    save_metrics_excel(
        output_metrics,
        run_summary=run_summary,
        ablation_samples=ablation_samples,
        ablation_summary=ablation_summary,
        shap_samples=shap_samples,
        shap_summary=shap_summary,
        diagnostic_summary=diagnostic_summary,
    )

    logging.info("Analisis completado. Salida HTML: %s", output_html)
    logging.info("Metricas agregadas guardadas en: %s", output_metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
