import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =========================
# FIXED CONFIGURATION
# =========================
# Edit these values if needed. The script runs with no CLI parameters.
DEFAULT_MODEL_PATH = "/mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_Models/bert_ivanova_100_patient_classifier_len256"
DEFAULT_DATA_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection"
DEFAULT_DATASETS = [
    "test_baycrest.jsonl",
    "test_delaware.jsonl",
    "test_kempler.jsonl",
    "test_lu.jsonl",
    "test_taukadial.jsonl",
    "test_vas.jsonl",
    "test_wls.jsonl",
]
DEFAULT_FALLBACK_LABELS = ["Dementia", "MCI", "HC"]
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_SAVE_JSON_PATH: Optional[str] = None


def validate_config(
    max_length: int,
    batch_size: int,
    datasets: List[str],
    fallback_labels: List[str],
) -> None:
    if max_length <= 0:
        raise ValueError("DEFAULT_MAX_LENGTH must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("DEFAULT_BATCH_SIZE must be a positive integer.")
    if not datasets:
        raise ValueError("DEFAULT_DATASETS cannot be empty.")
    if not fallback_labels:
        raise ValueError("DEFAULT_FALLBACK_LABELS cannot be empty.")


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _labels_are_generic(labels: List[str]) -> bool:
    if not labels:
        return True
    return all(re.fullmatch(r"LABEL_\d+", label.upper()) is not None for label in labels)


def normalize_label(value: Any, fallback_labels: Optional[List[str]] = None) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, int):
        if fallback_labels and 0 <= value < len(fallback_labels):
            return fallback_labels[value]
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.isdigit():
        idx = int(text)
        if fallback_labels and 0 <= idx < len(fallback_labels):
            return fallback_labels[idx]

    key = re.sub(r"[^a-z0-9]+", "", text.lower())

    direct_map = {
        "dementia": "Dementia",
        "dem": "Dementia",
        "ad": "Dementia",
        "alz": "Dementia",
        "alzheimer": "Dementia",
        "alzheimers": "Dementia",
        "probabledementia": "Dementia",
        "mci": "MCI",
        "amci": "MCI",
        "mcip": "MCI",
        "mildcognitiveimpairment": "MCI",
        "mildcognitiveimpair": "MCI",
        "mildimpairment": "MCI",
        "hc": "HC",
        "control": "HC",
        "healthycontrol": "HC",
        "healthycontrols": "HC",
        "normal": "HC",
        "cn": "HC",
        "cognitivelynormal": "HC",
        "nci": "HC",
    }
    if key in direct_map:
        return direct_map[key]

    if "dement" in key or "alzh" in key:
        return "Dementia"
    if "mci" in key or "mildcognitive" in key:
        return "MCI"
    if "healthy" in key or "control" in key or "normal" in key:
        return "HC"

    return None


def resolve_model_labels(model, fallback_labels: List[str]) -> Tuple[List[str], List[str]]:
    config = model.config
    id2label = getattr(config, "id2label", None)
    num_labels = int(getattr(config, "num_labels", 0) or 0)

    parsed: Dict[int, str] = {}
    if isinstance(id2label, dict):
        for key, value in id2label.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            parsed[idx] = str(value)

    if num_labels <= 0:
        if parsed:
            num_labels = max(parsed.keys()) + 1
        else:
            num_labels = len(fallback_labels)

    raw_labels = [parsed.get(i, f"LABEL_{i}") for i in range(num_labels)]

    if not parsed or _labels_are_generic(raw_labels):
        raw_labels = [
            fallback_labels[i] if i < len(fallback_labels) else f"LABEL_{i}"
            for i in range(num_labels)
        ]

    normalized_labels: List[str] = []
    for idx, label in enumerate(raw_labels):
        normalized = normalize_label(label, fallback_labels=fallback_labels)
        if normalized is None:
            normalized = fallback_labels[idx] if idx < len(fallback_labels) else label
        normalized_labels.append(normalized)

    return raw_labels, normalized_labels


def load_model_and_tokenizer(
    model_path: str,
    fallback_labels: List[str],
    device: torch.device,
):
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path not found: {model_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"Model path is not a directory: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    raw_labels, normalized_labels = resolve_model_labels(model, fallback_labels)
    return tokenizer, model, raw_labels, normalized_labels


def predict_text(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> Tuple[int, List[float]]:
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)
        logits = output.logits.squeeze(0)
        probabilities = torch.softmax(logits, dim=-1)

    pred_idx = int(torch.argmax(probabilities).item())
    return pred_idx, probabilities.detach().cpu().tolist()


def predict_text_batch(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> List[int]:
    predictions: List[int] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            batch_preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            predictions.extend(int(x) for x in batch_preds)

    return predictions


def _read_jsonl_records(
    dataset_path: Path,
    fallback_labels: List[str],
) -> Tuple[List[Dict[str, Any]], Counter]:
    records: List[Dict[str, Any]] = []
    issue_counts: Counter = Counter()

    with dataset_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            issue_counts["total_lines"] += 1
            raw_line = line.strip()

            if not raw_line:
                issue_counts["empty_line"] += 1
                continue

            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError:
                issue_counts["malformed_json"] += 1
                continue

            if not isinstance(item, dict):
                issue_counts["non_object_json"] += 1
                continue

            if "Text_interviewer_participant" not in item:
                issue_counts["missing_text_field"] += 1
                continue

            if "diagnosis" not in item:
                issue_counts["missing_diagnosis_field"] += 1
                continue

            text = str(item.get("Text_interviewer_participant", "")).strip()
            if not text:
                issue_counts["empty_text"] += 1
                continue

            diagnosis_raw = item.get("diagnosis")
            diagnosis_norm = normalize_label(diagnosis_raw, fallback_labels=fallback_labels)
            if diagnosis_norm is None:
                issue_counts["unknown_diagnosis_label"] += 1
                continue

            records.append(
                {
                    "line_number": line_number,
                    "text": text,
                    "diagnosis_raw": diagnosis_raw,
                    "diagnosis": diagnosis_norm,
                }
            )

    issue_counts["valid_records"] = len(records)
    issue_counts["skipped_records"] = issue_counts["total_lines"] - len(records)
    return records, issue_counts


def _ordered_distribution(labels: List[str], label_order: List[str]) -> Dict[str, int]:
    counts = Counter(labels)
    ordered: Dict[str, int] = {}

    for label in label_order:
        ordered[label] = int(counts.get(label, 0))

    extra_labels = sorted([label for label in counts.keys() if label not in ordered])
    for label in extra_labels:
        ordered[label] = int(counts[label])

    return ordered


def evaluate_dataset(
    dataset_path: Path,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
    canonical_labels: List[str],
    model_idx_to_label: List[str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "dataset_name": dataset_path.name,
        "dataset_path": str(dataset_path),
    }

    if not dataset_path.exists():
        result["error"] = f"File not found: {dataset_path}"
        return result
    if not dataset_path.is_file():
        result["error"] = f"Path is not a file: {dataset_path}"
        return result

    try:
        records, issue_counts = _read_jsonl_records(dataset_path, fallback_labels=canonical_labels)
    except Exception as exc:
        result["error"] = f"Failed to read dataset: {exc}"
        return result

    result["issue_counts"] = dict(issue_counts)

    if not records:
        result.update(
            {
                "num_valid": 0,
                "accuracy": None,
                "true_distribution": {},
                "pred_distribution": {},
                "confusion_labels": canonical_labels,
                "confusion_matrix": [],
                "classification_report": "No valid samples to evaluate.",
                "y_true": [],
                "y_pred": [],
            }
        )
        return result

    texts = [record["text"] for record in records]
    y_true = [record["diagnosis"] for record in records]

    pred_indices = predict_text_batch(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    y_pred: List[str] = []
    unknown_pred_count = 0
    for pred_idx in pred_indices:
        if 0 <= pred_idx < len(model_idx_to_label):
            pred_label = model_idx_to_label[pred_idx]
        else:
            pred_label = "UNKNOWN"

        pred_norm = normalize_label(pred_label, fallback_labels=canonical_labels)
        if pred_norm is None:
            pred_norm = "UNKNOWN"
            unknown_pred_count += 1

        y_pred.append(pred_norm)

    metric_labels = list(canonical_labels)
    if "UNKNOWN" in y_pred or "UNKNOWN" in y_true:
        metric_labels.append("UNKNOWN")

    accuracy = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=metric_labels)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=metric_labels,
        target_names=metric_labels,
        zero_division=0,
    )

    result.update(
        {
            "num_valid": len(records),
            "accuracy": accuracy,
            "unknown_pred_count": unknown_pred_count,
            "true_distribution": _ordered_distribution(y_true, metric_labels),
            "pred_distribution": _ordered_distribution(y_pred, metric_labels),
            "confusion_labels": metric_labels,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_text,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    return result


def _format_distribution(dist: Dict[str, int]) -> str:
    if not dist:
        return "{}"
    return ", ".join(f"{label}: {count}" for label, count in dist.items())


def _format_confusion_matrix(labels: List[str], matrix: List[List[int]]) -> str:
    if not labels or not matrix:
        return "No confusion matrix available."

    width = max(10, max(len(label) for label in labels) + 2)
    header = "true\\pred".ljust(width) + "".join(label.ljust(width) for label in labels)
    rows = [header]

    for label, row in zip(labels, matrix):
        row_values = "".join(str(value).ljust(width) for value in row)
        rows.append(label.ljust(width) + row_values)

    return "\n".join(rows)


def print_dataset_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"Dataset: {report.get('dataset_name', 'unknown')}")
    print(f"Path: {report.get('dataset_path', 'unknown')}")

    if "error" in report:
        print(f"Status: ERROR - {report['error']}")
        return

    issue_counts = report.get("issue_counts", {})
    total_lines = issue_counts.get("total_lines", 0)
    valid_records = report.get("num_valid", 0)
    skipped_records = issue_counts.get("skipped_records", 0)

    print(f"Total lines read: {total_lines}")
    print(f"Valid samples evaluated: {valid_records}")
    print(f"Skipped samples: {skipped_records}")

    skip_reason_keys = [
        "empty_line",
        "malformed_json",
        "non_object_json",
        "missing_text_field",
        "missing_diagnosis_field",
        "empty_text",
        "unknown_diagnosis_label",
    ]
    printed_skip_lines = False
    for key in skip_reason_keys:
        value = issue_counts.get(key, 0)
        if value:
            if not printed_skip_lines:
                print("Skip details:")
                printed_skip_lines = True
            print(f"  - {key}: {value}")

    accuracy = report.get("accuracy")
    if accuracy is None:
        print("Accuracy: N/A (no valid samples)")
        return

    print(f"Accuracy: {accuracy:.4f}")

    print("True distribution:")
    print(f"  {_format_distribution(report.get('true_distribution', {}))}")

    print("Predicted distribution:")
    print(f"  {_format_distribution(report.get('pred_distribution', {}))}")

    unknown_pred_count = report.get("unknown_pred_count", 0)
    if unknown_pred_count:
        print(f"Unknown predicted labels after normalization: {unknown_pred_count}")

    print("Confusion matrix (rows=true, cols=pred):")
    print(_format_confusion_matrix(report.get("confusion_labels", []), report.get("confusion_matrix", [])))

    print("Classification report:")
    print(report.get("classification_report", ""))


def build_global_summary(
    reports: List[Dict[str, Any]],
    canonical_labels: List[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "datasets": [],
        "overall": None,
    }

    all_true: List[str] = []
    all_pred: List[str] = []

    for report in reports:
        item = {
            "dataset_name": report.get("dataset_name"),
            "dataset_path": report.get("dataset_path"),
            "num_valid": report.get("num_valid", 0),
            "accuracy": report.get("accuracy"),
            "error": report.get("error"),
            "true_distribution": report.get("true_distribution", {}),
            "pred_distribution": report.get("pred_distribution", {}),
            "issue_counts": report.get("issue_counts", {}),
        }
        summary["datasets"].append(item)

        if report.get("error") is None and report.get("y_true") and report.get("y_pred"):
            all_true.extend(report["y_true"])
            all_pred.extend(report["y_pred"])

    if not all_true:
        return summary

    metric_labels = list(canonical_labels)
    if "UNKNOWN" in all_true or "UNKNOWN" in all_pred:
        metric_labels.append("UNKNOWN")

    overall_accuracy = float(accuracy_score(all_true, all_pred))
    overall_cm = confusion_matrix(all_true, all_pred, labels=metric_labels).tolist()
    overall_report_text = classification_report(
        all_true,
        all_pred,
        labels=metric_labels,
        target_names=metric_labels,
        zero_division=0,
    )

    summary["overall"] = {
        "num_samples": len(all_true),
        "accuracy": overall_accuracy,
        "true_distribution": _ordered_distribution(all_true, metric_labels),
        "pred_distribution": _ordered_distribution(all_pred, metric_labels),
        "confusion_labels": metric_labels,
        "confusion_matrix": overall_cm,
        "classification_report": overall_report_text,
    }
    return summary


def print_global_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "#" * 100)
    print("GLOBAL SUMMARY")

    datasets = summary.get("datasets", [])
    if not datasets:
        print("No datasets processed.")
        return

    print("Per-dataset comparison:")
    for item in datasets:
        name = item.get("dataset_name", "unknown")
        num_valid = item.get("num_valid", 0)
        error = item.get("error")
        if error:
            print(f"  - {name}: ERROR ({error})")
            continue

        accuracy = item.get("accuracy")
        accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"

        pred_dist = item.get("pred_distribution", {})
        majority_label = "N/A"
        if pred_dist:
            majority_label = max(pred_dist.items(), key=lambda kv: kv[1])[0]

        print(
            f"  - {name}: n={num_valid}, accuracy={accuracy_str}, "
            f"predicted_majority={majority_label}, pred_dist=[{_format_distribution(pred_dist)}]"
        )

    overall = summary.get("overall")
    if not overall:
        print("\nNo global metrics available (no valid predictions across datasets).")
        return

    print("\nOverall across all datasets:")
    print(f"  Samples: {overall['num_samples']}")
    print(f"  Accuracy: {overall['accuracy']:.4f}")
    print(f"  True distribution: {_format_distribution(overall['true_distribution'])}")
    print(f"  Pred distribution: {_format_distribution(overall['pred_distribution'])}")

    print("\nOverall confusion matrix (rows=true, cols=pred):")
    print(_format_confusion_matrix(overall["confusion_labels"], overall["confusion_matrix"]))

    print("Overall classification report:")
    print(overall["classification_report"])


def save_summary_json(summary: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    model_path = DEFAULT_MODEL_PATH
    data_dir = Path(DEFAULT_DATA_DIR)
    datasets = list(DEFAULT_DATASETS)
    max_length = DEFAULT_MAX_LENGTH
    batch_size = DEFAULT_BATCH_SIZE
    fallback_labels = list(DEFAULT_FALLBACK_LABELS)
    save_json_path = DEFAULT_SAVE_JSON_PATH

    try:
        validate_config(
            max_length=max_length,
            batch_size=batch_size,
            datasets=datasets,
            fallback_labels=fallback_labels,
        )

        device = select_device()
        print(f"Device: {device}")

        tokenizer, model, model_raw_labels, model_normalized_labels = load_model_and_tokenizer(
            model_path=model_path,
            fallback_labels=fallback_labels,
            device=device,
        )

        print(f"Model path: {model_path}")
        print(f"Data dir: {data_dir}")
        print(f"Datasets: {datasets}")
        print(f"Model labels (raw): {model_raw_labels}")
        print(f"Model labels (normalized): {model_normalized_labels}")

        reports: List[Dict[str, Any]] = []

        for dataset_name in datasets:
            dataset_path = data_dir / dataset_name
            report = evaluate_dataset(
                dataset_path=dataset_path,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
                batch_size=batch_size,
                canonical_labels=fallback_labels,
                model_idx_to_label=model_normalized_labels,
            )
            reports.append(report)
            print_dataset_report(report)

        summary = build_global_summary(reports, canonical_labels=fallback_labels)
        print_global_summary(summary)

        if save_json_path:
            save_summary_json(summary, save_json_path)
            print(f"\nJSON summary saved to: {save_json_path}")

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
