import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = "/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/BERT_Models/512FINAL_multiclass_pitt_patient_classifier"
DATA_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/jsonl/individual_sets"
DATASETS = [
    "test_pitt.jsonl", # hc/dementia/mci
    "test_delaware.jsonl", # hc/mci
    "test_lu.jsonl", # hc/dementia/mci
    "test_taukadial.jsonl", # hc/mci
    "test_vas.jsonl", # hc/dementia/mci
    "test_wls.jsonl", # hc/dementia
]
MAX_LENGTH = 256
BATCH_SIZE = 16
FALLBACK_ID2LABEL = {0: "Dementia", 1: "HC", 2: "MCI"}


def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(MODEL_PATH)
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Model path does not exist or is not a directory: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    config_id2label = getattr(model.config, "id2label", None)
    parsed_id2label = {}

    if isinstance(config_id2label, dict):
        for key, value in config_id2label.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            parsed_id2label[idx] = str(value)

    use_fallback = False

    if len(parsed_id2label) < 3:
        use_fallback = True
    else:
        values = [parsed_id2label.get(i, "") for i in range(3)]
        if values == ["LABEL_0", "LABEL_1", "LABEL_2"]:
            use_fallback = True

    if use_fallback:
        id2label = FALLBACK_ID2LABEL.copy()
    else:
        id2label = {i: parsed_id2label.get(i, FALLBACK_ID2LABEL[i]) for i in range(3)}

    return tokenizer, model, device, id2label


def read_jsonl_records(dataset_path):
    texts = []
    y_true = []

    stats = {
        "total_lines": 0,
        "valid_samples": 0,
        "omitted_samples": 0,
        "malformed_json": 0,
        "missing_text_or_label": 0,
        "empty_text": 0,
    }

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            stats["total_lines"] += 1
            line = line.strip()

            if not line:
                stats["omitted_samples"] += 1
                stats["malformed_json"] += 1
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                stats["omitted_samples"] += 1
                stats["malformed_json"] += 1
                continue

            if "Text_interviewer_participant" not in item or "Diagnosis" not in item:
                stats["omitted_samples"] += 1
                stats["missing_text_or_label"] += 1
                continue

            text = str(item["Text_interviewer_participant"]).strip()
            diagnosis = item["Diagnosis"]

            if not text:
                stats["omitted_samples"] += 1
                stats["empty_text"] += 1
                continue

            texts.append(text)
            y_true.append(diagnosis)
            stats["valid_samples"] += 1

    return texts, y_true, stats


def predict_batch(texts, tokenizer, model, device, id2label):
    y_pred = []

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start : start + BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            pred_indices = torch.argmax(outputs.logits, dim=-1).detach().cpu().tolist()

            for idx in pred_indices:
                label = id2label.get(idx, f"LABEL_{idx}")
                y_pred.append(label)

    return y_pred


def evaluate_dataset(dataset_name, tokenizer, model, device, id2label):
    dataset_path = Path(DATA_DIR) / dataset_name

    report = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "error": None,
        "total_lines": 0,
        "valid_samples": 0,
        "omitted_samples": 0,
        "accuracy": None,
        "true_distribution": {},
        "pred_distribution": {},
        "confusion_labels": [],
        "confusion_matrix": [],
        "classification_report": "",
        "y_true": [],
        "y_pred": [],
        "details": {},
    }

    if not dataset_path.exists():
        report["error"] = f"File not found: {dataset_path}"
        return report

    if not dataset_path.is_file():
        report["error"] = f"Not a file: {dataset_path}"
        return report

    try:
        texts, y_true, stats = read_jsonl_records(dataset_path)
    except Exception as e:
        report["error"] = f"Failed reading JSONL: {e}"
        return report

    report["total_lines"] = stats["total_lines"]
    report["valid_samples"] = stats["valid_samples"]
    report["omitted_samples"] = stats["omitted_samples"]
    report["details"] = stats

    if not y_true:
        return report

    y_pred = predict_batch(texts, tokenizer, model, device, id2label)

    base_labels = ["Dementia", "MCI", "HC"]
    extra_labels = sorted(set(y_true + y_pred) - set(base_labels))
    metric_labels = base_labels + extra_labels

    report["accuracy"] = float(accuracy_score(y_true, y_pred))
    report["true_distribution"] = dict(Counter(y_true))
    report["pred_distribution"] = dict(Counter(y_pred))
    report["confusion_labels"] = metric_labels
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=metric_labels).tolist()
    report["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=metric_labels,
        target_names=metric_labels,
        zero_division=0,
    )
    report["y_true"] = y_true
    report["y_pred"] = y_pred

    return report


def print_confusion_matrix(labels, matrix):
    if not labels or not matrix:
        print("No confusion matrix available.")
        return

    width = max(10, max(len(lbl) for lbl in labels) + 2)
    header = "true\\pred".ljust(width) + "".join(lbl.ljust(width) for lbl in labels)
    print(header)

    for label, row in zip(labels, matrix):
        row_str = "".join(str(v).ljust(width) for v in row)
        print(label.ljust(width) + row_str)


def print_dataset_report(report):
    print("\n" + "=" * 100)
    print(f"Dataset: {report['dataset_name']}")
    print(f"Path: {report['dataset_path']}")

    if report["error"]:
        print(f"Error: {report['error']}")
        return

    print(f"Total lines: {report['total_lines']}")
    print(f"Valid samples: {report['valid_samples']}")
    print(f"Omitted samples: {report['omitted_samples']}")

    details = report.get("details", {})
    if details:
        print(
            "Omitted breakdown: "
            f"malformed_json={details.get('malformed_json', 0)}, "
            f"missing_text_or_label={details.get('missing_text_or_label', 0)}, "
            f"empty_text={details.get('empty_text', 0)}"
        )

    if report["accuracy"] is None:
        print("Accuracy: N/A (no valid samples)")
        return

    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"True distribution: {report['true_distribution']}")
    print(f"Pred distribution: {report['pred_distribution']}")

    print("Confusion matrix (rows=true, cols=pred):")
    print_confusion_matrix(report["confusion_labels"], report["confusion_matrix"])

    print("Classification report:")
    print(report["classification_report"])


def main():
    tokenizer, model, device, id2label = load_model_and_tokenizer()

    print(f"Device: {device}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Datasets: {DATASETS}")
    print(f"id2label used: {id2label}")
    
    print("[LABEL MAPPING]")
    for idx in sorted(id2label):
        print(f"  {idx} -> {id2label[idx]}")

    all_true = []
    all_pred = []

    for dataset_name in DATASETS:
        report = evaluate_dataset(dataset_name, tokenizer, model, device, id2label)
        print_dataset_report(report)

        if report["error"] is None and report["y_true"]:
            all_true.extend(report["y_true"])
            all_pred.extend(report["y_pred"])

    print("\n" + "#" * 100)
    print("GLOBAL SUMMARY")

    if not all_true:
        print("No valid samples available across all datasets.")
        return

    base_labels = ["Dementia", "MCI", "HC"]
    extra_labels = sorted(set(all_true + all_pred) - set(base_labels))
    metric_labels = base_labels + extra_labels

    global_accuracy = float(accuracy_score(all_true, all_pred))
    global_true_dist = dict(Counter(all_true))
    global_pred_dist = dict(Counter(all_pred))
    global_cm = confusion_matrix(all_true, all_pred, labels=metric_labels).tolist()

    print(f"Global samples: {len(all_true)}")
    print(f"Global accuracy: {global_accuracy:.4f}")
    print(f"Global true distribution: {global_true_dist}")
    print(f"Global pred distribution: {global_pred_dist}")

    print("Global confusion matrix (rows=true, cols=pred):")
    print_confusion_matrix(metric_labels, global_cm)


if __name__ == "__main__":
    main()
