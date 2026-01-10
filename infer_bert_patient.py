import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# =========================
# CONFIG
# =========================
MODEL_DIR = "/mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_Models/bert_patient_classifier_len256"
MAX_LEN = 256
BATCH_SIZE = 8


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_artifacts(model_dir: str, device: torch.device):
    # Modelo + tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Label encoder (guardado con torch.save)
    le_path = os.path.join(model_dir, "label_encoder.pth")
    label_encoder = torch.load(le_path, map_location="cpu", weights_only=False)

    if not isinstance(label_encoder, LabelEncoder):
        raise TypeError("El archivo label_encoder.pth no parece ser un sklearn.preprocessing.LabelEncoder.")

    return model, tokenizer, label_encoder


@torch.no_grad()
def predict_texts(texts, model, tokenizer, label_encoder, device, max_len=256, batch_size=8):
    """
    Devuelve una lista de dicts por texto:
      {
        'text': ...,
        'pred_id': int,
        'pred_label': str,
        'probs': {label: prob, ...},
        'confidence': float
      }
    """
    results = []
    model.eval()

    # labels en el orden del modelo (id -> nombre)
    id2label = {i: c for i, c in enumerate(label_encoder.classes_)}

    # batching simple
    for start in range(0, len(texts), batch_size):
        batch_texts = [str(t) for t in texts[start:start + batch_size]]

        enc = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, num_labels]
        probs = torch.softmax(logits, dim=-1)  # [B, num_labels]

        pred_ids = torch.argmax(probs, dim=-1).tolist()
        probs_list = probs.detach().cpu().tolist()

        for txt, pid, pvec in zip(batch_texts, pred_ids, probs_list):
            prob_dict = {id2label[i]: float(pvec[i]) for i in range(len(pvec))}
            pred_label = id2label[pid]
            confidence = float(max(pvec))

            results.append({
                "text": txt,
                "pred_id": pid,
                "pred_label": pred_label,
                "probs": prob_dict,
                "confidence": confidence,
            })

    return results


def main():
    device = get_device()
    print(f"[DEVICE] {device}")

    model, tokenizer, label_encoder = load_artifacts(MODEL_DIR, device)

    print("[LABELS] id -> class")
    for i, c in enumerate(label_encoder.classes_):
        print(f"  {i} -> {c}")

    # =========================
    # EJEMPLOS (puedes cambiar/añadir)
    # =========================
    demo_texts = [
        # ===== AD-like (1) =====
        "INV: Please describe everything you see happening in the picture. PAR: Uh… there’s a woman… she’s at the… the sink. The water is, um, it’s going over. The kids are there and the boy is on the… the chair thing. He’s getting cookies, I think. The girl wants one. The mother doesn’t see it.",

        # ===== AD-like (2) =====
        "INV: What is going on in this picture? PAR: The boy is… is taking the cookies from the jar. He’s standing on something and it’s not safe. The lady is washing dishes and the water keeps running. I can’t remember what else is there.",

        # ===== AD-like (3) =====
        "INV: Describe the scene in as much detail as you can. PAR: Well, the sink is… overflowing and she’s there washing. The boy is reaching up and giving the cookie to the girl. The chair might fall. I know there’s more but I can’t think of it.",

        # ===== AD-like (4) =====
        "INV: Tell me everything you see. PAR: Um… the children are taking… the biscuits, cookies. The boy is up on the thing you stand on. The water is coming out of the sink. The woman doesn’t notice. I’m… I’m not sure about the rest.",

        # ===== AD-like (5) =====
        "INV: What do you notice first? PAR: I see the boy. He’s getting the cookies. The girl is waiting. The mother is doing dishes and the water is going down. The words aren’t coming to me very well.",

        # ===== HC-like (6) =====
        "INV: Please describe everything you see happening in the picture. PAR: The mother is standing at the sink washing dishes, but the water is overflowing because she left the tap on. Behind her, a boy is standing on a stool to reach a cookie jar on a high shelf, and he’s taking cookies out. His sister is reaching up to get one. The stool looks unstable, so he could fall.",

        # ===== HC-like (7) =====
        "INV: What is going on here? PAR: Two children are sneaking cookies while their mother is distracted at the sink. The faucet is running and water is spilling onto the floor. The boy is balancing on a stool and handing a cookie to the girl.",

        # ===== HC-like (8) =====
        "INV: Describe the picture in detail. PAR: The kitchen scene shows a woman washing dishes with the sink overflowing. A boy is reaching into a cookie jar on a high shelf while standing on a stool, and his sister is waiting beside him. The open cabinets and spilled water suggest a bit of chaos.",

        # ===== HC-like (9) =====
        "INV: Tell me what you see. PAR: The mother is focused on washing dishes and doesn’t notice the water spilling over the sink. Meanwhile, her son is taking cookies from a jar while standing on a stool, and his sister is reaching for one. The stool looks like it could tip over.",

        # ===== HC-like (10) =====
        "INV: Summarize what is happening. PAR: The children are stealing cookies from a jar while their mother is busy at the sink. The water is overflowing, and the boy is in danger of falling from the stool."
    ]


    preds = predict_texts(demo_texts, model, tokenizer, label_encoder, device, max_len=MAX_LEN, batch_size=BATCH_SIZE)

    print("\n--- PREDICTIONS ---")
    for i, r in enumerate(preds, 1):
        print(f"\n[{i}] pred_label={r['pred_label']}  confidence={r['confidence']:.3f}")
        # imprime probs ordenadas de mayor a menor
        probs_sorted = sorted(r["probs"].items(), key=lambda x: x[1], reverse=True)
        print(" probs:", ", ".join([f"{k}:{v:.3f}" for k, v in probs_sorted]))
        print(" text :", r["text"][:250], "..." if len(r["text"]) > 250 else "")


if __name__ == "__main__":
    main()
