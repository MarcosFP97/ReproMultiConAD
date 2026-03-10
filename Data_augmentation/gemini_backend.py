import base64
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from prompt_system import PromptSpec, prepare_prompt_payload

GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

COOKIE_THEFT_IMAGE_INLINE: dict[str, str] | None = None


def load_cookie_theft_image_inline(path: Path) -> dict[str, str]:
    """Carga la imagen Cookie Theft y la deja en base64 para Gemini multimodal."""
    global COOKIE_THEFT_IMAGE_INLINE

    if COOKIE_THEFT_IMAGE_INLINE is not None:
        return COOKIE_THEFT_IMAGE_INLINE

    if not path.exists():
        sys.exit(f"No se ha encontrado la imagen Cookie Theft en: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "image/jpeg"

    image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    COOKIE_THEFT_IMAGE_INLINE = {"mime_type": mime_type, "data": image_b64}
    print(f"[INFO] Imagen Cookie Theft cargada: {path} ({mime_type})")
    return COOKIE_THEFT_IMAGE_INLINE


def build_gemini_generation_config(
    spec: PromptSpec,
    default_temperature: float = 0.7,
    default_top_p: float = 0.95,
    default_top_k: int = 40,
    default_max_output_tokens: int = 220,
) -> dict[str, Any]:
    """
    Construye la generationConfig final para Gemini:
    - Base estable del pipeline.
    - Overrides por dataset definidos en PromptSpec.generation_options.
    """
    generation_options = spec.generation_options
    max_output_tokens = generation_options.get(
        "max_output_tokens",
        generation_options.get("num_predict", default_max_output_tokens),  # compat temporal
    )

    return {
        "temperature": float(generation_options.get("temperature", default_temperature)),
        "topP": float(generation_options.get("top_p", default_top_p)),
        "topK": int(generation_options.get("top_k", default_top_k)),
        "maxOutputTokens": int(max_output_tokens),
    }


def extract_text_gemini(payload: dict[str, Any]) -> str | None:
    """Extrae texto plano de la respuesta REST de Gemini."""
    candidates = payload.get("candidates", [])
    if not candidates:
        return None

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
    text = "\n".join(t for t in texts if t).strip()
    return text or None


def call_gemini_api_multimodal(
    *,
    model_name: str,
    api_key: str,
    system_txt: str,
    user_txt: str,
    image_inline: dict[str, str],
    generation_config: dict[str, Any],
    max_retries: int = 3,
) -> str | None:
    """Llama a Gemini via REST con texto + imagen (multimodal)."""
    if not api_key:
        sys.exit(
            "No se encontró la API key de Gemini. "
            "Define GEMINI_API_KEY en el entorno o en el archivo .env."
        )

    endpoint = GEMINI_API_URL_TEMPLATE.format(model=model_name, api_key=api_key)
    payload: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_txt},
                    {"inline_data": image_inline},
                ],
            }
        ],
        "generationConfig": generation_config,
    }

    if system_txt.strip():
        payload["system_instruction"] = {"parts": [{"text": system_txt.strip()}]}

    raw = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(endpoint, data=raw, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
                text = extract_text_gemini(parsed)
                if text:
                    return text
                print(f"[WARN] Respuesta Gemini sin texto útil. Intento {attempt}/{max_retries}.")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            print(f"[WARN] HTTPError Gemini ({e.code}) intento {attempt}/{max_retries}: {body[:400]}")
            if e.code not in (429, 500, 503, 504):
                break
        except Exception as e:
            print(f"[WARN] Error Gemini intento {attempt}/{max_retries}: {e}")

        if attempt < max_retries:
            time.sleep(2 ** attempt)

    return None


def _estimate_tokens(text: str) -> int:
    """Estimación ligera: ~1 token cada 4 caracteres."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def generar_dialogo_paciente_prompt(
    *,
    dataset_name: str,
    target: dict,
    vecinos: pd.DataFrame,
    ctx_size: int,
    basic: bool,
    model_name: str,
    api_key: str,
    cookie_theft_image_path: Path,
    token_counter: Callable[[str], int] | None = None,
    zero_shot: bool = False,
    max_retries: int = 3,
) -> str | None:
    """Construye prompt dataset-aware, llama a Gemini y devuelve el texto generado."""
    prompt_payload = prepare_prompt_payload(
        dataset_name=dataset_name,
        target=target,
        vecinos=vecinos,
        basic=basic,
        zero_shot=zero_shot,
    )
    spec: PromptSpec = prompt_payload["spec"]
    system_txt: str = prompt_payload["system_txt"]
    user_txt: str = prompt_payload["user_txt"]

    counter = token_counter or _estimate_tokens
    sys_tok = counter(system_txt)
    usr_tok = counter(user_txt)
    tot_tok = counter(system_txt + "\n" + user_txt)
    print(f"[TOKENS] system={sys_tok} | user={usr_tok} | total={tot_tok} | ctx_ref={ctx_size}")

    image_inline = load_cookie_theft_image_inline(cookie_theft_image_path)
    generation_config = build_gemini_generation_config(spec)

    return call_gemini_api_multimodal(
        model_name=model_name,
        api_key=api_key,
        system_txt=system_txt,
        user_txt=user_txt,
        image_inline=image_inline,
        generation_config=generation_config,
        max_retries=max_retries,
    )
