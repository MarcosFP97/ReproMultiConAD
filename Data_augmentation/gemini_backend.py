import io 
import PIL.Image
import base64
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from google import genai
from google.genai import types

from prompt_system import PromptSpec, prepare_prompt_payload


COOKIE_THEFT_IMAGE_INLINE: dict[str, str] | None = None

def load_cookie_theft_image_inline(path: Path) -> dict[str, str]:
    global COOKIE_THEFT_IMAGE_INLINE
    
    if COOKIE_THEFT_IMAGE_INLINE is not None:
        return COOKIE_THEFT_IMAGE_INLINE

    if not path.exists():
        sys.exit(f"No se ha encontrado la imagen Cookie Theft en: {path}")

    # CONVERSIÓN FORZADA A JPEG
    print(f"[INFO] Convirtiendo {path.name} a JPEG para compatibilidad...")
    with PIL.Image.open(path) as img:
        img = img.convert("RGB") # Asegura que no haya canales extra
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    COOKIE_THEFT_IMAGE_INLINE = {"mime_type": "image/jpeg", "data": image_b64}
    print(f"[INFO] Imagen Cookie Theft cargada y convertida a image/jpeg")
    return COOKIE_THEFT_IMAGE_INLINE


def build_gemini_generation_config(
    spec: PromptSpec,
    override_max_output_tokens: int,
    default_temperature: float = 0.7,
    default_top_p: float = 0.95,
    default_top_k: int = 40,
) -> types.GenerateContentConfig:
    generation_options = spec.generation_options

    return types.GenerateContentConfig(
        temperature=float(generation_options.get("temperature", default_temperature)),
        top_p=float(generation_options.get("top_p", default_top_p)),
        top_k=int(generation_options.get("top_k", default_top_k)),
        max_output_tokens=int(override_max_output_tokens),
    )


def extract_text_from_response(response) -> str | None:
    text = getattr(response, "text", None)
    if text:
        return text.strip() or None
    return None


def call_gemini_api(
    *,
    model_name: str,
    api_key: str,
    system_txt: str,
    user_txt: str,
    image_inline: dict[str, str] | None,
    generation_config: types.GenerateContentConfig,
    max_retries: int = 3,
) -> str | None:
    if not api_key:
        sys.exit(
            "No se encontró la API key de Gemini. "
            "Define GEMINI_API_KEY en el entorno o en el archivo .env."
        )

    client = genai.Client(api_key=api_key)

    contents = [types.Part.from_text(text=user_txt)]
    if image_inline is not None:
        contents.append(
            types.Part.from_bytes(
                data=base64.b64decode(image_inline["data"]),
                mime_type=image_inline["mime_type"],
            )
        )

    config = generation_config
    if system_txt.strip():
        config = types.GenerateContentConfig(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            max_output_tokens=generation_config.max_output_tokens,
            system_instruction=system_txt.strip(),
        )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            text = extract_text_from_response(response)
            if text:
                return text
            print(f"[WARN] Respuesta Gemini sin texto útil. Intento {attempt}/{max_retries}.")
        except Exception as e:
            print(f"[WARN] Error Gemini intento {attempt}/{max_retries}: {e}")

        if attempt < max_retries:
            time.sleep(2 ** attempt)

    return None


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def generar_dialogo_paciente_prompt(
    *,
    dataset_name: str,
    target: dict,
    vecinos: pd.DataFrame,
    basic: bool,
    model_name: str,
    api_key: str,
    cookie_theft_image_path: Path,
    max_output_tokens: int,
    token_counter: Callable[[str], int] | None = None,
    zero_shot: bool = False,
    max_retries: int = 3,
) -> str | None:
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
    print(f"[TOKENS] system={sys_tok} | user={usr_tok} | total={tot_tok}")

    image_inline = None
    if spec.uses_cookie_theft_image:
        image_inline = load_cookie_theft_image_inline(cookie_theft_image_path)
    generation_config = build_gemini_generation_config(
        spec,
        override_max_output_tokens=max_output_tokens,
    )

    return call_gemini_api(
        model_name=model_name,
        api_key=api_key,
        system_txt=system_txt,
        user_txt=user_txt,
        image_inline=image_inline,
        generation_config=generation_config,
        max_retries=max_retries,
    )
