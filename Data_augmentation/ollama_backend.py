import sys
from typing import Any

import pandas as pd

from prompt_system import PromptSpec, prepare_prompt_payload


def build_ollama_options(spec: PromptSpec,ctx_size: int,default_temperature: float = 1.0,base_repeat_penalty: float = 1.1,) -> dict[str, Any]:
    """
    Construye opciones finales para Ollama:
    - Base estable del pipeline.
    - Overrides por dataset definidos en PromptSpec.generation_options.
    """
    options: dict[str, Any] = {
        "temperature": float(default_temperature),
        "repeat_penalty": float(base_repeat_penalty),
        "num_ctx": int(ctx_size),
    }

    # Mapeo semántico -> Ollama.
    # max_output_tokens es agnóstico; en Ollama equivale a num_predict.
    for key, value in spec.generation_options.items():
        target_key = "num_predict" if key == "max_output_tokens" else key
        if isinstance(value, bool):
            options[target_key] = value
        elif isinstance(value, int):
            options[target_key] = int(value)
        elif isinstance(value, float):
            options[target_key] = float(value)
        else:
            options[target_key] = value

    # Garantizamos tipos serializables en num_ctx incluso con overrides.
    options["num_ctx"] = int(options.get("num_ctx", ctx_size))
    if "temperature" in options:
        options["temperature"] = float(options["temperature"])
    if "repeat_penalty" in options:
        options["repeat_penalty"] = float(options["repeat_penalty"])

    return options


def generar_dialogo_paciente_prompt(
    dataset_name: str,
    target: dict,
    vecinos: pd.DataFrame,
    ctx_size: int,
    basic: bool,
    model_name: str,
    tokenizer,
    zero_shot: bool = False,
) -> str | None:
    """Construye prompt dataset-aware, llama a Ollama y devuelve el texto generado."""
    prompt_payload = prepare_prompt_payload(
        dataset_name=dataset_name,
        target=target,
        vecinos=vecinos,
        basic=basic,
        zero_shot=zero_shot,
    )
    spec: PromptSpec = prompt_payload["spec"]
    messages: list[dict[str, str]] = prompt_payload["messages"]
    system_txt: str = prompt_payload["system_txt"]
    user_txt: str = prompt_payload["user_txt"]

    options = build_ollama_options(spec, ctx_size)

    sys_tok = len(tokenizer.encode(system_txt, add_special_tokens=False))
    usr_tok = len(tokenizer.encode(user_txt, add_special_tokens=False))
    both_txt = system_txt + "\n" + user_txt
    tot_tok = len(tokenizer.encode(both_txt, add_special_tokens=False))

    print(f"[TOKENS] system={sys_tok} | user={usr_tok} | total={tot_tok} | ctx={ctx_size}")

    try:
        from ollama import chat
    except Exception as e:
        sys.exit(f"No se pudo importar ollama: {e}")

    response = chat(model=model_name, messages=messages, options=options)
    return response.message.content.strip()
