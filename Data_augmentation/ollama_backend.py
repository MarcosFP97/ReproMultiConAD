"""
Ollama backend for synthetic transcription generation.

Translates the generation options from PromptSpec to Ollama-compatible parameters
(num_predict, num_ctx, repeat_penalty) and calls the local chat API.
"""

import sys
from typing import Any

import pandas as pd

from prompt_system import PromptSpec, prepare_prompt_payload


def get_ollama_generation_options(spec: PromptSpec) -> dict[str, Any]:
    """Merges common and Ollama-specific options defined in the PromptSpec."""
    return {
        **spec.generation_options,
        **spec.ollama_generation_options,
    }


def resolve_ollama_num_predict(spec: PromptSpec, default_num_predict: int = 1024) -> int:
    """Returns the actual output budget that Ollama will apply as num_predict."""
    merged_options = get_ollama_generation_options(spec)
    raw_value = merged_options.get("max_output_tokens", merged_options.get("num_predict", default_num_predict))
    return int(raw_value)


def build_ollama_options(spec: PromptSpec,num_ctx: int,default_temperature: float = 1.0,base_repeat_penalty: float = 1.1,) -> dict[str, Any]:
    """
    Builds the final options for Ollama:
    - Stable base for the pipeline.
    - Common overrides in PromptSpec.generation_options.
    - Ollama-specific overrides in PromptSpec.ollama_generation_options.
    """
    options: dict[str, Any] = {
        "temperature": float(default_temperature),
        "repeat_penalty": float(base_repeat_penalty),
        "num_ctx": int(num_ctx),
    }

    # Semantic mapping -> Ollama.
    # max_output_tokens is generic; in Ollama it corresponds to num_predict.
    merged_options = get_ollama_generation_options(spec)
    for key, value in merged_options.items():
        target_key = "num_predict" if key == "max_output_tokens" else key
        if isinstance(value, bool):
            options[target_key] = value
        elif isinstance(value, int):
            options[target_key] = int(value)
        elif isinstance(value, float):
            options[target_key] = float(value)
        else:
            options[target_key] = value

    if "num_predict" not in options:
        options["num_predict"] = resolve_ollama_num_predict(spec)

    # Ensure serializable types for num_ctx even with overrides.
    options["num_ctx"] = int(options.get("num_ctx", num_ctx))
    options["num_predict"] = int(options["num_predict"])
    if "temperature" in options:
        options["temperature"] = float(options["temperature"])
    if "repeat_penalty" in options:
        options["repeat_penalty"] = float(options["repeat_penalty"])

    return options


def generate_dialog_pacient_prompt(
    dataset_name: str,
    target: dict,
    vecinos: pd.DataFrame,
    num_ctx: int,
    basic: bool,
    model_name: str,
    tokenizer,
    zero_shot: bool = False,
) -> str | None:
    """Builds a dataset-aware prompt, calls Ollama, and returns the generated text."""
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

    options = build_ollama_options(spec, num_ctx)

    sys_tok = len(tokenizer.encode(system_txt, add_special_tokens=False))
    usr_tok = len(tokenizer.encode(user_txt, add_special_tokens=False))
    both_txt = system_txt + "\n" + user_txt
    tot_tok = len(tokenizer.encode(both_txt, add_special_tokens=False))

    print(
        f"[TOKENS] system={sys_tok} | user={usr_tok} | total={tot_tok} "
        f"| num_ctx_applied={options['num_ctx']} | num_predict={options.get('num_predict', 'default')}"
    )

    try:
        from ollama import chat
    except Exception as e:
        sys.exit(f"Could not import ollama: {e}")

    response = chat(model=model_name, messages=messages, options=options)
    return response.message.content.strip()
