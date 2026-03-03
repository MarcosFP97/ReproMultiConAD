import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

ValidatorFn = Callable[[str | None, "PromptSpec"], bool]

IVANOVA_REQUIRED_PASSAGE = (
    "En un lugar de La Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía "
    "un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. "
    "Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los "
    "sábados, lentejas los viernes, algún palomino de añadidura los domingos, consumían las "
    "tres partes de su hacienda."
)

IVANOVA_TOPIC_KEYWORDS = (
    "mancha",
    "tiempo",
    "hidalgo",
    "galgo",
    "corredor",
    "carnero",
    "lentejas",
    "palomino",
    "domingos",
    "hacienda",
)


@dataclass(frozen=True)
class PromptSpec:
    system_template: str
    user_template: str
    roles_required: tuple[str, ...]
    dataset_rules: tuple[str, ...]
    neighbor_header_template: str
    validators: tuple[ValidatorFn, ...]
    basic_user_template: str | None = None
    zero_shot_user_template: str | None = None
    zero_shot_rules: tuple[str, ...] | None = None
    ollama_options: dict[str, Any] = field(default_factory=dict)


def validate_chat(texto: str | None) -> bool:
    """
    Validacion minima CHAT:
    - Debe empezar con *INV: o *PAR:
    - Debe tener al menos una linea *INV: y una *PAR:
    """
    if not texto:
        return False

    lines = [l.strip() for l in texto.strip().splitlines() if l.strip()]
    if not lines:
        return False

    if not re.match(r"^\*(INV|PAR):", lines[0]):
        return False

    # Verificar códigos de tiempo prohibidos
    if "\x15" in texto or re.search(r"\x15\d+_\d+\x15", texto):
        return False

    # Rechazar code / markdown / notebooks
    if "```" in texto:
        return False
    if re.search(r"\.ipynb\b|%matplotlib\b|\bimport\b|\bdef\b|\bclass\b|\btensorflow\b", texto):
        return False
    if re.search(r"^\+{3,}", texto, flags=re.MULTILINE):
        return False

    has_inv = any(l.startswith("*INV:") for l in lines)
    has_par = any(l.startswith("*PAR:") for l in lines)
    return has_inv and has_par


def validate_required_roles(texto: str | None, spec: PromptSpec) -> bool:
    """Valida que estén todos los speakers requeridos y que la primera línea use uno permitido."""
    if not texto:
        return False

    lines = [l.strip() for l in texto.strip().splitlines() if l.strip()]
    if not lines:
        return False
    if not spec.roles_required:
        return True

    first_ok = any(lines[0].startswith(role) for role in spec.roles_required)
    if not first_ok:
        return False

    return all(any(line.startswith(role) for line in lines) for role in spec.roles_required)


def validate_chat_for_spec(texto: str | None, _: PromptSpec) -> bool:
    """Adapter para reutilizar la validación CHAT existente en specs concretos."""
    return validate_chat(texto)


def validate_only_par_lines(texto: str | None, _: PromptSpec) -> bool:
    """Toda línea no vacía debe comenzar exactamente con 'PAR:'."""
    if not texto:
        return False
    for line in texto.splitlines():
        if line.strip() and not line.startswith("PAR:"):
            return False
    return True


def _normalize_for_keyword_match(texto: str) -> str:
    """Normaliza para matching robusto de keywords (case/acento-insensitive)."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")
    return texto


def validate_contains_passage_verbatim(texto: str | None, _: PromptSpec) -> bool:
    """
    Validación temática suave para Ivanova:
    comprueba que el texto trate el contenido de lectura esperado
    mediante coincidencia de palabras clave del pasaje.
    """
    if not texto:
        return False
    normalized = _normalize_for_keyword_match(texto)
    hits = {
        kw for kw in IVANOVA_TOPIC_KEYWORDS
        if re.search(rf"\b{re.escape(kw)}\b", normalized)
    }
    return len(hits) >= 3


def validate_no_other_speakers(texto: str | None, _: PromptSpec) -> bool:
    """Rechaza cualquier etiqueta de speaker distinta de PAR:."""
    if not texto:
        return False
    for line in texto.splitlines():
        if not line.strip():
            continue
        match = re.match(r"^\s*(\w+)\s*:", line)
        if match and match.group(1) != "PAR":
            return False
    return True


def validate_no_timecodes(texto: str | None, _: PromptSpec) -> bool:
    """Rechaza \\x15 y patrones de timecode."""
    if not texto:
        return False
    if "\x15" in texto:
        return False
    if re.search(r"\x15.*?\x15", texto, flags=re.DOTALL):
        return False
    return True


PROMPT_REGISTRY: dict[str, PromptSpec] = {
    # PITT
    "pitt": PromptSpec(
        system_template=(
            "You generate synthetic Pitt Corpus dialogues in CHAT format. "
            "Output only turns from interviewer (*INV:) and participant (*PAR:)."
        ),
        user_template="""
            Use these neighbors as style anchors:

            {selected_transcripts}

            TARGET:
            Diagnosis: {Diagnosis}
            Age: {Age}
            MMSE: {MMSE}
            Gender: {Gender}

            Rules:
            {rules_block}
            """.strip(),
        roles_required=("*INV:", "*PAR:"),
        dataset_rules=(
            "Include both speakers.",
            "MIMIC the broken speech patterns found in the neighbors (do not correct grammar).",
            'YOU MUST INCLUDE CHAT CODES if the neighbors have them. Examples :\n- Pauses: (.) or (..)\n- Repetitions: [/] (e.g., "the [/] the cookie")\n- Revisions: [//] (e.g., "girl [//] boy")\n- Fillers: &-uh, &-um',
            "Keep Cookie Theft context.",
            "CRITICAL: DO NOT include time alignment bullets (e.g., \\x15123_456\\x15). Since this is synthetic text without audio, time codes are invalid.",
        ),
        neighbor_header_template="--- Neighbor {i} | Diagnosis: {Diagnosis} | Age: {Age} | MMSE: {MMSE} | Gender: {Gender} ---",
        validators=(validate_chat_for_spec, validate_required_roles),
        basic_user_template="""
            Based on these similar transcripts:

            {selected_transcripts}

            Generate a new Pitt CHAT transcript for:
            Diagnosis: {Diagnosis}
            Age: {Age}
            MMSE: {MMSE}
            Gender: {Gender}
            """.strip(),
        zero_shot_user_template="""
            TARGET PATIENT:

            Diagnosis: {Diagnosis}
            Age: {Age}
            MMSE: {MMSE}
            Gender: {Gender}

            TASK: Generate a synthetic Pitt Corpus CHAT transcript for the Cookie Theft picture description task.

            Adapt the participant's speech fluency, grammar, and lexical access to strictly match their Diagnosis and MMSE score.

            Rules:

            {rules_block}
        """.strip(),
        zero_shot_rules=(
            "Include both speakers (*INV: and *PAR:).",
            "MIMIC the cognitive decline",
            'YOU MUST INCLUDE CHAT CODES to reflect the cognition: Pauses (.) or (..), Repetitions [/], Revisions [//], and Fillers (&-uh, &-um).',
            "Keep the Cookie Theft picture context",
        ),
        ollama_options={
            "temperature": 1.0,
        },
                ),
    # IVANOVA
    "ivanova": PromptSpec(
        system_template=(
            "Generas transcripciones sintéticas en español para el dataset Ivanova. "
            "Salida estricta: solo líneas del participante con prefijo exacto PAR:."
        ),
        user_template="""
            Usa estos vecinos como anclas de estilo y capacidad cognitiva de lectura:

            {selected_transcripts}

            TARGET:
            Diagnosis: {Diagnosis}
            Age: {Age}
            MMSE: {MMSE}
            Gender: {Gender}

            TAREA:
            Lectura en español de las dos primeras frases de Don Quijote de Cervantes.

            PASAJE OBLIGATORIO (debe aparecer literal y exactamente una vez):
            "{required_passage}"

            Rules:
            {rules_block}
            """.strip(),
        roles_required=("PAR:",),
        dataset_rules=(
            'Solo se permiten líneas de participante; cada línea no vacía DEBE comenzar exactamente con "PAR:"',
            "No incluyas entrevistador ni otros speakers; no agregues prosa fuera de las líneas del transcript.",
            "La salida debe estar en español.",
            "El pasaje debe aparecer literal y en el mismo orden, exactamente una vez.",
            "Modela el nivel cognitivo/fluidez según vecinos: mayor capacidad = lectura más fluida; menor capacidad = más vacilaciones, repeticiones, reparaciones, sustituciones, omisiones y reinicios.",
            "No cambies el pasaje obligatorio; los errores solo pueden aparecer como disfluencias y marcas CHAT *alrededor* del pasaje, sin alterar su texto.",
            "Imita el estilo de disfluencias de los vecinos si existe ((.), (..), [/], [//], &-eh, &-em, etc.) y evita inventar estilos ajenos salvo mínimo necesario.",
            "CRITICAL: prohibidos símbolos de alineación temporal (\\x15 o patrones \\x15...\\x15).",
            "FORMATO/CONTROL: produce entre 2 y 6 líneas PAR: y termina. No añadas líneas extra.",
            "ANTI-LOOP: no repitas el pasaje ni vuelvas a recitarlo; no repitas secuencias largas (>8 palabras) del pasaje.",
            "Prohibidos markdown/code fences, headings, listas, explicaciones y artefactos de notebook/código.",
        ),

        neighbor_header_template="--- Vecino {i} | Diagnostico: {Diagnosis} | Edad: {Age} | MMSE: {MMSE} | Genero: {Gender} ---",
        validators=(
            validate_required_roles,
            validate_only_par_lines,
            validate_contains_passage_verbatim,
            validate_no_other_speakers,
            validate_no_timecodes,
        ),
        basic_user_template="""
            Usa estos vecinos como anclas de estilo:

            {selected_transcripts}

            Genera un transcript SOLO de lectura para:
            Diagnosis: {Diagnosis}
            Age: {Age}
            MMSE: {MMSE}
            Gender: {Gender}

            Salida estricta:
            1. Solo líneas con prefijo exacto "PAR:".
            2. Debe incluir este pasaje una vez:
            "{required_passage}"
            3. Mimetiza la fluidez/disfluencia según los vecinos.
            """.strip(),
        zero_shot_user_template="""
            PACIENTE OBJETIVO:

            Diagnosis: {Diagnosis}
            Age: {Age}
            MMSE: {MMSE}
            Gender: {Gender}

            TAREA:

            Lectura en español de las dos primeras frases de Don Quijote de Cervantes.

            Genera un transcript de cómo leería este paciente exacto el pasaje, adaptando su fluidez a su diagnóstico y puntuación MMSE.

            PASAJE OBLIGATORIO (debe aparecer literal y exactamente una vez):

            "{required_passage}"

            Rules:

            {rules_block}
        """.strip(),
        zero_shot_rules=(
            'Solo se permiten líneas de participante; cada línea no vacía DEBE comenzar exactamente con "PAR:".',
            "No incluyas entrevistador ni otros speakers.",
            "El pasaje debe aparecer literal y en el mismo orden, exactamente una vez.",
            "MODELA EL NIVEL COGNITIVO EN LA TRANSCRIPCIÓN",
            "No cambies el pasaje obligatorio; los errores de lectura solo pueden aparecer como disfluencias y marcas CHAT *alrededor* del pasaje.",   
            "ANTI-LOOP: no repitas el pasaje ni vuelvas a recitarlo.",
            "Prohibidos markdown/code fences, headings o explicaciones.",
        ),
        ollama_options={
            "temperature": 0.25,
            "num_predict": 150,
            "repeat_penalty": 1.25,
            "top_p": 0.9,
            "top_k": 40,
        },
                ),
    # DEFAULT
    "default": PromptSpec(
        system_template=(
            "You generate synthetic interview transcripts. "
            "Use only the required speaker tags and return plain text dialogue."
        ),
        user_template="""
            Use these neighbors as style anchors:

            {selected_transcripts}

            TARGET:
            Diagnosis: {Diagnosis}
            Age: {Age}
            MMSE: {MMSE}
            Gender: {Gender}

            Rules:
            {rules_block}
            """.strip(),
        roles_required=("SpeakerA:", "SpeakerB:"),
        dataset_rules=(
            "Include all required speakers exactly with their prefixes.",
            "Follow the style and lexical complexity of the neighbors.",
            "Do not emit CHAT timing codes or markdown/code blocks.",
            "Stay in the same elicitation context suggested by the neighbors.",
        ),
        neighbor_header_template="[Neighbor {i}] Diagnosis={Diagnosis} Age={Age} MMSE={MMSE} Gender={Gender}",
        validators=(validate_required_roles,),
        basic_user_template="""
            Based on these similar transcripts:

            {selected_transcripts}

            Generate a new transcript for:
            Diagnosis: {Diagnosis}
            Age: {Age}
            MMSE: {MMSE}
            Gender: {Gender}
            """.strip(),
                ),
            }


def get_prompt_spec(dataset_name: str) -> PromptSpec:
    """Devuelve el PromptSpec por dataset con fallback seguro."""
    key = (dataset_name or "").strip().lower()
    if key in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[key]
    if key.startswith("pitt"):
        return PROMPT_REGISTRY["pitt"]
    return PROMPT_REGISTRY["default"]


def format_neighbors(vecinos: pd.DataFrame, spec: PromptSpec) -> str:
    """Formatea vecinos según el encabezado definido por el spec."""
    bloques = []
    for i, r in enumerate(vecinos.itertuples(index=False), 1):
        row = {
            "i": i,
            "Diagnosis": getattr(r, "Diagnosis", ""),
            "Age": getattr(r, "Age", ""),
            "MMSE": getattr(r, "MMSE", ""),
            "Gender": getattr(r, "Gender", ""),
            "text": str(getattr(r, "Text_interviewer_participant", "")),
        }
        header = spec.neighbor_header_template.format(**row)
        bloques.append(f"{header}\n{row['text']}")
    return "\n\n".join(bloques)


def build_messages(
    spec: PromptSpec,
    target: dict,
    vecinos: pd.DataFrame,
    basic: bool = False,
    zero_shot: bool = False,
) -> list[dict]:
    """Construye mensajes para el modelo usando plantillas del PromptSpec."""
    rules = spec.zero_shot_rules if zero_shot else spec.dataset_rules
    rules_block = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules or (), 1))

    fmt_args = {
        "Diagnosis": target.get("Diagnosis", ""),
        "Age": target.get("Age", ""),
        "MMSE": target.get("MMSE", ""),
        "Gender": target.get("Gender", ""),
        "required_passage": IVANOVA_REQUIRED_PASSAGE,
        "rules_block": rules_block,
    }

    if not zero_shot:
        fmt_args["selected_transcripts"] = format_neighbors(vecinos, spec)

    if basic:
        user_template = spec.zero_shot_user_template if zero_shot else (spec.basic_user_template or spec.user_template)
        prompt_user = user_template.format(**fmt_args).strip()
        return [{"role": "user", "content": prompt_user}]

    user_template = spec.zero_shot_user_template if zero_shot else spec.user_template
    prompt_user = user_template.format(**fmt_args).strip()
    prompt_system = spec.system_template.format(**fmt_args).strip()

    if prompt_system:
        return [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ]
    return [{"role": "user", "content": prompt_user}]


def validate_generated_text(texto: str | None, spec: PromptSpec) -> bool:
    """Aplica validadores del dataset en cadena."""
    if not spec.validators:
        return bool(texto and texto.strip())
    return all(validator(texto, spec) for validator in spec.validators)


def build_ollama_options(
    spec: PromptSpec,
    ctx_size: int,
    default_temperature: float = 1.0,
    base_repeat_penalty: float = 1.1,
) -> dict[str, Any]:
    """
    Construye opciones finales para Ollama:
    - Base estable del pipeline.
    - Overrides por dataset definidos en PromptSpec.ollama_options.
    """
    options: dict[str, Any] = {
        "temperature": float(default_temperature),
        "repeat_penalty": float(base_repeat_penalty),
        "num_ctx": int(ctx_size),
    }

    for key, value in spec.ollama_options.items():
        if isinstance(value, bool):
            options[key] = value
        elif isinstance(value, int):
            options[key] = int(value)
        elif isinstance(value, float):
            options[key] = float(value)
        else:
            options[key] = value

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
    spec = get_prompt_spec(dataset_name)
    messages = build_messages(spec, target, vecinos, basic=basic, zero_shot=zero_shot)

    options = build_ollama_options(spec, ctx_size)

    system_txt = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_txt = next((m["content"] for m in messages if m["role"] == "user"), "")

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


def self_check_prompt_specs() -> None:
    """Autocheck rápido de construcción de mensajes sin llamar a Ollama."""
    dummy_target = {
        "Diagnosis": "HC",
        "Age": 71,
        "MMSE": 28,
        "Gender": "F",
    }
    dummy_neighbors = pd.DataFrame(
        [
            {
                "Diagnosis": "HC",
                "Age": 70,
                "MMSE": 29,
                "Gender": "F",
                "Text_interviewer_participant": "*INV: what do you see?\n*PAR: the boy takes cookies.",
            },
            {
                "Diagnosis": "HC",
                "Age": 73,
                "MMSE": 27,
                "Gender": "F",
                "Text_interviewer_participant": "*INV: anything else?\n*PAR: mother is washing dishes.",
            },
        ]
    )

    for ds in ("pitt", "default"):
        spec = get_prompt_spec(ds)
        messages = build_messages(spec, dummy_target, dummy_neighbors)
        assert messages and messages[-1]["role"] == "user", f"Mensajes inválidos para {ds}"
        assert "{selected_transcripts}" not in messages[-1]["content"], f"Template sin formatear en {ds}"
        print(
            f"[SELF-CHECK] dataset={ds} | messages={len(messages)} "
            f"| roles_required={spec.roles_required} | validators={len(spec.validators)}"
        )
        print(f"[SELF-CHECK] user_preview_{ds}: {messages[-1]['content'][:160].replace(chr(10), ' ')}...")

    assert validate_generated_text("*INV: hi\n*PAR: hello", get_prompt_spec("pitt"))
    assert validate_generated_text("SpeakerA: hi\nSpeakerB: hello", get_prompt_spec("default"))

    ivanova_target = {
        "Diagnosis": "MCI",
        "Age": 76,
        "MMSE": 22,
        "Gender": "F",
    }
    ivanova_neighbors = pd.DataFrame(
        [
            {
                "Diagnosis": "MCI",
                "Age": 77,
                "MMSE": 21,
                "Gender": "F",
                "Text_interviewer_participant": "PAR: En un lugar de La Mancha (.) de cuyo nombre no quiero acordarme...",
            }
        ]
    )

    ivanova_spec = get_prompt_spec("ivanova")
    ivanova_messages = build_messages(ivanova_spec, ivanova_target, ivanova_neighbors)
    ivanova_user = next((m["content"] for m in ivanova_messages if m["role"] == "user"), "")

    assert "PAR:" in ivanova_user
    assert IVANOVA_REQUIRED_PASSAGE in ivanova_user

    print(
        f"[SELF-CHECK] dataset=ivanova | messages={len(ivanova_messages)} "
        f"| roles_required={ivanova_spec.roles_required} | validators={len(ivanova_spec.validators)}"
    )
    print("[SELF-CHECK] ivanova user prompt (first 40 lines):")
    for line in ivanova_user.splitlines()[:40]:
        print(line)

    ivanova_ok = f"PAR: {IVANOVA_REQUIRED_PASSAGE}"
    ivanova_bad_prefix = f"*PAR: {IVANOVA_REQUIRED_PASSAGE}"
    ivanova_bad_speaker = f"INV: hola\nPAR: {IVANOVA_REQUIRED_PASSAGE}"
    ivanova_bad_timecode = f"PAR: {IVANOVA_REQUIRED_PASSAGE}\nPAR: \x15123_456\x15"
    ivanova_bad_topic = "PAR: Hoy fuimos al parque y comimos helado bajo el sol."

    assert validate_generated_text(ivanova_ok, ivanova_spec)
    assert not validate_generated_text(ivanova_bad_prefix, ivanova_spec)
    assert not validate_generated_text(ivanova_bad_speaker, ivanova_spec)
    assert not validate_generated_text(ivanova_bad_timecode, ivanova_spec)
    assert not validate_generated_text(ivanova_bad_topic, ivanova_spec)

    sample_ctx = 8192
    pitt_options = build_ollama_options(get_prompt_spec("pitt"), sample_ctx)
    ivanova_options = build_ollama_options(get_prompt_spec("ivanova"), sample_ctx)
    print(f"[SELF-CHECK] options_pitt(ctx={sample_ctx})={pitt_options}")
    print(f"[SELF-CHECK] options_ivanova(ctx={sample_ctx})={ivanova_options}")
    assert abs(float(pitt_options["temperature"]) - 1.0) < 1e-9
    assert abs(float(ivanova_options["temperature"]) - 0.5) < 1e-9
    assert int(pitt_options["num_ctx"]) == sample_ctx
    assert int(ivanova_options["num_ctx"]) == sample_ctx

    print("[SELF-CHECK] Prompt construction and validation passed for pitt/default/ivanova.")
