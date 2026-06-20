"""
Especificación de prompts y validación por dataset para la generación de transcripciones sintéticas.

Define los PromptSpec de cada corpus (Pitt, Ivanova, default), validadores de formato CHAT,
utilidades de formateo de vecinos KNN y la función central prepare_prompt_payload()
usada por ambos backends (Gemini y Ollama).
"""

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
    generation_options: dict[str, Any] = field(default_factory=dict)
    ollama_generation_options: dict[str, Any] = field(default_factory=dict)
    uses_cookie_theft_image: bool = False


def validate_chat(texto: str | None) -> bool:
    """
    Validacion minima CHAT:
    - Debe empezar con INV: o PAR:
    - Debe tener al menos una linea INV: y una PAR:
    """
    if not texto:
        return False

    lines = [l.strip() for l in texto.strip().splitlines() if l.strip()]
    if not lines:
        return False

    if not re.match(r"^\*?(INV|PAR):", lines[0]):
        return False

    # Rechazar code / markdown / notebooks
    if "```" in texto:
        return False
    if re.search(r"\.ipynb\b|%matplotlib\b|\bimport\b|\bdef\b|\bclass\b|\btensorflow\b", texto):
        return False
    if re.search(r"^\+{3,}", texto, flags=re.MULTILINE):
        return False

    has_inv = any(re.match(r"^\*?INV:", l) for l in lines)
    has_par = any(re.match(r"^\*?PAR:", l) for l in lines)
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

    first_ok = any(re.match(rf"^\*?{re.escape(role)}", lines[0]) for role in spec.roles_required)
    if not first_ok:
        return False

    return all(
        any(re.match(rf"^\*?{re.escape(role)}", line) for line in lines)
        for role in spec.roles_required
    )


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


def validate_no_speaker_labels(texto: str | None, _: PromptSpec) -> bool:
    """Rechaza etiquetas de hablante (INV:/PAR:/SpeakerX:): el corpus real usa solo ' : '."""
    if not texto:
        return False
    return re.search(r"\b(INV|PAR|SpeakerA|SpeakerB)\s*:", texto) is None


def validate_no_code_artifacts(texto: str | None, _: PromptSpec) -> bool:
    """Rechaza markdown/code fences y artefactos de notebook/código."""
    if not texto or not texto.strip():
        return False
    if "```" in texto:
        return False
    if re.search(r"\.ipynb\b|%matplotlib\b|\bimport\b|\bdef\b|\bclass\b|\btensorflow\b", texto):
        return False
    if re.search(r"^\+{3,}", texto, flags=re.MULTILINE):
        return False
    return True



PROMPT_REGISTRY: dict[str, PromptSpec] = {
    # PITT
    "pitt": PromptSpec(
        system_template=(
            "You generate synthetic Pitt Corpus Cookie Theft transcripts in CLAN/CHAT .cha output style. "
            "Write ONE continuous lowercase line, separating every turn with a space-colon-space ' : ' and putting spaces around punctuation ( . and ? ). "
            "Return only the transcript line, nothing else."
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
        roles_required=(),
        dataset_rules=(
            "Imitate the surface format of the neighbors: lowercase, turns separated by ' : ', spaces around punctuation ( . and ? ).",
            "Reproduce the SAME CHAT markers the neighbors use, with the same notation: pauses (.) (..) (...), repetitions 'word [/] word', revisions 'word [//] word', fillers &-uh &-um, overlaps +<, interruptions +/. and +/?, exclamations [+ exc], and www for unintelligible interviewer speech.",
            "Calibrate disfluency density to the cognitive level: HC/MMSE>=25 mostly fluent with few markers; MCI/MMSE 20-24 moderate hesitations; Dementia/MMSE<20 frequent pauses, repetitions, word-finding errors and broken syntax.",
            "MIMIC the broken speech patterns of the matching neighbors; do NOT correct grammar.",
            "Keep the Cookie Theft scene (boy, stool, cookie jar, mother, sink overflowing, dishes).",
            "Match the length of the neighbors (roughly 120-220 words).",
            "No markdown, headings, explanations, bullets, quotes or code.",
        ),
        neighbor_header_template="--- Neighbor {i} | Diagnosis: {Diagnosis} | Age: {Age} | MMSE: {MMSE} | Gender: {Gender} ---",
        validators=(validate_no_code_artifacts, validate_no_speaker_labels),
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
            "Write ONE continuous lowercase line, separating turns with ' : ' and putting spaces around punctuation ( . and ? ).",
            "Start with a short interviewer turn (e.g. ': all that you see going on in the picture ?') and then the participant's description.",
            "Include CHAT markers to reflect cognition: pauses (.) (..) (...), repetitions 'word [/] word', revisions 'word [//] word', fillers &-uh &-um, overlaps +<, interruptions +/. and +/?, exclamations [+ exc].",
            "Calibrate disfluency density to the cognitive level: HC/MMSE>=25 mostly fluent; MCI/MMSE 20-24 moderate hesitations; Dementia/MMSE<20 frequent pauses, repetitions, word-finding errors and broken syntax.",
            "Keep the Cookie Theft scene (boy, stool, cookie jar, mother, sink overflowing, dishes); aim for roughly 120-200 words.",
            "Return only the transcript. No explanation, no markdown, no bullets, no title, no code.",
        ),
        generation_options={
            "temperature": 0.8,
        },
        ollama_generation_options={
            "max_output_tokens": 320,
            "repeat_penalty": 1.15,
        },
        uses_cookie_theft_image=True,
                ),
    # IVANOVA
    "ivanova": PromptSpec(
        system_template=(
            "Generas transcripciones sintéticas en español del dataset Ivanova: una lectura en voz alta del inicio del Quijote, imitando el formato de salida CHAT/.cha. "
            "Escribe la transcripción separando los fragmentos con ' : ', conservando mayúscula inicial de frase y comas, como en una lectura real. "
            "Devuelve solo la transcripción."
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
            El paciente INTENTA leer en voz alta las dos primeras frases del Quijote, cometiendo errores reales de lectura según su nivel cognitivo (NO copiar literal).

            PASAJE DE REFERENCIA (es lo que el paciente trata de leer, no a transcribir tal cual):
            "{required_passage}"

            Rules:
            {rules_block}
            """.strip(),
        roles_required=(),
        dataset_rules=(
            "Imita el formato de los vecinos: fragmentos separados por ' : ', con mayúsculas iniciales de frase y comas.",
            "La salida debe estar en español.",
            "El paciente NO reproduce el pasaje literal: introduce errores reales de lectura escalados por gravedad (sustituciones de palabras, palabras partidas/repetidas como 'domin domingo', reinicios, omisiones, parafasias), tal como hacen los vecinos.",
            "Usa SOLO la marca CHAT [/] para repeticiones (ej. 'de [/] de los'), como los vecinos. NO uses (.), (..), [//], &-eh ni &-em (no aparecen en este corpus); las demás disfluencias van escritas como palabras realmente mal leídas.",
            "Calibra la cantidad de errores: MMSE alto (27-30)/HC = lectura casi correcta con algún tropiezo; MCI/MMSE 20-26 = vacilaciones y alguna sustitución; Demencia/MMSE<20 = lectura muy fragmentada, sustituciones frecuentes y pérdida de estructura.",
            "Mantén el tema y el orden del pasaje (lugar de la Mancha, hidalgo, lanza, adarga, rocín, galgo, olla, vaca, carnero, salpicón, sábados, lentejas, viernes, palomino, domingos, hacienda).",
            "Longitud parecida a los vecinos (aprox. 70-100 palabras). No recites el pasaje entero más de una vez.",
            "Prohibidos markdown/code fences, encabezados, listas, explicaciones, acotaciones (*pausa*) y artefactos de notebook/código.",
        ),

        neighbor_header_template="--- Vecino {i} | Diagnostico: {Diagnosis} | Edad: {Age} | MMSE: {MMSE} | Genero: {Gender} ---",
        validators=(
            validate_no_speaker_labels,
            validate_contains_passage_verbatim,
            validate_no_code_artifacts,
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
            1. Fragmentos separados por ' : ', con mayúsculas iniciales y comas, como una lectura real.
            2. El paciente intenta leer este pasaje cometiendo errores reales según su nivel (NO copiar literal):
            "{required_passage}"
            3. Mimetiza la fluidez/disfluencia según los vecinos (solo marca [/] para repeticiones).
            """.strip(),
        zero_shot_user_template="""
            PACIENTE OBJETIVO:

            Diagnosis: {Diagnosis}
            Age: {Age}
            MMSE: {MMSE}
            Gender: {Gender}

            TAREA:
            El paciente intenta leer en voz alta las dos primeras frases del Quijote.

            PASAJE DE REFERENCIA (lo que trata de leer, NO a copiar literal):
            "{required_passage}"

            OBJETIVO DE LA SIMULACIÓN:
            Genera la transcripción de CÓMO leería este paciente el pasaje en TIEMPO REAL.
            Debes tropezar, dudar y equivocarte MIENTRAS lees, adaptando la cantidad de errores al nivel cognitivo (Diagnosis y MMSE).

            Rules:
            {rules_block}
        """.strip(),
        zero_shot_rules=(
            "Escribe la transcripción separando los fragmentos con ' : ', con mayúsculas iniciales de frase y comas, como una lectura real.",
            "PROHIBIDO usar acotaciones teatrales (ej. *pausa*, *murmura*, *lee*) o comentarios del paciente; solo el intento de lectura.",
            "El paciente NO copia el pasaje literal: incrusta errores reales DURANTE la lectura (sustituciones, palabras partidas/repetidas, reinicios, omisiones).",
            "Usa SOLO la marca CHAT [/] para repeticiones (ej. 'un [/] un hidalgo). NO uses (.), [//], &-eh ni &-em; las demás disfluencias van como palabras realmente mal leídas.",
            "MODELA EL NIVEL COGNITIVO: MMSE alto (27-30)/HC = lectura casi perfecta con algún tropiezo; MMSE 20-26/MCI = vacilaciones y alguna sustitución; MMSE<20/Demencia = lectura muy fragmentada, sustituciones frecuentes y pérdida de estructura.",
            "Avanza hasta 'tres partes de su hacienda' y termina ahí; no recites el pasaje más de una vez.",
        ),
        generation_options={
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
        },
        ollama_generation_options={
            "max_output_tokens": 220,
            "repeat_penalty": 1.25,
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


def prepare_prompt_payload(
    *,
    target: dict,
    vecinos: pd.DataFrame,
    dataset_name: str | None = None,
    spec: PromptSpec | None = None,
    basic: bool = False,
    zero_shot: bool = False,
) -> dict[str, Any]:
    """
    Prepara un payload de prompt agnóstico de backend.

    Devuelve:
    {
        "spec": PromptSpec,
        "messages": list[dict],
        "system_txt": str,
        "user_txt": str,
    }
    """
    if spec is None:
        if dataset_name is None:
            raise ValueError("Debes proporcionar 'spec' o 'dataset_name'.")
        spec = get_prompt_spec(dataset_name)

    messages = build_messages(spec, target, vecinos, basic=basic, zero_shot=zero_shot)
    system_txt = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_txt = next((m["content"] for m in messages if m["role"] == "user"), "")

    return {
        "spec": spec,
        "messages": messages,
        "system_txt": system_txt,
        "user_txt": user_txt,
    }


def validate_generated_text(texto: str | None, spec: PromptSpec) -> bool:
    """Aplica validadores del dataset en cadena."""
    if not spec.validators:
        return bool(texto and texto.strip())
    return all(validator(texto, spec) for validator in spec.validators)


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
                "Text_interviewer_participant": "INV: what do you see?\n*PAR: the boy takes cookies.",
            },
            {
                "Diagnosis": "HC",
                "Age": 73,
                "MMSE": 27,
                "Gender": "F",
                "Text_interviewer_participant": "INV: anything else?\n*PAR: mother is washing dishes.",
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

    assert validate_generated_text(
        ": all that you see going on in the picture ? : the &-uh boy (i)s in the cookie jar (.) handing a cookie .",
        get_prompt_spec("pitt"),
    )
    assert not validate_generated_text("INV: hi : PAR: hello", get_prompt_spec("pitt"))
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

    assert IVANOVA_REQUIRED_PASSAGE in ivanova_user

    print(
        f"[SELF-CHECK] dataset=ivanova | messages={len(ivanova_messages)} "
        f"| roles_required={ivanova_spec.roles_required} | validators={len(ivanova_spec.validators)}"
    )
    print("[SELF-CHECK] ivanova user prompt (first 40 lines):")
    for line in ivanova_user.splitlines()[:40]:
        print(line)

    ivanova_ok = f": {IVANOVA_REQUIRED_PASSAGE}"
    ivanova_bad_speaker = f"INV: hola : PAR: {IVANOVA_REQUIRED_PASSAGE}"
    ivanova_bad_topic = ": Hoy fuimos al parque y comimos helado bajo el sol."

    assert validate_generated_text(ivanova_ok, ivanova_spec)
    assert not validate_generated_text(ivanova_bad_speaker, ivanova_spec)
    assert not validate_generated_text(ivanova_bad_topic, ivanova_spec)

    print("[SELF-CHECK] Prompt construction and validation passed for pitt/default/ivanova.")
