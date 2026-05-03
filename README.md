# ConvoCognition: Detección Temprana de Alzheimer en Conversaciones Inglés-Español

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="NLP" src="https://img.shields.io/badge/NLP-Clinical%20Language-5A4FCF?style=for-the-badge">
  <img alt="Transformers" src="https://img.shields.io/badge/HuggingFace-Transformers-FFCC00?style=for-the-badge&logo=huggingface&logoColor=black">
  <img alt="LLMs" src="https://img.shields.io/badge/LLMs-Gemini%20%7C%20Mistral-111827?style=for-the-badge">
  <img alt="HPC" src="https://img.shields.io/badge/HPC-SLURM-ED1C24?style=for-the-badge">
</p>

Repositorio del Trabajo de Fin de Grado sobre **detección automática de deterioro cognitivo y Enfermedad de Alzheimer a partir de transcripciones conversacionales en inglés y español**. El proyecto combina modelos clásicos (TF-IDF), representaciones densas (E5), modelos transformers (BERT), aumento sintético con LLMs e interpretabilidad con SHAP.

> Este repositorio tiene fines de investigación. No constituye una herramienta clínica ni un sistema de diagnóstico médico.

---

## Punto de Partida: MultiConAD

El proyecto parte del pipeline de **MultiConAD: A Unified Multilingual Conversational Dataset for Early Alzheimer's Detection** [[arXiv:2502.19208]](https://arxiv.org/abs/2502.19208), que proporciona la infraestructura base para:

- normalizar datasets conversacionales sobre demencia;
- unificar metadatos clínicos y demográficos;
- evaluar modelos en escenarios monolingües y multilingües.

Este TFG restringe el estudio a **inglés y español**, incorporando análisis por dataset, marcas CHAT, aumento sintético y transferencia cross-dataset.

---

## Datasets

| Dataset | Idioma | Tarea | Clases |
|---|---|---|---|
| Pitt (DementiaBank) | Inglés | Cookie Theft | HC, MCI, Dementia |
| Delaware | Inglés | Varias | HC, MCI, Dementia |
| Lu | Inglés | Varias | HC, Dementia |
| TAUKADIAL | Inglés | Picture description | HC, Dementia |
| VAS | Inglés | Varias | HC, MCI |
| WLS | Inglés | Varias | HC, Dementia |
| Ivanova | Español | Varias | HC, MCI, Dementia |

Los datos normalizados siguen el esquema `NormalizedDataPoint` en JSONL:

```json
{
  "PID": "...", "Dataset": "Pitt", "Diagnosis": "Dementia",
  "Age": 75, "Gender": "F", "MMSE": 21,
  "Text_participant": "PAR: ..."
}
```

Los datasets originales no se distribuyen en este repositorio. Para ejecutar el proyecto en otra máquina, adaptar las rutas en los scripts (o en `CLAUDE.md`).

---

## Pipeline

```
Ficheros CHAT (.cha)
    │
    ▼
extracting_data/          ← Parseo y normalización a JSONL
    │
    ▼
preprocessing_text/       ← Limpieza de texto + marcas CHAT → tokens especiales
    │
    ▼
┌───────────────────────────────────────────────┐
│               EXPERIMENTOS                     │
│  TF-IDF baselines → E5 embeddings → BERT       │
└───────────────────────────────────────────────┘
    │
    ▼
data_augmentation/        ← Generación sintética (Gemini / Mistral)
    │
    ▼
BERT_balanced.py          ← Entrenamiento con datos reales + sintéticos
    │
    ▼
shap_analysis.py          ← Interpretabilidad
infer_bert_patient.py     ← Análisis cross-dataset
```

---

## Experimentos

| # | Experimento | Script principal |
|---|---|---|
| 1 | Baselines TF-IDF (SVM, RF, NB, DT, LR) | `experiments/TF_IDF_classifier.py` |
| 2 | TF-IDF por dataset individual (balanceado/no) | `experiments/TF_IDF_single_classifier.py` |
| 3 | Embeddings densos E5 | `experiments/e5_larg_classifier.py` |
| 4 | BERT binario y multiclase | `experiments/BERT_classification.py` |
| 5 | BERT con marcas CHAT (`[PAUSE]`, `[REP]`, `[REF]`) | `experiments/BERT_tokenizer.py` |
| 6 | BERT con balanceo de clases | `experiments/BERT_balanced.py` |
| 7 | Aumento sintético (0–80% datos reales) | `data_augmentation/generacion_sintetica_*.py` |
| 8 | Interpretabilidad SHAP | `experiments/shap_analysis.py` |
| 9 | Transferencia cross-dataset | `infer_bert_patient.py` |

### Marcas CHAT como tokens especiales

| Fenómeno | Notación original | Token |
|---|---|---|
| Pausas | `(.)`, `(..)`, `(1.2)` | `[PAUSE]` |
| Repeticiones | `[/]` | `[REP]` |
| Reformulaciones | `[//]` | `[REF]` |

### Aumento sintético

La generación se condiciona con diagnóstico, edad, género, MMSE y ejemplos reales, probando proporciones de datos reales:

```
0% · 20% · 40% · 60% · 80% · 100%
```

Modelos disponibles: **Gemini** (API) y **Mistral** (Ollama en HPC).

---

## Estructura del Repositorio

```text
ConvoCognition/
├── audio_transcription/       # Transcripción automática de audio
├── data_augmentation/         # Generación sintética con Gemini y Mistral
├── experiments/               # TF-IDF, E5, BERT, SHAP y evaluación
├── extracting_data/           # Extracción y normalización de datos CHAT
├── markers_analysis/          # Análisis de pausas, repeticiones y reformulaciones
├── metadata_integration/      # Loaders y Enrichers por dataset
├── preprocessing_text/        # Limpieza textual y marcas CHAT
├── scripts/                   # Lanzadores SLURM numerados por orden de ejecución
└── infer_bert_patient.py      # Evaluación cross-dataset / transferencia
```

Los scripts SLURM en `scripts/` están numerados (`00a`, `00b`, `01a`, `01b`, …) para reflejar el orden de ejecución del pipeline: la letra `a` corresponde a experimentos en inglés y la `b` a español.

---

## Configuración del Entorno

```bash
conda activate sara_tfg
```

Para generación con Gemini, añadir un fichero `.env`:

```
GEMINI_API_KEY=tu_clave_aqui
```

Para generación con Mistral (Ollama):

```bash
ollama serve  # debe estar activo durante la ejecución
```

---

## Ejecución en HPC (SLURM)

Los scripts esperan datos y modelos en:

```
/mnt/beegfs/groups/irgroup/sara_tfg/
├── jsonl/          # Datos normalizados
├── results/        # Resultados y métricas
├── logs/           # Logs de ejecución
└── MultiConAD/Experiments/BERT_Models/  # Checkpoints
```

Lanzar un experimento:

```bash
sbatch scripts/04a_bert_classification_english.sh
sbatch scripts/09a_generate_gemini_english.sh
```

---

## Limitaciones

- Los datasets son pequeños y heterogéneos; los resultados pueden ser sensibles a la partición.
- La clase MCI está subrepresentada y es la más difícil de detectar.
- Las tareas cognitivas no siempre son comparables entre datasets.
- Los datos sintéticos pueden introducir artefactos generativos.
- El código contiene rutas absolutas del entorno HPC y requiere adaptación para otros entornos.

---

## Referencia Base

```bibtex
@misc{shakeri2025multiconadunifiedmultilingualconversational,
  title={MultiConAD: A Unified Multilingual Conversational Dataset for Early Alzheimer's Detection},
  author={Arezo Shakeri and Mina Farmanbar and Krisztian Balog},
  year={2025},
  eprint={2502.19208},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2502.19208}
}
```
