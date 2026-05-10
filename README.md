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

Los datasets originales no se distribuyen en este repositorio. Para ejecutar el proyecto en otra máquina, adaptar las rutas en los scripts.

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
BERT_balanced.py          ← Entrenamiento de los datasets con datos reales + sintéticos + cross-dataset

```

---

## Experimentos

| # | Experimento | Script principal |
|---|---|---|
| 1 | Baselines TF-IDF (SVM, RF, NB, DT, LR) | `experiments/TF_IDF_classifier.py` |
| 2 | Embeddings densos E5 | `experiments/e5_larg_classifier.py` |
| 3 | BERT | `experiments/BERT_classification.py` |
| 4 | BERT con marcas CHAT (`[PAUSE]`, `[REP]`, `[REF]`) | `experiments/BERT_tokenizer.py` |
| 5 | TF-IDF por dataset individual (no balanceado/balanceado) | `experiments/TF_IDF_single_classifier.py` |
| 6 | BERT con balanceo de clases para experimentos individuales y cross-task | `experiments/BERT_balanced.py` |
| 7 | Aumento sintético (0–80% datos reales) | `data_augmentation/generacion_sintetica_*.py` |
| 8 | Análisis final cross-task | `experiments/cross_task_analysis.py` |

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

Modelos disponibles: **Gemini 2.5 Flash** (API) y **Mistral Small 3.2** (Ollama en HPC).

### Análisis cross-task

El script `experiments/cross_task_analysis.py` analiza los resultados del experimento cross-task final. Este experimento entrena BERT balanceado con Pitt completo, mapeando `HC -> NoDisease` y `MCI/Dementia -> Disease`, y evalúa el mismo modelo en WLS y Taukadial. Además compara contra un baseline trivial `DummyClassifier(strategy="most_frequent")`.

Ejemplo:

```bash
python experiments/cross_task_analysis.py \
  --results-dir /mnt/beegfs/groups/irgroup/sara_tfg/results/BERT_synthetic_analysis \
  --output-dir /mnt/beegfs/groups/irgroup/sara_tfg/results/cross_task_analysis
```

Análisis generados:

- Tabla resumen BERT vs baseline mayoritario.
- Comparativa de `Accuracy`, `Macro-F1`, `Disease_recall` y `NoDisease_recall`.
- Gráfica específica de recall de `Disease`.
- Matrices de confusión de BERT y baseline para WLS y Taukadial.
- Tipos de error (`TP_Disease`, `FN_Disease`, `FP_Disease`, `TN_NoDisease`).
- Distribución de confianza para aciertos y errores de BERT.
- Resumen en `summary.md`.

---

## Estructura del Repositorio

```text
ConvoCognition/
├── audio_transcription/       # Transcripción automática de audio
├── data_augmentation/         # Generación sintética con Gemini y Mistral
├── experiments/               # TF-IDF, E5, BERT
├── extracting_data/           # Extracción y normalización de datos CHAT
├── markers_analysis/          # Análisis de pausas, repeticiones y reformulaciones
├── metadata_integration/      # Loaders y Enrichers por dataset (para incorporar informacion externa a los jsonl)
├── preprocessing_text/        # Limpieza textual y marcas CHAT
├── scripts/
│   ├── datasets_creation/     # Parseo, limpieza y creación de JSONL
│   └── experimental_pipeline/ # Lanzadores SLURM de experimentos
└── infer_bert_patient.py      # Evaluación cross-dataset / transferencia
```

Los scripts SLURM están separados por propósito:

- `scripts/datasets_creation/`: transcripción, parseo `.cha`, limpieza de JSONL y variantes con marcas CHAT.
- `scripts/experimental_pipeline/`: experimentos numerados (`01a`, `01b`, …). La letra `a` corresponde a inglés y la `b` a español.

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
├── results/        # Resultados y métricas en .xlsx
├── logs/           # Logs de ejecución
└── MultiConAD/Experiments/BERT_Models/  # Checkpoints
```

Lanzar un experimento:

```bash
sbatch scripts/experimental_pipeline/03a_bert_classification_english.sh
sbatch scripts/experimental_pipeline/09a_generate_gemini_english.sh
```

Crear datasets:

```bash
sbatch scripts/datasets_creation/01_parse_cha_files.sh
sbatch scripts/datasets_creation/02a_preprocess_english.sh
sbatch scripts/datasets_creation/02b_preprocess_spanish.sh
sbatch scripts/datasets_creation/03_preprocess_marker_features.sh
```

---

## Limitaciones

- Los datasets son pequeños y heterogéneos; los resultados pueden ser muy sensibles a la partición.
- La clase MCI está subrepresentada y es la más difícil de detectar; esto se contempla con intensidad en los análisis de datasets individuales
- Las tareas cognitivas no siempre son comparables entre datasets (descripción de imágen, tarea narrativa, entrevista personal...)
- Los datos sintéticos pueden introducir artefactos generativos (a pesar de que se intente controlar con firmeza a traves del codigo).
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
