# ConvoCognition: Detección Temprana de Alzheimer en Conversaciones Inglés-Español

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="NLP" src="https://img.shields.io/badge/NLP-Clinical%20Language-5A4FCF?style=for-the-badge">
  <img alt="Transformers" src="https://img.shields.io/badge/HuggingFace-Transformers-FFCC00?style=for-the-badge&logo=huggingface&logoColor=black">
  <img alt="LLMs" src="https://img.shields.io/badge/LLMs-Gemini%20%7C%20Mistral-111827?style=for-the-badge">
  <img alt="HPC" src="https://img.shields.io/badge/HPC-SLURM-ED1C24?style=for-the-badge">
</p>

Repositorio del Trabajo de Fin de Grado sobre **detección automática de deterioro cognitivo y Enfermedad de Alzheimer a partir de transcripciones conversacionales en inglés y español**. El proyecto combina modelos clásicos (TF-IDF), representaciones densas (E5), modelos Transformer (BERT), aumento sintético con LLMs e interpretabilidad con SHAP, estructurado en cinco fases experimentales.

> Este repositorio tiene fines de investigación. No constituye una herramienta clínica ni un sistema de diagnóstico médico.

---

## Punto de partida: MultiConAD

El proyecto parte del pipeline de **MultiConAD: A Unified Multilingual Conversational Dataset for Early Alzheimer's Detection** [[arXiv:2502.19208]](https://arxiv.org/abs/2502.19208), que proporciona la infraestructura base para normalizar datasets conversacionales sobre demencia, unificar metadatos clínicos y demográficos, y evaluar modelos en escenarios monolingües y multilingües.

Este TFG restringe el estudio a **inglés y español**, incorporando análisis por dataset, marcas CHAT como tokens especiales, aumento sintético controlado y transferencia cross-dataset.

---

## Datasets

| Dataset | Idioma | Tarea | Clases |
|---|---|---|---|
| Pitt (DementiaBank) | Inglés | Cookie Theft | HC, MCI, Dementia |
| Delaware | Inglés | Varias | HC, MCI, Dementia |
| Lu | Inglés | Varias | HC, Dementia |
| Taukadial | Inglés | Picture description | HC, Dementia |
| VAS | Inglés | Varias | HC, MCI |
| WLS | Inglés | Varias | HC, Dementia |
| Ivanova | Español | Lectura (El Quijote) | HC, MCI, Dementia |

Los datos normalizados siguen el esquema `NormalizedDataPoint` en JSONL:

```json
{
  "PID": "...", "Dataset": "Pitt", "Diagnosis": "Dementia",
  "Age": 75, "Gender": "F", "MMSE": 21,
  "Text_participant": "PAR: ..."
}
```

Los datasets originales no se distribuyen en este repositorio. Para ejecutar el proyecto en otra máquina, adaptar las rutas en los scripts SLURM o pasar las rutas como argumentos CLI.

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
experiments/BERT_balanced.py   ← Entrenamiento con datos reales + sintéticos + cross-dataset
```

---

## Experimentos

| # | Experimento | Script principal |
|---|---|---|
| 1 | Baselines TF-IDF (SVM, RF, NB, DT, LR) | `experiments/TF_IDF_classifier.py` |
| 2 | Embeddings densos E5 | `experiments/e5_larg_classifier.py` |
| 3 | BERT ajuste fino | `experiments/BERT_classification.py` |
| 4 | BERT con marcas CHAT (`[PAUSE]`, `[REP]`, `[REF]`) | `experiments/BERT_tokenizer.py` |
| 5 | SHAP agregado y ablación de tokens CHAT | `experiments/shap_analysis.py` |
| 6 | TF-IDF por dataset individual (balanceado/no balanceado) | `experiments/TF_IDF_single_classifier.py` |
| 7 | BERT balanceado: individual, cross-dataset y sintético | `experiments/BERT_balanced.py` |
| 8 | Generación sintética (0–100 % datos reales) | `data_augmentation/generacion_sintetica_*.py` |
| 9 | Análisis cross-dataset final | `experiments/cross_task_analysis.py` |

### Marcas CHAT como tokens especiales

| Fenómeno | Notación original | Token |
|---|---|---|
| Pausas | `(.)`, `(..)`, `(1.2)` | `[PAUSE]` |
| Repeticiones | `[/]` | `[REP]` |
| Reformulaciones | `[//]` | `[REF]` |

El argumento `--mode` controla qué tokens se activan: `pause`, `rep`, `ref`, `all` o `none` (sin marcadores, referencia controlada). El análisis SHAP guarda los HTML y los Excel agregados en `results/BERT_tokenizer/`. En español solo se analiza `[REP]`, ya que Ivanova no ofrece cobertura útil de pausas o reformulaciones.

### Aumento sintético

La generación se condiciona con diagnóstico, edad, género, MMSE y ejemplos reales, evaluando proporciones de datos reales del 0 % al 100 %:

```
0% · 20% · 40% · 60% · 80% · 100%
```

Modelos disponibles: **Gemini 2.5 Flash** (API, con imagen Cookie Theft en Pitt) y **Mistral Small 3.2** (local vía Ollama).

### Transferencia cross-dataset

`experiments/BERT_balanced.py --mode cross` entrena en Pitt y evalúa directamente sobre WLS y Taukadial sin ajuste adicional. `experiments/cross_task_analysis.py` genera las tablas y figuras comparativas del análisis.

---

## Estructura del repositorio

```
ConvoCognition/
├── audio_transcription/       # Transcripción automática de audio (Taukadial)
├── data_augmentation/         # Generación sintética con Gemini y Mistral
├── experiments/               # Clasificadores: TF-IDF, E5, BERT, SHAP
├── extracting_data/           # Parseo y normalización de ficheros CHAT
├── markers_analysis/          # Análisis de pausas, repeticiones y reformulaciones
├── metadata_integration/      # Loaders y Enrichers por dataset
├── preprocessing_text/        # Limpieza textual y conversión de marcas CHAT
├── scripts/
│   ├── datasets_creation/     # Parseo, limpieza y creación de JSONL
│   └── experimental_pipeline/ # Scripts SLURM numerados (00–11)
├── environment.yml            # Dependencias del entorno Conda
└── infer_bert_patient.py      # Evaluación cross-dataset sobre todos los corpus
```

Los scripts SLURM están separados por propósito:

- `scripts/datasets_creation/`: transcripción, parseo `.cha`, limpieza y variantes con marcas CHAT.
- `scripts/experimental_pipeline/`: experimentos numerados (`01a`, `01b`, …). El sufijo `a` corresponde a inglés y `b` a español.

---

## Configuración del entorno

### Instalación

```bash
git clone https://github.com/Saracas-Code/ConvoCognition.git
cd ConvoCognition
conda env create -f environment.yml
conda activate sara_tfg
```

### Generación con Gemini

Crear un fichero `.env` en la raíz del repositorio:

```
GEMINI_API_KEY=tu_clave_aqui
```

### Generación con Mistral (Ollama)

```bash
ollama pull mistral-small3.2
ollama serve   # debe estar activo durante la ejecución
```

---

## Ejecución en HPC (SLURM)

Los scripts esperan datos y modelos en:

```
/mnt/beegfs/groups/irgroup/sara_tfg/
├── jsonl/          # Datos normalizados
├── results/        # Resultados y métricas en .xlsx
├── logs/           # Logs de ejecución
└── MultiConAD/Experiments/BERT_Models/  # Checkpoints BERT
```

Lanzar un experimento:

```bash
sbatch scripts/experimental_pipeline/03a_bert_classification_english.sh
sbatch scripts/experimental_pipeline/09a_generate_gemini_english.sh
```

Crear datasets desde cero:

```bash
sbatch scripts/datasets_creation/01_parse_cha_files.sh
sbatch scripts/datasets_creation/02a_preprocess_english.sh
sbatch scripts/datasets_creation/02b_preprocess_spanish.sh
sbatch scripts/datasets_creation/03_preprocess_marker_features.sh
```

---

## Ejecución en local

Ajustar las rutas de datos y resultados según el entorno. Ejemplos de uso habitual:

```bash
# TF-IDF combinado
python experiments/TF_IDF_classifier.py --test_language en --task binary --translated no

# BERT ajuste fino
python experiments/BERT_classification.py --language en --task binary

# BERT con marcas CHAT
python experiments/BERT_tokenizer.py --language en --task binary --mode ref

# BERT balanceado (corpus individual)
python experiments/BERT_balanced.py --mode individual --train-dataset pitt --task binary

# BERT balanceado (transferencia cross-dataset)
python experiments/BERT_balanced.py --mode cross --train-dataset pitt --test-dataset wls --task binary

# BERT balanceado (datos sintéticos)
python experiments/BERT_balanced.py --mode synthetic --real-percentage 80 --task binary

# Generación sintética
python data_augmentation/generacion_sintetica_gemini.py --dataset pitt --real-percentage 20
python data_augmentation/generacion_sintetica_mistral.py --dataset pitt --real-percentage 20

# Tests unitarios del parser CHAT
python -m pytest extracting_data/test_ch_collection.py
```

---

## Limitaciones

- Los datasets son pequeños y heterogéneos; los resultados son sensibles a la partición train/test.
- La clase MCI está subrepresentada y es la más difícil de detectar.
- Las tareas cognitivas no son comparables entre todos los datasets.
- Los datos sintéticos pueden introducir artefactos generativos, especialmente en corpus con tarea fija (Ivanova).
- Los scripts SLURM contienen rutas absolutas del entorno HPC y requieren adaptación para otros sistemas.

---

## Referencia base

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
