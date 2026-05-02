# ConvoCognition: Detección Temprana de Alzheimer en Conversaciones Inglés-Español

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="NLP" src="https://img.shields.io/badge/NLP-Clinical%20Language-5A4FCF?style=for-the-badge">
  <img alt="Transformers" src="https://img.shields.io/badge/Transformers-BERT-FFCC00?style=for-the-badge">
  <img alt="LLMs" src="https://img.shields.io/badge/LLMs-Gemini%20%7C%20Mistral-111827?style=for-the-badge">
</p>

Repositorio asociado a un Trabajo de Fin de Grado sobre **detección automática de deterioro cognitivo y Enfermedad de Alzheimer a partir de transcripciones conversacionales en inglés y español**.

El proyecto parte del pipeline original de **MultiConAD: A Unified Multilingual Conversational Dataset for Early Alzheimer's Detection**, pero adapta el alcance a una investigación más concreta: análisis hispano-inglés, detección de MCI, estudio de marcas conversacionales CHAT, aumento sintético con LLMs y transferencia entre datasets.

> Este repositorio tiene fines de investigación. No constituye una herramienta clínica ni un sistema de diagnóstico médico.

---

## Punto de Partida: MultiConAD

MultiConAD proporciona la base inicial para:

- normalizar datasets conversacionales relacionados con demencia;
- trabajar con transcripciones textuales y datos derivados de audio;
- unificar metadatos clínicos y demográficos;
- evaluar modelos en escenarios monolingües y multilingües.

Este TFG reutiliza esa infraestructura, pero restringe el estudio a **inglés y español** para analizar con más control la transferencia entre idiomas, tareas cognitivas y datasets.

Referencia original: <https://arxiv.org/abs/2502.19208>

---

## Qué Aporta Este TFG

Las aportaciones principales de esta investigación son:

- evaluación específica en **inglés y español**;
- comparación entre clasificación **binaria** y **multiclase**;
- atención especial a la clase **MCI**, por ser la más difícil y clínicamente relevante;
- análisis por dataset para estudiar el efecto de la heterogeneidad de tareas;
- uso de marcas **CHAT** como señales estructurales de pausas, repeticiones y reformulaciones;
- aumento de datos mediante generación sintética con **Gemini** y **Mistral**;
- análisis **cross-dataset**, entrenando en un dataset y evaluando en otro con la misma tarea para medir capacidad de transferencia.

---

## Experimentos

### 1. Baselines Clásicos

Se evalúan representaciones TF-IDF con clasificadores clásicos:

- SVM;
- Random Forest;
- Naive Bayes;
- Decision Tree;
- Logistic Regression.

Script principal:

```bash
python Experiments/TF_IDF_classifier.py --test_language en --task binary --translated no
```

### 2. Representaciones Densas

Se incluyen experimentos con embeddings densos E5 para comparar frente a TF-IDF.

```bash
python Experiments/e5_larg_classifier.py --test_language spa --task binary --translated no
```

### 3. BERT Binario y Multiclase

Se entrenan modelos BERT para:

- clasificación binaria: `Dementia` vs `HC`;
- clasificación multiclase: `Dementia`, `MCI`, `HC`;
- entrenamiento balanceado con pesos de clase;
- comparación entre inglés y español.

Scripts relevantes:

```text
Experiments/BERT_classification.py
Experiments/BERT_balanced.py
```

### 4. BERT con Marcas CHAT

Se estudia si conservar información estructural de las transcripciones mejora la detección. Las marcas CHAT se transforman en tokens especiales:

| Fenómeno | Marca original | Token |
| --- | --- | --- |
| Pausas | `(.)`, `(..)`, `(1.2)` | `[PAUSE]` |
| Repeticiones | `[/]` | `[REP]` |
| Reformulaciones | `[//]` | `[REF]` |

Scripts relevantes:

```text
Preprocessing_text/preprocess_language_features.py
Experiments/BERT_tokenizer.py
```

### 5. Aumento Sintético con LLMs

Se generan conversaciones sintéticas para estudiar escenarios de bajo recurso y desbalance de clases. La generación se condiciona con variables como diagnóstico, edad, género, MMSE y ejemplos reales cercanos.

Modelos utilizados:

- **Gemini**, mediante API;
- **Mistral**, mediante Ollama en entorno HPC.

Se comparan porcentajes de datos reales:

```text
0%, 20%, 40%, 60%, 80%, 100%
```

Scripts principales:

```text
Data_augmentation/create_stratified_slices.py
Data_augmentation/generacion_sintetica_gemini.py
Data_augmentation/generacion_sintetica_mistral.py
Data_augmentation/prompt_system.py
```

### 6. Interpretabilidad

Se usa SHAP para analizar qué partes del texto influyen en las predicciones de los modelos BERT.

```text
Experiments/shap_analysis.py
```

### 7. Análisis Cross-Dataset

Como experimento final, se estudia la transferencia entre datasets: entrenar un modelo en un corpus y evaluarlo sobre el test de otro corpus que comparta una tarea cognitiva comparable.

El objetivo es comprobar si el modelo aprende patrones generales de deterioro cognitivo o si se ajusta demasiado a características propias del dataset de entrenamiento.

Script relacionado:

```text
infer_bert_patient.py
```

---

## Estructura del Repositorio

```text
MultiConAD/
├── Audio_transcription/       # Transcripción automática de audio
├── Data_augmentation/         # Generación sintética con Gemini y Mistral
├── Experiments/               # TF-IDF, E5, BERT, SHAP y evaluación
├── Extracting_data/           # Extracción y normalización de datos
├── Markers_analysis/          # Análisis de pausas, reformulaciones y diagnósticos
├── Metadata_integration/      # Integración de metadatos
├── Preprocessing_text/        # Limpieza textual y marcas CHAT
├── Translation/               # Traducción automática heredada de MultiConAD
├── scripts/                   # Lanzadores SLURM para cluster
└── infer_bert_patient.py      # Evaluación por dataset / transferencia
```

---

## Datos y Ejecución

Los datasets originales no se distribuyen en este repositorio. El pipeline espera datos normalizados en JSONL con campos como:

```json
{
  "Diagnosis": "Dementia",
  "Age": 75,
  "Gender": "F",
  "MMSE": 21,
  "Dataset": "Ivanova",
  "Text_interviewer_participant": "PAR: ..."
}
```

Muchos scripts están preparados para ejecución en cluster con SLURM y rutas absolutas del entorno de trabajo:

```text
/mnt/beegfs/groups/irgroup/sara_tfg/
```

Para ejecutar el proyecto en otra máquina, es necesario adaptar rutas de entrada, salida, modelos y logs.

---

## Limitaciones

- Los datasets son pequeños y heterogéneos.
- La clase MCI está menos representada y es más difícil de detectar.
- Las tareas cognitivas no siempre son comparables entre datasets.
- Los datos sintéticos pueden introducir artefactos generativos.
- El código conserva dependencias y rutas propias del entorno HPC usado durante el TFG.

---

## Cita de MultiConAD

Este repositorio deriva del código y pipeline de MultiConAD. La cita:

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
