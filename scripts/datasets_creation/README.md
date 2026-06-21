# Dataset Creation Scripts

Esta carpeta contiene solo ejecuciones necesarias para construir los JSONL que consume el pipeline experimental.

Orden recomendado:

1. `00_audio_transcription.sh`
   - Transcribe audio cuando el dataset parte de audio y no de `.cha` ya anotado.
   - Uso: `sbatch scripts/datasets_creation/00_audio_transcription.sh train` o `sbatch scripts/datasets_creation/00_audio_transcription.sh test`.
   - Usa directamente `TAUKADIAL-24-train` o `TAUKADIAL-24-test`.
   - También normaliza TAUKADIAL a `jsonl/results_cha_collection/taukadial_English_{train|test}.jsonl`.

2. `01_parse_cha_files.sh`
   - Parsea ficheros CHAT `.cha` y escribe JSONL normalizados en `jsonl/results_cha_collection/`.
   - Usa array SLURM para: `Baycrest`, `Delaware`, `Ivanova`, `Kempler`, `Lu`, `PerLA`, `Pitt`, `VAS`, `WLS`.

3. `02a_preprocess_english.sh`
   - Ejecuta `preprocessing_text/preprocess_global_english.py`.
   - Combina datasets ingleses y genera un unico split train/test.
   - `train_en.jsonl` y `test_en.jsonl`: limpieza fuerte para TF-IDF.
   - `train_en_e5.jsonl` y `test_en_e5.jsonl`: limpieza debil/light para E5, BERT y marcadores.
   - Salidas en la raiz de `jsonl/`: `combined_jsonl_english.jsonl`, `train_en.jsonl`, `test_en.jsonl`, `train_en_e5.jsonl`, `test_en_e5.jsonl`.

4. `02b_preprocess_spanish.sh`
   - Ejecuta `preprocessing_text/preprocess_global_spanish.py`.
   - Combina datasets españoles y genera un unico split train/test.
   - `train_spa.jsonl` y `test_spa.jsonl`: limpieza fuerte para TF-IDF.
   - `train_spa_e5.jsonl` y `test_spa_e5.jsonl`: limpieza debil/light para E5, BERT y marcadores.
   - Salidas en la raiz de `jsonl/`: `combined_jsonl_spanish.jsonl`, `train_spa.jsonl`, `test_spa.jsonl`, `train_spa_e5.jsonl`, `test_spa_e5.jsonl`.

5. `03_preprocess_marker_features.sh`
   - Genera variantes con marcas CHAT convertidas a tokens especiales.
   - Incluye todos los corpus globales, también Taukadial y PerLA.
   - Salidas en `jsonl/markers_collections/` con nombres esperados por `experiments/BERT_tokenizer.py`:
     `train_{en|spa}_e5_markers_{none|pause|rep|ref|all}.jsonl` y equivalentes de test.
   - La variante `none` usa los mismos corpus que las variantes con tokens,
     pero elimina las tres marcas CHAT sin añadir tokens especiales. Es el baseline
     controlado de la Fase 2.

6. `04a_preprocess_individual_sets_english.sh`
   - Ejecuta `preprocessing_text/preprocess_individual_english.py`.
   - Crea versiones light en `jsonl/individual_sets/train_{dataset}.jsonl` y `test_{dataset}.jsonl`.
   - Crea versiones TF-IDF con limpieza fuerte en `jsonl/individual_sets/TFIDF/train_{dataset}.jsonl` y `test_{dataset}.jsonl`.
   - Por defecto procesa `pitt`, `delaware`, `lu`, `taukadial`, `vas`, `wls`.
   - Se puede restringir con `sbatch scripts/datasets_creation/04a_preprocess_individual_sets_english.sh pitt`.

7. `04b_preprocess_individual_sets_spanish.sh`
   - Ejecuta `preprocessing_text/preprocess_individual_spanish.py`.
   - Crea versiones light en `jsonl/individual_sets/train_ivanova.jsonl` y `test_ivanova.jsonl`.
   - Crea versiones TF-IDF con limpieza fuerte en `jsonl/individual_sets/TFIDF/train_ivanova.jsonl` y `test_ivanova.jsonl`.
   - Se puede restringir con `sbatch scripts/datasets_creation/04b_preprocess_individual_sets_spanish.sh ivanova`.

Rutas fijas usadas por los scripts:

- Repositorio: `/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition`.
- Datasets `.cha`: `/mnt/beegfs/groups/irgroup/datasets/sara_tfg_multiconad`.
- JSONL generados: `/mnt/beegfs/groups/irgroup/sara_tfg/jsonl`.
