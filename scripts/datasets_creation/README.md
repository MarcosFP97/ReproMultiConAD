# Dataset Creation Scripts

This directory contains only the scripts required to build the JSONL files consumed by the experimental pipeline.

Recommended order:

1. `00_audio_transcription.sh`

   - Transcribes audio when the dataset consists of audio recordings rather than already annotated `.cha` files.
   - Usage: `sh scripts/datasets_creation/00_audio_transcription.sh train` or `sh scripts/datasets_creation/00_audio_transcription.sh test`.
   - Directly uses `TAUKADIAL-24-train` or `TAUKADIAL-24-test`.
   - Also normalizes TAUKADIAL into `jsonl/results_cha_collection/taukadial_English_{train|test}.jsonl`.

2. `01_parse_cha_files.sh`

   - Parses CHAT `.cha` files and writes normalized JSONL files to `jsonl/results_cha_collection/`.
   - Uses a SLURM array for: `Baycrest`, `Delaware`, `Ivanova`, `Kempler`, `Lu`, `PerLA`, `Pitt`, `VAS`, `WLS`.

3. `02a_preprocess_english.sh`

   - Runs `preprocessing_text/preprocess_global_english.py`.
   - Combines English datasets and generates a single train/test split.
   - `train_en.jsonl` and `test_en.jsonl`: strong cleaning for TF-IDF.
   - `train_en_e5.jsonl` and `test_en_e5.jsonl`: light cleaning for E5, BERT, and marker-based experiments.
   - Outputs are stored in the root of `jsonl/`: `combined_jsonl_english.jsonl`, `train_en.jsonl`, `test_en.jsonl`, `train_en_e5.jsonl`, `test_en_e5.jsonl`.

4. `02b_preprocess_spanish.sh`

   - Runs `preprocessing_text/preprocess_global_spanish.py`.
   - Combines Spanish datasets and generates a single train/test split.
   - `train_spa.jsonl` and `test_spa.jsonl`: strong cleaning for TF-IDF.
   - `train_spa_e5.jsonl` and `test_spa_e5.jsonl`: light cleaning for E5, BERT, and marker-based experiments.
   - Outputs are stored in the root of `jsonl/`: `combined_jsonl_spanish.jsonl`, `train_spa.jsonl`, `test_spa.jsonl`, `train_spa_e5.jsonl`, `test_spa_e5.jsonl`.

5. `03_preprocess_marker_features.sh`

   - Generates variants with CHAT markers converted into special tokens.
   - Includes all global corpora, including Taukadial and PerLA.
   - Outputs are stored in `jsonl/markers_collections/` with the filenames expected by `experiments/BERT_tokenizer.py`:
     `train_{en|spa}_e5_markers_{none|pause|rep|ref|all}.jsonl` and the corresponding test files.
   - The `none` variant uses the same corpora as the token-based variants, but removes the three CHAT markers without adding any special tokens. This serves as the controlled baseline for Phase 2.

6. `04a_preprocess_individual_sets_english.sh`

   - Runs `preprocessing_text/preprocess_individual_english.py`.
   - Creates light-cleaned versions in `jsonl/individual_sets/train_{dataset}.jsonl` and `test_{dataset}.jsonl`.
   - Creates TF-IDF versions with strong cleaning in `jsonl/individual_sets/TFIDF/train_{dataset}.jsonl` and `test_{dataset}.jsonl`.
   - By default, processes `pitt`, `delaware`, `lu`, `taukadial`, `vas`, and `wls`.
   - The datasets can be restricted, e.g. `sh scripts/datasets_creation/04a_preprocess_individual_sets_english.sh pitt`.

7. `04b_preprocess_individual_sets_spanish.sh`

   - Runs `preprocessing_text/preprocess_individual_spanish.py`.
   - Creates light-cleaned versions in `jsonl/individual_sets/train_ivanova.jsonl` and `test_ivanova.jsonl`.
   - Creates TF-IDF versions with strong cleaning in `jsonl/individual_sets/TFIDF/train_ivanova.jsonl` and `test_ivanova.jsonl`.
   - The dataset can be restricted, e.g. `sh scripts/datasets_creation/04b_preprocess_individual_sets_spanish.sh ivanova`.