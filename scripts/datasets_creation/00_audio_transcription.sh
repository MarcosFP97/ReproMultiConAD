#!/bin/bash

cd /ReproMultiConAD
mkdir -p ./taukdial_treatment

if [[ "$1" == "train" ]]; then
  python -m audio_transcription.ASR_audio_dataset \
    --audio-dir ./TAUKADIAL-24-train \
    --output-path ./taukdial_treatment/taukdial_train_transcrpt.json \
    --model large-v3 \
    --language-filter en

  python -m extracting_data.ASR_collection \
    --input-json ./taukdial_treatment/taukdial_train_transcrpt.json \
    --split train \
    --output-dir ./results_cha_collection

elif [[ "$1" == "test" ]]; then
  python -m audio_transcription.ASR_audio_dataset \
    --audio-dir ./TAUKADIAL-24-test \
    --output-path ./taukdial_treatment/taukdial_test_transcrpt.json \
    --model large-v3 \
    --language-filter en

  python -m extracting_data.ASR_collection \
    --input-json ./taukdial_treatment/taukdial_test_transcrpt.json \
    --split test \
    --output-dir ./results_cha_collection

else
  echo "Uso: sbatch scripts/datasets_creation/00_audio_transcription.sh train|test"
  exit 1
fi
