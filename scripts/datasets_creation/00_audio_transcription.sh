#!/bin/bash
#SBATCH --job-name=tfg_audio_transcription
#SBATCH --nodelist=hpc-gpu4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100_80:1
##SBATCH --constraint=cpu_amd
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/%x_%j.log

source ~/.bashrc
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

if [[ $# -ne 1 ]]; then
  echo "Uso: sbatch scripts/datasets_creation/00_audio_transcription.sh train|test"
  exit 1
fi

cd /mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition
mkdir -p /mnt/beegfs/groups/irgroup/sara_tfg/jsonl/taukdial_treatment

if [[ "$1" == "train" ]]; then
  python -m audio_transcription.ASR_audio_dataset \
    --audio-dir /mnt/beegfs/groups/irgroup/datasets/sara_tfg_multiconad/TAUKADIAL-24-train \
    --output-path /mnt/beegfs/groups/irgroup/sara_tfg/jsonl/taukdial_treatment/taukdial_train_transcrpt.json \
    --model large-v3 \
    --language-filter en

  python -m extracting_data.ASR_collection \
    --input-json /mnt/beegfs/groups/irgroup/sara_tfg/jsonl/taukdial_treatment/taukdial_train_transcrpt.json \
    --split train \
    --output-dir /mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection

elif [[ "$1" == "test" ]]; then
  python -m audio_transcription.ASR_audio_dataset \
    --audio-dir /mnt/beegfs/groups/irgroup/datasets/sara_tfg_multiconad/TAUKADIAL-24-test \
    --output-path /mnt/beegfs/groups/irgroup/sara_tfg/jsonl/taukdial_treatment/taukdial_test_transcrpt.json \
    --model large-v3 \
    --language-filter en

  python -m extracting_data.ASR_collection \
    --input-json /mnt/beegfs/groups/irgroup/sara_tfg/jsonl/taukdial_treatment/taukdial_test_transcrpt.json \
    --split test \
    --output-dir /mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection

else
  echo "Uso: sbatch scripts/datasets_creation/00_audio_transcription.sh train|test"
  exit 1
fi
