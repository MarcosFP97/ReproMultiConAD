#!/bin/bash
#SBATCH --job-name=parse_CHA_to_JSONL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/%x_%A_%a.log
#SBATCH --array=0-8

set -euo pipefail

source ~/.bashrc
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

cd /mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition
mkdir -p /mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection

datasets=(Baycrest Delaware Ivanova Kempler Lu PerLA Pitt VAS WLS)
languages=(english english spanish english english spanish english english english)

i="${SLURM_ARRAY_TASK_ID:-0}"
dataset="${datasets[$i]}"
language="${languages[$i]}"

python -m extracting_data.cha_collection \
  --input-dir "/mnt/beegfs/groups/irgroup/datasets/sara_tfg_multiconad/$dataset" \
  --language "$language" \
  --output-dir /mnt/beegfs/groups/irgroup/sara_tfg/jsonl/results_cha_collection \
  --output-name "${dataset}.jsonl"
