#!/bin/bash
#SBATCH --job-name=GEN_Gemini_SPA_real_pct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/GEN_%x_%j.log

set -euo pipefail

source ~/.bashrc
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

SCRIPT_PATH="/mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_gemini.py"

for dataset in ivanova; do
  for real_percentage in 0 20 40 60 80; do
    python -u "$SCRIPT_PATH" \
      --dataset "$dataset" \
      --real-percentage "$real_percentage"
  done
done
