#!/bin/bash
#SBATCH --job-name=create_real_slices
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/%x_%j.log

set -euo pipefail

source ~/.bashrc
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

SCRIPT_PATH="/mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/create_stratified_slices.py"

for dataset in pitt ivanova; do
  echo "=== Creating stratified real slices for dataset=$dataset ==="
  python -u "$SCRIPT_PATH" --dataset "$dataset"
done
