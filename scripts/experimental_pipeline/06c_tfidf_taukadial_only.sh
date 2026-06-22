#!/bin/bash
#SBATCH --job-name=TFIDF_taukadial_EN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/TFIDF/%x_%j.log

source ~/.bashrc
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

PROJECT_PATH="/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition"
SCRIPT_PATH="/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/TF_IDF_single_classifier.py"

echo "=== Regenerando Taukadial en inglés ==="
cd "$PROJECT_PATH"
python -m preprocessing_text.preprocess_individual_english \
  --dataset taukadial \
  --data-root /mnt/beegfs/groups/irgroup/sara_tfg/jsonl

echo "=== Dataset: taukadial | binary | unbalanced ==="
python "$SCRIPT_PATH" --dataset taukadial --task binary

echo "=== Dataset: taukadial | binary | balanced ==="
python "$SCRIPT_PATH" --dataset taukadial --task binary --balanced
