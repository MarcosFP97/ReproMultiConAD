#!/bin/bash
#SBATCH --job-name=BERT_Tokenizer_Spanish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/%x_%j.log

set -euo pipefail

source ~/.bashrc
conda activate sara_tfg

SCRIPT="/mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_tokenizer.py"

echo ">>> [SPA] Empezando BINARY"
python -u "$SCRIPT" --language spa --task binary --mode pause
python -u "$SCRIPT" --language spa --task binary --mode rep
python -u "$SCRIPT" --language spa --task binary --mode ref
python -u "$SCRIPT" --language spa --task binary --mode all

echo ">>> [SPA] Empezando MULTICLASS"
python -u "$SCRIPT" --language spa --task multiclass --mode pause
python -u "$SCRIPT" --language spa --task multiclass --mode rep
python -u "$SCRIPT" --language spa --task multiclass --mode ref
python -u "$SCRIPT" --language spa --task multiclass --mode all
