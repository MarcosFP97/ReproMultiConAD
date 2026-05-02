#!/bin/bash
#SBATCH --job-name=BERT_Full_Multiclass_Pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/%x_%j.log

source ~/.bashrc
conda activate sara_tfg

# ==========================================
# 1. EXPERIMENTOS BASELINE (Sin marcadores)
# ==========================================
echo ">>> [EN] Empezando BASELINE"
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_classification.py --language en --task multiclass

echo ">>> [SPA] Empezando BASELINE"
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_classification.py --language spa --task multiclass

# ==========================================
# 2. EXPERIMENTOS CON MARCADORES (Tokens)
# ==========================================
echo ">>> [EN] Empezando PAUSE"
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_tokenizer.py --language en --task multiclass --mode pause

echo ">>> [EN] Empezando REP"
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_tokenizer.py --language en --task multiclass --mode rep

echo ">>> [EN] Empezando REF"
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_tokenizer.py --language en --task multiclass --mode ref

echo ">>> [EN] Empezando ALL"
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_tokenizer.py --language en --task multiclass --mode all

echo ">>> [SPA] Empezando PAUSE"
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_tokenizer.py --language spa --task multiclass --mode pause

echo ">>> [SPA] Empezando REP"
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_tokenizer.py --language spa --task multiclass --mode rep

echo ">>> [SPA] Empezando REF"
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_tokenizer.py --language spa --task multiclass --mode ref

echo ">>> [SPA] Empezando ALL"
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_tokenizer.py --language spa --task multiclass --mode all

echo ">>> TODOS LOS EXPERIMENTOS FINALIZADOS CON ÉXITO"
