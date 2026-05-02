#!/bin/bash
#SBATCH --job-name=balancedBERT_SEEDS_FINAL_MODELS_PITT           # Job name
#SBATCH --nodes=1                    # -N Run all processes on a single node   
#SBATCH --ntasks=1                   # -n Run a single task   
#SBATCH --cpus-per-task=8            # -c Run 1 processor per task
#SBATCH --gres=gpu:1
#SBATCH --mem=40G                    # Job memory request
#SBATCH --time=12:00:00              # Time limit hrs:min:sec
#SBATCH --qos=regular                 # Cola
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/BERT/%x_%j.log       # Standard output and error log

source ~/.bashrc
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Pitt
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_balanced.py --dataset pitt --task multiclass --percentage 100 --seed 42
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_balanced.py --dataset pitt --task multiclass --percentage 100 --seed 52
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_balanced.py --dataset pitt --task multiclass --percentage 100 --seed 62