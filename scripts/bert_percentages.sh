#!/bin/bash
#SBATCH --job-name=BERT_Percentages           # Job name
#SBATCH --nodes=1                    # -N Run all processes on a single node   
#SBATCH --ntasks=1                   # -n Run a single task   
#SBATCH --cpus-per-task=8            # -c Run 1 processor per task
#SBATCH --gres=gpu:1
#SBATCH --mem=40G                    # Job memory request
#SBATCH --time=10:00:00              # Time limit hrs:min:sec
#SBATCH --qos=regular                 # Cola
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/BERT/%x_%j.log       # Standard output and error log

source ~/.bashrc
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_classification.py --dataset ivanova --percentage 20
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_classification.py --dataset ivanova --percentage 40
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_classification.py --dataset ivanova --percentage 60
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_classification.py --dataset ivanova --percentage 80
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_classification.py --dataset pitt --percentage 20
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_classification.py --dataset pitt --percentage 40
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_classification.py --dataset pitt --percentage 60
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_classification.py --dataset pitt --percentage 80
