#!/bin/bash
#SBATCH --job-name=E5_spa_binary     # Job name
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sara.castro.lopez@rai.usc.es
#SBATCH --nodes=1                    # -N Run all processes on a single node   
#SBATCH --ntasks=1                   # -n Run a single task   
#SBATCH --cpus-per-task=8            # -c Run 1 processor per task       
#SBATCH --gres=gpu:1
##SBATCH --constraint=cpu_amd     
#SBATCH --mem=40G                    # Job memory request
#SBATCH --time=04:00:00              # Time limit hrs:min:sec
#SBATCH --qos=regular                # Cola
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/E5/%x_%j.log       # Standard output and error log

source ~/.bashrc
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/e5_larg_classifier.py --test_language spa --task binary --translated no
