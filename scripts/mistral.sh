#!/bin/bash
#SBATCH --job-name=mistral_prueba
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sara.castro.lopez@rai.usc.es
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/mistral/%x_%j.log

source ~/.bashrc
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export OLLAMA_MODELS="/mnt/beegfs/groups/irgroup/sara_tfg/ollama_models"
export OLLAMA_LOAD_TIMEOUT="20m"
export OLLAMA_MAX_LOADED_MODELS=1
/mnt/beegfs/groups/irgroup/sara_tfg/ollama_bin/bin/ollama serve &
sleep 20
python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_mistral.py --dataset ivanova --augmented
