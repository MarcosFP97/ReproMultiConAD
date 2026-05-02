#!/bin/bash
#SBATCH --job-name=Generar_Sinteticos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=48:00:00 
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/GEN_%x_%j.log

source ~/.bashrc
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Lanzamos ollama
export OLLAMA_MODELS="/mnt/beegfs/groups/irgroup/sara_tfg/ollama_models"
export OLLAMA_LOAD_TIMEOUT="20m"
export OLLAMA_MAX_LOADED_MODELS=1
/mnt/beegfs/groups/irgroup/sara_tfg/ollama_bin/bin/ollama serve &
sleep 20

# Lanzamos la generación para Ivanova (100% datos sintéticos)
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_mistral.py --dataset ivanova --slice 20
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_mistral.py --dataset ivanova --slice 40
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_mistral.py --dataset ivanova --slice 60
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_mistral.py --dataset ivanova --slice 80
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_mistral.py --dataset ivanova --slice 100

# Lanzamos la generación para Pitt (100% datos sintéticos)
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_mistral.py --dataset pitt --slice 20
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_mistral.py --dataset pitt --slice 40
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_mistral.py --dataset pitt --slice 60
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_mistral.py --dataset pitt --slice 80
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_mistral.py --dataset pitt --slice 100
