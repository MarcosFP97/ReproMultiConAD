#!/bin/bash
#SBATCH --job-name=Generar_Sinteticos_GEMINI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      
#SBATCH --mem=32G              
#SBATCH --time=48:00:00 
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/GEN_%x_%j.log

source ~/.bashrc
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Lanzamos la generación para Ivanova (100% datos sintéticos cuando slice==0)
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_gemini.py --dataset ivanova --slice 0

# Lanzamos la generación para Pitt (100% datos sintéticos cuando slice==0)
python -u /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/generacion_sintetica_gemini.py --dataset pitt --slice 0
