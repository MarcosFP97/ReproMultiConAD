#!/bin/bash
#SBATCH --job-name=preprocess_EN_JSONL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/%x_%j.log

source ~/.bashrc
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

cd /mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition
python -m Preprocessing_text.preprocess_global_english
