#!/bin/bash
#SBATCH --job-name=infer_patients
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/%x_%j.log

python /mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/infer_bert_patient.py