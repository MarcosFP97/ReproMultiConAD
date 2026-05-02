#!/bin/bash
#SBATCH --job-name=Mistral_Clinical_Analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/%x_%j.log

# 1. Cargar el entorno de trabajo
# Ajusta esta ruta si tu conda está en otra ubicación
source ~/.bashrc
conda activate sara_tfg

# Arrancar el servidor de Ollama
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export OLLAMA_MODELS="/mnt/beegfs/groups/irgroup/sara_tfg/ollama_models"
export OLLAMA_LOAD_TIMEOUT="20m"
export OLLAMA_MAX_LOADED_MODELS=1
/mnt/beegfs/groups/irgroup/sara_tfg/ollama_bin/bin/ollama serve &
sleep 20

# 3. Lanzar el analizador
echo "Iniciando análisis clínico-lingüístico con Mistral Small 3.2..."
echo "Procesando dataset Ivanova Real para extraer patrones de deterioro..."

python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Data_augmentation/analista_clinico.py

echo "Análisis finalizado con éxito."
