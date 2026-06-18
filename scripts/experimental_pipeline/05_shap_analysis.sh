#!/bin/bash
#SBATCH --job-name=SHAP_Analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/SHAP_%j.log
#SBATCH --array=0-15  # Define las 16 combinaciones

source ~/.bashrc
conda activate sara_tfg

# Ejecutamos el análisis

# 2. Definir las variantes que quieres procesar
# Idiomas: en, spa
# Tasks: binary (puedes añadir multiclass si tienes los modelos)
# Markers: rep, ref, pause, all
languages=("spa" "en")
tasks=("binary" "multiclass")
markers=("rep" "ref" "pause" "all")

# 3. Lógica para convertir el ID del array en una combinación única
# Calculamos el total de combinaciones (2 * 2 * 4 = 16)
# Lanzaremos el sbatch con --array=0-15

total_markers=${#markers[@]}
total_tasks=${#tasks[@]}

lang_idx=$(( SLURM_ARRAY_TASK_ID / (total_tasks * total_markers) ))
task_idx=$(( (SLURM_ARRAY_TASK_ID / total_markers) % total_tasks ))
marker_idx=$(( SLURM_ARRAY_TASK_ID % total_markers ))

current_lang=${languages[$lang_idx]}
current_task=${tasks[$task_idx]}
current_marker=${markers[$marker_idx]}

echo "------------------------------------------------------------"
echo "PROCESANDO: Idioma=$current_lang | Task=$current_task | Marker=$current_marker"
echo "------------------------------------------------------------"

# 4. Ejecutar el script de Python con los argumentos correspondientes
python /mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/shap_analysis.py \
    --language "$current_lang" \
    --task "$current_task" \
    --marker "$current_marker" \
    --sample-size 30

echo "Finalizado."
