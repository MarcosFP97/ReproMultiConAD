#!/bin/bash
#SBATCH --job-name=SHAP_Analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/SHAP_%j.log
#SBATCH --array=0-9

source ~/.bashrc
conda activate sara_tfg

# En inglés se estudian los cuatro modos. En español solo REP tiene cobertura
# real suficiente en Ivanova; PAUSE y REF no constituyen comparaciones válidas.
combinations=(
    "en binary rep"
    "en binary ref"
    "en binary pause"
    "en binary all"
    "en multiclass rep"
    "en multiclass ref"
    "en multiclass pause"
    "en multiclass all"
    "spa binary rep"
    "spa multiclass rep"
)

read -r current_lang current_task current_marker <<< "${combinations[$SLURM_ARRAY_TASK_ID]}"

echo "------------------------------------------------------------"
echo "PROCESANDO: Idioma=$current_lang | Task=$current_task | Marker=$current_marker"
echo "------------------------------------------------------------"

# 4. Ejecutar el script de Python con los argumentos correspondientes
python /mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/shap_analysis.py \
    --language "$current_lang" \
    --task "$current_task" \
    --marker "$current_marker" \
    --sample-size 60 \
    --bootstrap-iterations 2000 \
    --output-root /mnt/beegfs/groups/irgroup/sara_tfg/results/BERT_tokenizer

echo "Finalizado."
