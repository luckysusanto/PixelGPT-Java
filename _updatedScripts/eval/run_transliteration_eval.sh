#!/bin/bash
# =========================================================================================
# EVALUATION SCRIPT (Dual Dataset Support)
# =========================================================================================
#$ -P multilm
#$ -l gpus=1
#$ -l gpu_type=A40
#$ -l h_rt=04:00:00
#$ -N DualGPT_Eval_Dual
#$ -j y
#$ -o DualGPT_Eval_Dual-$JOB_ID.log

# Paths
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/_updatedScripts/eval"
SCRIPT_NAME="transliteration_eval.py" 

# --- CONFIGURATION ---

# 1. Model to Evaluate
MODEL_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/_updated_experiment_output/DatasetDebug-Finetune"
OUTPUT_DIR="${MODEL_PATH}/evaluation_results"

# 2. Dataset A (Required)
DATASET_A_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt-debug/default/0.0.0"
DATASET_A_LANG="javanese-ours"

# 3. Dataset B (Optional - set to empty string "" to disable)
DATASET_B_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt-debug/default/0.0.0"
DATASET_B_LANG="javanese-ours-2"

# 4. Common Settings
EVAL_SPLIT="test"
TEXT_COLUMN="grapheme_token_ids"

# --- EXECUTION ---
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"

echo "Evaluating Model: ${MODEL_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"

# Construct arguments array
CMD=(
    python "${PROJECT_DIR}/${SCRIPT_NAME}"
    --model_path "${MODEL_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --dataset_a_name "${DATASET_A_PATH}"
    --dataset_a_lang "${DATASET_A_LANG}"
    --eval_split "${EVAL_SPLIT}"
    --text_column "${TEXT_COLUMN}"
    --max_new_tokens 256
)

# Add Dataset B if defined
if [ ! -z "$DATASET_B_PATH" ] && [ ! -z "$DATASET_B_LANG" ]; then
    echo "Adding Dataset B: ${DATASET_B_LANG}"
    CMD+=( 
        --dataset_b_name "${DATASET_B_PATH}" 
        --dataset_b_lang "${DATASET_B_LANG}"
    )
fi

# Run
"${CMD[@]}"

echo "Done."