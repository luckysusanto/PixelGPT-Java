#!/bin/bash
# =========================================================================================
# SUBMISSION SCRIPT FOR TEXT-TO-IMAGE (SCRIPT RENDERING) SSIM EVALUATION
# =========================================================================================

#$ -P multilm
#$ -l gpus=1
#$ -pe omp 1
#$ -l gpu_type=A40
#$ -l h_rt=02:00:00
#$ -N ErniePixel_Eval_SSIM      # <-- RENAMED FOR CLARITY
#$ -j y
#$ -o ErniePixel_Eval_TTI_SSIM$JOB_ID.log # <-- RENAMED FOR CLARITY

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/scripts/eval"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

# --- CONFIGURATION ---
# IMPORTANT: This must point to your TEXT-TO-IMAGE model checkpoint
FINETUNED_MODEL_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/finetune_imageGen/checkpoint-37500"

EVAL_OUTPUT_DIR="./eval_finetune_tti_ssim" # <-- RENAMED FOR CLARITY
# Make sure this is the name you saved the SSIM Python script as
PYTHON_SCRIPT_NAME="eval_tti.py" 
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"
EVAL_DATASET_PATH="izzako/javanese-pixelgpt-test"
EVAL_SPLIT="train"

# --- ENVIRONMENT SETUP AND EXECUTION ---
echo "==================================================================================="
echo "JOB NAME:         $JOB_NAME"
echo "JOB ID:           $JOB_ID"
echo "Evaluating Model (SSIM): ${FINETUNED_MODEL_PATH}" # <-- UPDATED MESSAGE
echo "==================================================================================="

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
export HF_HOME="${CACHE_PATH}"

echo "Starting evaluation..."

# The command is now much simpler
python "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}" \
    --model_path "${FINETUNED_MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --dataset_name "${EVAL_DATASET_PATH}" \
    --eval_split "${EVAL_SPLIT}" \
    --output_dir "${EVAL_OUTPUT_DIR}"

    # --max_new_tokens ${MAX_NEW_TOKENS} # <-- REMOVED, this was the cause of the error

echo "Evaluation complete. Results are in ${EVAL_OUTPUT_DIR}"