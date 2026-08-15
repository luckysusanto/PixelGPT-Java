#!/bin/bash
# =========================================================================================
# SUBMISSION SCRIPT FOR ERNIEPIXEL TRANSLITERATION EVALUATION (MANUAL LOOP)
# =========================================================================================

#$ -P multilm
#$ -l gpus=1
#$ -pe omp 1
#$ -l gpu_type=A40
#$ -l h_rt=02:00:00
#$ -N ErniePixel_Eval_Manual
#$ -j y
#$ -o ErniePixel_Eval_ITT_JavaModel_BaliEval_Manual$JOB_ID.log

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/scripts/eval"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

# --- CONFIGURATION ---
FINETUNED_MODEL_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/finetune_transliteration/checkpoint-37500"
EVAL_OUTPUT_DIR="./eval_finetune_itt/javaModel/baliEval"
PYTHON_SCRIPT_NAME="eval_itt.py"
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"
EVAL_DATASET_PATH="izzako/balinese-pixelgpt-test"
EVAL_SPLIT="train"
MAX_NEW_TOKENS=256

# --- ENVIRONMENT SETUP AND EXECUTION ---
echo "==================================================================================="
echo "JOB NAME:         $JOB_NAME"
echo "JOB ID:           $JOB_ID"
echo "Evaluating Model (Manual Loop): ${FINETUNED_MODEL_PATH}"
echo "==================================================================================="

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
export HF_HOME="${CACHE_PATH}"

echo "Starting evaluation..."

python "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}" \
    --model_path "${FINETUNED_MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --dataset_name "${EVAL_DATASET_PATH}" \
    --eval_split "${EVAL_SPLIT}" \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --max_new_tokens ${MAX_NEW_TOKENS}

echo "Evaluation complete. Results are in ${EVAL_OUTPUT_DIR}"