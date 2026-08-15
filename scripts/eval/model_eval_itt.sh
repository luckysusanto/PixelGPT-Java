#!/bin/bash
# =========================================================================================
# COMBINED SUBMISSION SCRIPT FOR ERNIEPIXEL EVALUATION (JAVA & BALI)
# Runs Java evaluation followed immediately by Bali evaluation in one job.
# =========================================================================================

#$ -P multilm
#$ -l gpus=1
#$ -pe omp 1
#$ -l gpu_type=A40
#$ -l h_rt=04:00:00
#$ -N ITT_eval_mergedPretrain
#$ -j y
#$ -o ITT_eval_mergedPretrain_$JOB_ID.log

# =========================================================================================
# 1. USER CONFIGURATION (EDIT THESE 3 VARIABLES)
# =========================================================================================

FINETUNED_MODEL_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/mergedData-finetune-output/checkpoint-5064"
JAVA_EVAL_OUTPUT_DIR="./eval_finetune_itt/mergedPretrain/javaEval"
BALI_EVAL_OUTPUT_DIR="./eval_finetune_itt/mergedPretrain/baliEval"

# =========================================================================================
# 2. STATIC CONFIGURATION & ENVIRONMENT SETUP
# =========================================================================================

# Paths
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/scripts/eval"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

# Script Params
PYTHON_SCRIPT_NAME="eval_itt.py"
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"
MAX_NEW_TOKENS=256

# Setup Environment
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
export HF_HOME="${CACHE_PATH}"

echo "==================================================================================="
echo "JOB STARTED"
echo "Job ID: $JOB_ID"
echo "Model:  $FINETUNED_MODEL_PATH"
echo "==================================================================================="

# =========================================================================================
# 3. RUN TASK 1: JAVA EVALUATION
# =========================================================================================
JAVA_DATASET="izzako/javanese-pixelgpt-test"
JAVA_SPLIT="test"

echo ""
echo ">>> [1/2] STARTING JAVA EVALUATION"
echo "Dataset: $JAVA_DATASET | Split: $JAVA_SPLIT"
echo "Output:  $JAVA_EVAL_OUTPUT_DIR"
echo "-----------------------------------------------------------------------------------"

python "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}" \
    --model_path "${FINETUNED_MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --dataset_name "${JAVA_DATASET}" \
    --eval_split "${JAVA_SPLIT}" \
    --output_dir "${JAVA_EVAL_OUTPUT_DIR}" \
    --max_new_tokens ${MAX_NEW_TOKENS}

echo ">>> [1/2] JAVA EVALUATION COMPLETE"

# =========================================================================================
# 4. RUN TASK 2: BALI EVALUATION
# =========================================================================================
BALI_DATASET="izzako/balinese-pixelgpt-test"
BALI_SPLIT="test"

echo ""
echo ">>> [2/2] STARTING BALI EVALUATION"
echo "Dataset: $BALI_DATASET | Split: $BALI_SPLIT"
echo "Output:  $BALI_EVAL_OUTPUT_DIR"
echo "-----------------------------------------------------------------------------------"

python "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}" \
    --model_path "${FINETUNED_MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --dataset_name "${BALI_DATASET}" \
    --eval_split "${BALI_SPLIT}" \
    --output_dir "${BALI_EVAL_OUTPUT_DIR}" \
    --max_new_tokens ${MAX_NEW_TOKENS}

echo ">>> [2/2] BALI EVALUATION COMPLETE"

echo ""
echo "==================================================================================="
echo "ALL JOBS FINISHED SUCCESSFULLY"
echo "==================================================================================="