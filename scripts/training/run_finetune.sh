#!/bin/bash
# =========================================================================================
# QSUB SUBMISSION SCRIPT FOR DUALGPT FINE-TUNING (TRANSLITERATION) ON BU SCC
# =========================================================================================

# --- BU SCC Grid Engine Directives ---
#$ -P multilm
#$ -l gpus=2
#$ -pe omp 2
#$ -l gpu_type=A40
#$ -l h_rt=24:00:00
#$ -N DualGPT_Finetune_Translit
#$ -j y
#$ -o DualGPT_Finetune_$JOB_ID.log

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/scripts/training"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

# Output for the Fine-tuned model
OUTPUT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/finetune_transliteration"

# Save the python code from the previous answer as this filename:
PYTHON_SCRIPT_NAME="run_finetune.py" 
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"

# --- MODEL ARGUMENTS ---
# Point this to the output directory of your PRETRAINING job.
# Example: ".../experiment_output/mixedPretrain" or ".../mixedPretrain/checkpoint-50000"
PRETRAINED_MODEL_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/mixedPretrain"

# --- DATASETS ---
# The Fine-tuning dataset (Must have Image and Text pairs)
DATASET_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt-poc-2/default/0.0.0"

# --- Dataset Columns ---
# Ensure these match your Fine-Tuning dataset columns
IMAGE_COLUMN="pixel_values"
TEXT_COLUMN="token_ids"

# --- Training Arguments ---
# Fine-tuning typically uses smaller batches and lower learning rates
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=2e-5
NUM_EPOCHS=5
DATALOADER_WORKERS=0

# --- Logging ---
LOGGING_INTEGRATION="tensorboard"
RUN_NAME="dualgpt-finetune-translit-$(date +%Y-%m-%d-%H-%M)"

# -----------------------------------------------------------------------------------------
# --- ENVIRONMENT SETUP ---
# -----------------------------------------------------------------------------------------
echo "==================================================================================="
echo "JOB NAME:         $JOB_NAME"
echo "JOB ID:           $JOB_ID"
echo "CUDA VISIBLE DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==================================================================================="

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
export HF_HOME="${CACHE_PATH}"
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=0

mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------------------------------------------------
# --- EXECUTION ---
# -----------------------------------------------------------------------------------------
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "Starting fine-tuning job..."
echo "Loading weights from: ${PRETRAINED_MODEL_PATH}"

accelerate launch --main_process_port 29701 "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}" \
    --model_name_or_path "${PRETRAINED_MODEL_PATH}" \
    --dataset_name "${DATASET_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --image_column "${IMAGE_COLUMN}" \
    --text_column "${TEXT_COLUMN}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataloader_num_workers ${DATALOADER_WORKERS} \
    --do_train \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --report_to "${LOGGING_INTEGRATION}" \
    --run_name "${RUN_NAME}" \
    --dataloader_drop_last \
    --bf16

echo "Fine-tuning job complete."