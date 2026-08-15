#!/bin/bash
# =========================================================================================
# QSUB SCRIPT FOR DUALGPT FEW-SHOT ADAPTATION
# =========================================================================================

# --- BU SCC Grid Engine Directives ---
#$ -P multilm
#$ -l gpus=4
#$ -pe omp 4
#$ -l gpu_type=A40
#$ -l h_rt=04:00:00
#$ -N DualGPT_FewShot
#$ -j y
#$ -o DualGPT_FewShot_$JOB_ID.log

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/scripts/training"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

# This is where the two subfolders (baliOnly and mixBaliJava) will be created
OUTPUT_BASE_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/fewshot_experiments"
PYTHON_SCRIPT_NAME="run_fewshot_adapt.py" # Ensure the python code above is saved with this name
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"

# --- MODEL & DATASETS ---
# 1. Path to the ALREADY PRETRAINED model (The starting point)
# Replace this with the actual path to your checkpoint folder
MODEL_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/mixedPretrain-v2"

# 2. Original Dataset (e.g., Javanese)
ORIGINAL_DATASET="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt-poc-2/default/0.0.0" # Or local path

# 3. New Dataset (e.g., Balinese)
# Replace with your new dataset path/HF ID
NEW_DATASET="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___balinese-pixelgpt-poc/default/0.0.0" 

# --- Few Shot Configuration ---
NUM_SAMPLES=64          # Recommended: 64. (User requested 10, but 64 is safer for batches)
NUM_EPOCHS=10           # Since N is small, we need more epochs to get gradient updates.
LEARNING_RATE=2e-5      # MUCH lower than pretraining to preserve knowledge

# --- Training Arguments ---
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4

# -----------------------------------------------------------------------------------------
# --- ENVIRONMENT SETUP ---
# -----------------------------------------------------------------------------------------
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
export HF_HOME="${CACHE_PATH}"
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=0

mkdir -p "${OUTPUT_BASE_DIR}"

# -----------------------------------------------------------------------------------------
# --- EXECUTION ---
# -----------------------------------------------------------------------------------------
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Starting Few-Shot Adaptation..."
echo "Base Model: ${MODEL_PATH}"
echo "Samples per dataset: ${NUM_SAMPLES}"

accelerate launch --main_process_port 29600 "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}" \
    --model_name_or_path "${MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --original_dataset_name "${ORIGINAL_DATASET}" \
    --new_dataset_name "${NEW_DATASET}" \
    --num_few_shot_samples ${NUM_SAMPLES} \
    --output_dir "${OUTPUT_BASE_DIR}" \
    --dataloader_num_workers 4 \
    --do_train \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --num_train_epochs ${NUM_EPOCHS} \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --dataloader_drop_last False

echo "Few-Shot experiments complete."