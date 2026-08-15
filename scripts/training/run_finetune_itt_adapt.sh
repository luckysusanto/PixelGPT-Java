#!/usr/bin/env bash
# =========================================================================================
# QSUB SCRIPT FOR DUALGPT FEW-SHOT FINE-TUNING (Text-to-Image)
# =========================================================================================

# --- BU SCC Grid Engine Directives ---
#$ -P multilm
#$ -l gpus=4
#$ -pe omp 4
#$ -l gpu_type=A40
#$ -l h_rt=04:00:00
#$ -N DualGPT_Finetune_FewShot
#$ -j y
#$ -o DualGPT_Finetune_$JOB_ID.log

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/scripts/training"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

# Output for Fine-Tuning
OUTPUT_BASE_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/finetune_transliteration-adapt"
PYTHON_SCRIPT_NAME="run_finetune_itt_adapt.py" 
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"

# --- MODEL & DATASETS ---
# 1. Base Model (Can be the Pretrained one, or an already fine-tuned one)
MODEL_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/finetune_transliteration"

# 2. Datasets
ORIGINAL_DATASET="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt-poc-2/default/0.0.0"
NEW_DATASET="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___balinese-pixelgpt-poc/default/0.0.0"

# --- Configuration ---
NUM_SAMPLES=64          # 64 Bali, and (64 Bali + 64 Java) for mixed.
NUM_EPOCHS=20           # Higher for fine-tuning on small data
LEARNING_RATE=2e-5      

# --- Training Args ---
PER_DEVICE_BATCH_SIZE=2 # Small batch for safety
GRADIENT_ACCUMULATION_STEPS=4

# -----------------------------------------------------------------------------------------
# --- ENVIRONMENT ---
# -----------------------------------------------------------------------------------------
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
export HF_HOME="${CACHE_PATH}"
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=0

mkdir -p "${OUTPUT_BASE_DIR}"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Starting Few-Shot Fine-Tuning..."

accelerate launch --main_process_port 29601 "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}" \
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

echo "Fine-tuning experiments complete."