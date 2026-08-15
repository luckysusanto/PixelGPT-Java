#!/bin/bash
# =========================================================================================
# QSUB SUBMISSION SCRIPT FOR DUALGPT PRETRAINING (Constant LR + Epoch Folders)
# =========================================================================================

# --- BU SCC Grid Engine Directives ---
#$ -P multilm
#$ -l gpus=4
#$ -pe omp 4
#$ -l gpu_type=A40
#$ -l h_rt=36:00:00
#$ -N DualGPT_mixedPretrain-constantLR-Bali
#$ -j y
#$ -o DualGPT_mixedPretrain-constantLR-Bali-$JOB_ID.log

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/scripts/training"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

# This will be the "Root" folder.
# The script will automatically create "epoch-0", "epoch-1", etc. INSIDE this folder.
OUTPUT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/mixedPretrain-constantLR-Bali"

# Make sure this matches the filename of the Python code provided above
PYTHON_SCRIPT_NAME="run_erniepixel_constantLR.py" 
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"

# --- DATASETS ---
# 1. The original paired dataset
DATASET_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___balinese-pixelgpt-poc/default/0.0.0"

# 2. Text-Only Dataset
TEXT_ONLY_DATASET_PATH="${DATASET_PATH}"

# 3. Image-Only Dataset
IMAGE_ONLY_DATASET_PATH="${DATASET_PATH}"

# --- Dataset Arguments ---
IMAGE_COLUMN="pixel_values"
TEXT_COLUMN="token_ids"

# --- Training Arguments ---
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4

# -----------------------------------------------------------------------
# UPDATED OPTIMIZER SETTINGS
# -----------------------------------------------------------------------
LEARNING_RATE=5e-4 
WEIGHT_DECAY=0.1

# CRITICAL: Use 'constant_with_warmup' so the LR doesn't decay to 0 at the end of epoch-0.
# This ensures epoch-1 starts with a healthy Learning Rate.
LR_SCHEDULER="constant_with_warmup" 
WARMUP_STEPS=1000

# This controls how many sequential epoch folders are created.
# 5 means: epoch-0, epoch-1, epoch-2, epoch-3, epoch-4
NUM_EPOCHS=5 

DATALOADER_WORKERS=0
LOGGING_INTEGRATION="tensorboard"
RUN_NAME="dualgpt-pretrain-mixed-constant-$(date +%Y-%m-%d-%H-%M)"

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

echo "Starting pretraining job..."
echo "Output Root: ${OUTPUT_DIR}"

accelerate launch --main_process_port 29700 "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}" \
    --dataset_name "${DATASET_PATH}" \
    --text_only_dataset_name "${TEXT_ONLY_DATASET_PATH}" \
    --image_only_dataset_name "${IMAGE_ONLY_DATASET_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --image_column "${IMAGE_COLUMN}" \
    --text_column "${TEXT_COLUMN}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataloader_num_workers ${DATALOADER_WORKERS} \
    --do_train \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --lr_scheduler_type ${LR_SCHEDULER} \
    --warmup_steps ${WARMUP_STEPS} \
    --num_train_epochs ${NUM_EPOCHS} \
    --report_to "${LOGGING_INTEGRATION}" \
    --run_name "${RUN_NAME}" \
    --dataloader_drop_last

echo "Pretraining job complete."