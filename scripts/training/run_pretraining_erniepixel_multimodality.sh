#!/bin/bash
# =========================================================================================
# QSUB SUBMISSION SCRIPT FOR DUALGPT PRETRAINING ON BOSTON UNIVERSITY (BU) SCC
# =========================================================================================

# --- BU SCC Grid Engine Directives ---
#$ -P multilm
#$ -l gpus=4
#$ -pe omp 4
#$ -l gpu_type=A40
#$ -l h_rt=36:00:00
#$ -N DualGPT_mixedPretrain-v2
#$ -j y
#$ -o DualGPT_mixedPretrain-v2$JOB_ID.log

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/scripts/training"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"
OUTPUT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/mixedPretrain-v2"
PYTHON_SCRIPT_NAME="run_pretraining_erniepixel_multimodality.py" 
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"

# --- DATASETS ---
# 1. The original paired dataset
DATASET_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt-poc-2/default/0.0.0"

# 2. [NEW] Text-Only Dataset Path (Set this to your actual path/HF ID)
# If you don't have one yet, you can comment this line out, but the script expects it if flags are passed.
TEXT_ONLY_DATASET_PATH="${DATASET_PATH}"

# 3. [NEW] Image-Only Dataset Path (Set this to your actual path/HF ID)
IMAGE_ONLY_DATASET_PATH="${DATASET_PATH}"


# --- Dataset Arguments ---
# ENSURE ALL DATASETS USE THESE COLUMN NAMES
IMAGE_COLUMN="pixel_values"
TEXT_COLUMN="token_ids"

# --- Training Arguments ---
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
# Standard pretraining LR is usually around 5e-4 or 1e-4. 
# Since we removed the auto-scaling, we set the MAX LR here directly.
LEARNING_RATE=5e-4 
WEIGHT_DECAY=0.1
LR_SCHEDULER="cosine"
WARMUP_STEPS=1000 # or use --warmup_ratio 0.03
NUM_EPOCHS=1 
DATALOADER_WORKERS=0
# --- Logging ---
LOGGING_INTEGRATION="tensorboard"
RUN_NAME="dualgpt-pretrain-mixed-$(date +%Y-%m-%d-%H-%M)"

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

# NOTE: If you are missing one of the datasets (e.g., Image Only), 
# simply remove the --image_only_dataset_name flag below.

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