#!/bin/bash
# =========================================================================================
# QSUB SUBMISSION SCRIPT FOR MERGED DUALGPT PRETRAINING (Scratch + Constant LR)
# =========================================================================================

# --- BU SCC Grid Engine Directives ---
#$ -P multilm
#$ -l gpus=4
#$ -pe omp 4
#$ -l gpu_type=A40
#$ -l h_rt=36:00:00
#$ -N DualGPT_MergedPretrain
#$ -j y
#$ -o DualGPT_MergedPretrain-$JOB_ID.log

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/scripts/training"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

# Root Output Directory
OUTPUT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/merged-pretrain-output"
PYTHON_SCRIPT_NAME="run_erniepixel_constantLR_multilang.py" 
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"

# --- DATASETS ---
# Define your two datasets here.
DATASET_A_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt-poc-2/default/0.0.0"
DATASET_B_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___balinese-pixelgpt-poc/default/0.0.0"

# --- Training Arguments ---
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4

# --- Optimizer ---
LEARNING_RATE=5e-4 
WEIGHT_DECAY=0.1
LR_SCHEDULER="constant_with_warmup" 
WARMUP_STEPS=1000

# Total number of epoch folders to create
NUM_EPOCHS=1 

DATALOADER_WORKERS=0
LOGGING_INTEGRATION="tensorboard"
RUN_NAME="dualgpt-merged-$(date +%Y-%m-%d-%H-%M)"

# -----------------------------------------------------------------------------------------
# --- ENVIRONMENT SETUP ---
# -----------------------------------------------------------------------------------------
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
export HF_HOME="${CACHE_PATH}"
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=0

mkdir -p "${OUTPUT_DIR}"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Starting merged pretraining..."
echo "Output Root: ${OUTPUT_DIR}"
echo "Dataset A: ${DATASET_A_PATH}"
echo "Dataset B: ${DATASET_B_PATH}"

accelerate launch --main_process_port 29700 "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}" \
    --dataset_a_name "${DATASET_A_PATH}" \
    --dataset_b_name "${DATASET_B_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
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

echo "Job complete."