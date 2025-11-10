#!/bin/bash
# =========================================================================================
# QSUB SUBMISSION SCRIPT FOR DUALGPT PRETRAINING ON BOSTON UNIVERSITY (BU) SCC
#
# This script is tailored to the BU SCC environment and uses Accelerate's
# automatic GPU detection via the CUDA_VISIBLE_DEVICES environment variable.
# =========================================================================================

# --- BU SCC Grid Engine Directives ---
#$ -P multilm                                    # Project allocation
#$ -l gpus=1                                     # Request 4 GPUs on a single node. The scheduler will set CUDA_VISIBLE_DEVICES.
#$ -l gpu_type=A40                               # Request specific GPU type
#$ -l h_rt=24:00:00                              # Request 24 hours of runtime
#$ -N OneGPU_OneCore_DualGPT_java_pretrain                      # Job name
#$ -j y                                          # Join stdout and stderr
#$ -o 1Core_1GPU_dualgpt_pretrain_$JOB_ID.log               # Specify the log file name

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/scripts/training"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
DATASET_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt-poc/default/0.0.0"
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"
OUTPUT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/experiment_output/pretraining"
PYTHON_SCRIPT_NAME="run_pretraining_erniepixel.py" 

# --- Dataset Arguments ---
IMAGE_COLUMN="pixel_values"
TEXT_COLUMN="token_ids"

# --- Training Arguments ---
PER_DEVICE_BATCH_SIZE=2 #1x1024 image patches is huge, I'm making the batch miniscule.
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-5
NUM_EPOCHS=3 # We continue one by one, just a test run
DATALOADER_WORKERS=0

# --- Logging ---
LOGGING_INTEGRATION="tensorboard"
RUN_NAME="dualgpt-pretrain-$(date +%Y-%m-%d-%H-%M)"

# -----------------------------------------------------------------------------------------
# --- ENVIRONMENT SETUP ---
# -----------------------------------------------------------------------------------------

echo "==================================================================================="
echo "JOB NAME:         $JOB_NAME"
echo "JOB ID:           $JOB_ID"
echo "HOSTNAME:         $HOSTNAME"
echo "CPU SLOTS ($NSLOTS):  $NSLOTS" # This is the number of CPU cores
echo "WORKING DIRECTORY:  $SGE_O_WORKDIR"
# --- Add a diagnostic line to prove GPU detection will work
echo "CUDA VISIBLE DEVICES: $CUDA_VISIBLE_DEVICES" # This should be set by the scheduler (e.g., "0,1")
echo "==================================================================================="

# Activate your Python environment
source "${VENV_PATH}/bin/activate"

# Add project root to PYTHONPATH so `from src...` works
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Set Hugging Face cache directory
export HF_HOME="${CACHE_PATH}"

# Set up environment variables for distributed training
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=0

# -----------------------------------------------------------------------------------------
# --- PREPARATION AND DEBUGGING ---
# -----------------------------------------------------------------------------------------

# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# Log the configuration Accelerate is using
echo "--- ACCELERATE CONFIGURATION DETECTED ---"
accelerate env
echo "-----------------------------------------"

# -----------------------------------------------------------------------------------------
# --- GPU MEMORY LOGGING ---
# -----------------------------------------------------------------------------------------
GPU_LOG_FILE="${OUTPUT_DIR}/gpu_memory_usage_${JOB_ID}.csv"
echo "Starting GPU memory logging to ${GPU_LOG_FILE}"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 5 >> "${GPU_LOG_FILE}" &
GPU_MONITOR_PID=$!

# -----------------------------------------------------------------------------------------
# --- EXECUTION ---
# -----------------------------------------------------------------------------------------

# We are running the python script directly to get the cleanest possible error message.
echo "Starting DEBUGGING job on a single GPU..."
python "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}" \
    --dataset_name "${DATASET_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --image_column "${IMAGE_COLUMN}" \
    --text_column "${TEXT_COLUMN}" \
    --output_dir "${OUTPUT_DIR}/DEBUG_RUN" \
    --dataloader_num_workers ${DATALOADER_WORKERS} \
    --do_train \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --report_to "${LOGGING_INTEGRATION}" \
    --run_name "DEBUG_RUN_${RUN_NAME}" \
    --dataloader_drop_last True \
    --logging_steps 10 # Log more frequently
# -----------------------------------------------------------------------------------------
# --- CLEANUP ---
# -----------------------------------------------------------------------------------------
echo "Training finished. Stopping GPU memory logger (PID: ${GPU_MONITOR_PID})."
kill $GPU_MONITOR_PID

echo "Pretraining job complete."