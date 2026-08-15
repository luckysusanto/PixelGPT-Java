#!/bin/bash
# =========================================================================================
# QSUB SUBMISSION SCRIPT FOR DUALGPT FINE-TUNING (TRANSLITERATION)
# Updated for: Multi-Dataset, Early Stopping, Time Caps, and Cluster Stability
# =========================================================================================

# --- BU SCC Grid Engine Directives ---
#$ -P multilm
#$ -l gpus=4
#$ -pe omp 4
#$ -l gpu_type=A40
#$ -l h_rt=26:00:00
#$ -N F_B-BO-LT
#$ -j y
#$ -o F_B-BO-LT-$JOB_ID.log

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/_updatedScripts/train"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

# Output for the Fine-tuned model
OUTPUT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/_updated_experiment_output/Finetune_Bali-Bali_Only-Llama_Tokenizer"

# The Python script we just created
PYTHON_SCRIPT_NAME="finetune_transliteration_erniepixel.py" 
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"

# --- MODEL ARGUMENTS ---
# Point this to the output directory of your PRETRAINING job.
# No need to pin it to certain checkpoint, the main folder is already BEST, unless you want latest checkpoint instead.
PRETRAINED_MODEL_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/_updated_experiment_output/Pretrain_Bali-Llama_Tokenizer"

# --- DATASETS ---
# Define your datasets here. 
# DATASET_A_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt/default/0.0.0"
DATASET_A_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___balinese-pixelgpt/default/0.0.0"

# Optional: Leave empty string "" if not using a second dataset
DATASET_B_PATH=""
# DATASET_B_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___balinese-pixelgpt/default/0.0.0"

# --- Dataset Columns ---
IMAGE_COLUMN="pixel_values"
TEXT_COLUMN="llama_token_ids"

# --- Training Arguments ---
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5
NUM_EPOCHS=10  # High number, relying on Early Stopping to finish it
DATALOADER_WORKERS=0 # Must be 0 to prevent hangs

# Validation / Stopping
VAL_SAMPLES=1000
EVAL_STEPS=1000
MAX_TIME_HOURS=24.0

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

# --- CRITICAL EXPORTS FOR STABILITY ---
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Fixes dataset loading hangs on shared filesystems
export HF_DATASETS_LOCKING_DISABLED=true 

echo "Starting fine-tuning job..."
echo "Loading weights from: ${PRETRAINED_MODEL_PATH}"

# -----------------------------------------------------------------------------------------
# --- CONSTRUCT COMMAND ---
# -----------------------------------------------------------------------------------------

CMD=(
    accelerate launch 
    --num_processes 4 
    --main_process_port 29720 
    "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}"
    --model_name_or_path "${PRETRAINED_MODEL_PATH}"
    --dataset_a_name "${DATASET_A_PATH}"
    --dataset_b_name "${DATASET_B_PATH}"
    --tokenizer_path "${TOKENIZER_PATH}"
    --image_column "${IMAGE_COLUMN}"
    --text_column "${TEXT_COLUMN}"
    --output_dir "${OUTPUT_DIR}"
    --dataloader_num_workers ${DATALOADER_WORKERS}
    --do_train
    --do_eval
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE}
    --per_device_eval_batch_size ${PER_DEVICE_BATCH_SIZE}
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
    --learning_rate ${LEARNING_RATE}
    --num_train_epochs ${NUM_EPOCHS}
    --report_to "${LOGGING_INTEGRATION}"
    --run_name "${RUN_NAME}"
    --dataloader_drop_last
    --bf16
    --validation_samples_per_dataset ${VAL_SAMPLES}
    --max_training_time_hours ${MAX_TIME_HOURS}
    --evaluation_strategy "steps"
    --save_strategy "steps"
    --eval_steps ${EVAL_STEPS}
    --save_steps ${EVAL_STEPS}
    --save_total_limit 2
    --load_best_model_at_end True
    --metric_for_best_model "eval_loss" 
    # Critical DDP Flags
    --ddp_find_unused_parameters True
    --ddp_broadcast_buffers False
)

# Add Dataset B if it exists
if [ ! -z "$DATASET_B_PATH" ]; then
    echo "Adding Dataset B: ${DATASET_B_PATH}"
    CMD+=( --dataset_b_name "${DATASET_B_PATH}" )
else
    echo "Training on Single Dataset A"
fi

# Execute
"${CMD[@]}"

echo "Fine-tuning job complete."