#!/bin/bash
# =========================================================================================
# QSUB SUBMISSION SCRIPT FOR PIXELS-ONLY DUALGPT PRETRAINING
# =========================================================================================

# --- BU SCC Grid Engine Directives ---
#$ -P multilm
#$ -l gpus=4
#$ -pe omp 4
#$ -l gpu_type=A40
#$ -l h_rt=25:00:00
#$ -N P_PixelsOnly_PretrainJB # <-- RENAMED JOB
#$ -j y
#$ -o P_PixelsOnly_PretrainJB-$JOB_ID.log # <-- RENAMED LOG

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/_updatedScripts/train"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

# Root Output Directory
OUTPUT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/_updated_experiment_output/Pretrain_PixelsOnly-JB" # <-- NEW OUTPUT DIR
PYTHON_SCRIPT_NAME="run_pretrain_imageOnly.py" 

# --- MODEL & TOKENIZER ---
# We are training from scratch, so we don't provide a model path.
# We still need a tokenizer path to build the model config, but it won't be used for data.
TOKENIZER_PATH="izzako/javanese-llama-tokenizer" # NEEDED FOR MODEL CONFIG, BUT IS NOT USED

# --- DATASETS ---
# You only need a dataset that has the 'pixel_values' column. The text column will be ignored.
DATASET_A_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt/default/0.0.0"

# Optional: Leave empty string "" if not using a second dataset
DATASET_B_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___balinese-pixelgpt/default/0.0.0"

# --- Training Arguments ---
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4

# --- Optimizer ---
LEARNING_RATE=5e-4 
WEIGHT_DECAY=0.1
LR_SCHEDULER="cosine" 
WARMUP_STEPS=1000
NUM_EPOCHS=50 
MAX_TIME_HOURS=24.0

# Validation settings
VAL_SAMPLES_PER_DS=1000
EVAL_STEPS=1000

DATALOADER_WORKERS=0
LOGGING_INTEGRATION="tensorboard"
RUN_NAME="dualgpt-pixels-only-$(date +%Y-%m-%d-%H-%M)"

# -----------------------------------------------------------------------------------------
# --- ENVIRONMENT SETUP (NO CHANGES HERE) ---
# -----------------------------------------------------------------------------------------
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
export HF_HOME="${CACHE_PATH}"
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=0

mkdir -p "${OUTPUT_DIR}"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Starting PIXELS-ONLY pretraining..."
echo "Output Root: ${OUTPUT_DIR}"
echo "Dataset A: ${DATASET_A_PATH}"

# Construct Command Array
CMD=(
    accelerate launch --main_process_port 29704 --num_processes 4 "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}"
    # --model_name_or_path is REMOVED to train from scratch
    --tokenizer_path "${TOKENIZER_PATH}" # Still needed for config
    --dataset_a_name "${DATASET_A_PATH}"
    --output_dir "${OUTPUT_DIR}"
    
    # --- THIS IS THE KEY CHANGE ---
    --image_only_pretrain True

    # --text_column is KEPT, only for initialization phase
    --text_column "grapheme_token_ids"
    --dataloader_num_workers ${DATALOADER_WORKERS}
    --do_train
    --do_eval 
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE}
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
    --learning_rate ${LEARNING_RATE}
    --weight_decay ${WEIGHT_DECAY}
    --lr_scheduler_type ${LR_SCHEDULER}
    --warmup_steps ${WARMUP_STEPS}
    --num_train_epochs ${NUM_EPOCHS}
    --report_to "${LOGGING_INTEGRATION}"
    --run_name "${RUN_NAME}"
    --dataloader_drop_last
    --validation_samples_per_dataset ${VAL_SAMPLES_PER_DS}
    --eval_steps ${EVAL_STEPS}
    --save_steps ${EVAL_STEPS}
    --evaluation_strategy "steps"
    --save_strategy "steps"
    --save_total_limit 2
    --load_best_model_at_end True
    --metric_for_best_model "eval_loss"
    --max_training_time_hours ${MAX_TIME_HOURS}
)

# This logic for Dataset B can remain, but DATASET_B_PATH is set to "" above
if [ ! -z "$DATASET_B_PATH" ]; then
    echo "Dataset B: ${DATASET_B_PATH}"
    CMD+=( --dataset_b_name "${DATASET_B_PATH}" )
else
    echo "Dataset B: Not provided"
fi

# Execute
"${CMD[@]}"

echo "Job complete."