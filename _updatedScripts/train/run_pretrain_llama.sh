#!/bin/bash
# =========================================================================================
# QSUB SUBMISSION SCRIPT FOR MERGED DUALGPT PRETRAINING (Time-Capped + Early Stopping)
# =========================================================================================

# --- BU SCC Grid Engine Directives ---
#$ -P multilm
#$ -l gpus=4
#$ -pe omp 4
#$ -l gpu_type=A40
#$ -l h_rt=25:00:00  # Give 1 additional hour for data loading and training.
#$ -N P_J-Llama_Tokenizer
#$ -j y
#$ -o P_J-Llama_Tokenizer-$JOB_ID.log

# --- Paths ---
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/_updatedScripts/train"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
CACHE_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

# Root Output Directory
OUTPUT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/_updated_experiment_output/Pretrain_Java-Llama_Tokenizer"
PYTHON_SCRIPT_NAME="pretrain_erniepixel.py" 
TOKENIZER_PATH="izzako/javanese-llama-tokenizer"
DEFAULT_TOKENIZER_PATH="ernie-research/DualGPT"
MODEL_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/models/dualGPT-vocabResize"
DEFAULT_MODEL_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/models/dualGPT"

# --- DATASETS ---
# Define your datasets here. 
# DATASET_A_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___balinese-pixelgpt/default/0.0.0"
DATASET_A_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___javanese-pixelgpt/default/0.0.0"

# Optional: Leave empty string "" if not using a second dataset
DATASET_B_PATH=""
# DATASET_B_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/izzako___balinese-pixelgpt/default/0.0.0"

TEXT_COLUMN="grapheme_token_ids"
LLAMA_TEXT_COLUMN="llama_token_ids"
# --- Training Arguments ---
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4

# --- Optimizer (Using Standard Cosine now) ---
LEARNING_RATE=5e-4 
WEIGHT_DECAY=0.1
LR_SCHEDULER="cosine" 
WARMUP_STEPS=1000

# High Epoch count because we rely on Early Stopping/Time Limit now
NUM_EPOCHS=50 

# Time Limit in Hours 
MAX_TIME_HOURS=24.0 # TEST RUN

# Validation settings
VAL_SAMPLES_PER_DS=1000
EVAL_STEPS=1000 # Check validation every 1000 steps (~30-60 mins depending on speed)

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

# Construct Command Array
CMD=(
    accelerate launch --main_process_port 29703 --num_processes 4 "${PROJECT_DIR}/${PYTHON_SCRIPT_NAME}"
    --model_name_or_path "${DEFAULT_MODEL_PATH}"
    --dataset_a_name "${DATASET_A_PATH}"
    --tokenizer_path "${DEFAULT_TOKENIZER_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --dataloader_num_workers ${DATALOADER_WORKERS}
    --do_train
    --do_eval
    --text_column "${LLAMA_TEXT_COLUMN}"
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

# Only add Dataset B argument if the path is not empty
if [ ! -z "$DATASET_B_PATH" ]; then
    echo "Dataset B: ${DATASET_B_PATH}"
    CMD+=( --dataset_b_name "${DATASET_B_PATH}" )
else
    echo "Dataset B: Not provided (Training Single Language)"
fi

# Execute
"${CMD[@]}"

echo "Job complete."