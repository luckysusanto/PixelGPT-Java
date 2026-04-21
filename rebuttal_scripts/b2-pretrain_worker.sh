#!/bin/bash
#$ -pe omp 4
#$ -l gpus=2
#$ -l gpu_type=A40
#$ -l h_rt=48:00:00
#$ -j y

PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
SCRIPT_PATH="${PROJECT_DIR}/rebuttal_scripts/pretrain_erniepixel.py"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env"
CACHE_DIR="${PROJECT_DIR}/hf_cache_updated"

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}"
export HF_HOME="${CACHE_DIR}"
export MASTER_ADDR=$HOSTNAME

echo "=========================================================="
echo "EXPERIMENT: $EXP_NAME"
echo "PORT:       $PORT"
echo "TOKENIZER:  $TEXT_COL"
echo "=========================================================="

# Use the dynamic port passed from the launcher
accelerate launch --main_process_port "$PORT" --num_processes 2 "$SCRIPT_PATH" \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_path "$TOK_PATH" \
    --dataset_a_name "$DS_A" \
    ${DS_B:+--dataset_b_name "$DS_B"} \
    --output_dir "$OUTPUT_DIR" \
    --text_column "$TEXT_COL" \
    --run_name "$RUN_NAME"

echo "Done."