#!/bin/bash
#$ -pe omp 4
#$ -l gpus=2
#$ -l gpu_type=A100
#$ -l gpu_memory=80G
#$ -l h_rt=25:00:00
#$ -j y

# =============================================================================
# FINETUNING WORKER SCRIPT
# =============================================================================
# This script is called by the master script with the following env vars:
# - PHASE: "phase1" or "phase2"
# - EXP_NAME: Experiment name
# - FINETUNE_LANG: Target language for finetuning (bali, java, lampung, sunda)
# - PRETRAIN_BASE: Pretrained model base name
# - TEXT_COL: Text column name (tok_llama2, tok_komodo, tok_mt5, tok_grapheme)
# - MODEL_PATH: Path to pretrained model
# - TOK_PATH: Path to tokenizer (passed for reference; tokenizer loaded from MODEL_PATH)
# - OUTPUT_DIR: Output directory
# - RUN_NAME: Run name for logging
# - PORT: Master process port
# =============================================================================

PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
SCRIPT_PATH="${PROJECT_DIR}/rebuttal_scripts/finetune_erniepixel-withLog.py"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env"
CACHE_DIR="${PROJECT_DIR}/hf_cache_updated"

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}"
export HF_HOME="${CACHE_DIR}"
export MASTER_ADDR=$HOSTNAME

# Dataset paths
DATASET_BALI="${CACHE_DIR}/datasets/Exqrch___rebuttal-balinese-pixelgpt/default/0.0.0/4a31992e9daac65666bc3a814085bc8e48352210"
DATASET_JAVA="${CACHE_DIR}/datasets/Exqrch___rebuttal-javanese-pixelgpt/default/0.0.0/c5fef00960c24c242eeb63befca570f18c1b3ec7"
DATASET_LAMPUNG="${CACHE_DIR}/datasets/Exqrch___rebuttal-lampung-pixelgpt/default/0.0.0/efac497f7298c8234298699767f8f46db161eb73"
DATASET_SUNDA="${CACHE_DIR}/datasets/Exqrch___rebuttal-sundanese-pixelgpt/default/0.0.0/dcd6ec05b35e3845faf930e710924114233c6f55"

# =============================================================================
# Helper function to select dataset path based on language
# =============================================================================
get_dataset_path() {
    local LANG=$1
    case $LANG in
        "bali")    echo "$DATASET_BALI" ;;
        "java")    echo "$DATASET_JAVA" ;;
        "lampung") echo "$DATASET_LAMPUNG" ;;
        "sunda")   echo "$DATASET_SUNDA" ;;
        *)         echo "ERROR: Unknown language $LANG" ;;
    esac
}

# =============================================================================
# Get the correct dataset for finetuning language.
# CUSTOM_DATASET (passed from master) takes precedence when set —
# used for bali_bali-tok experiments which need DATASET_BALI_PURE
# rather than the standard DATASET_BALI.
# FIX: exit 1 inside $(...) only kills the subshell, not the parent script.
# Capture output first, then check for error prefix and exit the parent.
# =============================================================================
if [ -n "$CUSTOM_DATASET" ]; then
    DATASET_PATH="$CUSTOM_DATASET"
else
    DATASET_PATH=$(get_dataset_path "$FINETUNE_LANG")
    if [[ "$DATASET_PATH" == ERROR* ]]; then
        echo "$DATASET_PATH"
        exit 1
    fi
fi

echo "=========================================================="
echo "FINETUNING EXPERIMENT"
echo "=========================================================="
echo "Phase:           $PHASE"
echo "Experiment:      $EXP_NAME"
echo "Finetune Lang:   $FINETUNE_LANG"
echo "Pretrain Base:   $PRETRAIN_BASE"
echo "Text Column:     $TEXT_COL"
echo "Model Path:      $MODEL_PATH"
echo "Tokenizer:       $TOK_PATH"
echo "Dataset:         $DATASET_PATH"
echo "Output Dir:      $OUTPUT_DIR"
echo "Port:            $PORT"
echo "=========================================================="

# =============================================================================
# Run finetuning with accelerate
# NOTE: --tokenizer_path is passed for reference/logging but the Python script
# always loads the tokenizer from --model_name_or_path for consistency.
# =============================================================================
accelerate launch --main_process_port "$PORT" --num_processes 2 "$SCRIPT_PATH" \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_path "$TOK_PATH" \
    --dataset_a_name "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --text_column "$TEXT_COL" \
    --run_name "$RUN_NAME"

echo "Done."