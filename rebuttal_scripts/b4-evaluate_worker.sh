#!/bin/bash
#$ -S /bin/bash
#$ -l h_rt=04:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -j y
#$ -V
# =============================================================================
# EVAL WORKER SCRIPT
# Runs ONE evaluation job for ONE finetuned model.
#
# Required env vars (passed via qsub -v):
#   MODEL_PATH       - path to the finetuned checkpoint
#   TOK_PATH         - tokenizer path / HF id used during finetuning
#   TEXT_COL         - tokenized text column name (e.g. tok_grapheme)
#   DATASET_PATH     - path to the HF dataset directory (or HF hub id)
#   LANG_CODE        - language code for the test set (used for naming/logs)
#   OUTPUT_DIR       - where eval_result.json + eval_verbose.csv will be written
#                      (typically the same as the finetune model dir)
#   PROJECT_DIR      - project root containing src/ and the eval script
#
# Optional:
#   EVAL_SPLIT          (default: test)
#   MAX_NEW_TOKENS      (default: 256)
#   MAX_EVAL_SAMPLES    (default: unset = full eval)
#   REPETITION_PENALTY  (default: 1.0)
# =============================================================================

set -e

cd "$PROJECT_DIR"

# Activate environment (adjust if your setup differs)
# module load conda
# source activate pixelgpt

# Defaults
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
SCRIPT_PATH="${PROJECT_DIR}/rebuttal_scripts/evaluate_pixel.py"
CACHE_DIR="${PROJECT_DIR}/hf_cache_updated"
VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env"

source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}"
export HF_HOME="${CACHE_DIR}"
export PYTHONPATH="${PROJECT_DIR}"
echo "=========================================================="
echo "EVAL WORKER STARTING"
echo "  Model      : $MODEL_PATH"
echo "  Tokenizer  : $TOK_PATH"
echo "  Text col   : $TEXT_COL"
echo "  Dataset    : $DATASET_PATH (split=$EVAL_SPLIT)"
echo "  Lang       : $LANG_CODE"
echo "  Output dir : $OUTPUT_DIR"
echo "  Max tokens : $MAX_NEW_TOKENS"
echo "  Rep penalty: $REPETITION_PENALTY"
if [ -n "$MAX_EVAL_SAMPLES" ]; then
    echo "  Sample cap : $MAX_EVAL_SAMPLES"
fi
echo "=========================================================="

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=""
if [ -n "$MAX_EVAL_SAMPLES" ]; then
    EXTRA_ARGS="--max_eval_samples $MAX_EVAL_SAMPLES"
fi

python "$SCRIPT_PATH" \
    --model_path        "$MODEL_PATH" \
    --tokenizer_path    "$TOK_PATH" \
    --dataset_path      "$DATASET_PATH" \
    --lang_code         "$LANG_CODE" \
    --output_dir        "$OUTPUT_DIR" \
    --eval_split        "$EVAL_SPLIT" \
    --max_new_tokens    "$MAX_NEW_TOKENS" \
    --repetition_penalty "$REPETITION_PENALTY" \
    $EXTRA_ARGS

echo "=========================================================="
echo "EVAL DONE"
echo "  Results written to: $OUTPUT_DIR/eval_result.json"
echo "  Verbose CSV:        $OUTPUT_DIR/eval_verbose.csv"
echo "=========================================================="