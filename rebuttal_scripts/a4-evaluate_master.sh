#!/bin/bash

# =============================================================================
# EVALUATION MASTER SCRIPT - Phase 1 & Phase 2
# =============================================================================
# Mirrors finetune_master.sh exactly: same experiment matrix, same naming.
# For each finetuned model, queues one qsub job that:
#   1. Loads the finetuned checkpoint from its OUTPUT_DIR
#   2. Runs evaluation on the matching language's TEST split
#   3. Writes eval_result.json + eval_verbose.csv into the SAME model dir
#
# Usage:
#   1. Set ASSIGNED_TOKENIZER below (matches the value used during finetuning)
#   2. ./b4-eval_master.sh
# =============================================================================

# =============================================================================
# 1. ASSIGNMENT CONFIGURATION (CHANGE THIS PER PERSON / PER RUN)
# =============================================================================
ASSIGNED_TOKENIZER="grapheme"   # Options: "llama2", "komodo", "mt5", "grapheme"

# =============================================================================
# 2. PATHS SETUP — MUST MATCH finetune_master.sh
# =============================================================================
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
OUTPUT_ROOT="${PROJECT_DIR}/rebuttal_experiment_output/finetune"
JOB_SCRIPT="${PROJECT_DIR}/rebuttal_scripts/b4-evaluate_worker.sh"
HF_CACHE_DIR="${PROJECT_DIR}/hf_cache_updated"
# Datasets (same as finetune; we use the test split of these)
DATASET_BALI="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-balinese-pixelgpt/default/0.0.0/4a31992e9daac65666bc3a814085bc8e48352210"
DATASET_JAVA="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-javanese-pixelgpt/default/0.0.0/c5fef00960c24c242eeb63befca570f18c1b3ec7"
DATASET_LAMPUNG="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-lampung-pixelgpt/default/0.0.0/efac497f7298c8234298699767f8f46db161eb73"
DATASET_SUNDA="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-sundanese-pixelgpt/default/0.0.0/dcd6ec05b35e3845faf930e710924114233c6f55"
DATASET_BALI_PURE="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-pure_bali-grapheme_experiment_only/default/0.0.0/97c9a64ef822783fe211834f590bc8e2c3a02b08"

# Tokenizer constants (same as finetune)
TOK_JAVA="izzako/javanese-llama-tokenizer"
TOK_BALI="izzako/balinese-llama-tokenizer"
TOK_SUNDA="izzako/sunda-llama-tokenizer"

# Per-tokenizer base config (same as finetune)
case $ASSIGNED_TOKENIZER in
    "llama2")   T_CODE="l2"; TOK_BASE="ernie-research/DualGPT" ;;
    "komodo")   T_CODE="ko"; TOK_BASE="Yellow-AI-NLP/komodo-7b-base" ;;
    "mt5")      T_CODE="m5"; TOK_BASE="google/mt5-small" ;;
    "grapheme") T_CODE="gr"; TOK_BASE="${TOK_JAVA}" ;;
esac

# Map a language code to its dataset path
lang_to_dataset() {
    case "$1" in
        bali)    echo "$DATASET_BALI" ;;
        java)    echo "$DATASET_JAVA" ;;
        lampung) echo "$DATASET_LAMPUNG" ;;
        sunda)   echo "$DATASET_SUNDA" ;;
        *)       echo "" ;;
    esac
}

# =============================================================================
# 3. QUEUE FUNCTION
# =============================================================================
submit_eval() {
    local PHASE=$1
    local NAME=$2
    local FINETUNE_LANG=$3
    local CUSTOM_TOK=$4         # Optional (grapheme uses different toks per language)
    local CUSTOM_DATASET=$5     # Optional (e.g. bali_bali-tok uses DATASET_BALI_PURE)

    # Reconstruct the model path from the finetune output convention
    local MODEL_DIR="${OUTPUT_ROOT}/finetune-${ASSIGNED_TOKENIZER}/${PHASE}/${NAME}"
    local TARGET_TOK="${CUSTOM_TOK:-$TOK_BASE}"
    local TARGET_DS="${CUSTOM_DATASET:-$(lang_to_dataset "$FINETUNE_LANG")}"

    if [ -z "$TARGET_DS" ]; then
        echo "ERROR: no dataset resolved for language '$FINETUNE_LANG' (exp=$NAME). Skipping."
        return
    fi

    if [ ! -d "$MODEL_DIR" ]; then
        echo "WARNING: model dir does not exist (yet): $MODEL_DIR — skipping eval submit."
        return
    fi

    local JOB_NAME="E_${T_CODE}_${PHASE:0:1}_${NAME}"
    local LOG_PATH="${MODEL_DIR}/scc_eval.log"

    echo "Queueing eval: $JOB_NAME"
    echo "  ↳ Model    : $MODEL_DIR"
    echo "  ↳ Tokenizer: $TARGET_TOK"
    echo "  ↳ Dataset  : $TARGET_DS  (lang=$FINETUNE_LANG)"

    qsub -N "$JOB_NAME" \
         -o "$LOG_PATH" \
         -v MODEL_PATH="$MODEL_DIR",TOK_PATH="$TARGET_TOK",DATASET_PATH="$TARGET_DS",LANG_CODE="$FINETUNE_LANG",OUTPUT_DIR="$MODEL_DIR",PROJECT_DIR="$PROJECT_DIR" \
         "$JOB_SCRIPT"
}

# =============================================================================
# 4. PHASE 1: MONOLINGUAL EVAL
# =============================================================================
echo "=========================================================="
echo "PHASE 1: MONOLINGUAL EVAL"
echo "Tokenizer: $ASSIGNED_TOKENIZER"
echo "=========================================================="

if [ "$ASSIGNED_TOKENIZER" == "grapheme" ]; then
    submit_eval "phase1" "mono_bali_java-tok"           "bali"    "${TOK_JAVA}"
    submit_eval "phase1" "mono_bali_bali-tok"           "bali"    "${TOK_BALI}"   "${DATASET_BALI_PURE}"
    submit_eval "phase1" "mono_java"                    "java"    "${TOK_JAVA}"
    submit_eval "phase1" "mono_lampung"                 "lampung" "${TOK_SUNDA}"
    submit_eval "phase1" "mono_sunda"                   "sunda"   "${TOK_SUNDA}"
    submit_eval "phase1" "dual_java_bali_ft-java"       "java"    "${TOK_JAVA}"
    submit_eval "phase1" "dual_java_bali_ft-bali"       "bali"    "${TOK_JAVA}"
    submit_eval "phase1" "dual_sunda_lampung_ft-sunda"  "sunda"   "${TOK_SUNDA}"
    submit_eval "phase1" "dual_sunda_lampung_ft-lampung" "lampung" "${TOK_SUNDA}"

elif [ "$ASSIGNED_TOKENIZER" == "llama2" ] || [ "$ASSIGNED_TOKENIZER" == "komodo" ] || [ "$ASSIGNED_TOKENIZER" == "mt5" ]; then
    submit_eval "phase1" "mono_bali"                     "bali"
    submit_eval "phase1" "mono_java"                     "java"
    submit_eval "phase1" "mono_lampung"                  "lampung"
    submit_eval "phase1" "mono_sunda"                    "sunda"
    submit_eval "phase1" "dual_java_bali_ft-java"        "java"
    submit_eval "phase1" "dual_java_bali_ft-bali"        "bali"
    submit_eval "phase1" "dual_sunda_lampung_ft-sunda"   "sunda"
    submit_eval "phase1" "dual_sunda_lampung_ft-lampung" "lampung"
fi

# =============================================================================
# 5. PHASE 2: CROSSLINGUAL EVAL
# =============================================================================
echo ""
echo "=========================================================="
echo "PHASE 2: CROSSLINGUAL EVAL"
echo "Tokenizer: $ASSIGNED_TOKENIZER"
echo "=========================================================="

if [ "$ASSIGNED_TOKENIZER" == "grapheme" ]; then
    submit_eval "phase2" "cross_bali_java-tok_ft-java"   "java"    "${TOK_JAVA}"
    submit_eval "phase2" "cross_java_ft-bali"            "bali"    "${TOK_JAVA}"
    submit_eval "phase2" "cross_lampung_ft-sunda"        "sunda"   "${TOK_SUNDA}"
    submit_eval "phase2" "cross_sunda_ft-lampung"        "lampung" "${TOK_SUNDA}"

elif [ "$ASSIGNED_TOKENIZER" == "llama2" ] || [ "$ASSIGNED_TOKENIZER" == "komodo" ] || [ "$ASSIGNED_TOKENIZER" == "mt5" ]; then
    submit_eval "phase2" "cross_bali_ft-java"     "java"
    submit_eval "phase2" "cross_java_ft-bali"     "bali"
    submit_eval "phase2" "cross_lampung_ft-sunda" "sunda"
    submit_eval "phase2" "cross_sunda_ft-lampung" "lampung"
fi

echo ""
echo "=========================================================="
echo "All eval jobs submitted for: $ASSIGNED_TOKENIZER"
echo "Each finetune model dir will receive eval_result.json + eval_verbose.csv on completion."
echo "=========================================================="