#!/bin/bash

# =============================================================================
# 1. ASSIGNMENT CONFIGURATION (CHANGE THIS PER PERSON)
# =============================================================================
ASSIGNED_TOKENIZER="grapheme" # Adapt this

# =============================================================================
# 2. PATHS SETUP
# =============================================================================
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt" # Adapt this
MODELS_ROOT="${PROJECT_DIR}/hf_cache_updated/models"
OUTPUT_ROOT="${PROJECT_DIR}/rebuttal_experiment_output/${ASSIGNED_TOKENIZER}"
JOB_SCRIPT="${PROJECT_DIR}/rebuttal_scripts/b2-pretrain_worker.sh"

# Adapt these
DATASET_BALI="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-balinese-pixelgpt/default/0.0.0/4a31992e9daac65666bc3a814085bc8e48352210"
DATASET_JAVA="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-javanese-pixelgpt/default/0.0.0/c5fef00960c24c242eeb63befca570f18c1b3ec7"
DATASET_LAMPUNG="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-lampung-pixelgpt/default/0.0.0/efac497f7298c8234298699767f8f46db161eb73"
DATASET_SUNDA="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-sundanese-pixelgpt/default/0.0.0/dcd6ec05b35e3845faf930e710924114233c6f55"
DATASET_BALI_PURE="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-pure_bali-grapheme_experiment_only/default/0.0.0/97c9a64ef822783fe211834f590bc8e2c3a02b08"

# Tokenizer Constants
TOK_JAVA="izzako/javanese-llama-tokenizer"
TOK_BALI="izzako/balinese-llama-tokenizer"
TOK_SUNDA="izzako/sunda-llama-tokenizer"

# Define Port Ranges, Model Bases, and Tokenizer Bases
case $ASSIGNED_TOKENIZER in
    "llama2")   T_CODE="l2"; TEXT_COL="tok_llama2";  BASE_PORT=29700; MODEL_BASE="dualGPT-vocabResize-llama2"; TOK_BASE="ernie-research/DualGPT" ;;
    "komodo")   T_CODE="ko"; TEXT_COL="tok_komodo";  BASE_PORT=29710; MODEL_BASE="dualGPT-vocabResize-komodo"; TOK_BASE="Yellow-AI-NLP/komodo-7b-base" ;;
    "mt5")      T_CODE="m5"; TEXT_COL="tok_mt5";     BASE_PORT=29720; MODEL_BASE="dualGPT-vocabResize-mt5";    TOK_BASE="google/mt5-small" ;;
    "grapheme") T_CODE="gr"; TEXT_COL="tok_grapheme"; BASE_PORT=29730; MODEL_BASE="dualGPT-vocabResize-java-grapheme"; TOK_BASE="${TOK_JAVA}" ;;
esac

# =============================================================================
# 3. QUEUE FUNCTION
# =============================================================================
CURRENT_OFFSET=0

submit_exp() {
    local NAME=$1
    local DA=$2
    local DB=$3
    local CUSTOM_MODEL=$4 
    local CUSTOM_TOK=$5 # <--- NEW PARAMETER
    
    local MODEL_NAME="${CUSTOM_MODEL:-$MODEL_BASE}"
    local TARGET_MODEL="${MODELS_ROOT}/${MODEL_NAME}"
    local TARGET_TOK="${CUSTOM_TOK:-$TOK_BASE}"
    
    local TARGET_OUT="${OUTPUT_ROOT}/${NAME}"
    local R_NAME="${ASSIGNED_TOKENIZER}-${NAME}"
    local MY_PORT=$((BASE_PORT + CURRENT_OFFSET))
    ((CURRENT_OFFSET++))

    mkdir -p "$TARGET_OUT"
    local JOB_NAME="P_${T_CODE}_${NAME}"
    local LOG_PATH="${TARGET_OUT}/scc_job.log"

    echo "Queueing: $JOB_NAME"
    echo "  ↳ Model: $MODEL_NAME | Tok: $TARGET_TOK | Port: $MY_PORT"

    # Pass TOK_PATH to the worker
    qsub -N "$JOB_NAME" \
         -o "$LOG_PATH" \
         -v PORT="$MY_PORT",EXP_NAME="$NAME",DS_A="$DA",DS_B="$DB",TEXT_COL="$TEXT_COL",MODEL_PATH="$TARGET_MODEL",TOK_PATH="$TARGET_TOK",OUTPUT_DIR="$TARGET_OUT",RUN_NAME="$R_NAME" \
         "$JOB_SCRIPT"
}

# =============================================================================
# 4. SUBMIT ALL 6 EXPERIMENTS
# =============================================================================

if [ "$ASSIGNED_TOKENIZER" == "grapheme" ]; then
    # 1. Javanese Mono
    submit_exp "DEBUG-mono_javanese" "${DATASET_JAVA}" "" "dualGPT-vocabResize-java-grapheme" "${TOK_JAVA}"
    
    # 2. Balinese Mono & Pure Bali Mono
    submit_exp "DEBUG-mono_balinese" "${DATASET_BALI}" "" "dualGPT-vocabResize-java-grapheme" "${TOK_JAVA}"
    submit_exp "DEBUG-mono_pure-balinese" "${DATASET_BALI_PURE}" "" "dualGPT-vocabResize-bali-grapheme" "${TOK_BALI}"
    
    # 3. Sundanese Mono
    submit_exp "DEBUG-mono_sundanese" "${DATASET_SUNDA}" "" "dualGPT-vocabResize-sunda-grapheme" "${TOK_SUNDA}"
    
    # 4. Lampung Mono
    submit_exp "DEBUG-mono_lampung" "${DATASET_LAMPUNG}" "" "dualGPT-vocabResize-sunda-grapheme" "${TOK_SUNDA}"
    
    # 5. Dual Javanese + Balinese (Uses JAVA Tokenizer/Model)
    submit_exp "DEBUG-dual_java_bali" "${DATASET_JAVA}" "${DATASET_BALI}" "dualGPT-vocabResize-java-grapheme" "${TOK_JAVA}"
    
    # 6. Dual Sundanese + Lampung
    submit_exp "DEBUG-dual_sunda_lampung" "${DATASET_SUNDA}" "${DATASET_LAMPUNG}" "dualGPT-vocabResize-sunda-grapheme" "${TOK_SUNDA}"

else
    # STANDARD LOGIC (Llama2, Komodo, mT5) - TOK_PATH will default to TOK_BASE inside function
    submit_exp "mono_javanese"      "${DATASET_JAVA}" ""
    submit_exp "mono_balinese"      "${DATASET_BALI}" ""
    submit_exp "mono_sundanese"     "${DATASET_SUNDA}" ""
    submit_exp "mono_lampung"       "${DATASET_LAMPUNG}" ""
    submit_exp "dual_java_bali"     "${DATASET_JAVA}" "${DATASET_BALI}"
    submit_exp "dual_sunda_lampung" "${DATASET_SUNDA}" "${DATASET_LAMPUNG}"
fi