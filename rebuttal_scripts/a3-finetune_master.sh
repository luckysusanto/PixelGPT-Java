#!/bin/bash

# =============================================================================
# FINETUNING MASTER SCRIPT - Phase 1 & Phase 2
# =============================================================================
# Usage: ./finetune_master.sh
# Change ASSIGNED_TOKENIZER to run all experiments for that tokenizer
# =============================================================================

# =============================================================================
# 1. ASSIGNMENT CONFIGURATION (CHANGE THIS PER PERSON)
# =============================================================================
ASSIGNED_TOKENIZER="grapheme"  # Options: "llama2", "komodo", "mt5", "grapheme"

# =============================================================================
# 2. PATHS SETUP
# =============================================================================
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
MODELS_ROOT="${PROJECT_DIR}/hf_cache_updated/models"
OUTPUT_ROOT="${PROJECT_DIR}/rebuttal_experiment_output/finetune"
JOB_SCRIPT="${PROJECT_DIR}/rebuttal_scripts/b3-finetune_worker.sh"

DATASET_BALI="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-balinese-pixelgpt/default/0.0.0/4a31992e9daac65666bc3a814085bc8e48352210"
DATASET_JAVA="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-javanese-pixelgpt/default/0.0.0/c5fef00960c24c242eeb63befca570f18c1b3ec7"
DATASET_LAMPUNG="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-lampung-pixelgpt/default/0.0.0/efac497f7298c8234298699767f8f46db161eb73"
DATASET_SUNDA="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-sundanese-pixelgpt/default/0.0.0/dcd6ec05b35e3845faf930e710924114233c6f55"
# DATASET_BALI_PURE is the Balinese dataset tokenized with the native Balinese tokenizer.
# Used exclusively for bali_bali-tok experiments to measure the impact of pretraining
# Bali from scratch with its own tokenizer vs. using the Javanese tokenizer (DATASET_BALI).
DATASET_BALI_PURE="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-pure_bali-grapheme_experiment_only/default/0.0.0/97c9a64ef822783fe211834f590bc8e2c3a02b08"

# Tokenizer Constants
TOK_JAVA="izzako/javanese-llama-tokenizer"
TOK_BALI="izzako/balinese-llama-tokenizer"
TOK_SUNDA="izzako/sunda-llama-tokenizer"

# Define Port Ranges, Model Bases, and Tokenizer Bases
case $ASSIGNED_TOKENIZER in
    "llama2")   T_CODE="l2"; TEXT_COL="tok_llama2";   BASE_PORT=30700; MODEL_BASE="dualGPT-vocabResize-llama2";          TOK_BASE="ernie-research/DualGPT" ;;
    "komodo")   T_CODE="ko"; TEXT_COL="tok_komodo";   BASE_PORT=30710; MODEL_BASE="dualGPT-vocabResize-komodo";          TOK_BASE="Yellow-AI-NLP/komodo-7b-base" ;;
    "mt5")      T_CODE="m5"; TEXT_COL="tok_mt5";      BASE_PORT=30720; MODEL_BASE="dualGPT-vocabResize-mt5";             TOK_BASE="google/mt5-small" ;;
    "grapheme") T_CODE="gr"; TEXT_COL="tok_grapheme"; BASE_PORT=30730; MODEL_BASE="dualGPT-vocabResize-java-grapheme";   TOK_BASE="${TOK_JAVA}" ;;
esac

# =============================================================================
# 3. QUEUE FUNCTION
# =============================================================================
CURRENT_OFFSET=0

submit_exp() {
    local PHASE=$1
    local NAME=$2
    local PRETRAIN_BASE=$3
    local PRETRAIN_MODEL=$4
    local FINETUNE_LANG=$5
    local CUSTOM_TOK=$6
    local CUSTOM_DATASET=$7  # Optional: overrides the dataset resolved from FINETUNE_LANG in the worker

    local TARGET_MODEL="${MODELS_ROOT}/${PRETRAIN_MODEL}"
    local TARGET_TOK="${CUSTOM_TOK:-$TOK_BASE}"

    local TARGET_OUT="${OUTPUT_ROOT}/finetune-${ASSIGNED_TOKENIZER}/${PHASE}/${NAME}"
    local R_NAME="${ASSIGNED_TOKENIZER}-finetune-${PHASE}-${NAME}"
    local MY_PORT=$((BASE_PORT + CURRENT_OFFSET))
    ((CURRENT_OFFSET++))

    mkdir -p "$TARGET_OUT"
    local JOB_NAME="F_${T_CODE}_${PHASE:0:1}_${NAME}"
    local LOG_PATH="${TARGET_OUT}/scc_job.log"

    echo "Queueing: $JOB_NAME"
    echo "  ↳ Phase: $PHASE | Pretrain: $PRETRAIN_BASE | Finetune Lang: $FINETUNE_LANG"
    echo "  ↳ Model: $PRETRAIN_MODEL | Tok: $TARGET_TOK | Port: $MY_PORT"
    if [ -n "$CUSTOM_DATASET" ]; then
        echo "  ↳ Dataset override: $CUSTOM_DATASET"
    fi

    qsub -N "$JOB_NAME" \
         -o "$LOG_PATH" \
         -v PHASE="$PHASE",EXP_NAME="$NAME",FINETUNE_LANG="$FINETUNE_LANG",PRETRAIN_BASE="$PRETRAIN_BASE",TEXT_COL="$TEXT_COL",MODEL_PATH="$TARGET_MODEL",TOK_PATH="$TARGET_TOK",OUTPUT_DIR="$TARGET_OUT",RUN_NAME="$R_NAME",PORT="$MY_PORT",CUSTOM_DATASET="$CUSTOM_DATASET" \
         "$JOB_SCRIPT"
}

# =============================================================================
# 4. PHASE 1: MONOLINGUAL FINETUNING
# =============================================================================
# Full experiment matrix (Phase 1):
# Tokenizer | Pretrain Base         | Finetune Language
# ==========+=======================+==================
# Llama2    | Bali                  | Bali
#           | Java                  | Java
#           | Lampung               | Lampung
#           | Sunda                 | Sunda
#           | Bali + Java           | Java
#           | Bali + Java           | Bali       <- was missing for llama2/komodo/mt5
#           | Lampung + Sunda       | Sunda
#           | Lampung + Sunda       | Lampung    <- was missing for llama2/komodo/mt5
# Grapheme  | Bali (Java tok)       | Bali
#           | Bali (Bali tok)       | Bali
#           | Java                  | Java
#           | Lampung (Sunda tok)   | Lampung    (no Lampung-specific grapheme tok)
#           | Sunda                 | Sunda
#           | Bali + Java           | Java
#           | Bali + Java           | Bali
#           | Lampung + Sunda       | Sunda
#           | Lampung + Sunda       | Lampung
# Komodo    | (same as Llama2)
# mT5       | (same as Llama2)
# =============================================================================

echo "=========================================================="
echo "PHASE 1: MONOLINGUAL FINETUNING"
echo "Tokenizer: $ASSIGNED_TOKENIZER"
echo "=========================================================="

if [ "$ASSIGNED_TOKENIZER" == "grapheme" ]; then
    # 1. Bali (Java tok) pretrained -> finetune Bali
    submit_exp "phase1" "mono_bali_java-tok" "bali_java-tok" "dualGPT-vocabResize-java-grapheme" "bali" "${TOK_JAVA}"

    # 2. Bali (Bali tok) pretrained -> finetune Bali
    # Uses DATASET_BALI_PURE: Bali data tokenized with the native Balinese tokenizer.
    submit_exp "phase1" "mono_bali_bali-tok" "bali_bali-tok" "dualGPT-vocabResize-bali-grapheme" "bali" "${TOK_BALI}" "${DATASET_BALI_PURE}"

    # 3. Java pretrained -> finetune Java
    submit_exp "phase1" "mono_java" "java" "dualGPT-vocabResize-java-grapheme" "java" "${TOK_JAVA}"

    # 4. Lampung pretrained -> finetune Lampung
    # NOTE: No Lampung-specific grapheme tokenizer exists; Sunda grapheme tokenizer used.
    # This is intentional — measures partial tokenizer-script alignment.
    submit_exp "phase1" "mono_lampung" "lampung" "dualGPT-vocabResize-sunda-grapheme" "lampung" "${TOK_SUNDA}"

    # 5. Sunda pretrained -> finetune Sunda
    submit_exp "phase1" "mono_sunda" "sunda" "dualGPT-vocabResize-sunda-grapheme" "sunda" "${TOK_SUNDA}"

    # 6. Bali + Java pretrained -> finetune Java
    submit_exp "phase1" "dual_java_bali_ft-java" "dual_java_bali" "dualGPT-vocabResize-java-grapheme" "java" "${TOK_JAVA}"

    # 7. Bali + Java pretrained -> finetune Bali
    submit_exp "phase1" "dual_java_bali_ft-bali" "dual_java_bali" "dualGPT-vocabResize-java-grapheme" "bali" "${TOK_JAVA}"

    # 8. Lampung + Sunda pretrained -> finetune Sunda
    submit_exp "phase1" "dual_sunda_lampung_ft-sunda" "dual_sunda_lampung" "dualGPT-vocabResize-sunda-grapheme" "sunda" "${TOK_SUNDA}"

    # 9. Lampung + Sunda pretrained -> finetune Lampung
    submit_exp "phase1" "dual_sunda_lampung_ft-lampung" "dual_sunda_lampung" "dualGPT-vocabResize-sunda-grapheme" "lampung" "${TOK_SUNDA}"

elif [ "$ASSIGNED_TOKENIZER" == "llama2" ]; then
    submit_exp "phase1" "mono_bali"                    "bali"            "dualGPT-vocabResize-llama2" "bali"
    submit_exp "phase1" "mono_java"                    "java"            "dualGPT-vocabResize-llama2" "java"
    submit_exp "phase1" "mono_lampung"                 "lampung"         "dualGPT-vocabResize-llama2" "lampung"
    submit_exp "phase1" "mono_sunda"                   "sunda"           "dualGPT-vocabResize-llama2" "sunda"
    submit_exp "phase1" "dual_java_bali_ft-java"       "dual_java_bali"  "dualGPT-vocabResize-llama2" "java"
    submit_exp "phase1" "dual_java_bali_ft-bali"       "dual_java_bali"  "dualGPT-vocabResize-llama2" "bali"       # FIX: was missing
    submit_exp "phase1" "dual_sunda_lampung_ft-sunda"  "dual_sunda_lampung" "dualGPT-vocabResize-llama2" "sunda"
    submit_exp "phase1" "dual_sunda_lampung_ft-lampung" "dual_sunda_lampung" "dualGPT-vocabResize-llama2" "lampung" # FIX: was missing

elif [ "$ASSIGNED_TOKENIZER" == "komodo" ]; then
    submit_exp "phase1" "mono_bali"                    "bali"            "dualGPT-vocabResize-komodo" "bali"
    submit_exp "phase1" "mono_java"                    "java"            "dualGPT-vocabResize-komodo" "java"
    submit_exp "phase1" "mono_lampung"                 "lampung"         "dualGPT-vocabResize-komodo" "lampung"
    submit_exp "phase1" "mono_sunda"                   "sunda"           "dualGPT-vocabResize-komodo" "sunda"
    submit_exp "phase1" "dual_java_bali_ft-java"       "dual_java_bali"  "dualGPT-vocabResize-komodo" "java"
    submit_exp "phase1" "dual_java_bali_ft-bali"       "dual_java_bali"  "dualGPT-vocabResize-komodo" "bali"       # FIX: was missing
    submit_exp "phase1" "dual_sunda_lampung_ft-sunda"  "dual_sunda_lampung" "dualGPT-vocabResize-komodo" "sunda"
    submit_exp "phase1" "dual_sunda_lampung_ft-lampung" "dual_sunda_lampung" "dualGPT-vocabResize-komodo" "lampung" # FIX: was missing

elif [ "$ASSIGNED_TOKENIZER" == "mt5" ]; then
    submit_exp "phase1" "mono_bali"                    "bali"            "dualGPT-vocabResize-mt5" "bali"
    submit_exp "phase1" "mono_java"                    "java"            "dualGPT-vocabResize-mt5" "java"
    submit_exp "phase1" "mono_lampung"                 "lampung"         "dualGPT-vocabResize-mt5" "lampung"
    submit_exp "phase1" "mono_sunda"                   "sunda"           "dualGPT-vocabResize-mt5" "sunda"
    submit_exp "phase1" "dual_java_bali_ft-java"       "dual_java_bali"  "dualGPT-vocabResize-mt5" "java"
    submit_exp "phase1" "dual_java_bali_ft-bali"       "dual_java_bali"  "dualGPT-vocabResize-mt5" "bali"          # FIX: was missing
    submit_exp "phase1" "dual_sunda_lampung_ft-sunda"  "dual_sunda_lampung" "dualGPT-vocabResize-mt5" "sunda"
    submit_exp "phase1" "dual_sunda_lampung_ft-lampung" "dual_sunda_lampung" "dualGPT-vocabResize-mt5" "lampung"   # FIX: was missing

fi

# =============================================================================
# 5. PHASE 2: CROSSLINGUAL FINETUNING
# =============================================================================
# Full experiment matrix (Phase 2):
# Tokenizer | Pretrain Base         | Finetune Language
# ==========+=======================+==================
# Llama2    | Bali                  | Java
#           | Java                  | Bali
#           | Lampung               | Sunda
#           | Sunda                 | Lampung
# Grapheme  | Bali (Java tok)       | Java
#           | Bali (Bali tok)       | Java
#           | Java                  | Bali
#           | Lampung               | Sunda
#           | Sunda                 | Lampung
# Komodo    | (same as Llama2)
# mT5       | (same as Llama2)
# =============================================================================

echo ""
echo "=========================================================="
echo "PHASE 2: CROSSLINGUAL FINETUNING"
echo "Tokenizer: $ASSIGNED_TOKENIZER"
echo "=========================================================="

if [ "$ASSIGNED_TOKENIZER" == "grapheme" ]; then
    # 1. Bali (Java tok) pretrained -> finetune Java
    submit_exp "phase2" "cross_bali_java-tok_ft-java" "bali_java-tok" "dualGPT-vocabResize-java-grapheme" "java" "${TOK_JAVA}"

    # 2. Bali (Bali tok) pretrained -> finetune Java
    # NOTE: Intentional stress test — Bali tokenizer has never seen Java script.
    # Measures cross-script transfer with misaligned tokenizer.
    # Uses DATASET_BALI_PURE for the pretraining-side dataset consistency.
    submit_exp "phase2" "cross_bali_bali-tok_ft-java" "bali_bali-tok" "dualGPT-vocabResize-bali-grapheme" "java" "${TOK_BALI}" "${DATASET_BALI_PURE}"

    # 3. Java pretrained -> finetune Bali
    submit_exp "phase2" "cross_java_ft-bali" "java" "dualGPT-vocabResize-java-grapheme" "bali" "${TOK_JAVA}"

    # 4. Lampung pretrained -> finetune Sunda
    submit_exp "phase2" "cross_lampung_ft-sunda" "lampung" "dualGPT-vocabResize-sunda-grapheme" "sunda" "${TOK_SUNDA}"

    # 5. Sunda pretrained -> finetune Lampung
    submit_exp "phase2" "cross_sunda_ft-lampung" "sunda" "dualGPT-vocabResize-sunda-grapheme" "lampung" "${TOK_SUNDA}"

elif [ "$ASSIGNED_TOKENIZER" == "llama2" ]; then
    submit_exp "phase2" "cross_bali_ft-java"      "bali"    "dualGPT-vocabResize-llama2" "java"
    submit_exp "phase2" "cross_java_ft-bali"      "java"    "dualGPT-vocabResize-llama2" "bali"
    submit_exp "phase2" "cross_lampung_ft-sunda"  "lampung" "dualGPT-vocabResize-llama2" "sunda"
    submit_exp "phase2" "cross_sunda_ft-lampung"  "sunda"   "dualGPT-vocabResize-llama2" "lampung"

elif [ "$ASSIGNED_TOKENIZER" == "komodo" ]; then
    submit_exp "phase2" "cross_bali_ft-java"      "bali"    "dualGPT-vocabResize-komodo" "java"
    submit_exp "phase2" "cross_java_ft-bali"      "java"    "dualGPT-vocabResize-komodo" "bali"
    submit_exp "phase2" "cross_lampung_ft-sunda"  "lampung" "dualGPT-vocabResize-komodo" "sunda"
    submit_exp "phase2" "cross_sunda_ft-lampung"  "sunda"   "dualGPT-vocabResize-komodo" "lampung"

elif [ "$ASSIGNED_TOKENIZER" == "mt5" ]; then
    submit_exp "phase2" "cross_bali_ft-java"      "bali"    "dualGPT-vocabResize-mt5" "java"
    submit_exp "phase2" "cross_java_ft-bali"      "java"    "dualGPT-vocabResize-mt5" "bali"
    submit_exp "phase2" "cross_lampung_ft-sunda"  "lampung" "dualGPT-vocabResize-mt5" "sunda"
    submit_exp "phase2" "cross_sunda_ft-lampung"  "sunda"   "dualGPT-vocabResize-mt5" "lampung"

fi

echo ""
echo "=========================================================="
echo "All experiments submitted for: $ASSIGNED_TOKENIZER"
echo "=========================================================="