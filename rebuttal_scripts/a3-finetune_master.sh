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

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
DATASET_BALI="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-balinese-pixelgpt/default/0.0.0/4a31992e9daac65666bc3a814085bc8e48352210"
DATASET_JAVA="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-javanese-pixelgpt/default/0.0.0/c5fef00960c24c242eeb63befca570f18c1b3ec7"
DATASET_LAMPUNG="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-lampung-pixelgpt/default/0.0.0/efac497f7298c8234298699767f8f46db161eb73"
DATASET_SUNDA="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-sundanese-pixelgpt/default/0.0.0/dcd6ec05b35e3845faf930e710924114233c6f55"
# DATASET_BALI_PURE is the Balinese dataset tokenized with the native Balinese tokenizer.
# Used exclusively for bali_bali-tok experiments to measure the impact of pretraining
# Bali from scratch with its own tokenizer vs. using the Javanese tokenizer (DATASET_BALI).
DATASET_BALI_PURE="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-pure_bali-grapheme_experiment_only/default/0.0.0/97c9a64ef822783fe211834f590bc8e2c3a02b08"

# -----------------------------------------------------------------------------
# Pretrained Model Paths — SET THESE TO YOUR PRETRAINED CHECKPOINTS
# Each variable points to the model pretrained on that language/combination.
# For grapheme, there are separate models per tokenizer-script alignment.
# -----------------------------------------------------------------------------

# --- Shared tokenizers (llama2, komodo, mt5) ---
PRETRAIN_BALI="PLACEHOLDER_pretrain_bali"
PRETRAIN_JAVA="PLACEHOLDER_pretrain_java"
PRETRAIN_LAMPUNG="PLACEHOLDER_pretrain_lampung"
PRETRAIN_SUNDA="PLACEHOLDER_pretrain_sunda"
PRETRAIN_DUAL_JAVA_BALI="PLACEHOLDER_pretrain_dual_java_bali"
PRETRAIN_DUAL_SUNDA_LAMPUNG="PLACEHOLDER_pretrain_dual_sunda_lampung"

# --- Grapheme-specific (separate model per script-aligned tokenizer) ---
PRETRAIN_GRAPHEME_BALI_JAVA_TOK="PLACEHOLDER_pretrain_grapheme_bali_java-tok"
PRETRAIN_GRAPHEME_BALI_BALI_TOK="PLACEHOLDER_pretrain_grapheme_bali_bali-tok"
PRETRAIN_GRAPHEME_JAVA="PLACEHOLDER_pretrain_grapheme_java"
PRETRAIN_GRAPHEME_LAMPUNG="PLACEHOLDER_pretrain_grapheme_lampung"
PRETRAIN_GRAPHEME_SUNDA="PLACEHOLDER_pretrain_grapheme_sunda"
PRETRAIN_GRAPHEME_DUAL_JAVA_BALI="PLACEHOLDER_pretrain_grapheme_dual_java_bali"
PRETRAIN_GRAPHEME_DUAL_SUNDA_LAMPUNG="PLACEHOLDER_pretrain_grapheme_dual_sunda_lampung"

# -----------------------------------------------------------------------------
# Tokenizer Constants
# -----------------------------------------------------------------------------
TOK_JAVA="izzako/javanese-llama-tokenizer"
TOK_BALI="izzako/balinese-llama-tokenizer"
TOK_SUNDA="izzako/sunda-llama-tokenizer"

# Define Port Ranges, Model Bases, and Tokenizer Bases
case $ASSIGNED_TOKENIZER in
    "llama2")   T_CODE="l2"; TEXT_COL="tok_llama2";   BASE_PORT=30700; TOK_BASE="ernie-research/DualGPT" ;;
    "komodo")   T_CODE="ko"; TEXT_COL="tok_komodo";   BASE_PORT=30710; TOK_BASE="Yellow-AI-NLP/komodo-7b-base" ;;
    "mt5")      T_CODE="m5"; TEXT_COL="tok_mt5";      BASE_PORT=30720; TOK_BASE="google/mt5-small" ;;
    "grapheme") T_CODE="gr"; TEXT_COL="tok_grapheme"; BASE_PORT=30730; TOK_BASE="${TOK_JAVA}" ;;
esac

# =============================================================================
# 3. QUEUE FUNCTION
# =============================================================================
CURRENT_OFFSET=0

submit_exp() {
    local PHASE=$1
    local NAME=$2
    local PRETRAIN_BASE=$3
    local PRETRAIN_MODEL=$4      # Full model path (resolved from PRETRAIN_* variables above)
    local FINETUNE_LANG=$5
    local CUSTOM_TOK=$6
    local CUSTOM_DATASET=$7      # Optional: overrides the dataset resolved from FINETUNE_LANG in the worker

    local TARGET_MODEL="$PRETRAIN_MODEL"
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
    echo "  ↳ Model: $TARGET_MODEL | Tok: $TARGET_TOK | Port: $MY_PORT"
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
    submit_exp "phase1" "mono_bali_java-tok"          "bali_java-tok"       "$PRETRAIN_GRAPHEME_BALI_JAVA_TOK"      "bali"    "${TOK_JAVA}"

    # 2. Bali (Bali tok) pretrained -> finetune Bali
    # Uses DATASET_BALI_PURE: Bali data tokenized with the native Balinese tokenizer.
    submit_exp "phase1" "mono_bali_bali-tok"          "bali_bali-tok"       "$PRETRAIN_GRAPHEME_BALI_BALI_TOK"      "bali"    "${TOK_BALI}"  "${DATASET_BALI_PURE}"

    # 3. Java pretrained -> finetune Java
    submit_exp "phase1" "mono_java"                   "java"                "$PRETRAIN_GRAPHEME_JAVA"               "java"    "${TOK_JAVA}"

    # 4. Lampung pretrained -> finetune Lampung
    # NOTE: No Lampung-specific grapheme tokenizer exists; Sunda grapheme tokenizer used.
    # This is intentional — measures partial tokenizer-script alignment.
    submit_exp "phase1" "mono_lampung"                "lampung"             "$PRETRAIN_GRAPHEME_LAMPUNG"            "lampung" "${TOK_SUNDA}"

    # 5. Sunda pretrained -> finetune Sunda
    submit_exp "phase1" "mono_sunda"                  "sunda"               "$PRETRAIN_GRAPHEME_SUNDA"              "sunda"   "${TOK_SUNDA}"

    # 6. Bali + Java pretrained -> finetune Java
    submit_exp "phase1" "dual_java_bali_ft-java"      "dual_java_bali"      "$PRETRAIN_GRAPHEME_DUAL_JAVA_BALI"     "java"    "${TOK_JAVA}"

    # 7. Bali + Java pretrained -> finetune Bali
    submit_exp "phase1" "dual_java_bali_ft-bali"      "dual_java_bali"      "$PRETRAIN_GRAPHEME_DUAL_JAVA_BALI"     "bali"    "${TOK_JAVA}"

    # 8. Lampung + Sunda pretrained -> finetune Sunda
    submit_exp "phase1" "dual_sunda_lampung_ft-sunda"   "dual_sunda_lampung" "$PRETRAIN_GRAPHEME_DUAL_SUNDA_LAMPUNG" "sunda"   "${TOK_SUNDA}"

    # 9. Lampung + Sunda pretrained -> finetune Lampung
    submit_exp "phase1" "dual_sunda_lampung_ft-lampung" "dual_sunda_lampung" "$PRETRAIN_GRAPHEME_DUAL_SUNDA_LAMPUNG" "lampung" "${TOK_SUNDA}"

elif [ "$ASSIGNED_TOKENIZER" == "llama2" ]; then
    submit_exp "phase1" "mono_bali"                     "bali"               "$PRETRAIN_BALI"               "bali"
    submit_exp "phase1" "mono_java"                     "java"               "$PRETRAIN_JAVA"               "java"
    submit_exp "phase1" "mono_lampung"                  "lampung"            "$PRETRAIN_LAMPUNG"            "lampung"
    submit_exp "phase1" "mono_sunda"                    "sunda"              "$PRETRAIN_SUNDA"              "sunda"
    submit_exp "phase1" "dual_java_bali_ft-java"        "dual_java_bali"     "$PRETRAIN_DUAL_JAVA_BALI"     "java"
    submit_exp "phase1" "dual_java_bali_ft-bali"        "dual_java_bali"     "$PRETRAIN_DUAL_JAVA_BALI"     "bali"    # FIX: was missing
    submit_exp "phase1" "dual_sunda_lampung_ft-sunda"   "dual_sunda_lampung" "$PRETRAIN_DUAL_SUNDA_LAMPUNG" "sunda"
    submit_exp "phase1" "dual_sunda_lampung_ft-lampung" "dual_sunda_lampung" "$PRETRAIN_DUAL_SUNDA_LAMPUNG" "lampung" # FIX: was missing

elif [ "$ASSIGNED_TOKENIZER" == "komodo" ]; then
    submit_exp "phase1" "mono_bali"                     "bali"               "$PRETRAIN_BALI"               "bali"
    submit_exp "phase1" "mono_java"                     "java"               "$PRETRAIN_JAVA"               "java"
    submit_exp "phase1" "mono_lampung"                  "lampung"            "$PRETRAIN_LAMPUNG"            "lampung"
    submit_exp "phase1" "mono_sunda"                    "sunda"              "$PRETRAIN_SUNDA"              "sunda"
    submit_exp "phase1" "dual_java_bali_ft-java"        "dual_java_bali"     "$PRETRAIN_DUAL_JAVA_BALI"     "java"
    submit_exp "phase1" "dual_java_bali_ft-bali"        "dual_java_bali"     "$PRETRAIN_DUAL_JAVA_BALI"     "bali"    # FIX: was missing
    submit_exp "phase1" "dual_sunda_lampung_ft-sunda"   "dual_sunda_lampung" "$PRETRAIN_DUAL_SUNDA_LAMPUNG" "sunda"
    submit_exp "phase1" "dual_sunda_lampung_ft-lampung" "dual_sunda_lampung" "$PRETRAIN_DUAL_SUNDA_LAMPUNG" "lampung" # FIX: was missing

elif [ "$ASSIGNED_TOKENIZER" == "mt5" ]; then
    submit_exp "phase1" "mono_bali"                     "bali"               "$PRETRAIN_BALI"               "bali"
    submit_exp "phase1" "mono_java"                     "java"               "$PRETRAIN_JAVA"               "java"
    submit_exp "phase1" "mono_lampung"                  "lampung"            "$PRETRAIN_LAMPUNG"            "lampung"
    submit_exp "phase1" "mono_sunda"                    "sunda"              "$PRETRAIN_SUNDA"              "sunda"
    submit_exp "phase1" "dual_java_bali_ft-java"        "dual_java_bali"     "$PRETRAIN_DUAL_JAVA_BALI"     "java"
    submit_exp "phase1" "dual_java_bali_ft-bali"        "dual_java_bali"     "$PRETRAIN_DUAL_JAVA_BALI"     "bali"    # FIX: was missing
    submit_exp "phase1" "dual_sunda_lampung_ft-sunda"   "dual_sunda_lampung" "$PRETRAIN_DUAL_SUNDA_LAMPUNG" "sunda"
    submit_exp "phase1" "dual_sunda_lampung_ft-lampung" "dual_sunda_lampung" "$PRETRAIN_DUAL_SUNDA_LAMPUNG" "lampung" # FIX: was missing

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
    submit_exp "phase2" "cross_bali_java-tok_ft-java" "bali_java-tok" "$PRETRAIN_GRAPHEME_BALI_JAVA_TOK" "java" "${TOK_JAVA}"

    # 2. Bali (Bali tok) pretrained -> finetune Java
    # NOTE: Intentional stress test — Bali tokenizer has never seen Java script.
    # Measures cross-script transfer with misaligned tokenizer.
    submit_exp "phase2" "cross_bali_bali-tok_ft-java" "bali_bali-tok" "$PRETRAIN_GRAPHEME_BALI_BALI_TOK" "java" "${TOK_BALI}" "${DATASET_BALI_PURE}"

    # 3. Java pretrained -> finetune Bali
    submit_exp "phase2" "cross_java_ft-bali"          "java"          "$PRETRAIN_GRAPHEME_JAVA"          "bali"    "${TOK_JAVA}"

    # 4. Lampung pretrained -> finetune Sunda
    submit_exp "phase2" "cross_lampung_ft-sunda"      "lampung"       "$PRETRAIN_GRAPHEME_LAMPUNG"       "sunda"   "${TOK_SUNDA}"

    # 5. Sunda pretrained -> finetune Lampung
    submit_exp "phase2" "cross_sunda_ft-lampung"      "sunda"         "$PRETRAIN_GRAPHEME_SUNDA"         "lampung" "${TOK_SUNDA}"

elif [ "$ASSIGNED_TOKENIZER" == "llama2" ]; then
    submit_exp "phase2" "cross_bali_ft-java"     "bali"    "$PRETRAIN_BALI"    "java"
    submit_exp "phase2" "cross_java_ft-bali"     "java"    "$PRETRAIN_JAVA"    "bali"
    submit_exp "phase2" "cross_lampung_ft-sunda" "lampung" "$PRETRAIN_LAMPUNG" "sunda"
    submit_exp "phase2" "cross_sunda_ft-lampung" "sunda"   "$PRETRAIN_SUNDA"   "lampung"

elif [ "$ASSIGNED_TOKENIZER" == "komodo" ]; then
    submit_exp "phase2" "cross_bali_ft-java"     "bali"    "$PRETRAIN_BALI"    "java"
    submit_exp "phase2" "cross_java_ft-bali"     "java"    "$PRETRAIN_JAVA"    "bali"
    submit_exp "phase2" "cross_lampung_ft-sunda" "lampung" "$PRETRAIN_LAMPUNG" "sunda"
    submit_exp "phase2" "cross_sunda_ft-lampung" "sunda"   "$PRETRAIN_SUNDA"   "lampung"

elif [ "$ASSIGNED_TOKENIZER" == "mt5" ]; then
    submit_exp "phase2" "cross_bali_ft-java"     "bali"    "$PRETRAIN_BALI"    "java"
    submit_exp "phase2" "cross_java_ft-bali"     "java"    "$PRETRAIN_JAVA"    "bali"
    submit_exp "phase2" "cross_lampung_ft-sunda" "lampung" "$PRETRAIN_LAMPUNG" "sunda"
    submit_exp "phase2" "cross_sunda_ft-lampung" "sunda"   "$PRETRAIN_SUNDA"   "lampung"

fi

echo ""
echo "=========================================================="
echo "All experiments submitted for: $ASSIGNED_TOKENIZER"
echo "=========================================================="