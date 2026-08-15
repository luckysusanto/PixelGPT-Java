#!/bin/bash
# -----------------------------------------------------------------------------
# Run tokenizer_stats.py over the 5 locally-cached datasets.
#
# Each invocation produces one JSON file in $OUTPUT_DIR.
#
# Edit the TEMPLATE block below as needed:
#   - PROJECT_DIR         : root of your PixelGPT project (mirrors your setup)
#   - HF_HOME             : where HF caches tokenizers/datasets
#   - OUTPUT_DIR          : where the per-language JSON results are written
#   - TOK_*               : tokenizer paths (HF hub ids OR local paths)
#   - DATASET_*           : local dataset cache paths
# -----------------------------------------------------------------------------
set -euo pipefail

# 1) Load your .env if present
set -a
[ -f .env ] && source .env
set +a

# -----------------------------------------------------------------------------
# === TEMPLATE: edit these to taste =========================================
# -----------------------------------------------------------------------------
PROJECT_DIR="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
export HF_HOME="${PROJECT_DIR}/hf_cache_updated"
export PYTHONPATH="../../src:${PYTHONPATH:-}"

OUTPUT_DIR="${PROJECT_DIR}/rebuttal_experiment_output/tokenizer_stats_pt1"
mkdir -p "${OUTPUT_DIR}"

# Shared (non-grapheme) tokenizers
TOK_LLAMA2="ernie-research/DualGPT"
TOK_KOMODO="Yellow-AI-NLP/komodo-7b-base"
TOK_MT5="google/mt5-small"

# Local dataset cache paths (revision dirs inside HF cache)
DATASET_BALI="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-balinese-pixelgpt/default/0.0.0/4a31992e9daac65666bc3a814085bc8e48352210"
DATASET_JAVA="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-javanese-pixelgpt/default/0.0.0/c5fef00960c24c242eeb63befca570f18c1b3ec7"
DATASET_LAMPUNG="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-lampung-pixelgpt/default/0.0.0/efac497f7298c8234298699767f8f46db161eb73"
DATASET_SUNDA="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-sundanese-pixelgpt/default/0.0.0/dcd6ec05b35e3845faf930e710924114233c6f55"
DATASET_BALI_PURE="${PROJECT_DIR}/hf_cache_updated/datasets/Exqrch___rebuttal-pure_bali-grapheme_experiment_only/default/0.0.0/97c9a64ef822783fe211834f590bc8e2c3a02b08"

# Map each language entry to (lang_label, dataset_path, grapheme_tokenizer).
# The grapheme tokenizer here mirrors what your bash launcher used when the
# dataset was built (so that vocab_size / unk_token_id line up with the stored
# tok_grapheme ids):
#   - javanese, balinese  -> izzako/javanese-llama-tokenizer
#   - sundanese, lampung  -> izzako/sunda-llama-tokenizer
#   - pbali (pure_bali)   -> izzako/balinese-llama-tokenizer
# -----------------------------------------------------------------------------

# Each entry: "label|dataset_path|grapheme_tokenizer_path"
ENTRIES=(
    "javanese|${DATASET_JAVA}|izzako/javanese-llama-tokenizer"
    "sundanese|${DATASET_SUNDA}|izzako/sunda-llama-tokenizer"
    "balinese|${DATASET_BALI}|izzako/javanese-llama-tokenizer"
    "lampung|${DATASET_LAMPUNG}|izzako/sunda-llama-tokenizer"
    "pbali|${DATASET_BALI_PURE}|izzako/balinese-llama-tokenizer"
)

# -----------------------------------------------------------------------------
# Run loop
# -----------------------------------------------------------------------------
for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r LANG_LABEL DATASET_PATH TOK_GRAPHEME <<< "${entry}"

    echo "----------------------------------------------------"
    echo "TOKENIZER STATS  ::  ${LANG_LABEL}"
    echo "  dataset    : ${DATASET_PATH}"
    echo "  grapheme   : ${TOK_GRAPHEME}"
    echo "----------------------------------------------------"

    OUT_JSON="${OUTPUT_DIR}/stats_${LANG_LABEL}.json"
    LOG_FILE="${OUTPUT_DIR}/stats_${LANG_LABEL}.log"

    python tokenizer_stats_part1.py \
        --dataset_path     "${DATASET_PATH}" \
        --tok_grapheme_path "${TOK_GRAPHEME}" \
        --tok_llama2_path  "${TOK_LLAMA2}" \
        --tok_komodo_path  "${TOK_KOMODO}" \
        --tok_mt5_path     "${TOK_MT5}" \
        --output_path      "${OUT_JSON}" \
        --lang_label       "${LANG_LABEL}" \
        2>&1 | tee "${LOG_FILE}"

    echo "FINISHED ${LANG_LABEL}  ->  ${OUT_JSON}"
done

echo "ALL DONE.  Results in: ${OUTPUT_DIR}"