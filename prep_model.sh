#!/bin/bash
# =============================================================================
# RUN MODEL PREPARATION (SURGERY)
#
# This script prepares a pretrained model for continued pretraining with a
# new tokenizer by resizing its embedding layers.
# =============================================================================

# --- Paths ---
# The original pretrained model from the Hugging Face Hub
export BASE_MODEL="ernie-research/DualGPT"

# Your custom tokenizer from the Hugging Face Hub or a local path
export TOKENIZER_PATH="ernie-research/DualGPT"

# --- !!! SET THIS VARIABLE !!! ---
# This is the cache directory where the NEW, ADAPTED model will be saved.
export MODEL_CACHE="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache/models/dualGPT"

# Path to your Python environment and the surgery script
export VENV_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/"
export PYTHON_SCRIPT_PATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/prep_model.py"

# Set PYTHONPATH if your src folder is in a different location
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"

# -----------------------------------------------------------------------------

echo "======================================================"
echo "Starting Model Adaptation (Cold Start)"
echo "======================================================"
echo "Base Model:      ${BASE_MODEL}"
echo "New Tokenizer:   ${TOKENIZER_PATH}"
echo "Output Location: ${MODEL_CACHE}"
echo "------------------------------------------------------"

# Activate Python environment
source "${VENV_PATH}/bin/activate"

# Run the Python script
python "${PYTHON_SCRIPT_PATH}" \
    --base_model_path "${BASE_MODEL}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --output_model_path "${MODEL_CACHE}"

echo "------------------------------------------------------"
echo "Model preparation complete."
echo "You can now use the model located at ${MODEL_CACHE} for pretraining."
echo "======================================================"