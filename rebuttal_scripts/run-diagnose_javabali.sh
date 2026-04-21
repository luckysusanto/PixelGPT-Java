#!/bin/bash
#$ -pe omp 4

DATASET_BALI="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-balinese-pixelgpt/default/0.0.0/4a31992e9daac65666bc3a814085bc8e48352210"
DATASET_JAVA="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated/datasets/Exqrch___rebuttal-javanese-pixelgpt/default/0.0.0/c5fef00960c24c242eeb63befca570f18c1b3ec7"

export HF_HOME="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated"
export TRANSFORMERS_CACHE="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache_updated"

python diagnose_javabali.py \
    --dataset_a_name "$DATASET_JAVA" \
    --dataset_b_name "$DATASET_BALI" \
    --tokenizer_path "google/mt5-small" \
    --target_step 6688 \
    --per_device_batch_size 2 \
    --grad_accum_steps 8 \
    --world_size 2 \
    --seed 42