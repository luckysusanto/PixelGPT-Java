#!/bin/bash

# Set SCC project
#$ -P llamagrp

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=12:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N render_komodo_bali

# Combine output and error files into a single file
#$ -j y

# Request 2 core
#$ -pe omp 2

# Keep track of information related to the current job
# echo "=========================================================="
# echo "Start date : $(date)"
# echo "Job name : $JOB_NAME"
# echo "WORKING DIR: $TMPDIR"
# echo "Job ID : $JOB_ID"
# echo "=========================================================="

# module load cuda/12.2 gcc/12.2.0 miniconda/23.11.0

# conda activate "pixelgpt"

echo "Using Python: $(which python)"

set -a
source "../../pixelgpt_env"
set +a

languages=("balinese")

for lang in "${languages[@]}"; do
 echo "Creating language: ${lang}"
    python data_renderer2.py \
        --renderer_config_path "../../renderers/m4_renderer" \
        --lang "${lang}" \
        --tokenizer_path "Yellow-AI-NLP/komodo-7b-base" \
        --tokenizer_name "Komodo"
        &> "render_$lang.log"
done