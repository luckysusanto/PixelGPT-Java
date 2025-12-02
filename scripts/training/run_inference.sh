#!/bin/bash
#$ -P multilm                                    # Project allocation
#$ -l gpus=1                                     # Request 4 GPUs on a single node. The scheduler will set CUDA_VISIBLE_DEVICES.
#$ -l gpu_type=A40                               # Request specific GPU type
#$ -l h_rt=00:10:00                              # Request 24 hours of runtime
#$ -N Inference_Test                      # Job name
#$ -j y                                          # Join stdout and stderr
#$ -o Inference_mixedTraining.log               # Specify the log file name

# Activate your Python environment
source "/projectnb/multilm/lsusanto/PixelGPT/pixelgpt_env/bin/activate"

# Add project root to PYTHONPATH so `from src...` works
export PYTHONPATH="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt"
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Set Hugging Face cache directory
export HF_HOME="/projectnb/multilm/lsusanto/PixelGPT/pixelgpt/hf_cache"

echo "Starting Python inference script..."
python /projectnb/multilm/lsusanto/PixelGPT/pixelgpt/scripts/training/run_ernie_pixel_inference.py
echo "Script finished."