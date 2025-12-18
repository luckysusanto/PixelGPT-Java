#!/bin/bash -l
#$ -P multilm
#$ -N ali_dataset_prep
#$ -l mem_per_core=12G

export HF_DATASETS_DISABLE_MP=1

python download_dataset.py