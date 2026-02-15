#!/bin/bash
#SBATCH --job-name=cdd11_test
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --nodelist=gcp-us2-h200-hp0c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --output=./joblogs/cdd11_test_ep154_low_haze_rain.log
#SBATCH --error=./joblogs/cdd11_test_ep154_low_haze_rain.error


### ===== [1] Environment Setup =====
# CUDA
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH


# micromamba environment
source ~/.bashrc
micromamba activate anyir

# Project directory
cd /home/bin_ren/projects/low-level/AnyIR_MM25/repre_learning/anyir_cdd11


### ===== [2] Configurations =====
CDD11_PATH="/work/bin_ren/datasets/low-level/cdd11/"


### ===== [3] Launch Training =====
python test.py \
    --trainset CDD11_low_haze_rain \
    --cdd11_path "$CDD11_PATH" \
    --output_path output/cdd11/test_ep154_low_haze_rain \
    --ckpt_name cdd11/epoch=154-step=378045.ckpt


# Direct Testing without using SLURM
# python test.py \
#     --trainset CDD11_low_haze_rain \
#     --cdd11_path /work/bin_ren/datasets/low-level/cdd11/ \
#     --output_path output/cdd11/test_ep04 \
#     --ckpt_name cdd11/epoch=4-step=40.ckpt