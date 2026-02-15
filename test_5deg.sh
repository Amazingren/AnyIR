#!/bin/bash
#SBATCH --job-name=deg5_test
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --nodelist=gcp-eu-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --time=01:00:00
#SBATCH --output=./joblogs/deg5_test_ep109.log
#SBATCH --error=./joblogs/deg5_test_ep109.error

### ===== [1] Environment Setup =====
# CUDA
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH

# micromamba environment
source ~/.bashrc
micromamba activate anyir

# Project directory
cd /home/bin_ren/projects/low-level/MIR_NeurIPS25/ab_main_exps/acm_b_ctrs_spd

### ===== [2] Configurations =====
DENOISE_DIR="/work/bin_ren/datasets/low-level/test/denoise/"
DERAIN_DIR="/work/bin_ren/datasets/low-level/test/derain/"
DEHAZE_DIR="/work/bin_ren/datasets/low-level/test/dehaze/"
GOPRO_DIR="/work/bin_ren/datasets/low-level/test/deblur/"
ENHANCE_DIR="/work/bin_ren/datasets/low-level/test/enhance/"

### ===== [3] Launch Training =====
python test.py \
    --trainset AnyIR \
    --mode 6 \
    --denoise_path "$DENOISE_DIR" \
    --derain_path "$DERAIN_DIR" \
    --dehaze_path "$DEHAZE_DIR" \
    --gopro_path "$GOPRO_DIR" \
    --enhance_path "$ENHANCE_DIR" \
    --ckpt_name 5deg/epoch=109-step=546700.ckpt \
    --output_path ./outputs/5deg_test_ep109