#!/bin/bash
#SBATCH --job-name=deg3_test
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --nodelist=gcp-us2-h200-jlrn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --output=./joblogs/deg3_test_ep87.log
#SBATCH --error=./joblogs/deg3_test_ep87.error


### ===== [1] Environment Setup =====
# CUDA
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH

# micromamba environment
source ~/.bashrc
micromamba activate anyir

# Project directory
cd /home/bin_ren/projects/low-level/AnyIR_MM25/repre_learning/anyir_3deg_new


### ===== [2] Configurations =====
DENOISE_DIR="/work/bin_ren/datasets/low-level/test/denoise/"
DERAIN_DIR="/work/bin_ren/datasets/low-level/test/derain/"
DEHAZE_DIR="/work/bin_ren/datasets/low-level/test/dehaze/"
GOPRO_DIR="/work/bin_ren/datasets/low-level/test/deblur/"
ENHANCE_DIR="/work/bin_ren/datasets/low-level/test/enhance/"

### ===== [3] Launch Training =====
python test.py \
    --trainset AnyIR \
    --mode 5 \
    --denoise_path "$DENOISE_DIR" \
    --derain_path "$DERAIN_DIR" \
    --dehaze_path "$DEHAZE_DIR" \
    --gopro_path "$GOPRO_DIR" \
    --enhance_path "$ENHANCE_DIR" \
    --ckpt_name 3deg/epoch=129-step=563940.ckpt \
    --output_path ./outputs/3deg_test_ep129


python test.py \
    --trainset AnyIR \
    --mode 5 \
    --denoise_path "/home/bin_ren/projects/low-level/testset/35deg_test/denoise/" \
    --derain_path "/home/bin_ren/projects/low-level/testset/35deg_test/derain/" \
    --dehaze_path "/home/bin_ren/projects/low-level/testset/35deg_test/dehaze/" \
    --gopro_path "/home/bin_ren/projects/low-level/testset/35deg_test/deblur/" \
    --enhance_path "/home/bin_ren/projects/low-level/testset/35deg_test/enhance/" \
    --ckpt_name 3deg/epoch=129-step=563940.ckpt \
    --output_path ./outputs/3deg_test_ep129

