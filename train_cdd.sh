#!/bin/bash
#SBATCH --job-name=space_cdd
#SBATCH --partition=batch 
#SBATCH --constraint=type-gpu
#SBATCH --nodelist=gcp-us2-h200-c37g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --output=./joblogs/space_cdd11.log
#SBATCH --error=./joblogs/space_cdd11.error

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
CKPT_DIR="train_ckpt/cdd11"
CDD_DIR="/work/bin_ren/datasets/low-level/cdd11/"

DE_TYPES="denoise_15 denoise_25 denoise_50 derain dehaze"

NUM_GPUS=1
BATCH_SIZE=32
EPOCHS=200
FFT_LOSS_WEIGHT=0.1


### ===== [3] Launch Training =====
python train.py \
    --trainset CDD11_all \
    --ckpt_dir "$CKPT_DIR" \
    --de_type $DE_TYPES \
    --cdd11_path "$CDD_DIR" \
    --num_gpus $NUM_GPUS \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --fft_loss_weight $FFT_LOSS_WEIGHT \
    --resume_from epoch=19-step=48780.ckpt