#!/bin/bash
#SBATCH --job-name=anyir_3deg
#SBATCH --partition=batch 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --output=./joblogs/anyir_3deg.log
#SBATCH --error=./joblogs/anyir_3deg.error

### ===== [1] Environment Setup =====
# CUDA
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH

# micromamba environments
source ~/.bashrc
micromamba activate anyir

# Project directory
cd /.../.../AnyIR


### ===== [2] Configurations =====
CKPT_DIR="train_ckpt/3deg"
DENOISE_DIR="/work/bin_ren/datasets/low-level/Train/Denoise/"
DERAIN_DIR="/work/bin_ren/datasets/low-level/Train/Derain/"
DEHAZE_DIR="/work/bin_ren/datasets/low-level/Train/Dehaze/"
GOPRO_DIR="/work/bin_ren/datasets/low-level/Train/Deblur/"
ENHANCE_DIR="/work/bin_ren/datasets/low-level/Train/Enhance/"

DE_TYPES="denoise_15 denoise_25 denoise_50 derain dehaze"

NUM_GPUS=1
BATCH_SIZE=32
EPOCHS=130
FFT_LOSS_WEIGHT=0.1

### ===== [3] Launch Training =====
python train.py \
    --trainset AnyIR \
    --ckpt_dir "$CKPT_DIR" \
    --de_type $DE_TYPES \
    --denoise_dir "$DENOISE_DIR" \
    --derain_dir "$DERAIN_DIR" \
    --dehaze_dir "$DEHAZE_DIR" \
    --gopro_dir "$GOPRO_DIR" \
    --enhance_dir "$ENHANCE_DIR" \
    --num_gpus $NUM_GPUS \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --fft_loss_weight $FFT_LOSS_WEIGHT \
    # --resume_from epoch=78-step=342702.ckpt