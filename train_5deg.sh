#!/bin/bash
#SBATCH --job-name=t_5D_ctrs_spd
#SBATCH --partition=batch 
#SBATCH --constraint=type-gpu
#SBATCH --nodelist=msp3-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --output=./joblogs/t_5D_ctrs_spd.log
#SBATCH --error=./joblogs/t_5D_ctrs_spd.error

### ===== [1] Environment Setup =====
# CUDA
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH

# micromamba environment
source ~/.bashrc
micromamba activate anyir

# Project directory
cd /home/bin_ren/projects/low-level/MIR_NeurIPS25/ab_main_exps/acm_t_ctrs_spd



### ===== [2] Configurations =====
CKPT_DIR="train_ckpt/5deg"
DENOISE_DIR="/work/bin_ren/datasets/low-level/Train/Denoise/"
DERAIN_DIR="/work/bin_ren/datasets/low-level/Train/Derain/"
DEHAZE_DIR="/work/bin_ren/datasets/low-level/Train/Dehaze/"
GOPRO_DIR="/work/bin_ren/datasets/low-level/Train/Deblur/"
ENHANCE_DIR="/work/bin_ren/datasets/low-level/Train/Enhance/"

DE_TYPES="denoise_15 denoise_25 denoise_50 derain dehaze deblur enhance"

NUM_GPUS=1
BATCH_SIZE=32
EPOCHS=150
FFT_LOSS_WEIGHT=0.1


### ===== [3] Launch Training =====
python train.py \
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
    --resume_from