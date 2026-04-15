#!/bin/bash

#SBATCH --job-name=vlm_sub
#SBATCH --account=alhalah-gpu-np
#SBATCH --partition=alhalah-gpu-np

#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

#SBATCH --mem=180G
#SBATCH --time=144:00:00

#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/rai/vast1/alhalah/users/nikesh/qwen3vl_proj

accelerate launch --num_processes 1 --mixed_precision bf16 scripts/train_joint.py \
    --feat-dir  features/qwen_video_subsampled \
    --cache-dir cache/subsampled \
    --ckpt-dir  checkpoints/subsampled
