#!/bin/bash

#SBATCH --job-name=vlm_xsub
#SBATCH --account=alhalah-gpu-np
#SBATCH --partition=alhalah-gpu-np

#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

#SBATCH --mem=100G
#SBATCH --time=144:00:00

#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1590742@utah.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/rai/vast1/alhalah/kyungbok/VLM_Router

accelerate launch --num_processes 1 --mixed_precision bf16 scripts/train_joint.py \
    --feat-dir  /scratch/rai/vast1/alhalah/users/nikesh/qwen3vl_proj/features/qwen_video_extra_subsampled    \
    --cache-dir cache/extra_subsampled \
    --ckpt-dir  checkpoints/extra_subsampled_random_reward_warmup \


python scripts/test_qwen_video.py -c /scratch/rai/vast1/alhalah/kyungbok/VLM_Router/checkpoints/extra_subsampled_random_reward_warmup/best  --out_path results/random_warmup_reward_router.json