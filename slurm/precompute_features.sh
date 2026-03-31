#!/bin/bash
#SBATCH --job-name=feat_extract
#SBATCH --partition=alhalah-gpu-np
#SBATCH --account=alhalah-gpu-np
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/feat_extract_%A_%a.out
#SBATCH --error=logs/feat_extract_%A_%a.err

TOTAL_PARTS=8  # must match --array upper bound + 1

source ~/miniconda3/etc/profile.d/conda.sh
cd /scratch/rai/vast1/alhalah/users/nikesh/qwen3vl_proj
mkdir -p logs

source ~/.bashrc
conda activate qwen

export HF_HOME=/scratch/rai/vast1/alhalah/users/nikesh/models

python -u scripts/precompute_features.py \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --part $SLURM_ARRAY_TASK_ID \
    --total-parts $TOTAL_PARTS \
    --device cuda
