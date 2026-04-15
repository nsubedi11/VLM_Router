#!/bin/bash

#SBATCH --job-name=vlm
#SBATCH --account=alhalah-gpu-np
#SBATCH --partition=alhalah-gpu-np

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

#SBATCH --mem=60G  
#SBATCH --time=144:00:00  


#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err


#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1590742@utah.edu


source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen

cd /scratch/rai/vast1/alhalah/kyungbok/VLM_Router
python scripts/test_qwen_video.py --use_fixed_pooling 2 --out_path results/test_qwen_video_offline_pool2.json








