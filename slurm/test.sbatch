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
# python scripts/test_qwen_video.py -c /scratch/rai/vast1/alhalah/kyungbok/VLM_Router/checkpoints/extra_subsampled_fixed_pool_0/best --use_fixed_pooling 0 --out_path fixed_trained_results/pool0.json
# python scripts/test_qwen_video.py -c /scratch/rai/vast1/alhalah/kyungbok/VLM_Router/checkpoints/extra_subsampled_fixed_pool_1/best --use_fixed_pooling 1 --out_path fixed_trained_results/pool1.json
# python scripts/test_qwen_video.py -c /scratch/rai/vast1/alhalah/kyungbok/VLM_Router/checkpoints/extra_subsampled_fixed_pool_2/best --use_fixed_pooling 2 --out_path fixed_trained_results/pool2.json

# reward
# python scripts/test_qwen_video.py -c /scratch/rai/vast1/alhalah/kyungbok/VLM_Router/checkpoints/extra_subsampled --out_path results/reward.json
# random 
python scripts/test_qwen_video.py -c /scratch/rai/vast1/alhalah/kyungbok/VLM_Router/checkpoints/extra_subsampled/best --out_path results/reward.json








