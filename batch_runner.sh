#!/bin/bash
#SBATCH --job-name=sam_aerial
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=sam_aerial_%j.out
#SBATCH --error=sam_aerial_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sloan.40@wright.edu

# Load environment
cd ~/sam-aerial-analysis
source .venv/bin/activate

# Run inference
python sam_batch_inference.py