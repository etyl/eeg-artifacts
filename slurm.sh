#!/bin/bash
#SBATCH --job-name=reve_embeddings
#SBATCH --partition=parietal,normal
#SBATCH --time=72:00:00  # Adjust as needed
#SBATCH --output=logs/reve_embeddings.log
#SBATCH --error=logs/reve_embeddings_error.log
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G 

python generate_embeddings.py --max-recordings=2 --model=cbramod