#!/bin/bash
#SBATCH --job-name=p9-energytransformer_testing
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=03:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=10
#SBATCH --mem=25G

singularity exec container.sif python3 Whitebox_tuning.py