#!/bin/bash
#SBATCH --job-name=creating_energycontainer
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=03:00:00 # 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=25G

singularity exec energycontainer.sif sbatch energycontainer.def

 