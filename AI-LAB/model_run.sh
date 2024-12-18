#!/bin/bash
#SBATCH --job-name=p9-energytransformer tuning
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G

# Load the singularity module
module load singularity

# Acess script arguments
SCRIPT=$1

# Print out the script arguments for debugging
echo "SCRIPT: $SCRIPT"

singularity exec /ceph/container/pytorch/pytorch_24.09.sif python3 $SCRIPT

 