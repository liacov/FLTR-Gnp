#!/bin/bash

#SBATCH --job-name=MyArrayJob
#SBATCH -J maxpred_2k
#SBATCH -o ./maxpred_2k.%A_%a.res
#SBATCH -e ./maxpred_2k.%A_%a.err

#SBATCH --mail-user laura.iacovissi@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --array=4-18
#SBATCH --cpus-per-task=24
#SBATCH --mem=50G
#SBATCH --partition=medium

# Run the python script
python3 FLTR_maxpred.py --p $SLURM_ARRAY_TASK_ID --n 2000 --und --k 50 --yes_sample --sample 1000
