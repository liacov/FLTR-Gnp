#!/bin/bash

#SBATCH --job-name=MyArrayJob
#SBATCH -J maxpred_1k_refhigh_res3
#SBATCH -o ./maxpred_1k_refhigh_res3.%A_%a.res
#SBATCH -e ./maxpred_1k_refhigh_res3.%A_%a.err

#SBATCH --mail-user laura.iacovissi@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --array=0,1,2
#SBATCH --cpus-per-task=24
#SBATCH --mem=50G
#SBATCH --partition=medium

# Run the python script
python3 FLTR_maxpred.py --p $SLURM_ARRAY_TASK_ID --n 1000 --und --k 50 --yes_sample --sample 100
