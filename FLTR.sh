#!/bin/bash

#SBATCH --job-name=MyArrayJob
#SBATCH -J pred_1k_refhigh
#SBATCH -o ./pred_1k_refhigh.%A_%a.res
#SBATCH -e ./pred_1k_refhigh.%A_%a.err

#SBATCH --mail-user laura.iacovissi@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --array=0,1,2
#SBATCH --cpus-per-task=24
#SBATCH --mem=50G
#SBATCH --partition=medium

# Run the python script
python3 FLTR_pred_inflection.py --p $SLURM_ARRAY_TASK_ID --n 1000 --dir --k 50 --yes_sample --sample 100
