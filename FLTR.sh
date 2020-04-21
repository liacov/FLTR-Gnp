#!/bin/bash

#SBATCH --job-name=MyArrayJob
#SBATCH -J influence_1k
#SBATCH -o ./data/influence_1k.%A_%a.res
#SBATCH -e ./data/influence_1k.%A_%a.err

#SBATCH --mail-user laura.iacovissi@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --array=3-9
#SBATCH --cpus-per-task=24
#SBATCH --mem=50G
#SBATCH --partition=medium

# Run the python script
python3 FLTR_opt.py --p $SLURM_ARRAY_TASK_ID --n 1000 --und --k 50 --no_sample
