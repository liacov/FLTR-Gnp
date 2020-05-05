#!/bin/bash

#SBATCH --job-name=MyArrayJob
#SBATCH -J graph_1k_ref
#SBATCH -o graph_1k_ref.%A_%a.res
#SBATCH -e graph_1k_ref.%A_%a.err

#SBATCH --mail-user laura.iacovissi@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --array=0-8
#SBATCH --cpus-per-task=24
#SBATCH --mem=50G
#SBATCH --partition=short

# Run the python script
python3 numpy_generation.py --p $SLURM_ARRAY_TASK_ID --n 1000 --dir --k 50
