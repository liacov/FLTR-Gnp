#!/bin/bash

#SBATCH --job-name=MyArrayJob
#SBATCH -J graph_5k
#SBATCH -o ./data/graph_5k.%A_%a.res
#SBATCH -e ./data/graph_5k.%A_%a.err

#SBATCH --mail-user laura.iacovissi@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --array=0-9
#SBATCH --cpus-per-task=24
#SBATCH --mem=50G
#SBATCH --partition=short

# Run the python script
python3 numpy_generation.py --p $SLURM_ARRAY_TASK_ID --n 5000 --dir --k 50
