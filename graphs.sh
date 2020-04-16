#!/bin/bash

#SBATCH --job-name=MyArrayJob
#SBATCH -J graph_1k
#SBATCH -o ./data/graph_1k.%A_%a.res
#SBATCH -e ./data/graph_1k.%A_%a.err

#SBATCH --mail-user laura.iacovissi@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --array=0-3
#SBATCH --cpus-per-task=15
#SBATCH --mem=40G
#SBATCH --partition=medium

# Run the python script
python3 graphs_generation.py --p $SLURM_ARRAY_TASK_ID --n 1000 --directed False --k 500
