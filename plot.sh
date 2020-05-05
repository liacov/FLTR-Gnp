#!/bin/bash

#SBATCH -J plot_1k_dir_maxpred
#SBATCH -o images/plot_1k_dir_maxpred.res
#SBATCH -e plot_1k_dir_maxpred.%A.err

#SBATCH --mail-user laura.iacovissi@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --mem=50G
#SBATCH --partition=short

# Run the python script
python3 plot.py --n 1000 --dir --maxpred
