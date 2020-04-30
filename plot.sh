#!/bin/bash

#SBATCH -J plot_5k_dir_maxpred
#SBATCH -o images/plot_5k_dir_maxpred.res
#SBATCH -e plot_5k_dir_maxpred.%A.err

#SBATCH --mail-user laura.iacovissi@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --mem=50G
#SBATCH --partition=short

# Run the python script
python3 plot.py --n 5000 --dir --from_p 3 --maxpred
