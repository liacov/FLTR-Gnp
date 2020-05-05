#!/bin/bash

#SBATCH -J properties_1k_dir_maxpred
#SBATCH -o images/properties_1k_dir_maxpred.res
#SBATCH -e properties_1k_dir_maxpred.%A.err

#SBATCH --mail-user laura.iacovissi@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --mem=50G
#SBATCH --partition=short

# Run the python script
python3 phase_transition.py --n 1000 --dir --maxpred
