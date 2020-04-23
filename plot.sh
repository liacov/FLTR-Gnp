#!/bin/bash

#SBATCH -J plot_5k_dir
#SBATCH -o images/plot_5k_dir.res
#SBATCH -e plot_5k_dir.%A.err

#SBATCH --mail-user laura.iacovissi@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --mem=50G
#SBATCH --partition=short

# Run the python script
python3 plot.py --n 5000 --dir --from_p 3
