#!/bin/tcsh
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R span[hosts=1]
#BSUB -o out.%J
#BSUB -e err.%J

module load conda
conda activate /usr/local/usrapps/statistics/hwang77/env_rl
python sim_ope.py --mc_size 25 --dropout_scheme '0' --ipw False --estimate_missing_prob False
conda deactivate