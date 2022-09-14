#!/bin/tcsh
#BSUB -W 12:00
#BSUB -n 4
#BSUB -R span[hosts=1]
#BSUB -o out.%J
#BSUB -e err.%J

module load conda
conda activate /usr/local/usrapps/statistics/hwang77/env_rl
python ope_cover.py --mc_size 10
conda deactivate