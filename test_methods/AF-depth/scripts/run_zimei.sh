#!/bin/bash
##SBATCH -J A40
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p A40
#SBATCH --gres=gpu:1
#SBATCH -o ../log/test.log
#SBATCH -e ../log/test.err

source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate alphafold
ml mathlib/cuda/11.2.2_460.32.03

seq=$1 # target sequence
python run_zimei.py --pro $seq --max_extra_msa 8 16 32 64 128
