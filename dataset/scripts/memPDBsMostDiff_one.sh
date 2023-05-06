#!/bin/bash
source ~/.bashrc 

ml gcc/gcc-5.5.0 
conda activate py36 

pdb4seq=$1
n_proc=$2
filepath=$3

python memPDBsMostDiff_one.py --pdbs "$pdb4seq" --npool $n_proc --filepath $filepath
