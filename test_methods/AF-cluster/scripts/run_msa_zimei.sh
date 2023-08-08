#!/bin/bash
##SBATCH -J AF_Cluster
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p yuhuan
##SBATCH -o test.log
##SBATCH -e test.err

source ~/.bashrc
conda activate py36

raw_msa=$1
clus_msa_root=$2

tmp_dir=/tmp/$USER/$SLURM_JOB_ID
if [ ! -d $tmp_dir  ]; then
    mkdir -p $tmp_dir
fi

python ../../../AF-cluster/scripts/ClusterMSA.py EX -i $raw_msa -o $tmp_dir --breakEPS 30 --max_eps 30
rsync -az $tmp_dir"/" $clus_msa_root
rm -r $tmp_dir
