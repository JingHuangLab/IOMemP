#!/bin/bash

source ~/.bashrc
conda activate py36

out_dir=../output/dataset
data_dir='../../AlphaFold/output/dataset'
for pro in `ls $out_dir`
do
    clus_msa_root=$out_dir"/$pro"

    file=$clus_msa_root/EX_000.a3m
    if [ -f $file ]; then
        continue
    fi
    
    n=`squeue -u $USER -o %j|grep $pro'_AF_Cluster'|wc -l`
    if [ $n -eq 1 ]; then
        echo $pro was submitted.
        continue
    fi

    echo $pro
    raw_msa=$data_dir"/$pro/msas/merge90.a3m"
    sbatch --job-name=$pro"_AF_Cluster" --output ../log/$pro".log" --error ../log/$pro".err" ./run_msa_zimei.sh $raw_msa $clus_msa_root
    #break
done
