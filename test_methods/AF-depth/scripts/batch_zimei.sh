#!/bin/bash

dir=../../AlphaFold/output/dataset
for pro in $(ls $dir);
do
    # check output
    pkl="../output/$pro/model50_64_128.pkl"
    if [ -f "$pkl" ]; then
        echo $pro has pkl file.
        continue
    fi

    suffix='_af2_conformation'
    n=`squeue -u $USER -o %j|grep $pro$suffix|wc -l`
    if [ $n -eq 1 ]; then
        #echo $pdb was submitted.
        continue
    fi

    echo $pro
    fas=$dir'/'$pro'/'$pro'.fasta'
    #sbatch --job-name=$pro$suffix --output=../log/$pro'.log' --error=../log/$pro'.err' run_zimei.sh $pro
    #break
done
