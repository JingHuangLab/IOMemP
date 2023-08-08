#!/bin/bash

test_name=msa_in
dir=../inputs/decoy_set_$test_name/decoy_pdbs
output=../output/eva_results_$test_name/results
af2_dir=../../AlphaFold/output/dataset
for pro in $(ls $dir);
do
    #echo $pro
    pro_dir=$dir/$pro
    result=$output'/../pdbs/'$pro'_no_template24.pdb' # be caerful the result file name
    if [ -f "$result" ]; then
        #echo $result exists.
        continue
    fi

    n=`squeue -u $USER -o %j|grep $pro'_AF2Rank'|wc -l`
    if [ $n -eq 1 ]; then
        echo $pro was submitted.
        continue
    fi

    echo $pro

    uppercase_pro=${pro^^}
    alias_pro="${uppercase_pro:0:4}"_"${uppercase_pro:4:5}" 
    # echo $alias_pro
    msa_dir=$af2_dir/$alias_pro/msas
    # echo $msa_dir
    sbatch --job-name=$pro'_AF2Rank' --output=../log/$pro'.log' --error=../log/$pro'.err' run_zimei.sh $pro eva_results_$test_name $msa_dir
    #break
done
