#!/bin/bash

out_dir=../output/dataset

for pro in `ls $out_dir`
do
    clus_msa_root=$out_dir/"$pro"/

    n_EX=`grep "not clustered" $clus_msa_root/EX.log|awk ' {print $1} '`
    r_n_EX=$((n_EX-1))

    if [ $pro == "4M64_A" ]; then
        r_n_EX=466
    fi

    if [ $r_n_EX -lt 10 ]; then
        code=EX_00"$r_n_EX"
    elif [ $r_n_EX -lt 100 ]; then
        code=EX_0"$r_n_EX"
    else
        code=EX_"$r_n_EX"
    fi

    # check if the last pdb exists
    last_pdb=$clus_msa_root/"$code".pdb
    if [ -f $last_pdb ];then
        #echo $last_pdb exists.
        continue
    fi

    # run prediction
    n=`squeue -u $USER -o %j|grep $pro"_AF_Cluster"|wc -l`
    if [ $n -eq 1 ]; then
        echo $pro was submitted.
        continue
    fi

    echo $pro
    echo $r_n_EX
    #sbatch --job-name=$pro"_AF_Cluster" --output ../log/$pro".log" --error ../log/$pro".err" ./run_pdb_dongpo.sh $r_n_EX $clus_msa_root
    #break
done
