#!/bin/bash
#SBATCH -J test_AF-cluster
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o test.log
#SBATCH -e test.err

r_n_EX=$1
clus_msa_root=$2

echo $r_n_EX
for i in `seq 0 $r_n_EX`;
do
    # msa file
    if [ $i -lt 10 ]; then
        code=EX_00"$i"
    elif [ $i -lt 100 ]; then
        code=EX_0"$i"
    else
        code=EX_"$i"
    fi
    msa=$clus_msa_root/"$code".a3m 

    # check if pdb exists
    pdb=$clus_msa_root/"$code".pdb
    if [ -f $pdb ];then
        echo $pdb exists.
        continue
    fi

    python ../../../AF-cluster/scripts/RunAF2.py $msa --af2_dir ../../../AlphaFold/ --output_dir $clus_msa_root

done
