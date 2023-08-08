#!/bin/bash
# coda activate py37

out_dir=../output/dataset
data_dir='../../AlphaFold/output/dataset'
for pro in `ls $out_dir`
do
    clus_msa_root=$out_dir"/$pro"

    file=$clus_msa_root/EX_cluster_metadata.tsv
    if [ -f $file ]; then
        continue
    fi

    echo $pro
    raw_msa=$data_dir"/$pro/msas/merge90.a3m"
    ./run_msa_dafu.sh $raw_msa $clus_msa_root 
    #break
done
