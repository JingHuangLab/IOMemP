#!/bin/bash

msa=$1
dir=$2
python $HOME/software/AF_Cluster_20221027/scripts/ClusterMSA.py EX -i $msa -o $dir
mv EX.log $dir

