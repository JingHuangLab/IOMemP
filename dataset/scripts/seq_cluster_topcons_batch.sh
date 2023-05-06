#!/bin/bash
#SBATCH -J TOPCONS
#SBATCH -c 72
#SBATCH -p yuhuan
#SBATCH -o ../log/batch.log
#SBATCH -e ../log/batch.err

ml apps/singularity/3.7.0

# set the path of downloaded sequences
dir='/u/xiety/database/bc-70-Down-span'

#parallel -j2 ./topconsOne.sh $dir/{} {} ::: 11AS_A 11AS_A
parallel -j72 ./topconsOne.sh $dir/{} {} ::: $(ls $dir)

