#!/bin/bash

# set the path of uniref90 and image
database=/u/xiety/database/uniref90
img=/home/xiety/software/TOPCONS_singularity/topcons2.img

# $1: protein directory
span=$1/$2'.span'
span_lines=10
echo $span
if [ -f "$span" ]
then
    span_lines=`wc -l < $span`
fi

# contain_name=topcons
echo span_lines, $span_lines
if [ $span_lines -lt 5 ] || [ ! -f "$span" ]
then
    singularity exec  -B $database:/data/topcons2_database -B $1:/scratch $img /app/topcons2/run_topcons2.sh /scratch/$2'.fasta' -outpath /scratch/TOPCONS
    if [ "$?" -ne "0" ]
    then
        echo "Docker run/exec failed!"
        exit 1
    fi
    ./octopus_in_topcons2span.pl $1/TOPCONS/$2/query.result.txt TOPCONS> $span
    if [ "$?" -ne "0" ] 
    then
        ./octopus_in_topcons2span.pl $1/TOPCONS/$2/query.result.txt OCTOPUS> $span
        if [ "$?" -ne "0" ]
        then
            ./octopus_in_topcons2span.pl $1/TOPCONS/$2/query.result.txt Philius> $span
            if [ "$?" -ne "0" ]
            then
                echo TOPCONS failed to generate $fasta
            fi
        fi
    fi

fi

