#!/bin/bash

rm -r /tmp/resultDB* /tmp/tmp/*

query=$1
target=$2

query_seq=../output/PDBs_tm_common_res_dis/$1'_ignorechain.fasta'
target_seq=../output/PDBs_tm_common_res_dis/$2'_ignorechain.fasta'

# createdb
mmseqs createdb $query_seq /tmp/queryDB
mmseqs createdb $target_seq /tmp/targetDB

# align
mmseqs search /tmp/queryDB /tmp/targetDB /tmp/resultDB /tmp/tmp 
sleep 1s

seq_id=`cat /tmp/resultDB | awk ' { print($3) }'`
echo $1 $2 $seq_id >> ../output/seq_identity.txt
