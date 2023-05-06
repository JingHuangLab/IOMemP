#/bin/bash
#filter proteins with TM regions, which is recognized by TOPCONS.

# set the path of downloaded sequence directory, output file, and sequence cluster data
dir='/u/xiety/database/bc-70-Down-span'
memPDBs=../output/seq_cluster70_memPDBs
seqs=../input/seq_cluster_rcsb/bc-70.out

if [ -f "$memPDBs" ]
then
    rm $memPDBs
fi

while IFS= read -r line;
do
    pdbid=`echo "$line"|awk '{ print $1 }'`
    span=$dir/$pdbid/$pdbid'.span'

    if [ ! -f "$span" ]
    then
       echo $span does not exist. 
       continue
    fi

    span_lines=`wc -l < $span`
    if [ $span_lines -lt 4 ]
    then
        echo $span does not generated successfully.
    fi

    if [ $span_lines -gt 4 ]
    then
        echo $line >> $memPDBs
        #break
    fi
done < "$seqs"
