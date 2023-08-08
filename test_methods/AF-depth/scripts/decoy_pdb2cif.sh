#/bin/bash

root_dir='../output/dataset'
for pro in $(ls $root_dir);
do
    echo $pro
    pro_dir=$root_dir'/'$pro
    cd $pro_dir
    for p in $(ls $pro_dir);
    do
        name=`basename -s .pdb $p`;
        cif=$name'.cif'
        if [ ! -f $cif ];
        then
            pdb_extract.py -iPDB $p -o $cif >/dev/null 2>/dev/null
        fi
        break
    done
    break
done
