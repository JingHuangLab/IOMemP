#!/bin/bash
source ~/.bashrc
conda activate py27

dir=../output/dataset/

for pro in $(ls $dir);
do
    echo $pro
    msa_dir=$dir/$pro/msas
    mg_sto=$msa_dir/mgnify_hits.sto
    uniref_sto=$msa_dir/uniref90_hits.sto
    bfd_a3m=$msa_dir/bfd_uniclust_hits.a3m

    if [ ! -f $mg_sto ]; then
        echo $mg_sto does not exist.
        continue
    fi

    if [ ! -f $uniref_sto ]; then
        echo $uniref_sto does not exist.
        continue
    fi

    if [ ! -f $bfd_a3m ]; then
        echo $bfd_a3m does not exist.
        continue
    fi

    merge90_a3m=$msa_dir/merge90.a3m
    if [ ! -f $merge90_a3m ]; then
      mg_a3m=$msa_dir/mgnify_hits.a3m
      # reformat
      tmp_mg_sto=/tmp/mgnify_hits.sto
      tmp_mg_a3m=/tmp/mgnify_hits.a3m
      cp $mg_sto $tmp_mg_sto
      $HOME/software/hh-suite/scripts/reformat.pl sto a3m $tmp_mg_sto $tmp_mg_a3m
      if [ $? -eq 1 ]; then
          echo reformat.pl sto a3m $mg_sto $mg_a3m failed.
          rm $tmp_mg_sto
          exit 1
      fi
      rm $tmp_mg_sto
      mv $tmp_mg_a3m $mg_a3m

      uniref_a3m=$msa_dir/uniref90_hits.a3m
      # reformat
      tmp_uniref_sto=/tmp/uniref90_hits.sto
      tmp_uniref_a3m=/tmp/uniref90_hits.a3m
      cp $uniref_sto $tmp_uniref_sto
      $HOME/software/hh-suite/scripts/reformat.pl sto a3m $tmp_uniref_sto $tmp_uniref_a3m
      if [ $? -eq 1 ]; then
          echo reformat.pl sto a3m $uniref_sto $uniref_a3m failed.
          rm $tmp_uniref_sto
          exit 1
      fi
      rm $tmp_uniref_sto
      mv $tmp_uniref_a3m $uniref_a3m

      # merge
      tmp_merge_a3m=/tmp/merge.a3m
      cat $mg_a3m $uniref_a3m $bfd_a3m > $tmp_merge_a3m

      # filter
      tmp_merge90_a3m=/tmp/merge90.a3m
      hhfilter -id 90 -i $tmp_merge_a3m -o $tmp_merge90_a3m
      mv $tmp_merge90_a3m $merge90_a3m
    fi

    merge90_fas=$msa_dir/merge90.fas
    if [ ! -f $merge90_fas ]; then
        $HOME/software/hh-suite/scripts/reformat.pl a3m fas $merge90_a3m $merge90_fas -r
    fi

    merge90_aln=$msa_dir/merge90.aln
    if [ ! -f $merge90_aln ]; then
        $HOME/software/CCMpred/scripts/convert_alignment.py $merge90_fas fasta $merge90_aln
    fi

    neff=$msa_dir/merge90_neff.txt
    if [ ! -f $neff ]; then
        ~/software/GREMLIN_CPP/gremlin_cpp -only_neff -i $merge90_aln > $neff
    fi

    #break

done
