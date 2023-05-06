# The curation of the dataset with IF/OF states

#### Download sequence cluster data from rcsb PDB  and save to `input/seq_cluster_rcsb/bc-70.out` (56,823 protein clusters)

#### Download the sequences of the first member of each protein cluster and save to directory `bc-70-Down-span` with the following arrangement `$ID/$ID.fasta`.

#### Membrane protein recognition

1. Sequence-based topology prediction
   * Download sequence dataset  uniref90 and singularity image; set their path in `topconsOne.sh`.
   * Run `seq_cluster_topcons_batch.sh` for the topology of those downloaded sequneces.
   * Run `seq_cluster_filter_memPDBs.sh` for transmembrane proteins (2,719).
   * Filter for protein cluster with more than one member (2,124).   

2. Structure-based  membrane orientation prediction
   * Requiriments: TMalign, memembed, all PDB and mmCIF files from RCSB PDB.
   * Predict the membrane orientation and filter non-transmembrane proteins (1098). 
        
        ```python memPDBsMostDiff.py --seq_cluster_mem_TOPC ../output/seq_cluster70_memPDBs_multiple --mem```

#### Filtration for proteins with alternative states

1. Filter those proteins with less than 80 residues in transmembrane regions and find the two members with largest intra-distance difference for each protein cluster (860).
   
   ```python memPDBsMostDiff.py --seq_cluster_mem_TOPC ../output/seq_cluster70_memPDBs_multiple  --dir4mostdiff ../output/mostdiff_files_seq_cluster70 --mdiff```

2. Rank all protein clusters by the largest intra-distance differences. 
    
    ```python memPDBsMostDiff.py --seq_cluster_mem_TOPC ../output/seq_cluster70_memPDBs_multiple -rank --suffix _seq_cluster70 ```
    
3. Sixteen proteins with resolution less than 3.5 $\unicode{x212B}$ are mannually selected as the databse IOMemP.

4. Plot intra-distance difference histogram.
 
    ``` python plot_dataset.py ```

#### Analysis on IOMemP

1. Record the distance matric of the two members of each protein cluster to `tm_common_res_dis.pickle` , which will be gold standard for our benchmark.

    ``` python tm_common_res_dis.py ```
2. Calculate sequence identities by `MMseqs2`.

    ``` bash mostdiff_files_seq_cluster70_manu_sel_in_out.identity.sh ```

