# Use AlphaFold to predicted the proteins in IOMemP and generate unified coevolution

## Modify the `$AlaphaFold/alphafold/data/pipeline.py` to force hhblits to seach BFD insead of BFD+Uniclust30.

In Line 145, replace `databases=[bfd_database_path, uniclust30_database_path])` with `databases=[bfd_database_path])`

## Run AlphaFold-2.1 and get `pkl` files for contact maps and `pdb` files for 3D coordinates.

## Unify the coevolution information and format them in `fas`/`a3m`/`aln` and calculate NEFF.

Requirements: HH-suite3, CCMpred, GREMLIN_CPP

```
bash batch_msa_unify.sh
```

## Analyze the prediction results.

Calculate $p_\text{sh}, p_1, p_2, c_\text{sh}, c_1, c_2$.

```
python analyze.py --analyze
```

Plot the $p_\text{sh}, p_1, p_2$ for the *ranked_0* model

```
python analyze.py --plot
```

Plot the $\Delta{D}$ of AF2 models to IF/OF structures.

```
python analyze.py --pre_exp_delta_af2
```

