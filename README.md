# Can Protein Structure Prediction Methods Capture Alternative Conformations of Membrane Proteins?

In this study, we benchmarked 12 representative protein structure methods on alternative conformation prediction in a curated dataset that consists of Inward-facing and Outward-facing states of Membrane Proteins (IOMemP) from the Protein Data Bank (PDB). 
The methods were benchmarked using the same input multiple sequence alignments (MSAs). 
Additionally, we tested 2 AlphaFold-based methods that manipulate MSAs to predict alternative conformations. 
Our dataset IOMemP and benchmark results could promote the development of alternative conformation prediction.

## The curation of the dataset
The details can be found [here](https://github.com/JingHuangLab/IOMemP/tree/master/dataset/scripts#the-curation-of-the-dataset-with-ifof-states).

## The benchmarked methods and their configurations
[MI]( http://dca.rice.edu/portal/dca)
[mfDCA]( http://dca.rice.edu/portal/dca)
[PSICOV]( https://github.com/psipred/psicov)
[CCMpred]( https://github.com/soedinglab/CCMpred)
[plmDCA]( https://github.com/pagnani/PlmDCA)
[RaptorX-Contact]( https://github.com/j3xugit/RaptorX-Contact)
[ResPRE]( https://zhanggroup.org/ResPRE)
[trRosetta]( https://yanglab.nankai.edu.cn/trRosetta)
[RaptorX-3DModeling](https://github.com/j3xugit/RaptorX-3DModeling)
[ESM]( https://github.com/facebookresearch/esm)
[RoseTTAFold]( https://github.com/RosettaCommons/RoseTTAFold)
[AlphaFold]( https://github.com/deepmind/alphafold)
[AF-Depth]( https://github.com/delalamo/af2_conformations)
[AF-Cluster]( https://github.com/HWaymentSteele/AF_Cluster)

All methods are executed with default settings if there is no specific statement. 

As for AF-cluster, AF-depth, and AlphaFold (v2.1.2), the mini modifications for their localization are provided in the corresponding patch files. 

Note that AF-cluster default uses Alphafold (v2.2.0) as its prediction engine. 

## The calculation of the composite confidence score of IF and OF structures

We add an alignment function to [AF2Rank](https://github.com/jproney/AF2Rank) (renamed as AF2Rank_homo_template) for the homologous template assignment, considering the different sequences lying between IF and OF structures. 

## The benchmark on AlphaFold input with MSAs, IF and OF structures, or their combinations

We also make AF2Rank_homo_template accept multiple templates and predict multiple times for models 1-5, so that we can AlphaFold predictions under different input configurations. 

