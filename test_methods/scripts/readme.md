# Analyze benchmark methods

## Calculate percentage

```
python compare_methods.py --methods MI mfDCA PSICOV CCMpred plmDCA RaptorX-Contact ResPRE trRosetta RaptorX-3DModeling ESM-1b RoseTTAFold AlphaFold --percentage_co_state_unique
```

## Calculate coverage

```
python compare_methods.py --methods MI mfDCA PSICOV CCMpred plmDCA RaptorX-Contact ResPRE trRosetta RaptorX-3DModeling ESM-1b RoseTTAFold AlphaFold --recision_co_state_unique
```

## Calculate the correlation of percentages and coverages betwwen shared contacts and state-specific contacts, or that of precisions and percentages for state-specific contacts

```
python compare_methods.py --methods MI mfDCA PSICOV CCMpred plmDCA RaptorX-Contact ResPRE trRosetta RaptorX-3DModeling ESM-1b RoseTTAFold AlphaFold --percentage_corrcoef
```

```
python compare_methods.py --methods MI mfDCA PSICOV CCMpred plmDCA RaptorX-Contact ResPRE trRosetta RaptorX-3DModeling ESM-1b RoseTTAFold AlphaFold --coverage_corrcoef
```

## Calculate the number of methods that prefer IF or OF state

```
python compare_methods.py --methods mfDCA PSICOV CCMpred plmDCA RaptorX-Contact ResPRE trRosetta RaptorX-3DModeling ESM-1b RoseTTAFold AlphaFold --num_methods_in_states
```

```
python compare_methods.py --methods mfDCA PSICOV CCMpred plmDCA RaptorX-Contact ResPRE trRosetta RaptorX-3DModeling ESM-1b RoseTTAFold AlphaFold --num_methods_in_states
```

## Calculate the averaged preference degree related to percentage and coverage

```
python compare_methods.py --methods mfDCA PSICOV CCMpred plmDCA RaptorX-Contact ResPRE trRosetta RaptorX-3DModeling ESM-1b RoseTTAFold AlphaFold --p_pre_unprefer
```

```
python compare_methods.py --methods mfDCA PSICOV CCMpred plmDCA RaptorX-Contact ResPRE trRosetta RaptorX-3DModeling ESM-1b RoseTTAFold AlphaFold --c_pre_unprefer
```

Run `plot.ipynb` to plot the the averaged preference degree.

## Plot NEFF of MSAs whose sequences are derived from IF and OF states in the context of sequence identity

```
python neff_pro_pair.py
```

## Calculate the precision of the 12 PSP methods

```
python compare_methods.py --methods MI mfDCA PSICOV CCMpred plmDCA RaptorX-Contact ResPRE trRosetta RaptorX-3DModeling ESM-1b RoseTTAFold AlphaFold --precision_all
```

