# Key modifications on original AF2Rank

## Homologous template's confidence score estimation

When the target sequence is from an IF structure, the corresponding OF structure, holding different sequence, need to be aligned to the target seqeuence before its coordinates are assigned to the target sequence.

We use model 1 to estimate the confidence score (energy) of a input template. 

## Different AF input configurations

The following configurations are implemented:

1. both IF and OF structures as templates (no MSAs)
2. solely MSAs
3. the combination of MSAs and both IF and OF structures as templates

We use AF models 1-5 with each repeated 5 times to generate 25 models for each target sequence with evergy configuration.