#!/bin/bash
#SBATCH -J AF2Rank
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p A40
#SBATCH --gres=gpu:1
#SBATCH -o ../log/run_zimei.log
#SBATCH -e ../log/run_zimei.err

source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate alphafold
ml mathlib/cuda/11.2.2_460.32.03

pro=$1
name=$2
msa_dir=$3

# run with msa but no template
python ../../../AF2Rank_homo_template/test_templates.py $name --target_list $pro --output_dir ../output/ --decoy_dir ../inputs/decoy_set_if_of_all_in/ --seq_replacement - --mask_sidechains_add_cb --use_precomputed_msas --msa_dir $msa_dir --no_template --verbose 

# run with msa and templates (IF+OF)
# python ../../../AF2Rank_homo_template/test_templates.py $name --target_list $pro --output_dir ../output/ --decoy_dir ../inputs/decoy_set_if_of_all_in/ --seq_replacement - --mask_sidechains_add_cb --use_native  --multiple_decoy --use_precomputed_msas --msa_dir $msa_dir --verbose 

# run with only IF and OF structures as templates
# python ../../../AF2Rank_homo_template/test_templates.py $name --target_list $pro --output_dir ../output/ --decoy_dir ../inputs/decoy_set_if_of_all_in/ --seq_replacement - --mask_sidechains_add_cb --use_native  --multiple_decoy --verbose 

# run with an IF or OF structure as a template
# python ../../../AF2Rank_homo_template/test_templates.py $name --target_list $pro --output_dir ../output/ --decoy_dir ../inputs/decoy_set_if_of_all_in/ --seq_replacement - --mask_sidechains_add_cb --use_native --verbose 
