# predict with various settings for MSA depth:
# max_msa_clusters: the number of clusters
# max_extra_msa: number of extra sequences (randomly selected from clusters) default=5120

import sys
sys.path.append("../../../af2_conformations_20220515")
sys.path.append("../../../alphafold-2.1.2")

import os
import argparse
from scripts import predict

database = "../../AlphaFold/output/dataset"

def process_alignment(a3m_files):

    r"""Process sequence alignment
    (modified from ColabFold)

    Parameters
    ----------
    a3m_files : List of files to parse
    token : Token to look for when parsing

    Returns
    ----------
    Tuple with [0] string with alignment, and [1] path to template

    """

    a3m_lines = ""

    for line in open(a3m_files, "r"):
        if len(line) > 0:
            a3m_lines += line.replace("\x00", "")

    return a3m_lines

parser = argparse.ArgumentParser()
parser.add_argument('--pro', help='path of fasta file')
parser.add_argument('--dir', default="../output", help='directory of output')
parser.add_argument('--num', default=50, help='repeat number prediction process')
parser.add_argument('--max_extra_msa', nargs='+', type=int, help='a list of max_extra_msa settings')

args = parser.parse_args()

fasta = os.path.join(database, args.pro, args.pro+'.fasta')
a3m_path = os.path.join(database, args.pro, 'msas', 'merge90.a3m')
a3m_lines = process_alignment(a3m_path)

with open(fasta, 'r') as f:
    lines = f.readlines()
    sequence = ''.join(lines[1:])
    sequence = sequence.replace("\n", "")
    print(sequence)

pro_dir = os.path.join(args.dir, args.pro)
if not os.path.exists(pro_dir):
    os.mkdir(pro_dir)

for repeat_index in range(1, args.num+1):
    for max_extra_msa_ in args.max_extra_msa:
        predict.predict_structure_no_templates(sequence, pro_dir,
                 a3m_lines, model_id = 1, max_msa_clusters = int(max_extra_msa_/2),
                 max_extra_msa = max_extra_msa_, max_recycles = 1, n_struct_module_repeats = 8, repeat_index=repeat_index)
