#/usr/bin/python3.7

"""
Compare prediction methods by precision and coverage
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from matplotlib import rcParams
rcParams.update({'font.size': 8})
rcParams.update({'font.weight': 'bold'})
rcParams.update({'axes.labelsize': 8})
rcParams.update({'axes.labelweight': 'bold'})

def get_neff(pro):
    root_path = '../AlphaFold/output/dataset'
    neff_file = os.path.join(root_path, pro, 'msas/merge90_neff.txt')
    with open(neff_file, 'r') as f:
        line = f.readlines()[-1]
        neff = line.split()[-1]
    return float(neff)

with open('../../dataset/output/mostdiff_files_seq_cluster70_manu_sel_in_out_uppercase.txt','r') as f:
    pairs = []
    for line in f.readlines():
        if line[0] != '#':
            pairs.append(line.split())


print(pairs)

neffs1 = []
neffs2 = []
seq_id_f = "../../dataset/output/seq_identity.txt"
with open(seq_id_f, 'r') as f:
    seq_identity = []
    for line in f.readlines():
        one_seq_id = float(line.split()[-1]) * 100
        seq_identity.append(one_seq_id)
for pair in pairs:
    neff1 = get_neff(pair[0])
    neff2 = get_neff(pair[1])

    neffs1.append(neff1)
    neffs2.append(neff2)

neffs1 = np.array(neffs1)
neffs2 = np.array(neffs2)
print(neffs1)
print(neffs2)

logneffs1 = np.log(neffs1)
logneffs2 = np.log(neffs2)

fig, ax = plt.subplots(figsize=(3.5,3))
ax.axline((7, 7), slope=1, linestyle='dashed', color='gray', alpha=0.5)
sc = ax.scatter(logneffs1, logneffs2, c=seq_identity, vmin = 65, vmax = 100, cmap = 'cool', alpha=0.5)

x_major_locator = MultipleLocator(0.5)
y_major_locator = MultipleLocator(0.5)

ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

plt.colorbar(sc, label='sequence identity (%)')
plt.xlim(7, 9.5)
plt.ylim(7, 9.5)
plt.xticks(np.arange(7, 10, 0.5))
plt.yticks(np.arange(7, 10, 0.5))
plt.xlabel("log NEFF")
plt.ylabel("log NEFF")
plt.savefig("../output/neff_of_pro_pair.png", bbox_inches='tight')
plt.savefig("../output/neff_of_pro_pair.pdf", bbox_inches='tight')
print("../output/neff_of_pro_pair.png")
