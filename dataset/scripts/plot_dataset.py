# plot the histogram of delta D for

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.weight': 'bold'})
plt.rcParams.update({'axes.labelweight': 'bold'})

data = pd.read_csv("../output/mostdiff_files_seq_cluster70_merge.csv")
deltaD = data['distance'][:860]

print(deltaD)

fig, ax = plt.subplots(figsize=(5,5))
ax.hist(deltaD, bins=20, rwidth=0.8)
ax.set_xlabel(r'Intra-distance difference score (${\rm \AA}$)')
ax.set_ylabel('Count')
plt.xlim(0,6)
plt.ylim(0,280)
plt.xticks(np.arange(0, 8, 2))
plt.yticks(np.arange(0, 350, 70))
fig.savefig("../output/deltaD_hist.pdf", bbox_inches='tight')

