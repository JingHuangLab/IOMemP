#!/usr/bin/python

"""
quantify the mutistate information contained in contacts
"""

import os
from Bio import SeqIO
import numpy as np
import pandas as pd
import argparse
import pickle
import json
import glob
#from jax.nn import softmax
from scipy.special import softmax
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from memPDBsMostDiff_one import *

from matplotlib import cm

plt.rcParams.update({'font.size': 14})

import sys
sys.path.append("../../../alphafold-2.2.0")

from alphafold.model import model
from alphafold.model import config
from alphafold.model import data

from alphafold.data import parsers
from alphafold.data import pipeline

from alphafold.common import protein
from alphafold.common import residue_constants


CONTACT_CUTOFF=8
ALL_RANGE=6
SHORT_RANGE=[6,11]
MEDIUM_RANGE=[12,23]
LONG_RANGE=24

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory of contacts')
    parser.add_argument('--plot', action='store_true', help='plot figures')
    parser.add_argument('--pre_exp_delta_af2', action='store_true', help='delta D of af2 predicted and experimental structres')
    parser.add_argument('--force', action='store_true', help='force calculate')

    args = parser.parse_args()

    if args.pre_exp_delta_af2:
        plt.rcParams.update({'font.size': 10})
        plt.rcParams.update({'font.weight': 'bold'})
        plt.rcParams.update({'axes.labelsize': 10})
        plt.rcParams.update({'axes.labelweight': 'bold'})
        plt.rcParams.update({'axes.titlesize': 10})
        plt.rcParams.update({'axes.titleweight': 'bold'})
        # read exp. delta D between in and out
        exp_deltaD = {}
        with open('../../../dataset/output/deltaD', 'r') as f:
            for l in f.readlines():
                pro, deltaD = l.split()[:2]
                exp_deltaD[str(pro)] = float(deltaD)

        with open('../../../dataset/output/tm_common_res_dis.pickle', 'rb') as handle:
            dic_distance = pickle.load(handle)

        # recognize in/out-ward
        with open('../../../dataset/output/mostdiff_files_seq_cluster70_manu_sel_in_out.txt') as handle:
            in_out_pairs = []
            for line in handle.readlines():
                in_out_pair = line.split()[:2]
                in_out_pairs.append(in_out_pair[0].upper())
                in_out_pairs.append(in_out_pair[1].upper())

        for pair, pair_data in dic_distance.items():
            # if pair != "6BVG_A+5IWS_A": continue
            print(pair)

            index_pro = in_out_pairs.index(pair.split('+')[0])
            if index_pro % 2 == 0: 
                order_ = [0, 1]
            else:
                order_ = [1, 0]

            pros = pair.split('+')
            pro_in = pros[order_[0]]
            pro_out = pros[order_[1]]

            # exp_in = '~/projects/coevo2struc/protein_data/output/PDBs_tm_common_res_dis/'+pro_in+'.pdb'
            # exp_out = '~/projects/coevo2struc/protein_data/output/PDBs_tm_common_res_dis/'+pro_out+'.pdb'

            exp_in_resid = np.asarray(dic_distance[pair][pro_in]['index'])
            exp_out_resid = np.asarray(dic_distance[pair][pro_out]['index'])

            d_in = np.asarray(dic_distance[pair][pro_in]['distance matrix'])
            d_out = np.asarray(dic_distance[pair][pro_out]['distance matrix'])

            for pro, pro_data in pair_data.items():

                print(pro)
                pro_resid = np.asarray(dic_distance[pair][pro]['index'])

                # min delta D to inward
                min_d_in = 10
                min_d_in_2out = None
                # min delta D to outward
                min_d_out = 10
                min_d_out_2in = None

                # plot d_in and d_out for at defferent depth
                fig, ax = plt.subplots(figsize=(1.9, 1.9))
                plt.title(pro)

                d_in_out_path = os.path.join('../output/', pro, "d_in_out.csv")
                if not os.path.exists(d_in_out_path) or args.force:
                    d_pre_ins = []
                    d_pre_outs = []
                    plddts = []
                    n_seq_all = 0

                    d_pre_in_min = 10
                    d_pre_in_min_file = None
                    d_pre_out_min = 10
                    d_pre_out_min_file = None

                    find_pdbs = glob.glob("../output/"+pro+"/*.pdb")
                    n_pdbs = len(find_pdbs)

                    print("len. of pdbs: ", n_pdbs)

                    for p in range(n_pdbs):
                        pkl = "../output/"+pro+"/EX_"+f'{p:03d}'+".pkl"
                        with open(pkl, 'rb') as f:
                            pre_info = pickle.load(f)
                            plddt = pre_info['pLDDT']
                        if plddt < 50:
                            continue

                        plddts.append(plddt)

                        pre = "../output/"+pro+"/EX_"+f'{p:03d}'+".pdb"
                        #print("pdb file:", pre)

                        d_pre = intra_dis(pro_resid, file_=pre)

                        d_pre_in = np.mean(abs(d_pre - d_in))
                        d_pre_out = np.mean(abs(d_pre - d_out))
                        
                        d_pre_ins.append(d_pre_in)
                        d_pre_outs.append(d_pre_out)


                        msa = "../output/"+pro+"/EX_"+f'{p:03d}'+".a3m"
                        with open(msa, 'r') as f:
                            n_seq = len(f.readlines())
                            #print(n_seq)
                            n_seq_all = n_seq_all + n_seq

                        if d_pre_in < d_pre_in_min:
                            d_pre_in_min = d_pre_in
                            d_pre_in_min_file = pre
                        if d_pre_out < d_pre_out_min:
                            d_pre_out_min = d_pre_out
                            d_pre_out_min_file = pre
                        #break
                    
                    # print file with minimum delta_in or delta_out
                    print("d_pre_in_min: ", d_pre_in_min)
                    print("d_pre_in_min_file: ", d_pre_in_min_file)
                    print("d_pre_out_min: ", d_pre_out_min)
                    print("d_pre_out_min_file: ", d_pre_out_min_file)
                    #exit(0)
                    
                    plddts = np.asarray(plddts)

                    df = pd.DataFrame()
                    df["d_in"] = d_pre_ins
                    df["d_out"] = d_pre_outs
                    df["plddt"] = plddts
                    df["ave_n_seq"] = n_seq_all / n_pdbs
                    df.to_csv(d_in_out_path, index=False)
                else:
                    df = pd.read_csv(d_in_out_path)
                    d_pre_ins = df["d_in"]
                    d_pre_outs = df["d_out"]
                    plddts = df["plddt"]

                    min_index = df.idxmin()
                    print(min_index)
                    min_in_tmp = df.iloc[min_index[0]]['d_in']
                    min_out_tmp = df.iloc[min_index[1]]['d_out']

                    if min_in_tmp < min_d_in:
                        min_d_in = min_in_tmp
                        min_d_in_2out = df.iloc[min_index[0]]['d_out']
                    if min_out_tmp < min_d_out:
                        min_d_out = min_out_tmp 
                        min_d_out_2in = df.iloc[min_index[1]]['d_in']
                        

                norm = cm.colors.Normalize(vmax=100, vmin=50)
                cmap = cm.PRGn
                im = ax.scatter(d_pre_ins, d_pre_outs, c=plddts, cmap='viridis', norm=norm, marker='o', alpha=0.5)

                # plot exp delta D
                plt.hlines(exp_deltaD[pro_in]/2, linestyles=':', xmin=0, xmax=5.5, color='black')
                plt.vlines(exp_deltaD[pro_out]/2, linestyles=':', ymin=0, ymax=5.5, color='black')

                # plot min delta D points
                ax.plot(min_d_in, min_d_in_2out, '^', markeredgecolor='black', fillstyle='none', markersize=8)
                ax.plot(min_d_out_2in, min_d_out, '^', markeredgecolor='black', fillstyle='none', markersize=8)

                # ax.legend()
                plt.xlim(0, 5.5)
                plt.ylim(0, 5.5)
                plt.xticks(np.arange(0, 6, 1))
                plt.yticks(np.arange(0, 6, 1))
                plt.xlabel("$\Delta D $ to IF "+pro_in+" ($\mathrm{\AA}$)")
                plt.ylabel("$\Delta D $ to OF "+pro_out+" ($\mathrm{\AA}$)")                
                # create an Axes on the right side of ax. The width of cax will be 5%
                # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                #divider = make_axes_locatable(ax)
                #cax = divider.append_axes("right", size="5%", pad=0.05)
                #plt.colorbar(im, cax=cax, label='plDDT')

                path = "../output/" + pro + "/deltaD_in_out.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.savefig(path[:-4]+'.pdf', bbox_inches='tight')
                plt.clf()
                plt.close()

            # break #pair

