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
#from jax.nn import softmax
from scipy.special import softmax
import matplotlib.pyplot as plt
from matplotlib import colors
from memPDBsMostDiff_one import *


CONTACT_CUTOFF=8
ALL_RANGE=6
SHORT_RANGE=[6,11]
MEDIUM_RANGE=[12,23]
LONG_RANGE=24

plt.rcParams.update({'font.size': 14})

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
            #if pair != "6BVG_A+5IWS_A": continue
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

            depth = [8, 16, 32, 64, 128, 5120]
            colors = ['red','darkorange', 'gold', 'darkkhaki', 'silver', 'grey']
            colors = ['grey','silver','darkkhaki','gold','darkorange', 'red']

            for pro, pro_data in pair_data.items():
                #if pro != '6QV1_B': continue
                print(pro)
                pro_resid = np.asarray(dic_distance[pair][pro]['index'])
                # plot d_in and d_out for at defferent depth
                fig, ax = plt.subplots(figsize=(1.9, 1.9))
                plt.title(pro)

                # min delta D to inward
                min_d_in = 10
                min_d_in_2out = None
                # min delta D to outward
                min_d_out = 10
                min_d_out_2in = None

                for depth_, color_  in zip(depth, colors):
                    d_in_out_path = os.path.join('../output', pro, "d_in_out_"+str(depth_)+".csv")
                    if not os.path.exists(d_in_out_path) or args.force:
                        d_pre_ins = []
                        d_pre_outs = []
                        saved_pres = []

                        if depth_ == 5120:
                            n_model = 5
                            pre_root = "../../AlphaFold/output/dataset/"+pro
                            model_indices = range(0, n_model)
                        else:
                            n_model = 50
                            pre_root = "../output/dataset/"+pro
                            model_indices = range(1, n_model+1)
                        for model_index in model_indices:
                            if depth_ == 5120:
                                pre = pre_root+'/ranked_'+str(model_index)+'.pdb'
                                saved_pre = 'ranked_'+str(model_index)+'.pdb'
                            else:
                                pre = pre_root+'/model'+str(model_index)+'_'+str(int(depth_/2))+'_'+str(depth_)+'.pdb'
                                saved_pre = 'model'+str(model_index)+'_'+str(int(depth_/2))+'_'+str(depth_)+'.pdb'
                            d_pre = intra_dis(pro_resid, file_=pre)

                            d_pre_in = np.mean(abs(d_pre - d_in))
                            d_pre_out = np.mean(abs(d_pre - d_out))
                            
                            d_pre_ins.append(d_pre_in)
                            d_pre_outs.append(d_pre_out)
                            saved_pres.append(saved_pre)

                        df = pd.DataFrame()
                        df["model_name"] = saved_pres
                        df["d_in"] = d_pre_ins
                        df["d_out"] = d_pre_outs
                        df.to_csv(d_in_out_path, index=False)
                    else:
                        df = pd.read_csv(d_in_out_path)
                        d_pre_ins = df["d_in"]
                        d_pre_outs = df["d_out"]

                    min_index = df[['d_in','d_out']].idxmin()
                    print(min_index)
                    min_in_tmp = df.iloc[min_index[0]]['d_in']
                    min_out_tmp = df.iloc[min_index[1]]['d_out']

                    if min_in_tmp < min_d_in:
                        min_d_in = min_in_tmp
                        min_d_in_2out = df.iloc[min_index[0]]['d_out']
                    if min_out_tmp < min_d_out:
                        min_d_out = min_out_tmp 
                        min_d_out_2in = df.iloc[min_index[1]]['d_in']
                        
                    ax.scatter(d_pre_ins, d_pre_outs, color=color_, marker='o', alpha=0.5, label=str(depth_))

                # plot delta D
                plt.hlines(exp_deltaD[pro_in]/2, linestyles=':', xmin=0, xmax=5.5, color='black')
                plt.vlines(exp_deltaD[pro_out]/2, linestyles=':', ymin=0, ymax=5.5, color='black')

                # plot min delta D points
                ax.plot(min_d_in, min_d_in_2out, '^', markeredgecolor='black', fillstyle='none', markersize=8)
                ax.plot(min_d_out_2in, min_d_out, '^', markeredgecolor='black', fillstyle='none', markersize=8)

                plt.xlim(0, 5.5)
                plt.ylim(0, 5.5)
                plt.xticks(np.arange(0, 6, 1))
                plt.yticks(np.arange(0, 6, 1))
                plt.xlabel("$\Delta D $ to IF "+pro_in+" ($\mathrm{\AA}$)")
                plt.ylabel("$\Delta D $ to OF "+pro_out+" ($\mathrm{\AA}$)")                
                
                #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                path = "../output/dataset/" + pro + "/deltaD_in_out.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.savefig(path[:-4]+'.pdf', bbox_inches='tight')
                plt.clf()
                plt.close()

            #break #par

    if args.plot:
        #data = pd.read_csv("../output/acc_on_states_one_model.csv", index_col=False ) 

        #with open('../output/5C78_A_6HRC_D_acc_on_states.pickle', 'rb') as handle:
        with open('../output/acc_on_states.pickle', 'rb') as handle:
            all_data = pickle.load(handle)

        # read pair pros with the order of inward-open and outward-open
        with open('../../../dataset/output/mostdiff_files_seq_cluster70_manu_sel_in_out.txt') as handle:
            in_out_pairs = []
            for line in handle.readlines():
                in_out_pair = line.split()[:2]
                in_out_pairs.extend(in_out_pair)

        ranges = ["Long_range"]
        top_describe = ['Top L']

        for range_ in ranges:
            for top_n in top_describe:
                for pair, pair_data in all_data.items():
                    print(pair)
                    # recognize in/out-ward
                    index_pro = in_out_pairs.index(pair.split('+')[0])
                    if index_pro % 2 == 0: 
                        order_ = [1, 2]
                    else:
                        order_ = [2, 1]
                    
                    for pro, pro_data in pair_data.items():
                        print(pro)

                        fig, ax = plt.subplots(figsize=(6,6))

                        depth = [128, 64, 32, 16, 8]
                        colors = ['darkorange', 'gold', 'darkkhaki', 'silver', 'grey']

                        for depth_, color_  in zip(depth, colors):
                            p_in_out = []
                            for index in order_:

                                # pro_labels.append(pro)
                                # index: 1, 2 -> order_
                                p_in_out.append(pro_data[range_][top_n][str(depth_)][:,index])
                                # p_co, p1, p2, c_co, c1, c2
                                # print(pro_data[range_][top_n][depth_].shape)
                                # print("raw p_co, p1, p2:")
                                # print(pro_data[range_][top_n][depth_][:,0])
                                # print(pro_data[range_][top_n][depth_][:,1])
                                # print(pro_data[range_][top_n][depth_][:,2])
                                # print("mean p_co, p1, p2:")
                                # print(np.mean(pro_data[range_][top_n][depth_][:,0]))
                                # print(np.mean(pro_data[range_][top_n][depth_][:,1]))
                                # print(np.mean(pro_data[range_][top_n][depth_][:,2]))

                            print(p_in_out[0])
                            print(p_in_out[1])
                            ax.scatter(p_in_out[0], p_in_out[1], color=color_, marker='.', alpha=0.2, s=8)
                            #break
                        # plot 50 moldel
                        fig.tight_layout()
                        path = "../output/dataset/" + pro + "/topL_p_in_out.png"
                        plt.xlim(0, 0.1)
                        plt.ylim(0, 0.1)
                        plt.savefig(path, dpi=300)
                        plt.clf()
                        plt.close()

                    break
                #break
            #break
        
