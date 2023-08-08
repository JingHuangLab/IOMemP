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
# from jax.nn import softmax
from scipy.special import softmax
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import MultipleLocator
from ../../../dataset/scripts/memPDBsMostDiff_one import *
import random

from matplotlib import rcParams
rcParams.update({'font.size': 7})
rcParams.update({'font.weight': 'bold'})
rcParams.update({'axes.labelweight': 'bold'})

CONTACT_CUTOFF=8
ALL_RANGE=6
SHORT_RANGE=[6,11]
MEDIUM_RANGE=[12,23]
LONG_RANGE=24

def plot4metrics_m0(x_labels, v1, v2, v3, v_labels, range_, top_n, y_label_prefix):
    v1 = np.nan_to_num(v1)
    v2 = np.nan_to_num(v2)
    v3 = np.nan_to_num(v3)
    y_labels = y_label_prefix + " (" + range_ + ", " + str(top_n) + ")"

    x = np.arange(len(x_labels))
    width=0.25

    fig, ax = plt.subplots(figsize=(6.0, 1.4))
    rect1 = ax.bar(x- width*3/2, v1, width, color='C0')
    rect2 = ax.bar(x- width/2, v2, width, color="C1")
    rect3 = ax.bar(x+ width/2, v3, width, color="C2")

    lines = np.arange(len(x_labels)+1, step=2)-0.5-0.125
    ax.vlines(lines, 0, 1, 'gray', linestyle='dashed', linewidth=1)

    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.set_ylabel("AlphaFold", weight='bold', rotation="90")

    ax.set_xlim(lines[0], lines[-1])
    ax.set_ylim(0,1)
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    #ax.set_ylabel(y_labels)
    ax.set_xticks(x - 0.125)
    ax.set_xticklabels(x_labels, rotation="45", ha='right', fontsize=6.5)
    colors = ['C1', 'C2'] * int(len(x_labels)/2)

    for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)

    fig.tight_layout()
    path = "../output/" + y_label_prefix + "_" + range_ + "_" + top_n[:4] + top_n[-1] + "_AlphaFold"
    plt.savefig(path, dpi=600)
    pdfpath = path+'.pdf'
    plt.savefig(pdfpath)

    ax.bar_label(rect1,fmt="%.4f", padding=3, fontsize=2,rotation="90")
    ax.bar_label(rect2,fmt="%.4f", padding=3, fontsize=2,rotation="90")
    ax.bar_label(rect3,fmt="%.4f", padding=3, fontsize=2,rotation="90")
    path = path + "_Number"
    plt.savefig(path, dpi=600)
    plt.clf()
    plt.close()
    print("../output/" + y_label_prefix + "_" + range_ + "_" + top_n[:4] + top_n[-1])
def plot4metrics(x_labels, v1, v2, v3, v1_std, v2_std, v3_std, v_labels, range_, top_n, y_label_prefix):
    v1 = np.nan_to_num(v1)
    v2 = np.nan_to_num(v2)
    v3 = np.nan_to_num(v3)
    v1_std = np.nan_to_num(v1_std)
    v2_std = np.nan_to_num(v2_std)
    v3_std = np.nan_to_num(v3_std)
    y_labels = y_label_prefix + " (" + range_ + ", " + str(top_n) + ")"

    x = np.arange(len(x_labels))
    width=0.25

    fig, ax = plt.subplots()
    rect1 = ax.bar(x- width*3/2, v1, width, yerr=v1_std)
    rect2 = ax.bar(x- width/2, v2, width, yerr=v2_std, color="C1")
    rect3 = ax.bar(x+ width/2, v3, width, yerr=v3_std, color="C2")

    #ax.bar(x_labels, v1, width)
    #ax.bar(x_labels, v2, width, color="red")
    #ax.bar(x_labels, v3, width, color="blue")
    
    ax.legend(labels=v_labels)
    ax.set_ylim(0,1)
    ax.set_ylabel(y_labels)
    ax.set_xticks(x-0.125)
    ax.set_xticklabels(x_labels, fontsize=8,rotation="90")
    colors = ['C1', 'C2'] * int(len(x_labels)/2)
    for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)

    fig.tight_layout()
    path = "../output/" + y_label_prefix + "_" + range_ + "_" + top_n[:4] + top_n[-1] + "_AlphaFold"
    plt.savefig(path, dpi=600)

    ax.bar_label(rect1,fmt="%.4f", padding=3, fontsize=2,rotation="90")
    ax.bar_label(rect2,fmt="%.4f", padding=3, fontsize=2,rotation="90")
    ax.bar_label(rect3,fmt="%.4f", padding=3, fontsize=2,rotation="90")
    path = path + "_Number"
    plt.savefig(path, dpi=600)

    plt.clf()
    plt.close()
    print("../output/" + y_label_prefix + "_" + range_ + "_" + top_n[:4] + top_n[-1])

def get_long_range_filter(resid1, resid2):
    len_ = len(resid1)
    assert len(resid1) == len(resid2)
    filter_ = np.zeros((len_,len_))

    for i in range(len_):
        for j in range(len_):
            if abs(resid1[i] - resid1[j]) >= LONG_RANGE or abs(resid2[i] - resid2[j]) >= LONG_RANGE:
                filter_[i][j] = 1
    return filter_
def plot_contact(data, path):
    print("plot_contact:", data.dtype)
    plt.cla()
    fig, ax = plt.subplots()
    plt.imshow(data, cmap='viridis')
    ax.invert_yaxis()
    plt.colorbar()
    plt.savefig(path, dpi=600)
    plt.clf()
    plt.close()

def plot_real_contact(data, path,merge=False):

    plt.cla()
    fig, ax = plt.subplots()
    if merge:
        cmap = colors.ListedColormap(['white', 'yellow','red','blue'])
        bounds=[-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(data,cmap=cmap, norm=norm)
        plt.colorbar()
    else:
        plt.imshow(data, cmap='Greys')

    ax.invert_yaxis()
    plt.savefig(path, dpi=600)
    plt.clf()
    plt.close()

def plot_acc(data):
    """
    plot accuracy of common, state1, state2 contacts
    """
    labels = data['Structures'].tolist()
    acc_co = data["Accurary common state"].to_numpy()
    acc1 = data["Accuracy in state1"].to_numpy()
    acc2 = data["Accuracy in state2"].to_numpy()
    print("Mean acc. common:", np.mean(acc_co))
    print("Std acc. common:", np.std(acc_co))
    print("Mean acc. in the two state:", np.mean([acc1, acc2]))
    print("Std acc. in the two state:", np.std([acc1, acc2]))
    print("Delta of acc. in the two state:", np.mean(np.abs(acc1-acc2)))
    print("Mean acc. in the prefer state:", np.mean(np.max([acc1, acc2],axis=0)))
    print("Std acc. in the prefer state:", np.std(np.max([acc1, acc2],axis=0)))
    print("Mean acc. in the unprefer state:", np.mean(np.min([acc1, acc2],axis=0)))
    print("Std acc. in the unprefer state:", np.std(np.min([acc1, acc2],axis=0)))


    x = np.arange(len(labels))
    width = 0.25
    
    plt.cla()
    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width/2, acc1, width, label="Acc. in state1", color='green')
    rects2 = ax.bar(x + width/2, acc2, width, label="Acc. in state2", color='red')
    
    ax.set_xticks(x, labels=labels, rotation='vertical')
    ax.set_ylim(0, max(np.max(acc1),np.max(acc2))+0.05)
    ax.set_xlabel("Protein names")
    ax.set_ylabel("Accuracy in the two states")
    ax.set_title("Accuracy in alternative states")
    ax.legend()

    # twin object for another y-axis
    ax2 = ax.twinx()
    rects_co = ax2.bar(x+3*width/2, acc_co, width, label="Acc. in common",color='blue')
    ax2.set_ylim(np.min(acc_co), 1)
    ax2.set_ylabel("Accuracy in common cantact map")
    ax2.legend()

    plt.savefig("../output/accuracy.png", bbox_inches='tight', dpi=500)
    print("../output/accuracy.png")
    plt.close()


def plot_distribution(contact_state1, contact_state2, contact_common, dir_, pro, prefix):
    """
    plot contact distribution
    """
    plt.cla()
    fig, ax = plt.subplots()
    #raw_contact_flat = raw_contact.flatten()
    #partial_contact_flat = partial_contact.flatten()
    ## exclude those residues with prob. no more than 0.00001
    #raw_contact_valid = [ n for n in raw_contact_flat if n > 0.02]
    #partial_contact_valid = [ n for n in partial_contact_flat if n > 0.00001]
    
    ax.hist(contact_common.flatten(), bins=100, range=(0.00001,1), density=True, label="Common", alpha=0.5)
    ax.hist(contact_state1.flatten(), bins=100, range=(0.00001,1), density=True, label="State1", alpha=0.5, color='red')
    ax.hist(contact_state2.flatten(), bins=100, range=(0.00001,1), density=True, label="State2", alpha=0.5, color='blue')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Contacts probability")
    ax.set_ylabel("Percentage(%)")
    ax.set_title("Histogram of contact probability (p>0.00001)")

    plt.legend()
    path = os.path.join(dir_, pro, "pre_contact_p_"+prefix+".png")
    plt.savefig(path, bbox_inches='tight')
    print(path)
    plt.close()

def get_range_filter(resid1, resid2, range_='Long_range'):
    len_ = len(resid1)
    assert len(resid1) == len(resid2)
    filter_ = np.zeros((len_,len_))

    if range_ == 'Long_range':
        for i in range(len_):
            for j in range(len_):
                if abs(resid1[i] - resid1[j]) >= LONG_RANGE or abs(resid2[i] - resid2[j]) >= LONG_RANGE:
                    filter_[i][j] = 1
    elif range_ == 'Medium_range':
        for i in range(len_):
            for j in range(len_):
                gap1 = abs(resid1[i] - resid1[j])
                gap2 = abs(resid2[i] - resid2[j])
                if (gap1 >= MEDIUM_RANGE[0] and gap1 <= MEDIUM_RANGE[1]) or (gap2 >= MEDIUM_RANGE[0] and gap2 <= MEDIUM_RANGE[1]): 
                    filter_[i][j] = 1
    elif range_ == 'Short_range':
        for i in range(len_):
            for j in range(len_):
                gap1 = abs(resid1[i] - resid1[j])
                gap2 = abs(resid2[i] - resid2[j])
                if (gap1 >= SHORT_RANGE[0] and gap1 <= SHORT_RANGE[1]) or (gap2 >= SHORT_RANGE[0] and gap2 <= SHORT_RANGE[1]):
                    filter_[i][j] = 1
    elif range_ == 'All_range':
        for i in range(len_):
            for j in range(len_):
                if abs(resid1[i] - resid1[j]) >= SHORT_RANGE[0] or abs(resid2[i] - resid2[j]) >= SHORT_RANGE[0] :
                    filter_[i][j] = 1
    else:
        print("set range among All_range, Short_range, Medium_range, Long_range")

    return filter_

def long_range_contact(contact):
    """
    exclude short-midium contacts 
    """
    long_contact = []
    L = len(contact)
    for i in range(L):
        for j in range(L):
            if abs(i-j) > LONG_RANGE:
                long_contact.append(contact[i, j])
    long_contact = np.array(long_contact)
    return long_contact

def long_range(contact):
    """
    set short-midium contacts as -1 to exclude them
    """
    l = contact.shape[-1]
    for i in range(l):
        for j in range(l):
            if abs(i-j) < LONG_RANGE:
                contact[:, i, j] = -1

    return contact

def analyze(pre_contact, state1, state2, state_common, key, top_n, real_resid1, real_resid2, range_, rank_i, dir_, pro):
    """
    analyze predicted contact to ground truth
    """
    # contact probability for paired residues, states of which are converted.
    # compare the distribution of raw contacts and contacts only for those state-converted states
    range_filter= get_range_filter(real_resid1, real_resid2, range_)
    state_common = np.logical_and(state_common, range_filter)
    state1 = np.logical_and(state1, range_filter)
    state2 = np.logical_and(state2, range_filter)

    print("Analyze:")
    print("No. of " + range_ + " contacts only in state1:", np.sum(state1))
    print("No. of " + range_ + " contacts only in state2:", np.sum(state2))

    pre_contact = np.multiply(pre_contact, range_filter)

    contact_state1 = np.multiply(pre_contact, state1)
    contact_state2 = np.multiply(pre_contact, state2)
    contact_common = np.multiply(pre_contact, state_common)
    #p_state1 = np.mean(contact_state1)
    #p_state2 = np.mean(contact_state2)
    #p_common = np.mean(contact_common)
    #print("mean pro of all common contact:", p_state1, "mean pro of all state1 contact:", p_state2, "mean pro of all state2 contact:", p_common)
    #print("No. of true state1, state2, common:", np.sum(state1), np.sum(state2), np.sum(state_common))
    if range_ == "All_range":
        if key == "Top L": # only plot one
            # plot histogram of pre_contact
            prefix = range_ + " of rank " + str(rank_i)
            plot_distribution(contact_state1, contact_state2, contact_common, dir_, pro, prefix) 
        

    # Top n contacts
    ind = np.unravel_index(np.argsort(pre_contact, axis=None), pre_contact.shape)
    top_n_selector = np.zeros(pre_contact.shape)
    for i, j in zip(ind[0][-top_n:], ind[1][-top_n:]):
        top_n_selector[i][j] = 1
    assert int(np.sum(top_n_selector)) == top_n

    top_n_pre_contact = np.multiply(pre_contact, top_n_selector)
    print("Mean of " + key + " contact:", np.sum(top_n_pre_contact)/top_n)

    top_n_state1 = np.logical_and(top_n_selector, state1)
    top_n_state2 = np.logical_and(top_n_selector, state2)
    top_n_state_common = np.logical_and(top_n_selector, state_common)

    top_n_num_common = np.sum(top_n_state_common)
    top_n_num_state1 = np.sum(top_n_state1)
    top_n_num_state2 = np.sum(top_n_state2)
    top_n_num_false = top_n - top_n_num_common - top_n_num_state1 - top_n_num_state2

    print("No. of predicted all, common, state1, state2, and false:", top_n, top_n_num_common, top_n_num_state1, top_n_num_state2, top_n_num_false) 

    # proportion
    p_co = top_n_num_common / top_n 
    p1 = top_n_num_state1 / top_n 
    p2 = top_n_num_state2 / top_n 
    print("p_co:", p_co, "p1:", p1, "p2:", p2)

    # coverage
    c_co = top_n_num_common / np.sum(state_common)
    c12 = (top_n_num_state1 + top_n_num_state2) / (np.sum(state1) + np.sum(state2))
    c1 = top_n_num_state1 / np.sum(state1)
    c2 = top_n_num_state2 / np.sum(state2)
    print("c_co:", c_co, "c1:", c1, "c2:", c2)

    # precision
    if top_n_num_common == 0 and top_n_num_false == 0: 
        pp_co = 0
    else:
        pp_co = top_n_num_common / (top_n_num_common + top_n_num_false)

    if top_n_num_state1 == 0 and top_n_num_state2 == 0 and top_n_num_false == 0:
        pp12 = 0
    else:
        pp12 = (top_n_num_state1 + top_n_num_state2) / (top_n_num_state1 + top_n_num_state2 + top_n_num_false)

    if top_n_num_state1 == 0 and top_n_num_false == 0:
        pp1 = 0
    else:
        pp1 = top_n_num_state1 / (top_n_num_state1 + top_n_num_false)

    if top_n_num_state2 == 0 and top_n_num_false == 0:
        pp2 = 0
    else:
        pp2 = top_n_num_state2 / (top_n_num_state2 + top_n_num_false)

    #          1,  2,  3,  4,    5,   6,  7,  8,     9,   10,  11
    #       p_co, p1, p2, c_co, c12, c1, c2, pp_co, pp12, pp1, pp2
    return [p_co, p1, p2, c_co, c12, c1, c2, pp_co, pp12, pp1, pp2]

def get_contacts(dir, pro):
    js = os.path.join(dir, pro, 'ranking_debug.json')
    if not os.path.exists(js):
        return -1

    with open(js) as f:
        rank = json.load(f)

    model_order = rank['order']
    contacts = []
    for m in model_order:
        pkl = os.path.join(dir, pro, 'result_' + m + '.pkl')
        with open(pkl, 'rb') as f:
            prediction_result = pickle.load(f)
        dist_bins = np.append([0],prediction_result["distogram"]["bin_edges"])
        contact_mtx = softmax(prediction_result["distogram"]["logits"],axis=-1)[:,:,dist_bins < CONTACT_CUTOFF].sum(-1)

        contacts.append(contact_mtx)

    return np.asarray(contacts) # for 5 models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='../output/dataset', help='directory of AlphaFold results')
    parser.add_argument('--analyze', action='store_true', help='calculate evalution metrics at a serious ranges')
    parser.add_argument('--force', action='store_true', help='force calculate')
    parser.add_argument('--plot', action='store_true', help='plot figures')
    parser.add_argument('--pre_exp_delta_af2', action='store_true', help='delta D of af2 predicted and experimental structres')
    parser.add_argument('--simple_contact_map', action='store_true', help='plot predicted contact map for model 0')

    args = parser.parse_args()

    # get IF/OF info.
    in_out_pairs = []
    with open('../../../dataset/output/mostdiff_files_seq_cluster70_manu_sel_in_out.txt') as handle:
        for line in handle.readlines():
            in_out_pair = line.split()[:2]
            in_out_pairs.append(in_out_pair[0].upper())
            in_out_pairs.append(in_out_pair[1].upper())

    print(in_out_pairs)

    if args.simple_contact_map:
        with open('../../../dataset/output/tm_common_res_dis.pickle', 'rb') as handle:
            dic_distance = pickle.load(handle)

        for pair in dic_distance:
            if pair != '6BTX_A+5AYM_A': continue
            pair_data = {}
            pros = pair.split("+")

            print("State1:", pros[0])
            print("State2:", pros[1])
            dis_maxtrix1 = np.asarray(dic_distance[pair][pros[0]]['distance matrix'])
            dis_maxtrix2 = np.asarray(dic_distance[pair][pros[1]]['distance matrix'])
            real_resid1 = np.asarray(dic_distance[pair][pros[0]]['index']) 
            real_resid2 = np.asarray(dic_distance[pair][pros[1]]['index'])
            #print(real_resid1)
            #print(real_resid2)

            resid1 = real_resid1 - 1 
            resid2 = real_resid2 - 1
            resids = [resid1, resid2]
            #print("dis_maxtrix1:", dis_maxtrix1)
            #print("dis_maxtrix2:", dis_maxtrix2)
            gt_contact1 = dis_maxtrix1 < CONTACT_CUTOFF
            gt_contact2 = dis_maxtrix2 < CONTACT_CUTOFF

            # contact in common
            state_common = np.logical_and(gt_contact1, gt_contact2)
            # contact in state1 or state2
            state_conversion = np.logical_xor(gt_contact1, gt_contact2)
            state1 = np.logical_and(state_conversion, gt_contact1)
            state2 = np.logical_and(state_conversion, gt_contact2)
            print("No. of all-range contacts only in state1:", np.sum(state1))
            print("No. of all-range contacts only in state2:", np.sum(state2))

            # merge common contact(1), contact in state1(2), contact in state2(3), non-contact(0)
            L = len(resid1)
            merge_contact = np.zeros((L,L))
            merge_contact = merge_contact + state_common.astype(int) * 1 + state1.astype(int)*2 + state2.astype(int)*3
            merge_contat_path = os.path.join(args.dir, pros[0], "merge_contact.png")
            plot_real_contact(merge_contact, merge_contat_path, True)

            
            # plot a contact map
            png_path_t1 = os.path.join(args.dir, pros[0], "all_range_state1.png")
            png_path_t2 = os.path.join(args.dir, pros[1], "all_range_state2.png")
            plot_real_contact(state1.astype(int), png_path_t1)
            plot_real_contact(state2.astype(int), png_path_t2)

            for p in range(len(pros)):
                pro = pros[p]
                # if pro != '5AYM_A': continue
                # check fasta length
                new_fasta_file = os.path.join("../../../dataset/output/PDBs_tm_common_res_dis",
                        pro[:4].lower()+pro[5:] + "_ignorechain.fasta")
                seq1 = list(SeqIO.parse(new_fasta_file, "fasta"))[0].seq
                raw_fasta_file = os.path.join("../seq_cluster70_prots_final_fas_from_pdb",
                        pro, pro+".fasta")
                seq2 = list(SeqIO.parse(raw_fasta_file, "fasta"))[0].seq
                print("Checking ", pro)
                assert str(seq1) == str(seq2)
                #continue

                resid = resids[p]
                #print("No. of TM residues:", len(resid))

                # predicted contact for the sequences of protein
                c_seq = get_contacts(args.dir, pro)
                
                partial_c = np.take(c_seq, resid, axis=1) 
                partial_c = np.take(partial_c, resid, axis=2) 
                print("c_seq shape:", c_seq.shape)
                print("Truncated Size:", partial_c.shape)

                one_data = {}
                #ranges = ["Long_range", "Medium_range", "Short_range", "All_range"]
                #top_describe = {'Top L': L, 'Top L/2': int(L/2), 'Top L/5': int(L/5)}
                ranges = ["Long_range"]
                top_describe = {'Top L': L}
                # top_describe = {'Top 2L': int(2*L)}
                for range_ in ranges:
                    range_data = {}
                    for key, top_n in top_describe.items():
                        metrics = []
                        for rank_i, model_contact in enumerate(partial_c):
                            print("===========================")
                            print("Model rank ",rank_i)
                            print(range_, key, top_n)
                            # plot a contact map
                            png_path = os.path.join(args.dir, pro, "raw_contact_model0.png")
                            plot_contact(model_contact, png_path)
                            break
                # break

    if args.pre_exp_delta_af2:
        rcParams.update({'font.size': 7})
        path = '../output/delta_prefer_unprefer.png'
        if not os.path.exists(path) or args.force:
            with open('../../../dataset/output/tm_common_res_dis.pickle', 'rb') as handle:
                dic_distance = pickle.load(handle)

            prefer_unprefer_path = '../../output/af2_preferable_state.txt'
            with open(prefer_unprefer_path, 'r') as f:
                prefer_unprefer_list = f.readlines()

            # print(prefer_unprefer_list)

            pros_ = []
            deltas_prefer_unprefer = []
            for pair, prefer_unprefer in zip(dic_distance, prefer_unprefer_list):
                print(pair)
                single = prefer_unprefer.split()[1]
                
                pros_.append(prefer_unprefer.split()[0])
                pros_.append(prefer_unprefer.split()[2])
                
                pair_data = {}
                pros = pair.split("+")

                print("State1:", pros[0])
                print("State2:", pros[1])
                real_resid1 = np.asarray(dic_distance[pair][pros[0]]['index']) 
                real_resid2 = np.asarray(dic_distance[pair][pros[1]]['index'])
                dis_maxtrix1 = np.asarray(dic_distance[pair][pros[0]]['distance matrix'])
                dis_maxtrix2 = np.asarray(dic_distance[pair][pros[1]]['distance matrix'])

                pre1 = '../output/dataset/'+pros[0]+'/ranked_0.pdb'
                pre2 = '../output/dataset/'+pros[1]+'/ranked_0.pdb'
                exp1 = '../../../dataset/output/PDBs_tm_common_res_dis/'+pros[0]+'.pdb'
                exp2 = '../../../dataset/output/PDBs_tm_common_res_dis/'+pros[1]+'.pdb'

                d1 = intra_dis(real_resid1, file_=pre1)
                d2 = intra_dis(real_resid2, file_=pre2)

                d11 = np.mean(abs(d1 - dis_maxtrix1))
                d12 = np.mean(abs(d1 - dis_maxtrix2))

                d21 = np.mean(abs(d2 - dis_maxtrix1))
                d22 = np.mean(abs(d2 - dis_maxtrix2))

                if single == '>':
                    delta_prefer_unprefer1 = [d11, d12]
                    delta_prefer_unprefer2 = [d21, d22]

                else:
                    print(single)
                    delta_prefer_unprefer1 = [d12, d11]
                    delta_prefer_unprefer2 = [d22, d21]

                print(delta_prefer_unprefer1)
                print(delta_prefer_unprefer2)

                deltas_prefer_unprefer.append(delta_prefer_unprefer1)
                deltas_prefer_unprefer.append(delta_prefer_unprefer2)

            deltas_prefer_unprefer = np.array(deltas_prefer_unprefer)
            print(deltas_prefer_unprefer)
            print(deltas_prefer_unprefer.shape)

            df = pd.DataFrame()
            df['pro_seq'] = pros_
            df['delta_prefer'] = deltas_prefer_unprefer[:, 0]
            df['delta_unprefer'] = deltas_prefer_unprefer[:, 1]
            df.to_csv("../output/delta_prefer_unprefer.csv", index=False)

        df = pd.read_csv("../output/delta_prefer_unprefer.csv")

        # calculate R^2
        model = LinearRegression()
        X = df['delta_prefer'].to_numpy().reshape(-1, 1)
        model.fit(X, df['delta_unprefer'])
        r_squared = model.score(X, df['delta_unprefer'])
        print("r_squared:", r_squared)


        fig, ax = plt.subplots(figsize=(2.9, 2.9))

        for i, pro in enumerate(df['pro_seq']):
            try:
                index_ = in_out_pairs.index(pro)
            except:
                continue

            excluded_pros = ['6QV1_B', '4Q4A_B', 
                             '6GCI_A', '4C9J_B', 
                             '6VS1_A', '6GV1_A',
                             '6QV1_C', '3QF4_A', 
                             '3VVS_A', '6FHZ_A', 
                             '6E9N_B', '6E9O_B', 
                             '6RAJ_B', '5MKK_B', 
                             '5IWS_A', '6BVG_A', 
                             '5C78_A', '6HRC_D']

            if index_ % 2 == 0:
                plt.scatter(df['delta_prefer'][i], df['delta_unprefer'][i],c='C1', alpha=0.7, s=8, edgecolors='none')
                if pro in excluded_pros:
                    continue

                if pro == '6BVG_A':
                    ax.annotate(pro, (df['delta_prefer'][i] + 1.35, df['delta_unprefer'][i] ),c='C1')
                    continue
                if pro == '6S3Q_A':
                    ax.annotate(pro, (df['delta_prefer'][i] + 0.05, df['delta_unprefer'][i] + 0.15),c='C1')
                    continue
                if pro == '6XMS_A':
                    ax.annotate(pro, (df['delta_prefer'][i] + 0.05, df['delta_unprefer'][i] - 0.15),c='C1')
                    continue
                if pro == '6NC7_A':
                    ax.annotate(pro, (df['delta_prefer'][i] + 0.05, df['delta_unprefer'][i] - 0.3),c='C1')
                    continue
                if pro == '6BTX_A':
                    ax.annotate(pro, (df['delta_prefer'][i] - 0.4, df['delta_unprefer'][i] + 0.1) ,c='C1')
                    continue
                ax.annotate(pro, (df['delta_prefer'][i] + 0.05, df['delta_unprefer'][i]),c='C1')

            else:
                plt.scatter(df['delta_prefer'][i], df['delta_unprefer'][i],c='C2', alpha=0.5, s=8, marker='^', edgecolors='none')
                if pro in excluded_pros:
                    continue
                if pro == '6XMU_A' :
                    ax.annotate(pro, (df['delta_prefer'][i] + 0.05, df['delta_unprefer'][i] + 0.15),c='C2')
                    continue
                if pro == '7N98_A' :
                    ax.annotate(pro, (df['delta_prefer'][i] + 0.05, df['delta_unprefer'][i] - 0.2),c='C2')
                    continue
                if pro == '7DQV_A' :
                    ax.annotate(pro, (df['delta_prefer'][i] + 0.05, df['delta_unprefer'][i] - 0.2),c='C2')
                    continue
                if pro == '6X3E_A' :
                    ax.annotate(pro, (df['delta_prefer'][i] + 0.08 , df['delta_unprefer'][i] - 0.12),c='C2')
                    continue
                if pro == '4M64_A' :
                    ax.annotate(pro, (df['delta_prefer'][i] + 0.05 , df['delta_unprefer'][i] - 0.35),c='C2')
                    continue
                ax.annotate(pro, (df['delta_prefer'][i] + 0.05, df['delta_unprefer'][i]),c='C2')

        print('../output/delta_prefer_unprefer.png')
        plt.xlim(0, 7)
        plt.ylim(0, 7)
        plt.legend(['OF','IF'])
        plt.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='k', lw=0.5)

        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        
        plt.xlabel(r'$\Delta D $ to the contact-preferable structure (${\mathrm{\AA}}$)')
        plt.ylabel(r'$\Delta D $ to the contact-unpreferable structure ($\mathrm{\AA}$)')
        path_pdf = '../output/delta_prefer_unprefer.pdf'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.savefig(path_pdf, bbox_inches='tight')
        plt.clf()
        plt.close()
        
    if args.analyze:
        # get ground truth
        with open('../../../dataset/output/tm_common_res_dis.pickle', 'rb') as handle:
            dic_distance = pickle.load(handle)

        accs_common = []
        accs_state1 = []
        accs_state2 = []
        pro_all = []

        all_data = {}
        for pair in dic_distance:
            print(pair)
            pair_data = {}
            pros = pair.split("+")

            print("State1:", pros[0])
            print("State2:", pros[1])
            dis_maxtrix1 = np.asarray(dic_distance[pair][pros[0]]['distance matrix'])
            dis_maxtrix2 = np.asarray(dic_distance[pair][pros[1]]['distance matrix'])
            real_resid1 = np.asarray(dic_distance[pair][pros[0]]['index']) 
            real_resid2 = np.asarray(dic_distance[pair][pros[1]]['index'])
            print(real_resid1)
            print(real_resid2)

            resid1 = real_resid1 - 1 
            resid2 = real_resid2 - 1
            resids = [resid1, resid2]
            #print("dis_maxtrix1:", dis_maxtrix1)
            #print("dis_maxtrix2:", dis_maxtrix2)
            gt_contact1 = dis_maxtrix1 < CONTACT_CUTOFF
            gt_contact2 = dis_maxtrix2 < CONTACT_CUTOFF

            # contact in common
            state_common = np.logical_and(gt_contact1, gt_contact2)
            # contact in state1 or state2
            state_conversion = np.logical_xor(gt_contact1, gt_contact2)
            state1 = np.logical_and(state_conversion, gt_contact1)
            state2 = np.logical_and(state_conversion, gt_contact2)
            print("No. of all-range contacts common in two states:", np.sum(state_common))
            print("No. of all-range contacts only in state1:", np.sum(state1))
            print("No. of all-range contacts only in state2:", np.sum(state2))

            # merge common contact(1), contact in state1(2), contact in state2(3), non-contact(0)
            L = len(resid1)
            merge_contact = np.zeros((L,L))
            merge_contact = merge_contact + state_common.astype(int) * 1 + state1.astype(int)*2 + state2.astype(int)*3
            merge_contat_path = os.path.join(args.dir, pros[0], "merge_contact.png")
            plot_real_contact(merge_contact, merge_contat_path, True)

            # check
            #index = np.where(state_conversion == True)
            #index1 = index[0]
            #index2 = index[1]
            #for r1, r2 in zip(index1, index2):
            #    if dis_maxtrix1[r1, r2] < dis_maxtrix2[r1, r2]:
            #        print("state1 converted residue pair:", real_resid1[r1], real_resid1[r2])
            #        print("state2 converted residue pair:", real_resid2[r1], real_resid2[r2])
            #        print("distance of distance matrix 1:", dis_maxtrix1[r1, r2])
            #        print("distance of distance matrix 2:", dis_maxtrix2[r1, r2])
            
            # plot a contact map
            png_path_t1 = os.path.join(args.dir, pros[0], "all_range_state1.png")
            png_path_t2 = os.path.join(args.dir, pros[1], "all_range_state2.png")
            plot_real_contact(state1.astype(int), png_path_t1)
            plot_real_contact(state2.astype(int), png_path_t2)

            for p in range(len(pros)):
                pro = pros[p]
                # check fasta length
                new_fasta_file = os.path.join("../../../dataset/output/PDBs_tm_common_res_dis",
                        pro[:4].lower()+pro[5:] + "_ignorechain.fasta")
                seq1 = list(SeqIO.parse(new_fasta_file, "fasta"))[0].seq
                raw_fasta_file = os.path.join("../seq_cluster70_prots_final_fas_from_pdb",
                        pro, pro+".fasta")
                seq2 = list(SeqIO.parse(raw_fasta_file, "fasta"))[0].seq
                print("Checking ", pro)
                assert str(seq1) == str(seq2)
                #continue

                resid = resids[p]
                pro_all.append(pro)
                #print("No. of TM residues:", len(resid))

                # predicted contact for the sequences of protein
                c_seq = get_contacts(args.dir, pro)
                
                partial_c = np.take(c_seq, resid, axis=1) 
                partial_c = np.take(partial_c, resid, axis=2) 
                print("c_seq shape:", c_seq.shape)
                print("Truncated Size:", partial_c.shape)

                one_data = {}
                ranges = ["Long_range", "Medium_range", "Short_range", "All_range"]
                top_describe = {'Top L': L, 'Top L/2': int(L/2), 'Top L/5': int(L/5)}
                # top_describe = {'Top 2L': int(2*L), 'Top L': L, 'Top L/2': int(L/2), 'Top L/5': int(L/5)}
                for range_ in ranges:
                    range_data = {}
                    for key, top_n in top_describe.items():
                        metrics = []
                        for rank_i, model_contact in enumerate(partial_c):
                            print("===========================")
                            print("Model rank ",rank_i)
                            print(range_, key, top_n)
                            # plot a contact map
                            png_path = os.path.join(args.dir, pro, "raw_contact.png")
                            plot_contact(model_contact, png_path)

                            m = analyze(model_contact, state1, state2, state_common, key, top_n, real_resid1, real_resid2, range_, rank_i, args.dir, pro)

                            metrics.append(m)
                            #break # models
                        #break # top_ns
                        range_data[key] = np.asarray(metrics)
                    one_data[range_] = range_data
                    #break # ranges
                pair_data[pro] = one_data
                #break # pros
            all_data[pair] = pair_data
            #break # pairs

        print(all_data)
        matrics_pickle = "../output/acc_on_states.pickle"
        with open(matrics_pickle, "wb") as handle:
            pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Save to ../output/acc_on_states.pickle")

    if args.plot:
        with open('../output/acc_on_states.pickle', 'rb') as handle:
            all_data = pickle.load(handle)

        ranges = ["Long_range", "Medium_range", "Short_range", "All_range"]
        top_describe = ['Top L', 'Top L/2', 'Top L/5']

        for range_ in ranges:
            for top_n in top_describe:
                #plot a figure at this range and top 

                pro_labels = []

                p_cos0 = []
                p1s0 = []
                p2s0 = []
                p_cos_mean = []
                p1s_mean = []
                p2s_mean = []
                p_cos_std = []
                p1s_std = []
                p2s_std = []

                c_cos0 = []
                c1s0 = []
                c2s0 = []
                c_cos_mean = []
                c1s_mean = []
                c2s_mean = []
                c_cos_std = []
                c1s_std = []
                c2s_std = []

                for pair, pair_data in all_data.items():
                    print(pair)
                    try:
                        index_pro = in_out_pairs.index(pair.split('+')[0])
                    except:
                        continue
                    if index_pro % 2 == 0:
                        order_ = [0, 1]
                        order4p = [1,2]
                        order4c = [4,5]
                    else:
                        order_ = [1, 0]
                        order4p = [2,1]
                        order4c = [5,4]

                    pros = pair.split('+')
                    pro_in = pros[order_[0]]
                    pro_out = pros[order_[1]]
                    

                    for order_index in order_:
                        pro = pros[order_index]

                        try:
                            pro_data = pair_data[pro]
                        except:
                            continue

                        pro_labels.append(pro)
                        # p_co, p1, p2, c_co, c1, c2
                        print(pro_data[range_][top_n].shape)
                        print("raw p_co, p1, p2:")
                        print(pro_data[range_][top_n][:,0])
                        print(pro_data[range_][top_n][:,1])
                        print(pro_data[range_][top_n][:,2])
                        print("mean p_co, p1, p2:")
                        print(np.mean(pro_data[range_][top_n][:,0]))
                        print(np.mean(pro_data[range_][top_n][:,order4p[0]]))
                        print(np.mean(pro_data[range_][top_n][:,order4p[1]]))

                        p_cos0.append(pro_data[range_][top_n][0,0])
                        p1s0.append(pro_data[range_][top_n][0,order4p[0]])
                        p2s0.append(pro_data[range_][top_n][0,order4p[1]])
                        p_cos_mean.append(np.mean(pro_data[range_][top_n][:,0]))
                        p1s_mean.append(np.mean(pro_data[range_][top_n][:,order4p[0]]))
                        p2s_mean.append(np.mean(pro_data[range_][top_n][:,order4p[1]]))

                        p_cos_std.append(np.std(pro_data[range_][top_n][:,0]))
                        p1s_std.append(np.std(pro_data[range_][top_n][:,order4p[0]]))
                        p2s_std.append(np.std(pro_data[range_][top_n][:,order4p[0]]))

                        print("raw c_co, c1, c2:")
                        print(pro_data[range_][top_n][:,3])
                        print(pro_data[range_][top_n][:,order4c[0]])
                        print(pro_data[range_][top_n][:,order4c[1]])
                        print("mean c_co, c1, c2:")
                        print(np.mean(pro_data[range_][top_n][:,3]))
                        print(np.mean(pro_data[range_][top_n][:,order4c[0]]))
                        print(np.mean(pro_data[range_][top_n][:,order4c[1]]))

                        c_cos0.append(pro_data[range_][top_n][0,3])
                        c1s0.append(pro_data[range_][top_n][0,order4c[0]])
                        c2s0.append(pro_data[range_][top_n][0,order4c[1]])
                        c_cos_mean.append(np.mean(pro_data[range_][top_n][:,3]))
                        c1s_mean.append(np.mean(pro_data[range_][top_n][:,order4c[0]]))
                        c2s_mean.append(np.mean(pro_data[range_][top_n][:,order4c[1]]))

                        c_cos_std.append(np.std(pro_data[range_][top_n][:,3]))
                        c1s_std.append(np.std(pro_data[range_][top_n][:,4]))
                        c2s_std.append(np.std(pro_data[range_][top_n][:,5]))
                    #break

                print(len(p1s0))
                print(p1s0[22], p2s0[22])
                print(p1s0[23], p2s0[23])
                plot4metrics_m0(pro_labels, p_cos0, p1s0, p2s0, ["shared","inward-open","outward-open"], range_, top_n, y_label_prefix="Precision_model0") # Here Precision actuallly is percentage in main text.
                plot4metrics_m0(pro_labels, c_cos0, c1s0, c2s0, ["shared","inward-open","outward-open"], range_, top_n, y_label_prefix="Coverage_model0")

                plot4metrics(pro_labels, p_cos_mean, p1s_mean, p2s_mean, p_cos_std, p1s_std, p2s_std, ["shared","inward-open","outward-open"], range_, top_n, y_label_prefix="Precision")
                plot4metrics(pro_labels, c_cos_mean, c1s_mean, c2s_mean, c_cos_std, c1s_std, c2s_std, ["shared","inward-open","outward-open"], range_, top_n, y_label_prefix="Coverage")
                print("mean of p_cos0:", np.mean(p_cos0))
                print("std of p_cos0:", np.std(p_cos0))
                print("mean of p1s0:", np.mean(p1s0))
                print("std of p2s0:", np.std(p_cos0))
                # break
            # break
        
