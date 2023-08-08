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
from scipy.special import softmax
import matplotlib.pyplot as plt
from matplotlib import colors

from matplotlib import rcParams
rcParams.update({'font.size': 7})
rcParams.update({'font.weight': 'bold'})
rcParams.update({'axes.labelweight': 'bold'})

CONTACT_CUTOFF=8
ALL_RANGE=6
SHORT_RANGE=[6,11]
MEDIUM_RANGE=[12,23]
LONG_RANGE=24

def plot4metrics(x_labels, v1, v2, v3, v_labels, range_, top_n, y_label_prefix):
    v1 = np.nan_to_num(v1)
    v2 = np.nan_to_num(v2)
    v3 = np.nan_to_num(v3)
    y_labels = y_label_prefix + " (" + range_ + ", " + str(top_n) + ")"

    x = np.arange(len(x_labels))
    width=0.25

    fig, ax = plt.subplots(figsize=(5.9, 1.0))
    rect1 = ax.bar(x- width*3/2, v1, width, color="C0")
    rect2 = ax.bar(x- width/2, v2, width, color="C1")
    rect3 = ax.bar(x+ width/2, v3, width, color="C2")

    lines = np.arange(len(x_labels)+1, step=2)-0.5-0.125
    ax.vlines(lines, 0, 1, 'gray', linestyle='dashed', linewidth=1)

    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.set_ylabel("mfDCA", weight='bold', rotation="90")

    #ax.legend(labels=v_labels)
    ax.set_xlim(lines[0], lines[-1])
    ax.set_ylim(0,1)
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    #ax.set_ylabel(y_labels)
    ax.set_xticks([])
    #ax.set_xticklabels(x_labels, fontsize=11,rotation="0")
    colors = ['C1', 'C2'] * int(len(x_labels)/2)
    for xtick, color in zip(ax.get_xticklabels(), colors):
            xtick.set_color(color)

    fig.tight_layout()
    path = "../output/" + y_label_prefix + "_" + range_ + "_" + top_n[:4] + top_n[-1] + "_mfDCA"
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

def analyze(pre_contact, state1, state2, state_common, key, top_n, real_resid1, real_resid2, range_, dir_, pro):
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
            prefix = range_ 
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

def get_contacts(dir_, pro):
    raw_fasta_file = os.path.join(dir_, pro, pro+".fasta")
    seq2 = list(SeqIO.parse(raw_fasta_file, "fasta"))[0].seq
    length = len(seq2)

    contact_mtx = np.zeros((length, length))
    c_file = os.path.join(dir_, pro, pro+".contact")
    with open(c_file, 'r') as f:
        for line in f:
            i, j, mi, dca = line.split()[:4]
            contact_mtx[int(i)-1,int(j)-1] = dca
            contact_mtx[int(j)-1,int(i)-1] = dca

    return contact_mtx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory of contacts')
    parser.add_argument('--analyze', action='store_true', help='analyze result')
    parser.add_argument('--plot', action='store_true', help='plot figures')
    args = parser.parse_args()

    # recognize in/out-ward
    in_out_pairs = []
    with open('../../../dataset/output/mostdiff_files_seq_cluster70_manu_sel_in_out.txt') as handle:
        for line in handle.readlines():
            in_out_pair = line.split()[:2]
            in_out_pairs.append(in_out_pair[0].upper())
            in_out_pairs.append(in_out_pair[1].upper())

    print(in_out_pairs)

    if args.analyze:
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
                raw_fasta_file = os.path.join("../dataset",
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
                if c_seq == 'empty': break
                
                partial_c = np.take(c_seq, resid, axis=0) 
                partial_c = np.take(partial_c, resid, axis=1) 
                print("c_seq shape:", c_seq.shape)
                print("Truncated Size:", partial_c.shape)

                one_data = {}
                ranges = ["Long_range", "Medium_range", "Short_range", "All_range"]
                top_describe = {'Top L': L, 'Top L/2': int(L/2), 'Top L/5': int(L/5)}
                top_describe = {'Top 2L': int(2*L), 'Top L': L, 'Top L/2': int(L/2), 'Top L/5': int(L/5)}
                for range_ in ranges:
                    range_data = {}
                    for key, top_n in top_describe.items():
                        print("===========================")
                        print(range_, key, top_n)
                        # plot a contact map
                        png_path = os.path.join(args.dir, pro, "raw_contact.png")
                        plot_contact(partial_c, png_path)

                        m = analyze(partial_c, state1, state2, state_common, key, top_n, real_resid1, real_resid2, range_, args.dir, pro)

                        #break # top_ns
                        range_data[key] = np.asarray(m)
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

                p_cos = []
                p1s = []
                p2s = []

                c_cos = []
                c1s = []
                c2s = []
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
                    
                        print(pro)
                        # print(pro_data)

                        pro_labels.append(pro)
                        # p_co, p1, p2, c_co, c1, c2
                        print(pro_data[range_][top_n].shape)
                        print("raw p_co, p1, p2:")
                        print(pro_data[range_][top_n][0])
                        print(pro_data[range_][top_n][order4p[0]])
                        print(pro_data[range_][top_n][order4p[1]])

                        p_cos.append(pro_data[range_][top_n][0])
                        p1s.append(pro_data[range_][top_n][order4p[0]])
                        p2s.append(pro_data[range_][top_n][order4p[1]])


                        print("raw c_co, c1, c2:")
                        print(pro_data[range_][top_n][3])
                        print(pro_data[range_][top_n][order4c[0]])
                        print(pro_data[range_][top_n][order4c[1]])
                        c_cos.append(pro_data[range_][top_n][3])
                        c1s.append(pro_data[range_][top_n][order4c[0]])
                        c2s.append(pro_data[range_][top_n][order4c[1]])

                plot4metrics(pro_labels, p_cos, p1s, p2s, ["shared","inward-open","outward-open"], range_, top_n, y_label_prefix="Precision")
                plot4metrics(pro_labels, c_cos, c1s, c2s, ["shared","inward-open","outward-open"], range_, top_n, y_label_prefix="Coverage")

                # statistics
                print("##############statistics##########")
                print(range_,top_n)
                print("mean of p_cos:", np.mean(p_cos))
                print("std of p_cos:", np.std(p_cos))

                p12 = [i+j for i, j in zip(p1s, p2s)]
                print("mean of p12:", np.mean(p12))
                print("std of p12:", np.std(p12))

                p_favor = [max(i,j) for i, j in zip(p1s, p2s)]
                print("mean of p_favor:", np.mean(p_favor))
                print("std of p_favor:", np.std(p_favor))

                p_unfavor = [min(i,j) for i, j in zip(p1s, p2s)]
                print("mean of p_unfavor:", np.mean(p_unfavor))
                print("std of p_unfavor:", np.std(p_unfavor))
                #break
            #break
        
