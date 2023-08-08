#!/usr/bin/python

"""
The seqeuence distance profile for state-unique contacts
"""

import os
import numpy as np
import pandas as pd
import argparse
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors

CONTACT_CUTOFF = 8
ALL_RANGE=6
SHORT_RANGE=[6,11]
MEDIUM_RANGE=[12,23]
LONG_RANGE=24

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='force to calculate the sequence distace')
    parser.add_argument('--type_flag', action='store_true')
    parser.add_argument('--type', choices=['common', 'state'], help='sequence distace for common/state-unique contact')
    parser.add_argument('--proportion', action='store_true', help='proportion of shared-, IF-, OF- contacts at different range')
    args = parser.parse_args()

    if args.proportion:
        in_out_pairs = []
        with open('../output/mostdiff_files_seq_cluster70_manu_sel_in_out.txt') as handle:
            for line in handle.readlines():
                in_out_pair = line.split()[:2]
                # in_out_pairs.append(in_out_pair[0].upper())
                # in_out_pairs.append(in_out_pair[1].upper())
                in_out_pairs.append([in_out_pair[0].upper(), in_out_pair[1].upper()])

        with open('../output/tm_common_res_dis.pickle', 'rb') as handle:
            dic_distance = pickle.load(handle)

        # record the number of contacts and L
        n_contacts = []
        n_res = []

        #for pair in dic_distance:
        for pros in in_out_pairs:
            #if pair == "7BH2_B+5MRW_F": continue
            pair = pros[0]+'+'+pros[1]
            if not pair in dic_distance.keys(): pair = pros[1]+'+'+pros[0]

            print(pair)
            pair_data = {}
            #pros = pair.split("+")

            # if/of order
            # try:
            #     index_pro = in_out_pairs.index(pair.split('+')[0])
            # except:
            #     continue
            # if index_pro % 2 == 0:
            #     order_ = [0, 1]
            # else:
            #     order_ = [1, 0]

            dis_maxtrix1 = np.asarray(dic_distance[pair][pros[0]]['distance matrix'])
            dis_maxtrix2 = np.asarray(dic_distance[pair][pros[1]]['distance matrix'])
            real_resid1 = np.asarray(dic_distance[pair][pros[0]]['index']) 
            real_resid2 = np.asarray(dic_distance[pair][pros[1]]['index'])
            resid1 = real_resid1 - 1 
            resid2 = real_resid2 - 1
            resids = [resid1, resid2]

            n_res.append(len(resid1))

            gt_contact1 = dis_maxtrix1 < CONTACT_CUTOFF
            gt_contact2 = dis_maxtrix2 < CONTACT_CUTOFF

            # contact in common
            state_common = np.logical_and(gt_contact1, gt_contact2)
            # contact in state1 or state2
            state_conversion = np.logical_xor(gt_contact1, gt_contact2)
            state1 = np.logical_and(state_conversion, gt_contact1)
            state2 = np.logical_and(state_conversion, gt_contact2)

            print('range_, n_common, n_if, n_of:')

            n_contact_ranges = []
            ranges = ['Long_range', 'Medium_range', 'Short_range','All_range']
            for range_ in ranges:
                range_filter= get_range_filter(real_resid1, real_resid2, range_)
                filtered_state_common = np.logical_and(state_common, range_filter)
                filtered_state1 = np.logical_and(state1, range_filter)
                filtered_state2 = np.logical_and(state2, range_filter)

                n_common = np.sum(filtered_state_common)
                n_state1 = np.sum(filtered_state1)
                n_state2 = np.sum(filtered_state2)

                #n_if = [n_state1, n_state2][order_[0]]
                #n_of = [n_state1, n_state2][order_[1]]
                n_if = n_state1
                n_of = n_state2

                print(range_, n_common, n_if, n_of)
                n_contact_ranges.append(f'{n_common}/{n_if}/{n_of}')

            n_contacts.append(n_contact_ranges)

        n_contacts = np.array(n_contacts)

        d = {'n_contacts_long': n_contacts[:, 0],
             'n_contacts_medium': n_contacts[:, 1],
             'n_contacts_short': n_contacts[:, 2],
             'n_contacts_all': n_contacts[:, 3],
             'the number of residues': n_res}
        df = pd.DataFrame(data=d)
        df_path = '../output/number_N_L.csv'
        print(df_path)
        df.to_csv(df_path, index=False)

    if args.type_flag:
        seq_dises = []
        p_short_pros = []
        p_medium_pros = []
        p_long_pros = []
        p_all_pros = []

        result_path = '../output/seq_dis_' + args.type + '.npy'
        num_contacts_over_L_path = '../output/num_contacts_over_L.csv'
        if not os.path.exists(result_path) or not os.path.exists(num_contacts_over_L_path) or args.force:
            with open('../output/tm_common_res_dis.pickle', 'rb') as handle:
                dic_distance = pickle.load(handle)
            for pair in dic_distance:
                if pair == "7BH2_B+5MRW_F": continue
                print(pair)
                pair_data = {}
                #if pair == "7E7I_A+7LKZ_A": continue
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
                #print("resid1:", resid1)
                #print("resid2:", resid2)

                gt_contact1 = dis_maxtrix1 < CONTACT_CUTOFF
                gt_contact2 = dis_maxtrix2 < CONTACT_CUTOFF

                seq_dis = []

                if args.type == 'state':

                    # contact in state1 or state2
                    state_conversion = np.logical_xor(gt_contact1, gt_contact2)

                    state1 = np.logical_and(state_conversion, gt_contact1)
                    state2 = np.logical_and(state_conversion, gt_contact2)
                    state1 = np.transpose(np.nonzero(state1))
                    state2 = np.transpose(np.nonzero(state2))

                    for index_pair in state1:
                        dis1 = abs(resid1[index_pair[0]] - resid1[index_pair[1]])
                        if dis1 >= 6 :
                            seq_dis.append(dis1)

                    for index_pair in state2:
                        dis2 = abs(resid2[index_pair[0]] - resid2[index_pair[1]])
                        if dis2 >= 6 :
                            seq_dis.append(dis2)

                #print("No. of all-range contacts only in state1:", np.sum(state1))
                #print("No. of all-range contacts only in state2:", np.sum(state2))

                # calculate sequence distance for state-converted contacts.
                # state_converted_index = np.transpose(np.nonzero(state_conversion))

                # calculate sequence distance for common contacts.

                if args.type == 'common':
                    common = np.logical_and(gt_contact1, gt_contact2)
                    common = np.transpose(np.nonzero(common))

                    for index_pair in common:
                        dis1 = abs(resid1[index_pair[0]] - resid1[index_pair[1]])
                        if dis1 >= 6 :
                            seq_dis.append(dis1)

                        dis2 = abs(resid2[index_pair[0]] - resid2[index_pair[1]])
                        if dis2 >= 6 :
                            seq_dis.append(dis2)

                seq_dis = np.array(seq_dis)
                seq_dises.extend(seq_dis)
                
                L = len(resid1)
                p_short_pro = ((seq_dis >= 6) & ( seq_dis <= 11)).sum() / L / 2
                p_medium_pro = ((seq_dis >= 12) & ( seq_dis <= 23)).sum() / L / 2
                p_long_pro = (seq_dis >= 24).sum() / L
                p_all_pro = p_short_pro + p_medium_pro + p_long_pro

                p_short_pros.append(p_short_pro)
                p_medium_pros.append(p_medium_pro)
                p_long_pros.append(p_long_pro)
                p_all_pros.append(p_all_pro)

                #break

            seq_dises = np.array(seq_dises).flatten()
            p_short_pros = np.array(p_short_pros)
            p_medium_pros = np.array(p_medium_pros)
            p_long_pros = np.array(p_long_pros)
            p_all_pros = np.array(p_all_pros)

            with open(result_path, 'wb') as f:
                np.save(f, seq_dises)

            df = pd.DataFrame()
            df['short'] = p_short_pros
            df['medium'] = p_medium_pros
            df['long'] = p_long_pros
            df['all'] = p_all_pros
            df.to_csv(num_contacts_over_L_path,index=False)

        with open(result_path, 'rb') as f:
            seq_dises = np.load(f, allow_pickle=True)

        df = pd.read_csv(num_contacts_over_L_path)
        print(df.mean(axis=0))

        seq_dises = np.array(seq_dises)
        # number of short-contact, medium-contact, long-range contact
        n_short = ((seq_dises >= 6) & ( seq_dises <= 11)).sum()
        n_medium = ((seq_dises >= 12) & ( seq_dises <= 23)).sum()
        n_long = (seq_dises >= 24).sum()

        print("No. of contact:", len(seq_dises))
        print("No. of short-range contacts:", n_short)
        print("No. of medium-range contacts:", n_medium)
        print("No. of long-range contacts:", n_long)
        
        fig, ax = plt.subplots()
        ax.hist(seq_dises, bins=100)
        #ax.hist(seq_dises, bins=[6,12,24,700])


        #ax.set_ylabel()
        fig.tight_layout()
        path = "../output/seq_dis_profile.png"
        print("Save to../output/seq_dis_profile.png")
        plt.savefig(path, dpi=600)
        plt.clf()
        plt.close()
