#!/usr/bin/python

"""
Calculate deltaD of the output structures of AF2 relative to the inward/outward-open structure from PDB.
"""
import os
import numpy as np
import pandas as pd
import argparse
import pickle
import glob
from memPDBsMostDiff_one import *

def model_deltaD(pdb_f, pro_resid, d_in, d_out):
    """
    Caculate the deltaD of model to in/outward-open
    """
    d_pre = intra_dis(pro_resid, file_=pdb_f)

    d_pre_in = np.mean(abs(d_pre - d_in))
    d_pre_out = np.mean(abs(d_pre - d_out))

    return d_pre_in, d_pre_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='../output/eva_results_from_reduce_MSA_depth', help='directory of contacts')
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

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
        # if pair == "6XMU_A+6XMS_A": continue

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
            # if pro != '6S3Q_A': continue
            print(pro)
            pro_alias = pro[:4].lower() + pro[-1]
            pro_resid = np.asarray(dic_distance[pair][pro]['index'])

            d_in_out_path = os.path.join(args.dir, 'results', 'results_'+pro_alias+'_deltaD.csv')
            df = pd.DataFrame()
            
            pre_root = os.path.join(args.dir, 'pdbs')

            if not os.path.exists(d_in_out_path) or args.force:
                d_pre_ins = []
                d_pre_outs = []
                saved_pres = []

                for pdb_f in glob.glob(pre_root + '/' + pro_alias + "*.pdb"):
                    # print(pdb_f)
                    saved_pre = os.path.basename(pdb_f)[6:] 
                    
                    # if saved_pre == 'na.pdb':# skip the native here
                    #     continue 
                    # else:
                    #     saved_pre = os.path.basename(pdb_f)[6:-4]+".cif"
                    
                    d_pre_in, d_pre_out = model_deltaD(pdb_f, pro_resid, d_in, d_out)
                    
                    d_pre_ins.append(d_pre_in)
                    d_pre_outs.append(d_pre_out)
                    
                    saved_pres.append(saved_pre)
                                      
                df["d_in"] = d_pre_ins
                df["d_out"] = d_pre_outs
                df["decoy_id"] = saved_pres
                
                df.to_csv(d_in_out_path, index=False)
            
            #break
        #break
