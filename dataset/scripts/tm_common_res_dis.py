#/usr/bin/python3

"""
Calculate the distance of residues in TM regions for all 
proteins with alternative states
"""

import numpy as np
import argparse
import pickle
from memPDBsMostDiff_one import *

def pymol_align(struct1, struct2):
    pdb_file1 = os.path.join(TMPDir, struct1[:4].lower() + struct1[5:]+ ".pdb")
    pdb_file2 = os.path.join(TMPDir, struct2[:4].lower() + struct2[5:]+ ".pdb")

    pymol.cmd.load(pdb_file1, struct1)
    pymol.cmd.load(pdb_file2, struct2)
    pymol.cmd.align(struct1, struct2, object=struct1+'_'+struct2)
    pymol.cmd.save(TMPDir+'/'+struct1+'_'+struct2+'.aln', struct1+'_'+struct2)

    f = open(TMPDir+'/'+struct1+'_'+struct2+'.aln', 'r')
    lines = f.readlines()
    repeat = int((len(lines)-2)/4)
    print("repeat:",repeat)
    seq1 = ''.join([lines[i*4+2][13:75] for i in range(repeat)])
    seq2 = ''.join([lines[i*4+3][13:75] for i in range(repeat)])
    symble = ''.join([lines[i*4+4][13:75] for i in range(repeat)])

    #print(seq1)
    #print(seq2)
    #print(symble)
    #print("-----------")
    #sys.stdout.flush()

    f.close()

    return seq1, symble, seq2

def pymol_common_res(struct1, struct2, in_mem = True):
    """
    get common residues by align command in pymol
    """
    chain1 = get_chain(struct1, in_mem, renumber=True)
    chain2 = get_chain(struct2, in_mem, renumber=True)
    
    if chain1 is None or chain2 is None:
        return -1, -1

    seq1, align_symble, seq2 = pymol_align(struct1, struct2)
    res1 = []
    res2 = []
    reses1 = get_index(seq1, chain1)
    reses2 = get_index(seq2, chain2)

    print("struct1: ", struct1) 
    print("struct2: ", struct2)
    print("align_symble:\n", align_symble)
    print("seq1:\n", seq1)
    print("seq2:\n", seq2)
    sys.stdout.flush()

    hit_res_index1 = []
    hit_res_index2 = []
    align_len = len(align_symble)
    for i in range(align_len):
        symble = align_symble[i]
        if reses1[i] == -1 or reses2[i] == -1: continue
        if symble == '*' or symble == ".": # res1 and res2 are aligend
            res1_ = reses1[i] 
            res2_ = reses2[i]

            if in_mem:
                z1 = res1_['CA'].get_vector()[-1]
                z2 = res2_['CA'].get_vector()[-1]
                if (z1 > AROUND_MEM or z1 < -AROUND_MEM ) and \
                    (z2 > AROUND_MEM or z2 < -AROUND_MEM):
                    continue # skip the ith residue

            res1.append(res1_)
            res2.append(res2_)

            hit_res_index1.append(res1_.id[1])
            hit_res_index2.append(res2_.id[1])

    print("Common residues' index:")
    print(hit_res_index1)
    print(hit_res_index2)
    sys.stdout.flush()
    
    # check common res in mem
    if in_mem:
        if len(res1) < MIN_RES: # Exclude those pair has limited common residues in membrane.
            print("Number of common residues is " + str(len(res1)) + ", less than " + str(MIN_RES) + ", so the pair is excluded.")
            return -1, -1
        else:
            print("Number of common residues is " + str(len(res1)) + ", larger than " + str(MIN_RES) + ", so the pair is included.")
            
    return res1, res2


def pair_structs_dis(pair):
    struct1 = pair[0]
    struct2 = pair[1]
    res1, res2 = pymol_common_res(struct1, struct2)
    if res1 == -1: return -1, -1, -1, -1

    intra_dis1 = intra_dis(res1)
    intra_dis2 = intra_dis(res2)

    resid1 = [r.id[1] for r in res1]
    resid2 = [r.id[1] for r in res2]
    return intra_dis1, intra_dis2, resid1, resid2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proteins', default="../output/mostdiff_files_seq_cluster70_manu_sel.txt", help='a file listing proteins with a pair of structures') 
    parser.add_argument('--dir', help='directory of proteins')
    args = parser.parse_args()

    pros = np.genfromtxt(args.proteins, dtype=str)
    dis = {}
    for pair in pros:
        dis1, dis2, index1, index2 = pair_structs_dis(pair)
        if type(dis1) == 'int': print(pair, "fail in distance matrix")
        info1 = {'distance matrix': dis1, 'index': index1}
        info2 = {'distance matrix': dis2, 'index': index2}
        dis[pair[0]+'+'+pair[1]] = {pair[0]:info1, pair[1]:info2}
        #break

    with open('../output/tm_common_res_dis.pickle', 'wb') as handle:
        pickle.dump(dis, handle, protocol=2)
    print('Save to ../output/tm_common_res_dis.pickle')
