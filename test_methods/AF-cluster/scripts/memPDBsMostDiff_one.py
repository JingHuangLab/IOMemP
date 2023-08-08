
"""
find most different pdbs from similiar structures
ml gcc/gcc-5.5.0
"""

import os
import sys
import gzip
import glob
import pymol
import time
import timeit
import random
import argparse
import numpy as np
import pandas as pd

from itertools import compress
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO
from multiprocessing import Pool
from Bio.PDB.Polypeptide import three_to_one

MIN_RES=80
NRESMEM=60
TMALIGN='/home/xiety/software/TMalign/TMalign'
PDBDir='/u/xiety/database/pdb/data/biounit/PDB/all'
CIFDir='/u/xiety/database/pdb/data/biounit/mmCIF/all'
AROUND_MEM=15 # 15+5
MAX_COUNT = 20000
STANDARD_RES = ['R','H','K','D','E','S','T','N','Q','C','U','G','P','A','V','I','L','M','F','Y','W']

TMPDir='/tmp/xiety'
if not os.path.exists(TMPDir):
    os.mkdir(TMPDir)
    
def format_pdb(pdb_id, chain_id):
    pdb_files = glob.glob(str(os.path.join(PDBDir, pdb_id+'*')))
    cif_files = glob.glob(str(os.path.join(CIFDir, pdb_id+'*')))

    new_file_pdb = os.path.join(TMPDir, pdb_id + chain_id + '.pdb')

    if len(pdb_files) ==0 and len(cif_files) == 0:
        print(pdb_id + ' do not have cif or pdb')
        raise ValueError()

    
    if len(pdb_files) > 0:
        for pdb_file in pdb_files:
            pymol.cmd.load(pdb_file, pdb_id)
            chain_ids = pymol.cmd.get_chains(pdb_id)
            if chain_id in chain_ids:
                pymol.cmd.select(pdb_id + ' and chain ' + chain_id)

                pymol.cmd.save(new_file_pdb, 'sele')
                pymol.cmd.delete(pdb_id)

                with open(new_file_pdb,'r') as f:
                    n_lines = len(f.readlines())
                if n_lines == 1: # in case pymol has deceptive chain
                    os.remove(new_file_pdb)
                    continue
                else:
                    break
            else:
                pymol.cmd.delete(pdb_id)
                continue
            
    if os.path.exists(new_file_pdb): return new_file_pdb

    if len(cif_files) > 0:
        for cif_file in cif_files:
            pymol.cmd.load(cif_file, pdb_id)
            pymol.cmd.select(pdb_id + ' and chain ' + chain_id)

            try:
                pymol.cmd.save(new_file_pdb, 'sele')
            except:
                pymol.cmd.delete(pdb_id)
                continue
    if os.path.exists(new_file_pdb): 
        return new_file_pdb
    else: 
        raise ValueError()


def count_res(model):
    count = 0
    for chain in model:
        for residue in chain:
            try:
                z = residue['CA'].get_vector()[-1]
                if z >= -15 and z <= 15:
                    count = count + 1
            except:
                continue

    return count
def mem_confirm(pdb):
    pdb_id = pdb[:4].lower()
    chain_id = pdb[5:]
    pdb_file = os.path.join(TMPDir, pdb_id + chain_id + ".pdb")
    cif_file = os.path.join(TMPDir, pdb_id + chain_id + ".cif")

    format_file = None
    if (not os.path.exists(pdb_file)) and \
       (not os.path.exists(cif_file)):
        try:
            format_file = format_pdb(pdb_id, chain_id)
        except:
            print("chain extract failed for " + pdb)
            return False

    model = None
    
    if os.path.exists(pdb_file):
        pdb_mem_file = str(pdb_file)[:-4] + '_EMBED.pdb'

        if not os.path.exists(pdb_mem_file):
            os.system('memembed ' + str(pdb_file) + '> /dev/null 2>&1')
        if not os.path.exists(pdb_mem_file):
            print('memembed failed on ' + pdb)
            return False
        parser = PDBParser(QUIET=True)
        model = parser.get_structure(pdb, pdb_mem_file)[0]
    else:
        print('memembed failed on' + pdb + ' or pdb file not generated')
        return False

    n_res_mem = count_res(model)

    if n_res_mem > NRESMEM:
        #print(pdb + ' perhaps is a mem prot. we need.')
        return True
    else:
        #print(pdb + ' is not a mem prot. we need.')
        return False

def scan_pdb4mem(pdb4seq):
    mem_checks = []
    for pdb in pdb4seq:
        mem_check = mem_confirm(pdb)
        mem_checks.append(mem_check)

    return mem_checks

def mem_pdb_file(pdb_file, cif=False, renumber=False):
    if cif is True:
        # try to convert cif to pdb
        pymol.cmd.load(str(pdb_file), "pdb_id")
        pdb_file = str(pdb_file)[:-4] + '.pdb'
        pymol.cmd.save(str(pdb_file))
        pymol.cmd.delete("pdb_id")

    if renumber:
        os.system("python /home/xiety/minitools/clean_pdb/clean_pdb.py " + pdb_file + " ignorechain")
        pdb_file = str(pdb_file)[:-4] + '_ignorechain.pdb'
        print(pdb_file)

    pdb_mem_file = str(pdb_file)[:-4] + '_EMBED.pdb'

    if not os.path.exists(pdb_mem_file):
        os.system('memembed ' + str(pdb_file) + '> /dev/null 2>&1')
    if not os.path.exists(pdb_mem_file):
        print('memembed failed on ' + str(pdb_file))
        return None

    pdb_mem_file_aligned = str(pdb_file)[:-4] + '_EMBED_align.pdb'
    pymol.cmd.load(str(pdb_file), "pdb_id")
    pymol.cmd.load(str(pdb_mem_file), "pdb_id_mem")
    pymol.cmd.align("pdb_id", "pdb_id_mem")
    pymol.cmd.save(str(pdb_mem_file_aligned), "pdb_id")
    pymol.cmd.delete("pdb_id")
    pymol.cmd.delete("pdb_id_mem")

    parser = PDBParser(QUIET=True)
    try:
        model = parser.get_structure("mem_pdb", pdb_mem_file_aligned)
        return model
    except:
        print("Failed parsing " + pdb_mem_file_aligned)
        return None

def chain_pymol(structure, chain_id): # in case pymol save pdb only the first letter of chain id.
    model = None
    try:
        model = structure[0]
    except:
        return None
    if model.has_id(chain_id): return model[chain_id]
    if model.has_id(chain_id[0]): return model[chain_id[0]]
    return None
def get_chain(pdb, in_mem, renumber=False):
    pdb_id = pdb[:4].lower()
    chain_id = pdb[5:]
    pdb_file = os.path.join(TMPDir, pdb_id + chain_id + ".pdb")
    cif_file = os.path.join(TMPDir, pdb_id + chain_id + ".cif")

    format_file = None
    if (not os.path.exists(pdb_file)) and \
       (not os.path.exists(cif_file)):
        try:
            format_file = format_pdb(pdb_id, chain_id)
        except:
            print("chain extract failed for " + pdb)
            return None
    
    if os.path.exists(pdb_file):
        if in_mem:
            structure = mem_pdb_file(pdb_file, renumber=renumber)
            if structure is None:
                return None
            else:
                return chain_pymol(structure, chain_id)

        else:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb, pdb_file)
            return chain_pymol(structure, chain_id)
    if os.path.exists(cif_file):
        if in_mem:
            structure = mem_pdb_file(cif_file, cif=True)
            if structure is None:
                return None
            else:
                try:
                    chain_ = structure[0][chain_id]
                    return chain_
                except:
                    return None
        else:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(pdb, cif_file)
            return structure[0][chain_id]

def TMalign4seqAlign(struct1, struct2):
    '''
    To get sequence alignment after structural alignment
    '''
    pdb_file1 = os.path.join(TMPDir, struct1[:4].lower() + struct1[5:]+ ".pdb")
    pdb_file2 = os.path.join(TMPDir, struct2[:4].lower() + struct2[5:]+ ".pdb")
    grep_string = '"denotes residue pairs of"'
    result = os.popen(TMALIGN + " -byresi 0 -a T  " + pdb_file1 + ' ' + pdb_file2 +
                      '|grep ' + grep_string + " -A 3|tail -n 3")
    aligned_seqs = result.read().split('\n')[:3]
    return aligned_seqs[0],aligned_seqs[1],aligned_seqs[2]

def res_index(reses, res_ids, res, slide_index_initial):
    '''
    find the first hit residue in chain starting from a index
    return hit_index
    '''
    slide_index = slide_index_initial
    while 1:
        ifexist = False
        if slide_index in res_ids:
            ifexist = True

        if ifexist:
            list_index = res_ids.index(slide_index)
            hit_res = reses[list_index]
            res_name = hit_res.resname

            try:
                res_name = three_to_one(res_name)
            except:
                return hit_res, slide_index + 1 # not standard res.
            if res_name == res:
                return hit_res, slide_index + 1

        slide_index = slide_index + 1
        
    if slide_index > MAX_COUNT: return np.nan, slide_index_initial

def get_index(seq, chain):
    """
    map aligned seq (one-letter and -) to the chain
    """
    # get first res index
    reses = []
    res_ids = []
    for res in chain:
        reses.append(res)
        res_ids.append(res.id[1])

    slide_index = res_ids[0]

    align_reses = []
    for res in seq:
        if res in STANDARD_RES: # A residue
            #print("reses:",reses)
            #print("res_ids:", res_ids)
            #print("res:",res)
            #print("slide_index:", slide_index)
            #sys.stdout.flush()
            align_res, slide_index  = res_index(reses, res_ids, res, slide_index)
            align_reses.append(align_res)
        else:
            align_reses.append(-1)
    return align_reses

def catch_common_res(struct1, struct2, in_mem = True):
    chain1 = get_chain(struct1, in_mem)
    chain2 = get_chain(struct2, in_mem)
    
    if chain1 is None or chain2 is None:
        return -1, -1

    #id1 = -1
    #for i in chain1.get_residues():
    #    id1 = i.get_id()[1]
    #id2 = -1
    #for i in chain1.get_residues():
    #    id2 = i.get_id()[1]
    #max_res = max(id1,id2)
    #common_id = []
    #res1 = []
    #res2 = []
    #for i in range(1, max_res+1):
    #    if chain1.has_id(i) and chain2.has_id(i):
    #        res1_ = chain1[i]
    #        res2_ = chain2[i]

    #        if not (res1_.has_id('CA') and res2_.has_id('CA')):
    #            continue

    #        if in_mem:
    #            z1 = res1_['CA'].get_vector()[-1]
    #            z2 = res2_['CA'].get_vector()[-1]
    #            if (z1 > AROUND_MEM or z1 < -AROUND_MEM ) and \
    #                (z2 > AROUND_MEM or z2 < -AROUND_MEM):
    #                continue # skip the ith residue

    #        common_id.append(i)
    #    
    #        res1.append(chain1[i])
    #        res2.append(chain2[i])
    seq1, align_symble, seq2 = TMalign4seqAlign(struct1, struct2)
    res1 = []
    res2 = []
    reses1 = get_index(seq1, chain1)
    reses2 = get_index(seq2, chain2)

    print("struct1: ", struct1) 
    print("struct2: ", struct2)
    print("align_symble: ", align_symble)
    print("seq1: ", seq1)
    print("seq2: ", seq2)
    sys.stdout.flush()

    hit_res_index1 = []
    hit_res_index2 = []
    align_len = len(align_symble)
    for i in range(align_len):
        symble = align_symble[i]
        if reses1[i] == -1 or reses2[i] == -1: continue
        if symble != ' ': # res1 and res2 are aligend
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

def intra_dis(res_, chain_='A', file_=None):
    if file_ is not None:
        parser = PDBParser(QUIET=True)
        chain = parser.get_structure('test', file_)[0][chain_]

        res = []
        for i in res_:
            res.append(chain[" ", i, " "])
    else:
        res = res_

    n = len(res)
    dis_matrix = np.zeros([n, n]) 
    for i in range(n-1):
        for j in range(i, n):
            if res[i].get_resname() == 'GLY':
                atom1 = res[i]['CA']
            else:
                atom1 = res[i]['CB']
            if res[j].get_resname() == 'GLY':
                atom2 = res[j]['CA']
            else:
                atom2 = res[j]['CB']

            dis_matrix[i,j] = atom1 -atom2
            dis_matrix[j,i] = dis_matrix[i,j]
    return dis_matrix
    
def inter_dis(structs):
    begin = time.time()
    struct1 = structs[0]
    struct2 = structs[1]
    if struct1 == "2RCR_M" or struct2 == "2RCR_M": return -1
    res1, res2 = catch_common_res(struct1, struct2)
    if res1 == -1: return -1
    intra_dis1 = intra_dis(res1)
    intra_dis2 = intra_dis(res2)

    inter_dis_ = np.mean(abs(intra_dis1 - intra_dis2))
    finish = time.time()
    duration = int(finish) - int(begin)
    print("distance: " + str(inter_dis_))
    print("time: " + str(duration))
    sys.stdout.flush()
    return inter_dis_

def most_diff(pdbs4seq, npool):
    # find most different two structures
    n_struct = len(pdbs4seq)
    structs = [None, None]
    most_dis = -1

    pair_pdbs = []
    for i in range(n_struct-1):
        for j in range(i+1, n_struct):
            pair_pdbs.append([pdbs4seq[i], pdbs4seq[j]])

    print(str(len(pair_pdbs)) + " pdb pairs ")
    inter_dis_ = None
    with Pool(npool) as pool:
        inter_dis_ = pool.map(inter_dis, pair_pdbs)

    most_dis = np.nanmax(inter_dis_)
    most_index = inter_dis_.index(most_dis)
    structs = pair_pdbs[most_index]

    return structs, most_dis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', \
                        help='pdb list with the almost similiar structures')
    parser.add_argument('--npool', default=6, type=int, \
                        help='number of cpus for multiple processing')
    parser.add_argument('--filepath',  help='file path of result')

    args = parser.parse_args()
    pdbs = args.pdbs.split()
    file_ = args.filepath
    print("pdbs: ", pdbs)

    mdiff_structs, most_dis = most_diff(pdbs, args.npool)
    with open(file_, "w") as f:
        f.write(mdiff_structs[0] + ' ' + mdiff_structs[1] + ' ' + str(most_dis) + '\n')
    print("Result was wrote to " + file_)

    # if empty
    file_size = os.path.getsize(file_)
    if file_size == 0:
        os.remove(file_)
        print(file_ + " is empty and removed")

