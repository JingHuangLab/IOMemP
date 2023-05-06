
"""
Among memPDBs with multiple structures, find out those different a lot
ml gcc/gcc-5.5.0
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import memPDBsMostDiff_one

NCPU = 1
PARTITION='yuhuan'

def submit2cluster(pdb4seq_, file_):
    """
    submit to cluster for most diff structures
    """
    pdb_list = pdb4seq_.split()
    n_proc = min (NCPU, int(len(pdb_list) * (len(pdb_list) - 1) / 2 / 100) + 1)
    print("Rep pdb " + str(pdb_list[0]))
    
    log = "../log/" + pdb_list[0] + ".log"
    err = "../log/" + pdb_list[0] + ".err"

    if args.runlocal:
        com = "./memPDBsMostDiff_one.sh \"{pdb4seq_}\" {n_proc} {file_path}" 
    else:
        com = "sbatch --job-name={rep_pdb} --nodes=1 --cpus-per-task={n_proc} --ntasks=1 --partition={PARTITION} --output={log} --error={err} memPDBsMostDiff_one.sh \"{pdb4seq_}\" {n_proc} {file_path}" 

    #print(com.format(rep_pdb=pdb_list[0], n_proc=n_proc, log=log, err=err, pdb4seq_=pdb4seq_[:-1], file_path=file_))
    os.system(com.format(rep_pdb=pdb_list[0], n_proc=n_proc, log=log, err=err, pdb4seq_=pdb4seq_, file_path=file_, PARTITION=PARTITION))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mem', action='store_true', \
                        help='memembed to filter non-mem seq')
    parser.add_argument('--seq_cluster_mem_TOPC', default='../output/mem_seq_mulpdbs', \
                        help='file of mem. and clustered seq by TOPCONS')
    parser.add_argument('--mdiff', action='store_true', \
                        help='most diff structure for mem seq')
    parser.add_argument('--npool', default=1, type=int, \
                        help='number of cpus for multiple processing')
    parser.add_argument('--dir4mostdiff', default='../output/mostdiff_files', \
                        help='dir. of result')
    parser.add_argument('--force', action='store_true', \
                        help='force to regenerate all data')
    parser.add_argument('--rank', action='store_true', help='rank for all')
    parser.add_argument('--suffix', default='_default', help='suffix of output file')
    parser.add_argument('--runlocal', action='store_true', help='run locally')
    parser.add_argument('--onecase', default=None, help='run only one case: pdb_id')

    args = parser.parse_args()
    if not os.path.exists(args.dir4mostdiff): os.makedirs(args.dir4mostdiff)

    pdb4seq = [] # representative pdb for each seq.
    with open(args.seq_cluster_mem_TOPC, 'r') as f:
        for line in f.readlines():
            pdb4seq.append(line.split()[0])

    if args.mem:
        mem_checks =  memPDBsMostDiff_one.scan_pdb4mem(pdb4seq)
        mem_rep_pdbs = args.seq_cluster_mem_TOPC + "_memembed"
        with open(mem_rep_pdbs, "w") as f:
            f.write("# memembed checking each pdb for all sequences, which have TM regions recognized by TOPCONS\n")
            for pdb, mem_check in zip(pdb4seq, mem_checks):
                if mem_check is True:
                    f.write(pdb + "\n")
    if args.mdiff:
        global mempdbs
        mem_rep_pdbs = args.seq_cluster_mem_TOPC + "_memembed"
        mem_pdbs = np.genfromtxt(mem_rep_pdbs, dtype=str) 

        pdb4seqs = [] 
        with open(args.seq_cluster_mem_TOPC, 'r') as f:
            for line in f.readlines():
                pdb4seqs.append(line)
        pdb4seqs = list(reversed(pdb4seqs))

        count = 0
        for pdb4seq_ in pdb4seqs:
            rep_pdb = pdb4seq_.split()[0]
            file_ = os.path.join(args.dir4mostdiff, rep_pdb + '.txt')

            if not args.runlocal:
                run_state = int(os.popen("squeue -n " + rep_pdb + "|wc -l").read()) -1 
                if run_state > 0 : continue
            if not os.path.exists(file_) or args.force:
                if args.onecase != None:
                    if rep_pdb != args.onecase: continue # only run the specified case
                if rep_pdb in mem_pdbs:
                    submit2cluster(pdb4seq_, file_)

                    count = count + 1
                    if count == -1:
                        break
            else:
                print("Result existed in " + file_)

    if args.rank:
        results = []
        # merge data of each kind of structure
        merge_file = '../output/mostdiff_files' + args.suffix + '.txt'
        os.system("for i in $(ls {datadir}/*); do c=`cat $i`;echo $c;done>{mergefile}".format(datadir=args.dir4mostdiff, mergefile=merge_file))
        with open(merge_file,'r') as f:
            for line in f:
                struct1, struct2, dis = line.split()[:3]
                result = {'struct1': str(struct1),
                          'struct2': str(struct2),
                          'distance': float(dis)}
                results.append(result)
                    
        df = pd.DataFrame()
        df = df.append(results, ignore_index=True)
        df = df.sort_values(by='distance', ascending=False)
        csv_file = '../output/mostdiff_files' + args.suffix + '_merge.csv'
        df.to_csv(csv_file, index = False)
        print("save to " + csv_file)

        fig, ax = plt.subplots()
        ax.hist(df['distance'], bins=50, range=(0,6))
        ax.set_xlabel("Differences of intra distances")
        ax.set_ylim(0,120)
        plt.savefig('../output/mostdiff_files' + args.suffix + '_merge.png', transparent=False)

