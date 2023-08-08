#/usr/bin/python3.7 
"""
Compare prediction methods by precision and coverage
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from matplotlib import rcParams

size = 7
rcParams.update({'font.size': size})
rcParams.update({'font.weight': 'bold'})
rcParams.update({'axes.titlesize': size})
rcParams.update({'axes.titleweight': 'bold'})
rcParams.update({'axes.labelweight': 'bold'})
rcParams.update({'axes.labelsize': size})

parser = argparse.ArgumentParser()
parser.add_argument('--methods', nargs='+', help='the names of methods')
parser.add_argument('--precision_all', action='store_true', help='calculate precision of all contacts')
parser.add_argument('--percentage_co_state_unique', action='store_true', help='calculate percentage')
parser.add_argument('--precision_co_state_unique', action='store_true', help='calculate precision')
parser.add_argument('--coverage_co_state_unique', action='store_true', help='calculate coverage')
parser.add_argument('--p_pre_unprefer', action='store_true', help='calculate coverage')
parser.add_argument('--c_pre_unprefer', action='store_true', help='calculate coverage')
parser.add_argument('--same_prefer_state_count', action='store_true', help='calculate coverage')
parser.add_argument('--num_methods_in_states', action='store_true', help='calculate coverage')
parser.add_argument('--bar_number', action='store_true', help='plot the number of bars')
parser.add_argument('--precision_corrcoef', action='store_true', help='calculate corrcoef')
parser.add_argument('--percentage_corrcoef', action='store_true', help='calculate percentage')
parser.add_argument('--coverage_corrcoef', action='store_true', help='calculate corrcoef')
parser.add_argument('--pre_all_prop_sh_state_unique', action='store_true', help='calculate corrcoef of the precision for all contacts over the proportion of shared-/state-unique- contacts')
parser.add_argument('--r_square', action='store_true', help='calculate the r square of p_co, p1, and p2 between two different MSAs for each method')


args = parser.parse_args()

ranges = ["Long_range", "Medium_range", "Short_range", "All_range"]
top_describe = ['Top L', 'Top L/2', 'Top L/5']

width=0.4
x = np.arange(len(args.methods))

data_methods = {}
print(args.methods)

# recognize in/out-ward
in_out_pairs = []
with open('../../dataset/output/mostdiff_files_seq_cluster70_manu_sel_in_out.txt') as handle:
    for line in handle.readlines():
        in_out_pair = line.split()[:2]
        in_out_pairs.append(in_out_pair[0].upper())
        in_out_pairs.append(in_out_pair[1].upper())

print(in_out_pairs)

# reoganized data structure

for m in args.methods:
    f_ = "../" + m + "/output/acc_on_states.pickle"
    print(f_)
    with open(f_, 'rb') as handle:
        all_data = pickle.load(handle, encoding='latin1')

    data_ranges = {}
    for range_ in ranges:
        data_tops = {}
        for top_n in top_describe:
            #plot a figure at this range and top 

            p_cos = []
            p1s = []
            p2s = []

            c_cos = []
            c12s = []
            c1s = []
            c2s = []

            pp_cos = []
            pp12s = []
            pp1s = []
            pp2s = []

            pros = []
            for pair, pair_data in all_data.items():
                try:
                    index_pro = in_out_pairs.index(pair.split('+')[0])
                except:
                    continue
                if index_pro % 2 == 0:
                    order_ = [0, 1]
                    order_12 = [1, 2, 5, 6, 9, 10]
                else:
                    order_ = [1, 0]
                    order_12 = [2, 1, 6, 5, 10, 9]

                pros = pair.split('+')
                pro_in = pros[order_[0]]
                pro_out = pros[order_[1]]
                

                for order_index in order_:
                    pro = pros[order_index]

                    try:
                        pro_data = pair_data[pro]
                    except:
                        continue

                    pros.append(pro)
                    #print(pro)


                    if m == "AlphaFold":
                        p_cos.append(pro_data[range_][top_n][0,0])
                        p1s.append(pro_data[range_][top_n][0,order_12[0]])
                        p2s.append(pro_data[range_][top_n][0,order_12[1]])
                        
                        c_cos.append(pro_data[range_][top_n][0,3])
                        c12s.append(pro_data[range_][top_n][0,4])
                        c1s.append(pro_data[range_][top_n][0,order_12[2]])
                        c2s.append(pro_data[range_][top_n][0,order_12[3]])

                        pp_cos.append(pro_data[range_][top_n][0,7])
                        pp12s.append(pro_data[range_][top_n][0,8])
                        pp1s.append(pro_data[range_][top_n][0,order_12[4]])
                        pp2s.append(pro_data[range_][top_n][0,order_12[5]])

                    else:
                        p_cos.append(pro_data[range_][top_n][0])
                        p1s.append(pro_data[range_][top_n][order_12[0]])
                        p2s.append(pro_data[range_][top_n][order_12[1]])
                        
                        c_cos.append(pro_data[range_][top_n][3])
                        c12s.append(pro_data[range_][top_n][4])
                        c1s.append(pro_data[range_][top_n][order_12[2]])
                        c2s.append(pro_data[range_][top_n][order_12[3]])

                        pp_cos.append(pro_data[range_][top_n][7])
                        pp12s.append(pro_data[range_][top_n][8])
                        pp1s.append(pro_data[range_][top_n][order_12[4]])
                        pp2s.append(pro_data[range_][top_n][order_12[5]])
                #break
            #if range_ == 'Long_range' and top_n == 'Top L':
            #    print(len(p1s))
            #    for i in range(0, 33, 2):
            #        #print(i)
            #        if p1s[i] - p2s[i] > 0:
            #            print("precision:", pros[i], " > ", pros[i+1])
            #        else:
            #            print("precision:", pros[i], " < ", pros[i+1])
                    #if p1s[i+1] - p2s[i+1] > 0:
                    #    print("precision:", pros[i], " > ", pros[i+1])
                    #else:
                    #    print("precision:", pros[i], " < ", pros[i+1])


            #print(range_,top_n)
            #print("mean of p_cos0:", np.mean(p_cos0))
            #print("std of p_cos0:", np.std(p_cos0))

            #p12 = [i+j for i, j in zip(p1s0, p2s0)]
            #print("mean of p12:", np.mean(p12))
            #print("std of p12:", np.std(p12))

            #p_favor = [max(i,j) for i, j in zip(p1s0, p2s0)]
            #print("mean of p_favor:", np.mean(p_favor))
            #print("std of p_favor:", np.std(p_favor))

            #p_unfavor = [min(i,j) for i, j in zip(p1s0, p2s0)]
            #print("mean of p_unfavor:", np.mean(p_unfavor))
            #print("std of p_unfavor:", np.std(p_unfavor))
            data_tops[top_n] = np.array([p_cos, p1s, p2s, c_cos, c12s, c1s, c2s, pp_cos, pp12s, pp1s, pp2s])
            #break # top n
        data_ranges[range_] = data_tops
        #break # ranges
    data_methods[m] = data_ranges
    #break # methods

#print(data_methods)

# analyze proportion of common and state-unique contacts
if args.precision_all:
    ranges = ['Long_range']
    width = 0.5
    for range_ in ranges:
        subplot_id = 0
        data_top = {}
        top_describe = ['Top L']
        for top_n in top_describe:
            fig, axs = plt.subplots(1, figsize=(2.9, 1.6))

            pp_all_means = []
            pp_sh_mean = []
            pp_sh_1_means = []
            pp_sh_2_means = []
            for m in args.methods:
                p_c = data_methods[m][range_][top_n] 
                pp_all_mean = np.mean(p_c[0,:]+p_c[1,:]+p_c[2,:])
                pp_all_means.append(pp_all_mean)

                pp_sh_mean.append(np.mean(p_c[0,:]))
                pp_sh_1_means.append(np.mean(p_c[0,:]+p_c[1,:]))
                pp_sh_2_means.append(np.mean(p_c[0,:]+p_c[2,:]))

            # save data to csv
            merge_array = np.array([pp_sh_mean, pp_sh_1_means, pp_sh_2_means, pp_all_means])
            np.savetxt("../output/ave_precision.csv", merge_array, fmt='%.3f', delimiter=',')
            print("../output/ave_precision.csv")
            # plot
            rect1 = axs.bar(x, pp_all_means, width)

            # bar label
            if args.bar_number:
                axs.bar_label(rect1, fmt="%.3f", size='3')

            axs.set_title("Precision", weight='bold')
            axs.set_ylabel('%s, %s'%(range_, top_n), weight='bold')
            axs.set_xticks([])
            axs.set_yticks(np.arange(0, 1.2, step=0.2))
            axs.set_ylim(0, 1)

            shorten_methods = []
            for m in args.methods:
                if m == 'RaptorX-Contact': m = 'RaptorX-C'
                if m == 'RaptorX-3DModeling': m = 'RaptorX-3D'
                shorten_methods.append(m)

            plt.xticks(x, shorten_methods, rotation="70", ha='right')

            if top_n == "Top L/2": top_n = "L2"
            if top_n == "Top L/5": top_n = "L5"

            if args.bar_number:
                path = "../output/precision"+range_+top_n+"_Number.png"
                print(path)
            else:
                path = "../output/precision"+range_+top_n+"_noNumber.png"
                print(path)

            plt.savefig(path, bbox_inches='tight', dpi=300)
            plt.savefig(path[:-4]+'.pdf', bbox_inches='tight')
            plt.clf()
            plt.close()
# analyze proportion of common and state-unique contacts
if args.percentage_co_state_unique:
    #ranges = ["Medium_range"]
    for range_ in ranges:
        subplot_id = 0
        data_top = {}
        # top_describe = ['Top L']
        for top_n in top_describe:
            fig, axs = plt.subplots(1, figsize=(2.9, 1.6))

            data_m = {}
            p_co_means = []
            p_co_stds = []
            
            p12_means = []
            p12_stds = []
            for m in args.methods:
                p_c = data_methods[m][range_][top_n] 

                p_co_mean = np.mean(p_c[0,:])
                if range_ == "Long_range" and top_n == "Top L":
                    print(m, " p_co_mean: ", p_co_mean)
                p_co_std = np.std(p_c[0,:])
                p_co_means.append(p_co_mean)
                p_co_stds.append(p_co_std)

                p12_mean = np.mean(p_c[1,:]+p_c[2,:])
                p12_std = np.std(p_c[1,:]+p_c[2,:])
                p12_means.append(p12_mean)
                p12_stds.append(p12_std)

            # rect1 = axs[subplot_id].bar(x - width/2, p_co_means, width, yerr=p_co_stds)
            # rect2 = axs[subplot_id].bar(x + width/2, p12_means, width, yerr=p12_stds)
            rect1 = axs.bar(x - width/2, p_co_means, width)
            rect2 = axs.bar(x + width/2, p12_means, width)

            p_co_means = np.asarray(p_co_means)
            p12_means = np.asarray(p12_means)
            # Pearson product-moment correlation coefficients
            R = np.corrcoef(p_co_means,p12_means)
            print(range_, top_n, "Coefficients:", R)
            R_non_DL = np.corrcoef(p_co_means[:5],p12_means[:5])
            print(range_, top_n, "Coefficients for non-DL:", R_non_DL)
            R_DL = np.corrcoef(p_co_means[[5, 6, 7, 8, 10, 11]],p12_means[[5, 6, 7, 8, 10, 11]])
            print(range_, top_n, "Coefficients for DL:", R_DL)


            # bar label
            if args.bar_number:
                axs.bar_label(rect1, fmt="%.3f", size='3')
                axs.bar_label(rect2, fmt="%.3f", size='3')

            axs.set_title("Percentage", weight='bold')
            axs.set_ylabel('%s, %s'%(range_, top_n), weight='bold')
            axs.set_xticks([])
            axs.set_yticks(np.arange(0, 1.2, step=0.2))
            axs.legend(loc="upper left", labels=["shared", "state-specific "], fontsize=5)
            #axs[subplot_id].set_ylim(-0.1, 1.1)
            subplot_id = subplot_id + 1

            shorten_methods = []
            for m in args.methods:
                if m == 'RaptorX-Contact': m = 'RaptorX-C'
                if m == 'RaptorX-3DModeling': m = 'RaptorX-3D'
                shorten_methods.append(m)

            plt.xticks(x, shorten_methods, rotation="70", ha='right')

            if top_n == "Top L/2": top_n = "L2"
            if top_n == "Top L/5": top_n = "L5"

            if args.bar_number:
                path = "../output/percentage"+range_+top_n+"_Number.png"
                print(path)
            else:
                path = "../output/percentage"+range_+top_n+"_noNumber.png"
                print(path)

            plt.savefig(path, bbox_inches='tight')
            plt.savefig(path[:-4]+'.pdf', bbox_inches='tight')
            plt.clf()
            plt.close()

# coverage for common and state-unique contacts
# analyze precision of common and state-unique contacts
if args.precision_co_state_unique:
    #ranges = ["Medium_range"]
    for range_ in ranges:
        subplot_id = 0
        data_top = {}
        # top_describe = ['Top L']
        for top_n in top_describe:
            fig, axs = plt.subplots(1, figsize=(2.9, 1.6))

            data_m = {}
            pp_co_means = []
            pp_co_stds = []
            
            pp12_means = []
            pp12_stds = []
            for m in args.methods:
                metrics = data_methods[m][range_][top_n] # 6 * No. of pros

                pp_co_mean = np.mean(metrics[7,:])
                pp_co_std = np.std(metrics[7,:])
                pp_co_means.append(pp_co_mean)
                pp_co_stds.append(pp_co_std)

                if range_ == "Long_range" and top_n == "Top L":
                    print(m, " pp_co_mean: ", pp_co_mean)

                pp12_mean = np.mean(metrics[8,:])
                pp12_std = np.std(metrics[8,:])
                pp12_means.append(pp12_mean)
                pp12_stds.append(pp12_std)

            # rect1 = axs[subplot_id].bar(x - width/2, p_co_means, width, yerr=p_co_stds)
            # rect2 = axs[subplot_id].bar(x + width/2, p12_means, width, yerr=p12_stds)
            rect1 = axs.bar(x - width/2, pp_co_means, width)
            rect2 = axs.bar(x + width/2, pp12_means, width)

            p_co_means = np.asarray(pp_co_means)
            p12_means = np.asarray(pp12_means)
            ## Pearson product-moment correlation coefficients
            #R = np.corrcoef(pp_co_means,pp12_means)
            #print(range_, top_n, "Coefficients:", R)
            #R_non_DL = np.corrcoef(pp_co_means[:5],pp12_means[:5])
            #print(range_, top_n, "Coefficients for non-DL:", R_non_DL)
            #R_DL = np.corrcoef(pp_co_means[[5, 6, 7, 8, 10, 11]],pp12_means[[5, 6, 7, 8, 10, 11]])
            #print(range_, top_n, "Coefficients for DL:", R_DL)


            # bar label
            if args.bar_number:
                axs.bar_label(rect1, fmt="%.3f", size='3')
                axs.bar_label(rect2, fmt="%.3f", size='3')

            axs.set_title("Precision", weight='bold')
            axs.set_ylabel('%s, %s'%(range_, top_n), weight='bold')
            axs.set_xticks([])
            axs.set_yticks(np.arange(0, 1.2, step=0.2))
            axs.legend(loc="upper left", labels=["shared", "state-specific "], fontsize=5)
            #axs[subplot_id].set_ylim(-0.1, 1.1)
            subplot_id = subplot_id + 1

            shorten_methods = []
            for m in args.methods:
                if m == 'RaptorX-Contact': m = 'RaptorX-C'
                if m == 'RaptorX-3DModeling': m = 'RaptorX-3D'
                shorten_methods.append(m)

            plt.xticks(x, shorten_methods, rotation="70", ha='right')

            if top_n == "Top L/2": top_n = "L2"
            if top_n == "Top L/5": top_n = "L5"

            if args.bar_number:
                path = "../output/precision"+range_+top_n+"_Number.png"
                print(path)
            else:
                path = "../output/precision"+range_+top_n+"_noNumber.png"
                print(path)

            plt.savefig(path, bbox_inches='tight')
            plt.savefig(path[:-4]+'.pdf', bbox_inches='tight')
            plt.clf()
            plt.close()

# coverage for common and state-unique contacts
if args.coverage_co_state_unique:
    #ranges = ["Medium_range"]
    for range_ in ranges:
        subplot_id = 0
        data_top = {}
        for top_n in top_describe:
            fig, axs = plt.subplots(1, figsize=(2.9, 1.6))
            data_m = {}
            c_co_means = []
            c_co_stds = []
            
            c12_means = []
            c12_stds = []
            for m in args.methods:
                metrics = data_methods[m][range_][top_n] # 6 * No. of pros

                c_co_mean = np.nanmean(metrics[3,:])
                c_co_std = np.nanstd(metrics[3,:])
                c_co_means.append(c_co_mean)
                c_co_stds.append(c_co_std)

                c12_mean = np.nanmean(metrics[4,:])
                c12_std = np.nanstd(metrics[4,:])
                c12_means.append(c12_mean)
                c12_stds.append(c12_std)

            
            print(c_co_means)
            print(c12_means)
            rect1 = axs.bar(x - width/2, c_co_means, width)
            rect2 = axs.bar(x + width/2, c12_means, width)
            # Pearson product-moment correlation coefficients
            c_co_means = np.asarray(c_co_means)
            c12_means = np.asarray(c12_means)
            R = np.corrcoef(c_co_means,c12_means)
            print(range_, top_n, "Coefficients:", R)
            R_non_DL = np.corrcoef(c_co_means[:5],c12_means[:5])
            print(range_, top_n, "Coefficients for non-DL:", R_non_DL)
            R_DL = np.corrcoef(c_co_means[[5, 6, 7, 8, 10, 11]],c12_means[[5, 6, 7, 8, 10, 11]])
            print(range_, top_n, "Coefficients for DL:", R_DL)

            # bar label
            if args.bar_number:
                axs.bar_label(rect1, fmt="%.3f", size='3')
                axs.bar_label(rect2, fmt="%.3f", size='3')

            axs.set_title("Coverage", weight='bold')
            axs.set_ylabel('%s, %s'%(range_, top_n), weight='bold')
            axs.set_xticks([])
            axs.set_yticks(np.arange(0, 1.2, step=0.2))
            axs.legend(loc="upper left", labels=["shared", "state-specific "], fontsize=5)
            #axs[subplot_id].set_ylim(-0.1, 1.1)
            subplot_id = subplot_id + 1

            shorten_methods = []
            for m in args.methods:
                if m == 'RaptorX-Contact': m = 'RaptorX-C'
                if m == 'RaptorX-3DModeling': m = 'RaptorX-3D'
                shorten_methods.append(m)

            plt.xticks(x, shorten_methods, rotation="70", ha='right')

            if top_n == "Top L/2": top_n = "L2"
            if top_n == "Top L/5": top_n = "L5"

            if args.bar_number:
                path = "../output/coverage"+range_+top_n+"_Number.png"
            else:
                path = "../output/coverage"+range_+top_n+"_noNumber.png"

            print(path)
            plt.savefig(path, bbox_inches='tight')
            plt.savefig(path[:-4]+'.pdf', bbox_inches='tight')
            plt.clf()
            plt.close()

# precision for prefered and unprefered state contacts
if args.p_pre_unprefer:
    width= 1 / 4
    top_describe = ['Top L']
    for range_ in ranges:
        fig, ax = plt.subplots(1, figsize=(2.9, 1.6))

        shorten_methods = []
        for m in args.methods:
            if m == 'RaptorX-Contact': m = 'RaptorX-C'
            if m == 'RaptorX-3DModeling': m = 'RaptorX-3D'
            shorten_methods.append(m)

        plt.xticks(x, shorten_methods, rotation="70", ha='right')
        #subplot_id = 0
        data_top = {}
        for top_n in top_describe:
            data_m = {}
            
            p1_means = []
            p1_stds = []

            p2_means = []
            p2_stds = []

            delta_means = []
            delta_stds = []
            for m in args.methods:
                p_c = data_methods[m][range_][top_n] # 6 * No. of pros

                raw_p1 = p_c[1,:] 
                raw_p2 = p_c[2,:]
                
                prefer_p = [max(i,j) for i, j in zip(raw_p1, raw_p2)]
                unprefer_p = [min(i,j) for i, j in zip(raw_p1, raw_p2)]
                deltas = []
                for i, j in zip(prefer_p, unprefer_p):
                    if i + j == 0:
                        continue
                    else:
                        delta = i/(i+j) 

                    deltas.append(delta)

                p1_mean = np.mean(prefer_p)
                p1_std = np.std(prefer_p)
                p1_means.append(p1_mean)
                p1_stds.append(p1_std)

                p2_mean = np.mean(unprefer_p)
                p2_std = np.std(unprefer_p)
                p2_means.append(p2_mean)
                p2_stds.append(p2_std)

                delta_mean = np.mean(deltas)
                delta_std = np.std(deltas)
                delta_means.append(delta_mean)
                delta_stds.append(delta_std)
            
            rect1 = ax.bar(x - width, p1_means, width, color='C4')
            rect2 = ax.bar(x , p2_means, width, color='C9')
            #rect1 = axs[subplot_id].bar(x - width, p1_means, width, yerr=p1_stds, color='C3')
            #rect2 = axs[subplot_id].bar(x , p2_means, width, yerr=p2_stds, color='C4')
            # bar label
            #axs[subplot_id].bar_label(rect1, fmt="%.3f", size='6')
            #axs[subplot_id].bar_label(rect2, fmt="%.3f", size='6')
            # ax_delta = ax.twinx()
            # ax_delta.set_ylabel(r"$d_{\mathrm{percentage}}$", weight='bold',color='C6')
            #rect3 = ax_delta.bar(x + width, delta_means, width, yerr=delta_std, color='C7')
            # rect3 = ax_delta.bar(x + width, delta_means, width, color='C6')
            
            # save delta_means
            delta_means = np.array(delta_means)
            np.savetxt("../output/degree_percentage"+range_+".csv", delta_means, fmt="%.3f")
            print("../output/degree_percentage"+range_+".csv")

            # bar label
            # if args.bar_number:
            #     ax_delta.bar_label(rect3,fmt="%.3f", size='6')
            # ax_delta.set_ylim(0.6,1)
            #ax_delta.set_xticks(x, args.methods, rotation="90")

            ax.set_ylabel('%s, %s'%(range_, top_n), weight='bold')
            ax.set_xticks([])
            ax.legend(loc="upper left", labels=["Preferable state", "Unpreferable state"])
            ax.set_ylim(0, 0.25)
            #subplot_id = subplot_id + 1

        plt.title("Percentage", weight='bold')
        plt.xticks(x, shorten_methods, rotation="70", ha='right')

        path = "../output/pre_unprefer_percentage"+range_+".png"
        if args.bar_number:
            path = "../output/pre_unprefer_percentage"+range_+"_number.png"
        plt.savefig(path, bbox_inches='tight')
        plt.savefig(path[:-4]+'.pdf', bbox_inches='tight')
        print(path)
        plt.clf()
        plt.close()

# coverage for prefered and unprefered state contacts
if args.c_pre_unprefer:
    width= 1 / 4
    top_describe = ['Top L']
    for range_ in ranges:
        fig, ax = plt.subplots(1, figsize=(2.9, 1.6))
        shorten_methods = []
        for m in args.methods:
            if m == 'RaptorX-Contact': m = 'RaptorX-C'
            if m == 'RaptorX-3DModeling': m = 'RaptorX-3D'
            shorten_methods.append(m)

        plt.xticks(x, shorten_methods, rotation="70", ha='right')
        #subplot_id = 0
        data_top = {}
        for top_n in top_describe:
            data_m = {}
            
            c1_means = []
            c1_stds = []

            c2_means = []
            c2_stds = []

            delta_means = []
            delta_stds = []
            for m in args.methods:
                p_c = data_methods[m][range_][top_n] # 6 * No. of pros

                raw_c1 = p_c[5,:] 
                raw_c2 = p_c[6,:]
                
                # prefer_p = p_c[1:2,:].max(axis=1)
                prefer_c = [max(i,j) for i, j in zip(raw_c1, raw_c2)]
                unprefer_c = [min(i,j) for i, j in zip(raw_c1, raw_c2)]
                deltas = []
                for i, j in zip(prefer_c, unprefer_c):
                    if i + j == 0:
                        continue
                    else:
                        delta = i / (i+j) 

                    deltas.append(delta)

                c1_mean = np.nanmean(prefer_c)
                c1_std = np.nanstd(prefer_c)
                c1_means.append(c1_mean)
                c1_stds.append(c1_std)

                c2_mean = np.nanmean(unprefer_c)
                c2_std = np.nanstd(unprefer_c)
                c2_means.append(c2_mean)
                c2_stds.append(c2_std)

                delta_mean = np.mean(deltas)
                delta_std = np.std(deltas)
                delta_means.append(delta_mean)
                delta_stds.append(delta_std)
            
            #rect1 = axs[subplot_id].bar(x - width, c1_means, width, yerr=c1_stds, color='#D3C3F7')
            #rect2 = axs[subplot_id].bar(x , c2_means, width, yerr=c2_stds, color='#98DAA7')
            rect1 = ax.bar(x - width, c1_means, width, color="C4")
            rect2 = ax.bar(x , c2_means, width, color="C9")
            # bar label
            #axs[subplot_id].bar_label(rect1, fmt="%.3f", size='6')
            #axs[subplot_id].bar_label(rect2, fmt="%.3f", size='6')
            # ax_delta = ax.twinx()
            # ax_delta.set_ylabel(r"$d_{\mathrm{coverage}}$", weight='bold',c='C6')
            #rect3 = ax_delta.bar(x + width, delta_means, width, yerr=delta_std, color='C7')
            # rect3 = ax_delta.bar(x + width, delta_means, width, color="C6")
            
            # save delta_means
            delta_means = np.array(delta_means)
            np.savetxt("../output/degree_coverage"+range_+".csv", delta_means, fmt="%.3f")
            print("../output/degree_coverage"+range_+".csv")

            # bar label
            # if args.bar_number:
            #     ax_delta.bar_label(rect3,fmt="%.3f", size='6')
            # ax_delta.set_ylim(0.6,1)
            #ax_delta.set_xticks(x, args.methods, rotation='vertical')

            ax.set_ylabel('%s, %s'%(range_, top_n), weight='bold')
            #ax.set_xticks([])
            ax.legend(loc="upper left", labels=["Preferable state", "Unpreferable state"])
            ax.set_ylim(0, 0.25)
            #subplot_id = subplot_id + 1

        plt.title('Coverage')
        plt.xticks(x, shorten_methods, rotation="70", ha='right')

        path = "../output/pre_unprefer_coverage"+range_+".png"
        if args.bar_number:
            path = "../output/pre_unprefer_coverage"+range_+"_number.png"
        plt.savefig(path, bbox_inches='tight')
        plt.savefig(path[:-4]+'.pdf', bbox_inches='tight')
        print(path)
        plt.clf()
        plt.close()


if args.same_prefer_state_count: # only long-range and Top L contacts

    count_list = []
    all_count_list = []
    for m in args.methods:
        f_ = "../" + m + "/output/acc_on_states.pickle"
        print(f_)
        with open(f_, 'rb') as handle:
            all_data = pickle.load(handle, encoding='latin1')

        data_ranges = {}
        range_ = 'Long_range'
        top_n = 'Top L'

        count = 0
        all_count = 0
        for pair, pair_data in all_data.items():
            if not pair_data:
                continue

            all_count = all_count + 1

            p1s = []
            p2s = []
            for pro, pro_data in pair_data.items():
                #print(pro)
                if m == "AlphaFold":
                    #p_cos.append(pro_data[range_][top_n][0,0])
                    p1s.append(pro_data[range_][top_n][0,1])
                    p2s.append(pro_data[range_][top_n][0,2])
                    
                    #c_cos.append(pro_data[range_][top_n][0,3])
                    #c1s.append(pro_data[range_][top_n][0,4])
                    #c2s.append(pro_data[range_][top_n][0,5])

                else:
                    #p_cos.append(pro_data[range_][top_n][0])
                    p1s.append(pro_data[range_][top_n][1])
                    p2s.append(pro_data[range_][top_n][2])


                    #c_cos.append(pro_data[range_][top_n][3])
                    #c1s.append(pro_data[range_][top_n][4])
                    #c2s.append(pro_data[range_][top_n][5])
            #print(p1s)
            #print(p2s)
            if (p1s[0] - p2s[0]) * (p1s[1] - p2s[1]):
                count = count + 1
            else:
                print(pair)
        count_list.append(count)
        all_count_list.append(all_count)

    df = pd.DataFrame()
    df['Methods'] = args.methods
    df['No. of proteins with the same preferable state'] = count_list
    df['No. of proteins with results'] = all_count_list
    df.to_csv("../output/number_of_pros_with_same_preferable_state.csv", index=False)

if args.num_methods_in_states: # only long-range and Top L contacts
    count_methods = {}

    for pro in in_out_pairs:
        count_methods[pro] = 0

    for m in args.methods:
        f_ = "../" + m + "/output/acc_on_states.pickle"
        print(f_)
        with open(f_, 'rb') as handle:
            all_data = pickle.load(handle, encoding='latin1')

        data_ranges = {}
        range_ = 'Long_range'
        top_n = 'Top L'

        for pair, pair_data in all_data.items():
            if not pair_data:
                continue

            p1s = []
            p2s = []
            flag = False
            
            if pair == "7BH2_B+5MRW_F": continue

            try:
                index_pro = in_out_pairs.index(pair.split('+')[0])
            except:
                continue

            if index_pro % 2 == 0:
                order_ = [0, 1]
                order4p = [1,2]
            else:
                order_ = [1, 0]
                order4p = [2,1]

            pros = pair.split('+')
            pro_in = pros[order_[0]]
            pro_out = pros[order_[1]]
            
            for order_index in order_:
                pro = pros[order_index]

                try:
                    pro_data = pair_data[pro]
                except:
                    flag = True
                    continue

                if m == "AlphaFold":
                    #p_cos.append(pro_data[range_][top_n][0,0])
                    p1s.append(pro_data[range_][top_n][0,order4p[0]])
                    p2s.append(pro_data[range_][top_n][0,order4p[1]])
                    
                    #c_cos.append(pro_data[range_][top_n][0,3])
                    #c1s.append(pro_data[range_][top_n][0,4])
                    #c2s.append(pro_data[range_][top_n][0,5])

                else:
                    #p_cos.append(pro_data[range_][top_n][0])
                    p1s.append(pro_data[range_][top_n][order4p[0]])
                    p2s.append(pro_data[range_][top_n][order4p[1]])


                    #c_cos.append(pro_data[range_][top_n][3])
                    #c1s.append(pro_data[range_][top_n][4])
                    #c2s.append(pro_data[range_][top_n][5])

        
            if flag: 
                continue

            print(p1s)
            if (p1s[0] - p2s[0]) * (p1s[1] - p2s[1]) > 0:
                if p1s[0] - p2s[0] >= 0:
                    count_methods[pro_in] = count_methods[pro_in] + 1
                    #if pair_pros[0] in ['6XMU_A','6QV1_B','6BVG_A','7BH2_B','4M64_A','7N98_A', '6RAJ_B']:
                    print(pro_in, m)
                else:
                    count_methods[pro_out] = count_methods[pro_out] + 1
                    #if pair_pros[1] in ['6HRC_D', '5AYM_A', '4C9J_B', '6S3Q_A', '6FHZ_A','6VS1_A','6E9O_B']:
                    print(pro_out, m)
            else:
                print(m, "have no preference on ", pair)

        #break

    print(count_methods)
    df = pd.DataFrame()
    df['pair_pros'] = list(count_methods.keys())
    df['num_methods'] = list(count_methods.values())
    df.to_csv("../output/number_methods_prefer_state.csv", index=False)
    print("../output/number_methods_prefer_state.csv")

    # split the number of hit methods for inward and outward
    fig, ax = plt.subplots(figsize=(5.9, 2.0))
    x = np.arange(len(count_methods.keys())/2)
    n_pair = int(len(count_methods.keys())/2)
    id_in = np.arange(0, n_pair*2, 2)
    id_out = np.arange(1, n_pair*2, 2)
    c_in = [list(count_methods.values())[i] for i in id_in]
    c_out = [list(count_methods.values())[i] for i in id_out]
    k_in = [list(count_methods.keys())[i] for i in id_in]
    k_out = [list(count_methods.keys())[i] for i in id_out]

    # plot stacked bar
    c_in = np.array(c_in)
    c_out = np.array(c_out)
    c_none = 11 - c_in - c_out
    print(c_in)
    print(c_out)
    print(c_none)

    bottom = np.zeros(n_pair)

    p = ax.bar(x, c_none, 0.6, color='lightgray', bottom=bottom)
    bottom += c_none
    labels = []
    for v in c_none:
        if v > 0:
            labels.append(str(v))
        else: 
            labels.append('')
    ax.bar_label(p, labels=labels, fmt="%g", size='7', label_type='center')

    p = ax.bar(x, c_in, 0.6, color='C1', label='IF', bottom=bottom)
    bottom += c_in
    labels = []
    for v in c_in:
        if v > 0:
            labels.append(str(v))
        else: 
            labels.append('')
    ax.bar_label(p, labels=labels, fmt="%g", size='7', label_type='center')

    p = ax.bar(x, c_out, 0.6, color='C2', label='OF', bottom=bottom)
    bottom += c_out
    labels = []
    for v in c_out:
        if v > 0:
            labels.append(str(v))
        else: 
            labels.append('')
    ax.bar_label(p, labels=labels, fmt="%g", size='7', label_type='center')

    colors = ['C1', 'C2'] * n_pair
    # rect_in = ax.bar(x-0.15, c_in, 0.3, color='C1', label='IF')
    # rect_out = ax.bar(x+0.15, c_out, 0.3, color='C2', label='OF')
    # 
    # ax.bar_label(rect_in, fmt="%g", size='8')
    # ax.bar_label(rect_out, fmt="%g", size='8')

    ax.set_xticks(np.sort(np.append(x-0.05, x+0.35)))
    ax.set_xticklabels(count_methods.keys(), rotation="70", ha='right')

    for xtick, color in zip(ax.get_xticklabels(), colors):
        xtick.set_color(color)

    ax.xaxis.set_ticks_position('none')

    ax.set_xlabel("Prefererable states",  weight='bold')
    ax.set_ylabel("No. of methods",  weight='bold')

    ax.set_ylim(0,13)
    ax.set_yticks(np.arange(2, 14, 2))
    fig.tight_layout()
    path = "../output/number_methods_prefer_state.png"
    print(path)
    plt.legend(fontsize=4, loc='upper left', ncol=2)
    plt.savefig(path, dpi=300)
    plt.savefig(path[:-4]+'.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()


def linear_regression(X, y):
    # linear regression


    # Create a linear regression model and fit it to the data
    model = LinearRegression()
    model.fit(X, y)

    # Calculate the R squared value for the model
    r2 = r2_score(y, model.predict(X))
    return model, r2

# analyze  the correlation of pp_co and pp12
if args.precision_corrcoef:
    size = 12
    rcParams.update({'font.size': size})
    rcParams.update({'font.weight': 'bold'})
    rcParams.update({'axes.titlesize': size})
    rcParams.update({'axes.titleweight': 'bold'})
    rcParams.update({'axes.labelweight': 'bold'})
    rcParams.update({'axes.labelsize': size})
    ranges = ['Long_range']

    for range_ in ranges:
        subplot_id = 0
        data_top = {}
        for top_n in top_describe:
            data_m = {}
            pp_co_means = []
            pp_co_stds = []
            
            pp12_means = []
            pp12_stds = []
            for m in args.methods:
                metrics = data_methods[m][range_][top_n] 

                pp_co_mean = np.mean(metrics[7,:])
                pp_co_std = np.std(metrics[7,:])
                pp_co_means.append(pp_co_mean)
                pp_co_stds.append(pp_co_std)

                if range_ == "Long_range" and top_n == "Top L":
                    print(m, " pp_co_mean: ", pp_co_mean)

                pp12_mean = np.mean(metrics[8,:])
                pp12_std = np.std(metrics[8,:])
                pp12_means.append(pp12_mean)
                pp12_stds.append(pp12_std)

            # scatter for p_co and p12
            if top_n == "Top L":
                path = "../output/precision_"+range_+"TopL_co_vs_p12.pdf"
                x = np.array(pp12_means)
                y = np.array(pp_co_means)

                xy = np.array([x, y])
                np.savetxt("../output/precision_"+range_+"TopL_co_vs_p12.csv", xy, fmt='%.4f', delimiter=',')
                print("../output/precision_"+range_+"TopL_co_vs_p12.csv")

                non_dl_x = x[:5] # MI mfDCA PSICOV CCMpred plmDCA
                non_dl_X = non_dl_x.reshape((-1, 1))
                non_dl_y = y[:5]
                nondl_model, nondl_r2 = linear_regression(non_dl_X, non_dl_y)
                print("Non-DL fitting model:", nondl_model.coef_)
                print("Non-DL fitting model r2:", nondl_r2)

                dl_x = x[[5,6,7,8,10,11]]
                dl_X = dl_x.reshape((-1, 1))
                dl_y = y[[5,6,7,8,10,11]]
                dl_model, dl_r2 = linear_regression(dl_X, dl_y)
                print("DL fitting model:", dl_model.coef_)
                print("DL fitting model r2:", dl_r2)
                

                fig, ax = plt.subplots(figsize=(5,5))
                ax.scatter(non_dl_x, non_dl_y, c='C0')
                ax.scatter(dl_x, dl_y, c='C1', alpha=0.6)
                ax.scatter(x[9], y[9], c='cyan')

                # plot regression line
                ax.plot(non_dl_x, nondl_model.predict(non_dl_X), c = 'C0', linewidth=2, alpha=.3, solid_capstyle='round', label=r"R$^2$: %.2f"%(nondl_r2))
                ax.plot(dl_x, dl_model.predict(dl_X), c = 'C1', linewidth=2, alpha=.3, solid_capstyle='round', label=r"R$^2$: %.2f"%(dl_r2))

                ax.set_xlabel("State-specific contacts",weight='bold')
                ax.set_ylabel("Shared contacts",weight='bold')
                for i, label in enumerate(args.methods):
                    if label == "RaptorX-3DModeling":
                        plt.annotate("RaptorX-3D", (pp12_means[i]+0.005, pp_co_means[i] - 0.015), size=6)
                        continue
                    if label == "trRosetta":
                        plt.annotate(label, (pp12_means[i]+0.005, pp_co_means[i] - 0.04), size=6)
                        continue
                    if label == "ResPRE":
                        plt.annotate(label, (pp12_means[i]+0.005, pp_co_means[i] - 0.07), size=6)
                        continue
                    if label == "RaptorX-Contact": label = "RaptorX-C"
                    plt.annotate(label, (pp12_means[i]+0.05, pp_co_means[i]), size=6)

                ax.set_xlim(0, 1)
                ax.set_xticks(np.arange(0, 1.2, step=0.2))
                ax.set_ylim(0, 1)
                ax.set_yticks(np.arange(0, 1.2, step=0.2))

                ax.set_box_aspect(1)
                plt.legend(loc='lower right')
                plt.title("Precision", weight='bold')
                plt.savefig(path, bbox_inches='tight')
                plt.savefig(path[:-4]+'.png', bbox_inches='tight')

                print(path)
                plt.clf()
                plt.close()
# analyze  the correlation of p_co and p12
if args.percentage_corrcoef:
    size = 12
    rcParams.update({'font.size': size})
    rcParams.update({'font.weight': 'bold'})
    rcParams.update({'axes.titlesize': size})
    rcParams.update({'axes.titleweight': 'bold'})
    rcParams.update({'axes.labelweight': 'bold'})
    rcParams.update({'axes.labelsize': size})
    ranges = ['Long_range']

    for range_ in ranges:
        subplot_id = 0
        data_top = {}
        top_describe = {'Top L'}
        for top_n in top_describe:
            data_m = {}

            # precision of common or state-unique contacts
            pp_all_means = []

            p_co_means = []
            p_co_stds = []
            
            p12_means = []
            p12_stds = []
            for m in args.methods:
                p_c = data_methods[m][range_][top_n] # 6 * No. of pros

                p_co_mean = np.mean(p_c[0,:])
                if range_ == "Long_range" and top_n == "Top L":
                    print(m, " p_co_mean: ", p_co_mean)
                p_co_std = np.std(p_c[0,:])
                p_co_means.append(p_co_mean)
                p_co_stds.append(p_co_std)

                p12_mean = np.mean(p_c[1,:]+p_c[2,:])
                p12_std = np.std(p_c[1,:]+p_c[2,:])
                p12_means.append(p12_mean)
                p12_stds.append(p12_std)

                pp_all_mean = np.mean(p_c[0,:]+p_c[1,:]+p_c[2,:])
                pp_all_means.append(pp_all_mean)

            # scatter for p_co and p_12
            if top_n == "Top L":
                path = "../output/percentage_TopL_pro_co_pro_12"+range_+".pdf"

                x = np.array(p12_means)
                y = np.array(p_co_means)

                print(x)
                print(y)
                xy = np.array([x, y])
                np.savetxt("../output/percetage_"+range_+"TopL_co_vs_p12.csv", xy, fmt='%.4f', delimiter=',')
                print("../output/percentage_"+range_+"TopL_co_vs_p12.csv")

                non_dl_x = x[:5] # MI mfDCA PSICOV CCMpred plmDCA
                non_dl_X = non_dl_x.reshape((-1, 1))
                non_dl_y = y[:5]
                nondl_model, nondl_r2 = linear_regression(non_dl_X, non_dl_y)
                print("Non-DL fitting model:", nondl_model.coef_)
                print("Non-DL fitting model r2:", nondl_r2)

                dl_x = x[[5,6,7,8,10,11]]
                dl_X = dl_x.reshape((-1, 1))
                dl_y = y[[5,6,7,8,10,11]]
                dl_model, dl_r2 = linear_regression(dl_X, dl_y)
                print("DL fitting model:", dl_model.coef_)
                print("DL fitting model r2:", dl_r2)
                

                fig, ax = plt.subplots(figsize=(5,5))
                ax.scatter(non_dl_x, non_dl_y, c='C0')
                ax.scatter(dl_x, dl_y, c='C1', alpha=0.6)
                ax.scatter(x[9], y[9], c='cyan')

                # plot regression line
                ax.plot(non_dl_x, nondl_model.predict(non_dl_X), c = 'C0', linewidth=2, alpha=.3, solid_capstyle='round', label=r"R$^2$: %.2f"%(nondl_r2))
                ax.plot(dl_x, dl_model.predict(dl_X), c = 'C1', linewidth=2, alpha=.3, solid_capstyle='round', label=r"R$^2$: %.2f"%(dl_r2))

                ax.set_xlabel("State-specific contacts",weight='bold')
                ax.set_ylabel("Shared contacts",weight='bold')
                for i, label in enumerate(args.methods):
                    if label == "RaptorX-3DModeling":
                        plt.annotate("RaptorX-3D", (x[i]+0.005, y[i] - 0.015), size=12)
                        continue
                    if label == "trRosetta":
                        plt.annotate(label, (x[i]+0.005, y[i] - 0.04), size=12)
                        continue
                    if label == "ResPRE":
                        plt.annotate(label, (x[i]+0.005, y[i] - 0.07), size=12)
                        continue
                    if label == "RaptorX-Contact": label = "RaptorX-C"
                    plt.annotate(label, (x[i]+0.005, y[i]), size=12)

                ax.set_xlim(0, 0.24)
                ax.set_xticks(np.arange(0, 0.30, step=0.06))
                ax.set_ylim(0, 1)
                ax.set_yticks(np.arange(0, 1.2, step=0.2))

                ax.set_box_aspect(1)
                plt.legend(loc='lower right')
                plt.title("Percentage", weight='bold')
                plt.savefig(path, bbox_inches='tight')
                plt.savefig(path[:-4]+'.png', bbox_inches='tight')

                print(path)

            # scatter for pp_all and p_co
            if top_n == "Top L":
                path = "../output/TopL_pre_all_pro_co"+range_+".pdf"

                x = np.array(p_co_means)
                y = np.array(pp_all_means)

                non_dl_x = x[:5] # MI mfDCA PSICOV CCMpred plmDCA
                non_dl_X = non_dl_x.reshape((-1, 1))
                non_dl_y = y[:5]
                nondl_model, nondl_r2 = linear_regression(non_dl_X, non_dl_y)
                print("Non-DL fitting model:", nondl_model.coef_)
                print("Non-DL fitting model r2:", nondl_r2)

                dl_x = x[[5,6,7,8,10,11]]
                dl_X = dl_x.reshape((-1, 1))
                dl_y = y[[5,6,7,8,10,11]]
                dl_model, dl_r2 = linear_regression(dl_X, dl_y)
                print("DL fitting model:", dl_model.coef_)
                print("DL fitting model r2:", dl_r2)
                

                fig, ax = plt.subplots(figsize=(5,5))
                ax.scatter(non_dl_x, non_dl_y, c='C0')
                ax.scatter(dl_x, dl_y, c='C1', alpha=0.6)
                ax.scatter(x[9], y[9], c='cyan')

                # plot regression line
                ax.plot(non_dl_x, nondl_model.predict(non_dl_X), c = 'C0', linewidth=2, alpha=.3, solid_capstyle='round', label=r"R$^2$: %.2f"%(nondl_r2))
                ax.plot(dl_x, dl_model.predict(dl_X), c = 'C1', linewidth=2, alpha=.3, solid_capstyle='round', label=r"R$^2$: %.2f"%(dl_r2))

                ax.set_xlabel("Percentage for shared contacts",weight='bold')
                ax.set_ylabel("Precision",weight='bold')
                for i, label in enumerate(args.methods):
                    if label == "RaptorX-3DModeling":
                        plt.annotate("RaptorX-3D", (x[i]+0.03, y[i] - 0.02), size=8)
                        continue
                    if label == "trRosetta":
                        plt.annotate(label, (x[i]+0.005, y[i] - 0.04), size=8)
                        continue
                    if label == "ResPRE":
                        plt.annotate(label, (x[i]+0.005, y[i] - 0.07), size=8)
                        continue
                    if label == "RaptorX-Contact": 
                        label = "RaptorX-C"
                        plt.annotate(label, (x[i]+0.005, y[i]-0.01 ), size=8)
                        continue
                    plt.annotate(label, (x[i]+0.015, y[i]), size=8)

                ax.set_xlim(0, 1)
                ax.set_xticks(np.arange(0, 1.2, step=0.2))
                ax.set_ylim(0, 1)
                ax.set_yticks(np.arange(0, 1.2, step=0.2))

                ax.set_box_aspect(1)
                plt.legend(loc='lower right')
                #plt.title("Proportion", weight='bold')
                plt.savefig(path, bbox_inches='tight')
                plt.savefig(path[:-4]+'.png', bbox_inches='tight')

                print(path)

            # scatter for pp_all and p12
            if top_n == "Top L":
                path = "../output/TopL_pre_all_pro_12"+range_+".pdf"
                x = np.array(p12_means)
                y = np.array(pp_all_means)

                non_dl_x = x[:5] # MI mfDCA PSICOV CCMpred plmDCA
                non_dl_X = non_dl_x.reshape((-1, 1))
                non_dl_y = y[:5]
                nondl_model, nondl_r2 = linear_regression(non_dl_X, non_dl_y)
                print("Non-DL fitting model:", nondl_model.coef_)
                print("Non-DL fitting model r2:", nondl_r2)

                dl_x = x[[5,6,7,8,10,11]]
                dl_X = dl_x.reshape((-1, 1))
                dl_y = y[[5,6,7,8,10,11]]
                dl_model, dl_r2 = linear_regression(dl_X, dl_y)
                print("DL fitting model:", dl_model.coef_)
                print("DL fitting model r2:", dl_r2)
                

                fig, ax = plt.subplots(figsize=(5,5))
                ax.scatter(non_dl_x, non_dl_y, c='C0')
                ax.scatter(dl_x, dl_y, c='C1', alpha=0.6)
                ax.scatter(x[9], y[9], c='cyan')

                # plot regression line
                ax.plot(non_dl_x, nondl_model.predict(non_dl_X), c = 'C0', linewidth=2, alpha=.3, solid_capstyle='round', label=r"R$^2$: %.2f"%(nondl_r2))
                ax.plot(dl_x, dl_model.predict(dl_X), c = 'C1', linewidth=2, alpha=.3, solid_capstyle='round', label=r"R$^2$: %.2f"%(dl_r2))

                ax.set_xlabel("Percentage for state-specific contacts",weight='bold')
                ax.set_ylabel("Precision",weight='bold')
                for i, label in enumerate(args.methods):
                    if label == "RaptorX-3DModeling":
                        plt.annotate("RaptorX-3D", (x[i]+0.005, y[i] - 0.015), size=8)
                        continue
                    if label == "trRosetta":
                        plt.annotate(label, (x[i]+0.005, y[i] - 0.04), size=8)
                        continue
                    if label == "ResPRE":
                        plt.annotate(label, (x[i]+0.005, y[i] - 0.07), size=8)
                        continue
                    if label == "RoseTTAFold":
                        plt.annotate(label, (x[i]-0.045, y[i]), size=8)
                        continue
                    if label == "RaptorX-Contact": label = "RaptorX-C"
                    plt.annotate(label, (x[i]+0.005, y[i]), size=8)
                ax.set_xlim(0, 0.18)
                ax.set_xticks(np.arange(0, 0.21, step=0.03))
                ax.set_ylim(0, 1)
                ax.set_yticks(np.arange(0, 1.2, step=0.2))

                ax.set_box_aspect(1)
                plt.legend(loc='lower right')
                #plt.title("Proportion", weight='bold')
                plt.savefig(path, bbox_inches='tight')
                plt.savefig(path[:-4]+'.png', bbox_inches='tight')

                print(path)
                plt.clf()
                plt.close()

# analyze  the correlation of c_co and c12
if args.coverage_corrcoef:
    size = 12
    rcParams.update({'font.size': size})
    rcParams.update({'font.weight': 'bold'})
    rcParams.update({'axes.titlesize': size})
    rcParams.update({'axes.titleweight': 'bold'})
    rcParams.update({'axes.labelweight': 'bold'})
    rcParams.update({'axes.labelsize': size})

    width= 1 / 4
    for range_ in ranges:
        data_top = {}
        for top_n in top_describe:
            data_m = {}
            c_co_means = []
            c_co_stds = []
            
            c12_means = []
            c12_stds = []
            for m in args.methods:
                c_c = data_methods[m][range_][top_n] # 6 * No. of pros

                c_co_mean = np.nanmean(c_c[3,:])
                c_co_std = np.nanstd(c_c[3,:])
                c_co_means.append(c_co_mean)
                c_co_stds.append(c_co_std)

                c12_mean = np.nanmean(c_c[4,:])
                c12_std = np.nanstd(c_c[4,:])
                c12_means.append(c12_mean)
                c12_stds.append(c12_std)

            # scatter for c_co and c12
            if top_n == "Top L":
                x = np.array(c12_means)
                y = np.array(c_co_means)

                xy = np.array([x, y])
                np.savetxt("../output/coverage_"+range_+"TopL_co_vs_p12.csv", xy, fmt='%.4f', delimiter=',')
                print("../output/coverage_"+range_+"TopL_co_vs_p12.csv")

                non_dl_x = x[:5] # MI mfDCA PSICOV CCMpred plmDCA
                non_dl_X = non_dl_x.reshape((-1, 1))
                non_dl_y = y[:5]
                nondl_model, nondl_r2 = linear_regression(non_dl_X, non_dl_y)
                print("Non-DL fitting model:", nondl_model.coef_)
                print("Non-DL fitting model r2:", nondl_r2)

                dl_x = x[[5,6,7,8,10,11]]
                dl_X = dl_x.reshape((-1, 1))
                dl_y = y[[5,6,7,8,10,11]]
                dl_model, dl_r2 = linear_regression(dl_X, dl_y)
                print("DL fitting model:", dl_model.coef_)
                print("DL fitting model r2:", dl_r2)
                

                fig, ax = plt.subplots(figsize=(5,5))
                ax.scatter(non_dl_x, non_dl_y, c='C0')
                ax.scatter(dl_x, dl_y, c='C1', alpha=0.6)
                ax.scatter(x[9], y[9], c='cyan')

                # plot regression line
                ax.plot(non_dl_x, nondl_model.predict(non_dl_X), c = 'C0', linewidth=2, alpha=.3, solid_capstyle='round', label=r"R$^2$: %.2f"%(nondl_r2))
                ax.plot(dl_x, dl_model.predict(dl_X), c = 'C1', linewidth=2, alpha=.3, solid_capstyle='round', label=r"R$^2$: %.2f"%(dl_r2))

                ax.set_xlabel("State-specific contacts",weight='bold')
                ax.set_ylabel("Shared contacts",weight='bold')
                for i, label in enumerate(args.methods):
                    if label == "RoseTTAFold":
                        plt.annotate(label, (c12_means[i] - 0.050, c_co_means[i]-0.009), size=12)
                        continue
                    if label == "RaptorX-Contact":
                        plt.annotate("RaptorX-C", (c12_means[i] -0.042, c_co_means[i]-0.01 ), size=12)
                        continue
                    if label == "RaptorX-3DModeling":
                        plt.annotate("RaptorX-3D", (c12_means[i] , c_co_means[i] + 0.01), size=12)
                        continue
                    if label == "trRosetta":
                        plt.annotate(label, (c12_means[i], c_co_means[i] + 0.01), size=12)
                        continue
                    if label == "ResPRE":
                        plt.annotate(label, (c12_means[i]-0.01, c_co_means[i]-0.025 ), size=12)
                        continue
                    if label == "plmDCA":
                        plt.annotate(label, (c12_means[i] - 0.032, c_co_means[i]), size=12)
                        continue
                    plt.annotate(label, (c12_means[i]+0.003, c_co_means[i]), size=12)
                path = "../output/coverage_"+range_+"TopL_co_vs_p12.pdf"
                plt.legend(loc='lower right')
                plt.xlabel("State-specific contacts")
                plt.ylabel("Shared contacts")
                plt.title("Coverage")
                ax.set_xlim(0, 0.15)
                ax.set_xticks(np.arange(0, 0.18, step=0.03))
                ax.set_ylim(0, 0.6)
                ax.set_yticks(np.arange(0, 0.8, step=0.2))

                ax.set_box_aspect(1)
                plt.savefig(path, bbox_inches='tight')
                plt.savefig(path[:-4]+".png", bbox_inches='tight')
                print(path)
                plt.clf()
                plt.close()


# analyze  r_square

if args.r_square:

    ranges = ['Long_range']
    top_describe = ['Top L']
    for range_ in ranges:
        for top_n in top_describe:

            r_cos = []
            r1s = []
            r2s = []
            r_delta_p12s = []

            for m in args.methods:
                metrics = data_methods[m][range_][top_n] # 6 * No. of pros

                p_co_in = metrics[0, 0::2]
                p_co_out = metrics[0, 1::2]

                p1_in = metrics[1, 0::2]
                p1_out = metrics[1, 1::2]

                p2_in = metrics[2, 0::2]
                p2_out = metrics[2, 1::2]

                delta_p12_in = (metrics[1, 0::2] - metrics[2, 0::2])
                delta_p12_out = (metrics[1, 1::2] - metrics[2, 1::2])

                # assume the in and out follow y=x
                p_co_out = np.asarray(p_co_out).reshape((-1,1))
                p1_out = np.asarray(p1_out).reshape((-1,1))
                p2_out = np.asarray(p2_out).reshape((-1,1))
                delta_p12_out = np.asarray(delta_p12_out).reshape((-1,1))

                r_co = r2_score(p_co_in, p_co_out)
                r1 = r2_score(p1_in, p1_out)
                r2 = r2_score(p2_in, p2_out)
                r_delta_p12 = r2_score(delta_p12_in, delta_p12_out)

                r_cos.append(r_co)
                r1s.append(r1)
                r2s.append(r2)
                r_delta_p12s.append(r_delta_p12)
            
            data = {'methods': args.methods,
                    'proportion_co': r_cos,
                    'proportion1': r1s, 
                    'proportion2': r2s,
                    'proportion1-proportion2': r_delta_p12s
                    }

            df = pd.DataFrame(data=data)
            df.to_csv("../output/r_square.csv", float_format="%.2f", index=False)
            print("../output/r_square.csv")
