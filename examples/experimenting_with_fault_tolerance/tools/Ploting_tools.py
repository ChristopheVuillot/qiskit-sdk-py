import ast
import os
import matplotlib.pyplot as plt
import re
import statistics
import numpy as np
from numpy import array
from scipy.stats import t, norm

from tools.Experiment_tools import CIRCUIT_NAMES, rank_qubit_pairs_conf, convert_parameter

def plot_everything_raw(folder):
    list_file = os.listdir(folder)
    n_circuit = len(list_file)
    n_skipped = 0
    n_keapt = 0
    cmap = plt.cm.get_cmap('gist_ncar')
    plt.figure(figsize=(20, 20))
    for j, circuit_filename in enumerate(list_file):
        with open(folder+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        stat_dist = []
        qasm_count = []
        for expe_data_string in expe_list:
            try:
                expe_data = ast.literal_eval(expe_data_string)
                qasm_count.append(expe_data['qasm_count'])
                stat_dist.append(expe_data['stat_dist'])
                n_keapt += 1
            except SyntaxError as syntax_error:
                n_skipped += 1
        plt.scatter(qasm_count, stat_dist, label=circuit_filename, marker='x', c=cmap(j/n_circuit))
        #if circuit_filename[0] == 'b' or 'nft' in circuit_filename or '|0+>' in circuit_filename:
        #    color = 'r'
        #else:
        #    color = 'b'
        #plt.scatter(qasm_count, stat_dist, label=circuit_filename, c=color)
    plt.title('all experiments')
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0))
    plt.yscale('log')
    plt.grid()
    plt.show()
    print(n_skipped, n_keapt)

PLOT_LABELS = ['bare[1, 0]',
               'bare[2, 0]',
               'bare[2, 1]',
               'bare[2, 4]',
               'bare[3, 2]',
               'bare[3, 4]',
               'encoded|00>ftv1',
               'encoded|00>ftv2',
               'encoded|00>nftv1',
               'encoded|00>nftv2',
               'encoded|0+>',
               'encoded|00>+|11>']

def plot_everything_averaged(folder, logscaley=True, sublabels=PLOT_LABELS, ci=.99):
    list_file = os.listdir(folder)
    n_skipped = 0
    n_kept = 0
    re_labels = [re.compile('[\\S]*\\[1, 0\\].txt'),
                 re.compile('[\\S]*\\[2, 0\\].txt'),
                 re.compile('[\\S]*\\[2, 1\\].txt'),
                 re.compile('[\\S]*\\[2, 4\\].txt'),
                 re.compile('[\\S]*\\[3, 2\\].txt'),
                 re.compile('[\\S]*\\[3, 4\\].txt'),
                 re.compile('[\\S]*>ftv1.txt'),
                 re.compile('[\\S]*>ftv2.txt'),
                 re.compile('[\\S]*nftv1.txt'),
                 re.compile('[\\S]*nftv2.txt'),
                 re.compile('e[\\S]*\\|0\\+>.txt'),
                 re.compile('e[\\S]*\\|00>\\+\\|11>.txt')]
    labels = ['bare[1, 0]',
              'bare[2, 0]',
              'bare[2, 1]',
              'bare[2, 4]',
              'bare[3, 2]',
              'bare[3, 4]',
              'encoded|00>ftv1',
              'encoded|00>ftv2',
              'encoded|00>nftv1',
              'encoded|00>nftv2',
              'encoded|0+>',
              'encoded|00>+|11>']
    cmap = plt.cm.get_cmap('Paired')
    colors = [cmap(j/12) for j in [1,5,10,11,4,0,8,9,6,7,2,3]]
    qasm_counts = [[] for j in range(0, 12)]
    circuit_indices = [[] for j in range(0, 12)]
    stat_dists = [[] for j in range(0, 12)]
    stdevs = [[] for j in range(0, 12)]
    conf_ints = [[] for j in range(0, 12)]
    fig, ax = plt.subplots(figsize=(20, 20))
    for j, circuit_filename in enumerate(list_file):
        total = 0
        stat_dist_avg = 0
        values = []
        with open(folder+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        for k, reg_ex in enumerate(re_labels):
            if reg_ex.match(circuit_filename):
                index = k
                break
        for expe_data_string in expe_list:
            try:
                expe_data = ast.literal_eval(expe_data_string)
                total += 1
                stat_dist_avg += expe_data['stat_dist']
                values.append(expe_data['stat_dist'])
                n_kept += 1
            except SyntaxError:
                n_skipped += 1
        #if stat_dist_avg/total > .2:
            #print(circuit_filename, stat_dist_avg/total)
        stat_dists[index].append(stat_dist_avg/total)
        qasm_counts[index].append(expe_data['qasm_count'])
        for l,cn in enumerate(CIRCUIT_NAMES):
            if cn in circuit_filename:
                circuit_index = l+1
        circuit_indices[index].append(circuit_index)
        stdevs[index].append(statistics.stdev(values))
        ct = t.interval(ci, len(values)-1, loc=0, scale=1)[1]
        conf_ints[index].append(ct*statistics.stdev(values)/np.sqrt(len(values)))
    #plots = [plt.scatter(qasm_counts[j], stat_dists[j], marker='x', label=labels[j], c=colors[j]) for j in range(0, 12)]
    indices_to_plot = [labels.index(pl) for pl in sublabels]
    for j in indices_to_plot:
        ax.errorbar(np.array(circuit_indices[j]), np.array(stat_dists[j]), yerr=np.array(conf_ints[j]), markersize=15, mew=3, fmt='x', label=labels[j], c=colors[j])
    l1, l2 = zip(*sorted(zip(circuit_indices[0], qasm_counts[0])))
    ax2 = ax.twinx()
    ax2.plot(l1, l2, 'k-') 
    handles, labs = ax.get_legend_handles_labels()
    ax.set_title('all experiments')
    #ax.legend([h[0] for h in handles], labels, loc='lower left', bbox_to_anchor=(1, 0))
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    if logscaley:
        ax.set_yscale('log')
    #ax.set_xticks(range(1,21))
    #ax.set_xticklabels(CIRCUIT_NAMES)
    #ax.tick_params(axis='x', labelrotation=90)
    ax.grid(True)
    fig.tight_layout()
    plt.sca(ax)
    plt.xticks(range(1,21), [c[1:] for c in CIRCUIT_NAMES], rotation=60, horizontalalignment='right')
    plt.show()
    print(n_skipped, n_kept)
    for k in range(0,6):
        print(labels[k],statistics.mean(stat_dists[k]))

def plot_everything_averaged_diff(folder, logscaley=True, bareindex=1, ci=.99):
    list_file = os.listdir(folder)
    n_skipped = 0
    n_kept = 0
    re_labels = [re.compile('[\\S]*\\[1, 0\\].txt'),
                 re.compile('[\\S]*\\[2, 0\\].txt'),
                 re.compile('[\\S]*\\[2, 1\\].txt'),
                 re.compile('[\\S]*\\[2, 4\\].txt'),
                 re.compile('[\\S]*\\[3, 2\\].txt'),
                 re.compile('[\\S]*\\[3, 4\\].txt'),
                 re.compile('[\\S]*>ftv1.txt'),
                 re.compile('[\\S]*>ftv2.txt'),
                 re.compile('[\\S]*nftv1.txt'),
                 re.compile('[\\S]*nftv2.txt'),
                 re.compile('e[\\S]*\\|0\\+>.txt'),
                 re.compile('e[\\S]*\\|00>\\+\\|11>.txt')]
    labels = ['bare[1, 0]',
              'bare[2, 0]',
              'bare[2, 1]',
              'bare[2, 4]',
              'bare[3, 2]',
              'bare[3, 4]',
              'encoded|00>ftv1',
              'encoded|00>ftv2',
              'encoded|00>nftv1',
              'encoded|00>nftv2',
              'encoded|0+>',
              'encoded|00>+|11>']
    cmap = plt.cm.get_cmap('Paired')
    colors = [cmap(j/12) for j in [1,5,10,11,4,0,8,9,6,7,2,3]]
    qasm_counts = [[] for j in range(0, 12)]
    circuit_indices = [[] for j in range(0, 12)]
    stat_dists = [[] for j in range(0, 12)]
    stdevs = [[] for j in range(0, 12)]
    conf_ints = [[] for j in range(0, 12)]
    fig, ax = plt.subplots(figsize=(20, 20))
    for j, circuit_filename in enumerate(list_file):
        total = 0
        stat_dist_avg = 0
        values = []
        with open(folder+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        for k, reg_ex in enumerate(re_labels):
            if reg_ex.match(circuit_filename):
                index = k
                break
        for expe_data_string in expe_list:
            try:
                expe_data = ast.literal_eval(expe_data_string)
                total += 1
                stat_dist_avg += expe_data['stat_dist']
                values.append(expe_data['stat_dist'])
                n_kept += 1
            except SyntaxError:
                n_skipped += 1
        #if stat_dist_avg/total > .2:
            #print(circuit_filename, stat_dist_avg/total)
        stat_dists[index].append(stat_dist_avg/total)
        qasm_counts[index].append(expe_data['qasm_count'])
        for l,cn in enumerate(CIRCUIT_NAMES):
            if cn in circuit_filename:
                circuit_index = l+1
        circuit_indices[index].append(circuit_index)
        stdevs[index].append(statistics.stdev(values))
        ct = t.interval(ci, len(values)-1, loc=0, scale=1)[1]
        conf_ints[index].append(ct*statistics.stdev(values)/np.sqrt(len(values)))
    #plots = [plt.scatter(qasm_counts[j], stat_dists[j], marker='x', label=labels[j], c=colors[j]) for j in range(0, 12)]
    l1, l2 = zip(*sorted(zip(circuit_indices[bareindex], qasm_counts[bareindex])))
    ax2 = ax.twinx()
    ax2.plot(l1, l2, 'k-') 
    indices_to_plot = [pl for pl in range(6,12)]
    ax.plot([j for j in range(-1,22)], [0 for j in range(-1,22)], '-r')
    for j in indices_to_plot:
        for k in range(0,len(stat_dists[j])):
            bare_ref = stat_dists[bareindex][circuit_indices[bareindex].index(circuit_indices[j][k])]
            stat_dists[j][k] -= bare_ref
        ax.errorbar(np.array(circuit_indices[j]), np.array(stat_dists[j]), yerr=np.array(conf_ints[j]), markersize=15, mew=3, fmt='x', label=labels[j], c=colors[j])
    handles, labs = ax.get_legend_handles_labels()
    ax.set_title('Encoded circuits compared to bare qubit pair '+labels[bareindex][4:])
    #ax.legend([h[0] for h in handles], labels, loc='lower left', bbox_to_anchor=(1, 0))
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    if logscaley:
        ax.set_yscale('log')
    #ax.set_xticks(range(1,21))
    #ax.set_xticklabels(CIRCUIT_NAMES)
    #ax.tick_params(axis='x', labelrotation=90)
    ax.grid(True)
    ax.set_xlim([0,21])
    plt.sca(ax)
    plt.xticks(range(1,21), [c[1:] for c in CIRCUIT_NAMES], rotation=60, horizontalalignment='right')
    fig.tight_layout()
    plt.show()
    print(n_skipped, n_kept)
    for k in range(0,6):
        print(labels[k],statistics.mean(stat_dists[k]))

def plot_everything_calib_data(folder, qubit_index, parameter_name, multi_qubit_param=False, logscaley=True, sublabels=PLOT_LABELS, ci=.99, x_range=[0,10**(-5)]):
    list_file = os.listdir(folder)
    n_skipped = 0
    n_kept = 0
    re_labels = [re.compile('[\\S]*\\[1, 0\\].txt'),
                 re.compile('[\\S]*\\[2, 0\\].txt'),
                 re.compile('[\\S]*\\[2, 1\\].txt'),
                 re.compile('[\\S]*\\[2, 4\\].txt'),
                 re.compile('[\\S]*\\[3, 2\\].txt'),
                 re.compile('[\\S]*\\[3, 4\\].txt'),
                 re.compile('[\\S]*>ftv1.txt'),
                 re.compile('[\\S]*>ftv2.txt'),
                 re.compile('[\\S]*nftv1.txt'),
                 re.compile('[\\S]*nftv2.txt'),
                 re.compile('e[\\S]*\\|0\\+>.txt'),
                 re.compile('e[\\S]*\\|00>\\+\\|11>.txt')]
    labels = ['bare[1, 0]',
              'bare[2, 0]',
              'bare[2, 1]',
              'bare[2, 4]',
              'bare[3, 2]',
              'bare[3, 4]',
              'encoded|00>ftv1',
              'encoded|00>ftv2',
              'encoded|00>nftv1',
              'encoded|00>nftv2',
              'encoded|0+>',
              'encoded|00>+|11>']
    cmap = plt.cm.get_cmap('Paired')
    colors = [cmap(j/12) for j in [1,5,10,11,4,0,8,9,6,7,2,3]]
    stat_dists = [[] for j in range(0, 12)]
    parameter = [[] for j in range(0, 12)]
    fig, ax = plt.subplots(figsize=(20, 20))
    for j, circuit_filename in enumerate(list_file):
        with open(folder+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        for k, reg_ex in enumerate(re_labels):
            if reg_ex.match(circuit_filename):
                index = k
                break
        for expe_data_string in expe_list:
            try:
                expe_data = ast.literal_eval(expe_data_string)
                stat_dists[index].append(expe_data['stat_dist'])
                if multi_qubit_param:
                    parameter[index].append(convert_parameter(expe_data['calibration']['multiQubitGates'][qubit_index][parameter_name]))
                else:
                    parameter[index].append(convert_parameter(expe_data['calibration']['qubits'][qubit_index][parameter_name]))
                n_kept += 1
            except SyntaxError:
                n_skipped += 1
    indices_to_plot = [labels.index(pl) for pl in sublabels]
    for j in indices_to_plot:
        ax.scatter(np.array(parameter[j]), np.array(stat_dists[j]), marker='x', label=labels[j], c=colors[j])
    handles, labs = ax.get_legend_handles_labels()
    ax.set_title('all experiments vs {} for {} '.format(parameter_name, qubit_index))
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    if logscaley:
        ax.set_yscale('log')
    ax.grid(True)
    ax.set_xlim(x_range)
    fig.tight_layout()
    plt.show()
    print(n_skipped, n_kept)
    for k in range(0,12):
        print(labels[k],statistics.mean(stat_dists[k]))

def save_plot_data_averaged_diff(folder_data, folder_save, bareindex=1, ci=.99):
    list_file = os.listdir(folder_data)
    n_skipped = 0
    n_kept = 0
    re_labels = [re.compile('[\\S]*\\[1, 0\\].txt'),
                 re.compile('[\\S]*\\[2, 0\\].txt'),
                 re.compile('[\\S]*\\[2, 1\\].txt'),
                 re.compile('[\\S]*\\[2, 4\\].txt'),
                 re.compile('[\\S]*\\[3, 2\\].txt'),
                 re.compile('[\\S]*\\[3, 4\\].txt'),
                 re.compile('[\\S]*>ftv1.txt'),
                 re.compile('[\\S]*>ftv2.txt'),
                 re.compile('[\\S]*nftv1.txt'),
                 re.compile('[\\S]*nftv2.txt'),
                 re.compile('e[\\S]*\\|0\\+>.txt'),
                 re.compile('e[\\S]*\\|00>\\+\\|11>.txt')]
    qasm_counts = [[] for j in range(0, 12)]
    circuit_indices = [[] for j in range(0, 12)]
    stat_dists = [[] for j in range(0, 12)]
    stdevs = [[] for j in range(0, 12)]
    conf_ints = [[] for j in range(0, 12)]
    for j, circuit_filename in enumerate(list_file):
        total = 0
        stat_dist_avg = 0
        values = []
        with open(folder_data+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        for k, reg_ex in enumerate(re_labels):
            if reg_ex.match(circuit_filename):
                index = k
                break
        for expe_data_string in expe_list:
            try:
                expe_data = ast.literal_eval(expe_data_string)
                total += 1
                stat_dist_avg += expe_data['stat_dist']
                values.append(expe_data['stat_dist'])
                n_kept += 1
            except SyntaxError:
                n_skipped += 1
        stat_dists[index].append(stat_dist_avg/total)
        qasm_counts[index].append(expe_data['qasm_count'])
        for l,cn in enumerate(CIRCUIT_NAMES):
            if cn in circuit_filename:
                circuit_index = l+1
        circuit_indices[index].append(circuit_index)
        stdevs[index].append(statistics.stdev(values))
        ct = t.interval(ci, len(values)-1, loc=0, scale=1)[1]
        conf_ints[index].append(ct*statistics.stdev(values)/np.sqrt(len(values)))
    indices_to_plot = [pl for pl in range(6,12)]
    for j in indices_to_plot:
        for k in range(0,len(stat_dists[j])):
            bare_ref = stat_dists[bareindex][circuit_indices[bareindex].index(circuit_indices[j][k])]
            stat_dists[j][k] -= bare_ref
    print(n_skipped, n_kept)

PLOT_LABELS_RANK = ['bare1',
                    'bare2',
                    'bare3',
                    'bare4',
                    'bare5',
                    'bare6',
                    'encoded|00>ftv1',
                    'encoded|00>ftv2',
                    'encoded|00>nftv1',
                    'encoded|00>nftv2',
                    'encoded|0+>',
                    'encoded|00>+|11>']

# The ranking function seems to be not a good ranking function for the performances
# of the bare circuits.
def plot_everything_averaged_bare_ranked(folder, logscaley=True, sublabels=PLOT_LABELS_RANK, ci=.99):
    list_file = os.listdir(folder)
    n_skipped = 0
    n_kept = 0
    re_labels = [re.compile('[\\S]*\\[1, 0\\].txt'),
                 re.compile('[\\S]*\\[2, 0\\].txt'),
                 re.compile('[\\S]*\\[2, 1\\].txt'),
                 re.compile('[\\S]*\\[2, 4\\].txt'),
                 re.compile('[\\S]*\\[3, 2\\].txt'),
                 re.compile('[\\S]*\\[3, 4\\].txt'),
                 re.compile('[\\S]*>ftv1.txt'),
                 re.compile('[\\S]*>ftv2.txt'),
                 re.compile('[\\S]*nftv1.txt'),
                 re.compile('[\\S]*nftv2.txt'),
                 re.compile('e[\\S]*\\|0\\+>.txt'),
                 re.compile('e[\\S]*\\|00>\\+\\|11>.txt')]
    pairs = [[1, 0], [2, 0], [2, 1], [2, 4], [3, 2], [3, 4]]
    labels = ['bare1',
              'bare2',
              'bare3',
              'bare4',
              'bare5',
              'bare6',
              'encoded|00>ftv1',
              'encoded|00>ftv2',
              'encoded|00>nftv1',
              'encoded|00>nftv2',
              'encoded|0+>',
              'encoded|00>+|11>']
    cmap = plt.cm.get_cmap('Paired')
    colors = [cmap(j/12) for j in [1,5,10,11,4,0,8,9,6,7,2,3]]
    qasm_counts = [[] for j in range(0, 12)]
    stat_dists = [[] for j in range(0, 12)]
    stdevs = [[] for j in range(0, 12)]
    conf_ints = [[] for j in range(0, 12)]
    bare_ranked_stat_dists = {}
    bare_ranked_qasm_counts = {}
    bare_ranked_stdevs = {}
    fig, ax = plt.subplots(figsize=(20, 20))
    for j, circuit_filename in enumerate(list_file):
        total = 0
        stat_dist_avg = 0
        values = []
        with open(folder+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        for k, reg_ex in enumerate(re_labels):
            if reg_ex.match(circuit_filename):
                index = k
                break
        if index < 6:
            circuit_name = circuit_filename[2:len(circuit_filename)-10]
            for expe_data_string in expe_list:
                try:
                    expe_data = ast.literal_eval(expe_data_string)
                    pair = pairs[index]
                    ranking, _ = rank_qubit_pairs_conf(expe_data['calibration'], pairs)
                    index = ranking.index(tuple(pair))
                    bare_ranked_stat_dists.setdefault(circuit_name, [[] for j in range(0, 6)])
                    bare_ranked_qasm_counts.setdefault(circuit_name, [0 for j in range(0, 6)])
                    bare_ranked_stat_dists[circuit_name][index].append(expe_data['stat_dist'])
                    bare_ranked_qasm_counts[circuit_name][index] = expe_data['qasm_count']
                    n_kept += 1
                except SyntaxError:
                    n_skipped += 1
        else:
            for expe_data_string in expe_list:
                try:
                    expe_data = ast.literal_eval(expe_data_string)
                    total += 1
                    stat_dist_avg += expe_data['stat_dist']
                    values.append(expe_data['stat_dist'])
                    n_kept += 1
                except SyntaxError:
                    n_skipped += 1
            stat_dists[index].append(stat_dist_avg/total)
            qasm_counts[index].append(expe_data['qasm_count'])
            stdevs[index].append(statistics.stdev(values))
            ct = t.interval(ci, len(values)-1, loc=0, scale=1)[1]
            conf_ints[index].append(ct*statistics.stdev(values)/np.sqrt(len(values)))
    for circuit_name, values_lists in bare_ranked_stat_dists.items():
        for index, values in enumerate(values_lists):
            stat_dists[index].append(statistics.mean(values))
            qasm_counts[index].append(bare_ranked_qasm_counts[circuit_name][index])
            stdevs[index].append(statistics.stdev(values))
            ct = t.interval(ci, len(values)-1, loc=0, scale=1)[1]
            conf_ints[index].append(ct*statistics.stdev(values)/np.sqrt(len(values)))
    #plots = [plt.scatter(qasm_counts[j], stat_dists[j], marker='x', label=labels[j], c=colors[j]) for j in range(0, 12)]
    indices_to_plot = [labels.index(pl) for pl in sublabels]
    for j in indices_to_plot:
        ax.errorbar(np.array(qasm_counts[j]), np.array(stat_dists[j]), yerr=np.array(conf_ints[j]), markersize=15, mew=3, fmt='x', label=labels[j], c=colors[j])
    handles, labs = ax.get_legend_handles_labels()
    ax.set_title('all experiments')
    #ax.legend([h[0] for h in handles], labels, loc='lower left', bbox_to_anchor=(1, 0))
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    if logscaley:
        ax.set_yscale('log')
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    print(n_skipped, n_kept)
    for k in range(0,6):
        print(labels[k],statistics.mean(stat_dists[k]))



# Plotting one bare run next to one encoded run with the expected output distribution
def plot_one_expe(analysed_data1,analysed_data2,confidence):
    N = 4;
    ind = np.arange(N)
    
    width = 0.25
    
    fig, ax = plt.subplots()
    hist1 = ax.bar(ind, analysed_data1['values']/analysed_data1['total_valid'], width, color='r', yerr=analysed_data1['stand_dev']*norm.ppf(1/2+confidence/2),label=analysed_data1['version']+' (stat dist : '+str(analysed_data1['stat_dist'])+')')
    hist2 = ax.bar(ind+width, analysed_data2['values']/analysed_data2['total_valid'], width, color='b', yerr=analysed_data2['stand_dev']*norm.ppf(1/2+confidence/2),label=analysed_data2['version']+' (stat dist : '+str(analysed_data2['stat_dist'])+')')
    hist3 = ax.bar(ind+2*width, analysed_data1['output_distribution'], width, color='g',label='Expectation')
    
    ax.set_ylabel('Frequencies')
    ax.set_title('Performance on the circuit : '+analysed_data1['circuit_desc']
                 +' (ratio of post-selection : '+str(analysed_data2['post_selected_ratio'])+')')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(analysed_data1['labels'])
    
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    
    plt.show()
