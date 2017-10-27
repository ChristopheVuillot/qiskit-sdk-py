from scipy.stats import gaussian_kde
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

def plot_everything_averaged(folder, logscaley=True, sublabels=PLOT_LABELS, ci=.99, save_data_folder_pref=None):
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
        l1, l2, l3 = zip(*sorted(zip(circuit_indices[j], stat_dists[j], conf_ints[j])))
        ax.errorbar(l1, l2, yerr=l3, markersize=15, mew=3, fmt='x', label=labels[j], c=colors[j])
        if save_data_folder_pref:
            with open(save_data_folder_pref + labels[j] + '.dat', 'w') as data_file:
                data_file.write('index stat_dist conf_int99\n')
                for tup in zip(l1, l2, l3):
                    data_file.write('{} {} {}\n'.format(*tup))
    l1, l2 = zip(*sorted(zip(circuit_indices[0], qasm_counts[0])))
    ax2 = ax.twinx()
    ax2.plot(l1, l2, 'k-') 
    if save_data_folder_pref:
        with open(save_data_folder_pref + 'bare_qasm_count.dat', 'w') as data_file:
            data_file.write('index qasm_count\n')
            for tup in zip(l1, l2):
                data_file.write('{} {}\n'.format(*tup))
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
    for k in range(0,12):
        print(labels[k],statistics.mean(stat_dists[k]))

def plot_everything_averaged_diff(folder, logscaley=True, bareindex=1, ci=.99, plot_qasm_count=False, save_data_folder_pref=None):
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
    if plot_qasm_count:
        ax2 = ax.twinx()
    ax.plot([j for j in range(-1,22)], [0 for j in range(-1,22)], '-r')
    indices_to_plot = [pl for pl in range(6,12)]
    for j in indices_to_plot:
        if plot_qasm_count:
            l1, l2 = zip(*sorted(zip(circuit_indices[j], qasm_counts[j])))
            ax2.plot(l1, l2, label=labels[j], c=colors[j]) 
        for k in range(0,len(stat_dists[j])):
            bare_ref = stat_dists[bareindex][circuit_indices[bareindex].index(circuit_indices[j][k])]
            stat_dists[j][k] -= bare_ref
        ax.errorbar(np.array(circuit_indices[j]), np.array(stat_dists[j]), yerr=np.array(conf_ints[j]), markersize=15, mew=3, fmt='x', label=labels[j], c=colors[j])
        if save_data_folder_pref:
            with open(save_data_folder_pref + labels[j] + '-' + labels[bareindex] + '.dat', 'w') as data_file:
                data_file.write('index stat_dist_diff conf_int99\n')
                for tup in sorted(zip(circuit_indices[j], stat_dists[j], conf_ints[j])):
                    data_file.write('{} {} {}\n'.format(*tup))
    handles, labs = ax.get_legend_handles_labels()
    ax.set_title('Encoded circuits compared to bare qubit pair '+labels[bareindex][4:])
    if plot_qasm_count:
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 0))
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
    for k in range(0,12):
        print(labels[k],statistics.mean(stat_dists[k]))

def plot_everything_calib_data(folder, qubit_index, parameter_name, multi_qubit_param=False, logscalex=True, logscaley=True, sublabels=PLOT_LABELS, ci=.99, x_range=[10**(-5),10**(-1)], y_range=[10**(-2), 1]):
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
    #cmap = plt.cm.get_cmap('Paired')
    #colors = [cmap(j/12) for j in [1,5,10,11,4,0,8,9,6,7,2,3]]
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
        x = np.array(parameter[j])
        y = np.array(stat_dists[j])

        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax.scatter(x, y, c=z, s=50, label=labels[j], edgecolor='')
        #ax.scatter(np.array(parameter[j]), np.array(stat_dists[j]), marker='x', label=labels[j], c=colors[j])
    handles, labs = ax.get_legend_handles_labels()
    ax.set_title('all experiments vs {} for {} '.format(parameter_name, qubit_index))
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    if logscaley:
        ax.set_yscale('log')
    if logscalex:
        ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
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

def save_relevant_ploting_data(folder, filename):
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
    with open(filename, 'w') as file_raw:
        with open('averaged_'+filename, 'w') as file_avg:
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
                        parameter[index].append(convert_parameter(expe_data['calibration']['multiQubitGates'][qubit_index][parameter_name]))
                        parameter[index].append(convert_parameter(expe_data['calibration']['qubits'][qubit_index][parameter_name]))
                        n_kept += 1
                    except SyntaxError:
                        n_skipped += 1
    print(n_skipped, n_kept)
    for k in range(0,12):
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
