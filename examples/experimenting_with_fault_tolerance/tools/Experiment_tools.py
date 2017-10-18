###########################################################################################
#            Tools for demonstrating fault-tolerance on the IBM 5Q chip
#
#   contributor : Christophe Vuillot
#   affiliations : - JARA Institute for Quantum Information, RWTH Aachen university
#                  - QuTech, TU Delft
#
###########################################################################################

import re
import os
import time
import statistics
import numpy as np
# next import is used when using eval function on stored processed_data. Very clean yes.
from numpy import array
import matplotlib.pyplot as plt
from scipy.stats import t, norm
from qiskit import QISKitError

# Functions that helps deciding what is the "best" pair of qubits
#################################################################
def convert_rate(time_t, time_gate):
    if time_t['unit'] == 'ns':
        multiplier_t = 10**(-9)
    elif time_t['unit'] == 'µs':
        multiplier_t = 10**(-6)
    else:
        multiplier_t = 1
    if time_gate['unit'] == 'ns':
        multiplier_gate = 10**(-9)
    elif time_gate['unit'] == 'µs':
        multiplier_gate = 10**(-6)
    else:
        multiplier_gate = 1
    return 1-np.exp(-time_gate['value']*multiplier_gate/(time_t['value']*multiplier_t))


def rank_qubit_pairs(quantump, backend):
    config = quantump.get_backend_configuration(backend, list_format=True)
    calib = quantump.get_backend_calibration(backend)
    param = quantump.get_backend_parameters(backend)
    pairs_numbers = dict([(tuple(c), []) for c in config['coupling_map']])
    for pair, numbers in pairs_numbers.items():
        numbers.append(calib['qubits'][pair[0]]['gateError']['value'])
        numbers.append(calib['qubits'][pair[0]]['readoutError']['value'])
        numbers.append(calib['qubits'][pair[1]]['gateError']['value'])
        numbers.append(calib['qubits'][pair[1]]['readoutError']['value'])
        for multiq_calib in calib['multi_qubit_gates']:
            if multiq_calib['qubits'] == list(pair):
                numbers.append(multiq_calib['gateError']['value'])
        numbers.append(convert_rate(param['qubits'][pair[0]]['T1'], param['qubits'][pair[0]]['gateTime']))
        numbers.append(convert_rate(param['qubits'][pair[0]]['T2'], param['qubits'][pair[0]]['gateTime']))
        numbers.append(convert_rate(param['qubits'][pair[1]]['T1'], param['qubits'][pair[1]]['gateTime']))
        numbers.append(convert_rate(param['qubits'][pair[1]]['T2'], param['qubits'][pair[1]]['gateTime']))
    for pair, numbers in pairs_numbers.items():
        pairs_numbers[pair] = sorted(numbers, reverse=True)
    return (sorted(pairs_numbers, key=pairs_numbers.__getitem__), pairs_numbers)

# Functions that create all the circuits inside a given QuantumProgram module
#############################################################################

# Misc aux circuits
###################
def swap_circuit(pair, quantump, qri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuitswap = quantump.create_circuit("SWAP"+str(pair), qrs, crs)
    qcircuitswap.cx(qrs[qri][pair[0]], qrs[qri][pair[1]])
    qcircuitswap.h(qrs[qri][pair[0]])
    qcircuitswap.h(qrs[qri][pair[1]])
    qcircuitswap.cx(qrs[qri][pair[0]], qrs[qri][pair[1]])
    qcircuitswap.h(qrs[qri][pair[0]])
    qcircuitswap.h(qrs[qri][pair[1]])
    qcircuitswap.cx(qrs[qri][pair[0]], qrs[qri][pair[1]])
    return qcircuitswap

def measure_all(quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuitmeasure = quantump.create_circuit("Measure all", qrs, crs)
    qcircuitmeasure.measure(qrs[qri][0], crs[cri][0])
    qcircuitmeasure.measure(qrs[qri][1], crs[cri][1])
    qcircuitmeasure.measure(qrs[qri][2], crs[cri][2])
    qcircuitmeasure.measure(qrs[qri][3], crs[cri][3])
    qcircuitmeasure.measure(qrs[qri][4], crs[cri][4])
    return qcircuitmeasure

# The encoded preparations
##########################
def encoded_00_prep_ftv1(quantump, qri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qc_ftv1 = quantump.create_circuit("e|00>ftv1", qrs, crs)
    qc_ftv1.h(qrs[qri][2])
    qc_ftv1.cx(qrs[qri][2], qrs[qri][0])
    qc_ftv1.cx(qrs[qri][2], qrs[qri][1])
    qc_ftv1.h(qrs[qri][2])
    qc_ftv1.h(qrs[qri][3])
    qc_ftv1.cx(qrs[qri][3], qrs[qri][2])
    qc_ftv1.h(qrs[qri][2])
    qc_ftv1.h(qrs[qri][3])
    qc_ftv1.cx(qrs[qri][2], qrs[qri][4])
    qc_ftv1.cx(qrs[qri][2], qrs[qri][0])
    #qc_ftv1.measure(qrs[qri][0],crs[cri][0])
    return qc_ftv1

def encoded_00_prep_nftv1(quantump, qri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qc_nftv1 = quantump.create_circuit("e|00>nftv1", qrs, crs)
    qc_nftv1.h(qrs[qri][3])
    qc_nftv1.cx(qrs[qri][3], qrs[qri][4])
    qc_nftv1.cx(qrs[qri][3], qrs[qri][2])
    qc_nftv1.cx(qrs[qri][2], qrs[qri][1])
    return qc_nftv1

def encoded_00_prep_ftv2(quantump, qri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qc_ftv2 = quantump.create_circuit("e|00>ftv2", qrs, crs)
    qc_ftv2.h(qrs[qri][3])
    qc_ftv2.cx(qrs[qri][3], qrs[qri][2])
    qc_ftv2.h(qrs[qri][2])
    qc_ftv2.h(qrs[qri][3])
    qc_ftv2.cx(qrs[qri][2], qrs[qri][1])
    qc_ftv2.cx(qrs[qri][3], qrs[qri][4])
    qc_ftv2.h(qrs[qri][4])
    qc_ftv2.extend(swap_circuit([2, 4], quantump))
    qc_ftv2.cx(qrs[qri][2], qrs[qri][0])
    qc_ftv2.cx(qrs[qri][1], qrs[qri][0])
    qc_ftv2.h(qrs[qri][4])
    #qc_ftv2.measure(qrs[qri][0],crs[cri][0])
    return qc_ftv2

def encoded_00_prep_nftv2(quantump, qri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qc_nftv2 = quantump.create_circuit("e|00>nftv2", qrs, crs)
    qc_nftv2.h(qrs[qri][2])
    qc_nftv2.h(qrs[qri][3])
    qc_nftv2.cx(qrs[qri][3], qrs[qri][4])
    qc_nftv2.h(qrs[qri][4])
    qc_nftv2.cx(qrs[qri][2], qrs[qri][4])
    qc_nftv2.h(qrs[qri][4])
    qc_nftv2.extend(swap_circuit([2, 1], quantump))
    qc_nftv2.cx(qrs[qri][3], qrs[qri][2])
    qc_nftv2.cx(qrs[qri][2], qrs[qri][0])
    qc_nftv2.h(qrs[qri][0])
    qc_nftv2.cx(qrs[qri][1], qrs[qri][0])
    qc_nftv2.h(qrs[qri][0])
    qc_nftv2.h(qrs[qri][1])
    #qc_nftv2.measure(qrs[qri][0],crs[cri][0])
    return qc_nftv2

def encoded_0p_prep(quantump, qri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qc_0p = quantump.create_circuit("e|0+>", qrs, crs)
    qc_0p.h(qrs[qri][1])
    qc_0p.h(qrs[qri][3])
    qc_0p.cx(qrs[qri][3], qrs[qri][2])
    qc_0p.extend(swap_circuit([2,1], quantump))
    qc_0p.cx(qrs[qri][2], qrs[qri][4])
    return qc_0p

def encoded_2cat_prep(quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qc_2cat = quantump.create_circuit("e|00>+|11>", qrs, crs)
    qc_2cat.h(qrs[qri][2])
    qc_2cat.h(qrs[qri][3])
    qc_2cat.cx(qrs[qri][2], qrs[qri][1])
    qc_2cat.cx(qrs[qri][3], qrs[qri][4])
    return qc_2cat

# The bare preparations
#######################
def bare_00_prep(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_00 = quantump.create_circuit("b|00>"+str(pair), qrs, crs)
    return qcircuit_bare_00

def bare_0p_prep(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_0p = quantump.create_circuit("b|0+>"+str(pair), qrs, crs)
    qcircuit_bare_0p.h(qrs[qri][pair[1]])
    return qcircuit_bare_0p

def bare_2cat_prep(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_2cat = quantump.create_circuit("b|00>+|11>"+str(pair), qrs, crs)
    qcircuit_bare_2cat.h(qrs[qri][pair[0]])
    qcircuit_bare_2cat.cx(qrs[qri][pair[0]], qrs[qri][pair[1]])
    return qcircuit_bare_2cat

# The encoded gates
###################
def encoded_X1_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_X1 = quantump.create_circuit("eX1", qrs, crs)
    qcircuit_encoded_X1.x(qrs[qri][mapping[0]])
    qcircuit_encoded_X1.x(qrs[qri][mapping[1]])
    return qcircuit_encoded_X1

def encoded_X2_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_X2 = quantump.create_circuit("eX2", qrs, crs)
    qcircuit_encoded_X2.x(qrs[qri][mapping[0]])
    qcircuit_encoded_X2.x(qrs[qri][mapping[2]])
    return qcircuit_encoded_X2

def encoded_Z1_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_Z1 = quantump.create_circuit("eZ1", qrs, crs)
    qcircuit_encoded_Z1.z(qrs[qri][mapping[1]])
    qcircuit_encoded_Z1.z(qrs[qri][mapping[3]])
    return qcircuit_encoded_Z1

def encoded_Z2_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_Z2 = quantump.create_circuit("eZ2", qrs, crs)
    qcircuit_encoded_Z2.z(qrs[qri][mapping[2]])
    qcircuit_encoded_Z2.z(qrs[qri][mapping[3]])
    return qcircuit_encoded_Z2

def encoded_CZ_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_CZ = quantump.create_circuit("eCZ", qrs, crs)
    qcircuit_encoded_CZ.s(qrs[qri][mapping[0]])
    qcircuit_encoded_CZ.s(qrs[qri][mapping[1]])
    qcircuit_encoded_CZ.s(qrs[qri][mapping[2]])
    qcircuit_encoded_CZ.s(qrs[qri][mapping[3]])
    return qcircuit_encoded_CZ

def encoded_HHS_circuit(mapping, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_encoded_HHS = quantump.create_circuit("eHHS", qrs, crs)
    qcircuit_encoded_HHS.h(qrs[qri][mapping[0]])
    qcircuit_encoded_HHS.h(qrs[qri][mapping[1]])
    qcircuit_encoded_HHS.h(qrs[qri][mapping[2]])
    qcircuit_encoded_HHS.h(qrs[qri][mapping[3]])
    return qcircuit_encoded_HHS

# The bare gates
################
def bare_X1_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_X1 = quantump.create_circuit("bX1"+str(pair), qrs, crs)
    qcircuit_bare_X1.x(qrs[qri][pair[0]])
    return qcircuit_bare_X1

def bare_X2_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_X2 = quantump.create_circuit("bX2"+str(pair), qrs, crs)
    qcircuit_bare_X2.x(qrs[qri][pair[1]])
    return qcircuit_bare_X2

def bare_Z1_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_Z1 = quantump.create_circuit("bZ1"+str(pair), qrs, crs)
    qcircuit_bare_Z1.z(qrs[qri][pair[1]])
    return qcircuit_bare_Z1

def bare_Z2_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_Z2 = quantump.create_circuit("bZ2"+str(pair), qrs, crs)
    qcircuit_bare_Z2.z(qrs[qri][pair[1]])
    return qcircuit_bare_Z2

def bare_CZ_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_CZ = quantump.create_circuit("bCZ"+str(pair), qrs, crs)
    qcircuit_bare_CZ.h(qrs[qri][pair[1]])
    qcircuit_bare_CZ.cx(qrs[qri][pair[0]],qrs[0][pair[1]])
    qcircuit_bare_CZ.h(qrs[qri][pair[1]])
    return qcircuit_bare_CZ

def bare_HHS_circuit(pair, quantump, qri=0, cri=0):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    qcircuit_bare_HHS = quantump.create_circuit("bHHS"+str(pair), qrs, crs)
    qcircuit_bare_HHS.h(qrs[qri][pair[0]])
    qcircuit_bare_HHS.h(qrs[qri][pair[1]])
    return qcircuit_bare_HHS

# The dictionaries for all circuits
###################################
DICT_ENCODED = dict(zip(['eX1','eX2','eZ1','eZ2','eHHS','eCZ','e|00>ftv1','e|00>ftv2','e|00>nftv1','e|00>nftv2','e|0+>','e|00>+|11>'],
                        [encoded_X1_circuit,
                         encoded_X2_circuit,
                         encoded_Z1_circuit,
                         encoded_Z2_circuit,
                         encoded_HHS_circuit,
                         encoded_CZ_circuit,
                         encoded_00_prep_ftv1,
                         encoded_00_prep_ftv2,
                         encoded_00_prep_nftv1,
                         encoded_00_prep_nftv2,
                         encoded_0p_prep,
                         encoded_2cat_prep]))

DICT_BARE = dict(zip(['bX1','bX2','bZ1','bZ2','bHHS','bCZ','b|00>','b|0+>','b|00>+|11>'],
                     [bare_X1_circuit,
                      bare_X2_circuit,
                      bare_Z1_circuit,
                      bare_Z2_circuit,
                      bare_HHS_circuit,
                      bare_CZ_circuit,
                      bare_00_prep,
                      bare_0p_prep,
                      bare_2cat_prep]))


# The circuits for the experiment with input state and output distribution
##########################################################################
CIRCUITS = [[['X1', 'HHS', 'CZ', 'X2'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['HHS', 'Z1', 'CZ'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['HHS', 'Z1', 'Z2'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['HHS', 'Z2', 'CZ'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['Z2', 'X2'], '|00>+|11>', [0, .5, .5, 0]],
            [['X1', 'Z2'], '|0+>', [0, 0, .5, .5]],
            [['HHS', 'Z1'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['HHS', 'CZ'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['X1', 'X2'], '|00>', [0, 0, 0, 1]],
            [['HHS', 'Z2'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['X1'], '|00>+|11>', [0, .5, .5, 0]],
            [['X1'], '|0+>', [0, 0, .5, .5]],
            [['HHS'], '|00>', [0.25, 0.25, 0.25, 0.25]],
            [['Z2'], '|00>+|11>', [.5, 0, 0, .5]],
            [['Z2'], '|0+>', [.5, .5, 0, 0]],
            [['X1'], '|00>', [0, 0, 1, 0]],
            [['X2'], '|00>', [0, 1, 0, 0]],
            [[], '|00>+|11>', [.5, 0, 0, .5]],
            [[], '|0+>', [.5, .5, 0, 0]],
            [[], '|00>', [1, 0, 0, 0]]]

# The names of the different versions for encoding |00> and the chosen mapping
##############################################################################
ENCODED_VERSION_LIST = ['ftv1', 'ftv2', 'nftv1', 'nftv2']
MAPPING = [3, 2, 1, 4]
CODEWORDS = [['0000', '1111'], ['1100', '0011'], ['1010', '0101'], ['1001', '0110']]
MAPPED_CODEWORDS = [[], [], [], []]
for i, cl in enumerate(CODEWORDS):
    for c in cl:
        MAPPED_CODEWORDS[i].append(''.join(list(reversed([c[j-1] for j in MAPPING])))+'0')

# Function that assembles all circuits within a given QuantumProgram module
###########################################################################
def all_circuits(quantump, possible_pairs, mapping=MAPPING, circuits=CIRCUITS, dict_bare=DICT_BARE, dict_encoded=DICT_ENCODED, encoded_version_list=ENCODED_VERSION_LIST):
    qrs = [quantump.get_quantum_register(qrn) for qrn in quantump.get_quantum_register_names()]
    crs = [quantump.get_classical_register(crn) for crn in quantump.get_classical_register_names()]
    circuit_names = []
    for lc in circuits:
        for pair in possible_pairs:
            qcirc = quantump.create_circuit('bM'+'-'.join(reversed(lc[0]))+lc[1]+str(pair), qrs, crs)
            circuit_names.append('bM'+'-'.join(reversed(lc[0]))+lc[1]+str(pair))
            qcirc.extend(dict_bare['b'+lc[1]](pair, quantump))
            number_swap = 0
            for g in lc[0]:
                if g[0] == 'X' or g[0] == 'Z':
                    key = 'b'+g[0]+str(((int(g[1]) - 1 + number_swap) % 2) + 1)
                elif g[0] == 'H':
                    number_swap += 1
                    key = 'b'+g
                else:
                    key = 'b'+g
                qcirc.extend(dict_bare[key](pair, quantump))
            qcirc.extend(measure_all(quantump))
        if lc[1] == '|00>':
            for v in encoded_version_list:
                qcirc = quantump.create_circuit('eM'+'-'.join(reversed(lc[0]))+lc[1]+v, qrs, crs)
                circuit_names.append('eM'+'-'.join(reversed(lc[0]))+lc[1]+v)
                qcirc.extend(dict_encoded['e'+lc[1]+v](quantump))
                for g in lc[0]:
                    qcirc.extend(dict_encoded['e'+g](mapping, quantump))
                qcirc.extend(measure_all(quantump))
        else:
            qcirc = quantump.create_circuit('eM'+'-'.join(reversed(lc[0]))+lc[1], qrs, crs)
            circuit_names.append('eM'+'-'.join(reversed(lc[0]))+lc[1])
            qcirc.extend(dict_encoded['e'+lc[1]](quantump))
            for gate in lc[0]:
                qcirc.extend(dict_encoded['e'+gate](mapping, quantump))
            qcirc.extend(measure_all(quantump))
    return circuit_names


# Callback function for the run circuits
########################################
def post_treatment(res):
    '''Callback function to write the results into a file after the jobs are finished.
    '''
    with open('data/callback.log', 'a') as logfile:
        logfile.write(str(time.asctime(time.localtime(time.time())))+':'+res.get_status()+' - id: '+res.get_job_id()+'\n')
    circuit_names = res.get_names()
    try:
        for circuit_name in circuit_names:
            circuit_data = res.get_data(circuit_name)
            filename = 'data/Raw_counts/' + circuit_name + '_' + circuit_data['date']+'.txt'
            with open(filename, 'w') as data_file:
                data_file.write(str(circuit_data['counts']))
        with open('data/completed.txt', 'a') as completed_file:
            completed_file.write(res.get_job_id()+'\n')
    except QISKitError as qiskit_err:
        if str(qiskit_err) == '\'Time Out\'':
            with open('data/timed_out.txt', 'a') as timed_out_file:
                timed_out_file.write(res.get_job_id()+'\n')


def post_treatment_list(results):
    '''Callback function to write the results into a file after the jobs are finished.
    '''
    for res in results:
        with open('data/callback.log', 'a') as logfile:
            logfile.write(str(time.asctime(time.localtime(time.time())))+':'+res.get_status()+' - id: '+res.get_job_id()+'\n')
        circuit_names = res.get_names()
        try:
            for circuit_name in circuit_names:
                circuit_data = res.get_data(circuit_name)
                filename = 'data/Raw_counts/' + circuit_name + '_' + circuit_data['date']+'.txt'
                with open(filename, 'w') as data_file:
                    data_file.write(str(circuit_data['counts']))
            with open('data/completed.txt', 'a') as completed_file:
                completed_file.write(res.get_job_id()+'\n')
        except QISKitError as qiskit_err:
            if str(qiskit_err) == '\'Time Out\'':
                with open('data/timed_out.txt', 'a') as timed_out_file:
                    timed_out_file.write(res.get_job_id()+'\n')


# Function to fetch previously timed out results
def fetch_previous(filename, api):
    '''Function that fetch previously ran experiements whose ids are stored in data/filename
    '''
    new = 0
    with open('data/'+filename, 'r') as ids_file_read:
        id_lines = ids_file_read.readlines()
    with open('data/'+filename, 'w') as ids_file_write:
        for id_line in id_lines:
            id_string = id_line.rstrip()
            job_result = api.get_job(id_string)
            if not job_result['status'] == 'COMPLETED':
                ids_file_write.write(id_line)
            else:
                new += 1
                with open('data/completed_'+filename, 'a') as comp_file:
                    comp_file.write(id_line)
                with open('data/API_dumps/api_dump_'+id_string+'.txt', 'w') as data_file:
                    data_file.write(str(job_result))
    return new


# Functions to analyse gathered data
####################################

# Function to get the dictionary of qasm vs circuit name
def get_qasm_name_dict(compiled_qobj_list):
    dictionary = {}
    for n, v in zip(sum([[circuit['compiled_circuit_qasm'] for circuit in batch['circuits']] for batch in compiled_qobj_list],[]),
                    sum([[circuit['name'] for circuit in batch['circuits']] for batch in compiled_qobj_list],[])):
        dictionary.setdefault(n, []).append(v)
    return dictionary

def api_data_to_dict(res, name):
    data_dict = {'name' : name}
    data_dict.setdefault('raw_counts', {}).update(res['data']['counts'])
    data_dict['counts'] = {'00' : 0, '01' : 0, '10' : 0, '11' : 0, 'err' : 0, 'total_valid' : 0}
    data_dict['qasm_count'] = len(res['qasm'].split('\n')) - 5
    n = len(name)
    number_H = name.count('H')/2

    if name[0] == 'b':
        circuit_info = [c for c in CIRCUITS if '-'.join(reversed(c[0]))+c[1] == name[2:n-6]][0]
        data_dict['expected_distribution_array'] = np.array(circuit_info[2], dtype=float)
        pair = eval(name[n-6:n])
        data_dict['version'] = 'bare'
        if number_H % 2 == 1:
            pair.reverse()
        for key in res['data']['counts']:
            data_dict['counts'][''.join([key[4-j] for j in pair])] += res['data']['counts'][key]
            data_dict['counts']['total_valid'] += res['data']['counts'][key]

    elif name[0] == 'e':
        if 'nftv' in name[n-5:n]:
            circuit_info = [c for c in CIRCUITS if '-'.join(reversed(c[0]))+c[1] == name[2:n-5]][0]
        elif 'ftv' in name[n-5:n]:
            circuit_info = [c for c in CIRCUITS if '-'.join(reversed(c[0]))+c[1] == name[2:n-4]][0]
        else:
            circuit_info = [c for c in CIRCUITS if '-'.join(reversed(c[0]))+c[1] == name[2:n]][0]
        data_dict['expected_distribution_array'] = np.array(circuit_info[2], dtype=float)
        data_dict['version'] = 'encoded'
        for key in res['data']['counts']:
            found = False
            for i,codeword_list in enumerate(MAPPED_CODEWORDS):
                if key in codeword_list:
                    data_dict['counts']["{0:02b}".format(i)] += res['data']['counts'][key]
                    data_dict['counts']['total_valid'] += res['data']['counts'][key]
                    found = True
                    break
            if not found:
                data_dict['counts']['err'] += res['data']['counts'][key]

    data_dict['experimental_distribution_array'] = np.array([data_dict['counts'][s]/data_dict['counts']['total_valid']
                                                             for s in ['00', '01', '10', '11']],dtype=float)
    data_dict['post_selection_ratio'] = data_dict['counts']['total_valid']/(data_dict['counts']['err']+data_dict['counts']['total_valid'])
    data_dict['stat_dist'] = .5*sum(np.abs(data_dict['experimental_distribution_array']-data_dict['expected_distribution_array']))
    data_dict['stand_dev'] = np.sqrt(data_dict['experimental_distribution_array']
                                     *(1-data_dict['experimental_distribution_array'])
                                     /data_dict['counts']['total_valid'])
    stat_dist_stand_dev = 0
    for j in range(0, 4):
        stat_dist_stand_dev += data_dict['experimental_distribution_array'][j]*(1-data_dict['experimental_distribution_array'][j])/(4*data_dict['counts']['total_valid'])
    for i in range(0, 4):
        for j in range(0, 4):
            if i != j:
                stat_dist_stand_dev += data_dict['experimental_distribution_array'][i]*data_dict['experimental_distribution_array'][j]/(4*data_dict['counts']['total_valid'])
    stat_dist_stand_dev = np.sqrt(stat_dist_stand_dev)
    data_dict['stat_dist_stand_dev'] = stat_dist_stand_dev
    return data_dict


def process_api_dump(filename, dict_qasm_name, dict_res={}):
    with open(filename, 'r') as api_dump_file:
        job_results = eval(api_dump_file.read())
    for res in job_results['qasms']:
        names = dict_qasm_name['OPENQASM 2.0;'+res['qasm']]
        for name in names:
            res_entry = api_data_to_dict(res, name)
            res_entry['calibration'] = job_results['calibration']
            dict_res.setdefault(name, []).append(res_entry)
            with open('data/Processed_data/' + name + '.txt', 'a') as circuit_file:
                circuit_file.write(str(res_entry) + '\n')
    return dict_res

def process_all_api_dumps(file_of_files_to_process, file_of_already_processed_files, dict_qasm_name):
    n_processed = 0
    with open(file_of_already_processed_files, 'r') as file_processed:
        processed = file_processed.readlines()
    with open(file_of_files_to_process, 'r') as file_to_process:
        to_process = file_to_process.readlines()
    with open(file_of_already_processed_files, 'a') as file_processed:
        for filename in to_process:
            if not filename in processed:
                n_processed += 1
                process_api_dump('data/API_dumps/api_dump_' + filename.rstrip() + '.txt', dict_qasm_name)
                file_processed.write(filename)
    return n_processed

def repair_processed_data(filename, new_ext='_repaired'):
    n_repaired = 0
    repaired_lines = []
    with open(filename, 'r') as file:
        lines = file.readlines()
    iter_lines = iter(lines)
    for line in iter_lines:
        n = len(line.rstrip())
        if line.rstrip()[n-1] == ',':
            next_line = next(iter_lines)
            repaired_lines.append(line.rstrip()+next_line)
            n_repaired += 1
        else:
            repaired_lines.append(line)
    if n_repaired > 0:
        new_filename = filename.split('.')[0] + new_ext + '.txt'
        with open(new_filename, 'w') as new_file:
            new_file.writelines(repaired_lines)
    return n_repaired


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
                expe_data = eval(expe_data_string)
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

def plot_everything_binned(folder):
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
                 re.compile('e[\\S]*\\|0+>.txt'),
                 re.compile('e[\\S]*\\|00>+\\|11>.txt')]
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
    colors = [cmap(j/12) for j in range(0, 12)]
    qasm_counts = [[] for j in range(0, 12)]
    stat_dists = [[] for j in range(0, 12)]
    plt.figure(figsize=(20, 20))
    for j, circuit_filename in enumerate(list_file):
        with open(folder+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        for j, reg_ex in enumerate(re_labels):
            if reg_ex.match(circuit_filename):
                color = colors[j]
                label = labels[j]
                break
        for expe_data_string in expe_list:
            try:
                expe_data = eval(expe_data_string)
                qasm_counts[j].append(expe_data['qasm_count'])
                stat_dists[j].append(expe_data['stat_dist'])
                n_kept += 1
            except SyntaxError:
                n_skipped += 1
    plots = [plt.scatter(qasm_counts[j], stat_dists[j], marker='x', label=labels[j], c=colors[j]) for j in range(0, 12)]
    plt.title('all experiments')
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0))
    plt.yscale('log')
    plt.grid()
    plt.show()
    print(n_skipped, n_kept)


def plot_everything_averaged(folder):
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
                 re.compile('e[\\S]*\\|0+>.txt'),
                 re.compile('e[\\S]*\\|00>+\\|11>.txt')]
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
    colors = [cmap(j/12) for j in range(0, 12)]
    qasm_counts = [[] for j in range(0, 12)]
    stat_dists = [[] for j in range(0, 12)]
    stdevs = [[] for j in range(0, 12)]
    fig, ax = plt.subplots(figsize=(20, 20))
    for j, circuit_filename in enumerate(list_file):
        total = 0
        stat_dist_avg = 0
        values = []
        with open(folder+circuit_filename, 'r') as circuit_file:
            expe_list = circuit_file.readlines()
        for j, reg_ex in enumerate(re_labels):
            if reg_ex.match(circuit_filename):
                break
        for expe_data_string in expe_list:
            try:
                expe_data = eval(expe_data_string)
                total += 1
                stat_dist_avg += expe_data['stat_dist']
                values.append(expe_data['stat_dist'])
                n_kept += 1
            except SyntaxError:
                n_skipped += 1
        stat_dists[j].append(stat_dist_avg/total)
        qasm_counts[j].append(expe_data['qasm_count'])
        stdevs[j].append(statistics.stdev(values))
    #plots = [plt.scatter(qasm_counts[j], stat_dists[j], marker='x', label=labels[j], c=colors[j]) for j in range(0, 12)]
    for j in range(0,12):
        ax.errorbar(np.array(qasm_counts[j]), np.array(stat_dists[j]), yerr=np.array(stdevs[j]), markersize=15, mew=3, fmt='x', label=labels[j], c=colors[j])
    handles, labs = ax.get_legend_handles_labels()
    ax.set_title('all experiments')
    ax.legend([h[0] for h in handles], labels, loc='lower left', bbox_to_anchor=(1, 0))
    ax.set_yscale('log')
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    print(n_skipped, n_kept)









# Function that analyse one run (8192 shots) of one circuit in its bare version
def analysis_one_bare_expe(expe_bare, circuit, cpp):

    raw_labels_list = [['0','0','0','0','0'],['0','0','0','0','0'],['0','0','0','0','0'],['0','0','0','0','0']];
    raw_labels_list[1][4-cpp[1]] = '1';
    raw_labels_list[2][4-cpp[0]] = '1';
    raw_labels_list[3][4-cpp[1]] = '1';
    raw_labels_list[3][4-cpp[0]] = '1';
    
    raw_labels = [''.join(rll) for rll in raw_labels_list];
    
    data_bare = expe_bare['result']['data']['counts']
    
    labels = ['00','01','10','11']
    labels_bare = sorted(data_bare)
    
    values_bare = np.array([0,0,0,0],dtype=float)
    total_valid_bare = 0
    total_err_bare = 0
    
    for label in labels_bare:
        if label==raw_labels[0]:
            values_bare[0] += data_bare[label]
            total_valid_bare += data_bare[label]
        elif label==raw_labels[1]:
            if circuit['nH']==0:
                values_bare[1] += data_bare[label]
            else:
                values_bare[2] += data_bare[label]
            total_valid_bare += data_bare[label]
        elif label==raw_labels[2]:
            if circuit['nH']==0:
                values_bare[2] += data_bare[label]
            else:
                values_bare[1] += data_bare[label]
            total_valid_bare += data_bare[label]
        elif label==raw_labels[3]:
            values_bare[3] += data_bare[label]
            total_valid_bare += data_bare[label]
        else:
            total_err_bare += data_bare[label]
        
    values_expectation = np.array(circuit['output_distribution'])
    
    stand_dev = np.sqrt(values_bare/total_valid_bare*(1-values_bare/total_valid_bare)/total_valid_bare)
    
    post_selected_ratio_bare = total_valid_bare/(total_valid_bare+total_err_bare)
    
    stat_dist_bare = .5*sum(np.abs(values_bare/total_valid_bare-values_expectation))
    
    stat_dist_stand_dev = 0
    for j in range(0,4):
        stat_dist_stand_dev += values_bare[j]/total_valid_bare*(1-values_bare[j]/total_valid_bare)/(4*total_valid_bare)
    for i in range(0,4):
        for j in range(0,4):
            if i!=j:
                stat_dist_stand_dev += values_bare[i]/total_valid_bare*values_bare[j]/total_valid_bare/(4*total_valid_bare)

    stat_dist_stand_dev = np.sqrt(stat_dist_stand_dev)
    
    return {'circuit_desc':circuit['circuit_desc'],
            'version':'bare',
            'gate_count':sum(circuit['gate_count_bare']),
            'input_state':circuit['input_state'],
            'labels':labels,
            'values':values_bare,
            'total_valid':total_valid_bare,
            'total_err':total_err_bare,
            'output_distribution':values_expectation,
            'stand_dev':stand_dev,
            'post_selected_ratio':post_selected_ratio_bare,
            'stat_dist':stat_dist_bare,
            'stat_dist_stand_dev':stat_dist_stand_dev}  

# Function that analyse one run (8192 shots) of one circuit in its encoded version
def analysis_one_encoded_expe(expe_encoded, circuit):

    data_encoded = expe_encoded['result']['data']['counts']
    
    labels = ['00','01','10','11']
    labels_encoded = sorted(data_encoded)
    
    values_encoded = np.array([0,0,0,0],dtype=float)
    total_valid_encoded = 0
    total_err_encoded = 0
    
    for label in labels_encoded:
        if label=='00000' or label=='11110':
            values_encoded[0] += data_encoded[label]
            total_valid_encoded += data_encoded[label]
        elif label=='01010' or label=='10100':
            values_encoded[1] += data_encoded[label]
            total_valid_encoded += data_encoded[label]
        elif label=='10010' or label=='01100':
            values_encoded[2] += data_encoded[label]
            total_valid_encoded += data_encoded[label]
        elif label=='11000' or label=='00110':
            values_encoded[3] += data_encoded[label]
            total_valid_encoded += data_encoded[label]
        else:
            total_err_encoded += data_encoded[label]

    values_expectation = np.array(circuit['output_distribution'])
    
    stand_dev = np.sqrt(values_encoded/total_valid_encoded*(1-values_encoded/total_valid_encoded)/total_valid_encoded)
    
    post_selected_ratio_encoded = total_valid_encoded/(total_valid_encoded+total_err_encoded)

    stat_dist_encoded = .5*sum(np.abs(values_encoded/total_valid_encoded-values_expectation))

    stat_dist_stand_dev = 0
    for j in range(0,4):
        stat_dist_stand_dev += values_encoded[j]/total_valid_encoded*(1-values_encoded[j]/total_valid_encoded)/(4*total_valid_encoded)
    for i in range(0,4):
        for j in range(0,4):
            if i!=j:
                stat_dist_stand_dev += values_encoded[i]/total_valid_encoded*values_encoded[j]/total_valid_encoded/(4*total_valid_encoded)

    stat_dist_stand_dev = np.sqrt(stat_dist_stand_dev)
    
    return {'circuit_desc':circuit['circuit_desc'],
            'version':'encoded',
            'gate_count':sum(circuit['gate_count_encoded']), 
            'input_state':circuit['input_state'],
            'labels':labels,
            'values':values_encoded,
            'total_valid':total_valid_encoded,
            'total_err':total_err_encoded,
            'output_distribution':values_expectation,
            'stand_dev':stand_dev,
            'post_selected_ratio':post_selected_ratio_encoded,
            'stat_dist':stat_dist_encoded,
            'stat_dist_stand_dev':stat_dist_stand_dev}

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
    
    plt.show();

    
# Function that analyse all the runs per circuit
def analyse_all_expe(listlist_bare, listlist_encoded, confidence):
    
    all_expe = []
    
    for expe in range(0,20):
        bare_runs = [e[expe] for e in listlist_bare]
        encoded_runs = [e[expe] for e in listlist_encoded]
        
        bare_mean_stat_dist = 0
        encoded_mean_stat_dist = 0
        
        bare_std_dev = 0
        encoded_std_dev = 0
        
        for r in bare_runs:
            bare_mean_stat_dist += r['stat_dist']/len(bare_runs)
            
        for r in encoded_runs:
            encoded_mean_stat_dist += r['stat_dist']/len(encoded_runs)
            
        for r in bare_runs:
            bare_std_dev += (r['stat_dist']-bare_mean_stat_dist)**2/(len(bare_runs)-1)
        bare_std_dev = np.sqrt(bare_std_dev)
        ct = t.interval(confidence, len(bare_runs)-1, loc=0, scale=1)[1]
        bare_confi = ct*bare_std_dev/np.sqrt(len(bare_runs))
            
        for r in encoded_runs:
            encoded_std_dev += (r['stat_dist']-encoded_mean_stat_dist)**2/(len(encoded_runs)-1)
        encoded_std_dev = np.sqrt(encoded_std_dev)    
        ct = t.interval(confidence, len(encoded_runs)-1, loc=0, scale=1)[1]
        encoded_confi = ct*encoded_std_dev/np.sqrt(len(encoded_runs))
            
        all_expe.append({'circuit_desc':bare_runs[0]['circuit_desc'],
                         'gate_count_bare':bare_runs[0]['gate_count'],
                         'gate_count_encoded':encoded_runs[0]['gate_count'],
                         'input_state':bare_runs[0]['input_state'],
                         'output_distribution':bare_runs[0]['output_distribution'],
                         'bare_mean_stat_dist':bare_mean_stat_dist,
                         'encoded_mean_stat_dist':encoded_mean_stat_dist,
                         'bare_std_dev':bare_std_dev,
                         'encoded_std_dev':encoded_std_dev,
                         'bare_conf_int':bare_confi,
                         'encoded_conf_int':encoded_confi,
                         'confidence':confidence})
    return all_expe

# Plotting the difference in statistical distance between encoded and bare version for all circuits
def plot_stat_dist(all_expe):
    
    ng = np.array([e['gate_count_bare'] for e in all_expe])
    sdb = np.array([e['bare_mean_stat_dist'] for e in all_expe])
    sde = np.array([e['encoded_mean_stat_dist'] for e in all_expe])
    cib = np.array([e['bare_conf_int'] for e in all_expe])
    cie = np.array([e['encoded_conf_int'] for e in all_expe])
    
    fig, ax = plt.subplots();
    
    ax.errorbar(ng, sde-sdb, yerr=cib+cie, fmt='rx', label='Difference')
    
    ax.set_ylabel('Difference')
    ax.set_xlabel('Number of gates in the bare circuit')
    ax.set_title('Statistical distances from the ideal distribution\ndepending on the number of gates in the bare circuit\nConfidence interval at '+str(all_expe[0]['confidence']*100)+'%')
    
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0))
    plt.grid()
    plt.show()