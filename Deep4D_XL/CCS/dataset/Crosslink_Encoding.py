import argparse
import numpy as np
import pandas as pd
import os
import shutil
from .aa_onehot import onehot
from .mass_cal import mz_cal as m

def extract(str,symbol):
    index = []
    for i in range(len(str)):
        if str[i]==symbol:
            index.append(i)
    start_index = index[0]
    end_index = index[1]
    str1 = str[(start_index+1):end_index]
    return str1

def get_position_angle_vec(position,dim):
    return [position / np.power(10000, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]

def get_onehot(pep, cl_site):
    peptide = pep
    peptide_np = np.zeros((50, 23))
    if peptide[0] == 'a':
        x = 1
        pep_len = len(peptide) - 1
    else:
        x = 0
        pep_len = len(peptide)
    i = 0
    for j in range(x, len(peptide)):
        peptide_np[i, :] = onehot.AA_onehot[peptide[j]]
        i = i + 1
    if peptide[0] == 'a':
        peptide_np[0, 20] = 1
    peptide_np[int(cl_site-1), 22] = 1
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i, 23) for pos_i in range(pep_len)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    peptide_np[:pep_len, ] = peptide_np[:pep_len, ] + sinusoid_table
    return  peptide_np, pep_len

def crosslink_ccs_encoding(input,output):
    data = pd.read_csv(input, header=0)
    os.mkdir(output)
    peptide = np.array(data['combine_peptide'])
    peptide1 = np.array(data['peptide1'])
    peptide2 = np.array(data['peptide2'])
    site1 = np.array(data['site1'])
    site2 = np.array(data['site2'])
    CCS = np.array(data['ccs'])
    charge = np.array(data['charge'])
    peptide_num = len(peptide1)
    a = m()
    for h in range(peptide_num):
        if h % 200 == 0:
            print(h)
        pep = peptide[h]
        pep1 = peptide1[h]
        pep2 = peptide2[h]
        s1 = site1[h]
        s2 = site2[h]
        peptide_ccs = int(CCS[h] * 10000)
        peptide_charge = charge[h]
        if (len(pep1) < 50) and (len(pep2) < 50):
            pep1_np, pep1_len = get_onehot(pep1, s1)
            pep2_np, pep2_len = get_onehot(pep2, s2)
            peptide_np = np.row_stack((pep1_np, pep2_np))
            lenth = len(str(peptide_num))
            num = str(h).zfill(lenth)
            np.save(f'{output}/{num}%{mz}%#{pep}#_{peptide_ccs}_${pep1_len}$@{pep2_len}@*{peptide_charge}*.npy',peptide_np)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode crosslink peptides into npy files with CCS metadata.')
    parser.add_argument('--filename', type=str, default='total_crosslink_filter_precursor_test',
                        help='Base filename in ./data (without extension).')
    args = parser.parse_args()
    crosslink_ccs_encoding(f'./data/{args.filename}.csv', f'./data/{args.filename}')
