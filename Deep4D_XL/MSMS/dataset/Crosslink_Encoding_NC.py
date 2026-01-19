import numpy as np
import pandas as pd
import os
import shutil
import argparse
from aa_onehot import onehot

                                              

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

def peptide_onehot_encoding(peptide, cl_site):
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
    peptide_np[int(cl_site - 1), 22] = 1
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i, 23) for pos_i in range(pep_len)])                                    
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])                     
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])                      
    peptide_np[:pep_len, ] = peptide_np[:pep_len, ] + sinusoid_table                                   
    return peptide_np, pep_len

def peptide_msms_encoding(pep_dataframe, peptide1, peptide2):                                                          
    charge = np.array(pep_dataframe['Fragment_charge'])
    by_type = np.array(pep_dataframe['Fragment_type'])
    loss = np.array(pep_dataframe['Neutral_loss'])
    by_id = np.array(pep_dataframe['Fragment_num'])
    intensity = np.array(pep_dataframe['Fragment_intensity'])
    intensity = intensity/(np.max(intensity))
    peptide1 = peptide1.replace('a', '')
    peptide2 = peptide2.replace('a', '')
    len1 = len(peptide1)
    len2 = len(peptide2)
    data = np.zeros((49,20))
    data[(len1 - 1):, :10] = -1
    data[(len2 - 1):, 10:] = -1
                                
                                
    for i in range(len(charge)):
        if by_type[i] == '1b':
            if charge[i] == 1:
                    data[(by_id[i] - 1), 0] = intensity[i]
            elif charge[i] == 2:
                    data[(by_id[i] - 1), 1] = intensity[i]
            elif charge[i] == 3:
                    data[(by_id[i] - 1), 2] = intensity[i]
            elif charge[i] == 4:
                    data[(by_id[i] - 1), 3] = intensity[i]
            elif charge[i] == 5:
                    data[(by_id[i] - 1), 4] = intensity[i]
        elif by_type[i] == '1y':
            if charge[i] == 1:
                    data[(by_id[i] - 1), 5] = intensity[i]
            elif charge[i] == 2:
                    data[(by_id[i] - 1), 6] = intensity[i]
            elif charge[i] == 3:
                    data[(by_id[i] - 1), 7] = intensity[i]
            elif charge[i] == 4:
                    data[(by_id[i] - 1), 8] = intensity[i]
            elif charge[i] == 5:
                    data[(by_id[i] - 1), 9] = intensity[i]
        elif by_type[i] == '2b':
            if charge[i] == 1:
                    data[(by_id[i] - 1), 10] = intensity[i]
            elif charge[i] == 2:
                    data[(by_id[i] - 1), 11] = intensity[i]
            elif charge[i] == 3:
                    data[(by_id[i] - 1), 12] = intensity[i]
            elif charge[i] == 4:
                    data[(by_id[i] - 1), 13] = intensity[i]
            elif charge[i] == 5:
                    data[(by_id[i] - 1), 14] = intensity[i]
        elif by_type[i] == '2y':
            if charge[i] == 1:
                    data[(by_id[i] - 1), 15] = intensity[i]
            elif charge[i] == 2:
                    data[(by_id[i] - 1), 16] = intensity[i]
            elif charge[i] == 3:
                    data[(by_id[i] - 1), 17] = intensity[i]
            elif charge[i] == 4:
                    data[(by_id[i] - 1), 18] = intensity[i]
            elif charge[i] == 5:
                    data[(by_id[i] - 1), 19] = intensity[i]
    return data

def encoding(input, output):
    data = pd.read_csv(input)
    os.mkdir(f'{output}_msms')
    os.mkdir(f'{output}_onehot')
    charge_list = list(set(data['charge']))
    num = 0
    for z in charge_list:
        data1 = data[data['charge'] == z]
        if len(np.array(data1['title'])) > 0:
            data1 = data1.sort_values('title', ignore_index=True)             
            peptide_list = np.array(data1['title'])
            peptide = peptide_list[0]
            index_list = []
            peptide_num = len(set(data1['title']))
            for i in range(len(peptide_list)):
                if peptide_list[i] == peptide:
                    index_list.append(i)
                else:
                    num = num + 1
                    if num % 200 == 0:
                        print(num)
                    data2 = data1.iloc[index_list]
                    combine_peptide = np.array(data2['combine_peptide'])
                    com_peptide = combine_peptide[0]
                    peptide1 = data2['peptide1']
                    peptide2 = data2['peptide2']
                    site1 = data2['site1']
                    site2 = data2['site2']
                    peptide1 = np.array(peptide1)
                    peptide2 = np.array(peptide2)
                    site1 = np.array(site1)
                    site2 = np.array(site2)
                    pep1 = peptide1[0]
                    s1 = site1[0]
                    pep2 = peptide2[0]
                    s2 = site2[0]
                    msms_arrary = peptide_msms_encoding(data2, pep1, pep2).ravel()
                    pep1_np, pep1_len = peptide_onehot_encoding(pep1, s1)
                    pep2_np, pep2_len = peptide_onehot_encoding(pep2, s2)
                    onehot_arrary = np.row_stack((pep1_np, pep2_np))
                    num_id = str(num).zfill(7)
                    if (len(pep1) < 50) and (len(pep2) < 50):
                        np.save(f'{output}_msms/{num_id}_{com_peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', msms_arrary)
                        np.save(f'{output}_onehot/{num_id}_{com_peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', onehot_arrary)
                    index_list = []
                    index_list.append(i)
                    peptide = peptide_list[i]
            num = num + 1
            data2 = data1.iloc[index_list]
            combine_peptide = np.array(data2['combine_peptide'])
            com_peptide = combine_peptide[0]
            peptide1 = data2['peptide1']
            peptide2 = data2['peptide2']
            site1 = data2['site1']
            site2 = data2['site2']
            peptide1 = np.array(peptide1)
            peptide2 = np.array(peptide2)
            site1 = np.array(site1)
            site2 = np.array(site2)
            pep1 = peptide1[0]
            s1 = site1[0]
            pep2 = peptide2[0]
            s2 = site2[0]
            msms_arrary = peptide_msms_encoding(data2, pep1, pep2).ravel()
            pep1_np, pep1_len = peptide_onehot_encoding(pep1, s1)
            pep2_np, pep2_len = peptide_onehot_encoding(pep2, s2)
            onehot_arrary = np.row_stack((pep1_np, pep2_np))
            num_id = str(num).zfill(7)
            if (len(pep1) < 50) and (len(pep2) < 50):
                np.save(f'{output}_msms/{num_id}_{com_peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', msms_arrary)
                np.save(f'{output}_onehot/{num_id}_{com_peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', onehot_arrary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode noncleavable crosslink MS/MS data.')
    parser.add_argument('--filename', type=str, default='MSMS_train',
                        help='Base filename inside ./data (without extension).')
    args = parser.parse_args()
    encoding(f'./data/{args.filename}.csv', f'./data/{args.filename}')
  
