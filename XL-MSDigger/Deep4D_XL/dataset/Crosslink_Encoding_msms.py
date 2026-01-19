import numpy as np
import pandas as pd
import os
import shutil
import argparse
from Deep4D_XL.dataset.aa_onehot import onehot
from Deep4D_XL.dataset.mass_cal import mz_cal as m
                                              
def get_args():             
    parser = argparse.ArgumentParser(description='Train the transformer on peptide and ccs')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--type', type=str, default='DeepDIA')
    parser.add_argument('--label', type=int, default=1)
    parser.add_argument('--maxcharge', type=int, default=5)
    parser.add_argument('--slice', type=int, default=1)
    return parser.parse_args()

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
    if 'U' in peptide:
        peptide = peptide.replace('U', '')
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

def encoding_with_label(input):
    data = pd.read_csv(input)
    output = input.rsplit('/', 1)[0] + '/' + 'feature_encoding'
    if os.path.exists(output):
        shutil.rmtree(output)
    os.mkdir(output)
    os.mkdir(f'{output}/msms')
    os.mkdir(f'{output}/onehot')
    charge_list = list(set(data['charge']))
    num = 0
    for z in charge_list:
        data1 = data[data['charge'] == z]
        if len(np.array(data1['combine_peptide'])) > 0:
            data1 = data1.sort_values('combine_peptide', ignore_index=True)             
            peptide_list = np.array(data1['combine_peptide'])
            peptide = peptide_list[0]
            index_list = []
            for i in range(len(peptide_list)):
                if peptide_list[i] == peptide:
                    index_list.append(i)
                else:
                    num = num + 1
                    if num % 200 == 0:
                        print(num)
                    data2 = data1.iloc[index_list]
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
                        np.save(f'{output}/msms/{num_id}_{peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', msms_arrary)
                        np.save(f'{output}/onehot/{num_id}_{peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', onehot_arrary)
                    index_list = []
                    index_list.append(i)
                    peptide = peptide_list[i]
            num = num + 1
            data2 = data1.iloc[index_list]
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
                np.save(f'{output}/msms/{num_id}_{peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', msms_arrary)
                np.save(f'{output}/onehot/{num_id}_{peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', onehot_arrary)
    return output

def encoding_with_label_candidate(input):
    data = pd.read_csv(input)
    output = input.rsplit('/', 1)[0] + '/' + 'candidate_feature_encoding'
    if os.path.exists(output):
        shutil.rmtree(output)
    os.mkdir(output)
    os.mkdir(f'{output}/msms')
    os.mkdir(f'{output}/onehot')
    num = 0
    spectrum_list = np.array(data['title'])
    spectrum = spectrum_list[0]
    index_list = []
    total_num = len(set(spectrum_list))
    for i in range(len(spectrum_list)):
        if spectrum_list[i] == spectrum:
            index_list.append(i)
        else:
            num = num + 1
            if num % 200 == 0:
                print(num)
            data2 = data.iloc[index_list]
            z = np.array(data2['Charge'])[0]
            order = np.array(data2['Order'])[0]
            peptide = np.array(data2['combine_peptide'])[0]
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
            lenth = len(str(total_num))
            num_id = str(num).zfill(lenth)
            if (len(pep1) < 50) and (len(pep2) < 50):
                np.save(f'{output}/msms/{num_id}&{order}&_{peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', msms_arrary)
                np.save(f'{output}/onehot/{num_id}&{order}&_{peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', onehot_arrary)
            index_list = []
            index_list.append(i)
            spectrum = spectrum_list[i]
    num = num + 1
    data2 = data.iloc[index_list]
    z = np.array(data2['Charge'])[0]
    order = np.array(data2['Order'])[0]
    peptide = np.array(data2['combine_peptide'])[0]
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
    lenth = len(str(total_num))
    num_id = str(num).zfill(lenth)
    if (len(pep1) < 50) and (len(pep2) < 50):
        np.save(f'{output}/msms/{num_id}&{order}&_{peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', msms_arrary)
        np.save(f'{output}/onehot/{num_id}&{order}&_{peptide}_${pep1_len}$@{pep2_len}@#{z}#.npy', onehot_arrary)
    return output

def encoding_without_label(input, output, maxcharge, slice=1):                    
    data_total = pd.read_csv(input)
    name = list(data_total)
    total_lenth = len(list(data_total['combine_peptide']))
    data_total = np.array(data_total)
    offset = 0
    a = m()
    for slice_num in range(slice):
        print(f'{slice_num}/{slice} slices in encoding..........')
        if slice_num != (slice - 1):
            lenth1 = total_lenth // slice
        else:
            lenth1 = total_lenth // slice + total_lenth % slice
        data = data_total[offset:(offset + lenth1), :]
        offset = offset + lenth1
        data = pd.DataFrame(data, columns=name)
        os.mkdir(f'{output}_slice{slice_num}_onehot')
        charge_list = list(range(2, (maxcharge + 1)))                                               
        nn = 0
        lenth = len(str(len(data['combine_peptide'])))
        for z in charge_list:
            data1 = data[data['charge'] == z]
            peptide = np.array(data1['combine_peptide'])
                                          
            peptide1 = np.array(data1['peptide1'])
            peptide2 = np.array(data1['peptide2'])
            site1 = np.array(data1['site1'])
            site2 = np.array(data1['site2'])
            charge = np.array(data1['charge'])
            peptide_num = len(peptide1)
            for h in range(peptide_num):
                pep = peptide[h]
                pep1 = peptide1[h]
                pep2 = peptide2[h]
                s1 = site1[h]
                s2 = site2[h]
                peptide_charge = charge[h]
                             
                if (len(pep1) < 50) and (len(pep2) < 50):
                    nn = nn + 1
                    pep1_np, pep1_len = peptide_onehot_encoding(pep1, s1)
                    pep2_np, pep2_len = peptide_onehot_encoding(pep2, s2)
                    mz = a.crosslink_peptide_m_z(pep, peptide_charge, 'DSS')
                    onehot_arrary = np.row_stack((pep1_np, pep2_np))
                    num_id = str(nn).zfill(lenth)
                    np.save(f'{output}_slice{slice_num}_onehot/{num_id}%{mz}%_{pep}_${pep1_len}$@{pep2_len}@#{peptide_charge}#.npy',
                            onehot_arrary)                                                   

                                      
if __name__ == '__main__':
    encoding('./data/test.csv','./data/test')
