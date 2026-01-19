import numpy as np
import pandas as pd
from .mass_cal import mz_cal as m
import time

def merge_information(msms_dir, ccs_dir, rt_dir):
    msms_matrix = np.array(pd.read_csv(msms_dir))
    ccs_matrix = np.array(pd.read_csv(ccs_dir))
    rt_matrix = np.array(pd.read_csv(rt_dir))
    m_z = msms_matrix[:, 2]
    charge = msms_matrix[:, 1]
    ccs = ccs_matrix[:, 2]
    rt = rt_matrix[:, 2]
    k0 = []
    a = m()
    for i in range(len(ccs)):
        k0.append(a.calculate_k0(m_z[i], charge[i], ccs[i]))
    k0 = np.array(k0)
    rt = np.expand_dims(np.array(rt), axis=1)
    k0 = np.expand_dims(np.array(k0), axis=1)
    data = np.column_stack((k0, rt, msms_matrix))
    return data

def choose_top_n(data, n):
    name = list(data)
    inten = np.array(data['FI.Intensity'])
    data2 = data.copy()
    data2['FI.Intensity'] = data['FI.Intensity']/np.max(list(data['FI.Intensity']))
    data1 = np.array(data2)
    if len(data1) > n:
        data2 = data1[np.argsort(-inten)]
        data3 = data2[:n, :]
        data3 = pd.DataFrame(data3, columns=name)
        return data3
    else:
        return data

def extract_from_cl_peptide(cl_peptide):                                
    id1 = cl_peptide.find('X')
    pep1 = cl_peptide[:id1]
    pep2 = cl_peptide[(id1 + 1):]
    site1 = pep1.find('U')
    site2 = pep2.find('U')
    peptide1 = pep1.replace('U', '')
    peptide2 = pep2.replace('U', '')
    return peptide1, peptide2, site1, site2

def filter_msms(peptide):
    peptide1, peptide2, site1, site2 = extract_from_cl_peptide(peptide)
    len1, len2 = len(peptide1), len(peptide2)
    site1_y = len1 - site1 + 1
    site2_y = len2 - site2 + 1
    a = np.zeros((49, 20))
    a[:(site1 - 1), 1:5] = -1
    a[:(site2 - 1), 11:15] = -1
    a[:(site1_y - 1), 6:10] = -1
    a[:(site2_y - 1), 16:20] = -1
    a[(site1 - 1):, 0] = -1
    a[(site2 - 1):, 10] = -1
    a[(site1_y - 1):, 5] = -1
    a[(site2_y - 1):, 15] = -1
    a[(len1 - 1):, :10] = -1
    a[(len2 - 1):, 10:] = -1
    a = a.ravel()
    return a

def generate_4d_library(data):                                                           
    peptide_list, peptide_z_list, charge_list, m_z_list, rt_list, ccs_list, ion_charge, ion_type, ion_num, ion_loss, ion_inten = [], [], [], [], [], [], [], [], [], [], []
    peptide1_list, peptide2_list, site1_list, site2_list = [], [], [], []
    peptide = data[:, 2]
    charge = data[:, 3]
    pep_m_z = data[:, 4]
                          
    ccs = data[:, 0]
    rt = data[:, 1]
    pre_msms = data[:, 5:]
    aa = time.perf_counter()
    for j in range(len(peptide)):
        if len(pre_msms[j, :][(pre_msms[j, :] > 0)]) > 2:                      
            pep = peptide[j]
            pep_len = pep.find('X') - 1                                                              
            filter_array = filter_msms(pep)
            inten_list = pre_msms[j, :]
            inten_list[filter_array==-1]=-1
            inten_list1 = sorted(inten_list, reverse=True)
            inten_list = inten_list/inten_list1[0]
            limit = inten_list1[12]/inten_list1[0]                  
            for i in range(pep_len * 20):
                if (inten_list[i] >= 0.01) and (inten_list[i] > limit):                   
                    a = i % 20
                    num = (i // 20) + 1
                    peptide_list.append(peptide[j])
                    peptide1, peptide2, site1, site2 = extract_from_cl_peptide(peptide[j])
                    peptide1_list.append(peptide1)
                    peptide2_list.append(peptide2)
                    site1_list.append(site1)
                    site2_list.append(site2)
                    charge_list.append(charge[j])
                    peptide_z_list.append(str(peptide[j])+str(charge[j]))
                    m_z_list.append(pep_m_z[j])
                    rt_list.append(rt[j])
                    ccs_list.append(ccs[j])
                    ion_num.append(num)
                    ion_inten.append(inten_list[i])
                    if a == 0:
                        ion_charge.append(1)
                        ion_type.append('1b')
                        ion_loss.append('Noloss')
                    elif a == 1:
                        ion_charge.append(2)
                        ion_type.append('1b')
                        ion_loss.append('Noloss')
                    elif a == 2:
                        ion_charge.append(3)
                        ion_type.append('1b')
                        ion_loss.append('Noloss')
                    elif a == 3:
                        ion_charge.append(4)
                        ion_type.append('1b')
                        ion_loss.append('Noloss')
                    elif a == 4:
                        ion_charge.append(5)
                        ion_type.append('1b')
                        ion_loss.append('Noloss')
                    elif a == 5:
                        ion_charge.append(1)
                        ion_type.append('1y')
                        ion_loss.append('Noloss')
                    elif a == 6:
                        ion_charge.append(2)
                        ion_type.append('1y')
                        ion_loss.append('Noloss')
                    elif a == 7:
                        ion_charge.append(3)
                        ion_type.append('1y')
                        ion_loss.append('Noloss')
                    elif a == 8:
                        ion_charge.append(4)
                        ion_type.append('1y')
                        ion_loss.append('Noloss')
                    elif a == 9:
                        ion_charge.append(5)
                        ion_type.append('1y')
                        ion_loss.append('Noloss')
                    elif a == 10:
                        ion_charge.append(1)
                        ion_type.append('2b')
                        ion_loss.append('Noloss')
                    elif a == 11:
                        ion_charge.append(2)
                        ion_type.append('2b')
                        ion_loss.append('Noloss')
                    elif a == 12:
                        ion_charge.append(3)
                        ion_type.append('2b')
                        ion_loss.append('Noloss')
                    elif a == 13:
                        ion_charge.append(4)
                        ion_type.append('2b')
                        ion_loss.append('Noloss')
                    elif a == 14:
                        ion_charge.append(5)
                        ion_type.append('2b')
                        ion_loss.append('Noloss')
                    elif a == 15:
                        ion_charge.append(1)
                        ion_type.append('2y')
                        ion_loss.append('Noloss')
                    elif a == 16:
                        ion_charge.append(2)
                        ion_type.append('2y')
                        ion_loss.append('Noloss')
                    elif a == 17:
                        ion_charge.append(3)
                        ion_type.append('2y')
                        ion_loss.append('Noloss')
                    elif a == 18:
                        ion_charge.append(4)
                        ion_type.append('2y')
                        ion_loss.append('Noloss')
                    elif a == 19:
                        ion_charge.append(5)
                        ion_type.append('2y')
                        ion_loss.append('Noloss')
    bb = time.perf_counter()
    print(bb-aa)
    data1 = pd.DataFrame()
    data1['combine_peptide'] = peptide_list
    data1['combine_peptide_z'] = peptide_z_list
    data1['charge'] = charge_list
    data1['m_z'] = m_z_list
    data1['rt'] = rt_list
    data1['k0'] = ccs_list
    data1['Fragment_charge'] = ion_charge
    data1['Fragment_type'] = ion_type
    data1['Fragment_num'] = ion_num
    data1['Neutral_loss'] = ion_loss
    data1['Fragment_intensity'] = ion_inten
    data1['peptide1'] = peptide1_list
    data1['peptide2'] = peptide2_list
    data1['site1'] = site1_list
    data1['site2'] = site2_list    
    data = np.array(data1)
    msms_mz = data[:, 2].copy()
    a = m()
    aa = time.perf_counter()
    for i in range(len(data[:, 0])):
        x = a.crosslink_peptide_msms_m_z(data[i, 0], 'DSS', data[i, 7], data[i, 8], data[i, 6], data[i, 9])
        x = round(x,5)
        msms_mz[i] = x
    bb = time.perf_counter()
    print(bb - aa)
    data1['Fragment_m_z_calculation'] = list(msms_mz)
                                       
                              
                                                        
                        
              
                               
                              
                         
                                                 
                                                 
                                                       
                             
                                               
                             
                              
                    
                            
                                               
    data3 = pd.DataFrame()
    data3['ModifiedPeptide'] = data1['combine_peptide']
    data3['PrecursorCharge'] = data1['charge']
    data3['PrecursorMz'] = data1['m_z']
    data3['FragmentCharge'] = data1['Fragment_charge']
    data3['ProductMz'] = data1['Fragment_m_z_calculation']
    data3['Tr_recalibrated'] = data1['rt']
    data3['IonMobility'] = data1['k0']
    data3['LibraryIntensity'] = data1['Fragment_intensity']
    return data1, data3                                               

def move_modification(peptide):
    peptide = peptide.replace('a', '')
    peptide = peptide.replace('e', 'M')
    peptide = peptide.replace('s', 'S')
    peptide = peptide.replace('t', 'T')
    peptide = peptide.replace('y', 'Y')
    return peptide


def modif_as_DIA_NN(peptide):                                   
    peptide = peptide.replace('C', 'C(UniMod:4)')
    peptide = peptide.replace('s', 'S(UniMod:21)')
    peptide = peptide.replace('t', 'T(UniMod:21)')
    peptide = peptide.replace('y', 'Y(UniMod:21)')
    peptide = peptide.replace('e', 'M(UniMod:35)')
    peptide = peptide.replace('a', '(UniMod:1)')
                                                   
                                                   
                                                   
    return peptide

def modif_diann_to_normal(peptide):
    peptide = peptide.replace('(UniMod:4)','')
    peptide = peptide.replace('S(UniMod:21)', 's')
    peptide = peptide.replace('T(UniMod:21)', 't')
    peptide = peptide.replace('Y(UniMod:21)', 'y')
    peptide = peptide.replace('M(UniMod:35)', 'e')
    peptide = peptide.replace('(UniMod:1)', 'a')
    peptide = peptide.replace('(UniMod:26)', 'a')
    peptide = peptide.replace('(UniMod:27)', 'a')
    peptide = peptide.replace('(UniMod:28)', 'a')
    return peptide

def change_to_DIA_NN(data):
    name = ['PeptideSequence', 'PrecursorCharge', 'PrecursorMz', 'Tr_recalibrated', 'IonMobility',
            'FragmentCharge', 'FragmentType', 'FragmentSeriesNumber', 'FragmentLossType', 'LibraryIntensity',
            'ProductMz']
    data[:, 8][data[:, 8] == 'Noloss'] = 'noloss'
    mod_pep = data[:, 0].copy()
    msms_mz = data[:, 2].copy()
    data[:, 2] = np.array(data[:, 2], dtype=float).round(5)
    data[:, 3] = np.array(data[:, 3], dtype=float).round(4)
    data[:, 4] = np.array(data[:, 4], dtype=float).round(4)
    data[:, 9] = np.array(data[:, 9], dtype=float).round(4)
    for i in range(len(mod_pep)):
        mod_pep[i] = modif_as_DIA_NN(data[i, 0])
    data = pd.DataFrame(data, columns=name)
    data['ModifiedPeptide'] = list(mod_pep)
    return data


if __name__ == '__main__':
    pass
