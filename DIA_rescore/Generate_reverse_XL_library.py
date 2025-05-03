import numpy as np
import pandas as pd

def reverse_peptide(input_str): #####保留最后一位氨基酸，反转之前的肽段作为decoy peptide。
    reversed_str = input_str[::-1]
    return reversed_str[1:] + input_str[-1]

def reverse_peptide_site(peptide, site): ####计算反转序列之后交联位点在第几个氨基酸上，这里的site是python的index+1
    if int(site) == len(peptide):
        r_site = int(site)
    else:
        r_site = len(peptide) - int(site)
    return r_site

def combine_peptide(pep1, pep2, site1, site2, z): ###把两条交联肽合并表达，每个交联位点后面加U，两个交联肽用X连接,最后一位加上电荷
    pep1 = list(pep1)
    pep2 = list(pep2)
    pep1.insert(site1, 'U')
    pep2.insert(site2, 'U')
    pep1 = ''.join(pep1)
    pep2 = ''.join(pep2)
    pep = pep1 + 'X' + pep2
    pep_z = pep1 + 'X' + pep2 + str(z)
    return pep, pep_z

def generate_decoy_library(library):  ###生成三种decoy的肽：TD,DT,DD
    data = library
    peptide1_list, peptide2_list, charge_list, site1_list, site2_list = data['peptide1'], data['peptide2'], data['charge'], data['site1'], data['site2']
    m_z_list = data['m_z']
    peptide1_list_r, peptide2_list_r, site1_list_r, site2_list_r, combine_pep, combine_pep_z, z_list, type, m_z_list_r = [], [], [], [], [], [], [], [], []
    target_list = set(list(peptide1_list) + list(peptide2_list))
    for i in range(len(peptide1_list)):
        m_z = m_z_list[i]
        pep1 = reverse_peptide(peptide1_list[i])
        pep2 = peptide2_list[i]
        s1 = reverse_peptide_site(peptide1_list[i], site1_list[i])
        s2 = site2_list[i]
        z = charge_list[i]
        com_pep, com_pep_z = combine_peptide(pep1, pep2, s1, s2, z)
        if pep1 not in target_list:
            peptide1_list_r.append(pep1)
            peptide2_list_r.append(pep2)
            z_list.append(z)
            site1_list_r.append(s1)
            site2_list_r.append(s2)
            combine_pep.append(com_pep)
            combine_pep_z.append(com_pep_z)
            type.append('DT')
            m_z_list_r.append(m_z)
        pep2 = reverse_peptide(peptide2_list[i])
        pep1 = peptide1_list[i]
        s2 = reverse_peptide_site(peptide2_list[i], site2_list[i])
        s1 = site1_list[i]
        z = charge_list[i]
        com_pep, com_pep_z = combine_peptide(pep1, pep2, s1, s2, z)
        if pep2 not in target_list:
            peptide1_list_r.append(pep1)
            peptide2_list_r.append(pep2)
            z_list.append(z)
            site1_list_r.append(s1)
            site2_list_r.append(s2)
            combine_pep.append(com_pep)
            combine_pep_z.append(com_pep_z)
            type.append('TD')
            m_z_list_r.append(m_z)
        pep1 = reverse_peptide(peptide1_list[i])
        pep2 = reverse_peptide(peptide2_list[i])
        s1 = reverse_peptide_site(peptide1_list[i], site1_list[i])
        s2 = reverse_peptide_site(peptide2_list[i], site2_list[i])
        z = charge_list[i]
        com_pep, com_pep_z = combine_peptide(pep1,pep2,s1,s2,z)
        if (pep1 not in target_list) and (pep2 not in target_list):
            peptide1_list_r.append(pep1)
            peptide2_list_r.append(pep2)
            z_list.append(z)
            site1_list_r.append(s1)
            site2_list_r.append(s2)
            combine_pep.append(com_pep)
            combine_pep_z.append(com_pep_z)
            type.append('DD')
            m_z_list_r.append(m_z)
    data1 = pd.DataFrame()
    data1['peptide1'] = peptide1_list_r
    data1['peptide2'] = peptide2_list_r
    data1['charge'] = z_list
    data1['site1'] = site1_list_r
    data1['site2'] = site2_list_r
    data1['combine_peptide'] = combine_pep
    data1['combine_peptide_z'] = combine_pep_z
    data1['m_z'] = m_z_list_r
    data1['type'] = type
    data2 = data1[data1['type'] == 'DD']
    data3 = data1[data1['type'] != 'DD']
    return data2, data3


if __name__ == '__main__':
    # for i in range(18, 39):
    #     data =pd.read_csv(f'H:/20230708_new/dia_rescore/{i}/{i}_crosslink_filter_precursor.csv')
    #     data1, data2 = generate_decoy_library(data)
    #     data1.to_csv(f'H:/20230708_new/dia_rescore/{i}/{i}_crosslink_filter_precursor_DD.csv', index = False)
    #     data2.to_csv(f'H:/20230708_new/dia_rescore/{i}/{i}_crosslink_filter_precursor_TD.csv', index = False)

    # x = 21
    # data = pd.read_csv(f'F:/18_19_crosslink/18_19_crosslink_filter_precursor.csv')
    # data1, data2 = generate_decoy_library(data)
    # data1.to_csv(f'F:/18_19_crosslink/18_19_crosslink_filter_precursor_DD.csv', index=False)
    # data2.to_csv(f'F:/18_19_crosslink/18_19_crosslink_filter_precursor_TD.csv', index=False)

    data = pd.read_csv(f'G:/20230708_new/dia_rescore/100_low_protein/20_protein_1/20_protein_1_intra_peptide.csv')
    data1, data2 = generate_decoy_library(data)
    data1.to_csv(f'G:/20230708_new/dia_rescore/100_low_protein/20_protein_1/20_protein_1_intra_peptide_DD.csv', index=False)
    data2.to_csv(f'G:/20230708_new/dia_rescore/100_low_protein/20_protein_1/20_protein_1_intra_peptide_TD.csv', index=False)

    # data = pd.read_csv('E:/项目/Deep4D_cl/dia_Rescore/data/RL7A_RL34_cross-linked_peptides.csv')
    # data1, data2 = generate_decoy_library(data)
    # data1.to_csv('E:/项目/Deep4D_cl/dia_Rescore/data/RL7A_RL34_cross-linked_peptides_reverse_DD.csv', index=False)
    # data2.to_csv('E:/项目/Deep4D_cl/dia_Rescore/data/RL7A_RL34_cross-linked_peptides_reverse_TD.csv', index=False)