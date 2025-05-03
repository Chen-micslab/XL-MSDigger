import pandas as pd
import numpy as np

def calculate_msms_num(msms, target, tolerance):
    index = np.abs((msms - target)/msms) < tolerance
    result = msms[index]
    count = len(result)
    if count == 0:
        count = 1
    return count

def calculate_fragment_weight(library, tolerance=0.00002):  #####质量偏差20ppm
    library1 = library[library['type'] != 'test']
    fragment_mz1 = np.array(library1['Fragment_m_z_calculation'])
    fragment_mz = np.array(library['Fragment_m_z_calculation'])
    count_list = []
    for mz in fragment_mz:
        count = calculate_msms_num(fragment_mz1, mz, tolerance)
        count_list.append(count)
    library['fragment_count'] = count_list
    return library

def combine_normal_library(TT_lib, TD_lib, DD_lib): ####合并TT,DD,TD三种类型的normal library
    # xlist = ['combine_peptide', 'combine_peptide_z', 'charge', 'm_z', 'rt', 'k0', 'Fragment_charge', 'Fragment_type',
    #                  'Fragment_num', 'Neutral_loss', 'Fragment_intensity', 'peptide1', 'peptide2', 'site1', 'site2', 'Fragment_m_z_calculation']
    xlist = ['combine_peptide', 'combine_peptide_z', 'charge', 'm_z', 'rt', 'k0', 'Fragment_charge', 'Fragment_type',
                     'Fragment_num', 'Neutral_loss', 'Fragment_intensity', 'Fragment_m_z_calculation']

    TT_lib = TT_lib[xlist].copy()
    TD_lib = TD_lib[xlist].copy()
    DD_lib = DD_lib[xlist].copy()
    len_TT = len(TT_lib['combine_peptide'])
    x = ['TT']*len_TT
    TT_lib['type'] = x
    len_TD = len(TD_lib['combine_peptide'])
    x = ['TD']*len_TD
    TD_lib['type'] = x
    len_DD = len(DD_lib['combine_peptide'])
    x = ['DD']*len_DD
    DD_lib['type'] = x
    lib = pd.concat([TT_lib, TD_lib, DD_lib])
    return lib

def combine_DIANN_library(TT_lib, TD_lib, DD_lib):####合并TT,DD,TD三种类型的DIANN library
    lib = pd.concat([TT_lib, TD_lib, DD_lib])
    return lib

def combine_test_normal_library(normal_lib, test_lib): ####合并normal的实验library和test library
    # name_list = ['combine_peptide', 'combine_peptide_z', 'charge', 'm_z', 'rt', 'k0', 'Fragment_charge',
    #              'Fragment_type', 'Fragment_num', 'Neutral_loss', 'Fragment_intensity', 'Fragment_m_z_calculation']
    # test_lib = test_lib[name_list]
    pep_list = list(set(list(normal_lib['combine_peptide'])))
    test_lib = test_lib[~test_lib['combine_peptide'].isin(pep_list)]
    if 'fragment_count' in normal_lib.columns:
        normal_lib.drop('fragment_count', axis=1, inplace=True)
    len_test = len(test_lib['combine_peptide'])
    if 'type' not in test_lib.columns:
        x = ['TT']*len_test
        test_lib['type'] = x
    x = ['test']*len_test
    test_lib['type2'] = x
    len_test = len(normal_lib['combine_peptide'])
    x = ['exp']*len_test
    normal_lib['type2'] = x
    lib = pd.concat([normal_lib, test_lib])
    # lib1 = calculate_fragment_weight(lib)
    return lib

def combine_test_DIANN_library(DIANN_lib, test_lib): ####合并DIANN的实验library和test library
    pep_list = list(set(list(DIANN_lib['ModifiedPeptide'])))
    test_lib = test_lib[~test_lib['ModifiedPeptide'].isin(pep_list)]
    c = pd.concat([test_lib, DIANN_lib])
    return c


if __name__ == '__main__':

    # x = 'total'
    # data = pd.read_csv(f'H:/20230708_new/dia_rescore/{x}/{x}_crosslink_precursor_total_DIANN_library_exp.csv')
    # data1 = pd.read_csv(f'H:/20230708_new/dia_rescore/100_low_protein/20_protein_2/20_protein_2_intra_peptide_DIANN_library1.csv')
    # data3 = combine_test_DIANN_library(data, data1)
    # data3.to_csv(f'H:/20230708_new/dia_rescore/100_low_protein/20_protein_2/{x}_crosslink_precursor_total_DIANN_library_exp_20protein_2_1.csv',
    #     index=False)
    #
    # data = pd.read_csv(f'H:/20230708_new/dia_rescore/{x}/{x}_crosslink_precursor_total_normal_library_exp.csv')
    # data1 = pd.read_csv(f'H:/20230708_new/dia_rescore/ribosome/40S_PPI_inter_peptide_normal_library.csv')
    # data3 = combine_test_normal_library(data, data1)
    # data3.to_csv(f'H:/20230708_new/dia_rescore/ribosome/{x}_crosslink_precursor_total_normal_library_exp_40S_PPI.csv',
    #     index=False)

    data1 = pd.read_csv(f'G:/20230708_new/dia_rescore/100protein/20protein_5/20_protein_5_intra_peptide_normal_library.csv')
    data2 = pd.read_csv(f'G:/20230708_new/dia_rescore/100protein/20protein_5/20_protein_5_intra_peptide_TD_normal_library.csv')
    data3 = pd.read_csv(f'G:/20230708_new/dia_rescore/100protein/20protein_5/20_protein_5_intra_peptide_DD_normal_library.csv')
    data_normal = combine_normal_library(data1, data2, data3)
    data = pd.read_csv(f'G:/20230708_new/dia_rescore/total/total_crosslink_precursor_total_normal_library_exp.csv')
    data3 = combine_test_normal_library(data, data_normal)
    data3.to_csv(f'G:/20230708_new/dia_rescore/100protein/20protein_5/total_crosslink_precursor_total_normal_library_20protein_5.csv',
        index=False)


    data1 = pd.read_csv(f'G:/20230708_new/dia_rescore/100protein/20protein_5/20_protein_5_intra_peptide_DIANN_library.csv')
    data2 = pd.read_csv(f'G:/20230708_new/dia_rescore/100protein/20protein_5/20_protein_5_intra_peptide_TD_DIANN_library.csv')
    data3 = pd.read_csv(f'G:/20230708_new/dia_rescore/100protein/20protein_5/20_protein_5_intra_peptide_DD_DIANN_library.csv')
    data_normal = combine_DIANN_library(data1, data2, data3)
    data = pd.read_csv(f'G:/20230708_new/dia_rescore/total/total_crosslink_precursor_total_DIANN_library_exp.csv')
    data3 = combine_test_DIANN_library(data, data_normal)
    data3.to_csv(f'G:/20230708_new/dia_rescore/100protein/20protein_5/total_crosslink_precursor_total_DIANN_library_20protein_5.csv',
        index=False)

    # for i in range(1,5):
    # x = 'total'
    # data = pd.read_csv(f'F:/20230708_new/dia_rescore/{x}/{x}_crosslink_precursor_total_DIANN_library_exp.csv')
    # # data1 = pd.read_csv(f'F:/20230708_new/dia_rescore/ribosome/new/40S_intra_peptide_DIANN_library.csv')
    # data3 = combine_test_DIANN_library(data, data_DIANN)
    # data3.to_csv(
    #     f'F:/20230708_new/dia_rescore/ribosome/new/{x}_crosslink_precursor_total_DIANN_library_40S_intra_peptide.csv',
    #     index=False)

    # data = pd.read_csv(f'F:/20230708_new/dia_rescore/{x}/{x}_crosslink_precursor_total_normal_library_exp.csv')
    # # data1 = pd.read_csv(f'F:/20230708_new/dia_rescore/ribosome/new/40S_intra_peptide_normal_library.csv')
    # data3 = combine_test_normal_library(data, data_normal)
    # data3.to_csv(f'F:/20230708_new/dia_rescore/ribosome/new/{x}_crosslink_precursor_total_normal_library_40S_intra_peptide1.csv',
    #     index=False)
