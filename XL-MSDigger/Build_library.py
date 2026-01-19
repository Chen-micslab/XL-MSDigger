from DIA_rescore.feature_detection import feature_detect
from DIA_rescore.DIA_rescore import dnn_rescore
import pandas as pd
import argparse
from DIA_rescore.Generate_reverse_XL_library import generate_decoy_library
from DIA_rescore.combine_library import combine_normal_library, combine_DIANN_library, combine_test_normal_library, combine_test_DIANN_library
from Deep4D_XL.predict_4d_XL_library import generate_predicted_library
from DIA_rescore.Generate_potential_XL_peptide import generate_potential_XL_peptide
import os

def get_args():             
    parser = argparse.ArgumentParser(description='XL-MSDigger DIA')
    parser.add_argument('--experiment_library', type=str, default='/data/moran/XL-MSDigger-main/test_dia_1/experimental_library.csv')
    parser.add_argument('--aim_protein', type=str, default='/data/moran/XL-MSDigger-main/test_dia_1/test_PPI.csv')                              
    parser.add_argument('--aim_type', type=int, default=1)                         
    parser.add_argument('--fasta_dir', type=str, default='/data/moran/XL-MSDigger-main/test_dia_1/human_reviewed.fasta')        
    parser.add_argument('--msms_param_dir', type=str, default='Deep4D_XL/checkpoint/msms.pth')  
    parser.add_argument('--rt_param_dir', type=str, default='Deep4D_XL/checkpoint/rt.pth')  
    parser.add_argument('--ccs_param_dir', type=str, default='Deep4D_XL/checkpoint/ccs.pth')  
    parser.add_argument('--maxcharge', type=int, default=5) 
    parser.add_argument('--slice', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1000) 
    return parser.parse_args()

def transfer_to_DIANN_format(data):
    data2 = pd.DataFrame()
    data2['ModifiedPeptide'] = data['combine_peptide']
    data2['PrecursorCharge'] = data['charge']
    data2['PrecursorMz'] = data['m_z']
    data2['FragmentCharge'] = data['Fragment_charge']
    data2['ProductMz'] = data['Fragment_m_z_calculation']
    data2['Tr_recalibrated'] = data['rt']
    data2['IonMobility'] = data['k0']
    data2['LibraryIntensity'] = data['Fragment_intensity']
    return data2

def run():
    args = get_args()          
    exp_TT_normal_lib = pd.read_csv(args.experiment_library)
    print('Generating experiment decoy peptide...')
    exp_DD_pep, exp_TD_pep = generate_decoy_library(exp_TT_normal_lib)
    exp_TD_pep_dir = args.experiment_library.split('.csv')[0] + '_TD.csv'
    exp_DD_pep_dir = args.experiment_library.split('.csv')[0] + '_DD.csv'
    exp_TD_pep.to_csv(exp_TD_pep_dir, index = False)
    exp_DD_pep.to_csv(exp_DD_pep_dir, index = False)
    print('Generating experiment decoy library...')
    exp_TD_normal_lib, exp_TD_diann_lib = generate_predicted_library(exp_TD_pep_dir, args.maxcharge, args.slice, args.batch_size, 
                                                                        args.msms_param_dir, args.rt_param_dir, args.ccs_param_dir)
    exp_DD_normal_lib, exp_DD_diann_lib = generate_predicted_library(exp_DD_pep_dir, args.maxcharge, args.slice, args.batch_size, 
                                                                        args.msms_param_dir, args.rt_param_dir, args.ccs_param_dir)
    os.remove(exp_TD_pep_dir)
    os.remove(exp_DD_pep_dir)
    print('Combining experiment library...')
    exp_total_normal_lib = combine_normal_library(exp_TT_normal_lib, exp_TD_normal_lib, exp_DD_normal_lib)
    exp_total_DIANN_lib = transfer_to_DIANN_format(exp_total_normal_lib)
    exp_total_normal_lib_dir = args.experiment_library.split('.csv')[0] + '_total_normal_lib.csv'
    exp_total_DIANN_lib_dir = args.experiment_library.split('.csv')[0] + '_total_DIANN_lib.csv'
    print('Saving experiment library...')
    exp_total_normal_lib.to_csv(exp_total_normal_lib_dir, index = False)
    exp_total_DIANN_lib.to_csv(exp_total_DIANN_lib_dir, index = False)
    if args.aim_protein is not None:
        x = generate_potential_XL_peptide()
        if args.aim_type == 0:
            print('Generating intra aim library ...')
            aim_TT_pep, aim_TT_pep_dir = x.generate_intra_peptide_pair(args.aim_protein, args.fasta_dir)
        elif args.aim_type == 1:
            print('Generating inter aim library ...')
            aim_TT_pep, aim_TT_pep_dir = x.generate_inter_peptide_pair(args.aim_protein, args.fasta_dir)
        print('Generating decoy aim peptide ...')
        aim_DD_pep, aim_TD_pep = generate_decoy_library(aim_TT_pep)
        aim_TD_pep_dir = aim_TT_pep_dir.split('.csv')[0] + '_TD.csv'
        aim_DD_pep_dir = aim_TT_pep_dir.split('.csv')[0] + '_DD.csv'
        aim_TD_pep.to_csv(aim_TD_pep_dir, index = False)
        aim_DD_pep.to_csv(aim_DD_pep_dir, index = False)
        print('Generating predicted aim library ...')
        aim_TT_normal_lib, aim_TT_diann_lib = generate_predicted_library(aim_TT_pep_dir, args.maxcharge, args.slice, args.batch_size, 
                                                                            args.msms_param_dir, args.rt_param_dir, args.ccs_param_dir)       
        aim_TD_normal_lib, aim_TD_diann_lib = generate_predicted_library(aim_TD_pep_dir, args.maxcharge, args.slice, args.batch_size, 
                                                                            args.msms_param_dir, args.rt_param_dir, args.ccs_param_dir)
        aim_DD_normal_lib, aim_DD_diann_lib = generate_predicted_library(aim_DD_pep_dir, args.maxcharge, args.slice, args.batch_size, 
                                                                            args.msms_param_dir, args.rt_param_dir, args.ccs_param_dir)
        aim_total_normal_lib = combine_normal_library(aim_TT_normal_lib, aim_TD_normal_lib, aim_DD_normal_lib)
        aim_total_DIANN_lib = transfer_to_DIANN_format(aim_total_normal_lib)
        os.remove(aim_TT_pep_dir)
        os.remove(aim_TD_pep_dir)
        os.remove(aim_DD_pep_dir)
        print('Combining total library...')
        total_normal_lib = combine_test_normal_library(exp_total_normal_lib, aim_total_normal_lib)
        total_DIANN_lib = combine_test_DIANN_library(exp_total_DIANN_lib, aim_total_DIANN_lib)
        total_normal_lib_dir = args.experiment_library.split('.csv')[0] + '_with_aim_normal_lib.csv'
        total_DIANN_lib_dir = args.experiment_library.split('.csv')[0] + '_with_aim_DIANN_lib.csv'
        print('Saving total library...')
        total_normal_lib.to_csv(total_normal_lib_dir, index = False)
        total_DIANN_lib.to_csv(total_DIANN_lib_dir, index = False)

if __name__ == '__main__':
    run()