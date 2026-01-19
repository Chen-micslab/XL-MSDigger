from DIA_rescore.feature_detection import feature_detect
from DIA_rescore.DIA_rescore import dnn_rescore
from DIA_rescore.match_to_protein import run_for_rescore_results
import pandas as pd
import argparse

def get_args():             
    parser = argparse.ArgumentParser(description='XL-MSDigger DIA')
    parser.add_argument('--diann_report', type=str, default='/data/moran/XL-MSDigger-main/test_dia_2/report.tsv')
    parser.add_argument('--DIA_library', type=str, default='/data/moran/XL-MSDigger-main/test_dia_2/experimental_library_with_aim_normal_lib.csv')
    parser.add_argument('--fasta_dir', type=str, default='/data/moran/XL-MSDigger-main/test_dia_2/human_reviewed.fasta') 
    parser.add_argument('--peptide_protein_list', type=str, default='/data/moran/XL-MSDigger-main/test_dia_2/test_PPI_origin_inter_peptide&protein.csv')
    return parser.parse_args()

def run():
    args = get_args()          
    feature_det = feature_detect()
    print('feature extraction...')
    diann_feature = feature_det.total_process(args.diann_report, args.DIA_library)
    diann_feature_dir = args.diann_report.split('.tsv')[0] + '_feature.csv'
    diann_feature.to_csv(diann_feature_dir, index = False)
    print('rescoring...')
    rescore = dnn_rescore()
    rescore_results, rescore_test_results = rescore.run(diann_feature_dir)
    rescore_test_results_1 = run_for_rescore_results(rescore_test_results, args.peptide_protein_list, args.fasta_dir)
    rescore_test_results_dir = args.diann_report.split('.tsv')[0] + "target_rescore_results.csv"
    rescore_test_results_1.to_csv(rescore_test_results_dir, index=False)
if __name__ == '__main__':
    run()
