from DIA_rescore.feature_detection import feature_detect
from DIA_rescore.construct_library import construct_library
from DIA_rescore.DIA_rescore import dnn_rescore
import pandas as pd
import argparse

def get_args():  ##设置需要传入的参数
    parser = argparse.ArgumentParser(description='XL-MSDigger DIA')
    parser.add_argument('--diann_report', type=str, default='/data/moran/dia_test/report.tsv')
    parser.add_argument('--DIA_library', type=str, default='/data/moran/dia_test/total_crosslink_precursor_total_normal_library_HSP60_PPI_inter_peptide.csv')
    return parser.parse_args()

def run():
    args = get_args()  ##生成参数列表
    feature_det = feature_detect()
    diann_feature = feature_det.total_process(args.diann_report, args.DIA_library)
    diann_feature_dir = args.diann_report.split('.tsv')[0] + '_feature.csv'
    diann_feature.to_csv(diann_feature_dir, index = False)
    rescore = dnn_rescore()
    rescore_results, rescore_test_results = rescore.run(diann_feature_dir)

if __name__ == '__main__':
    run()
