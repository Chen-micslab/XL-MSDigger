import argparse
from Preprocess.pLink2_process import plink2_with_DA_mgf
from Deep4D_XL.Finetune import train_model
from DDA_rescore.predict_feature import generate_feature
from DDA_rescore.DDA_rescore import Rescore_SVM, Rescore_DNN
import os
import pandas as pd
import time 

def get_args():  ##设置需要传入的参数
    parser = argparse.ArgumentParser(description='Train the transformer on peptide and ccs')
    parser.add_argument('--plinkfile', type=str, default='/data/moran/XL-MSDigger-main/test_dda_rescore/test')
    parser.add_argument('--mgf_dir', type=str, default='/data/moran/XL-MSDigger-main/test_dda_rescore/test.mgf')
    parser.add_argument('--finetune', type=int, default=1)
    parser.add_argument('--rescore_model', type=str, default='dnn')
    parser.add_argument('--rescore_fdr', type=float, default=0.01)
    parser.add_argument('--rescore_batch_size', type=float, default=200)
    parser.add_argument('--rescore_vali_rate', type=float, default=0.1)
    parser.add_argument('--rescore_model_parameter', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()

def run():
    args = get_args()  ##生成参数列表
    plink = plink2_with_DA_mgf()
    msms_dir, ccs_dir, rt_dir, candidate_msms_dir, candidate_rtccs_dir = plink.process(args.plinkfile, args.mgf_dir)
    if args.finetune == 1:
        print('Finetuning the model......')
        train = train_model()
        msms_paradir, ccs_paradir, rt_paradir = train.finetune(msms_dir, ccs_dir, rt_dir)
    else:
        print('No Finetuning')
        base_dir = os.path.dirname(os.path.abspath(__file__))
        msms_paradir = os.path.join(base_dir, 'Deep4D_XL', 'checkpoint', 'msms.pth')
        ccs_paradir = os.path.join(base_dir, 'Deep4D_XL', 'checkpoint', 'ccs.pth')
        rt_paradir = os.path.join(base_dir, 'Deep4D_XL', 'checkpoint', 'rt.pth')
    generate = generate_feature()
    candidate_feature = generate.run(candidate_msms_dir, candidate_rtccs_dir, msms_paradir, ccs_paradir, rt_paradir)
    candidate_feature_dir = candidate_rtccs_dir.split('.csv')[0] + '_candidate_feature.csv'
    candidate_feature.to_csv(candidate_feature_dir, index=False)
    candidate_feature = pd.read_csv(candidate_feature_dir)
    if args.rescore_model == 'svm':
        print('SVM will be used for rescoring')
        dda_rescore = Rescore_SVM()
        rescore_results = dda_rescore.run(candidate_feature)
    elif args.rescore_model == 'dnn':
        print('DNN will be used for rescoring')
        dda_rescore= Rescore_DNN()
        rescore_results = dda_rescore.run(args, candidate_feature, candidate_rtccs_dir)
    else:
        print('Invalid model name entered, SVM will be used for rescoring')
        dda_rescore = Rescore_SVM()
        rescore_results = dda_rescore.run(candidate_feature)
    rescore_results = rescore_results[rescore_results['FDR'] <= args.rescore_fdr]
    rescore_results = rescore_results[rescore_results['Target_Decoy'] == 2]
    rescore_results_dir = candidate_rtccs_dir.split('.csv')[0] + '_rescore_inter_results.csv'
    rescore_results.to_csv(rescore_results_dir, index=False)
    os.remove(msms_dir)
    os.remove(ccs_dir)
    os.remove(rt_dir)
    os.remove(candidate_msms_dir)
    os.remove(candidate_rtccs_dir)
    folder_path = os.path.dirname(args.mgf_dir)
    import shutil
    candidate_feature_encoding_dir = folder_path + '/candidate_feature_encoding'
    checkpoint_dir = folder_path + '/checkpoint'
    feature_encoding_dir = folder_path + '/feature_encoding'
    shutil.rmtree(candidate_feature_encoding_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    shutil.rmtree(feature_encoding_dir, ignore_errors=True)

if __name__ == '__main__':
    run()