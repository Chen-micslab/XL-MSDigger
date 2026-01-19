import argparse
from Preprocess.plink_with_msconvert_mgf import plink_with_msconvert_mgf
from Deep4D_XL.Finetune_noccs import train_model
from DDA_rescore.predict_feature_noccs import generate_feature
from DDA_rescore.DDA_rescore_plink3 import Rescore_SVM, Rescore_DNN
from DDA_rescore.mzid_writer import build_mzid
import os
import pandas as pd
import time 

DEFAULT_MOD_INI = "/Users/moranchen/Documents/Project/Deep4D_XL/Review_data/data/mzidentML/modification.ini"

def get_args():             
    parser = argparse.ArgumentParser(description='Train the transformer on peptide and ccs')
    parser.add_argument('--plinkfile', type=str, default='/data/plinkfile')
    parser.add_argument('--mgf_dir', type=str, default='/data/plinkfile.mgf')
    parser.add_argument('--finetune', type=int, default=1)
    parser.add_argument('--rescore_model', type=str, default='dnn')
    parser.add_argument('--rescore_fdr', type=float, default=0.01)
    parser.add_argument('--rescore_batch_size', type=float, default=200)
    parser.add_argument('--rescore_vali_rate', type=float, default=0.1)
    parser.add_argument('--rescore_model_parameter', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fasta', type=str, default=None)
    parser.add_argument('--mzid', type=int, default=1)
    parser.add_argument('--mod_ini', type=str, default=DEFAULT_MOD_INI)
    return parser.parse_args()

def find_plink_params(plink_path):
    if os.path.isfile(plink_path) and plink_path.lower().endswith(".plink"):
        return plink_path
    if os.path.isdir(plink_path):
        candidates = sorted(
            os.path.join(plink_path, name)
            for name in os.listdir(plink_path)
            if name.lower().endswith(".plink")
        )
        return candidates[0] if candidates else None
    return None

def run():
    args = get_args()          
    plink = plink_with_msconvert_mgf()
    msms_dir, rt_dir, candidate_msms_dir, candidate_rtccs_dir = plink.process(args.plinkfile, args.mgf_dir)
    if args.finetune == 1:
        print('Finetuning the model......')
        train = train_model()
        msms_paradir, rt_paradir = train.finetune(msms_dir, rt_dir)
    else:
        print('No Finetuning')
        base_dir = os.path.dirname(os.path.abspath(__file__))
        msms_paradir = os.path.join(base_dir, 'Deep4D_XL', 'checkpoint', 'PXD017620', 'MSMS.pth')
        rt_paradir = os.path.join(base_dir, 'Deep4D_XL', 'checkpoint', 'PXD017620', 'RT.pth')
    generate = generate_feature()
    candidate_feature = generate.run(candidate_msms_dir, candidate_rtccs_dir, msms_paradir, rt_paradir)
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
    rescore_results_dir = candidate_rtccs_dir.split('.csv')[0] + '_rescore_results.csv'
    rescore_results.to_csv(rescore_results_dir, index=False)
    rescore_results = rescore_results[rescore_results['FDR'] <= args.rescore_fdr]
    rescore_results = rescore_results[rescore_results['Target_Decoy'] == 2]
    rescore_results_dir = candidate_rtccs_dir.split('.csv')[0] + '_rescore_results1.csv'
    rescore_results.to_csv(rescore_results_dir, index=False)
    if args.mzid == 1:
        if not args.fasta:
            print('WARN: --fasta not provided; skipping mzIdentML export.')
        else:
            plink_params_path = find_plink_params(args.plinkfile)
            if not plink_params_path:
                print('WARN: No .plink params file found; skipping mzIdentML export.')
            elif not os.path.isfile(args.mod_ini):
                print(f'WARN: modification.ini not found at {args.mod_ini}; skipping mzIdentML export.')
            else:
                mzid_out = rescore_results_dir.rsplit('.', 1)[0] + '.mzid'
                build_mzid(rescore_results_dir, args.fasta, args.mgf_dir, plink_params_path, args.mod_ini, mzid_out)
                print(f'mzIdentML saved: {mzid_out}')
    os.remove(msms_dir)
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
