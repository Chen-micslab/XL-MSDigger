import argparse
import  logging
import os
import sys
sys.path.append('./program')
import time
import numpy as  np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from dataset.Crosslink_Dataset import Mydata_label, Mydata_nolabel
from torch.utils.data import DataLoader
from utils.Eval_crosslink_model import eval_model, get_cosine, get_SA, get_pearson, get_spearman
from program.mass_cal import mz_cal 
                                                            

def get_args():             
    parser = argparse.ArgumentParser(description='Predict msms')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--msms_norm', type=float, default=10)            
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_dir', type=str, default='./output')    
    parser.add_argument('--load_msms_param_dir', type=str, default='Deep4D_XL/MSMS/checkpoint/msms_c.pth')                            
    parser.add_argument('--seed', type=int, default=1)                            
    parser.add_argument('--filename', type=str, default='merged_lib_test')
    parser.add_argument('--label', type=int, default=1)                        
    parser.add_argument('--slice', type=int, default=1)
    return parser.parse_args()

def get_mask(peptide,length): 
    mask = torch.zeros(peptide.size(0),peptide.size(1))  
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0
    return  mask

def predict_label_msms(args):
    data_dir = f'./dataset/data/{args.filename}'
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
    from model.crosslink_msms_model_cleavable import Transformer as deep_model
    model = deep_model(feature_len=args.feature_len,
                        d_model=args.d_model,
                        nhead=args.nheads,
                        num_encoder_layers=args.num_encoder_layers,
                        dim_feedforward=args.dim_feedforward,
                        dropout=args.dropout,
                        activation=args.activation)
    if args.load_msms_param_dir:
                                                                                                         
        model_state_dict = torch.load(args.load_msms_param_dir, map_location=device)
        model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
        logging.info(f'Model parameters loaded from {args.load_msms_param_dir}')
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device=device)
    model.eval()                   
    test_data = Mydata_label(data_dir)
    total_lenth = len(test_data)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
    n_val = len(test_loader)                
    norm = args.msms_norm
    num = 0
    x = -1 * norm
    index = 0
    total_peptide = np.zeros(total_lenth, dtype=object)
    total_peptide_charge = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
                                                                                           
    total_peptide_msms = torch.zeros((total_lenth, 1764)).to(device=device, dtype=torch.float32)
    total_peptide_msms_pre = torch.zeros((total_lenth, 1764)).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            peptide1 = batch['peptide1'].to(device=device, dtype=torch.float32)
            peptide2 = batch['peptide2'].to(device=device, dtype=torch.float32)
            pep1_len = batch['len1'].to(device=device, dtype=torch.float32)
            pep2_len = batch['len2'].to(device=device, dtype=torch.float32)
            peptide_msms = batch['peptide_msms'].to(device=device,dtype=torch.float32)
            charge = batch['charge'].to(device=device,dtype=torch.float32)
            peptide_seq = np.array(batch['peptide'])
            mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)              
            mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)              
            norm = args.msms_norm
            peptide_msms = norm * peptide_msms
            with torch.no_grad():               
                peptide_msms_pre = model(pep1 = peptide1, pep2 = peptide2, src_key_padding_mask_1 = mask1, src_key_padding_mask_2 = mask2, charge = charge)
                id = peptide_msms_pre.size(0)
                total_peptide[index:(index+id)] = peptide_seq
                total_peptide_msms[index:(index + id), :] = peptide_msms
                total_peptide_msms_pre[index:(index+id),:] = peptide_msms_pre
                total_peptide_charge[index:(index+id)] = charge
                                                                
                index = index + id
            pbar.update()
    model.train()                
    total_peptide_msms_pre[torch.where(total_peptide_msms == x)] = x
    total_peptide_msms = np.array(total_peptide_msms.to(device='cpu', dtype=torch.float32))
    total_peptide_msms_pre = np.array(total_peptide_msms_pre.to(device='cpu', dtype=torch.float32))
    total_peptide_charge = np.array(total_peptide_charge.to(device='cpu', dtype=torch.int))
                                                                                           
    cosine = []
    SA = []
    pearson = []
    spearman = []
    pep = []
    charge_list = []
    for i in range(len(total_peptide_msms)):
        a = total_peptide_msms[i, :]
        b = total_peptide_msms_pre[i, :]
                            
        select = (a != 0) * (a != x)
        if len(a[(a != 0) * (a != x)]) > 2:
            pep.append(total_peptide[i])
            charge_list.append(total_peptide_charge[i])
            cosine.append(get_cosine(a[select], b[select]))
            SA.append(get_SA(a[select], b[select]))
            pearson.append(get_pearson(a[select], b[select]))
            spearman.append(get_spearman(a[select], b[select]))

    data = np.column_stack((cosine, SA, pearson, spearman))
    pep = np.expand_dims(np.array(pep), axis=1)
    charge_list = np.expand_dims(np.array(charge_list), axis=1)
    data = np.column_stack((pep, charge_list, data))
    data = pd.DataFrame(data, columns=['Peptide', 'Charge', 'cosine', 'SA', 'pearson', 'spearman'])
    data.to_csv(f'./dataset/data/output/{args.filename}_pre_perform.csv', index=False)
    print('Validation Mean cosine: {}'.format(np.mean(cosine)))
    print('Validation Median cosine: {}%'.format(np.median(cosine)))
    print('Validation Mean SA: {}'.format(np.mean(SA)))
    print('Validation Median SA: {}%'.format(np.median(SA)))
    print('Validation Mean pearson: {}'.format(np.mean(pearson)))
    print('Validation Median pearson: {}%'.format(np.median(pearson)))
    print('Validation Mean spearman: {}'.format(np.mean(spearman)))
    print('Validation Median spearman: {}%'.format(np.median(spearman)))
    num = 0
    pep = []
    m_z_list = []
    charge_list = []
    pep_len = []
    total_peptide_msms_pre_1 = total_peptide_msms_pre
    m = mz_cal()
    for i in range(len(total_peptide_msms)):
        a = total_peptide_msms[i, :]
        b = total_peptide_msms_pre[i, :]
        if len(b[(b != 0) * (b != x)]) > 2:
            b = b / np.max(b)
            m_z = m.crosslink_peptide_m_z(total_peptide[i], total_peptide_charge[i], 'DSS')
            m_z_list.append(m_z)
            pep.append(total_peptide[i])
            charge_list.append(total_peptide_charge[i])
                                                    
            total_peptide_msms_pre_1[i,:] = b
            num = num + 1
    total_peptide_msms_pre_1 = total_peptide_msms_pre_1[:num, :]
    pep = np.expand_dims(np.array(pep), axis=1)
    m_z_list = np.expand_dims(np.array(m_z_list), axis=1)
    charge_list = np.expand_dims(np.array(charge_list), axis=1)
                                                         
    data = np.column_stack((pep, charge_list, m_z_list, total_peptide_msms_pre_1))
    data = pd.DataFrame(data)
    data.to_csv(f'./dataset/data/output/{args.filename}_pre.csv', index=False)

if __name__ == '__main__':
    args = get_args()          
    if args.label == 1:
        predict_label_msms(args)
    else:
        predict_nolabel_msms(args)

