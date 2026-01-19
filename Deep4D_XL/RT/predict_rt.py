import argparse
import logging
import os
import sys

import numpy as  np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from dataset.Dataset_Crosslink import Mydata_label
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.Eval_crosslink import eval_model
from model.selfatt_cnn_crosslink import Transformer


def get_args():             
    parser = argparse.ArgumentParser(description='Train the transformer on peptide and ccs')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--norm', type=float, default=100)            
    parser.add_argument('--validation', type=int, default=0)                           
    parser.add_argument('--vali_rate', type=float, default=0.1)                       
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_dir', type=str, default='./output')    
    parser.add_argument('--load_rt_param_dir', type=str, default='Deep4D_XL/RT/checkpoint/rt.pth')
    parser.add_argument('--seed', type=int, default=1)                            
    parser.add_argument('--filename', type=str, default='RT_test')
    parser.add_argument('--sch', type=int, default=0)
    return parser.parse_args()


def get_position_angle_vec(position, dim):                                   
    return [position / np.power(10000, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]


def get_position_coding(max_len, d_model):
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i, d_model) for pos_i in range(max_len)])                                    
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])                     
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])                      
    return sinusoid_table


def get_mask(peptide, length):
    mask = torch.zeros(peptide.size(0), peptide.size(1))
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0
    return mask


if __name__ == '__main__':
    args = get_args()          
    data_dir = f'./dataset/data/{args.filename}'
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
    model = Transformer(feature_len=args.feature_len,
                            d_model=args.d_model,
                            nhead=args.nheads,
                            num_encoder_layers=args.num_encoder_layers,
                            dim_feedforward=args.dim_feedforward,
                            dropout=args.dropout,
                            activation=args.activation)
    if args.load_rt_param_dir:
                                                                                                         
        model_state_dict = torch.load(args.load_rt_param_dir, map_location=device)
        model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
        logging.info(f'Model parameters loaded from {args.load_rt_param_dir}')
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device=device)
    model.eval()                   
    test_data = Mydata_label(data_dir)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
    n_val = len(test_loader)                
    norm = args.norm
    RT_pre = torch.tensor([]).to(device=device, dtype=torch.float32)
    RT = torch.tensor([]).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            peptide1 = batch['peptide1'].to(device=device, dtype=torch.float32)
            peptide2 = batch['peptide2'].to(device=device, dtype=torch.float32)
            pep1_len = batch['len1'].to(device=device, dtype=torch.float32)
            pep2_len = batch['len2'].to(device=device, dtype=torch.float32)
            rt = batch['RT'].to(device=device, dtype=torch.float32)
            norm = args.norm
            mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)              
            mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)              
            with torch.no_grad():               
                rt_pred = model(pep1=peptide1, pep2=peptide2, src_key_padding_mask_1=mask1, src_key_padding_mask_2=mask2)
            RT_pre = torch.cat((RT_pre,rt_pred*norm),0)
            RT = torch.cat((RT,rt),0)
            pbar.update()
    RT = RT.to(device='cpu', dtype=torch.float32).numpy().flatten()
    RT_pre = RT_pre.to(device='cpu', dtype=torch.float32).numpy().flatten()

          
    abs_err = np.abs(RT - RT_pre)
    mae = np.mean(abs_err)
    mse = np.mean(abs_err ** 2)

    print('MAE:', mae)
    print('MSE:', mse)

                               
    df = pd.DataFrame({
        'rt_true': RT,
        'rt_pred': RT_pre,
        'abs_err': abs_err
    })
                  
    df.loc['MAE'] = [np.nan, np.nan, mae]

    df.to_csv(f'./dataset/data/output/{args.filename}_pre.csv')