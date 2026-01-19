import argparse
import  logging
import os
import sys
import shutil
import numpy as  np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from pathlib import Path
import time
from model import selfatt_cnn_crosslink
from dataset.Dataset_Crosslink import Mydata_label
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.Eval_crosslink import eval_model
from model.selfatt_cnn_crosslink import Transformer
from dataset.Crosslink_Encoding import crosslink_ccs_encoding

                                                               


def get_args():             
    parser = argparse.ArgumentParser(description='Train the transformer on peptide and ccs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--max', type=float, default=629.5419871)
    parser.add_argument('--min', type=float, default=272.0595023)
    parser.add_argument('--norm', type=float, default=100)            
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_dir', type=str, default='./output')    
    parser.add_argument('--load_ccs_param_dir', type=str, default='Deep4D_XL/CCS/checkpoint/ccs.pth')                            
    parser.add_argument('--seed', type=int, default=1)                            
    parser.add_argument('--filename', type=str, default='CCS_test')
    return parser.parse_args()

def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0
    return  mask

def extract(str,symbol):                              
    index = []
    for i in range(len(str)):
        if str[i]==symbol:
            index.append(i)
    start_index = index[0]
    end_index = index[1]
    str1 = str[(start_index+1):end_index]
    return str1

def get_position_angle_vec(position,dim):                                  
    return [position / np.power(10000, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                     
    args = get_args()
    data_csvdir =f'./dataset/data/input/{args.filename}.csv'           
    data_csv = pd.read_csv(data_csvdir)
    data_npydir = './temporary'                  
    if Path(data_npydir).exists():
        shutil.rmtree(data_npydir)
    crosslink_ccs_encoding(data_csvdir,data_npydir)                    
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
    if args.load_ccs_param_dir:
                                                                                                         
        model_state_dict = torch.load(args.load_ccs_param_dir, map_location=device)
        model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
        logging.info(f'Model parameters loaded from {args.load_ccs_param_dir}')
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device=device)
    model.eval()                   
    test_data = Mydata_label(data_npydir)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
    n_val = len(test_loader)                
    tot_loss = 0                  
    tot_ARE = 0                          
    tot = torch.tensor([]).to(device=device, dtype=torch.float32)
    norm = args.norm
    CCS = torch.tensor([]).to(device=device, dtype=torch.float32)
    CCS_pre = torch.tensor([]).to(device=device, dtype=torch.float32)
    time_start = time.time()
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            peptide1 = batch['peptide1'].to(device=device, dtype=torch.float32)
            peptide2 = batch['peptide2'].to(device=device, dtype=torch.float32)
            m_z = batch['m_z'].to(device=device, dtype=torch.float32)
            pep1_len = batch['len1'].to(device=device, dtype=torch.float32)
            pep2_len = batch['len2'].to(device=device, dtype=torch.float32)
            ccs = batch['ccs'].to(device=device, dtype=torch.float32)
            charge = batch['charge'].to(device=device,dtype=torch.float32)
            norm = args.norm
            ccs = ccs/norm
            m_z = m_z/norm
            mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)              
            mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)              
            with torch.no_grad():               
                 ccs_pred = model(pep1 = peptide1,pep2 = peptide2, src_key_padding_mask_1 = mask1,
                                            src_key_padding_mask_2 = mask2,charge = charge, mz = m_z).view(ccs.size(0))                      
            CCS = torch.cat((CCS, ccs), 0)
            CCS_pre = torch.cat((CCS_pre, ccs_pred), 0)
            print(CCS.size())
            loss_f = nn.L1Loss()
            tot_loss += torch.abs(ccs - ccs_pred).mean()                    
            tot_ARE += (torch.abs(ccs - ccs_pred) / ccs).mean()
            tot = torch.cat([tot, torch.abs(ccs - ccs_pred) / ccs], 0)
            pbar.update()
    model.train()                
    shutil.rmtree(data_npydir)
    tot = np.array(tot.to(device='cpu', dtype=torch.float32))
    tot_MRE = np.median(tot)
    CCS = CCS.unsqueeze(1)
    CCS_pre = CCS_pre.unsqueeze(1)
    CCS_total = torch.cat((CCS, CCS_pre), 1)
    CCS_total = CCS_total.to(device='cpu', dtype=torch.float32).numpy()
    CCS_total = pd.DataFrame(CCS_total*norm,columns=['CCS','ccs_pre'])
    print('ARE:', tot_ARE / n_val)
    print('MRE:', tot_MRE)
    data = data_csv.join(CCS_total['ccs_pre'])
    data.to_csv(f'./dataset/data/output/{args.filename}_pre.csv',index=False)                       
    time_end = time.time()
    print('totally cost', time_end - time_start)

