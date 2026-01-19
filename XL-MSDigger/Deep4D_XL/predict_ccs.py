import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import logging
from Deep4D_XL.dataset.Crosslink_Dataset_ccs import Mydata_nolabel
from torch.utils.data import DataLoader
from Deep4D_XL.model.crosslink_ccs_model import Transformer

def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0
    return  mask

def predict_nolabel_ccs(task_dir, filename, slice, batch_size, load_ccs_param_dir):
    file_dir = task_dir
    for slice_num in range(slice):
        print(f'{slice_num}/{slice} slices in ccs prediction...........')
        data_dir = f'{file_dir}/{filename}_slice{slice_num}'
        torch.cuda.manual_seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
        model = Transformer()
        if load_ccs_param_dir:
            model_state_dict = torch.load(load_ccs_param_dir, map_location=device)
            model.load_state_dict({k.replace('module.', ''): v for k, v in model_state_dict.items()})
            logging.info(f'Model parameters loaded from {load_ccs_param_dir}')
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        model.to(device=device)
        model.eval()                   
        test_data = Mydata_nolabel(data_dir)
        total_lenth = len(test_data)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
        n_val = len(test_loader)                
        norm = 100
        index = 0
        total_peptide = np.zeros(total_lenth, dtype=object)
        total_peptide_charge = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
        CCS_pre = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
        with tqdm(total=n_val, desc='Prediction round', unit='batch', leave=False) as pbar:
            for batch in test_loader:
                peptide_seq = np.array(batch['peptide'])
                peptide1 = batch['peptide1'].to(device=device, dtype=torch.float32)
                peptide2 = batch['peptide2'].to(device=device, dtype=torch.float32)
                pep1_len = batch['len1'].to(device=device, dtype=torch.float32)
                pep2_len = batch['len2'].to(device=device, dtype=torch.float32)
                charge = batch['charge'].to(device=device, dtype=torch.float32)
                m_z = batch['m_z'].to(device=device, dtype=torch.float32)
                m_z = m_z/norm
                mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)              
                mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)              
                with torch.no_grad():               
                    ccs_pre = model(pep1=peptide1, pep2=peptide2, src_key_padding_mask_1=mask1,
                                    src_key_padding_mask_2=mask2, charge=charge, mz = m_z).view(charge.size(0))                       
                    id = charge.size(0)
                    total_peptide[index:(index + id)] = peptide_seq
                    CCS_pre[index:(index + id)] = ccs_pre
                    total_peptide_charge[index:(index + id)] = charge
                    index = index + id
                pbar.update()
        CCS_pre = CCS_pre.unsqueeze(1) * norm
        total_peptide_charge = total_peptide_charge.unsqueeze(1)
        total_peptide = np.expand_dims(np.array(total_peptide), axis=1)
        data_total = torch.cat((total_peptide_charge, CCS_pre), 1)
        data_total = data_total.to(device='cpu', dtype=torch.float32).numpy()
        data = np.column_stack((total_peptide, data_total))
        data = pd.DataFrame(data, columns=['combine_peptide', 'charge', 'ccs_pre'])
        data.to_csv(f'{file_dir}/output/{filename}_slice{slice_num}_pre_ccs.csv', index=False)                        
