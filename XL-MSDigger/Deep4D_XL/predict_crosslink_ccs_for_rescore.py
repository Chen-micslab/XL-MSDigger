import  logging
import numpy as  np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from Deep4D_XL.dataset.Crosslink_Dataset_ccs import Mydata_label
from torch.utils.data import DataLoader
from Deep4D_XL.model.crosslink_ccs_model import Transformer


                                                            
def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0
    return  mask

def predict(data_dir, load_ccs_param_dir, batch_size):
    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
    model = Transformer()
    model_state_dict = torch.load(load_ccs_param_dir, map_location=device, weights_only=True)
    model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
    logging.info(f'Model parameters loaded from {load_ccs_param_dir}')
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device=device)
    model.eval()                   
    test_data = Mydata_label(data_dir)
    total_lenth = len(test_data)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False, num_workers=3, pin_memory=True)
    n_val = len(test_loader)                
    norm = 100
    index = 0
    total_order = np.zeros(total_lenth, dtype=object)
    CCS = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
    CCS_pre = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            peptide1 = batch['peptide1'].to(device=device, dtype=torch.float32)
            peptide2 = batch['peptide2'].to(device=device, dtype=torch.float32)
            m_z = batch['m_z'].to(device=device, dtype=torch.float32)
            pep1_len = batch['len1'].to(device=device, dtype=torch.float32)
            pep2_len = batch['len2'].to(device=device, dtype=torch.float32)
            ccs = batch['ccs'].to(device=device, dtype=torch.float32)
            charge = batch['charge'].to(device=device,dtype=torch.float32)
            order = np.array(batch['order'])
            ccs = ccs/norm
            mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)              
            mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)              
            with torch.no_grad():               
                ccs_pre = model(pep1 = peptide1,pep2 = peptide2, src_key_padding_mask_1 = mask1,
                                src_key_padding_mask_2 = mask2,charge = charge, mz = m_z).view(ccs.size(0))                      
            id = ccs.size(0)
            CCS[index:(index + id)] = ccs
            CCS_pre[index:(index + id)] = ccs_pre
            total_order[index:(index + id)] = order
            index = index + id
            pbar.update()
    model.train()                
    CCS = CCS.unsqueeze(1)
    CCS_pre = CCS_pre.unsqueeze(1)
    CCS_total = torch.cat((CCS,CCS_pre),1)
    CCS_total = CCS_total.to(device='cpu', dtype=torch.float32).numpy()
    MRE_total = np.abs(CCS_total[:,0]-CCS_total[:,1])/CCS_total[:,0]
    total_order = np.expand_dims(np.array(total_order), axis=1)
    data = np.column_stack((total_order, CCS_total, MRE_total))
    ccs_feature = pd.DataFrame(data,columns=['Order', 'ccs', 'ccs_pre', 'ccs_RE'])
    return ccs_feature
