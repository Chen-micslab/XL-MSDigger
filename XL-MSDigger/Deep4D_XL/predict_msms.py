import numpy as  np
import pandas as pd
import torch
from tqdm import tqdm
from Deep4D_XL.dataset.Crosslink_Dataset_msms import Mydata_nolabel
from torch.utils.data import DataLoader
from Preprocess.mass_cal import mz_cal as m
import torch.nn as nn
import  logging


def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0
    return  mask
    
def predict_nolabel_msms(task_dir, filename, slice, batch_size, load_msms_param_dir):
    file_dir = task_dir
    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
    from Deep4D_XL.model.crosslink_msms_model import Transformer as deep_model
    model = deep_model()
    if load_msms_param_dir:
        model_state_dict = torch.load(load_msms_param_dir, map_location=device)
        model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
        logging.info(f'Model parameters loaded from {load_msms_param_dir}')
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device=device)
    model.eval()                   
    for slice_num in range(slice):
        print(f'{slice_num}/{slice} slices in msms prediction...........')
        data_dir = f'{file_dir}/{filename}_slice{slice_num}'
        test_data = Mydata_nolabel(data_dir)
        total_lenth = len(test_data)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
        norm = 10
        x = -1 * norm
        index = 0
        total_peptide = np.zeros(total_lenth, dtype=object)
        total_peptide_charge = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
        total_peptide_msms_pre = torch.zeros((total_lenth, 980)).to(device=device, dtype=torch.float32)
        with tqdm(total=len(test_loader), desc='Prediction round', unit='peptide', leave=False) as pbar:
            for batch in test_loader:
                peptide1 = batch['peptide1'].to(device=device, dtype=torch.float32)
                peptide2 = batch['peptide2'].to(device=device, dtype=torch.float32)
                pep1_len = batch['len1'].to(device=device, dtype=torch.float32)
                pep2_len = batch['len2'].to(device=device, dtype=torch.float32)
                charge = batch['charge'].to(device=device,dtype=torch.float32)
                peptide_seq = np.array(batch['peptide'])
                mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)              
                mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)              
                with torch.no_grad():               
                    peptide_msms_pre = model(pep1 = peptide1, pep2 = peptide2, src_key_padding_mask_1 = mask1,
                                             src_key_padding_mask_2 = mask2, charge = charge)
                    id = peptide_msms_pre.size(0)
                    total_peptide[index:(index + id)] = peptide_seq
                    total_peptide_msms_pre[index:(index + id), :] = peptide_msms_pre
                    total_peptide_charge[index:(index + id)] = charge
                    index = index + id
                pbar.update()
        total_peptide_msms_pre = np.array(total_peptide_msms_pre.to(device='cpu', dtype=torch.float32))
        total_peptide_charge = np.array(total_peptide_charge.to(device='cpu', dtype=torch.int))
        num = 0
        pep = []
        charge_list = []
        m_z_list = []
        total_peptide_msms_pre_1 = total_peptide_msms_pre
        for i in range(len(total_peptide_msms_pre)):
            b = total_peptide_msms_pre[i, :]
                                                         
                               
            a = m()
            m_z = a.crosslink_peptide_m_z(total_peptide[i], total_peptide_charge[i], 'DSS')
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
        data.to_csv(f'{file_dir}/output/{filename}_slice{slice_num}_pre_msms.csv', index=False)



