import  logging
import numpy as  np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from Deep4D_XL.dataset.Crosslink_Dataset_msms import Mydata_label
from torch.utils.data import DataLoader
from Deep4D_XL.utils.Eval_crosslink_msms import  get_cosine, get_SA, get_pearson, get_spearman

                                                            
def get_mask(peptide,length): 
    mask = torch.zeros(peptide.size(0),peptide.size(1))  
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0
    return  mask

def predict(data_dir, load_msms_param_dir, batch_size):
    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
    from Deep4D_XL.model.crosslink_msms_model_cleavable import Transformer as deep_model
    model = deep_model()
    if load_msms_param_dir:
        model_state_dict = torch.load(load_msms_param_dir, map_location=device, weights_only=True)
        model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
        logging.info(f'Model parameters loaded from {load_msms_param_dir}')
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device=device)
    model.eval()                   
    test_data = Mydata_label(data_dir)
    total_lenth = len(test_data)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
    n_val = len(test_loader)                
    norm = 10
    x = -1 * norm
    index = 0
    total_peptide = np.zeros(total_lenth, dtype=object)
    total_order = np.zeros(total_lenth, dtype=object)
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
            order = np.array(batch['order'])
            mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)              
            mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)              
            peptide_msms = norm * peptide_msms
            with torch.no_grad():               
                peptide_msms_pre = model(pep1 = peptide1, pep2 = peptide2, src_key_padding_mask_1 = mask1, src_key_padding_mask_2 = mask2, charge = charge)
                id = peptide_msms_pre.size(0)
                total_peptide[index:(index+id)] = peptide_seq
                total_order[index:(index+id)] = order
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
    pep, order_list = [], []
    charge_list = []
    match_num, match_num1, match_num2, both_m_p_num, both_m_p_num1, both_m_p_num2 = [],[],[],[],[],[]
    for i in range(len(total_peptide_msms)):
        a = total_peptide_msms[i, :]
        a1 = a.reshape(49, 36)[:, :18].ravel()
        a2 = a.reshape(49, 36)[:, 18:].ravel()
        b = total_peptide_msms_pre[i, :]
        b1 = b.reshape(49, 36)[:, :18].ravel()
        b2 = b.reshape(49, 36)[:, 18:].ravel()
        select = (a != 0) * (a != x)
        select1 = (a1 != 0) * (a1 != x)
        select2 = (a2 != 0) * (a2 != x)
        if len(a[(a != 0) * (a != x)]) >= 2:
            cosine.append(get_cosine(a[select], b[select]))
            SA.append(get_SA(a[select], b[select]))
            pearson.append(get_pearson(a[select], b[select]))
            spearman.append(get_spearman(a[select], b[select]))
        else:
            cosine.append(0)
            SA.append(0)
            pearson.append(0)
            spearman.append(0)
        pep.append(total_peptide[i])
        order_list.append(total_order[i])
        charge_list.append(total_peptide_charge[i])
        match_num.append(len(a[(a != 0) * (a != x)]))
        match_num1.append(len(a1[(a1 != 0) * (a1 != x)]))
        match_num2.append(len(a2[(a2 != 0) * (a2 != x)]))
        both_m_p_num.append((b[select] >= 0.01).sum())
        both_m_p_num1.append((b1[select1] >= 0.01).sum())
        both_m_p_num2.append((b2[select2] >= 0.01).sum())
    data = np.column_stack((match_num, match_num1, match_num2, both_m_p_num, both_m_p_num1, both_m_p_num2, cosine, SA, pearson, spearman))
    pep = np.expand_dims(np.array(pep), axis=1)
    order_list = np.expand_dims(np.array(order_list), axis=1)
    charge_list = np.expand_dims(np.array(charge_list), axis=1)
    data = np.column_stack((pep, order_list, charge_list, data))
    msms_feature = pd.DataFrame(data, columns=['Peptide', 'Order', 'Charge', 'match_num', 'match_num1', 'match_num2', 'both_m_p_num', 'both_m_p_num1', 'both_m_p_num2','cosine', 'SA', 'pearson', 'spearman'])
    return msms_feature


if __name__ == '__main__':
    args = get_args()          
    predict_label_msms(args)


