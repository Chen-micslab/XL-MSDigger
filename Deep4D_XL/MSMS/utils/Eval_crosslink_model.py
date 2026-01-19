import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import numpy.linalg as L
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def get_mask(peptide,length): 
    mask = torch.zeros(peptide.size(0),peptide.size(1))  
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0 
    return  mask

def get_cosine(msms, msms_pre):
    dot = np.dot(msms,msms_pre)
    return dot/(L.norm(msms)*L.norm(msms_pre))

def get_SA(msms, msms_pre):
    L2normed_act = msms / L.norm(msms)
    L2normed_pred = msms_pre / L.norm(msms_pre)
    inner_product = np.dot(L2normed_act, L2normed_pred)
    return 1 - 2*np.arccos(inner_product)/np.pi

def get_pearson(act, pred):
    return pearsonr(act, pred)[0]


def get_spearman(act, pred):
    return spearmanr(act, pred)[0]


def eval_model(model, loader, device, norm):
    model.eval()                   
    n_val = len(loader)                
    num = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            peptide1 = batch['peptide1'].to(device=device, dtype=torch.float32)
            peptide2 = batch['peptide2'].to(device=device, dtype=torch.float32)
            pep1_len = batch['len1'].to(device=device, dtype=torch.float32)
            pep2_len = batch['len2'].to(device=device, dtype=torch.float32)
            peptide_msms = batch['peptide_msms'].to(device=device, dtype=torch.float32)
            charge = batch['charge'].to(device=device, dtype=torch.float32)
            mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)              
            mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)              
            peptide_msms = norm * peptide_msms
            with torch.no_grad():               
                peptide_msms_pre = model(pep1 = peptide1,pep2 = peptide2, src_key_padding_mask_1 = mask1,src_key_padding_mask_2 = mask2,charge = charge)
            loss_f = nn.L1Loss()
            if num == 0:
                total_peptide_msms = peptide_msms
                total_peptide_msms_pre = peptide_msms_pre
            else:
                total_peptide_msms = torch.cat([total_peptide_msms,peptide_msms],0)
                total_peptide_msms_pre = torch.cat([total_peptide_msms_pre, peptide_msms_pre], 0)
            num += 1
            pbar.update()
    model.train()                
    total_peptide_msms = np.array(total_peptide_msms.to(device='cpu', dtype=torch.float32))
    total_peptide_msms_pre = np.array(total_peptide_msms_pre.to(device='cpu', dtype=torch.float32))
    cosine = []
    x = -1 * norm 
    for i in range(len(total_peptide_msms)):
        a = total_peptide_msms[i, :]
        b = total_peptide_msms_pre[i, :]
                          
        select =  (a != 0) * (a != x)
        if len(a[(a != 0) * (a != x)]) > 2:
            cosine.append(get_cosine(a[select], b[select]))                                    
    mean_cosine = np.mean(cosine)
    median_cosine = np.median(cosine)
    return mean_cosine, median_cosine