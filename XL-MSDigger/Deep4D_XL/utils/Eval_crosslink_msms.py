import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import numpy.linalg as L
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import math

def get_cosine(msms, msms_pre):
    dot = np.dot(msms,msms_pre)
    correlation = dot/(L.norm(msms)*L.norm(msms_pre))
    if np.isnan(correlation):
        return 0
    else:
        return correlation
    return correlation

def get_SA(msms, msms_pre):
    L2normed_act = msms / L.norm(msms)
    L2normed_pred = msms_pre / L.norm(msms_pre)
    inner_product = np.dot(L2normed_act, L2normed_pred)
    correlation = (1 - 2*np.arccos(inner_product)/np.pi)
    if np.isnan(correlation):
        return 0
    else:
        return correlation
    return correlation

def get_unweighted_entropy(msms, msms_pre):
    spectrum1_a = msms
    spectrum1_a = spectrum1_a / np.sum(spectrum1_a)
    spectrum2_a = spectrum1_a[spectrum1_a > 0]
    a_entropy = -1 * np.sum(spectrum2_a * np.log(spectrum2_a))

    spectrum1_b = msms_pre
    spectrum1_b = spectrum1_b / np.sum(spectrum1_b)
    spectrum2_b = spectrum1_b[spectrum1_b > 0]
    b_entropy = -1 * np.sum(spectrum2_b * np.log(spectrum2_b))

    spectrum1_ab = (spectrum1_a + spectrum1_b)/2
    spectrum2_ab = spectrum1_ab[spectrum1_ab > 0]
    ab_entropy = -1 * np.sum(spectrum2_ab * np.log(spectrum2_ab))
    unweighted_entropy_similarity = 1 - (2 * ab_entropy - a_entropy - b_entropy) / math.log(4)
    return unweighted_entropy_similarity

def get_pearson(act, pred):
    correlation = pearsonr(act, pred)[0]
                                
    if np.isnan(correlation):
        return 0
    else:
        return correlation


def get_spearman(act, pred):
                               
    correlation = spearmanr(act, pred)[0]
                                
    if np.isnan(correlation):
        return 0
    else:
        return correlation

def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0
    return  mask

def eval_model(model, loader, device  , norm):
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
            pbar.update()
    model.train()                
    total_peptide_msms = np.array(total_peptide_msms.to(device='cpu', dtype=torch.float32))
    total_peptide_msms_pre = np.array(total_peptide_msms_pre.to(device='cpu', dtype=torch.float32))
    cosine = []
    pearson = []
    x = -1 * norm
    for i in range(len(total_peptide_msms)):
        a = total_peptide_msms[i, :]
        b = total_peptide_msms_pre[i, :]
        select = (a != 0) * (a != x)
        if len(a[(a != 0) * (a != x)]) > 2:
            cosine.append(get_cosine(a[select], b[select]))
            pearson.append(get_pearson(a[select], b[select]))
    mean_cosine = np.mean(cosine)
    median_cosine = np.median(cosine)
    mean_pearson = np.mean(pearson)
    median_pearson = np.median(pearson)
    return mean_cosine, median_cosine, mean_pearson, median_pearson