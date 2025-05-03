import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0
    return  mask

def eval_model(model, loader, device, norm):
    model.eval()  ##将model调整为eval模式
    n_val = len(loader)  #val中的batch数量
    tot_MAE = 0  ##average relative error
    RT = torch.tensor([]).to(device=device, dtype=torch.float32)
    RT_pre = torch.tensor([]).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            peptide1 = batch['peptide1'].to(device=device, dtype=torch.float32)
            peptide2 = batch['peptide2'].to(device=device, dtype=torch.float32)
            pep1_len = batch['len1'].to(device=device, dtype=torch.float32)
            pep2_len = batch['len2'].to(device=device, dtype=torch.float32)
            rt = batch['RT'].to(device=device, dtype=torch.float32)
            rt = rt / norm
            mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)  ##求相应的mask矩阵
            mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)  ##求相应的mask矩阵
            # peptide1 = peptide1.transpose(0, 1)  ##self attention需要的输入是（max_length,batchsize,embedding），需要把peptide的前两维转置
            # peptide2 = peptide2.transpose(0, 1)  ##self attention需要的输入是（max_length,batchsize,embedding），需要把peptide的前两维转置
            with torch.no_grad():
                rt_pred = model(pep1=peptide1, pep2=peptide2, src_key_padding_mask_1=mask1,
                            src_key_padding_mask_2=mask2).view(rt.size(0))  ##将数据送入model，得到预测的ccs
            loss_f = nn.L1Loss()
            RT = torch.cat((RT, rt), 0)
            RT_pre = torch.cat((RT_pre, rt_pred), 0)
            pbar.update()
    model.train() ##将模型调回train模式
    tot_ARE = (torch.abs(RT - RT_pre)).mean()

    return tot_ARE*norm  ##返回loss的均值以及MRE