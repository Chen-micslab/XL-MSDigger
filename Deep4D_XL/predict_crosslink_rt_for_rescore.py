import  logging
import numpy as  np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
from Deep4D_XL.dataset.Crosslink_Dataset_rt import Mydata_label
from torch.utils.data import DataLoader
from Deep4D_XL.model.crosslink_rt_model import Transformer

#######这个程序适用于预测的肽特别多的时候，先将csv文件里的所有肽转换成包含npy文件的文件夹，然后在用这个程序
def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0
    return  mask

def predict(data_dir, load_rt_param_dir, batch_size):
    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ##判断使用GPU还是CPU
    model = Transformer()
    model_state_dict = torch.load(load_rt_param_dir, map_location=device, weights_only=True)
    model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
    logging.info(f'Model parameters loaded from {load_rt_param_dir}')
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device=device)
    model.eval()  ##将model调整为eval模式
    test_data = Mydata_label(data_dir)
    total_lenth = len(test_data)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
    n_val = len(test_loader)  # val中的batch数量
    norm = 100
    index = 0
    total_order = np.zeros(total_lenth, dtype=object)
    RT = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
    RT_pre = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            peptide1 = batch['peptide1'].to(device=device, dtype=torch.float32)
            peptide2 = batch['peptide2'].to(device=device, dtype=torch.float32)
            pep1_len = batch['len1'].to(device=device, dtype=torch.float32)
            pep2_len = batch['len2'].to(device=device, dtype=torch.float32)
            rt = batch['RT'].to(device=device, dtype=torch.float32)
            order = np.array(batch['order'])
            mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)  ##求相应的mask矩阵
            mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)  ##求相应的mask矩阵
            with torch.no_grad():  ##不生成梯度，减少运算量
                 rt_pre = model(pep1=peptide1, pep2=peptide2, src_key_padding_mask_1=mask1, src_key_padding_mask_2=mask2).view(rt.size(0))
            id = rt.size(0)
            RT[index:(index + id)] = rt
            RT_pre[index:(index + id)] = rt_pre*norm
            total_order[index:(index + id)] = order
            index = index + id
            pbar.update()
    model.train()  ##将模型调回train模式
    RT = RT.unsqueeze(1)
    RT_pre = RT_pre.unsqueeze(1)
    RT_total = torch.cat((RT, RT_pre), 1)
    RT_total = RT_total.to(device='cpu', dtype=torch.float32).numpy()
    AE_total = np.abs(RT_total[:, 0] - RT_total[:, 1])
    data = np.column_stack((total_order, RT_total, AE_total))
    rt_feature = pd.DataFrame(data, columns=['Order', 'rt', 'rt_pre', 'rt_AE'])
    return rt_feature
