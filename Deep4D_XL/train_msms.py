import  logging
import os
import math
import numpy as  np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from Deep4D_XL.dataset.Crosslink_Dataset_msms import Mydata_label
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from Deep4D_XL.utils.Eval_crosslink_msms import eval_model


def SL_loss(msms, msms_pre):
        msms = nn.functional.normalize(msms, p=2, dim=1) 
        msms_pre = nn.functional.normalize(msms_pre, p=2, dim=1)
        
        inner_product = torch.sum(msms * msms_pre, dim=1)
        loss = 2 * torch.acos(inner_product) / math.pi
        
        return loss

def get_mask(peptide,length): 
    mask = torch.zeros(peptide.size(0),peptide.size(1))  
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0 
    return  mask

###设置模型训练时的细节
def train(model,device,epochs=30, batch_size=50,lr=0.0001,val_percent=0.1, traindir=None, checkpoint_dir=None):
    mydata = Mydata_label(traindir)  ##导入dataset
    n_val = int(len(mydata) * val_percent)  ##计算validation data的数量
    n_train = len(mydata) - n_val  ##计算train data的数量
    train_data, val_data = random_split(mydata, [n_train, n_val])  ##随机分配validation data和train data
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)  ##生成train data
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True) ##生成validation data
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0
    optimizer =optim.Adam( model.parameters(), lr=lr, betas=(0.9,0.999),eps=1e-08, weight_decay=0, amsgrad=False) ##选择optimizer为Adam,注意传入的是model的parameters
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=10,T_mult=2,eta_min=0.00000001)
    best_index = 0
    for epoch in range(epochs):
        model.train()  ##设置模型为train模式
        local_step = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}',unit='peptide') as pbar: ##使用tqdm来显示训练的进度条
            for batch in train_loader:
                local_step += 1
                peptide1 = batch['peptide1'].to(device=device, dtype=torch.float32)
                peptide2 = batch['peptide2'].to(device=device, dtype=torch.float32)
                pep1_len = batch['len1'].to(device=device, dtype=torch.float32)
                pep2_len = batch['len2'].to(device=device, dtype=torch.float32)
                peptide_msms = batch['peptide_msms'].to(device=device,dtype=torch.float32)
                charge = batch['charge'].to(device=device,dtype=torch.float32)
                mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)  ##求相应的mask矩阵
                mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)  ##求相应的mask矩阵
                norm = 10
                peptide_msms = norm * peptide_msms
                peptide_msms_pre = model(pep1 = peptide1,pep2 = peptide2, src_key_padding_mask_1 = mask1,src_key_padding_mask_2 = mask2,charge = charge) ##将数据送入model，得到预测的ccs
                loss = SL_loss(peptide_msms,peptide_msms_pre)
                pbar.set_postfix(**{'loss (batch)': torch.mean(loss)})  #输入一个字典，来显示实验指标
                optimizer.zero_grad()  ##梯度清零
                loss.backward(loss.clone().detach())  ##反向传播
                optimizer.step()
                pbar.update(peptide1.shape[0])
                global_step += 1
                if global_step % (n_train // (2*batch_size)) == 0:
                    mean_cosine, median_cosine, mean_pearson, median_pearson = eval_model(model,val_loader,device,norm)
                    writer.add_scalar('learning rate',optimizer.param_groups[0]['lr'], global_step)
                    logging.info('Validation Mean cosine: {}'.format(mean_cosine))
                    logging.info('Validation Median cosine: {}'.format(median_cosine))
                    logging.info('Validation Mean pearson: {}'.format(mean_pearson))
                    logging.info('Validation Median pearson: {}'.format(median_pearson))
                    if median_cosine > best_index:
                        torch.save(model.state_dict(),
                                   checkpoint_dir + f'model_param_epoch{epoch + 1}global_step{global_step}#median_cosine{median_cosine}.pth')  ###只保存模型的参数到checkpoint_dir
                        logging.info(f'Checkpoint {epoch + 1}global_step{global_step} saved !')
                        best_index = median_cosine
                        best_checkpoint =  f'model_param_epoch{epoch + 1}global_step{global_step}#median_cosine{median_cosine}.pth'
        scheduler.step()
    writer.close()
    return best_checkpoint

def do_train(traindir, load_msms_param_dir, epochs, batch_size, lr, vali_rate):
    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ##判断使用GPU还是CPU
    print(device)
    logging.info(f'Using device {device}')      
    from Deep4D_XL.model.crosslink_msms_model import Transformer as deep_model
    checkpoint_dir = traindir.rsplit('/', 1)[0] + '/' + 'checkpoint/msms/'
    if os.path.exists(checkpoint_dir):
        pass
    else: os.makedirs(checkpoint_dir)
    model = deep_model()  ###生成模型
    logging.info(f'train msms model:')
    if load_msms_param_dir:
        model_state_dict = torch.load(load_msms_param_dir, map_location=device, weights_only=True)
        model.load_state_dict({k.replace('module.', ''):v for k, v in model_state_dict.items()})
        logging.info(f'Model parameters loaded from {load_msms_param_dir}')
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device=device)
    best_checkpoint = train(model=model,
                          device=device,
                          epochs=epochs,
                          batch_size=batch_size,
                          lr=lr,
                          val_percent=vali_rate,
                          traindir = traindir,
                          checkpoint_dir = checkpoint_dir)
    for filename in os.listdir(checkpoint_dir):
        if filename != best_checkpoint:
            file_path = os.path.join(checkpoint_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  #删除除了最佳checkpoint以外的所有文件
    best_file_path = os.path.join(checkpoint_dir, best_checkpoint)
    return best_file_path

if __name__ == '__main__':
    do_train()
