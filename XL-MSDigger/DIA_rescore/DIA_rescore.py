import argparse
import logging
import os
import sys
from torch.utils.data import Dataset,DataLoader
import numpy as  np
import pandas as pd
from torch import optim
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.init import xavier_uniform_
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler

class Mydata_label(Dataset):
    def __init__(self, feature, label):
        self.feature = np.array(feature) 
        self.label = np.array(label)

    def __getitem__(self, index):
        feature = self.feature[index]
        label = self.label[index]
        sample = {'feature': feature, 'label': label}
        return sample

    def __len__(self):
        return len(self.label)

class Mydata_nolabel(Dataset):
    def __init__(self, feature):
        self.feature = np.array(feature)  

    def __getitem__(self, index):
        feature = self.feature[index]
        sample = {'feature': feature}
        return sample

    def __len__(self):
        return len(self.feature)

class dnn_model(Module):
    def __init__(self, feature_num = 20):
        super(dnn_model, self).__init__()
        self.input = feature_num
        self.linear1 = Linear(self.input, 100)
        self.linear2 = Linear(100, 2000)
        self.linear3 = Linear(2000, 100)
        self.linear4 = Linear(100, 1)

    def forward(self, pep_feaure) -> Tensor:
        pep = self.linear1(pep_feaure)
        pep = F.relu(pep)
        pep = self.linear2(pep)
        pep = F.relu(pep)
        pep = self.linear3(pep)
        pep = F.relu(pep)
        rescore = self.linear4(pep)
        return rescore

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class dnn_rescore():
    def __init__(self):
        self.load_param_dir = None
        self.epoch = 10
        self.batch_size = 50
        self.lr = 0.001
        self.val_percent = 0.1
        self.cv_fold = 4
        self.fdr = 0.001
        self.max_train_sample = 5000

    def eval_model(self, model, loader, device):
        model.eval()                   
        n_val = len(loader)                
        ex = torch.tensor([]).to(device=device, dtype=torch.float32)
        pre = torch.tensor([]).to(device=device, dtype=torch.float32)
        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for batch in loader:
                feature = batch['feature'].to(device=device, dtype=torch.float32)
                label = batch['label'].to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    score_pre = model(feature).view(label.size(0))                       
                ex = torch.cat((ex, label), 0)
                pre = torch.cat((pre, score_pre), 0)
                pbar.update()
        model.train()                
        tot = torch.abs(ex - pre)
        mean_error = tot.mean()
        median_error = tot.median()
        return mean_error, median_error                  

    def train_DNN(self, model, device, epochs=10, batch_size=50, lr=0.001, val_percent=0.1, save_mp=True, checkpoint_dir=None,
                  feature=None, label=None):
        mydata = Mydata_label(feature, label)             
        n_val = int(len(mydata) * val_percent)                        
        n_train = len(mydata) - n_val                   
        train_data, val_data = random_split(mydata, [n_train, n_val])                                  
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)                
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0,
                                pin_memory=True, )                     
        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
        global_step = 0
        logging.info(f'''Starting training:
                Epochs:          {epochs}
                Batch size:      {batch_size}
                Learning rate:   {lr}
                Training size:   {n_train}
                Validation size: {n_val}
                Checkpoints:     {save_mp}
                Device:          {device.type}
            ''')
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                               amsgrad=False)                                           
                                                                                                                          
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)
        best_index = 1
        for epoch in range(epochs):
            model.train()                
            local_step = 0
            epoch_loss = 0                      
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='peptide') as pbar:                   
                for batch in train_loader:
                    local_step += 1
                    feature = batch['feature'].to(device=device, dtype=torch.float32)
                    label = batch['label'].to(device=device, dtype=torch.float32)
                    score_pre = model(feature).view(label.size(0))                       
                    loss_f = nn.MSELoss()
                    loss = loss_f(label, score_pre)                      
                    epoch_loss += loss.item()
                    writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})                  
                    optimizer.zero_grad()        
                    loss.backward()        
                    optimizer.step()
                    pbar.update(label.shape[0])
                    global_step += 1
                    if global_step % (n_train // (2 * batch_size)) == 0:
                        val_MeanE, val_MedianE = self.eval_model(model, val_loader, device)
                        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], global_step)
                        logging.info('Validation MeanE: {}'.format(val_MeanE))
                        logging.info('Validation MedianE: {}'.format(val_MedianE))
                        if val_MedianE < best_index:
                            torch.save(model.state_dict(),
                                       checkpoint_dir + f'model_param_epoch{epoch + 1}MedianE{val_MedianE}.pth')                            
                            para_dir = checkpoint_dir + f'model_param_epoch{epoch + 1}MedianE{val_MedianE}.pth'
                            logging.info(f'Checkpoint {epoch + 1}global_step{global_step} saved !')
                            best_index = val_MedianE
            scheduler.step()
        writer.close()
        return para_dir

    def do_train(self, feature, label, checkpoint_dir):
        if os.path.exists(checkpoint_dir):
            pass
        else: os.makedirs(checkpoint_dir)
        torch.cuda.manual_seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
        print(device)
        logging.info(f'Using device {device}')
        model = dnn_model()         
        if self.load_param_dir:
            model.load_state_dict(torch.load(self.load_param_dir, map_location=device))
            logging.info(f'Model parameters loaded from {self.load_param_dir}')
        model.to(device=device)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        para_dir = self.train_DNN(model = model,
                                 device = device,
                                 epochs = self.epoch,
                                 batch_size = self.batch_size,
                                 lr = self.lr,
                                 val_percent = self.val_percent,
                                 save_mp = True,
                                 checkpoint_dir = checkpoint_dir,
                                 feature=feature, label = label)
        return para_dir

    def do_predict(self, para_dir, test_feature, batch_size):
        model = dnn_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(para_dir, map_location=device))
        logging.info(f'Model parameters loaded from {para_dir}')
        test_data = Mydata_nolabel(test_feature)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
        n_test = len(test_loader)
        model.to(device=device)
        model.eval()                   
        pre = torch.tensor([]).to(device=device, dtype=torch.float32)
        with tqdm(total=n_test, desc='Validation round', unit='batch', leave=False) as pbar:
            for batch in test_loader:
                feature = batch['feature'].to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    score_pre = model(feature).view(feature.size(0))                       
                pre = torch.cat((pre, score_pre), 0)
                pbar.update()
        pre = pre.to(device='cpu', dtype=torch.float32).numpy()
        pre = list(pre)
        return pre

    def cv_score(self, feature_dir):                      
        data = pd.read_csv(feature_dir)
        feature_list = ['Evidence', 'CScore', 'Q.Value', 'PEP', 'pep1_num', 'pep2_num', 'pep1_num_matched', 'pep2_num_matched',
                        'pep_cosine', 'pep1_cosine', 'pep2_cosine',
                        'spec_frag_num', 'alpha_spec_frag_num', 'beta_spec_frag_num',
                        'pep_entropy', 'pep1_entropy', 'pep2_entropy',
                        'delta_RT', 'delta_IM', 'averge_corr_list']
        data_target = data[(data['type'] == 'TT') & (data['type2'] == 'exp')]
        data_decoy = data[(data['type'] == 'TD') | (data['type'] == 'DD')]
        type = set(list(data['type2']))
        if 'test' in type:
            test = data[data['type2'] == 'test']
        if self.cv_fold > 1:
            test_df_list = []
            for i in range(self.cv_fold):
                t_mask = np.ones(len(data_target), dtype=bool)
                slice_num = slice(i, len(data_target), self.cv_fold)
                t_mask[slice_num] = False
                cv_df_target = data_target[t_mask]
                train_t_df = cv_df_target[cv_df_target['Q.Value'] <= self.fdr]
                train_t_df = train_t_df[train_t_df['pep1_num_matched'] != 0]
                train_t_df = train_t_df[train_t_df['pep2_num_matched'] != 0]
                train_t_df = train_t_df[train_t_df['Ms1.Profile.Corr'] > 0.8]
                train_t_df = train_t_df[train_t_df['inten_ratio'] > 1]
                test_t_df = data_target[slice_num]

                d_mask = np.ones(len(data_decoy), dtype=bool)
                slice_num = slice(i, len(data_decoy), self.cv_fold)
                d_mask[slice_num] = False
                train_d_df = data_decoy[d_mask]
                test_d_df = data_decoy[slice_num]
                if len(train_t_df) > self.max_train_sample:
                    train_t_df = train_t_df.sample(n=self.max_train_sample, random_state=123)
                if len(train_d_df) > self.max_train_sample:
                    train_d_df = train_d_df.sample(n=self.max_train_sample, random_state=123)
                print(len(train_t_df['Modified.Sequence']))
                print(len(train_d_df['Modified.Sequence']))
                train_df = pd.concat((train_t_df, train_d_df))
                train_label = np.ones(len(train_df), dtype=np.int32)
                train_label[len(train_t_df):] = 0
                feature_train = train_df[feature_list]
                checkpoint_dir = feature_dir.rsplit('/', 1)[0] + '/' + 'checkpoint/'                             
                para_dir = self.do_train(feature_train, train_label, checkpoint_dir)
                test_df = pd.concat((test_t_df, test_d_df))
                feature_test = test_df[feature_list]
                pre = self.do_predict(para_dir, feature_test, self.batch_size)
                test_df.insert(0, 'ml_score', pre)
                test_df_list.append(test_df)
                if 'test' in type:
                    pre1 = np.array(self.do_predict(para_dir, test[feature_list], self.batch_size))
                                                      
                    if i == 0:
                        ml_score = pre1
                    else:
                        ml_score = ml_score + pre1
            data1 = pd.concat(test_df_list)
            if 'test' in type:
                ml_score = ml_score / self.cv_fold
                test.insert(0, 'ml_score', ml_score)
                data1 = pd.concat([data1, test])
            import shutil
            shutil.rmtree(checkpoint_dir)
            return data1

    def fdr_to_q_values(self, fdr_values):                 
        q_values = np.zeros_like(fdr_values)
        min_q_value = np.max(fdr_values)
        for i in range(len(fdr_values) - 1, -1, -1):
            fdr = fdr_values[i]
            if fdr < min_q_value:
                min_q_value = fdr
            q_values[i] = min_q_value
        return q_values

    def FDR_to_score(self, data, fdr):                
        data = data[data['type2'] == 'exp']
        data1 = data.sort_values('ml_score', ignore_index=True, ascending=False)
        type = np.array(data1['type'])
        id_TT = np.array(type == 'TT').astype(int)
        id_TD = np.array(type == 'TD').astype(int)
        id_DD = np.array(type == 'DD').astype(int)
        TT_cumsum = np.cumsum(id_TT)
                          
        TD_cumsum = np.cumsum(id_TD)
        DD_cumsum = np.cumsum(id_DD)
        fdrs = (TD_cumsum - DD_cumsum) / TT_cumsum
        q_values = self.fdr_to_q_values(fdrs)
                         
        data1.insert(0, 'FDR_rescore', q_values)
        data2 = data1[data1['FDR_rescore'] <= fdr]
        score = data2['ml_score']
        return np.min(score)

    def run(self, feature_dir):
        data1 = self.cv_score(feature_dir)
        data1 = data1[data1['pep1_num_matched'] != 0]
        data1 = data1[data1['pep2_num_matched'] != 0]
        data1 = data1[data1['Ms1.Profile.Corr'] > 0.8]
        data1 = data1[data1['inten_ratio'] > 1]
        score = self.FDR_to_score(data1, 0.05)
        data2 = data1[data1['type2'] == 'exp']
        print('total_exp_TT:', list(data2['type']).count('TT'))
        data3 = data2[data2['ml_score'] > score]
        print('now_exp_TT:', list(data3['type']).count('TT'))
        type = list(data2['type'])
        print('old_FDR:', (type.count('TD') - type.count('DD')) / type.count('TT'))
        type = list(data3['type'])
        print('new_FDR:', (type.count('TD') - type.count('DD')) / type.count('TT'))
        data2 = data1[data1['type'] == 'TT']
        data3 = data2[data2['ml_score'] > score]
        type1 = list(data3['type2'])
        print('test_num:', type1.count('test'))
        data4 = data3[data3['type2'] == 'test']
        return data1, data4
