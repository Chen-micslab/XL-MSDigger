import  logging
import os
from torch.utils.data import Dataset
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
from DDA_rescore.calculate_FDR import FDR_calculate as FDR

class DNN_parameter:
    para = {'epoch':[10,30],
            'cv':[2,3],
            'max_train_sample':[ 1000, 1500, 2000],
            'lr':[0.05,0.03, 0.01, 0.008,0.005, 0.003,0.001]}

class SVM_parameter:
    para = {'cv':[2,3],
            'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
            'c':[10, 50, 100, 200],
            'max_train_sample':[1000, 1500, 2000]}

class Mydata_label(Dataset):
    def __init__(self, feature, label):
        self.feature = np.array(feature)  ##这里data是输入的feature dataframe
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
        self.feature = np.array(feature)  ##这里data是输入的feature dataframe

    def __getitem__(self, index):
        feature = self.feature[index]
        sample = {'feature': feature}
        return sample

    def __len__(self):
        return len(self.feature)

class DNN_model(Module):

    def __init__(self, feature_num = 15):
        super(DNN_model, self).__init__()
        self.input = feature_num
        self.linear1 = Linear(self.input, 100)
        self.linear2 = Linear(100, 1000)
        self.linear3 = Linear(1000, 100)
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

class Rescore_DNN():
    ###设置模型训练时的细节
    def eval_model(self, model, loader, device):
        model.eval()  ##将model调整为eval模式
        n_val = len(loader)  # val中的batch数量
        ex= torch.tensor([]).to(device=device, dtype=torch.float32)
        pre = torch.tensor([]).to(device=device, dtype=torch.float32)
        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for batch in loader:
                feature = batch['feature'].to(device=device, dtype=torch.float32)
                label = batch['label'].to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    score_pre = model(feature).view(label.size(0))  ##将数据送入model，得到预测的ccs
                ex = torch.cat((ex, label), 0)
                pre = torch.cat((pre, score_pre), 0)
                pbar.update()
        model.train()  ##将模型调回train模式
        tot = torch.abs(ex - pre)
        mean_error = tot.mean()
        median_error = tot.median()
        return mean_error, median_error  ##返回loss的均值以及MRE

    def train_DNN(self, model, device, epochs=None, batch_size=50, lr=0.001, val_percent=0.1, save_mp=True, checkpoint_dir=None, feature =None, label = None):
        mydata = Mydata_label(feature, label)  ##导入dataset
        n_val = int(len(mydata) * val_percent)  ##计算validation data的数量
        n_train = len(mydata) - n_val  ##计算train data的数量
        train_data, val_data = random_split(mydata, [n_train, n_val])  ##随机分配validation data和train data
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)  ##生成train data
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,)  ##生成validation data
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
        optimizer =optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999),eps=1e-08, weight_decay=0, amsgrad=False) ##选择optimizer为Adam,注意传入的是model的parameters
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=10,T_mult=2,eta_min=0.000001)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=0.5)
        best_index = 1
        for epoch in range(epochs):
            model.train()  ##设置模型为train模式
            local_step = 0
            epoch_loss = 0  ##用于存储一个epoch中loss之和
            with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}',unit='peptide') as pbar: ##使用tqdm来显示训练的进度条
                for batch in train_loader:
                    local_step += 1
                    feature = batch['feature'].to(device=device, dtype=torch.float32)
                    label = batch['label'].to(device=device, dtype=torch.float32)
                    score_pre = model(feature).view(label.size(0))  ##将数据送入model，得到预测的ccs
                    loss_f = nn.MSELoss()
                    loss = loss_f(label,score_pre)  ##计算ccs预测值与真实值的均方根误差
                    epoch_loss += loss.item()
                    writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})  #输入一个字典，来显示实验指标
                    optimizer.zero_grad()  ##梯度清零
                    loss.backward()  ##反向传播
                    optimizer.step()
                    pbar.update(label.shape[0])
                    global_step += 1
                    if global_step % (n_train // (2 * batch_size)) == 0:
                        val_MeanE,val_MedianE = self.eval_model(model,val_loader,device)
                        writer.add_scalar('learning rate',optimizer.param_groups[0]['lr'], global_step)
                        logging.info('Validation MeanE: {}'.format(val_MeanE))
                        logging.info('Validation MedianE: {}'.format(val_MedianE))
                        if val_MedianE < best_index:
                                torch.save(model.state_dict(),
                                           checkpoint_dir + f'model_param_epoch{epoch + 1}MedianE{val_MedianE}.pth')  ###只保存模型的参数到checkpoint_dir
                                para_dir = checkpoint_dir + f'model_param_epoch{epoch + 1}MedianE{val_MedianE}.pth'
                                logging.info(f'Checkpoint {epoch + 1}global_step{global_step} saved !')
                                best_index = val_MedianE
            scheduler.step()
            if save_mp:
                logging.info('Created checkpoint directory')
                torch.save(model.state_dict(), checkpoint_dir + f'model_param_epoch{epoch + 1}.pth')###只保存模型的参数到checkpoint_dir
                logging.info(f'Checkpoint {epoch + 1} saved !')
        writer.close()
        best_index = float(best_index.to('cpu'))
        print(best_index)
        return para_dir, best_index

    def do_train(self, args, feature, label, epoch, lr, feature_dir):
        checkpoint_dir = feature_dir.rsplit('/', 1)[0] + '/' + 'checkpoint/dda_rescore/' ##checkpoint文件夹的路径，用于保存模型参数
        if os.path.exists(checkpoint_dir):
            pass
        else: os.makedirs(checkpoint_dir)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ##判断使用GPU还是CPU
        print(device)
        logging.info(f'Using device {device}')
        model = DNN_model()  ###生成模型
        if args.rescore_model_parameter:
            model.load_state_dict(torch.load(args.rescore_model_parameter, map_location=device))
            logging.info(f'Model parameters loaded from {args.rescore_model_parameter}')
        model.to(device=device)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        para_dir, best_index = self.train_DNN(model=model,
                          device=device,
                          epochs=epoch,
                          batch_size=args.rescore_batch_size,
                          lr=lr,
                          val_percent=args.rescore_vali_rate,
                          save_mp=True,
                          checkpoint_dir=checkpoint_dir,
                          feature = feature, label = label)
        return para_dir, best_index, checkpoint_dir

    def do_predict(self, para_dir, test_feature, batch_size):
        model = DNN_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(para_dir, map_location=device, weights_only=True))
        logging.info(f'Model parameters loaded from {para_dir}')
        test_data = Mydata_nolabel(test_feature)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
        n_test = len(test_loader)
        model.to(device=device)
        model.eval()  ##将model调整为eval模式
        pre = torch.tensor([]).to(device=device, dtype=torch.float32)
        with tqdm(total=n_test, desc='Validation round', unit='batch', leave=False) as pbar:
            for batch in test_loader:
                feature = batch['feature'].to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    score_pre = model(feature).view(feature.size(0))  ##将数据送入model，得到预测的ccs
                pre = torch.cat((pre, score_pre), 0)
                pbar.update()
        pre = pre.to(device='cpu', dtype=torch.float32).numpy()
        pre = list(pre)
        return pre

    def cv_score(self, args, data, feature_dir): ###data提取后的feature信息
        print(data)
        feature_list = ['Charge', 'Re-score_CSM', 'len1', 'len2', 'rt_AE', 'match_num', 'match_num1', 'match_num2',
                        'both_m_p_num', 'both_m_p_num1', 'both_m_p_num2', 'cosine', 'SA', 'pearson', 'spearman']
        # feature_list = ['Charge', 'SVM_Score', 'len1', 'len2', 'rt_AE', 'ccs_RE', 'match_num', 'match_num1', 'match_num2',
        #                 'both_m_p_num', 'both_m_p_num1', 'both_m_p_num2', 'cosine']
        # feature_list = ['Charge', 'SVM_Score', 'len1', 'len2', 'rt_AE', 'ccs_RE', 'match_num', 'match_num1', 'match_num2',
        #                 'both_m_p_num', 'both_m_p_num1', 'both_m_p_num2', 'cosine']
        data_target = data[data['Target_Decoy'] == 2]
        data_decoy =  data[data['Target_Decoy'] != 2]
        cv_list = DNN_parameter.para['cv']
        max_trian_number_list = DNN_parameter.para['max_train_sample']
        epoch_list = DNN_parameter.para['epoch']
        lr_list = DNN_parameter.para['lr']
        import os
        # 获取当前工作目录
        current_directory = os.getcwd()
        # 创建名为"temp"的文件夹
        temp_directory = os.path.join(current_directory, "temp")
        # 检查文件夹是否存在，如果不存在则创建它
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)
        new, old, cross, sen, feature1, feature2, feature3, feature4, validation_m = [], [], [] ,[], [], [], [] ,[] ,[]
        for cv in cv_list:
            for epoch in epoch_list:
                for lr in lr_list:
                    for max_trian_number in max_trian_number_list:
                        if cv > 1:
                            test_df_list = []
                            vali = 0
                            for i in range(cv):
                                t_mask = np.ones(len(data_target), dtype=bool)
                                slice_num = slice(i, len(data_target), cv)
                                t_mask[slice_num] = False
                                cv_df_target = data_target[t_mask]
                                train_t_df = cv_df_target[cv_df_target['Q-value_CSM'] <= args.rescore_fdr]
                                test_t_df = data_target[slice_num]

                                d_mask = np.ones(len(data_decoy), dtype=bool)
                                slice_num = slice(i, len(data_decoy), cv)
                                d_mask[slice_num] = False
                                train_d_df = data_decoy[d_mask]
                                test_d_df = data_decoy[slice_num]
                                if len(train_t_df) > max_trian_number:
                                    train_t_df = train_t_df.sample(n=max_trian_number, random_state=123)
                                if len(train_d_df) > max_trian_number:
                                    train_d_df = train_d_df.sample(n=max_trian_number, random_state=123)
                                print(len(train_t_df['Peptide']))
                                print(len(train_d_df['Peptide']))
                                train_df = pd.concat((train_t_df, train_d_df))
                                train_label = np.ones(len(train_df), dtype=np.int32)
                                train_label[len(train_t_df):] = 0
                                feature_train = train_df[feature_list]
                                feature_train = feature_train.fillna(0)
                                para_dir, best_index, checkpoint_dir = self.do_train(args, feature_train, train_label, epoch, lr, feature_dir)
                                vali = vali + best_index
                                test_df = pd.concat((test_t_df, test_d_df))
                                feature_test = test_df[feature_list]
                                feature_test = feature_test.fillna(0)
                                pre = self.do_predict(para_dir, feature_test, args.rescore_batch_size)
                                test_df.insert(0, 'ml_score', pre)
                                test_df_list.append(test_df)
                            data1 = pd.concat(test_df_list)
                            data2 = data1[data1['Protein_Type'] == 2]
                            F = FDR()
                            data2 = F.crosslink_FDR_plink(data2, col_name='ml_score')
                            data3 = data2[data2['FDR'] < args.rescore_fdr]
                            data2.to_csv(f'{temp_directory}/{cv}_{epoch}_{lr}_{max_trian_number}.csv', index=False)
                            new_num = len(data3['FDR'])
                            data4 = data2[data2['Q-value_CSM'] < args.rescore_fdr]
                            old_num = len(data4['Q-value_CSM'])
                            a = set(data3['Order'])
                            b = set(data4['Order'])
                            c = a & b
                            cross_num = len(c)
                            if old_num != 0:
                                sensitivity = cross_num/old_num
                            else:
                                sensitivity = 1
                            new.append(new_num)
                            old.append(old_num)
                            cross.append(cross_num)
                            sen.append(sensitivity)
                            feature1.append(cv)
                            feature2.append(epoch)
                            feature3.append(lr)
                            feature4.append(max_trian_number)
                            validation_m.append(vali/cv)
        data5 = pd.DataFrame()
        data5['cv'] = feature1
        data5['epoch'] = feature2
        data5['lr'] = feature3
        data5['max_trian_number'] = feature4
        data5['old_number'] = old
        data5['new_number'] = new
        data5['cross_number'] = cross
        data5['sensitivity'] = sen
        data5['vali_medianRE'] = validation_m
        data6 = choose_parameter(data5)
        cv = list(data6['cv'])[0]
        epoch = list(data6['epoch'])[0]
        lr = list(data6['lr'])[0]
        max_trian_number = list(data6['max_trian_number'])[0]
        data7 = pd.read_csv(f'{temp_directory}/{cv}_{epoch}_{lr}_{max_trian_number}.csv')
        import shutil
        shutil.rmtree(checkpoint_dir)
        shutil.rmtree(temp_directory)
        return data5, data7

    def merge_old_with_new(self, data_old, data_new):
        name = list(data_old)
        data_new1 = data_new[name]
        data_old = data_old[data_old['Q-value_CSM'] < 0.01]
        data_old = data_old[data_old['Protein_Type'] == 1]
        data_old1 = data_old[data_old['Target_Decoy'] == 2]
        data = pd.concat([data_new1, data_old1])
        return data

    def run(self, args, data, feature_dir):
        data1, data2 = self.cv_score(args, data, feature_dir)  ##data1是参数，data2是重打分结果
        return data2

class Rescore_SVM():
    def cv_score(self, data):  ###data提取后的feature信息
        # feature_list = ['Charge', 'SVM_Score', 'len1', 'len2', 'rt_AE', 'ccs_RE', 'match_num', 'match_num1', 'match_num2',
        #                 'both_m_p_num', 'both_m_p_num1', 'both_m_p_num2', 'cosine', 'SA', 'pearson', 'spearman']
        feature_list = ['Charge', 'Re-score_CSM', 'len1', 'len2', 'rt_AE', 'match_num', 'match_num1', 'match_num2',
                        'both_m_p_num', 'both_m_p_num1', 'both_m_p_num2', 'cosine', 'SA', 'pearson', 'spearman']

        data_target = data[data['Target_Decoy'] == 2]
        data_decoy = data[data['Target_Decoy'] != 2]
        cv_list = SVM_parameter.para['cv']
        gamma_list = SVM_parameter.para['gamma']
        c_list = SVM_parameter.para['c']
        max_trian_number_list = SVM_parameter.para['max_train_sample']
        import os
        # 获取当前工作目录
        current_directory = os.getcwd()
        # 创建名为"temp"的文件夹
        temp_directory = os.path.join(current_directory, "temp")
        # 检查文件夹是否存在，如果不存在则创建它
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)
        new, old, cross, sen, feature1, feature2, feature3, feature4 = [], [], [], [], [], [], [], []
        for cv_fold in cv_list:
            for gamma in gamma_list:
                for c_svm in c_list:
                    for max_trian_number in max_trian_number_list:
                        if cv_fold > 1:
                            test_df_list = []
                            for i in range(cv_fold):
                                print(i)
                                t_mask = np.ones(len(data_target), dtype=bool)
                                slice_num = slice(i, len(data_target), cv_fold)
                                t_mask[slice_num] = False
                                cv_df_target = data_target[t_mask]
                                train_t_df = cv_df_target[cv_df_target['Q-value_CSM'] < 0.01]
                                test_t_df = data_target[slice_num]

                                d_mask = np.ones(len(data_decoy), dtype=bool)
                                slice_num = slice(i, len(data_decoy), cv_fold)
                                d_mask[slice_num] = False
                                train_d_df = data_decoy[d_mask]
                                test_d_df = data_decoy[slice_num]
                                model = self.train(train_t_df, train_d_df, feature_list, max_trian_number, gamma, c_svm)
                                test_df = pd.concat((test_t_df, test_d_df))
                                test_df_list.append(self.predict(test_df, model, feature_list))
                            data1 = pd.concat(test_df_list)
                            data2 = data1[data1['Protein_Type'] == 2]
                            F = FDR()
                            data2 = F.crosslink_FDR_plink(data2, col_name='ml_score')
                            data3 = data2[data2['FDR'] < 0.01]
                            data2.to_csv(f'{temp_directory}/{cv_fold}_{gamma}_{c_svm}_{max_trian_number}.csv',
                                         index=False)
                            new_num = len(data3['FDR'])
                            data4 = data2[data2['Q-value_CSM'] < 0.01]
                            old_num = len(data4['Q-value_CSM'])
                            a = set(data3['Order'])
                            b = set(data4['Order'])
                            c = a & b
                            cross_num = len(c)
                            if old_num != 0:
                                sensitivity = cross_num / old_num
                            else:
                                sensitivity = 1
                            new.append(new_num)
                            old.append(old_num)
                            cross.append(cross_num)
                            sen.append(sensitivity)
                            feature1.append(cv_fold)
                            feature2.append(gamma)
                            feature3.append(c_svm)
                            feature4.append(max_trian_number)

        data5 = pd.DataFrame()
        data5['cv'] = feature1
        data5['gamma'] = feature2
        data5['c'] = feature3
        data5['max_trian_number'] = feature4
        data5['old_number'] = old
        data5['new_number'] = new
        data5['cross_number'] = cross
        data5['sensitivity'] = sen
        # data5['vali_medianRE'] = validation_m
        data6 = choose_parameter(data5)
        cv_fold = list(data6['cv'])[0]
        gamma = list(data6['gamma'])[0]
        c = list(data6['c'])[0]
        max_trian_number = list(data6['max_trian_number'])[0]
        data7 = pd.read_csv(f'{temp_directory}/{cv_fold}_{gamma}_{c}_{max_trian_number}.csv')
        import shutil
        shutil.rmtree(temp_directory)
        return data5, data7

    def train(self, train_t_df, train_d_df, feature_list, train_number, gamma, c):
        if len(train_t_df) > train_number:
            train_t_df = train_t_df.sample(n=train_number, random_state=123)
        if len(train_d_df) > train_number:
            train_d_df = train_d_df.sample(n=train_number, random_state=123)

        train_df = pd.concat((train_t_df, train_d_df))
        train_label = np.ones(len(train_df), dtype=np.int32)
        train_label[len(train_t_df):] = 0
        x = train_df[feature_list]
        x = x.infer_objects()  # 尝试推断对象类型
        x = x.fillna(0)
        # st = StandardScaler().fit(x)
        # x = st.transform(x)
        # pca = PCA(n_components=0.99)
        # pca.fit(x)
        # x = pca.transform(x)
        from sklearn.linear_model import LogisticRegression as lg
        from sklearn.svm import SVR
        # model = xgb.XGBRegressor(n_estimators=5000).fit(train_df[feature_list].values, train_label)
        # model = lg()
        model = SVR(gamma=gamma, C=c, kernel='rbf')
        # np.set_printoptions(linewidth=1000)
        model = model.fit(x, train_label)
        return model

    def predict(self, test_df, model, feature_list):
        x = test_df[feature_list]
        x = x.fillna(0)
        # st = StandardScaler().fit(x)
        # x = st.transform(x)
        # pca = PCA(n_components=0.99)
        # pca.fit(x)
        # x = pca.transform(x)
        # test_df['ml_score'] = model.predict(x)
        test_df.insert(0, 'ml_score', model.predict(x))
        return test_df

    def run(self, data):
        data1, data2 = self.cv_score(data)
        return data2

def choose_parameter(data):
    if (data['sensitivity'] >= 0.8).any():
        data1 = data[data['sensitivity'] >= 0.8]
        max_value = data1['new_number'].max()
        result_rows = data[data['new_number'] == max_value].head(1)
    else:
        max_value = data['sensitivity'].max()
        result_rows = data[data['sensitivity'] == max_value].head(1)
    return result_rows
