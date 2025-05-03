from torch.utils.data import Dataset
import numpy as np
from os import listdir
import os

class Mydata_label(Dataset):
    def __init__(self, datadir):
        self.msms_dir = f'{datadir}/msms' ##这里data_dir是文件的储存路径
        self.onehot_dir = f'{datadir}/onehot'  ##这里data_dir是文件的储存路径
        self.allpeptide = listdir(self.onehot_dir)  ##储存每一个peptide的文件名
        self.lenth = len(str(len(self.allpeptide)))  ###获得肽总数的位数
        self.allpeptide.sort(key=lambda x: int(x[:self.lenth]))  # 将所有文件名，按照序号从小到大排序

    def extract(self,str, symbol):  ###定义一个函数来提取相同的开始符和结束符中间的字符串,symbol可以设置为任意字符
        index = []
        for i in range(len(str)):
            if str[i] == symbol:
                index.append(i)
        start_index = index[0]
        end_index = index[1]
        str1 = str[(start_index + 1):end_index]
        return str1

    def __getitem__(self, index):     
        peptide_index = self.allpeptide[index]  ##按照index提取任意peptide的文件名
        peptide_msms_dir = os.path.join(self.msms_dir, peptide_index)  ##拼接文件名和路径
        peptide_onehot_dir = os.path.join(self.onehot_dir,peptide_index) ##拼接文件名和路径
        peptide_onehot = np.load(peptide_onehot_dir)  ##按照peptide_onehot的文件路径读取peptide的one-hot矩阵
        peptide1 = peptide_onehot[:50, :]
        peptide2 = peptide_onehot[50:, :]
        peptide_msms = np.load(peptide_msms_dir)  ##按照peptide_msms的文件路径读取peptide的msms矩阵
        peptide = self.extract(peptide_index, '_') ###交联肽序列
        # site1_np, site2_np = self.extract_from_combine_peptide(peptide)
        pep1_len = int(self.extract(peptide_index, '$')) ##通过自定义的extract函数提取出peptide文件名中的肽段长度
        pep2_len = int(self.extract(peptide_index, '@'))  ##通过自定义的extract函数提取出peptide文件名中的肽段长度n  
        peptide_charge = int(self.extract(peptide_index,'#'))
        if '&' in peptide_index:    ####只在DDA rescore中，预测所有candidate时候使用
            order = self.extract(peptide_index, '&')
            sample = {'peptide': peptide, 'peptide1': peptide1, 'peptide2': peptide2, 'peptide_msms':peptide_msms,
                  'len1':pep1_len, 'len2': pep2_len, 'charge':peptide_charge, 'order':order}
        else:
            sample = {'peptide': peptide, 'peptide1': peptide1, 'peptide2': peptide2, 'peptide_msms':peptide_msms,
                  'len1':pep1_len, 'len2': pep2_len, 'charge':peptide_charge}
        return sample

    def __len__(self):
        return len(self.allpeptide)

class Mydata_nolabel(Dataset):

    def __init__(self, datadir):
        self.onehot_dir = f'{datadir}_onehot'  ##这里data_dir是文件的储存路径
        self.allpeptide = listdir(self.onehot_dir)  ##储存每一个peptide的文件名
        self.lenth = len(str(len(self.allpeptide)))  ###获得肽总数的位数
        self.allpeptide.sort(key=lambda x: int(x[:self.lenth]))  # 将所有文件名，按照序号从小到大排序

    def extract(self, str, symbol):  ###定义一个函数来提取相同的开始符和结束符中间的字符串,symbol可以设置为任意字符
        index = []
        for i in range(len(str)):
            if str[i] == symbol:
                index.append(i)
        start_index = index[0]
        end_index = index[1]
        str1 = str[(start_index + 1):end_index]
        return str1

    def __getitem__(self, index):
        peptide_index = self.allpeptide[index]  ##按照index提取任意peptide的文件名
        peptide_onehot_dir = os.path.join(self.onehot_dir, peptide_index)  ##拼接文件名和路径
        peptide_onehot = np.load(peptide_onehot_dir)  ##按照peptide_onehot的文件路径读取peptide的one-hot矩阵
        peptide1 = peptide_onehot[:50, :]
        peptide2 = peptide_onehot[50:, :]
        peptide_seq = self.extract(peptide_index, '_')  ###交联肽序列
        pep1_len = int(self.extract(peptide_index, '$'))  ##通过自定义的extract函数提取出peptide文件名中的肽段长度
        pep2_len = int(self.extract(peptide_index, '@'))  ##通过自定义的extract函数提取出peptide文件名中的肽段长度
        charge = int(self.extract(peptide_index, '#'))
        sample = {'peptide': peptide_seq, 'peptide1': peptide1, 'peptide2': peptide2, 'len1': pep1_len, 'len2': pep2_len,
                  'charge': charge}
        return sample

    def __len__(self):
        return len(self.allpeptide)

class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)






