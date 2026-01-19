from torch.utils.data import Dataset,DataLoader
import numpy as np
from os import listdir
import os

class Mydata_label(Dataset):
    def __init__(self,data_dir):
        self.data_dir = f'{data_dir}/rt'                     
        self.allpeptide = listdir(self.data_dir)                    
        self.lenth = len(str(len(self.allpeptide)))             
        self.allpeptide.sort(key=lambda x: int(x[:self.lenth]))                     

    def extract(self,str, symbol):                                              
        index = []
        for i in range(len(str)):
            if str[i] == symbol:
                index.append(i)
        start_index = index[0]
        end_index = index[1]
        str1 = str[(start_index + 1):end_index]
        return str1

    def __getitem__(self, index):
        peptide_index = self.allpeptide[index]                          
        peptide_dir = os.path.join(self.data_dir,peptide_index)           
        peptide = np.load(peptide_dir)                                     
        peptide1 = peptide[:50, :]
        peptide2 = peptide[50:, :]
        peptide_seq = self.extract(peptide_index, '_')        
        RT = int(self.extract(peptide_index,'*'))/10000                                      
        charge = int(self.extract(peptide_index, '#'))
        pep1_len = int(self.extract(peptide_index,'$'))
        pep2_len = int(self.extract(peptide_index,'@'))
        if '&' in peptide_index:                                        
            order = self.extract(peptide_index, '&')
            sample = {'peptide': peptide_seq, 'peptide1':peptide1, 'peptide2':peptide2, 
                    'RT':RT, 'len1':pep1_len, 'len2':pep2_len, 'charge': charge, 'order':order}
        else:
            sample = {'peptide': peptide_seq, 'peptide1':peptide1, 'peptide2':peptide2, 
                    'RT':RT, 'len1':pep1_len, 'len2':pep2_len, 'charge': charge}
        return sample

    def __len__(self):
        return len(self.allpeptide)


class Mydata_nolabel(Dataset):
    
    def __init__(self, datadir):
        self.onehot_dir = f'{datadir}_onehot'                      
        self.allpeptide = listdir(self.onehot_dir)                    
        self.lenth = len(str(len(self.allpeptide)))             
        self.allpeptide.sort(key=lambda x: int(x[:self.lenth]))                     

    def extract(self, str, symbol):                                              
        index = []
        for i in range(len(str)):
            if str[i] == symbol:
                index.append(i)
        start_index = index[0]
        end_index = index[1]
        str1 = str[(start_index + 1):end_index]
        return str1

    def __getitem__(self, index):
        peptide_index = self.allpeptide[index]                          
        peptide_onehot_dir = os.path.join(self.onehot_dir, peptide_index)            
        peptide_onehot = np.load(peptide_onehot_dir)                                            
        peptide1 = peptide_onehot[:50, :]
        peptide2 = peptide_onehot[50:, :]
        peptide_seq = self.extract(peptide_index, '_')        
        charge = int(self.extract(peptide_index, '#'))
        pep1_len = int(self.extract(peptide_index,'$'))
        pep2_len = int(self.extract(peptide_index,'@'))
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



