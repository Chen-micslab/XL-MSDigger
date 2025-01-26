import numpy as np
import pandas as pd
import os

from Deep4D_XL.dataset.aa_onehot import onehot
from Deep4D_XL.dataset.mass_cal import mz_cal as m
###该程序用来将肽序列转换为相应的one-hot矩阵，并每个肽的one-hot矩阵存为一个单独的npy文件，相应的ccs值乘以10000并储存在文件名的两个_中间。

def extract(str,symbol):  ###定义一个函数来提取相同的开始符和结束符中间的字符串
    index = []
    for i in range(len(str)):
        if str[i]==symbol:
            index.append(i)
    start_index = index[0]
    end_index = index[1]
    str1 = str[(start_index+1):end_index]
    return str1

def get_position_angle_vec(position,dim): ##position encoding,sin和cos函数里面的值
    return [position / np.power(10000, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]

def get_onehot(pep, cl_site):
    peptide = pep
    peptide_np = np.zeros((50, 23))  ###创建一个50*23的0矩阵来存放peptide
    if 'U' in peptide:
        peptide = peptide.replace('U', '')
    if peptide[0] == 'a':  ##判断peptide序列的第一位是不是N端乙酰化修饰，如果是，那么one-hot编码就从第二位开始，求得肽段长度
        x = 1
        pep_len = len(peptide) - 1
    else:
        x = 0
        pep_len = len(peptide)
    i = 0
    for j in range(x, len(peptide)):
        peptide_np[i, :] = onehot.AA_onehot[peptide[j]]
        i = i + 1
    if peptide[0] == 'a':  ##再次判断如果第一位是a（乙酰化N端修饰），那么对one-hot矩阵的第一个氨基酸加入乙酰化的信息。
        peptide_np[0, 20] = 1
    peptide_np[int(cl_site-1), 22] = 1
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i, 23) for pos_i in range(pep_len)])  # 先计算position encoding sin和cos里面的值
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i  按照偶数来算sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1  按照奇数算cos
    peptide_np[:pep_len, ] = peptide_np[:pep_len, ] + sinusoid_table  ###将position encoding和one-hot合在一起
    return  peptide_np, pep_len

def crosslink_ccs_encoding_with_label(input):
    data = pd.read_csv(input, header=0)  ##读入数据
    output = input.rsplit('/', 1)[0] + '/' + 'feature_encoding'
    os.mkdir(f'{output}/ccs')
    peptide = np.array(data['combine_peptide'])
    peptide1 = np.array(data['peptide1'])
    peptide2 = np.array(data['peptide2'])
    site1 = np.array(data['site1'])
    site2 = np.array(data['site2'])
    CCS = np.array(data['ccs'])
    charge = np.array(data['charge'])
    peptide_num = len(peptide1)  ##计算肽段个数
    a = m()
    for h in range(peptide_num):
        if h % 200 == 0:
            print(h)
        pep = peptide[h]
        pep1 = peptide1[h]
        pep2 = peptide2[h]
        s1 = site1[h]
        s2 = site2[h]
        peptide_ccs = int(CCS[h] * 10000)
        peptide_charge = charge[h]
        if (len(pep1) < 50) and (len(pep2) < 50):
            pep1_np, pep1_len = get_onehot(pep1, s1)
            pep2_np, pep2_len = get_onehot(pep2, s2)
            mz = a.crosslink_peptide_m_z(pep, peptide_charge, 'DSS')
            peptide_np = np.row_stack((pep1_np, pep2_np))
            lenth = len(str(peptide_num))
            num = str(h).zfill(lenth)
            np.save(f'{output}/ccs/{num}%{mz}%_{pep}_*{peptide_ccs}*${pep1_len}$@{pep2_len}@#{peptide_charge}#.npy',peptide_np)  ##将每个肽的one-hot矩阵单独存为一个npy文件,名字中包含CCS值、肽段长度和charge

def crosslink_ccs_encoding_with_label_candidate(input):
    data = pd.read_csv(input, header=0)  ##读入数据
    output = input.rsplit('/', 1)[0] + '/' + 'candidate_feature_encoding'
    os.mkdir(f'{output}/ccs')
    order_list = np.array(data['Order'])
    peptide = np.array(data['combine_peptide'])
    peptide1 = np.array(data['peptide1'])
    peptide2 = np.array(data['peptide2'])
    site1 = np.array(data['site1'])
    site2 = np.array(data['site2'])
    CCS = np.array(data['ccs'])
    charge = np.array(data['charge'])
    peptide_num = len(peptide1)  ##计算肽段个数
    a = m()
    for h in range(peptide_num):
        if h % 200 == 0:
            print(h)
        order = order_list[h]
        pep = peptide[h]
        pep1 = peptide1[h]
        pep2 = peptide2[h]
        s1 = site1[h]
        s2 = site2[h]
        peptide_ccs = int(CCS[h] * 10000)
        peptide_charge = charge[h]
        if (len(pep1) < 50) and (len(pep2) < 50):
            pep1_np, pep1_len = get_onehot(pep1, s1)
            pep2_np, pep2_len = get_onehot(pep2, s2)
            mz = a.crosslink_peptide_m_z(pep, peptide_charge, 'DSS')
            peptide_np = np.row_stack((pep1_np, pep2_np))
            lenth = len(str(peptide_num))
            num = str(h).zfill(lenth)
            np.save(f'{output}/ccs/{num}&{order}&%{mz}%_{pep}_*{peptide_ccs}*${pep1_len}$@{pep2_len}@#{peptide_charge}#.npy',peptide_np)  ##将每个肽的one-hot矩阵单独存为一个npy文件,名字中包含CCS值、肽段长度和charge

def crosslink_ccs_encoding_without_label(input,output):
    data = pd.read_csv(input, header=0)  ##读入数据
    os.mkdir(output)  ###创建文件存放路径
    peptide = np.array(data['combine_peptide'])
    peptide1 = np.array(data['peptide1'])
    peptide2 = np.array(data['peptide2'])
    site1 = np.array(data['site1'])
    site2 = np.array(data['site2'])
    charge = np.array(data['charge'])
    peptide_num = len(peptide1)  ##计算肽段个数
    for h in range(peptide_num):
        if h % 200 == 0:
            print(h)
        pep = peptide[h]
        pep1 = peptide1[h]
        pep2 = peptide2[h]
        s1 = site1[h]
        s2 = site2[h]
        peptide_charge = charge[h]
        if (len(pep1) < 50) and (len(pep2) < 50):
            pep1_np, pep1_len = get_onehot(pep1, s1)
            pep2_np, pep2_len = get_onehot(pep2, s2)
            mz = a.crosslink_peptide_m_z(pep, peptide_charge, 'DSS')
            peptide_np = np.row_stack((pep1_np, pep2_np))
            lenth = len(str(peptide_num))
            num = str(h).zfill(lenth)
            np.save(f'{output}/{num}%{mz}%_{pep}_${pep1_len}$@{pep2_len}@#{peptide_charge}#.npy',peptide_np)  ##将每个肽的one-hot矩阵单独存为一个npy文件,名字中包含CCS值、肽段长度和charge

####这里的data应该只包含两列数据0
if __name__ == '__main__':
    get_onehot_file('./data/ribosome_sample2.csv','./data/ribosome_sample2')
