import numpy as np
import pandas as pd
from mass_cal import mz_cal as m
import os

class generate_potential_XL_peptide():####从ligand-receptors pairs（受体配体对）生成可能的交联肽,
    def generate_peptide_pair(self, fasta_file, genename_file):  ####从ligand-receptors pairs（受体配体对）生成可能的交联肽
        protein_dict = self.readFa_from_label(fasta_file, 'GN=')  ###key是基因名，value是蛋白序列
        name = list(protein_dict.keys())
        data = pd.read_csv(genename_file)
        # protein1_list = data['Ligand.ApprovedSymbol']
        # protein2_list = data['Receptor.ApprovedSymbol']
        protein1_list = data['Gene1']
        protein2_list = data['Gene2']
        peptide_list = []
        for i in range(len(protein1_list)):
            print(i)
            gene_1 = protein1_list[i]
            gene_2 = protein2_list[i]
            print(gene_1)
            if gene_1 in name:
                protein1 = protein_dict[gene_1]
                protein2 = protein_dict[gene_2]
                print(protein1)
                print(protein2)
                peptide1 = self.cut_protein(protein1)
                peptide2 = self.cut_protein(protein2)
                for pep1 in peptide1:
                    for pep2 in peptide2:
                        if ('K' in pep1) and ('K' in pep2):
                           if ('B' not in pep1) and ('J' not in pep1) and ('O' not in pep1) and ('U' not in pep1) and ('X' not in pep1) and ('Z' not in pep1):
                               if ('B' not in pep2) and ('J' not in pep2) and ('O' not in pep2) and ('U' not in pep2) and ('X' not in pep2) and ('Z' not in pep2):
                                    if len(pep1) > len(pep2):
                                        pep = pep1 + '-' + pep2
                                    else:
                                        pep = pep2 + '-' + pep1
                                    peptide_list.append(pep)
        peptide_list = set(peptide_list)
        print(len(peptide_list))
        final_file = self.generate_cl_csv(peptide_list)
        return final_file

    def generate_intra_peptide_pair(self, protein_list_dir, fasta_dir):
        protein = pd.read_csv(protein_list_dir)
        protein_list = protein['protein']
        fasta = self.extractFa_list(protein_list, fasta_dir)
        fasta_file = fasta_dir.split('.fasta')[0] + '_filter.fasta'
        fasta.to_csv(fasta_file, index=False, header=None)
        protein_dict = self.readFa(fasta_file)
        peptide_list, peptide_list1, protein_list = [], [], []
        i = 0
        for protein_name, protein in protein_dict.items():
            i = i + 1
            print(i)
            protein1 = protein
            print(len(protein1))
            peptide1 = self.cut_protein(protein1)
            peptide2 = peptide1
            for pep1 in peptide1:
                for pep2 in peptide2:
                    if ('K' in pep1) and ('K' in pep2):
                       if ('B' not in pep1) and ('J' not in pep1) and ('O' not in pep1) and ('U' not in pep1) and ('X' not in pep1) and ('Z' not in pep1):
                           if ('B' not in pep2) and ('J' not in pep2) and ('O' not in pep2) and ('U' not in pep2) and ('X' not in pep2) and ('Z' not in pep2):
                               if pep1 != pep2:
                                    if len(pep1) > len(pep2):
                                        pep = pep1 + '-' + pep2
                                    else:
                                        pep = pep2 + '-' + pep1
                                    peptide_list.append(pep)
                                    peptidex = pep.replace('-', 'X')
                                    find_all = lambda c, s: [x for x in range(c.find(s), len(c)) if c[x] == s]  ###函数，在字符串中寻找所有指定字符的index
                                    c = protein_name
                                    s = '|'  ##搜索的字符是空格符
                                    index_all = find_all(c, s)  ###所有空格符的index的lsit
                                    uniprot_id = protein_name[(index_all[0] + 1):index_all[1]]
                                    uniprot_id = uniprot_id + '-' + uniprot_id
                                    protein_list.append(uniprot_id)
                                    peptide_list1.append(peptidex)
        data = pd.DataFrame({'peptide':peptide_list1, 'protein':protein_list})
        dir = protein_list_dir.split('.csv')[0] + '_origin_intra_peptide&protein.csv'
        data.to_csv(dir, index=False)
        peptide_list = set(peptide_list)
        print(len(peptide_list))
        final_file = self.generate_cl_csv(peptide_list)
        os.remove(fasta_file)
        fasta_file_dir = protein_list_dir.split('.csv')[0] + '_intra_peptide.csv'
        final_file.to_csv(fasta_file_dir, index=False)

    def generate_inter_peptide_pair(self, protein_list_dir, fasta_dir):
        protein = pd.read_csv(protein_list_dir)
        protein1_list = protein['protein1']
        protein2_list = protein['protein2']
        peptide_list, peptide_list1, xl_protein_list = [], [], []
        for i in range(len(protein1_list)):
            print(i)
            pro1_id = protein1_list[i]
            pro2_id = protein2_list[i]
            xl_pro_id = pro1_id + '-' + pro2_id
            protein1 = self.extract_protein_by_uniprot_id(fasta_dir, pro1_id)
            protein2 = self.extract_protein_by_uniprot_id(fasta_dir, pro2_id)
            if (protein1 != -1) and (protein2 != -1):
                peptide1 = self.cut_protein(protein1)
                peptide2 = self.cut_protein(protein2)
                print(len(peptide1)*len(peptide2))
                for pep1 in peptide1:
                    for pep2 in peptide2:
                        if ('K' in pep1) and ('K' in pep2):
                            if ('B' not in pep1) and ('J' not in pep1) and ('O' not in pep1) and ('U' not in pep1) and ('X' not in pep1) and ('Z' not in pep1):
                                if ('B' not in pep2) and ('J' not in pep2) and ('O' not in pep2) and ('U' not in pep2) and ('X' not in pep2) and ('Z' not in pep2):
                                    if pep1 != pep2:
                                        if len(pep1) > len(pep2):
                                            pep = pep1 + '-' + pep2
                                        else:
                                            pep = pep2 + '-' + pep1
                                        peptide_list.append(pep)
                                        xl_protein_list.append(xl_pro_id)
                                        peptidex = pep.replace('-', 'X')
                                        peptide_list1.append(peptidex)
        data = pd.DataFrame({'peptide': peptide_list1, 'protein': xl_protein_list})
        dir = protein_list_dir.split('.csv')[0] + '_origin_inter_peptide&protein.csv'
        data.to_csv(dir, index=False)
        peptide_list = set(peptide_list)
        print(len(peptide_list))
        final_file = self.generate_cl_csv(peptide_list)
        fasta_file_dir = protein_list_dir.split('.csv')[0] + '_inter_peptide.csv'
        final_file.to_csv(fasta_file_dir, index=False)

    def find_cut_pos(self, pro, protease_type):  ####本函数用于计算蛋白序列中所有酶切位点的index
        if protease_type == "trypsin":
            index = []
            for x in range(len(pro)):
                if pro[x] == "K" or pro[x] == "R":
                    index.append(x)
        elif protease_type == "chymotrypsin":
            index = []
            for x in range(len(pro)):
                if pro[x] == "L" or pro[x] == "F" or pro[x] == "W" or pro[x] == "Y" or pro[x] == "y":
                    index.append(x)
        elif protease_type == "trypsin&chymotrypsin":
            index = []
            for x in range(len(pro)):
                if pro[x] == "K" or pro[x] == "R" or pro[x] == "L"  or pro[x] == "F" or pro[x] == "W" or pro[x] == "Y" or pro[x] == "y":
                    index.append(x)
        return [-1] + index + [len(pro)+1]

    def cut_protein(self, protein, miss_avg=2, protease_type='trypsin', detele_first=True):  ####本函数通过输入的蛋白序列和酶类型来生成理论的肽段
        if detele_first:
            protein = protein[1:]
        index = self.find_cut_pos(protein, protease_type)  ##获得每个酶切位点氨基酸index
        peplist = []  ##用于存放生成的多肽
        for i in range(len(index)):
            end = i + miss_avg + 2
            if i > (len(index) - miss_avg - 2):
                end = len(index)
            for j in range(i + 1, end):
                peptide = protein[(index[i] + 1):(index[j] + 1)]
                if len(peptide) >= 7 and len(peptide) < 20:
                    # if peptide[-2] != 'P':
                        peplist.append(peptide)
        #####生成N端的peptide
        if (len(index)-1) >= (miss_avg+1):
            site_n = miss_avg+1
        else:
            site_n = len(index)-1
        for i in range(site_n):
            pep = protein[:(index[i+1]+1)]
            if len(pep) >= 7 and len(pep) < 20:
                pep = '*' + pep
                peplist.append(pep)
        peplist = list(set(peplist))
        return peplist

    def readFa(self, fa):  ##### 本函数用于从fasta文件中读取所有蛋白序列，返回所有蛋白序列到一个list，所有蛋白名到一个list
        fr = open(fa, 'r')  ##读取文件
        sample = ""  ###用于在过程中储存单个蛋白序列
        samples = []  ###用于储存所有的蛋白
        samples_name = []
        before = ''
        for line in fr:
            if before == '':
                if '>sp' in line:
                    before = line
            else:
                if '>sp' in line:  ###如果识别到新的蛋白名字，则把储存在sample里的序列加入到list里面，但是这样会导致最后一个蛋白序列没法加入
                    samples_name.append(before)
                    samples.append(sample)
                    sample = ""
                    before = line
                else:
                    sample += line[:-1]
        samples_name.append(before)
        samples.append(sample)
        protein_dict = dict(zip(samples_name,samples))
        return protein_dict

    def extract_protein_by_uniprot_id(self, fasta_dir, uniprot_id):  ##### 本函数用于从fasta文件中读取所有蛋白名中包含指定label的蛋白序列，返回所有蛋白序列到一个list，所有蛋白名到一个list
        fr = open(fasta_dir, 'r')  ##读取文件
        sample = ""  ###用于在过程中储存单个蛋白序列
        proper = 0
        have_found = 0
        for line in fr:
            line = line.replace('\n', '')
            if '>sp' in line:
                find_all = lambda c, s: [x for x in range(c.find(s), len(c)) if c[x] == s]  ###函数，在字符串中寻找所有指定字符的index
                c = line
                s = '|'  ##搜索的字符是空格符
                index_all = find_all(c, s)  ###所有空格符的index的lsit
                label = line[(index_all[0]+1):index_all[1]]  ###从line中提取uniprot id,example: P54762
                if uniprot_id == label:
                    proper = 1
                    have_found = 1
                else:
                    proper = 0
            else:
                if proper == 1:
                    sample = sample + line
        if have_found:
            return sample
        else:
            print('No found fasta:', uniprot_id)
            return -1

    def readFa_from_label(self, fa, label):  ##### 本函数用于从fasta文件中读取所有蛋白名中包含指定label的蛋白序列，返回所有蛋白序列到一个list，所有蛋白名到一个list
        fr = open(fa, 'r')  ##读取文件
        sample = ""  ###用于在过程中储存单个蛋白序列
        samples = []  ###用于储存所有的蛋白
        samples_name = []
        before = ''
        for line in fr:
            if before == '':
                if line.startswith('>'):
                    before = line
            else:
                if line.startswith('>'):  ###如果识别到新的蛋白名字，则把储存在sample里的序列加入到list里面，但是这样会导致最后一个蛋白序列没法加入
                    if label in before:
                        b = before.index('GN=')
                        c = before.index('PE=')
                        samples_name.append(before[(b+3):(c-1)])
                        # if sample != "":
                        samples.append(sample)
                        sample = ""
                    else:
                        sample = ""
                    before = line
                else:
                    sample += line[:-1]
        if label in before:
            b = before.index('GN=')
            c = before.index('PE=')
            samples_name.append(before[(b + 3):(c - 1)])
            samples.append(sample)  ###将最后一个蛋白序列加入list
        protein_dict = dict(zip(samples_name,samples))
        return protein_dict

    def extractFa(self, id, fa):  ##### 本函数用于从fasta文件中读取所有protein_list中的蛋白序列，每个蛋白保存为一个元素，fa为文件路径
        # print(protein_list)
        fr = open(fa, 'r')  ##读取文件
        protein = []
        total = []  ###用于储存所有的蛋白
        proper = 0
        have_found = 0
        title = ''
        for line in fr:
            line = line.replace('\n', '')
            if line.startswith('>'):  ###如果识别到新的蛋白名字，则把储存在sample里的序列加入到list里面，但是这样会导致最后一个蛋白序列没法加入
                title = line
                # id1 = title.find('GN=')
                # id2 = title.find('PE=')
                # gene = title[(id1 + 3):(id2 - 1)]
                find_all = lambda c, s: [x for x in range(c.find(s), len(c)) if c[x] == s]  ###函数，在字符串中寻找所有指定字符的index
                c = line
                s = '|'  ##搜索的字符是空格符
                index_all = find_all(c, s)  ###所有空格符的index的lsit
                uniprot_id = line[(index_all[0]+1):index_all[1]]  ###从line中提取uniprot id,example: P54762
                if uniprot_id == id:
                    proper = 1
                    have_found = 1
                    protein.append(line)
                else:
                    proper = 0
            else:
                if proper == 1:
                    protein.append(line)
        if have_found:
            return protein
        else:
            print('No found fasta:',id)
            return -1

    def extractFa_list(self, id_list, fa):  ##### 本函数用于从fasta文件中读取所有protein_list中的蛋白序列，每个蛋白保存为一个元素，fa为文件路径
        # print(protein_list)
        fasta = []
        i = 0
        for id in id_list:
            i = i + 1
            print(i)
            if self.extractFa(id, fa) != -1:
                fasta = fasta + self.extractFa(id, fa)
        fasta = pd.DataFrame(fasta)
        return fasta

    def generate_cl_csv(self, peptide_list):
        a = m()
        peptide1_list, peptide2_list, site1_list, site2_list, charge_list, combine_pep, combine_pep_z, m_z_list, protein_list = [], [], [], [], [], [], [], [], []
        for z in [3, 4]:
            for peptide in peptide_list:
                id = peptide.find('-')
                pep1 = peptide[:id]
                pep2 = peptide[(id+1):]
                pep1_list, pep2_list = [], []
                if '*' in pep1:  #####N端也可以发生交联反应
                    s = 1
                    pep1 = pep1.replace('*', '')
                    pep1_1 = list(pep1)
                    pep1_1.insert(s, 'U')
                    pep1_1 = ''.join(pep1_1)
                    pep1_list.append(pep1_1)
                else:
                    pep1 = pep1.replace('*','')
                    site1 = [i+1 for i, char in enumerate(pep1) if char == 'K']
                    for s in site1:
                        pep1_1 = list(pep1)
                        pep1_1.insert(s, 'U')
                        pep1_1 = ''.join(pep1_1)
                        pep1_list.append(pep1_1)
                if '*' in pep2:
                    s = 1
                    pep2 = pep2.replace('*', '')
                    pep2_1 = list(pep2)
                    pep2_1.insert(s, 'U')
                    pep2_1 = ''.join(pep2_1)
                    pep2_list.append(pep2_1)
                else:
                    pep2 = pep2.replace('*','')
                    site2 = [i+1 for i, char in enumerate(pep2) if char == 'K']
                    for s in site2:
                        pep2_1 = list(pep2)
                        pep2_1.insert(s, 'U')
                        pep2_1 = ''.join(pep2_1)
                        pep2_list.append(pep2_1)
                for pep1 in pep1_list:
                    for pep2 in pep2_list:
                        peptide1 = pep1.replace('U','')
                        peptide2 = pep2.replace('U', '')
                        s1 = pep1.find('U')
                        s2 = pep2.find('U')
                        pep = pep1 + 'X' + pep2
                        pep_z = pep1 + 'X' + pep2 + str(z)
                        peptide1_list.append(peptide1)
                        peptide2_list.append(peptide2)
                        combine_pep.append(pep)
                        combine_pep_z.append(pep_z)
                        site1_list.append(s1)
                        site2_list.append(s2)
                        charge_list.append(z)
                        m_z_list.append(a.crosslink_peptide_m_z(pep, z, 'DSS'))
        data = pd.DataFrame()
        data['peptide1'] = peptide1_list
        data['peptide2'] = peptide2_list
        data['site1'] = site1_list
        data['site2'] = site2_list
        data['charge'] = charge_list
        data['combine_peptide'] = combine_pep
        data['combine_peptide_z'] = combine_pep_z
        data['m_z'] = m_z_list
        return data

    def test(self, protein_list_dir, fasta_dir):
        protein = pd.read_csv(protein_list_dir)
        protein_list = protein['protein']
        fasta = self.extractFa_list(protein_list, fasta_dir)
        fasta_file = fasta_dir.split('.fasta')[0] + '_filter.fasta'
        fasta.to_csv(fasta_file, index=False, header=None)
        protein_dict = self.readFa(fasta_file)
        print(protein_dict)
        aa_num = []
        for protein_name, protein_seq in protein_dict.items():
            len1 = len(protein_seq)
            aa_num.append(len1)
        protein['aa_num'] = aa_num
        return protein

def generate_theoritial_PPI(all_protein):
    protein1, protein2 = [], []
    for i in range(len(all_protein)):
        x = i + 1
        for j in range(x, len(all_protein)):
            protein1.append(all_protein[i])
            protein2.append(all_protein[j])
    data = pd.DataFrame()
    data['protein1'] = protein1
    data['protein2'] = protein2
    return data

if __name__ == '__main__':
    # data = pd.read_csv('H:/20230708_new/dia_rescore/ribosome/60S.csv')
    # protein = data['Entry']
    # data1 = generate_theoritial_PPI(protein)
    # data1.to_csv('H:/20230708_new/dia_rescore/ribosome/60S_PPI.csv', index=False)
    #
    x = generate_potential_XL_peptide()
    # x.generate_inter_peptide_pair('G:/20230708_new/dia_rescore/HSP90AA1_centric1/HSP90AA1_centric1_PPI.csv',
    #                               'G:/tims_data/fasta/human reviewed.fasta')
    # pep_list = x.generate_peptide_pair('./data/Human_reviewed.fasta', './data/RL7A_RL34.csv')
    # print(len(pep_list))
    # pep_list.to_csv('./data/RL7A_RL34_cross-linked_peptides.csv', index=False)
    # pep_list = x.generate_intra_peptide_pair1('./data/yeast_reviewed.fasta')
    # print(len(pep_list))
    # pep_list.to_csv('./data/yeast_intra_peptidex.csv', index=False)
    peptide = x.generate_intra_peptide_pair('G:/20230708_new/dia_rescore/100_low_protein/20_protein_5/20_protein_5.csv',
                                            'G:/tims_data/fasta/human reviewed.fasta')
    # peptide = x.generate_intra_peptide_pair('H:/20230708_new/dia_rescore/100protein/20protein_4/20protein_4.csv',
    #                                         'H:/tims_data/fasta/human reviewed.fasta')

    # protein = x.test('H:/20230708_new/dia_rescore/100protein/100protein.csv',
    #                                         'H:/tims_data/fasta/human reviewed.fasta')
    # protein.to_csv('H:/20230708_new/dia_rescore/100protein/100protein1.csv', index=False)
    # peptide.to_csv('F:/hela_invivo_crosslink/240/pre/low_quantity_intraprotein1_intro_peptide.csv', index=False)

    # protein = x.extractFa('PDPK2P', './data/Human_reviewed.fasta')
    # protein = pd.DataFrame(protein)
    # protein.to_csv('./data/PDPK2P.fasta',index=False,header=None)
    # print(protein)
    # peptide = x.generate_intra_peptide_pair('./data/PDPK2P.fasta')
    # peptide.to_csv('./data/PDPK2P_intro_peptide.csv', index=False)