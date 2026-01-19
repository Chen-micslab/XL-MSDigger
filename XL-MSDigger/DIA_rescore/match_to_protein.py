import numpy as np
import pandas as pd

def intra_peptide_matched_protein(peptide, peptide_protein_data):
    peptide = peptide.replace('U','')
    peptide_list = set(peptide_protein_data['peptide'])
    if peptide in peptide_list:
        a = peptide_protein_data[peptide_protein_data['peptide'] == peptide]
        protein = list(set(a['protein']))[0]
    else:
        protein = 'nomatch-nomatch'
    return protein

def inter_peptide_matched_protein(peptide, peptide_protein_data):
    peptide = peptide.replace('U','')
    peptide_list = set(peptide_protein_data['peptide'])
    if peptide in peptide_list:
        a = peptide_protein_data[peptide_protein_data['peptide'] == peptide]
        protein = list(set(a['protein']))[0]
    else:
        protein = 'nomatch-nomatch'
    return protein

def process_intra_results(peptide_results, peptide_protein_data):
    protein_list = []
    peptide_list = peptide_results['Modified.Sequence']
    for peptide in peptide_list:
        protein = intra_peptide_matched_protein(peptide, peptide_protein_data)
        protein_list.append(protein)
    peptide_results['protein'] = protein_list
    return peptide_results

def extract_from_cl_peptide(cl_peptide):                                
    id1 = cl_peptide.find('X')
    pep1 = cl_peptide[:id1]
    pep2 = cl_peptide[(id1 + 1):]
    site1 = pep1.find('U')
    site2 = pep2.find('U')
    peptide1 = pep1.replace('U', '')
    peptide2 = pep2.replace('U', '')
    return peptide1, peptide2, site1, site2

import re
def extract_uniprot_id(s):
                                     
    match = re.search(r'sp\|([A-Za-z0-9]+)\|', s)
    if match:
        return match.group(1)                    
    else:
        return None

def read_fasta(fasta_dir):
    fr = open(fasta_dir, 'r')        
    protein_name_list = []
    amino_acid_list = []              
    amino_seq = ''
    for line in fr:
            line = line.replace('\n','')
            if line.startswith('>'):                                                             
                if amino_seq != '':
                    amino_acid_list.append(amino_seq)
                uniprot_id = extract_uniprot_id(line)
                protein_name_list.append(uniprot_id)
                amino_seq = ''
            else:
                amino_seq = amino_seq + line
    if amino_seq != '':
        amino_acid_list.append(amino_seq)
    protein_dict = dict(zip(protein_name_list, amino_acid_list))
    return protein_dict

def extract_absolute_site(data, fasta_dir):
    protein_dict = read_fasta(fasta_dir)
    peptide1_list = list(data['peptide1'])
    peptide2_list = list(data['peptide2'])
    site1_list = list(data['site1'])
    site2_list = list(data['site2'])
    protein_uniprot_id_list = list(data['protein'])
    s1_list, s2_list, protein1_list, protein2_list = [], [], [], []
    for i in range(len(peptide1_list)):
        peptide1 = peptide1_list[i]
        peptide2 = peptide2_list[i]
        site1 = site1_list[i]
        site2 = site2_list[i]
        protein_uniprot_id = protein_uniprot_id_list[i]
        print(protein_uniprot_id)
        protein1_id, protein2_id = protein_uniprot_id.split('-')
        pro1 = protein_dict[protein1_id]
        pro2 = protein_dict[protein2_id]
        if peptide1 in pro1:
            s1 = pro1.find(peptide1) + site1
            s2 = pro2.find(peptide2) + site2
            protein1_list.append(protein1_id)
            protein2_list.append(protein2_id)
        else:
            s1 = pro2.find(peptide1) + site1
            s2 = pro1.find(peptide2) + site2
            protein1_list.append(protein2_id)
            protein2_list.append(protein1_id)
        s1_list.append(s1)
        s2_list.append(s2)
    data['protein1'] = protein1_list
    data['protein2'] = protein2_list
    data['protein_site1'] = s1_list
    data['protein_site2'] = s2_list
    return data

def run_for_rescore_results(rescore_results, peptide_protein_data_dir, fasta_dir):
    peptide_protein_data = pd.read_csv(peptide_protein_data_dir)
    data1 = process_intra_results(rescore_results, peptide_protein_data)
    data2 = data1[data1['type2']=='test'].copy()
    peptide = data2['Modified.Sequence']
    peptide1, peptide2, site1, site2, protein = [], [], [], [], []
    for pep in peptide:
        pep1, pep2, s1, s2 = extract_from_cl_peptide(pep)
        peptide1.append(pep1)
        peptide2.append(pep2)
        site1.append(s1)
        site2.append(s2)
    data2['peptide1'] = peptide1
    data2['peptide2'] = peptide2
    data2['site1'] = site1
    data2['site2'] = site2
    data3 = extract_absolute_site(data2, fasta_dir)
    return data3

if __name__ == '__main__':
    data = pd.read_csv('G:/20230708_new/dia_rescore/100_low_protein/20_protein_5/report_feature_results_dnn1_test.csv')
    peptide_protein_data = pd.read_csv('G:/20230708_new/dia_rescore/100_low_protein/20_protein_5/20_protein_5_origin_intra_peptide&protein.csv')
    data1 = process_intra_results(data, peptide_protein_data)
    data2 = data1[data1['type2']=='test'].copy()
    peptide = data2['Modified.Sequence']
    peptide1, peptide2, site1, site2, protein = [], [], [], [], []
    for pep in peptide:
        pep1, pep2, s1, s2 = extract_from_cl_peptide(pep)
        peptide1.append(pep1)
        peptide2.append(pep2)
        site1.append(s1)
        site2.append(s2)
    data2['peptide1'] = peptide1
    data2['peptide2'] = peptide2
    data2['site1'] = site1
    data2['site2'] = site2
    print(data2)
    data3 = extract_absolute_site(data2, 'G:/tims_data/fasta/human reviewed.fasta')
    data2.to_csv('G:/20230708_new/dia_rescore/100_low_protein/20_protein_5/report_feature_results_dnn1_test_protein.csv', index=False)

                            
                                  
                                           
                               
                            
                                  
                                           
                               
                            
                               
                             
                                                                                                                  
                                                       
