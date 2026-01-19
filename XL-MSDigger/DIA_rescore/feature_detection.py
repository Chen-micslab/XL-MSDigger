import numpy as np
import numpy.linalg as L
import pandas as pd
import math
from Preprocess.mass_cal import mz_cal
from scipy.stats import pearsonr
import re

class feature_detect():
    def get_cosine(self, msms, msms_pre):
        dot = np.dot(msms,msms_pre)
        return dot/(L.norm(msms)*L.norm(msms_pre))

    def get_pearson(self, act, pred):
        return pearsonr(act, pred)[0]

    def get_SA(self, msms, msms_pre):
        L2normed_act = msms / L.norm(msms)
        L2normed_pred = msms_pre / L.norm(msms_pre)
        inner_product = np.dot(L2normed_act, L2normed_pred)
        return 1 - 2*np.arccos(inner_product)/np.pi

    def get_unweighted_entropy(self, msms, msms_pre):
        spectrum1_a = msms
        spectrum1_a = spectrum1_a / np.sum(spectrum1_a)
        spectrum2_a = spectrum1_a[spectrum1_a > 0]
        a_entropy = -1 * np.sum(spectrum2_a * np.log(spectrum2_a))

        spectrum1_b = msms_pre
        spectrum1_b = spectrum1_b / np.sum(spectrum1_b)
        spectrum2_b = spectrum1_b[spectrum1_b > 0]
        b_entropy = -1 * np.sum(spectrum2_b * np.log(spectrum2_b))

        spectrum1_ab = (spectrum1_a + spectrum1_b)/2
        spectrum2_ab = spectrum1_ab[spectrum1_ab > 0]
        ab_entropy = -1 * np.sum(spectrum2_ab * np.log(spectrum2_ab))
        unweighted_entropy_similarity = 1 - (2 * ab_entropy - a_entropy - b_entropy) / math.log(4)
        return unweighted_entropy_similarity

    def extract_matched_msms_intensity(self, fragment_quant):
        inten = fragment_quant.split(';')
        inten = inten[:-1]
        inten = np.array(inten, dtype=float)
                                                       
        inten = inten/np.max(inten)
        inten[inten < 0.05] = 0                      
        return inten

    def calculate_msms_num(self, msms, target, tolerance):
        index = np.abs((msms - target)/msms) < tolerance
        result = msms[index]
        count = len(result)
        if count == 0:
            count = 1
        return count

    def calculate_fragment_weight(self, library, tolerance=0.00002):                
        library1 = library[library['type2'] != 'test']
        fragment_mz1 = np.array(library1['Fragment_m_z_calculation'])
        fragment_mz = np.array(library['Fragment_m_z_calculation'])
        count_list = []
        for mz in fragment_mz:
            count = self.calculate_msms_num(fragment_mz1, mz, tolerance)
            count_list.append(count)
        library['fragment_count'] = count_list
        return library

    def process_diann_report(self, diann_report, normal_library):
        data = diann_report[['Run','Precursor.Id', 'Modified.Sequence', 'Precursor.Charge', 'Q.Value', 'PEP', 'Evidence', 'Spectrum.Similarity',
                             'CScore', 'RT', 'iRT', 'IM', 'iIM', 'Fragment.Quant.Raw', 'Ms1.Profile.Corr', 'Mass.Evidence', 'Ms1.Area',
                             'Precursor.Quantity', 'Fragment.Correlations', 'MS2.Scan']].copy()
        peptide_list = list(data['Precursor.Id'])
        peptide_list1 = list(normal_library['combine_peptide_z'])
        z_list1 = list(normal_library['charge'])
        type_list1 = list(normal_library['type'])
        type_list2 = list(normal_library['type2'])
        type_list, type2_list, m_z_list = [], [], []
        m = mz_cal()
        for pep in peptide_list:
            id = int(peptide_list1.index(pep))
            z = int(z_list1[id])
            type_list.append(type_list1[id])
            type2_list.append(type_list2[id])
            m_z = m.crosslink_peptide_m_z(pep[:-1],z,'DSS')
            m_z = round(m_z,5)
            m_z_list.append(m_z)
        data['m_z'] = m_z_list
        data['type'] = type_list
        data['type2'] = type2_list
        data1 = self.filter_diann_results(data)
        return data1

    def filter_diann_results(self, data):                                                    
                                                                        
        data1 = data.sort_values('MS2.Scan', ignore_index=True)                  
        data1 = data1.groupby('MS2.Scan').apply(lambda x: x.sort_values('m_z'))                        
        data1 = data1.reset_index(drop=True)
        scan_list = np.array(data1['MS2.Scan'])
        mz_list = np.array(data1['m_z'])
        name = list(data1)
        data5 = np.array(data1)                                
        scan = scan_list[0]
        mz = mz_list[0]
        index_list = []
        num = 0
        lenth1 = 0
        for i in range(len(scan_list)):
            if (scan_list[i] == scan) and (mz_list[i]==mz):
                index_list.append(i)
            else:
                num = num + 1
                data2 = data1.iloc[index_list]
                q_list = list(np.array(data2['PEP']))
                data3 = data2[data2['PEP'] == min(q_list)]
                lenth2 = len(data3['Precursor.Id'])
                data5[lenth1:(lenth1+lenth2),:] = np.array(data3)                        
                lenth1 = lenth1 + lenth2
                index_list = []
                index_list.append(i)
                scan = scan_list[i]
                mz = mz_list[i]
        data2 = data1.iloc[index_list]
        q_list = list(np.array(data2['PEP']))
        data3 = data2[data2['PEP'] == min(q_list)]
        lenth2 = len(data3['Precursor.Id'])
        data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)                        
        lenth1 = lenth1 + lenth2
        data6 = pd.DataFrame(data5[:lenth1,:],columns=name)
        return data6

    def extract_from_cl_peptide(self, cl_peptide):                                
        id1 = cl_peptide.find('X')
        pep1 = cl_peptide[:id1]
        pep2 = cl_peptide[(id1 + 1):]
        site1 = pep1.find('U')
        site2 = pep2.find('U')
        peptide1 = pep1.replace('U', '')
        peptide2 = pep2.replace('U', '')
        return peptide1, peptide2, int(site1), int(site2)

    def calculate_spec_frag_num(self, data):
        num = 0
        num1 = 0
        num2 = 0
        peptide, Fragment_num, Fragment_type = list(data['combine_peptide']), list(data['Fragment_num']), list(data['Fragment_type'])
        for h in range(len(peptide)):
            f_num, type =  int(Fragment_num[h]), Fragment_type[h]
            pep1, pep2, s1, s2 = self.extract_from_cl_peptide(peptide[h])
            if type == '1b':
                if f_num >= s1:
                    num = num + 1
                    num1 = num1 + 1
            elif type == '1y':
                if f_num >= (len(pep1) - s1 + 1):
                    num = num + 1
                    num1 = num1 + 1
            elif type == '2b':
                if f_num >= s2:
                    num = num + 1
                    num2 = num2 + 1
            elif type == '2y':
                if f_num >= (len(pep2) - s2 + 1):
                    num = num + 1
                    num2 = num2 + 1
        return num, num1, num2

    def calculate_fragment_correlation(self, Fragment_Correlations):
        a = Fragment_Correlations.split(';')
        a = a[:-1]
        numbers = [float(num) for num in a]
        average_correlations = sum(numbers) / len(numbers)
        return average_correlations

    def feature_detection(self, diann_report, library):
        peptide_z_list = diann_report['Precursor.Id']
        fragment_inten_list = diann_report['Fragment.Quant.Raw']
        fragment_correlation_list = diann_report['Fragment.Correlations']
        ms1_area_list = diann_report['Ms1.Area']
        ms2_quantity_list = diann_report['Precursor.Quantity']
                                                                   
        pep1_num_list, pep2_num_list, pep1_num_matched_list, pep2_num_matched_list, pep_cosine, pep1_cosine, pep2_cosine = [], [], [], [], [], [], []
        pep_entropy, pep1_entropy, pep2_entropy = [], [], []
        highest_ms2_intensity_list, inten_ratio_list, averge_corr_list = [], [], []
        spec_frag_num_list, alpha_spec_frag_num_list, beta_spec_frag_num_list = [], [], []
        lib_filter = library[library['combine_peptide_z'].isin(peptide_z_list)]
                                                            
        for i in range(len(peptide_z_list)):
            peptide_z = peptide_z_list[i]
            ms1_area = float(ms1_area_list[i])
            ms2_quantity = float(ms2_quantity_list[i])
            max_ms2_inten = max(map(float, re.findall(r'(.*?);', fragment_inten_list[i])))
            inten_ratio = ms1_area/max_ms2_inten
            data = lib_filter[lib_filter['combine_peptide_z'] == peptide_z]
            data = data.sort_values(by='Fragment_intensity', ascending=False)                           
            matched_frag_inten = self.extract_matched_msms_intensity(fragment_inten_list[i])
            averge_corr = self.calculate_fragment_correlation(fragment_correlation_list[i])
            averge_corr_list.append(averge_corr)
            lenth = len(list(matched_frag_inten))
            data = data.head(lenth)
            data['matched_frag_inten'] = list(matched_frag_inten)
            data1 = data[data['matched_frag_inten']>0].copy()
            spec_frag_num, alpha_spec_frag_num, beta_spec_frag_num = self.calculate_spec_frag_num(data1)
            spec_frag_num_list.append(spec_frag_num)
            alpha_spec_frag_num_list.append(alpha_spec_frag_num)
            beta_spec_frag_num_list.append(beta_spec_frag_num)
            by_type = list(data['Fragment_type'])
            matched_by_type = list(data1['Fragment_type'])
            by_type = list(by_type)
            pep1_num = by_type.count('1b') + by_type.count('1y')
            pep2_num = by_type.count('2b') + by_type.count('2y')
            pep1_num_matched = matched_by_type.count('1b') + matched_by_type.count('1y')
            pep2_num_matched = matched_by_type.count('2b') + matched_by_type.count('2y')
            pep1_num_list.append(pep1_num)
            pep2_num_list.append(pep2_num)
            pep1_num_matched_list.append(pep1_num_matched)
            pep2_num_matched_list.append(pep2_num_matched)
            inten1 = data1['Fragment_intensity']
            inten2 = data1['matched_frag_inten']
            pep_entropy.append(self.get_unweighted_entropy(inten1, inten2))
            pep_cosine.append(self.get_cosine(inten1, inten2))
            data2 = data1[(data1['Fragment_type'] == '1b') | (data1['Fragment_type'] == '1y')]
            inten1 = data2['Fragment_intensity']
            inten2 = data2['matched_frag_inten']
            if len(inten1) > 1:
                cos1 = self.get_cosine(inten1, inten2)
                en1 = self.get_unweighted_entropy(inten1, inten2)
            elif len(inten1) == 1:
                cos1 = 0.5
                en1 = 0.5
            else:
                cos1 = 0
                en1 = 0
            data3 = data1[(data1['Fragment_type'] == '2b') | (data1['Fragment_type'] == '2y')]
            inten1 = data3['Fragment_intensity']
            inten2 = data3['matched_frag_inten']
            if len(inten1) > 1:
                cos2 = self.get_cosine(inten1, inten2)
                en2 = self.get_unweighted_entropy(inten1, inten2)
            elif len(inten1) == 1:
                cos2 = 0.5
                en2 = 0.5
            else:
                cos2 = 0
                en2 = 0
            pep1_cosine.append(cos1)
            pep2_cosine.append(cos2)
            pep1_entropy.append(en1)
            pep2_entropy.append(en2)
            highest_ms2_intensity_list.append(max_ms2_inten)
            inten_ratio_list.append(inten_ratio)
        diann_report['highest_ms2_intensity'] = highest_ms2_intensity_list
        diann_report['inten_ratio'] = inten_ratio_list
        diann_report['averge_corr_list'] = averge_corr_list
        diann_report['pep1_num'] = pep1_num_list
        diann_report['pep2_num'] = pep2_num_list
        diann_report['pep1_num_matched'] = pep1_num_matched_list
        diann_report['pep2_num_matched'] = pep2_num_matched_list
        diann_report['spec_frag_num'] = spec_frag_num_list
        diann_report['alpha_spec_frag_num'] = alpha_spec_frag_num_list
        diann_report['beta_spec_frag_num'] = beta_spec_frag_num_list
        diann_report['pep_cosine'] = pep_cosine
        diann_report['pep1_cosine'] = pep1_cosine
        diann_report['pep2_cosine'] = pep2_cosine
        diann_report['pep_entropy'] = pep_entropy
        diann_report['pep1_entropy'] = pep1_entropy
        diann_report['pep2_entropy'] = pep2_entropy

        RT = np.array(diann_report['RT'])
        iRT = np.array(diann_report['iRT'])
        delta_RT = np.abs(RT-iRT)
        diann_report['delta_RT'] = delta_RT
        IM = np.array(diann_report['IM'])
        iIM = np.array(diann_report['iIM'])
        delta_IM = np.abs(IM-iIM)/IM
        diann_report['delta_IM'] = delta_IM
        return diann_report

    def total_process(self, diann_results_dir, library_dir):
        data = pd.read_table(diann_results_dir)
        lib = pd.read_csv(library_dir)
        data1 = self.process_diann_report(data, lib)
        data2 = self.feature_detection(data1, lib)
        return data2
        outputdir = diann_results_dir.split('.tsv')[0] + '_feature.csv'
        data2.to_csv(outputdir, index=False)


