import pandas as pd
import numpy as np
import math
import os
import sys
from mass_cal import mz_cal as M


class plink_with_msconvert_mgf():
    """适用于msconvert软件生成的mgf文件和plink2搜库结果的处理类"""

    def __init__(self, crosslinker='DSS', fragment_ppm=0.00002, fragment_num=24, min_mz=0, max_mz=1700,
                 intensity_threshold=0, plink_score_cutoff=1):
        self.crosslinker = crosslinker
        self.frag_ppm = fragment_ppm
        self.frag_num = fragment_num
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.intensity_threshold = intensity_threshold
        self.plink_score_cutoff = plink_score_cutoff

    def extract_from_plink_xl_peptide(self, xl_peptide):
        """用于从plink的交联肽序列中提取两条肽和位点的信息"""
        id1 = xl_peptide.find('-')
        pep1 = xl_peptide[:id1]
        pep2 = xl_peptide[(id1 + 1):]
        id1 = pep1.find('(')
        id2 = pep1.find(')')
        peptide1 = pep1[:id1]
        site1 = pep1[(id1 + 1):id2]
        id1 = pep2.find('(')
        id2 = pep2.find(')')
        peptide2 = pep2[:id1]
        site2 = pep2[(id1 + 1):id2]
        return peptide1, peptide2, int(site1), int(site2)

    def modif_xlpeptide(self, pep1, pep2, mod):
        len1 = len(pep1)
        len2 = len(pep2)
        if 'M' in mod:
            find_all = lambda c, s: [x for x in range(c.find(s), len(c)) if c[x] == s]                         
            id_list = find_all(mod, 'M')
            for id in id_list:
                if mod[(id + 4)] == ')':
                    modid = int(mod[(id + 3)])
                    if modid < (len1 + len2 + 1):
                        if modid > len1:
                            b = list(pep2)
                            if b[(modid - len1 - 1)] == 'M':
                                b[(modid - len1 - 1)] = 'e'
                            pep2 = ''.join(b)
                        else:
                            b = list(pep1)
                            if b[(modid - len1 - 1)] == 'M':
                                b[(modid - len1 - 1)] = 'e'
                            pep1 = ''.join(b)
                elif mod[(id + 5)] == ')':
                    modid = int(mod[(id + 3):(id + 5)])
                    if modid < (len1 + len2 + 1):
                        if modid > len1:
                            b = list(pep2)
                            if b[(modid - len1 - 1)] == 'M':
                                b[(modid - len1 - 1)] = 'e'
                            pep2 = ''.join(b)
                        else:
                            b = list(pep1)
                            if b[(modid - len1 - 1)] == 'M':
                                b[(modid - len1 - 1)] = 'e'
                            pep1 = ''.join(b)
        return pep1, pep2

    def extract_from_combine_peptide(self, combine_peptide):
        """从combine_peptide中提取两条肽和位点的信息"""
        id1 = combine_peptide.find('X')
        pep1 = combine_peptide[:id1]
        pep2 = combine_peptide[(id1 + 1):]
        site1 = pep1.find('U')
        peptide1 = pep1.replace('U', '')
        site2 = pep2.find('U')
        peptide2 = pep2.replace('U', '')
        return peptide1, peptide2, site1, site2

    def calculate_ccs(self, peptide_m_z, peptide_charge, peptide_k0):
        """计算CCS值"""
        m = 28.00615
        t = 304.7527
        coeff = 18500 * peptide_charge * math.sqrt(
            (peptide_m_z * peptide_charge + m) / (peptide_m_z * peptide_charge * m * t))
        ccs = coeff * peptide_k0
        return ccs

    def parse_mgf_info(self, spectrum, title):
        """从msconvert生成的MGF文件中解析RT和Ion Mobility信息"""
        rt = None
        ion_mobility = None

        title_line = f'TITLE={title}'
        title_indices = np.where(spectrum == title_line)[0]
        if len(title_indices) == 0:
            return rt, ion_mobility

        id = title_indices[0]
                                     
        for i in range(id + 1, min(id + 10, len(spectrum))):
            line = str(spectrum[i])
            if line.startswith('RTINSECONDS='):
                rt = float(line.split('=')[1])
            elif line.startswith('ION_MOBILITY='):
                parts = line.split('=')[1].strip().split()
                if len(parts) >= 2:
                    ion_mobility = float(parts[1])
            elif line.startswith('CHARGE=') or line == 'END IONS':
                break

        return rt, ion_mobility

    def change_plink_filter_crosslink(self, plinkfile_dir, spectrum=None):
        """处理plink3生成的filtered_crosslinked_spectra文件"""
                                  
        plinkfile_dir = plinkfile_dir + '/reports/'
        spectra_file = None

        for filename in os.listdir(plinkfile_dir):
            if filename.startswith('._'):
                continue
            if 'filtered_crosslinked_spectra.csv' in filename or 'filtered_cross-linked_spectra.csv' in filename:
                spectra_file = plinkfile_dir + filename
                break

        if spectra_file is None:
            raise FileNotFoundError("未找到filtered_crosslinked_spectra.csv文件")

              
        data = pd.read_csv(spectra_file, header=0, index_col=False, engine='python', on_bad_lines='warn')
                                              
        if 'Charge' in data.columns:
            first_charge = data.iloc[0]['Charge']
            try:
                int(first_charge)
            except (ValueError, TypeError):
                print("Detected column misalignment, fixing...")

                                 
                original_columns = list(data.columns)[:-1]

                             
                data['Title'] = data['Title'].astype(str) + ',' + data['Charge'].astype(str)

                              
                data = data.drop('Charge', axis=1)

                              
                data.columns = original_columns

               
        cl_peptide_list, m_z_list, charge_list, peptide1_list, site1_list, peptide2_list, site2_list = [], [], [], [], [], [], []
        rt_list, k0_list, ccs_list, score_list, precursor_Mass_Error_list, intensity_list, type_list = [], [], [], [], [], [], []
        protein_list, protein_type_list, title_list, filename_list, cmpd_list = [], [], [], [], []

        for index, row in data.iterrows():
            peptide = row['Peptide']
            peptide1, peptide2, site1, site2 = self.extract_from_plink_xl_peptide(peptide)

                  
            modif = str(row['Modifications']) if pd.notna(row['Modifications']) else ''
            peptide1, peptide2 = self.modif_xlpeptide(peptide1, peptide2, modif)

                    
            if any(char in peptide for char in ['U', 'O', 'X', 'B', 'J', 'Z']):
                continue

                                
            title = row['Title']
            charge = row['Charge']
                                                        
            if 'Precursor_Mass' in row:
                precursor_mass = row['Precursor_Mass']
            else:
                precursor_mass = row['Precursor_MH']

            score = row['Score']                         
            precursor_mass_error = row['Precursor_Mass_Error(ppm)']
            protein = row['Proteins']
            protein_type = row['Protein_Type']

                   
            m_z = (float(precursor_mass) + (int(charge) - 1) * 1.00728) / int(charge)

                                      
            rt = None
            k0 = None
            ccs = None
            intensity = 1.0

            if spectrum is not None:
                rt, k0 = self.parse_mgf_info(spectrum, title)
                if k0 is not None:
                    ccs = self.calculate_ccs(m_z, int(charge), float(k0))

                          
            if '.' in title:
                filename = title.split('.')[0]
                cmpd = ''
            else:
                filename = title
                cmpd = ''

                   
            cl_peptide_list.append(peptide)
            m_z_list.append(m_z)
            charge_list.append(charge)
            peptide1_list.append(peptide1)
            peptide2_list.append(peptide2)
            site1_list.append(site1)
            site2_list.append(site2)
            rt_list.append(rt)
            k0_list.append(k0)
            ccs_list.append(ccs)
            score_list.append(score)
            precursor_Mass_Error_list.append(precursor_mass_error)
            title_list.append(title)
            filename_list.append(filename)
            intensity_list.append(intensity)
            cmpd_list.append(cmpd)
            type_list.append(3)
            protein_list.append(protein)
            protein_type_list.append(protein_type)

                                       
        result_data = pd.DataFrame()
        result_data['title'] = title_list
        result_data['filename'] = filename_list
        result_data['peptide'] = cl_peptide_list
        result_data['m_z'] = m_z_list
        result_data['charge'] = charge_list
        result_data['peptide1'] = peptide1_list
        result_data['peptide2'] = peptide2_list
        result_data['site1'] = site1_list
        result_data['site2'] = site2_list
        result_data['rt'] = rt_list
        result_data['k0'] = k0_list
        result_data['ccs'] = ccs_list
        result_data['score'] = score_list
        result_data['precursor_Mass_Error(ppm)'] = precursor_Mass_Error_list
        result_data['intensity'] = intensity_list
        result_data['cmpd'] = cmpd_list
        result_data['peptide_type'] = type_list
        result_data['protein'] = protein_list
        result_data['protein_type'] = protein_type_list

                             
        combine_pep, combine_pep_z = [], []
        for i in range(len(peptide1_list)):
            pep1 = peptide1_list[i]
            pep2 = peptide2_list[i]
            s1 = int(site1_list[i])
            s2 = int(site2_list[i])
            z = int(charge_list[i])

            pep1_list = list(pep1)
            pep2_list = list(pep2)
            pep1_list.insert(s1, 'U')
            pep2_list.insert(s2, 'U')
            pep1_mod = ''.join(pep1_list)
            pep2_mod = ''.join(pep2_list)

            pep = pep1_mod + 'X' + pep2_mod
            pep_z = pep1_mod + 'X' + pep2_mod + str(z)
            combine_pep.append(pep)
            combine_pep_z.append(pep_z)

        result_data['combine_peptide'] = combine_pep
        result_data['combine_peptide_z'] = combine_pep_z

        return result_data

    def change_plink_total_results(self, plinkfile_dir, spectrum=None):
        """处理plink的总结果文件 - 使用总鉴定文件（最短文件名）"""
        plinkfile_dir = plinkfile_dir + '/reports/'
        listdir = os.listdir(plinkfile_dir)
                           
        file = plinkfile_dir + min(listdir, key=len)

        data = pd.read_csv(file)
        data = data[data['Peptide_Type'] == 3]          

               
        m_z_list, charge_list, peptide1_list, site1_list, peptide2_list, site2_list = [], [], [], [], [], []
        rt_list, k0_list, ccs_list, combine_pep, combine_pep_z, len1_list, len2_list = [], [], [], [], [], [], []

        for index, row in data.iterrows():
            peptide = row['Peptide']
            modif = str(row['Modifications']) if pd.notna(row['Modifications']) else ''
            charge = row['Charge']

            peptide1, peptide2, site1, site2 = self.extract_from_plink_xl_peptide(peptide)
            len1 = len(peptide1)
            len2 = len(peptide2)
            peptide1, peptide2 = self.modif_xlpeptide(peptide1, peptide2, modif)

            z = int(charge)
            pep1_list = list(peptide1)
            pep2_list = list(peptide2)
            pep1_list.insert(site1, 'U')
            pep2_list.insert(site2, 'U')
            pep1_mod = ''.join(pep1_list)
            pep2_mod = ''.join(pep2_list)

            pep = pep1_mod + 'X' + pep2_mod
            pep_z = pep1_mod + 'X' + pep2_mod + str(z)

                                                   
            mass = row['Precursor_MH']
            title = row['Title']

                                      
            rt = None
            k0 = None
            ccs = None
            if spectrum is not None:
                rt, k0 = self.parse_mgf_info(spectrum, title)

            m_z = (float(mass) + (int(charge) - 1) * 1.00728) / int(charge)
            if k0 is not None:
                ccs = self.calculate_ccs(m_z, int(charge), float(k0))

            m_z_list.append(m_z)
            charge_list.append(charge)
            peptide1_list.append(peptide1)
            peptide2_list.append(peptide2)
            len1_list.append(len1)
            len2_list.append(len2)
            site1_list.append(site1)
            site2_list.append(site2)
            rt_list.append(rt)
            k0_list.append(k0)
            ccs_list.append(ccs)
            combine_pep.append(pep)
            combine_pep_z.append(pep_z)

                     
        data['m_z'] = m_z_list
        data['charge'] = charge_list
        data['peptide1'] = peptide1_list
        data['peptide2'] = peptide2_list
        data['len1'] = len1_list
        data['len2'] = len2_list
        data['site1'] = site1_list
        data['site2'] = site2_list
        data['rt'] = rt_list
        data['k0'] = k0_list
        data['ccs'] = ccs_list
        data['combine_peptide'] = combine_pep
        data['combine_peptide_z'] = combine_pep_z

              
        data = data[data['Charge'] < 6]
        data = data[data['len1'] < 50]
        data = data[data['len2'] < 50]
        
                                   
        noncanonical_aas = set(["B", "J", "O", "U", "X", "Z"])
        def contains_noncanonical(peptide: str):
            return any(aa in noncanonical_aas for aa in peptide)

        data = data[~data['peptide1'].apply(contains_noncanonical)]
        data = data[~data['peptide2'].apply(contains_noncanonical)]
                    
        data.rename(columns={'Title': 'title', 'Score': 'score'}, inplace=True)

        return data

    def match_msms(self, spectrum, m_z, title):
        """匹配MS/MS谱图 - 使用原版高效方法"""
        title_line = f'TITLE={title}'
        title_indices = np.where(spectrum == title_line)[0]
        if len(title_indices) == 0:
            m_z_1 = [0 if mz != -1 else -1 for mz in m_z]
            inten = [0 if mz != -1 else -1 for mz in m_z]
            return m_z_1, inten

        id = title_indices[0]
        msms = []
        intensity = []
        id = id + 5                                                       

        for i in range(id, len(spectrum)):
            if spectrum[i] == 'END IONS':
                break
            else:
                try:
                    a = str(spectrum[i])
                    if ' ' in a:
                        parts = a.split()
                        msms.append(float(parts[0]))
                        intensity.append(float(parts[1]))
                except (ValueError, IndexError):
                    continue

        if not msms:
            m_z_1 = [0 if mz != -1 else -1 for mz in m_z]
            inten = [0 if mz != -1 else -1 for mz in m_z]
            return m_z_1, inten

        msms = np.array(msms)
        intensity = np.array(intensity)

        m_z_1 = []
        inten = []
        for mz in m_z:
            if mz == -1:
                m_z_1.append(-1), inten.append(-1)
            else:
                msms1 = np.abs(msms - mz) / msms
                if np.min(msms1) <= self.frag_ppm:
                    m_z_1.append(msms[np.argmin(msms1)]), inten.append(intensity[np.argmin(msms1)])
                else:
                    m_z_1.append(0), inten.append(0)
        return m_z_1, inten

    def filter_plink_precursor_results(self, data):
        """筛选相同肽中score最小的precursor"""
        data1 = data.sort_values('combine_peptide_z', ignore_index=True)             
        peptide_list = np.array(data1['combine_peptide_z'])
        name = list(data1)
        data5 = np.array(data1)                                
        peptide = peptide_list[0]
        index_list = []
        peptide_num = len(set(data1['combine_peptide_z']))
        num = 0
        lenth1 = 0
        for i in range(len(peptide_list)):
            if peptide_list[i] == peptide:
                index_list.append(i)
            else:
                num = num + 1
                data2 = data1.iloc[index_list]
                q_list = list(data2['score'])
                psm_list = list(data2['title'])
                psm = psm_list[int(q_list.index(min(q_list)))]
                data3 = data2[data2['title'] == psm]
                lenth2 = len(data3['combine_peptide_z'])
                data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)                        
                lenth1 = lenth1 + lenth2
                index_list = []
                index_list.append(i)
                peptide = peptide_list[i]
        data2 = data1.iloc[index_list]
        q_list = list(data2['score'])
        psm_list = list(data2['title'])
        psm = psm_list[int(q_list.index(min(q_list)))]
        data3 = data2[data2['title'] == psm]
        lenth2 = len(data3['combine_peptide_z'])
        data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)                        
        lenth1 = lenth1 + lenth2
        data6 = pd.DataFrame(data5[:lenth1, :], columns=name)
        data6['score'] = data6['score'].astype(float)
        data7 = data6[data6['score'] < float(self.plink_score_cutoff)]
        return data7

    def filter_plink_peptide_results(self, data):
        """筛选相同肽中score最小的peptide"""
        data1 = data.sort_values('combine_peptide', ignore_index=True)             
        peptide_list = np.array(data1['combine_peptide'])
        name = list(data1)
        data5 = np.array(data1)                                
        peptide = peptide_list[0]
        index_list = []
        peptide_num = len(set(data1['combine_peptide']))
        num = 0
        lenth1 = 0
        for i in range(len(peptide_list)):
            if peptide_list[i] == peptide:
                index_list.append(i)
            else:
                num = num + 1
                data2 = data1.iloc[index_list]
                q_list = list(data2['score'])
                psm_list = list(data2['title'])
                psm = psm_list[int(q_list.index(min(q_list)))]
                data3 = data2[data2['title'] == psm]
                lenth2 = len(data3['combine_peptide'])
                data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)                        
                lenth1 = lenth1 + lenth2
                index_list = []
                index_list.append(i)
                peptide = peptide_list[i]
        data2 = data1.iloc[index_list]
        q_list = list(data2['score'])
        psm_list = list(data2['title'])
        psm = psm_list[int(q_list.index(min(q_list)))]
        data3 = data2[data2['title'] == psm]
        lenth2 = len(data3['combine_peptide'])
        data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)                        
        lenth1 = lenth1 + lenth2
        data6 = pd.DataFrame(data5[:lenth1, :], columns=name)
        data6['score'] = data6['score'].astype(float)
        data7 = data6[data6['score'] < float(self.plink_score_cutoff)]
        return data7

    def crosslink_ion_generation(self, peptide1, peptide2):
        """生成交联肽的碎片离子"""
        len1 = len(peptide1)
        len2 = len(peptide2)
        z = ['1', '2', '3', '4', '5']            
        l = ['noloss'] * 5          
        by_1 = (['1b'] * 5 + ['1y'] * 5) * (len1 - 1)
        c = []         
        for i in range(len1 - 1):
            j = i + 1
            c = c + [j] * 10
        for i in range(len2 - 1):
            j = i + 1
            c = c + [j] * 10
        by_2 = (['2b'] * 5 + ['2y'] * 5) * (len2 - 1)
        by = by_1 + by_2
        z = z * 2 * (len1 + len2 - 2)
        l = l * 2 * (len1 + len2 - 2)
        c = np.array(c)         
        by = np.array(by)           
        z = np.array(z)          
        l = np.array(l)         
        data = np.column_stack((c, by, z, l))
        return data

    def choose_top_n(self, data, n):
        """选择前n强度的碎片"""
        name = list(data)
        inten = np.array(data['Fragment_intensity'])
        data1 = np.array(data)
        if len(data1) > n:
            data2 = data1[np.argsort(-inten)]
            data3 = data2[:n, :]
            data3 = pd.DataFrame(data3, columns=name)
            return data3
        else:
            return data

    def genenrate_all_crosslink_fragment(self, plink_data, mgf_dir):
        """生成所有交联碎片离子库"""
        sys.stdout.write("Loading file......\r")

                       
        spectrum = np.array(pd.read_csv(mgf_dir, sep='!'))
        spectrum = spectrum.flatten()

                                                           
        data = plink_data
        name = list(data)
        pep1 = np.array(data['peptide1'])
        pep2 = np.array(data['peptide2'])

        num = 0
        for i in range(len(pep1)):
            num = num + (len(pep1[i]) + len(pep2[i]) - 2) * 10

        data_array = np.array(data)
        a = np.array(list(data_array[0, :]) + [0, 0, 0, 0, 0, 0, 0], dtype=object)
        data1 = np.tile(a, [num, 1])
        num = 0
        name = name + ['Fragment_num', 'Fragment_type', 'Fragment_charge', 'Neutral_loss']

        sys.stdout.write("Generating library......\r")

        for i in range(len(pep1)):
            peptide1 = pep1[i]
            peptide2 = pep2[i]
            len1 = (len(pep1[i]) + len(pep2[i]) - 2) * 10
            b = data_array[i, :]
            data_y = np.tile(b, [len1, 1])
            data_x = self.crosslink_ion_generation(peptide1, peptide2)
            data_xy = np.column_stack((data_y, data_x))
            data_xy = pd.DataFrame(data_xy, columns=name)

            xl_peptide = data_xy['combine_peptide']
            by_type = data_xy['Fragment_type']
            Fragment_num = data_xy['Fragment_num']
            Fragment_charge = data_xy['Fragment_charge']
            Neutral_loss = data_xy['Neutral_loss']
            title = data_xy['title']

            m_z = []
            m = M()
            for j in range(len(xl_peptide)):
                mz = m.crosslink_peptide_msms_m_z(xl_peptide.iloc[j], self.crosslinker, by_type.iloc[j],
                                                  int(Fragment_num.iloc[j]), int(Fragment_charge.iloc[j]),
                                                  Neutral_loss.iloc[j])
                mz = round(mz, 5)
                m_z.append(mz)

                        
            m_z_1, inten = self.match_msms(spectrum, m_z, title.iloc[0])

            if np.max(inten) > 0:
                inten = np.array(inten) / np.max(inten)
            else:
                inten = np.array(inten)

            m_z = np.array(m_z)
            m_z_1 = np.array(m_z_1)
            data_xy = np.array(data_xy)
            data_xy = np.column_stack((data_xy, m_z, m_z_1, inten))

            charge = int(data_xy[0, 4])
            for k in range(len(data_xy)):
                if int(data_xy[k, -5]) > charge:
                    data_xy[k, -1], data_xy[k, -2], data_xy[k, -3] = -1, -1, -1

            data1[num:(num + len1), :] = data_xy
            num = num + len1

        name = name + ['Fragment_m_z_calculation', 'Fragment_m_z_experiment', 'Fragment_intensity']

        for l in range(len(data1)):
            if data1[l, -1] < 0:
                data1[l, -1] = -1

        data1 = pd.DataFrame(data1, columns=name)
        data1 = data1[data1['Fragment_m_z_experiment'] > self.min_mz]
        data1 = data1[data1['Fragment_m_z_experiment'] < self.max_mz]
        data1 = data1[data1['Fragment_intensity'] > self.intensity_threshold]

        return data1

    def transfer_to_DIANN_format(self, data):
        """转换为DIA-NN格式 - 增加Ion Mobility支持"""
        data2 = pd.DataFrame()
        data2['ModifiedPeptide'] = data['combine_peptide']
        data2['PrecursorCharge'] = data['charge']
        data2['PrecursorMz'] = data['m_z']
        data2['FragmentCharge'] = data['Fragment_charge']
        data2['ProductMz'] = data['Fragment_m_z_calculation']
        data2['Tr_recalibrated'] = data['rt']
        if 'k0' in data.columns and data['k0'].notna().any():
            data2['IonMobility'] = data['k0']
        data2['LibraryIntensity'] = data['Fragment_intensity']
        return data2

    def filter_peptide_with_score_in_msms(self, data):
        """根据电荷态和PSM的score来filter"""
        data1 = data.sort_values('combine_peptide_z', ignore_index=True)             
        peptide_list = np.array(data1['combine_peptide_z'])
        name = list(data1)
        data5 = np.array(data1)                                
        peptide = peptide_list[0]
        index_list = []
        peptide_num = len(set(data1['combine_peptide_z']))
        num = 0
        lenth1 = 0
        for i in range(len(peptide_list)):
            if peptide_list[i] == peptide:
                index_list.append(i)
            else:
                num = num + 1
                data2 = data1.iloc[index_list]
                q_list = list(data2['score'])
                psm_list = list(data2['title'])
                psm = psm_list[int(q_list.index(min(q_list)))]
                data3 = data2[data2['title'] == psm]
                lenth2 = len(data3['combine_peptide_z'])
                data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)                        
                lenth1 = lenth1 + lenth2
                index_list = []
                index_list.append(i)
                peptide = peptide_list[i]
        data2 = data1.iloc[index_list]
        q_list = list(data2['score'])
        psm_list = list(data2['title'])
        psm = psm_list[int(q_list.index(min(q_list)))]
        data3 = data2[data2['title'] == psm]
        lenth2 = len(data3['combine_peptide_z'])
        data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)
        lenth1 = lenth1 + lenth2
        data6 = pd.DataFrame(data5[:lenth1, :], columns=name)
                                   
        data1 = data6
        name = list(data1)
        data3 = np.array(data1)
        peptide = set(list(data1['combine_peptide_z']))
        data2 = np.tile(data3[0, :], (len(peptide) * self.frag_num, 1))
        num1 = 0
        x = 0
        for pep in peptide:
            x = x + 1
            data_x = data1[data1['combine_peptide_z'] == pep]
            data_y = np.array(self.choose_top_n(data_x, self.frag_num))            
            lenth = len(data_y)
            data2[num1:(num1 + lenth), :] = data_y
            num1 = num1 + lenth
        data4 = data2[:num1, :]
        data4 = pd.DataFrame(data4, columns=name)
        data4_1 = data4[
            ['title', 'score', 'peptide1', 'peptide2', 'site1', 'site2', 'combine_peptide', 'combine_peptide_z',
             'charge',
             'm_z', 'rt', 'k0', 'Fragment_charge', 'Fragment_type',
             'Fragment_num', 'Neutral_loss', 'Fragment_intensity', 'Fragment_m_z_calculation']]
        data5 = self.transfer_to_DIANN_format(data4)
        return data6, data4_1, data5

    def process(self, plinkfile, mgf_dir):
        """主处理函数 - 增加Ion Mobility检测"""
        print('Processing data.......')

                                  
        spectrum = np.array(pd.read_csv(mgf_dir, sep='!'))
        spectrum = spectrum.flatten()
        has_ion_mobility = any('ION_MOBILITY=' in str(line) for line in spectrum)
        print(f"Ion Mobility detected: {has_ion_mobility}")

                    
        plink_crosslink_data = self.change_plink_filter_crosslink(plinkfile, spectrum)
                                                                                          

                   
        if has_ion_mobility:
            plink_crosslink_data_ccs = self.filter_plink_precursor_results(plink_crosslink_data)
        plink_crosslink_data_rt = self.filter_plink_peptide_results(plink_crosslink_data)

        complete_normal_library = self.genenrate_all_crosslink_fragment(plink_crosslink_data, mgf_dir)
                                                                                                              

        outputdir = mgf_dir.split('.mgf')[0]
        if has_ion_mobility:
            plink_crosslink_data_ccs.to_csv(f'{outputdir}_ccs.csv', index=False)
        plink_crosslink_data_rt.to_csv(f'{outputdir}_rt.csv', index=False)
                                                                                         
                                                                                             
        complete_normal_library.to_csv(f'{outputdir}_complete_normal_library.csv', index=False)

        rt_dir = f'{outputdir}_rt.csv'
        ccs_dir = f'{outputdir}_ccs.csv'
        msms_dir = f'{outputdir}_complete_normal_library.csv'
        candidate_msms_dir = f'{outputdir}_all_candidate_msms.csv'
        candidate_rtccs_dir = f'{outputdir}_all_candidate.csv'

        print(f"Processing completed. Ion Mobility support: {has_ion_mobility}")
        return msms_dir, ccs_dir, rt_dir, candidate_msms_dir, candidate_rtccs_dir, has_ion_mobility



if __name__ == '__main__':
    plink = plink_with_msconvert_mgf()
    msms_dir, ccs_dir, rt_dir, candidate_msms_dir, candidate_rtccs_dir = plink.process(
        'G:\moran\XXX\PXD017620\C_Lee_090916_ymitos_BS3_XL_B13_C1\C_Lee_090916_ymitos_BS3_XL_B13_C1_10_Rep1',
        'G:\moran\XXX\PXD017620\C_Lee_090916_ymitos_BS3_XL_B13_C1\C_Lee_090916_ymitos_BS3_XL_B13_C1_10_Rep1.mgf'
    )