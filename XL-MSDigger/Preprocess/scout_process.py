                      
                       

import pandas as pd
import numpy as np
import math
import os
import shutil
import sys
import re
from Preprocess.mass_cal import mz_cal as M

class scout_with_msconvert_mgf():                                                       

    def __init__(self, crosslinker='DSBU', fragment_ppm=0.00002, fragment_num=24, min_mz=0, max_mz=1700,
                 intensity_threshold=0, plink_score_cutoff=1):
        self.crosslinker = crosslinker
        self.frag_ppm = fragment_ppm                    
        self.frag_num = fragment_num                     
        self.min_mz = min_mz                               
        self.max_mz = max_mz                               
        self.intensity_threshold = intensity_threshold                        
        self.plink_score_cutoff = plink_score_cutoff                         

                                                      
    @staticmethod
    def _fdr_to_q(fdr: np.ndarray) -> np.ndarray:
        """将按分值降序累积得到的 FDR 序列转成单调非增的 q-value。"""
        q = np.empty_like(fdr, dtype=float)
        m = np.inf
        for i in range(len(fdr) - 1, -1, -1):
            v = fdr[i]
            if v < m:
                m = v
            q[i] = m
        return np.clip(q, 0.0, 1.0)

    @staticmethod
    def _parse_mgf_index(mgf_path: str) -> pd.DataFrame:
        """扫描 MGF，返回 index(1-based)、rt(分钟)、title。"""
        idx, titles, rts = 0, [], []
        with open(mgf_path, 'r', encoding='utf-8', errors='ignore') as f:
            in_block, cur_title, cur_rt = False, "", np.nan
            for raw in f:
                line = raw.strip()
                if not in_block:
                    if line == "BEGIN IONS":
                        in_block, cur_title, cur_rt = True, "", np.nan
                    continue
                if line.startswith("TITLE="):
                    cur_title = line[6:]
                    continue
                if line.startswith("RTINSECONDS="):
                    try:
                        cur_rt = float(line.split("=", 1)[1]) / 60.0
                    except:
                        cur_rt = np.nan
                    continue
                if line.startswith("RTINMINUTES="):
                    try:
                        cur_rt = float(line.split("=", 1)[1])
                    except:
                        cur_rt = np.nan
                    continue
                if line == "END IONS":
                    idx += 1
                    titles.append(cur_title)
                    rts.append(cur_rt)
                    in_block = False
        return pd.DataFrame({"index": np.arange(1, idx + 1, dtype=int), "rt": rts, "title": titles})

    def _clean_peptide(self, s: str) -> str:
        """
        将带修饰的肽串清洗为仅含氨基酸字母（保留 e/X/U）。
        规则：
        1) 把 Met 氧化的数值写法 M(+15.9949..)（含 ()[]{}、可带Da）统一替换为 'e'
        2) 兼容文字写法 M(ox)/M(Oxidation)/M[Oxidation] -> 'e'
        3) Carbamidomethyl Cys (C(+57.021460) / C[+57.021460]) 直接替换为 'C'
        4) 其余任意修饰去壳（保留氨基酸本体）
        5) 只保留 A–Z 字母以及 X/U/e
        """
        if s is None:
            return ""
        s = str(s)

                                                                                
        s = re.sub(r"M\s*[\(\[\{]\s*\+?\s*15\.99\d*\s*(?:Da)?\s*[\)\]\}]",
                "e", s, flags=re.IGNORECASE)

                                                             
        s = re.sub(r"M\s*[\(\[\{]\s*(?:ox|oxidation)\s*[\)\]\}]",
                "e", s, flags=re.IGNORECASE)

                                                                          
        s = re.sub(r"C\s*[\(\[\{]\s*\+?57\.02146\d*\s*[\)\]\}]",
                "C", s, flags=re.IGNORECASE)

                              
                                                            
        s = re.sub(r"([A-Za-z])\s*[\(\[\{][^\)\]\}]*[\)\]\}]", r"\1", s)

                                        
        s = re.sub(r"[^A-Za-zXUe]", "", s)

        return s


                                                                            
    def change_scout(self, scout_csv: str, mgf_path: str, q_threshold: float = 0.01):
        """
        一次解析 scout CSV + MGF：返回 (total_df, filtered_df=TT 且 Q-value<阈值)。
        列集合与后续流程保持兼容。
        """
        df = pd.read_csv(scout_csv).rename(columns={
            "AlphaPeptide": "peptide1_raw",
            "BetaPeptide": "peptide2_raw",
            "AlphaMappings": "protein1",
            "BetaMappings": "protein2",
            "ExperimentalMZ": "m_z",
        })

                                
        bad_aas = ["B", "J", "O", "U", "X", "Z"]
        pattern = "|".join(bad_aas)
        df = df[~df["peptide1_raw"].str.contains(pattern, na=False)]
        df = df[~df["peptide2_raw"].str.contains(pattern, na=False)]
               
        df["peptide1"] = df["peptide1_raw"].apply(self._clean_peptide)
        df["peptide2"] = df["peptide2_raw"].apply(self._clean_peptide)
        df["m_z"]    = pd.to_numeric(df["m_z"], errors="coerce")
        df["site1"]  = df["AlphaPos"].astype(int) + 1                       
        df["site2"]  = df["BetaPos"].astype(int) + 1
        df["len1"]   = df["peptide1"].astype(str).str.count(r"[A-Za-z]")
        df["len2"]   = df["peptide2"].astype(str).str.count(r"[A-Za-z]")
        df["charge"] = pd.to_numeric(df["Charge"], errors="coerce").astype("Int64")

                       
        def _make_comb(r):
            p1, p2 = list(str(r.peptide1)), list(str(r.peptide2))
            try: p1.insert(int(r.site1), 'U')
            except: pass
            try: p2.insert(int(r.site2), 'U')
            except: pass
            pep1, pep2 = ''.join(p1), ''.join(p2)
            pep = f"{pep1}X{pep2}"
            cz  = int(r.charge) if pd.notna(r.charge) else ''
            return pep, f"{pep}{cz}"
        comb = df.apply(_make_comb, axis=1, result_type="expand")
        df["combine_peptide"]   = comb[0]
        df["combine_peptide_z"] = comb[1]

                                        
        def _split_ids(s): return set(x.strip() for x in str(s).split(";") if x.strip())
        df["Protein_Type"] = df.apply(
            lambda r: 1 if _split_ids(r.protein1) & _split_ids(r.protein2) else 2, axis=1
        )

                         
        cls_map = {"fulltarget": 2, "fulldecoy": 0, "alphatarget": 1, "betatarget": 1}
        df["Target_Decoy"] = (
            df["Class"].astype(str).str.lower().str.replace(r"[\s_]+", "", regex=True)
              .map(cls_map).fillna(2).astype(int)
        )

                                     
        df["Q-value"] = np.nan
        inter = df["Protein_Type"] == 2
        if inter.any():
            sub = df.loc[inter].sort_values("XLScore", ascending=False)
            t  = sub["Target_Decoy"].to_numpy(int)
            TT = (t == 2).astype(int).cumsum()
            TD = (t == 1).astype(int).cumsum()
            DD = (t == 0).astype(int).cumsum()
            with np.errstate(divide="ignore", invalid="ignore"):
                fdr = (TD - DD) / TT
                fdr[~np.isfinite(fdr)] = 1.0
                fdr = np.maximum(fdr, 0.0)
            qv = self._fdr_to_q(fdr)
            df.loc[sub.index, "Q-value"] = qv
        df.loc[df["Protein_Type"] == 1, "Q-value"] = 0.0

                                                 
        mgf_idx = self._parse_mgf_index(mgf_path).set_index("index")
        scans = df["ScanNumber"].astype(int)
        df["rt"]    = scans.map(mgf_idx["rt"])
        df["title"] = scans.map(mgf_idx["title"])

                                        
        df["score"] = pd.to_numeric(df["XLScore"], errors="coerce")

               
        df["protein"]       = df["protein1"].astype(str) + "|" + df["protein2"].astype(str)
        df["protein_type"]  = df["Protein_Type"].astype(int)
        df["k0"]            = np.nan
        df["ccs"]           = np.nan
        df["precursor_Mass_Error(ppm)"] = np.nan
        df["intensity"]     = np.nan
        df["filename"]      = df["title"].astype(str).str.split(".", n=1).str[0]
        df["cmpd"]          = np.nan
        df["peptide_type"]  = 3                

        cols = [
            'title','score','peptide1','peptide2','site1','site2',
            'combine_peptide','combine_peptide_z','charge','m_z','rt','k0',
            'len1','len2','protein','protein_type','ccs',
            'precursor_Mass_Error(ppm)','intensity','filename','cmpd','peptide_type',
            'Target_Decoy','Q-value','ScanNumber'
        ]
        total_df    = df[cols].copy()
        total_df = total_df.rename(columns={'ScanNumber': 'Order'})
        filtered_df = df[(df["Target_Decoy"] == 2) & (df["Q-value"] < q_threshold)][cols].copy()
        filtered_df = filtered_df.rename(columns={'ScanNumber': 'Order'})
        total_df = total_df[(total_df["len1"] < 50) & (total_df["len2"] < 50)]
        filtered_df = filtered_df[(filtered_df["len1"] < 50) & (filtered_df["len2"] < 50)]
        return total_df, filtered_df

                                                             
    def extract_from_symbol(self, s, symbol):
        index = []
        for i in range(len(s)):
            if s[i] == symbol:
                index.append(i)
        start_index = index[0]
        end_index = index[1]
        return s[(start_index + 1):end_index]

    def modif_xlpeptide(self, pep1, pep2, mod):
        len1 = len(pep1)
        len2 = len(pep2)
        if 'M' in mod:
            find_all = lambda c, ss: [x for x in range(c.find(ss), len(c)) if c[x] == ss]
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
        id1 = combine_peptide.find('X')
        pep1 = combine_peptide[:id1]
        pep2 = combine_peptide[(id1 + 1):]
        site1 = pep1.find('U')
        peptide1 = pep1.replace('U', '')
        site2 = pep2.find('U')
        peptide2 = pep2.replace('U', '')
        return peptide1, peptide2, site1, site2

    def filter_plink_precursor_results(self, data):
        """同肽同电荷保留最优PSM（基于score最小），再用cutoff过滤。"""
        data1 = data.sort_values('combine_peptide_z', ignore_index=True)
        peptide_list = np.array(data1['combine_peptide_z'])
        name = list(data1)
        data5 = np.array(data1, dtype=object)
        peptide = peptide_list[0]
        index_list = []
        lenth1 = 0
        for i in range(len(peptide_list)):
            if peptide_list[i] == peptide:
                index_list.append(i)
            else:
                data2 = data1.iloc[index_list]
                q_list = list(data2['score'])
                psm_list = list(data2['title'])
                psm = psm_list[int(q_list.index(min(q_list)))]
                data3 = data2[data2['title'] == psm]
                lenth2 = len(data3['combine_peptide_z'])
                data5[lenth1:(lenth1 + lenth2), :] = np.array(data3, dtype=object)
                lenth1 = lenth1 + lenth2
                index_list = [i]
                peptide = peptide_list[i]
        data2 = data1.iloc[index_list]
        q_list = list(data2['score'])
        psm_list = list(data2['title'])
        psm = psm_list[int(q_list.index(min(q_list)))]
        data3 = data2[data2['title'] == psm]
        lenth2 = len(data3['combine_peptide_z'])
        data5[lenth1:(lenth1 + lenth2), :] = np.array(data3, dtype=object)
        lenth1 = lenth1 + lenth2
        data6 = pd.DataFrame(data5[:lenth1, :], columns=name)
        data6['score'] = pd.to_numeric(data6['score'], errors='coerce')
        data7 = data6[data6['score'] < float(self.plink_score_cutoff)]
        return data7

    def filter_plink_peptide_results(self, data):
        """不同电荷只留一个最佳PSM（基于score最小），再用cutoff过滤。"""
        data1 = data.sort_values('combine_peptide', ignore_index=True)
        peptide_list = np.array(data1['combine_peptide'])
        name = list(data1)
        data5 = np.array(data1, dtype=object)
        peptide = peptide_list[0]
        index_list = []
        lenth1 = 0
        for i in range(len(peptide_list)):
            if peptide_list[i] == peptide:
                index_list.append(i)
            else:
                data2 = data1.iloc[index_list]
                q_list = list(data2['score'])
                psm_list = list(data2['title'])
                psm = psm_list[int(q_list.index(min(q_list)))]
                data3 = data2[data2['title'] == psm]
                lenth2 = len(data3['combine_peptide'])
                data5[lenth1:(lenth1 + lenth2), :] = np.array(data3, dtype=object)
                lenth1 = lenth1 + lenth2
                index_list = [i]
                peptide = peptide_list[i]
        data2 = data1.iloc[index_list]
        q_list = list(data2['score'])
        psm_list = list(data2['title'])
        psm = psm_list[int(q_list.index(min(q_list)))]
        data3 = data2[data2['title'] == psm]
        lenth2 = len(data3['combine_peptide'])
        data5[lenth1:(lenth1 + lenth2), :] = np.array(data3, dtype=object)
        lenth1 = lenth1 + lenth2
        data6 = pd.DataFrame(data5[:lenth1, :], columns=name)
        data6['score'] = pd.to_numeric(data6['score'], errors='coerce')
        data7 = data6[data6['score'] < float(self.plink_score_cutoff)]
        return data7

                                                                 
    def match_msms(self, spectrum, m_z, precursor, spe, id_s):
        title = 'TITLE=' + precursor
        id_x = np.where(spe == title)[0]
        if len(id_x) == 0:
            print(f'[Warning] Cannot find spectrum title: {title}')
            return [0] * len(m_z), [0] * len(m_z), np.nan

        start_idx = id_s[int(id_x[0])]

                                   
        rt_min = np.nan
        j = start_idx
        while j >= 0 and str(spectrum[j]).strip() != "BEGIN IONS":
            line = str(spectrum[j]).strip()
            if line.startswith("RTINSECONDS="):
                try:
                    rt_min = float(line.split("=", 1)[1]) / 60.0
                except:
                    rt_min = np.nan
                break
            if line.startswith("RTINMINUTES="):
                try:
                    rt_min = float(line.split("=", 1)[1])
                except:
                    rt_min = np.nan
                break
            j -= 1

        msms, intensity = [], []
        for i in range(start_idx + 1, len(spectrum)):
            line = str(spectrum[i]).strip()
            if line == "END IONS":
                break
            parts = line.split()
            if len(parts) == 2:
                try:
                    mz_val = float(parts[0])
                    int_val = float(parts[1])
                    msms.append(mz_val)
                    intensity.append(int_val)
                except:
                    continue

        msms = np.array(msms)
        intensity = np.array(intensity)

        m_z_1, inten = [], []
        for mz in m_z:
            if mz == -1:
                m_z_1.append(-1)
                inten.append(-1)
            else:
                msms1 = np.abs(msms - mz) / msms
                if np.min(msms1) <= self.frag_ppm:                
                    m_z_1.append(msms[np.argmin(msms1)])
                    inten.append(intensity[np.argmin(msms1)])
                else:
                    m_z_1.append(0)
                    inten.append(0)

        return m_z_1, inten, rt_min

                                                       
    def crosslink_ion_generation(self, peptide1, peptide2):
        """可裂解交联版本：包含 1b/1y/2b/2y 及其短/长 stub（_s/_l），电荷 +1,+2,+3，noloss"""
        len1 = len(peptide1)
        len2 = len(peptide2)
        z = ['1', '2', '3']                  
        l = ['noloss'] * 3                 
        by_1 = (['1b'] * 3 + ['1y'] * 3 + ['1b_s'] * 3 + ['1y_s'] * 3 + ['1b_l'] * 3 + ['1y_l'] * 3) * (len1 - 1)
        c = []
        for i in range(len1 - 1):
            j = i + 1
            c = c + [j] * 18
        for i in range(len2 - 1):
            j = i + 1
            c = c + [j] * 18
        by_2 = (['2b'] * 3 + ['2y'] * 3 + ['2b_s'] * 3 + ['2y_s'] * 3 + ['2b_l'] * 3 + ['2y_l'] * 3) * (len2 - 1)
        by = by_1 + by_2
        z = z * 6 * (len1 + len2 - 2)
        l = l * 6 * (len1 + len2 - 2)
        data = np.column_stack((np.array(c), np.array(by), np.array(z), np.array(l)))
        return data

    def genenrate_all_crosslink_fragment(self, plink_data, mgf_dir):
        """
        生成完整碎片文库（complete_normal_library 等价物）
        依赖 plink_data 至少包含字段：
        'title','score','peptide1','peptide2','site1','site2',
        'combine_peptide','combine_peptide_z','charge','m_z','rt','k0'
        """
        sys.stdout.write("Loading file......\r")
        spectrum = np.array(pd.read_csv(mgf_dir, sep='!')).flatten()
        spe, id_s = [], []
        for i in range(len(spectrum)):
            if str(spectrum[i]).startswith('TIT'):
                spe.append(spectrum[i])
                id_s.append(i)
        spe = np.array(spe)
        id_s = np.array(id_s)

        data = plink_data.copy()
        name = list(data)
        pep1 = np.array(data['peptide1'])
        pep2 = np.array(data['peptide2'])

                                 
        total_rows = 0
        for i in range(len(pep1)):
            total_rows += (len(pep1[i]) + len(pep2[i]) - 2)
        total_rows *= 18

        data_np = np.array(data, dtype=object)
        template = np.array(list(data_np[0, :]) + [0, 0, 0, 0, 0, 0, 0], dtype=object)
        data1 = np.tile(template, [total_rows, 1])
        name_ext = name + ['Fragment_num', 'Fragment_type', 'Fragment_charge', 'Neutral_loss']

        sys.stdout.write("Generating library......\r")
        write_ptr = 0
        for i in range(len(pep1)):
            peptide1 = pep1[i]
            peptide2 = pep2[i]
            curr_rows = (len(peptide1) + len(peptide2) - 2) * 18

            base_row = data_np[i, :]
            data_y = np.tile(base_row, [curr_rows, 1])
            data_x = self.crosslink_ion_generation(peptide1, peptide2)
            data_xy = pd.DataFrame(np.column_stack((data_y, data_x)), columns=name_ext)

            cl_peptide = data_xy['combine_peptide']
            by_type = data_xy['Fragment_type']
            Fragment_num = data_xy['Fragment_num']
            Fragment_charge = data_xy['Fragment_charge']
            Neutral_loss = data_xy['Neutral_loss']
            title = data_xy['title']

                      
            m = M()
            mz_theory = []
            for j in range(len(cl_peptide)):
                mz = m.crosslink_peptide_msms_m_z_cleavable(
                    cl_peptide[j],
                    self.crosslinker,
                    by_type[j],
                    int(Fragment_num[j]),
                    int(Fragment_charge[j]),
                    Neutral_loss[j]
                )
                mz_theory.append(round(mz, 5))

                               
            mz_exp, inten, rt_min = self.match_msms(spectrum, mz_theory, title.iloc[0], spe, id_s)

                              
            if np.sum(np.array(inten) > 0) > 2:
                inten = (np.array(inten) / np.max(inten)).tolist()

                                                   
                data_xy_p = data_xy.copy()
                data_xy_p["Fragment_m_z_calculation"] = mz_theory
                data_xy_p["Fragment_m_z_experiment"]  = mz_exp
                data_xy_p["Fragment_intensity"]       = inten

                                     
                precursor_z = int(data_xy_p["charge"].iloc[0])

                                    
                mask = pd.to_numeric(data_xy_p["Fragment_charge"], errors="raise") > precursor_z
                cols_to_zero = ["Fragment_m_z_calculation", "Fragment_m_z_experiment", "Fragment_intensity"]
                data_xy_p.loc[mask, cols_to_zero] = -1

                                       
                data1[write_ptr:(write_ptr + curr_rows), :] = data_xy_p.to_numpy(dtype=object)
                write_ptr += curr_rows

                                         
                data.loc[data.index[i], "rt"] = rt_min

                    
        name_final = name_ext + ['Fragment_m_z_calculation', 'Fragment_m_z_experiment', 'Fragment_intensity']
        data1 = pd.DataFrame(data1[:write_ptr, :], columns=name_final)

                                   
        data1 = data1[data1['Fragment_m_z_experiment'] > self.min_mz]
        data1 = data1[data1['Fragment_m_z_experiment'] < self.max_mz]
        data1 = data1[data1['Fragment_intensity'] > self.intensity_threshold]

        complete_normal_library = data1
        return complete_normal_library

                                                                        
    def process(self, scout_csv, mgf_dir):
        print('Processing data.......')

                                  
        plink_crosslink_candidate, plink_crosslink_data = self.change_scout(
            scout_csv, mgf_dir, q_threshold=0.01
        )
                                                   

        plink_crosslink_data_rt  = self.filter_plink_peptide_results(plink_crosslink_data)

                                             
        complete_normal_library  = self.genenrate_all_crosslink_fragment(plink_crosslink_data, mgf_dir)
        candidate_normal_library = self.genenrate_all_crosslink_fragment(plink_crosslink_candidate, mgf_dir)

                      
        outputdir = scout_csv.split('.csv')[0]
        plink_crosslink_data_rt.to_csv(f'{outputdir}_rt.csv', index=False)
        plink_crosslink_candidate.to_csv(f'{outputdir}_all_candidate.csv', index=False)
        candidate_normal_library.to_csv(f'{outputdir}_all_candidate_msms.csv', index=False)
        complete_normal_library.to_csv(f'{outputdir}_complete_normal_library.csv', index=False)

        rt_dir = f'{outputdir}_rt.csv'
        msms_dir = f'{outputdir}_complete_normal_library.csv'
        candidate_msms_dir = f'{outputdir}_all_candidate_msms.csv'
        candidate_rtccs_dir = f'{outputdir}_all_candidate.csv'
        return msms_dir, rt_dir, candidate_msms_dir, candidate_rtccs_dir


if __name__ == '__main__':
    scout = scout_with_DA_mgf()
    scout.process(
        '/Users/moranchen/Documents/Project/Deep4D_XL/Review/data/PXD012546_DSBU/scout/unfilter_XL_A8.csv',
        '/Users/moranchen/Documents/Project/Deep4D_XL/Review/data/PXD012546_DSBU/scout/R2_A8.mgf'
    )
