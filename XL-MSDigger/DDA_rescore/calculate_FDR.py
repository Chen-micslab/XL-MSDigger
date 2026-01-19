import numpy as np
import pandas as pd

class FDR_calculate():
    def fdr_to_q_values(self, fdr_values):                
        q_values = np.zeros_like(fdr_values)
        min_q_value = np.max(fdr_values)
        for i in range(len(fdr_values) - 1, -1, -1):
            fdr = fdr_values[i]
            if fdr < min_q_value:
                min_q_value = fdr
            q_values[i] = min_q_value
        q_values = np.maximum(q_values, 0)
        return q_values

    def crosslink_FDR_plink(self, data, col_name):                 
        data1 = data.sort_values(col_name, ignore_index=True, ascending=False)
        type = np.array(data1['Target_Decoy'])
        id_TT = np.array(type == 2).astype(int)
        id_TD = np.array(type == 1).astype(int)
        id_DD = np.array(type == 0).astype(int)
        TT_cumsum = np.cumsum(id_TT)
                          
        TD_cumsum = np.cumsum(id_TD)
        DD_cumsum = np.cumsum(id_DD)
        fdrs = (TD_cumsum - DD_cumsum) / TT_cumsum
        q_values = self.fdr_to_q_values(fdrs)
                         
        data1.insert(0, 'FDR', q_values)
        return data1

    def linear_FDR_plink(self, data, col_name = 'SVM_Score'):                  
        data1 = data.sort_values(col_name, ignore_index=True, ascending=False)
        type = np.array(data1['Target_Decoy'])
        id_T = np.array(type == 2).astype(int)
        id_D = np.array(type == 0).astype(int)
        T_cumsum = np.cumsum(id_T)
        D_cumsum = np.cumsum(id_D)
        fdrs = D_cumsum / T_cumsum
        q_values = self.fdr_to_q_values(fdrs)
        return fdrs, q_values

if __name__ == '__main__':
    F = FDR_calculate()
    data = pd.read_csv('./dataset/data/yeast_26_1_crosslink_feature.csv')
    data1 = data[data['Protein_Type'] == 2]
    data2 = F.crosslink_FDR_plink(data1, col_name='SVM_Score')
    data2.to_csv('./dataset/data/yeast_26_1_crosslink_feature_intra.csv', index=False)