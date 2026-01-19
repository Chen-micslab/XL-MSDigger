from Deep4D_XL.predict_crosslink_msms_for_rescore_C import predict as predict_msms
from Deep4D_XL.predict_crosslink_ccs_for_rescore import predict as predict_ccs
from Deep4D_XL.predict_crosslink_rt_for_rescore import predict as predict_rt
from Deep4D_XL.dataset.Crosslink_Encoding_msms_C import encoding_with_label_candidate as encode_msms
from Deep4D_XL.dataset.Crosslink_Encoding_ccs_C import crosslink_ccs_encoding_with_label_candidate as encode_ccs
from Deep4D_XL.dataset.Crosslink_Encoding_rt_C import crosslink_rt_encoding_with_label_candidate as encode_rt
import pandas as pd

class generate_feature():
    def __init__(self):
        self.batch_size = 200

    def feature_encoding(self, rt_filedir, ccs_filedir, msms_filedir):
        feature_file = encode_msms(msms_filedir)
        encode_rt(rt_filedir)
                                 
        return feature_file

    def combine_feature(self, crosslink_data, msms_feature, rt_feature):
                                                                                       
        crosslink_data.reset_index(drop=True, inplace=True)
        msms_feature['pearson'].fillna(0, inplace=True)
        msms_feature['spearman'].fillna(0, inplace=True)
        feature_list1 = ['match_num', 'match_num1', 'match_num2',
                         'both_m_p_num', 'both_m_p_num1', 'both_m_p_num2', 'cosine', 'SA', 'pearson', 'spearman']
        crosslink_data['Order'] = pd.to_numeric(crosslink_data['Order'], errors='coerce').astype('Int64')
        msms_feature['Order'] = pd.to_numeric(msms_feature['Order'], errors='coerce').astype('Int64')
        rt_feature['Order'] = pd.to_numeric(rt_feature['Order'], errors='coerce').astype('Int64')
        crosslink_data = crosslink_data.merge(msms_feature[["Order"] + feature_list1], on="Order", how="left").merge(rt_feature[["Order", "rt_AE"]], on="Order", how="left")
                     
        return crosslink_data

    def run(self, msms_filedir, rtccs_filedir, msms_param_dir, rt_param_dir):
        feature_dir = self.feature_encoding(rtccs_filedir, rtccs_filedir, msms_filedir)
        msms_feature = predict_msms(feature_dir, msms_param_dir, self.batch_size)
        rt_feature = predict_rt(feature_dir, rt_param_dir, self.batch_size)
        crosslink_data = pd.read_csv(rtccs_filedir)
        total_feature = self.combine_feature(crosslink_data, msms_feature, rt_feature)
        return total_feature

