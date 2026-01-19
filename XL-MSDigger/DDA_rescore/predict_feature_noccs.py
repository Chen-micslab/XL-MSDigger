from Deep4D_XL.predict_crosslink_msms_for_rescore import predict as predict_msms
from Deep4D_XL.predict_crosslink_ccs_for_rescore import predict as predict_ccs
from Deep4D_XL.predict_crosslink_rt_for_rescore import predict as predict_rt
from Deep4D_XL.dataset.Crosslink_Encoding_msms import encoding_with_label_candidate as encode_msms
from Deep4D_XL.dataset.Crosslink_Encoding_ccs import crosslink_ccs_encoding_with_label_candidate as encode_ccs
from Deep4D_XL.dataset.Crosslink_Encoding_rt import crosslink_rt_encoding_with_label_candidate as encode_rt
import pandas as pd

class generate_feature():
    def __init__(self):
        self.batch_size = 200

    def feature_encoding(self, rt_filedir, msms_filedir):
        feature_file = encode_msms(msms_filedir)
        encode_rt(rt_filedir)
        return feature_file

    def combine_feature(self, crosslink_data, msms_feature, rt_feature):
        crosslink_data = crosslink_data[~crosslink_data['Peptide'].str.contains('U')]
        crosslink_data.reset_index(drop=True, inplace=True)
        msms_feature['pearson'].fillna(0, inplace=True)
        msms_feature['spearman'].fillna(0, inplace=True)
        feature_list1 = ['match_num', 'match_num1', 'match_num2',
                         'both_m_p_num', 'both_m_p_num1', 'both_m_p_num2', 'cosine', 'SA', 'pearson', 'spearman']
        crosslink_data = crosslink_data.sort_values('Order', ignore_index=True)
        msms_feature = msms_feature.sort_values('Order', ignore_index=True)
        rt_feature = rt_feature.sort_values('Order', ignore_index=True)
        crosslink_data[feature_list1] = msms_feature[feature_list1]
        crosslink_data['rt_AE'] = rt_feature['rt_AE']
        return crosslink_data

    def run(self, msms_filedir, rtccs_filedir, msms_param_dir, rt_param_dir):
        feature_dir = self.feature_encoding(rtccs_filedir, msms_filedir)
        msms_feature = predict_msms(feature_dir, msms_param_dir, self.batch_size)
        rt_feature = predict_rt(feature_dir, rt_param_dir, self.batch_size)
        crosslink_data = pd.read_csv(rtccs_filedir)
        total_feature = self.combine_feature(crosslink_data, msms_feature, rt_feature)
        return total_feature

