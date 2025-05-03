import os
from datetime import datetime
import numpy as  np
import pandas as pd
from  Deep4D_XL.predict_msms import predict_nolabel_msms
from  Deep4D_XL.predict_ccs import predict_nolabel_ccs
from  Deep4D_XL.predict_rt import predict_nolabel_rt
from  Deep4D_XL.dataset.Crosslink_Encoding_msms import encoding_without_label as encoding_msms
from  Deep4D_XL.dataset.Generate_spectral_library import merge_information, generate_4d_library

def create_taskname(filename):
    a = str(datetime.now())
    a = a.replace(':', '-')
    a = a.replace('.', '-')
    a = a.replace(' ', '-')
    a = a[:19]
    task_name = f'{filename}_task_{a}'
    return task_name

def generate_predicted_library(filedir, maxcharge, slice, batch_size, load_msms_param_dir, load_rt_param_dir, load_ccs_param_dir):
    filename = filedir.split('/')[-1].split('.')[0]
    task_name = create_taskname(filename)
    task_dir = filedir[:filedir.rfind('/')]
    os.mkdir(f'{task_dir}/{task_name}')
    os.mkdir(f'{task_dir}/{task_name}/output')
    task_dir1 = f'{task_dir}/{task_name}'
    onehot_dir = f'{task_dir}/{task_name}/{filename}'
    import time
    aa = time.perf_counter()
    encoding_msms(filedir, onehot_dir, maxcharge, slice)
    bb = time.perf_counter()
    print(bb-aa)
    predict_nolabel_msms(task_dir1, filename, slice, batch_size, load_msms_param_dir)
    predict_nolabel_ccs(task_dir1, filename, slice, batch_size, load_ccs_param_dir)
    predict_nolabel_rt(task_dir1, filename, slice, batch_size, load_rt_param_dir)
    print('merging prediction information.........')
    for slice_num in range(slice):
        msms_dir = f'{task_dir}/{task_name}/output/{filename}_slice{slice_num}_pre_msms.csv'
        ccs_dir = f'{task_dir}/{task_name}/output/{filename}_slice{slice_num}_pre_ccs.csv'
        rt_dir = f'{task_dir}/{task_name}/output/{filename}_slice{slice_num}_pre_rt.csv'
        print(f'{slice_num} in slice{slice}: merging 4d information...........')
        data = merge_information(msms_dir, ccs_dir, rt_dir)
        print(f'{slice_num} in slice{slice}: generating 4d library...........')
        data1, data2  = generate_4d_library(data)
        name1 = list(data1)
        data1 = np.array(data1)
        data2 = np.array(data2)
        if slice_num == 0:
            data_lib = data2
            data_lib1 = data1
        else:
            data_lib = np.row_stack((data_lib, data2))
            data_lib1 = np.row_stack((data_lib1, data1))
    normal_lib = pd.DataFrame(data_lib1, columns=name1)
    name = ['ModifiedPeptide','PrecursorCharge','PrecursorMz','FragmentCharge','ProductMz','Tr_recalibrated','IonMobility','LibraryIntensity']
    diann_lib = pd.DataFrame(data_lib, columns=name)
    import shutil
    shutil.rmtree(f'{task_dir}/{task_name}')
    return normal_lib, diann_lib

