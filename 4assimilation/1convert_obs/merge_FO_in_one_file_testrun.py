'''
This script merge ensemble of FO data(usually in ens_size number of file)
to a single file, for better fortran reading.

one line contains {ens_size} number of FO data.

written by Haoxing in 2023-05-12/NJU-Xianlin Campus 
'''
import os
import pandas as pd
import numpy as np
import subprocess
import logging


# def fmt_write(fname, array, fmt):
#     with open(fname, 'w') as f:
#         for index, row in df.iterrows():
#             write_fmt = ff.FortranRecordWriter(fmt)
#             out = write_fmt.write(
#                 [row['sitecode'], row['lat_y'], row['lon_y']]
#             )
#             #logging.info('Now write: '+out)
#             f.write(out)
#             f.write('\n')

#     return    

def load_clear_sky_indices(mask_file, nobs):
    if mask_file and os.path.exists(mask_file):
        mask = np.loadtxt(mask_file).astype(bool).reshape(-1)
        if mask.size != nobs:
            raise ValueError(f"clear-sky mask size {mask.size} does not match nobs {nobs}: {mask_file}")
        print(f"Using clear-sky mask {mask_file}: keep {np.sum(mask)} / {nobs}")
        return np.where(mask)[0]
    return None

if __name__ == "__main__":
    
    # dir_list="d01_0030 d01_0100 d02_0030 d02_0040 d02_0050 d02_0100 d03_0030 d03_0035 d03_0040 d03_0045 d03_0050 d03_0055 d03_0100".split(" ")
    # dir_list="d01_0030 d02_0030 d02_0040 d02_0050 d03_0030 d03_0035 d03_0040 d03_0045 d03_0050 d03_0055".split(" ")
    # time_from_env=os.environ.get("current_time")
    time_from_env='10_00_00'
    time_list = [time_from_env]
    domain='d01'
    sensor='AMSUA'
    ch=4
    ens_size = 50
    memlist=list(np.arange(1,ens_size+1))
    # memlist.extend([31,32,36,38,39,40,41,42,43,45,47,48,49,50,51,54,62,69,70,71,73])
    TOP_DIR="/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs"
    BT_DIR=os.environ.get('ENS_BT_DIR','/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/4ens_BT')
    OBS_BT_DIR=os.environ.get('OBS_BT_DIR',f'/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/3obs_BT/{sensor}')
    
    for time in time_list:
        output = []
        dir_name='ensBT'+domain+'_'+time
        OUT_DIR=f'{TOP_DIR}/mergeFO{domain}_{time}'
        subprocess.run(['mkdir','-p',f'{OUT_DIR}'])
        # expt = 'quantile_eakf'
        # work_dir = f'/share/home/lililei4/haoxing/quantile_data/data_BT/posterior_FO_data/{expt}'
        work_dir = f'{TOP_DIR}/{dir_name}/prior_BT/'
        subprocess.run(['mkdir','-p',f'{work_dir}'])
        
        ens_fname_prefix = f'ens_ch{ch}_mem'
        # ens_fname_prefix = 'interp2obs_ens_ch1025_mem'
        # ens_fname_prefix = 'posterior_FO_mem'
        # out_fname = 'merged_FO_data.txt_interp'
        out_fname = f'{OUT_DIR}/merged_FO_data.txt'
        fmtFO = '(50f11.7)'
        print(work_dir)
        bt_subdir=os.environ.get('ENS_BT_SUBDIR',f'BT_{time}')
        obs_subdir=os.environ.get('OBS_BT_SUBDIR',f'BT_{time}')
        mask_file=os.environ.get('CLEAR_SKY_MASK_FILE',f'{OBS_BT_DIR}/{obs_subdir}/clear_sky_mask.txt')
        if os.environ.get('USE_CLEAR_SKY_MASK', '1' if os.environ.get('rttov_scatt','0') == '0' else '0') != '1' and os.environ.get('CLEAR_SKY_MASK_FILE') is None:
            mask_file=''
        clear_sky_indices=None
        
        os.chdir(work_dir)
        for i,mem in enumerate(memlist):
            formatted_i = '{:03d}'.format(mem)
            in_fname = f'{ens_fname_prefix}{formatted_i}'
            subprocess.run(['ln','-sf',f'{BT_DIR}/mem{formatted_i}/{sensor}/{bt_subdir}/obs_{domain}_ch{ch}_totalline.txt',f'{work_dir}/{in_fname}'])
            arr = np.loadtxt(in_fname,dtype='str')
            if clear_sky_indices is None:
                clear_sky_indices=load_clear_sky_indices(mask_file, len(arr))
            if clear_sky_indices is not None:
                arr = arr[clear_sky_indices]
            output.append(arr)

        tmp = np.array(output).transpose() # make sure structure is [data_num,ens_size]
        #print(tmp[:,0])
        np.savetxt(out_fname, tmp, delimiter=' ',fmt='%s')
