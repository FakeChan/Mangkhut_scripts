import subprocess
import numpy as np
import os
import math

def read_every_nth_line(file_path, start_line, step):
    selected_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, start=1):  
            if i == start_line or (i > start_line and (i - start_line) % step == 0):
                selected_lines.append(line.strip())
    return selected_lines

def load_clear_sky_indices(mask_file, nobs):
    if mask_file and os.path.exists(mask_file):
        mask = np.loadtxt(mask_file).astype(bool).reshape(-1)
        if mask.size != nobs:
            raise ValueError(f"clear-sky mask size {mask.size} does not match nobs {nobs}: {mask_file}")
        print(f"Using clear-sky mask {mask_file}: keep {np.sum(mask)} / {nobs}")
        return np.where(mask)[0]
    return np.arange(nobs)

def get_lacc_obs_error_std(obs_err_std, obs_dir, obs_subdir):
    obs_err = float(obs_err_std)
    if os.environ.get('USE_LACC_OBS_ERROR_SCALING', '1') != '1':
        return obs_err
    times_file = os.environ.get('LACC_TIMES_FILE', f'{obs_dir}/{obs_subdir}/LACC_times.txt')
    if not obs_subdir.startswith('BT_LACC_') and not os.path.exists(times_file):
        return obs_err
    if os.path.exists(times_file):
        with open(times_file, 'r', encoding='utf-8') as file:
            ntime = sum(1 for line in file if line.startswith('lag_time='))
    else:
        ntime = len(str(os.environ.get('lacc_times', '')).split())
    if ntime > 1:
        obs_err = obs_err / math.sqrt(ntime)
        print(f"Using LACC obs error std scaled by sqrt({ntime}): {obs_err}")
    return obs_err

if __name__ == "__main__":
    #======================================================================================
    #basic model configure
    domain='d01'
    nobs=676
    use_quantile=True
    #======================================================================================
    #path
    cycle_flag=os.environ.get("cycle_flag")
    sensor='AMSUA'
    obs_dir=os.environ.get('OBS_BT_DIR',f'/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/3obs_BT/{sensor}')
    prof_dir=os.environ.get('PROFILE_DIR','/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/profile')
    output_dir='/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs'
    
    if cycle_flag=='1':
        day=os.environ.get("current_day")
        hour=os.environ.get("current_hour")
        min=os.environ.get("current_min")
        obs_err_from_envs=os.environ.get("obs_err_std")
    else:
        day='10'
        hour='00'
        min='00'
        obs_err_from_envs=1.5
    channel=4
    
    obs_subdir=os.environ.get('OBS_BT_SUBDIR',f'BT_{day}_{hour}_{min}')
    obs_file=f'{obs_dir}/{obs_subdir}/obs_d01_ch{channel}_totalline_withpert.txt'
    clear_sky_mask_file=os.environ.get('CLEAR_SKY_MASK_FILE',f'{obs_dir}/{obs_subdir}/clear_sky_mask.txt')
    if os.environ.get('USE_CLEAR_SKY_MASK', '1' if os.environ.get('rttov_scatt','0') == '0' else '0') != '1' and os.environ.get('CLEAR_SKY_MASK_FILE') is None:
        clear_sky_mask_file=''
    profile_subdir=os.environ.get('PROFILE_SUBDIR',f'profile_{domain}')
    para_file=os.environ.get('PARA_FILE',f'{prof_dir}/{profile_subdir}/prof{day}_{hour}:{min}.dat')
    #=======================================================================================
    #parameters read into DART

    #check https://nwp-saf.eumetsat.int/site/software/rttov/documentation/platforms-supported/ for rttov ids
    
    if sensor == 'AMSUA':
        #======================================
        platform=1              #NOAA        ==
        sat=18                  #N18         ==
        sensor=3                #AMSUA       ==
        #======================================
    elif sensor == 'AMSR2':
        #======================================
        platform=29              #GCOM-W     ==
        sat=1                    # 1         ==
        sensor=63                #AMSR2      ==
        #======================================

    intday=int(day)
    inthour=int(hour)
    intmin=int(min)
    if use_quantile:
        obstype=170  #when using DART_quantile
        suffix='quantile'
    else:
        obstype=170 #NOAA_18_AMSUA_TB =   170 when using DART_main
        suffix='main'
    year=2018
    month=9
    second=0
    obs_err=get_lacc_obs_error_std(obs_err_from_envs, obs_dir, obs_subdir)
    obs=np.loadtxt(obs_file)
    clear_sky_indices=load_clear_sky_indices(clear_sky_mask_file, nobs)
    angles=read_every_nth_line(para_file, start_line=74, step=69)
    locations=read_every_nth_line(para_file, start_line=72, step=69)
    #make sure output converting obs to {output_dir}/{domain}
    subprocess.run(['mkdir','-p',f'{output_dir}/obs_{domain}'])
    # subprocess.run(['cd',f'{output_dir}/{domain}/'])

    format_str = "{:3d} {:11.5f} {:11.5f} {:8.1f}" + \
             "{:5d} {:5d} {:5d} {:5d} {:5d} {:5d}" + \
             "{:11.5f} {:11.5f}" + \
             "{:11.1f} {:11.1f}" + \
             "{:5d} {:5d} {:5d} {:5d}"
    data=[]
    for i in clear_sky_indices:
        obs_value=obs[i]
        sat_ze=float(angles[i].split()[0])
        sat_az=float(angles[i].split()[1])
        hgt_obs=90000
        lat=float(locations[i].split()[1])
        lon=float(locations[i].split()[2])
        data.append((obstype,lat,lon,hgt_obs,year,month,intday,inthour,intmin,second,obs_value,obs_err,sat_az,sat_ze,platform,sat,sensor,channel))
    #sequnce: obstype(int),lat,lon,height of obs(hPa),year,month,day,hour,minute,second,
    #obs_value,obs_error,sat_az,sat_ze,platform_id, sat_id, sensor_id, channel
    # file.write(f'{obstype} {lat} {lon} {hgt_obs} {year} {month} {day} {hour} {min} ')
    # file.write(f'{second} {obs_value:.4f} {obs_err} {sat_az} {sat_ze} {platform} {sat} {sensor} {channel}\n')
    nobs_out=len(data)
    with open(f"{output_dir}/obs_{domain}/kctest_{nobs_out}_obsinput_{day}_{hour}_{min}_{suffix}.txt", "w", encoding="utf-8") as file:
        for row in data:
            file.write(format_str.format(*row) + "\n")
