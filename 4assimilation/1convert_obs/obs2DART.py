import subprocess
import numpy as np
import os

def read_every_nth_line(file_path, start_line, step):
    selected_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, start=1):  
            if i == start_line or (i > start_line and (i - start_line) % step == 0):
                selected_lines.append(line.strip())
    return selected_lines

if __name__ == "__main__":
    #======================================================================================
    #basic model configure
    domain='d01'
    nobs=676
    use_quantile=True
    #======================================================================================
    #path
    obs_dir='/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/3obs_BT/AMSUA'
    prof_dir='/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/profile'
    output_dir='/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs'
    day=os.environ.get("current_day")
    hour=os.environ.get("current_hour")
    min=os.environ.get("current_min")
    channel=4
    obs_err_from_envs=os.environ.get("obs_err_std")
    obs_file=f'{obs_dir}/BT_{day}_{hour}_{min}/obs_d01_ch{channel}_totalline_withpert.txt'
    para_file=f'{prof_dir}/profile_{domain}/prof{day}_{hour}:{min}.dat'
    #=======================================================================================
    #parameters read into DART

    #check https://nwp-saf.eumetsat.int/site/software/rttov/documentation/platforms-supported/ for rttov ids
    #======================================
    platform=1              #NOAA        ==
    sat=18                  #N18         ==
    sensor=3                #AMSUA       ==
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
    obs_err=float(obs_err_from_envs)
    obs=np.loadtxt(obs_file)
    angles=read_every_nth_line(para_file, start_line=188, step=185)
    locations=read_every_nth_line(para_file, start_line=186, step=185)
    #make sure output converting obs to {output_dir}/{domain}
    subprocess.run(['mkdir','-p',f'{output_dir}/obs_{domain}'])
    # subprocess.run(['cd',f'{output_dir}/{domain}/'])

    format_str = "{:3d} {:11.5f} {:11.5f} {:8.1f}" + \
             "{:5d} {:5d} {:5d} {:5d} {:5d} {:5d}" + \
             "{:11.5f} {:11.5f}" + \
             "{:11.1f} {:11.1f}" + \
             "{:5d} {:5d} {:5d} {:5d}"
    data=[]
    for i in range(nobs):
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
    with open(f"{output_dir}/obs_{domain}/kctest_{nobs}_obsinput_{day}_{hour}_{min}_{suffix}.txt", "w", encoding="utf-8") as file:
        for row in data:
            file.write(format_str.format(*row) + "\n")
