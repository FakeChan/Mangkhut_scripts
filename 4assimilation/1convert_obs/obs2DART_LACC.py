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

def load_clear_sky_indices(mask_file, nobs, required):
    if not mask_file or not os.path.exists(mask_file):
        if required:
            raise FileNotFoundError(
                f"Clear-sky mask is required but does not exist: {mask_file}"
            )
        return np.arange(nobs)

    raw_mask = np.loadtxt(mask_file).reshape(-1)

    if raw_mask.size != nobs:
        raise ValueError(
            f"clear-sky mask size {raw_mask.size} does not match nobs "
            f"{nobs}: {mask_file}"
        )

    if not np.all(np.isin(raw_mask, [0, 1])):
        raise ValueError(
            f"clear-sky mask must contain only 0 and 1: {mask_file}"
        )

    mask = raw_mask.astype(bool)
    indices = np.where(mask)[0]

    print(
        f"Using clear-sky mask {mask_file}: "
        f"keep {indices.size} / {nobs}"
    )
    return indices

def read_lacc_time_count(times_file):
    if not os.path.exists(times_file):
        raise FileNotFoundError(
            f"LACC times file does not exist: {times_file}"
        )

    with open(times_file, "r", encoding="utf-8") as file:
        count = sum(
            1 for line in file
            if line.strip().startswith("lag_time=")
        )

    if count <= 0:
        raise ValueError(
            f"No lag_time entries found in LACC times file: {times_file}"
        )

    expected = os.environ.get("EXPECTED_LACC_COUNT")
    if expected is not None and count != int(expected):
        raise ValueError(
            f"LACC time count mismatch: file contains {count}, "
            f"but EXPECTED_LACC_COUNT={expected}"
        )

    return count

if __name__ == "__main__":
    #======================================================================================
    #basic model configure (from environment variables)
    domain = os.environ.get("domain", "d01")
    nobs = int(os.environ.get("NOBS", "676"))
    use_quantile = True

    day = os.environ.get("current_day", "10")
    hour = os.environ.get("current_hour", "00")
    minute = os.environ.get("current_min", "00")

    channel = int(os.environ.get("assim_channel", "4"))

    obs_err_single = float(
        os.environ.get(
            "OBS_ERR_STD",
            os.environ.get("obs_err_std", "0.5")
        )
    )
    hgt_obs = int(os.environ.get("HGT_OBS", "100000"))
    #======================================================================================
    #path
    obs_dir = os.environ.get(
        "OBS_BT_DIR",
        "/share/home/lililei1/kcfu/tc_mangkhut/"
        "3create_obs/hx_rttov/3obs_BT_LACC/AMSUA"
    )
    prof_dir = os.environ.get(
        "PROFILE_DIR",
        "/share/home/lililei1/kcfu/tc_mangkhut/"
        "3create_obs/hx_rttov/profile"
    )
    output_dir = os.environ.get(
        "CONVERT_OUTPUT_DIR",
        "/share/home/lililei1/kcfu/tc_mangkhut/"
        "4assimilation/1convert_obs"
    )
    obs_subdir = os.environ.get(
        "OBS_BT_SUBDIR",
        f"BT_LACC_{day}_{hour}_{minute}"
    )
    obs_file = os.environ.get(
        "OBS_INPUT_FILE",
        f"{obs_dir}/{obs_subdir}/"
        f"obs_{domain}_ch{channel}_totalline_withpert.txt"
    )
    clear_sky_mask_file = os.environ.get(
        "CLEAR_SKY_MASK_FILE",
        f"{obs_dir}/{obs_subdir}/clear_sky_mask.txt"
    )
    profile_subdir = os.environ.get(
        "PROFILE_SUBDIR",
        f"profile_{domain}_LACC_{day}_{hour}_{minute}"
    )
    para_file = os.environ.get(
        "PARA_FILE",
        f"{prof_dir}/{profile_subdir}/"
        f"prof{day}_{hour}:{minute}.dat"
    )
    lacc_times_file = os.environ.get(
        "LACC_TIMES_FILE",
        f"{obs_dir}/{obs_subdir}/LACC_times.txt"
    )
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
    intmin=int(minute)
    if use_quantile:
        obstype=170  #when using DART_quantile
        suffix='quantile'
    else:
        obstype=170 #NOAA_18_AMSUA_TB =   170 when using DART_main
        suffix='main'
    year=2018
    month=9
    second=0

    # clear-sky mask switch
    use_clear_sky_mask = (
        os.environ.get("USE_CLEAR_SKY_MASK", "1") == "1"
    )

    clear_sky_indices = load_clear_sky_indices(
        clear_sky_mask_file,
        nobs,
        use_clear_sky_mask,
    )

    # LACC observation error: single-time sigma / sqrt(number of LACC times)
    n_lacc_times = read_lacc_time_count(lacc_times_file)
    obs_err = obs_err_single / math.sqrt(n_lacc_times)

    print(f"Single-time obs error std: {obs_err_single:.8f} K")
    print(f"Number of LACC times: {n_lacc_times}")
    print(f"LACC obs error std: {obs_err:.8f} K")

    obs=np.loadtxt(obs_file)
    angles=read_every_nth_line(para_file, start_line=188, step=185)
    locations=read_every_nth_line(para_file, start_line=186, step=185)

    if len(obs) != nobs:
        raise ValueError(
            f"observation count {len(obs)} does not match nobs {nobs}: {obs_file}"
        )
    if len(angles) != nobs:
        raise ValueError(
            f"angle count {len(angles)} does not match nobs {nobs}: {para_file}"
        )
    if len(locations) != nobs:
        raise ValueError(
            f"location count {len(locations)} does not match nobs {nobs}: {para_file}"
        )

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
        lat=float(locations[i].split()[1])
        lon=float(locations[i].split()[2])
        data.append((obstype,lat,lon,hgt_obs,year,month,intday,inthour,intmin,second,obs_value,obs_err,sat_az,sat_ze,platform,sat,sensor,channel))
    #sequnce: obstype(int),lat,lon,height of obs(hPa),year,month,day,hour,minute,second,
    #obs_value,obs_error,sat_az,sat_ze,platform_id, sat_id, sensor_id, channel

    obs_dart_input_file = os.environ.get("OBS_DART_INPUT_FILE")
    if obs_dart_input_file:
        out_path = obs_dart_input_file
    else:
        nobs_out = len(data)
        out_path = f"{output_dir}/obs_{domain}/kctest_{nobs_out}_obsinput_{day}_{hour}_{minute}_LACC.txt"

    with open(out_path, "w", encoding="utf-8") as file:
        for row in data:
            file.write(format_str.format(*row) + "\n")

    nobs_out = len(data)
    print(f"Filtered observation count: {nobs_out}")
    print(f"DART text observation input: {out_path}")
