'''
This script merge ensemble of FO data(usually in ens_size number of file)
to a single file, for better fortran reading.

one line contains {ens_size} number of FO data.

written by Haoxing in 2023-05-12/NJU-Xianlin Campus
'''
import os
import numpy as np
import subprocess

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

if __name__ == "__main__":

    time_from_env = os.environ.get(
        "current_time",
        "10_00_00"
    )

    time_list = [time_from_env]
    domain = os.environ.get("domain", "d01")
    sensor = os.environ.get("sensor", "AMSUA")
    ch = int(os.environ.get("assim_channel", "4"))
    ens_size = int(os.environ.get("ENS_SIZE", "50"))
    nobs = int(os.environ.get("NOBS", "676"))
    memlist = list(np.arange(1, ens_size + 1))

    TOP_DIR = "/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs"
    BT_DIR = os.environ.get(
        "ENS_BT_DIR",
        "/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/4ens_BT"
    )

    for time in time_list:
        output = []
        dir_name = 'ensBT' + domain + '_' + time
        OUT_DIR = f'{TOP_DIR}/mergeFO{domain}_{time}'
        subprocess.run(['mkdir', '-p', f'{OUT_DIR}'])
        work_dir = f'{TOP_DIR}/{dir_name}/prior_BT/'
        subprocess.run(['mkdir', '-p', f'{work_dir}'])

        ens_fname_prefix = f'ens_ch{ch}_mem'
        out_fname_default = f'{OUT_DIR}/merged_FO_data.txt'
        print(work_dir)

        bt_subdir = os.environ.get(
            "ENS_BT_SUBDIR",
            f"BT_{time}"
        )
        mask_file = os.environ.get(
            "CLEAR_SKY_MASK_FILE",
            ""
        )
        use_clear_sky_mask = (
            os.environ.get("USE_CLEAR_SKY_MASK", "1") == "1"
        )
        out_fname = os.environ.get("MERGED_FO_FILE", out_fname_default)

        clear_sky_indices = load_clear_sky_indices(
            mask_file, nobs, use_clear_sky_mask
        )

        os.chdir(work_dir)
        for i, mem in enumerate(memlist):
            formatted_i = '{:03d}'.format(mem)
            in_fname = f'{ens_fname_prefix}{formatted_i}'
            subprocess.run(['ln', '-sf',
                            f'{BT_DIR}/mem{formatted_i}/{sensor}/{bt_subdir}/obs_{domain}_ch{ch}_totalline.txt',
                            f'{work_dir}/{in_fname}'])
            if not os.path.exists(in_fname):
                raise FileNotFoundError(
                    f"member Hx file missing (not silently skipped): {work_dir}/{in_fname}"
                )
            arr = np.loadtxt(in_fname, dtype='str')
            if arr.size != nobs:
                raise ValueError(
                    f"member Hx raw row count {arr.size} does not match nobs "
                    f"{nobs}: {in_fname}"
                )
            arr = arr[clear_sky_indices]
            if arr.size != len(clear_sky_indices):
                raise ValueError(
                    f"member Hx filtered row count {arr.size} does not match "
                    f"expected {len(clear_sky_indices)}: {in_fname}"
                )
            output.append(arr)

        if len(output) != ens_size:
            raise ValueError(
                f"merged ensemble member count {len(output)} does not match "
                f"ens_size {ens_size}"
            )

        tmp = np.array(output).transpose()  # make sure structure is [data_num,ens_size]
        if tmp.shape != (len(clear_sky_indices), ens_size):
            raise ValueError(
                f"merged matrix shape {tmp.shape} does not match expected "
                f"{(len(clear_sky_indices), ens_size)}"
            )
        np.savetxt(out_fname, tmp, delimiter=' ', fmt='%s')

        print(f"Clear-sky mask file: {mask_file}")
        print(f"Raw grid point count: {nobs}")
        print(f"Retained grid point count: {len(clear_sky_indices)}")
        print(f"Ensemble member count: {ens_size}")
        print(f"Merged matrix shape: {tmp.shape}")
        print(f"Merged Hx file: {out_fname}")
