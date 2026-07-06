from pathlib import Path

import netCDF4 as nc
import numpy as np


# =====================
# User configuration
# =====================
work_dir = Path(".")
domains = ["d01", "d02"]
member_count = 50
filename_template = "firstguess_{domain}.mem{member:03d}"


def _read_2d_first_time(var):
    if var.ndim == 2:
        return var[:, :]
    if var.ndim == 3:
        return var[0, :, :]
    raise ValueError(f"{var.name} must be 2D or 3D with one Time level, got {var.shape}")


def _read_om_tmp_surface(var):
    if var.ndim == 3:
        return var[0, :, :]
    if var.ndim == 4:
        return var[0, 0, :, :]
    raise ValueError(f"{var.name} must be 3D or 4D with one Time level, got {var.shape}")


def _write_tsk_first_time(var, values):
    if var.ndim == 2:
        var[:, :] = values
    elif var.ndim == 3:
        var[0, :, :] = values
    else:
        raise ValueError(f"{var.name} must be 2D or 3D with one Time level, got {var.shape}")


def update_one_file(path):
    with nc.Dataset(path, "r+") as ds:
        for name in ["TSK", "XLAND", "OM_TMP"]:
            if name not in ds.variables:
                raise KeyError(f"{path}: missing variable {name}")

        tsk_var = ds.variables["TSK"]
        xland = _read_2d_first_time(ds.variables["XLAND"])
        om_tmp_surface = _read_om_tmp_surface(ds.variables["OM_TMP"])
        tsk = _read_2d_first_time(tsk_var)

        if tsk.shape != xland.shape or tsk.shape != om_tmp_surface.shape:
            raise ValueError(
                f"{path}: shape mismatch: "
                f"TSK{tsk.shape}, XLAND{xland.shape}, OM_TMP surface{om_tmp_surface.shape}"
            )

        ocean_mask = xland == 2
        updated_tsk = np.where(ocean_mask, om_tmp_surface, tsk)
        _write_tsk_first_time(tsk_var, updated_tsk)

    return int(np.count_nonzero(ocean_mask))


def main():
    total_files = 0
    total_points = 0

    for domain in domains:
        for member in range(1, member_count + 1):
            path = work_dir / filename_template.format(domain=domain, member=member)
            if not path.exists():
                raise FileNotFoundError(path)

            changed = update_one_file(path)
            total_files += 1
            total_points += changed
            print(f"updated {path}: {changed} ocean grid points")

    print(f"done: updated {total_files} files, {total_points} ocean grid-point values")


if __name__ == "__main__":
    main()
