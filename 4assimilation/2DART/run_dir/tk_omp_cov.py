"""
Compute ensemble covariance between OM_TMP level 0 and lowest-level air
temperature from WRF ensemble members.

For each horizontal grid point:

    cov(OM_TMP(member, ocean_level=0), T_air(member, bottom_level=0))

where T_air can be either true temperature TK derived from WRF perturbation
potential temperature T plus P/PB, or raw THM/T if requested.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import netCDF4 as nc
except ImportError as exc:
    raise ImportError("This script requires netCDF4. Install with: conda install -c conda-forge netcdf4") from exc


# =========================
# User configuration
# =========================
CONFIG = {
    "member_base_dir": r"/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst",
    "domain": "d01",
    "wrf_time": "2018-09-10_00:00:00",
    "member_start": 1,
    "member_end": 50,
    "omtmp_var": "OM_TMP",
    "omtmp_level": 0,
    # Use "TK" for true air temperature derived from T/THM and P/PB.
    # Use "T" or "THM" to compute true temperature from those perturbation
    # potential temperature variables if available.
    "air_temp_var": "TK",
    "air_level": 0,
    "lat_var": "XLAT",
    "lon_var": "XLONG",
    "output_npz": r"./diag/cov_omtmp0_tbottom.npz",
    "output_csv": r"./diag/cov_omtmp0_tbottom_summary.csv",
    "output_png": r"./diag/cov_omtmp0_tbottom.png",
}


def wrfout_path(base_dir: Path, member: int, domain: str, wrf_time: str) -> Path:
    return base_dir / f"mem{member:03d}" / f"wrfout_{domain}_{wrf_time}"


def squeeze_time(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def read_true_air_temperature(ds: nc.Dataset, level: int, air_temp_var: str) -> np.ndarray:
    var_upper = air_temp_var.upper()
    if var_upper in {"TK", "TEMP", "TEMPERATURE"}:
        theta_name = "T" if "T" in ds.variables else "THM"
    elif var_upper in {"T", "THM"}:
        theta_name = var_upper
    else:
        raise ValueError("air_temp_var must be TK, T, THM, TEMP, or TEMPERATURE.")

    if theta_name not in ds.variables:
        raise KeyError(f"{theta_name} is not in wrfout.")
    if "P" not in ds.variables or "PB" not in ds.variables:
        raise KeyError("P and PB are required to compute true air temperature.")

    theta_pert = squeeze_time(np.asarray(ds.variables[theta_name][:], dtype=float))[level, :, :]
    pressure = (
        squeeze_time(np.asarray(ds.variables["P"][:], dtype=float))[level, :, :]
        + squeeze_time(np.asarray(ds.variables["PB"][:], dtype=float))[level, :, :]
    )
    return (theta_pert + 300.0) * (pressure / 100000.0) ** 0.286


def read_member_fields(path: Path, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with nc.Dataset(path) as ds:
        omtmp = squeeze_time(np.asarray(ds.variables[config["omtmp_var"]][:], dtype=float))[
            int(config["omtmp_level"]), :, :
        ]
        tair = read_true_air_temperature(ds, int(config["air_level"]), str(config["air_temp_var"]))
        lat = squeeze_time(np.asarray(ds.variables[config["lat_var"]][:], dtype=float))
        lon = squeeze_time(np.asarray(ds.variables[config["lon_var"]][:], dtype=float))
    return omtmp, tair, lat, lon


def calculate(config: dict) -> dict[str, np.ndarray]:
    base_dir = Path(config["member_base_dir"])
    members = list(range(int(config["member_start"]), int(config["member_end"]) + 1))

    om_list = []
    t_list = []
    lat = lon = None
    for member in members:
        path = wrfout_path(base_dir, member, config["domain"], config["wrf_time"])
        if not path.exists():
            raise FileNotFoundError(path)
        omtmp, tair, this_lat, this_lon = read_member_fields(path, config)
        om_list.append(omtmp)
        t_list.append(tair)
        if lat is None:
            lat = this_lat
            lon = this_lon

    om = np.stack(om_list, axis=0)
    tair = np.stack(t_list, axis=0)

    om_anom = om - np.mean(om, axis=0, keepdims=True)
    t_anom = tair - np.mean(tair, axis=0, keepdims=True)
    cov = np.sum(om_anom * t_anom, axis=0) / (len(members) - 1)
    corr = cov / (np.std(om, axis=0, ddof=1) * np.std(tair, axis=0, ddof=1))

    return {
        "lat": lat,
        "lon": lon,
        "cov": cov,
        "corr": corr,
        "om_mean": np.mean(om, axis=0),
        "tair_mean": np.mean(tair, axis=0),
        "om_sd": np.std(om, axis=0, ddof=1),
        "tair_sd": np.std(tair, axis=0, ddof=1),
    }


def write_outputs(result: dict[str, np.ndarray], config: dict) -> None:
    output_npz = Path(config["output_npz"])
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **result)
    print(f"Wrote {output_npz}")

    cov = result["cov"]
    corr = result["corr"]
    summary = pd.DataFrame(
        [
            {
                "n_grid": cov.size,
                "cov_min": float(np.nanmin(cov)),
                "cov_mean": float(np.nanmean(cov)),
                "cov_median": float(np.nanmedian(cov)),
                "cov_max": float(np.nanmax(cov)),
                "cov_frac_positive": float(np.nanmean(cov > 0.0)),
                "cov_frac_negative": float(np.nanmean(cov < 0.0)),
                "corr_min": float(np.nanmin(corr)),
                "corr_mean": float(np.nanmean(corr)),
                "corr_median": float(np.nanmedian(corr)),
                "corr_max": float(np.nanmax(corr)),
            }
        ]
    )
    output_csv = Path(config["output_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    print(summary.to_string(index=False))


def plot_covariance(result: dict[str, np.ndarray], config: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    lat = result["lat"]
    lon = result["lon"]
    cov = result["cov"]
    finite = cov[np.isfinite(cov)]
    limit = np.nanpercentile(np.abs(finite), 98.0)
    if not np.isfinite(limit) or limit == 0.0:
        limit = np.nanmax(np.abs(finite))

    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=160)
    levels = np.linspace(-limit, limit, 25)
    cf = ax.contourf(lon, lat, cov, levels=levels, cmap="RdBu_r", extend="both")
    if np.nanmin(cov) <= 0.0 <= np.nanmax(cov):
        ax.contour(lon, lat, cov, levels=[0.0], colors="k", linewidths=0.8)
    ax.set_title(
        f"cov({config['omtmp_var']} level {config['omtmp_level']}, "
        f"{config['air_temp_var']} level {config['air_level']})"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.colorbar(cf, ax=ax, label="Covariance K^2")
    fig.tight_layout()

    output_png = Path(config["output_png"])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-base-dir", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--wrf-time", default=None)
    parser.add_argument("--member-start", type=int, default=None)
    parser.add_argument("--member-end", type=int, default=None)
    parser.add_argument("--omtmp-level", type=int, default=None)
    parser.add_argument("--air-level", type=int, default=None)
    parser.add_argument("--air-temp-var", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-png", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict:
    config = CONFIG.copy()
    for key, value in {
        "member_base_dir": args.member_base_dir,
        "domain": args.domain,
        "wrf_time": args.wrf_time,
        "member_start": args.member_start,
        "member_end": args.member_end,
        "omtmp_level": args.omtmp_level,
        "air_level": args.air_level,
        "air_temp_var": args.air_temp_var,
        "output_npz": args.output_npz,
        "output_csv": args.output_csv,
        "output_png": args.output_png,
    }.items():
        if value is not None:
            config[key] = value
    if args.no_plot:
        config["output_png"] = None
    return config


def main() -> None:
    config = merge_config(parse_args())
    result = calculate(config)
    write_outputs(result, config)
    if config["output_png"]:
        plot_covariance(result, config)


if __name__ == "__main__":
    main()
