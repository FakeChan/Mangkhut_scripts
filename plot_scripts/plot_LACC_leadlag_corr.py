"""
Calculate lead-lag spatial correlation between WRF atmospheric variables at
previous times and the reference-time TSK inside a fixed TC-centered radius.

The end_time is used as the reference time:
    1. Find the TC center at end_time.
    2. Build one fixed radius mask around that center.
    3. Read TSK at end_time inside that mask.
    4. For each wrfout from start_time to end_time, compute
       corr[ATM(time), TSK(end_time)] over the masked grid points.

Lag convention:
    lag_hours > 0 means the atmospheric variable leads the reference TSK:
        corr[ATM(end_time - lag), TSK(end_time)]

Edit the CONFIG block for normal use, or override selected options from
the command line. The script writes a CSV table and a simple figure.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd


# =========================
# User configuration
# =========================
CONFIG = {
    # Directory containing files named like wrfout_d01_2018-09-09_00:00:00.
    "wrfout_dir": r"/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain",
    "domain": "d01",
    "start_time": "2018-09-09_00:00:00",
    "end_time": "2018-09-10_00:00:00",
    "time_interval_hours": 3,
    # Keep only previous times whose lag from end_time is inside this range.
    # Positive lag: atmospheric variable leads reference-time TSK.
    "min_lag_hours": 3,
    "max_lag_hours": 24,
    "lag_interval_hours": 3,
    # Radius around the TC center.
    "radius_km": 200.0,
    # Atmospheric variables. Supported derived names:
    #   P_FULL = P + PB, Pa
    #   TK     = true air temperature from WRF T/P/PB, K
    # Raw WRF variable names such as P, T, QVAPOR, U, V also work.
    "atm_vars": ["TK"],
    "ocean_var": "TSK",
    # Optional vertical levels for 3D variables, e.g. [0, 5, 10].
    # Use None to process all model levels.
    "levels": None,
    "output_dir": r"./figs/LACC",
}


def parse_wrf_time(time_text: str) -> datetime:
    return datetime.strptime(time_text, "%Y-%m-%d_%H:%M:%S")


def make_time_list(start: datetime, end: datetime, interval_hours: int) -> list[datetime]:
    times = []
    current = start
    step = timedelta(hours=interval_hours)
    while current <= end:
        times.append(current)
        current += step
    return times


def wrfout_path(wrfout_dir: str, domain: str, time_value: datetime) -> Path:
    name = f"wrfout_{domain}_{time_value.strftime('%Y-%m-%d_%H:%M:%S')}"
    return Path(wrfout_dir) / name


def as_2d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 3:
        return arr[0, :, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected a 2D or 3D field, got shape {arr.shape}")


def as_var_array(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return np.asarray(arr, dtype=float)


def haversine_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    radius_earth_km = 6371.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat0_rad) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius_earth_km * np.arcsin(np.sqrt(a))


def find_tc_center(ds: nc.Dataset) -> tuple[float, float, int, int, str, float]:
    lat = as_2d(ds.variables["XLAT"][:])
    lon = as_2d(ds.variables["XLONG"][:])

    if "SLP" in ds.variables:
        center_field = as_2d(ds.variables["SLP"][:])
        center_var = "SLP"
    elif "slp" in ds.variables:
        center_field = as_2d(ds.variables["slp"][:])
        center_var = "slp"
    elif "PSFC" in ds.variables:
        center_field = as_2d(ds.variables["PSFC"][:])
        center_var = "PSFC"
    else:
        raise KeyError("Cannot locate TC center: no SLP, slp, or PSFC in wrfout.")

    j_center, i_center = np.unravel_index(np.nanargmin(center_field), center_field.shape)
    return (
        float(lat[j_center, i_center]),
        float(lon[j_center, i_center]),
        int(j_center),
        int(i_center),
        center_var,
        float(center_field[j_center, i_center]),
    )


def read_variable(ds: nc.Dataset, varname: str) -> np.ndarray:
    upper_name = varname.upper()

    if upper_name == "P_FULL":
        return as_var_array(ds.variables["P"][:]) + as_var_array(ds.variables["PB"][:])

    if upper_name in {"TK", "TEMP", "TEMPERATURE"}:
        theta = as_var_array(ds.variables["T"][:]) + 300.0
        pressure = as_var_array(ds.variables["P"][:]) + as_var_array(ds.variables["PB"][:])
        return theta * (pressure / 100000.0) ** 0.286

    if varname not in ds.variables:
        raise KeyError(f"{varname} is not in this wrfout file.")
    return as_var_array(ds.variables[varname][:])


def area_mean_by_level(field: np.ndarray, mask: np.ndarray, levels: list[int] | None) -> dict[int | str, float]:
    arr = np.asarray(field, dtype=float)

    if arr.ndim == 2:
        return {"surface": float(np.nanmean(arr[mask]))}

    if arr.ndim != 3:
        raise ValueError(f"Only 2D/3D fields are supported, got shape {arr.shape}")

    nz = arr.shape[0]
    selected_levels = list(range(nz)) if levels is None else levels
    means = {}
    for level in selected_levels:
        if level < 0 or level >= nz:
            raise IndexError(f"Requested level {level}, but field has {nz} levels.")
        means[int(level)] = float(np.nanmean(arr[level, :, :][mask]))
    return means


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x_valid = x[valid]
    y_valid = y[valid]
    if np.nanstd(x_valid) == 0.0 or np.nanstd(y_valid) == 0.0:
        return np.nan
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def collect_reference_data(config: dict) -> tuple[datetime, np.ndarray, np.ndarray, dict]:
    reference_time = parse_wrf_time(config["end_time"])
    reference_path = wrfout_path(config["wrfout_dir"], config["domain"], reference_time)
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing reference wrfout file: {reference_path}")

    with nc.Dataset(reference_path) as ds:
        lat = as_2d(ds.variables["XLAT"][:])
        lon = as_2d(ds.variables["XLONG"][:])
        center_lat, center_lon, j_center, i_center, center_var, center_value = find_tc_center(ds)
        distance = haversine_km(lat, lon, center_lat, center_lon)
        mask = distance <= float(config["radius_km"])
        if not np.any(mask):
            raise RuntimeError(f"No grid points inside {config['radius_km']} km at {reference_time}.")

        reference_tsk = read_variable(ds, config["ocean_var"])
        if np.asarray(reference_tsk).ndim != 2:
            raise ValueError(f"Reference ocean_var must be 2D, got shape {np.asarray(reference_tsk).shape}.")
        reference_values = np.asarray(reference_tsk, dtype=float)[mask]

    center_info = {
        "reference_time": reference_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "j_center": j_center,
        "i_center": i_center,
        "center_var": center_var,
        "center_value": center_value,
        "n_grid_points": int(mask.sum()),
        "note": "TC center and radius mask are fixed from end_time.",
    }
    return reference_time, mask, reference_values, center_info


def spatial_corr_by_level(field: np.ndarray, mask: np.ndarray, reference_values: np.ndarray, levels: list[int] | None) -> dict[int | str, tuple[float, int]]:
    arr = np.asarray(field, dtype=float)

    if arr.ndim == 2:
        values = arr[mask]
        return {"surface": (pearsonr(values, reference_values), int(np.isfinite(values).sum()))}

    if arr.ndim != 3:
        raise ValueError(f"Only 2D/3D fields are supported, got shape {arr.shape}")

    nz = arr.shape[0]
    selected_levels = list(range(nz)) if levels is None else levels
    correlations = {}
    for level in selected_levels:
        if level < 0 or level >= nz:
            raise IndexError(f"Requested level {level}, but field has {nz} levels.")
        values = arr[level, :, :][mask]
        correlations[int(level)] = (pearsonr(values, reference_values), int(np.isfinite(values).sum()))
    return correlations


def calculate_reference_correlations(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = parse_wrf_time(config["start_time"])
    end = parse_wrf_time(config["end_time"])
    times = make_time_list(start, end, int(config["time_interval_hours"]))
    levels = config["levels"]
    if levels is not None:
        levels = [int(level) for level in levels]

    reference_time, reference_mask, reference_values, center_info = collect_reference_data(config)
    min_lag_hours = int(config["min_lag_hours"])
    max_lag_hours = int(config["max_lag_hours"])
    lag_interval_hours = int(config["lag_interval_hours"])
    rows = []

    for time_value in times:
        path = wrfout_path(config["wrfout_dir"], config["domain"], time_value)
        if not path.exists():
            raise FileNotFoundError(f"Missing wrfout file: {path}")

        with nc.Dataset(path) as ds:
            for varname in config["atm_vars"]:
                lag_hours = int((reference_time - time_value).total_seconds() / 3600)
                if lag_hours < min_lag_hours or lag_hours > max_lag_hours:
                    continue
                if lag_interval_hours > 0 and lag_hours % lag_interval_hours != 0:
                    continue
                field = read_variable(ds, varname)
                if np.asarray(field).shape[-2:] != reference_mask.shape:
                    raise ValueError(
                        f"{path} {varname} horizontal shape {np.asarray(field).shape[-2:]} "
                        f"does not match reference mask shape {reference_mask.shape}."
                    )
                corr_by_level = spatial_corr_by_level(field, reference_mask, reference_values, levels)
                for level, (corr, n_pairs) in corr_by_level.items():
                    rows.append(
                        {
                            "atm_var": varname,
                            "ocean_var": config["ocean_var"],
                            "level": level,
                            "atm_time": time_value.strftime("%Y-%m-%d_%H:%M:%S"),
                            "reference_time": reference_time.strftime("%Y-%m-%d_%H:%M:%S"),
                            "lag_hours": lag_hours,
                            "lead_label": f"t-{lag_hours}" if lag_hours > 0 else "t",
                            "corr": corr,
                            "n_pairs": n_pairs,
                            "note": "positive lag means ATM(time) leads TSK(reference_time)",
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame([center_info])


def calculate_correlations(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    return calculate_reference_correlations(config)


def plot_correlations(corr_df: pd.DataFrame, output_png: Path) -> None:
    groups = list(corr_df.groupby(["atm_var", "level"], sort=False))
    if not groups:
        return

    ncols = min(3, len(groups))
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.4 * nrows), squeeze=False)

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, ((varname, level), group) in zip(axes.ravel(), groups):
        ax.set_visible(True)
        group = group.sort_values("lag_hours", ascending=False).reset_index(drop=True)
        xloc = np.arange(len(group))
        ax.plot(xloc, group["corr"], marker="o", linewidth=1.6)
        ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        ax.set_title(f"{varname}, level={level}")
        ax.set_xlabel("Lead time relative to reference TSK")
        ax.set_ylabel("Correlation")
        ax.set_xticks(xloc)
        ax.set_xticklabels(group["lead_label"], rotation=45, ha="right")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrfout-dir", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--time-interval-hours", type=int, default=None)
    parser.add_argument("--radius-km", type=float, default=None)
    parser.add_argument("--atm-vars", nargs="+", default=None)
    parser.add_argument("--ocean-var", default=None)
    parser.add_argument("--levels", nargs="+", type=int, default=None)
    parser.add_argument("--min-lag-hours", type=int, default=None)
    parser.add_argument("--max-lag-hours", type=int, default=None)
    parser.add_argument("--lag-interval-hours", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict:
    config = CONFIG.copy()
    arg_map = {
        "wrfout_dir": args.wrfout_dir,
        "domain": args.domain,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "time_interval_hours": args.time_interval_hours,
        "radius_km": args.radius_km,
        "atm_vars": args.atm_vars,
        "ocean_var": args.ocean_var,
        "levels": args.levels,
        "min_lag_hours": args.min_lag_hours,
        "max_lag_hours": args.max_lag_hours,
        "lag_interval_hours": args.lag_interval_hours,
        "output_dir": args.output_dir,
    }
    for key, value in arg_map.items():
        if value is not None:
            config[key] = value
    return config


def main() -> None:
    config = merge_config(parse_args())
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    corr_df, center_df = calculate_correlations(config)

    tag = (
        f"{config['domain']}_"
        f"{config['start_time'].replace(':', '').replace('-', '')}_"
        f"{config['end_time'].replace(':', '').replace('-', '')}_"
        f"r{int(config['radius_km'])}km"
    )
    corr_csv = output_dir / f"lead_lag_corr_{tag}.csv"
    center_csv = output_dir / f"tc_center_{tag}.csv"
    fig_png = output_dir / f"lead_lag_corr_{tag}.png"

    corr_df.to_csv(corr_csv, index=False)
    center_df.to_csv(center_csv, index=False)
    plot_correlations(corr_df, fig_png)

    print(f"Wrote correlation table: {corr_csv}")
    print(f"Wrote TC center table: {center_csv}")
    print(f"Wrote figure: {fig_png}")
    print("Lag convention: positive lag_hours means ATM(time) leads TSK(end_time).")


if __name__ == "__main__":
    main()
