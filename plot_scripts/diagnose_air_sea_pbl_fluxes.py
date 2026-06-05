#!/usr/bin/env python3
"""
Diagnose air-sea boundary-layer exchange from WRF sensitivity experiments.

This script compares three ocean-coupling experiment settings and two
assimilation methods:

    EXP1: no ocean coupling
    EXP2: ocean coupling without ocean update
    EXP3: ocean coupling with ocean update

For each EXP/assimilation pair it computes ocean-only means, strong-wind means,
cumulative fluxes, experiment differences, assimilation-method differences,
time-series figures, and selected spatial difference maps.
"""

import glob
import os
import re
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# User settings
# =============================================================================

base_dirs = {
    "EXP1": "/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim0Run0",
    "EXP2": "/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim0Run1",
    "EXP3": "/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim1Run1",
}

assim_methods = ["EAKF", "QCF_RHF"]
members = ["006", "015", "029", "037", "043", "044"]

# Change to "d02" if your files are named wrfout_d02_*.
domain = "d02"

# Nature-run files are used only to locate the typhoon center, then that center
# is mapped to the ensemble-member grid. If nr_base is a file, the same file is
# used for all times; if it is a directory, wrfout_{nr_domain}_YYYY-mm-dd_HH:MM:SS
# is searched recursively.
nr_base = "/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/"
nr_domain = "d03"
tc_radius_km = 150.0

output_dir = "./figs/pbl_flux_anal/"

# Physical constants and output interval.
Lv = 2.5e6  # J kg-1
rho = 1.2  # kg m-3
dt = 1800.0  # s, 30 min

strict_time_alignment = True
strong_wind_thresholds = [10.0, 15.0]
skip_initial_flux_time_in_plots = True

required_vars = ["UST", "HFX", "QFX", "LANDMASK", "U10", "V10", "TSK", "T2", "PBLH"]


# =============================================================================
# Basic helpers
# =============================================================================

def ensure_output_dirs():
    subdirs = [
        output_dir,
        os.path.join(output_dir, "csv"),
        os.path.join(output_dir, "figures"),
        os.path.join(output_dir, "figures", "timeseries"),
        os.path.join(output_dir, "figures", "spatial"),
    ]
    for path in subdirs:
        os.makedirs(path, exist_ok=True)


def warn(message):
    warnings.warn(message)
    print(f"WARNING: {message}")


def member_file_pattern(exp_name, assim_method, member):
    return os.path.join(base_dirs[exp_name], assim_method, member, f"wrfout_{domain}_*")


def sorted_member_wrf_files(exp_name, assim_method, member):
    pattern = member_file_pattern(exp_name, assim_method, member)
    files = sorted(glob.glob(pattern))
    if not files:
        recursive_pattern = os.path.join(base_dirs[exp_name], assim_method, member, "**", f"wrfout_{domain}_*")
        files = sorted(glob.glob(recursive_pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No wrfout files found: {pattern}")
    return files


def parse_time_from_filename(path):
    name = os.path.basename(path)
    match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}", name)
    if not match:
        return None
    return pd.to_datetime(match.group(0).replace("_", " "))


def build_member_file_maps(exp_name, assim_method):
    maps = {}
    for member in members:
        member_map = {}
        for path in sorted_member_wrf_files(exp_name, assim_method, member):
            file_time = parse_time_from_filename(path)
            if file_time is None:
                with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
                    file_time = parse_wrf_time(ds, path)
            if file_time in member_map:
                warn(f"Duplicate time {file_time} for {exp_name}/{assim_method}/{member}; using {path}")
            member_map[file_time] = path
        maps[member] = member_map

    common_times = None
    for member, member_map in maps.items():
        times = set(member_map)
        common_times = times if common_times is None else common_times & times
        if not times:
            raise FileNotFoundError(f"No usable wrfout times found for {exp_name}/{assim_method}/{member}")

    all_times = sorted(common_times)
    if not all_times:
        raise ValueError(f"No common wrfout times across members for {exp_name}/{assim_method}")

    for member, member_map in maps.items():
        missing = sorted(set().union(*[set(m) for m in maps.values()]) - set(member_map))
        if missing:
            warn(f"{exp_name}/{assim_method}/{member} is missing {len(missing)} times; ensemble mean uses common times only.")

    return maps, all_times


def parse_wrf_time(ds, file_path):
    """Read WRF Times variable, falling back to XTIME or filename if needed."""
    if "Times" in ds:
        raw = ds["Times"].values
        if raw.ndim == 2:
            text = b"".join(raw[0].astype("S1")).decode("utf-8").strip()
        else:
            item = raw[0] if raw.ndim > 0 else raw
            text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
        return pd.to_datetime(text.replace("_", " "))

    if "XTIME" in ds:
        units = ds["XTIME"].attrs.get("units", "")
        try:
            decoded = xr.decode_cf(ds[["XTIME"]])["XTIME"].values
            return pd.to_datetime(np.asarray(decoded).ravel()[0])
        except Exception:
            warn(f"Could not decode XTIME units '{units}' in {file_path}; using filename time.")

    name = os.path.basename(file_path)
    match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}", name)
    if match:
        try:
            return pd.to_datetime(match.group(0).replace("_", " "))
        except Exception:
            pass
    raise ValueError(f"Could not read time from Times/XTIME or filename: {file_path}")


def first_2d_array(ds, var_name, file_path, required=False):
    """Return a variable as a 2-D numpy array, taking the first time/level if present."""
    if var_name not in ds.variables:
        if required:
            raise KeyError(f"Required variable {var_name} not found in {file_path}")
        warn(f"Variable {var_name} not found in {file_path}; filling related diagnostics with NaN.")
        return None

    da = ds[var_name]
    indexers = {}
    for dim in da.dims:
        if dim in ["Time", "time", "DateStrLen"]:
            if dim != "DateStrLen":
                indexers[dim] = 0
        elif dim in ["bottom_top", "bottom_top_stag", "soil_layers_stag", "ocean_layer_stag"]:
            indexers[dim] = 0
    if indexers:
        da = da.isel(indexers)

    arr = np.asarray(da.squeeze().values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Variable {var_name} in {file_path} is not 2-D after slicing; shape={arr.shape}")
    return arr


def read_lat_lon(ds, file_path):
    if "XLAT" not in ds.variables or "XLONG" not in ds.variables:
        raise KeyError(f"XLAT/XLONG not found in {file_path}")
    lats = first_2d_array(ds, "XLAT", file_path, required=True)
    lons = first_2d_array(ds, "XLONG", file_path, required=True)
    return lats, lons


def find_nr_file(time):
    time_name = pd.to_datetime(time).strftime("%Y-%m-%d_%H:%M:%S")
    if os.path.isfile(nr_base):
        return nr_base
    pattern = os.path.join(nr_base, "**", f"wrfout_{nr_domain}_{time_name}*")
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No NR file matching wrfout_{nr_domain}_{time_name}* under {nr_base}")
    return matches[0]


def pressure_like_field(ds, file_path):
    for name in ["SLP", "slp", "PSFC", "PMSL"]:
        if name in ds.variables:
            return first_2d_array(ds, name, file_path, required=True), name

    try:
        from wrf import getvar, to_np
        import netCDF4
    except Exception as exc:
        raise KeyError(
            f"No SLP/slp/PSFC/PMSL variable found in {file_path}, and wrf-python is not available "
            "to diagnose sea-level pressure."
        ) from exc

    with netCDF4.Dataset(file_path) as nc:
        return np.asarray(to_np(getvar(nc, "slp", timeidx=0)), dtype=float), "wrf-python slp"


def nr_typhoon_center_lat_lon(time):
    nr_file = find_nr_file(time)
    with xr.open_dataset(nr_file, decode_times=False, mask_and_scale=True) as ds:
        pressure, pressure_name = pressure_like_field(ds, nr_file)
        lats, lons = read_lat_lon(ds, nr_file)

    if pressure.shape != lats.shape:
        raise ValueError(f"NR pressure shape {pressure.shape} does not match XLAT shape {lats.shape}: {nr_file}")

    min_idx = np.unravel_index(np.nanargmin(pressure), pressure.shape)
    center_lat = float(lats[min_idx])
    center_lon = float(lons[min_idx])
    min_pressure = float(pressure[min_idx])
    print(
        f"NR TC center at {pd.to_datetime(time)} from {os.path.basename(nr_file)} "
        f"using {pressure_name}: lat={center_lat:.4f}, lon={center_lon:.4f}, pmin={min_pressure:.2f}"
    )
    return center_lat, center_lon, nr_file


def latlon_distance_km(lats, lons, center_lat, center_lon):
    radius_earth_km = 6371.0
    lat1 = np.deg2rad(lats)
    lon1 = np.deg2rad(lons)
    lat2 = np.deg2rad(center_lat)
    lon2 = np.deg2rad(center_lon)
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_earth_km * np.arcsin(np.sqrt(a))


def tc_region_mask_from_nr(time, member_lats, member_lons):
    center_lat, center_lon, nr_file = nr_typhoon_center_lat_lon(time)
    dist_sq = (member_lats - center_lat) ** 2 + (member_lons - center_lon) ** 2
    center_j, center_i = np.unravel_index(np.nanargmin(dist_sq), dist_sq.shape)
    distances = latlon_distance_km(member_lats, member_lons, center_lat, center_lon)
    mask = distances <= tc_radius_km
    if not np.any(mask):
        raise ValueError(
            f"No member grid points found within tc_radius_km={tc_radius_km:g} of NR TC center "
            f"for {time}; NR={nr_file}"
        )
    print(
        f"Mapped NR TC center to member grid for {pd.to_datetime(time)}: "
        f"i={center_i}, j={center_j}, radius={tc_radius_km:g} km, points={int(mask.sum())}"
    )
    return mask


def ensemble_nanmean(arrays):
    valid_arrays = [arr for arr in arrays if arr is not None]
    if not valid_arrays:
        return None
    return np.nanmean(np.stack(valid_arrays, axis=0), axis=0)


def nanmean_masked(arr, mask):
    if arr is None:
        return np.nan
    selected = np.where(mask, arr, np.nan)
    if np.all(np.isnan(selected)):
        return np.nan
    return float(np.nanmean(selected))


def symmetric_levels(arr, n=21):
    finite = np.asarray(arr)[np.isfinite(arr)]
    if finite.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanmax(np.abs(finite)))
        if vmax == 0:
            vmax = 1.0
    return np.linspace(-vmax, vmax, n)


def safe_name_time(t):
    return pd.to_datetime(t).strftime("%Y%m%d_%H%M%S")


# =============================================================================
# Reading and diagnostics
# =============================================================================

def read_one_experiment(exp_name, assim_method):
    member_maps, times = build_member_file_maps(exp_name, assim_method)
    rows = []
    fields_by_time = defaultdict(dict)

    for i_time, time in enumerate(times):
        print(f"Reading {exp_name}/{assim_method}: time {i_time + 1}/{len(times)} {pd.to_datetime(time)}")

        member_arrays = {var: [] for var in required_vars}
        member_lats = None
        member_lons = None

        for member in members:
            path = member_maps[member][time]
            print(f"  member {member}: {os.path.basename(path)}")
            with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
                for var in required_vars:
                    arr = first_2d_array(ds, var, path, required=(var in ["UST", "HFX", "QFX", "LANDMASK"]))
                    member_arrays[var].append(arr)
                if member_lats is None or member_lons is None:
                    member_lats, member_lons = read_lat_lon(ds, path)

        arrays = {var: ensemble_nanmean(values) for var, values in member_arrays.items()}
        ocean_mask = arrays["LANDMASK"] < 0.5
        tc_region_mask = tc_region_mask_from_nr(time, member_lats, member_lons)
        analysis_mask = ocean_mask & tc_region_mask
        if not np.any(analysis_mask):
            warn(
                f"No ocean grid points found inside TC region for {exp_name}/{assim_method} at {time}; "
                "falling back to TC region without LANDMASK."
            )
            analysis_mask = tc_region_mask

        if not np.any(ocean_mask):
            warn(f"No ocean grid points found with LANDMASK == 0 for {exp_name}/{assim_method} at {time}")

        if arrays["U10"] is not None and arrays["V10"] is not None:
            wspd10 = np.sqrt(arrays["U10"] ** 2 + arrays["V10"] ** 2)
        else:
            wspd10 = None

        lh = Lv * arrays["QFX"]
        tau = rho * arrays["UST"] ** 2

        row = {
            "time": time,
            "experiment": exp_name,
            "assim_method": assim_method,
            "ensemble_members": len(members),
            "tc_region_points": int(tc_region_mask.sum()),
            "tc_ocean_points": int(analysis_mask.sum()),
            "UST_ocean_mean": nanmean_masked(arrays["UST"], analysis_mask),
            "tau_ocean_mean": nanmean_masked(tau, analysis_mask),
            "HFX_ocean_mean": nanmean_masked(arrays["HFX"], analysis_mask),
            "QFX_ocean_mean": nanmean_masked(arrays["QFX"], analysis_mask),
            "LH_ocean_mean": nanmean_masked(lh, analysis_mask),
            "PBLH_ocean_mean": nanmean_masked(arrays["PBLH"], analysis_mask),
            "TSK_ocean_mean": nanmean_masked(arrays["TSK"], analysis_mask),
            "T2_ocean_mean": nanmean_masked(arrays["T2"], analysis_mask),
        }

        for threshold in strong_wind_thresholds:
            suffix = f"wind{int(threshold)}"
            if wspd10 is None:
                wind_mask = np.zeros_like(ocean_mask, dtype=bool)
                warn(f"Cannot calculate WSPD10 > {threshold:g} m/s for {exp_name}/{assim_method} at {time}; U10/V10 missing.")
            else:
                wind_mask = analysis_mask & (wspd10 > threshold)
                if not np.any(wind_mask):
                    warn(f"No TC-ocean grid points satisfy WSPD10 > {threshold:g} m/s for {exp_name}/{assim_method} at {time}")
            row[f"UST_{suffix}_mean"] = nanmean_masked(arrays["UST"], wind_mask)
            row[f"tau_{suffix}_mean"] = nanmean_masked(tau, wind_mask)
            row[f"HFX_{suffix}_mean"] = nanmean_masked(arrays["HFX"], wind_mask)
            row[f"QFX_{suffix}_mean"] = nanmean_masked(arrays["QFX"], wind_mask)
            row[f"LH_{suffix}_mean"] = nanmean_masked(lh, wind_mask)

        rows.append(row)

        spatial_plot_mask = np.where(analysis_mask, 1.0, np.nan)
        fields_by_time[time]["UST"] = arrays["UST"] * spatial_plot_mask
        fields_by_time[time]["tau"] = tau * spatial_plot_mask
        fields_by_time[time]["HFX"] = arrays["HFX"] * spatial_plot_mask
        fields_by_time[time]["QFX"] = arrays["QFX"] * spatial_plot_mask
        fields_by_time[time]["LH"] = lh * spatial_plot_mask
        fields_by_time[time]["TSK"] = arrays["TSK"] * spatial_plot_mask if arrays["TSK"] is not None else None
        fields_by_time[time]["T2"] = arrays["T2"] * spatial_plot_mask if arrays["T2"] is not None else None
        fields_by_time[time]["PBLH"] = arrays["PBLH"] * spatial_plot_mask if arrays["PBLH"] is not None else None
        fields_by_time[time]["LANDMASK"] = arrays["LANDMASK"]
        fields_by_time[time]["TC_REGION_MASK"] = tc_region_mask

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    for base_col in ["HFX_ocean_mean", "QFX_ocean_mean", "LH_ocean_mean"]:
        cum_name = base_col.replace("_ocean_mean", "_cum")
        df[cum_name] = (df[base_col].fillna(0.0) * dt).cumsum()

    return df, dict(fields_by_time)


def read_all_results():
    frames = []
    fields = defaultdict(dict)
    results = defaultdict(dict)

    for exp_name in base_dirs:
        for assim_method in assim_methods:
            df, field_dict = read_one_experiment(exp_name, assim_method)
            frames.append(df)
            fields[exp_name][assim_method] = field_dict
            results[exp_name][assim_method] = df

    all_df = pd.concat(frames, ignore_index=True)
    return results, fields, all_df


def check_time_alignment(results):
    reference_key = None
    reference_times = None
    warnings_found = []

    for exp_name in base_dirs:
        for assim_method in assim_methods:
            times = tuple(pd.to_datetime(results[exp_name][assim_method]["time"]))
            key = f"{exp_name}_{assim_method}"
            if reference_times is None:
                reference_key = key
                reference_times = times
            elif times != reference_times:
                warnings_found.append(f"{key} times do not match {reference_key}")

    if warnings_found:
        message = "; ".join(warnings_found)
        if strict_time_alignment:
            raise ValueError(f"Time alignment error: {message}")
        warn(f"Time alignment warning: {message}")
    else:
        print("Time alignment check passed for all EXP/assimilation combinations.")


# =============================================================================
# Difference tables
# =============================================================================

diff_metric_columns = [
    "UST_ocean_mean",
    "tau_ocean_mean",
    "HFX_ocean_mean",
    "QFX_ocean_mean",
    "LH_ocean_mean",
    "PBLH_ocean_mean",
    "TSK_ocean_mean",
    "T2_ocean_mean",
    "UST_wind10_mean",
    "tau_wind10_mean",
    "HFX_wind10_mean",
    "QFX_wind10_mean",
    "LH_wind10_mean",
    "UST_wind15_mean",
    "tau_wind15_mean",
    "HFX_wind15_mean",
    "QFX_wind15_mean",
    "LH_wind15_mean",
    "HFX_cum",
    "QFX_cum",
    "LH_cum",
]


def difference_df(left, right, label, assim_method=None, experiment=None):
    common = pd.merge(
        left[["time"] + diff_metric_columns],
        right[["time"] + diff_metric_columns],
        on="time",
        how="inner",
        suffixes=("_left", "_right"),
    )
    if len(common) != len(left) or len(common) != len(right):
        warn(f"Difference {label} uses {len(common)} common times; input lengths are {len(left)} and {len(right)}.")

    out = pd.DataFrame({"time": common["time"], "difference": label})
    if experiment is not None:
        out["experiment"] = experiment
    if assim_method is not None:
        out["assim_method"] = assim_method
    for col in diff_metric_columns:
        out[col] = common[f"{col}_left"] - common[f"{col}_right"]
    return out


def write_difference_csvs(results):
    csv_dir = os.path.join(output_dir, "csv")
    exp_pairs = [("EXP2", "EXP1"), ("EXP3", "EXP2"), ("EXP3", "EXP1")]
    method_pairs = [("QCF_RHF", "EAKF")]

    combined_exp_diffs = {("EXP2", "EXP1"): [], ("EXP3", "EXP2"): [], ("EXP3", "EXP1"): []}

    for assim_method in assim_methods:
        for left_exp, right_exp in exp_pairs:
            label = f"{left_exp}_minus_{right_exp}"
            df = difference_df(
                results[left_exp][assim_method],
                results[right_exp][assim_method],
                label,
                assim_method=assim_method,
            )
            df.to_csv(os.path.join(csv_dir, f"diff_{assim_method}_{label}.csv"), index=False)
            combined_exp_diffs[(left_exp, right_exp)].append(df)

    for (left_exp, right_exp), frames in combined_exp_diffs.items():
        label = f"{left_exp}_minus_{right_exp}"
        pd.concat(frames, ignore_index=True).to_csv(os.path.join(csv_dir, f"diff_{label}.csv"), index=False)

    for exp_name in base_dirs:
        for left_method, right_method in method_pairs:
            label = f"{exp_name}_{left_method}_minus_{right_method}"
            df = difference_df(
                results[exp_name][left_method],
                results[exp_name][right_method],
                label,
                experiment=exp_name,
            )
            df.to_csv(os.path.join(csv_dir, f"diff_{label}.csv"), index=False)


# =============================================================================
# Plotting
# =============================================================================

plot_vars = [
    ("UST_ocean_mean", "UST ocean mean", "m s-1"),
    ("tau_ocean_mean", "Sea-surface stress ocean mean", "N m-2"),
    ("HFX_ocean_mean", "Sensible heat flux ocean mean", "W m-2"),
    ("QFX_ocean_mean", "Moisture flux ocean mean", "kg m-2 s-1"),
    ("LH_ocean_mean", "Latent heat flux ocean mean", "W m-2"),
    ("HFX_cum", "Cumulative sensible heat flux", "J m-2"),
    ("LH_cum", "Cumulative latent heat flux", "J m-2"),
]


flux_plot_columns = {
    "UST_ocean_mean",
    "tau_ocean_mean",
    "HFX_ocean_mean",
    "QFX_ocean_mean",
    "LH_ocean_mean",
    "HFX_cum",
    "QFX_cum",
    "LH_cum",
}


def dataframe_for_plot(df, columns):
    """Optionally drop the first time only for flux/exchange time-series plots."""
    if not skip_initial_flux_time_in_plots:
        return df
    if not set(columns).issubset(flux_plot_columns):
        return df
    if len(df) <= 1:
        return df
    return df.iloc[1:].copy()


def plot_timeseries_by_assim(results):
    fig_dir = os.path.join(output_dir, "figures", "timeseries")
    colors = {"EXP1": "#0072BD", "EXP2": "#009E73", "EXP3": "#D0002E"}

    for assim_method in assim_methods:
        for col, title, ylabel in plot_vars:
            fig, ax = plt.subplots(figsize=(10, 5))
            for exp_name in base_dirs:
                df = dataframe_for_plot(results[exp_name][assim_method], [col])
                ax.plot(df["time"], df[col], marker="o", ms=3, lw=1.8, label=exp_name, color=colors.get(exp_name))
            ax.set_title(f"{assim_method}: {title}")
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            ax.grid(True, ls="--", alpha=0.35)
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"timeseries_{assim_method}_{col}_EXP1_EXP2_EXP3.png"), dpi=300)
            plt.close(fig)


def plot_timeseries_by_experiment(results):
    fig_dir = os.path.join(output_dir, "figures", "timeseries")
    colors = {"EAKF": "#0072BD", "QCF_RHF": "#D0002E"}

    for exp_name in base_dirs:
        for col, title, ylabel in plot_vars:
            fig, ax = plt.subplots(figsize=(10, 5))
            for assim_method in assim_methods:
                df = dataframe_for_plot(results[exp_name][assim_method], [col])
                ax.plot(df["time"], df[col], marker="o", ms=3, lw=1.8, label=assim_method, color=colors.get(assim_method))
            ax.set_title(f"{exp_name}: {title}")
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            ax.grid(True, ls="--", alpha=0.35)
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"timeseries_{exp_name}_{col}_EAKF_QCF_RHF.png"), dpi=300)
            plt.close(fig)


def plot_diff_timeseries(diff_df, label, cols, outfile):
    diff_df = dataframe_for_plot(diff_df, cols)
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 2.6 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.axhline(0.0, color="k", lw=0.8)
        ax.plot(diff_df["time"], diff_df[col], marker="o", ms=3, lw=1.6)
        ax.set_ylabel(col.replace("_ocean_mean", "").replace("_mean", ""))
        ax.grid(True, ls="--", alpha=0.35)
    axes[0].set_title(label)
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_exp3_minus_exp2_timeseries(results):
    fig_dir = os.path.join(output_dir, "figures", "timeseries")
    for assim_method in assim_methods:
        diff = difference_df(
            results["EXP3"][assim_method],
            results["EXP2"][assim_method],
            "EXP3_minus_EXP2",
            assim_method=assim_method,
        )
        plot_diff_timeseries(
            diff,
            f"{assim_method}: EXP3 - EXP2 air-sea flux differences",
            ["UST_ocean_mean", "tau_ocean_mean", "HFX_ocean_mean", "QFX_ocean_mean", "LH_ocean_mean"],
            os.path.join(fig_dir, f"diff_{assim_method}_EXP3_minus_EXP2_UST_tau_HFX_QFX_LH.png"),
        )
        plot_diff_timeseries(
            diff,
            f"{assim_method}: EXP3 - EXP2 thermal/PBL differences",
            ["TSK_ocean_mean", "T2_ocean_mean", "PBLH_ocean_mean"],
            os.path.join(fig_dir, f"diff_{assim_method}_EXP3_minus_EXP2_TSK_T2_PBLH.png"),
        )


def common_times_for_fields(fields_a, fields_b):
    a_times = list(fields_a.keys())
    b_times = set(fields_b.keys())
    common = sorted([t for t in a_times if t in b_times])
    if not common:
        raise ValueError("No common times found for spatial difference maps.")
    return common


def selected_times(times):
    n = len(times)
    indexes = sorted(set([0, n // 2, n - 1]))
    return [times[i] for i in indexes]


def plot_spatial_diff(diff_arr, title, outfile):
    levels = symmetric_levels(diff_arr)
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(diff_arr, levels=levels, cmap="RdBu_r", extend="both")
    ax.set_title(title)
    ax.set_xlabel("West-East grid index")
    ax.set_ylabel("South-North grid index")
    cbar = fig.colorbar(cf, ax=ax, shrink=0.86)
    cbar.set_label("Difference")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def plot_spatial_differences(fields):
    fig_dir = os.path.join(output_dir, "figures", "spatial")
    variables = ["UST", "HFX", "QFX", "LH", "TSK"]

    comparisons = []
    for assim_method in assim_methods:
        comparisons.append((f"{assim_method}_EXP3_minus_EXP2", fields["EXP3"][assim_method], fields["EXP2"][assim_method]))
    for exp_name in base_dirs:
        comparisons.append((f"{exp_name}_QCF_RHF_minus_EAKF", fields[exp_name]["QCF_RHF"], fields[exp_name]["EAKF"]))

    for label, left_fields, right_fields in comparisons:
        times = selected_times(common_times_for_fields(left_fields, right_fields))
        for time in times:
            for var in variables:
                left = left_fields[time].get(var)
                right = right_fields[time].get(var)
                if left is None or right is None:
                    warn(f"Skipping spatial map {label} {var} {time}: missing variable.")
                    continue
                diff_arr = left - right
                tstr = safe_name_time(time)
                plot_spatial_diff(
                    diff_arr,
                    f"{label}: {var} at {pd.to_datetime(time)}",
                    os.path.join(fig_dir, f"spatial_{label}_{var}_{tstr}.png"),
                )


# =============================================================================
# Summary and automatic diagnosis
# =============================================================================

def summarize_means(results):
    lines = []
    lines.append("Time-mean ocean diagnostics by experiment and assimilation method")
    lines.append("=" * 72)
    cols = ["UST_ocean_mean", "tau_ocean_mean", "HFX_ocean_mean", "QFX_ocean_mean", "LH_ocean_mean", "TSK_ocean_mean", "T2_ocean_mean", "PBLH_ocean_mean"]

    for exp_name in base_dirs:
        for assim_method in assim_methods:
            df = results[exp_name][assim_method]
            means = df[cols].mean(skipna=True)
            lines.append(f"\n{exp_name} / {assim_method}")
            for col in cols:
                lines.append(f"  mean({col}) = {means[col]:.6g}")

    lines.append("\nKey average differences")
    lines.append("=" * 72)

    key_diffs = []
    for assim_method in assim_methods:
        key_diffs.append((f"{assim_method} EXP3 - EXP2", difference_df(results["EXP3"][assim_method], results["EXP2"][assim_method], "EXP3_minus_EXP2", assim_method=assim_method)))
    for exp_name in base_dirs:
        key_diffs.append((f"{exp_name} QCF_RHF - EAKF", difference_df(results[exp_name]["QCF_RHF"], results[exp_name]["EAKF"], f"{exp_name}_QCF_RHF_minus_EAKF", experiment=exp_name)))

    for label, diff in key_diffs:
        means = diff[cols].mean(skipna=True)
        lines.append(f"\n{label}")
        for col in cols:
            lines.append(f"  mean(diff {col}) = {means[col]:.6g}")

    return "\n".join(lines)


def automatic_diagnosis(results):
    lines = []
    lines.append("Automatic physical diagnosis")
    lines.append("=" * 72)
    eps = 1.0e-12

    for assim_method in assim_methods:
        diff = difference_df(results["EXP3"][assim_method], results["EXP2"][assim_method], "EXP3_minus_EXP2", assim_method=assim_method)
        means = diff[["HFX_ocean_mean", "QFX_ocean_mean", "LH_ocean_mean", "UST_ocean_mean", "tau_ocean_mean", "TSK_ocean_mean"]].mean(skipna=True)
        hfx = means["HFX_ocean_mean"]
        qfx = means["QFX_ocean_mean"]
        lh = means["LH_ocean_mean"]
        ust = means["UST_ocean_mean"]
        tau = means["tau_ocean_mean"]
        tsk = means["TSK_ocean_mean"]

        lines.append(f"\n{assim_method}: mean EXP3 - EXP2")
        lines.append(f"  HFX={hfx:.6g}, QFX={qfx:.6g}, LH={lh:.6g}, UST={ust:.6g}, tau={tau:.6g}, TSK={tsk:.6g}")

        if hfx < -eps and qfx < -eps:
            lines.append("  海洋更新可能削弱了海洋向大气的感热和水汽供应。")
        if qfx < -eps or lh < -eps:
            lines.append("  海洋更新可能削弱了潜热和水汽输入。")
        if ust < -eps or tau < -eps:
            lines.append("  海洋更新可能削弱了近海面动量交换或机械湍流。")
        if tsk < -eps and (hfx < -eps or qfx < -eps or lh < -eps):
            lines.append("  可能存在海洋冷却反馈。")
        if not any([hfx < -eps and qfx < -eps, qfx < -eps or lh < -eps, ust < -eps or tau < -eps, tsk < -eps and (hfx < -eps or qfx < -eps or lh < -eps)]):
            lines.append("  未触发预设的削弱/冷却反馈判据，请结合空间图和时间序列进一步判断。")

    for exp_name in base_dirs:
        diff = difference_df(results[exp_name]["QCF_RHF"], results[exp_name]["EAKF"], f"{exp_name}_QCF_RHF_minus_EAKF", experiment=exp_name)
        means = diff[["HFX_ocean_mean", "QFX_ocean_mean", "LH_ocean_mean", "UST_ocean_mean", "tau_ocean_mean", "TSK_ocean_mean"]].mean(skipna=True)
        lines.append(f"\n{exp_name}: mean QCF_RHF - EAKF")
        lines.append(
            "  "
            + ", ".join(
                [
                    f"HFX={means['HFX_ocean_mean']:.6g}",
                    f"QFX={means['QFX_ocean_mean']:.6g}",
                    f"LH={means['LH_ocean_mean']:.6g}",
                    f"UST={means['UST_ocean_mean']:.6g}",
                    f"tau={means['tau_ocean_mean']:.6g}",
                    f"TSK={means['TSK_ocean_mean']:.6g}",
                ]
            )
        )

    return "\n".join(lines)


def write_reports(results):
    report = summarize_means(results) + "\n\n" + automatic_diagnosis(results)
    path = os.path.join(output_dir, "air_sea_flux_diagnosis_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSummary report saved to: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_output_dirs()

    print("Starting WRF air-sea flux/PBL diagnostics")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"File glob domain: wrfout_{domain}_*")

    results, fields, all_df = read_all_results()
    check_time_alignment(results)

    csv_dir = os.path.join(output_dir, "csv")
    all_csv = os.path.join(csv_dir, "air_sea_flux_ocean_statistics_all_experiments.csv")
    all_df.to_csv(all_csv, index=False)
    print(f"Saved combined statistics CSV: {all_csv}")

    write_difference_csvs(results)
    print(f"Saved difference CSV files in: {csv_dir}")

    plot_timeseries_by_assim(results)
    plot_timeseries_by_experiment(results)
    plot_exp3_minus_exp2_timeseries(results)
    plot_spatial_differences(fields)
    print(f"Saved PNG figures in: {os.path.join(output_dir, 'figures')}")

    write_reports(results)
    print("Done.")


if __name__ == "__main__":
    main()
