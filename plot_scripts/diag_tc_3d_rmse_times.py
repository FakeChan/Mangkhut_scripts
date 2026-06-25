"""
Plot time series of TC-region 2-D/3-D variable RMSE against the NR run.

The TC center is defined by the minimum PSFC in NR d03. NR d03 is thinned from
300 m to about the d02 1.5 km spacing, then each ensemble-member d02 field is
linearly interpolated to that thinned NR grid. RMSE is computed only within
TC_RADIUS_KM of the NR center. The script computes

    1. RMSE for each ensemble member relative to NR.
    2. RMSE of the ensemble-mean field relative to NR.

All parameters are configured at the top of this file. No command-line parser is
used.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

# =============================================================================
# User settings
# =============================================================================

START_TIME = datetime(2018, 9, 10, 0, 0)
END_TIME = datetime(2018, 9, 10, 6, 0)
INTERVAL = timedelta(minutes=30)

NR_DIR = Path("/scratch/lililei1/kcfu/tc_mangkhut/NR")
NR_DOMAIN = "d03"
MEMBER_DOMAIN = "d02"

# d03 is 300 m and d02 is 1.5 km, so every fifth d03 grid point is used.
NR_THIN_FACTOR = 5
# "stride" means sparse sampling. Change to "coarsen_mean" for block averages.
NR_THIN_METHOD = "stride"
TC_RADIUS_KM = 150.0
CENTER_PRESSURE_VAR = "PSFC"

FILTER_KINDS = ["EAKF"]

EXP_DIRS = {
    "Exp_oceanAssim0Run0": Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim0Run0"),
    "Exp_oceanAssim0Run1": Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim0Run1"),
    "Exp_oceanAssim1Run1": Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim1Run1"),
}

MEMBERS = ["006", "015", "029", "037", "043", "044"]

# Select the model-level index here. Use None for 2-D variables.
VERTICAL_LEVEL_INDEX = 14

TARGET_VARIABLES = [
    {
        "name": "MU",
        "level_index": None,
        "add_offset": 0.0,
        "scale": 1.0,
        "unit": "Pa",
        "label": "MU",
    },
    {
        "name": "QVAPOR",
        "level_index": VERTICAL_LEVEL_INDEX,
        "add_offset": 0.0,
        "scale": 1000.0,
        "unit": "g kg-1",
        "label": "QVAPOR",
    },
    {
        "name": "P",
        "level_index": VERTICAL_LEVEL_INDEX,
        "add_offset": 0.0,
        "scale": 1.0,
        "unit": "Pa",
        "label": "P",
    },
    {
        "name": "T",
        "level_index": VERTICAL_LEVEL_INDEX,
        "add_offset": 300.0,
        "scale": 1.0,
        "unit": "K",
        "label": "T + 300",
    },
]

OUTPUT_DIR = Path("./figs/tc_3d_rmse_times")
SAVE_CSV = True

# Colors follow diag_tc_diagnostics_err_times.py.
EXP_COLORS = {
    "Exp_oceanAssim0Run0": "goldenrod",
    "Exp_oceanAssim0Run1": "dodgerblue",
    "Exp_oceanAssim1Run1": "crimson",
}


# =============================================================================
# Basic helpers
# =============================================================================

def build_times(start_time: datetime, end_time: datetime, interval: timedelta) -> list[datetime]:
    times = []
    curr_time = start_time
    while curr_time <= end_time:
        times.append(curr_time)
        curr_time += interval
    return times


def wrf_time_name(time_obj: datetime) -> str:
    return time_obj.strftime("%Y-%m-%d_%H:%M:%S")


def find_wrf_file(base_dir: Path, domain: str, time_obj: datetime) -> Path:
    pattern = f"wrfout_{domain}_{wrf_time_name(time_obj)}"
    direct = base_dir / pattern
    if direct.exists():
        return direct

    matches = sorted(base_dir.rglob(pattern))
    if not matches:
        matches = sorted(base_dir.rglob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} under {base_dir}")
    return matches[0]


def find_member_file(exp_base_dir: Path, filter_kind: str, member: str, time_obj: datetime) -> Path:
    member_dir = exp_base_dir / filter_kind / member
    return find_wrf_file(member_dir, MEMBER_DOMAIN, time_obj)


def find_nr_file(time_obj: datetime) -> Path:
    if NR_DIR.is_file():
        return NR_DIR
    return find_wrf_file(NR_DIR, NR_DOMAIN, time_obj)


def isel_time0(da: xr.DataArray) -> xr.DataArray:
    for dim in da.dims:
        if dim.lower() == "time":
            return da.isel({dim: 0})
    return da


def read_lat_lon(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    if "XLAT" not in ds or "XLONG" not in ds:
        raise KeyError("XLAT/XLONG not found")
    lats = np.asarray(isel_time0(ds["XLAT"]).values, dtype=float)
    lons = np.asarray(isel_time0(ds["XLONG"]).values, dtype=float)
    return lats, lons


def read_2d_variable(ds: xr.Dataset, name: str, path: Path) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"{name} not found in {path}")
    data = isel_time0(ds[name])
    if data.ndim != 2:
        raise ValueError(f"{name} should be 2-D after Time selection, got dims={data.dims}: {path}")
    return np.asarray(data.values, dtype=float)


def select_2d_field(ds: xr.Dataset, variable: dict, path: Path) -> np.ndarray:
    name = variable["name"]
    if name not in ds:
        raise KeyError(f"{name} not found in {path}")

    data = isel_time0(ds[name])
    level_index = variable["level_index"]
    if data.ndim == 2:
        if level_index is not None:
            raise ValueError(
                f"{name} is 2-D after Time selection, but level_index={level_index} was configured: {path}"
            )
        values = np.asarray(data.values, dtype=float)
        return (values + variable["add_offset"]) * variable["scale"]

    if data.ndim != 3:
        raise ValueError(f"{name} should be 2-D or 3-D after Time selection, got dims={data.dims}: {path}")

    if level_index is None:
        raise ValueError(f"{name} is 3-D after Time selection. Set level_index to an integer: {path}")

    vertical_dim = data.dims[0]
    if level_index < 0 or level_index >= data.sizes[vertical_dim]:
        raise IndexError(
            f"{name} level_index={level_index} is outside {vertical_dim} size "
            f"{data.sizes[vertical_dim]} in {path}"
        )

    values = np.asarray(data.isel({vertical_dim: level_index}).values, dtype=float)
    return (values + variable["add_offset"]) * variable["scale"]


def variable_signature(variable: dict) -> tuple[str, int | None, float, float]:
    return (
        variable["name"],
        None if variable["level_index"] is None else int(variable["level_index"]),
        float(variable["add_offset"]),
        float(variable["scale"]),
    )


def variable_from_signature(name: str, level_index: int | None, add_offset: float, scale: float) -> dict:
    return {
        "name": name,
        "level_index": level_index,
        "add_offset": add_offset,
        "scale": scale,
    }


@lru_cache(maxsize=None)
def read_fields_and_grid_cached(
    path_str: str,
    variable_signatures: tuple[tuple[str, int | None, float, float], ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    path = Path(path_str)
    with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
        lats, lons = read_lat_lon(ds)
        fields = {}
        for signature in variable_signatures:
            variable = variable_from_signature(*signature)
            values = select_2d_field(ds, variable, path)
            if values.shape != lats.shape:
                raise ValueError(
                    f"{variable['name']} shape {values.shape} does not match XLAT shape {lats.shape}: {path}"
                )
            fields[variable_key(variable)] = values

    return lats, lons, fields


def variable_signatures(variables: list[dict]) -> tuple[tuple[str, int | None, float, float], ...]:
    return tuple(variable_signature(variable) for variable in variables)


def read_fields_and_grid(path: Path, variables: list[dict]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    return read_fields_and_grid_cached(str(path), variable_signatures(variables))


def thin_2d(values: np.ndarray, factor: int, method: str) -> np.ndarray:
    if factor <= 1:
        return values
    if method == "stride":
        return values[::factor, ::factor]
    if method == "coarsen_mean":
        da = xr.DataArray(values, dims=("south_north", "west_east"))
        return da.coarsen(south_north=factor, west_east=factor, boundary="trim").mean().values
    raise ValueError(f"Unsupported NR_THIN_METHOD={method!r}")


def latlon_distance_km(lats: np.ndarray, lons: np.ndarray, center_lat: float, center_lon: float) -> np.ndarray:
    earth_radius_km = 6371.0
    lat1 = np.deg2rad(lats)
    lon1 = np.deg2rad(lons)
    lat2 = np.deg2rad(center_lat)
    lon2 = np.deg2rad(center_lon)
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


@lru_cache(maxsize=None)
def read_nr_time_data_cached(
    path_str: str,
    variable_signatures_: tuple[tuple[str, int | None, float, float], ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, float, float, float]:
    path = Path(path_str)
    with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
        lats, lons = read_lat_lon(ds)
        center_pressure = read_2d_variable(ds, CENTER_PRESSURE_VAR, path)
        fields = {}
        for signature in variable_signatures_:
            variable = variable_from_signature(*signature)
            values = select_2d_field(ds, variable, path)
            if values.shape != lats.shape:
                raise ValueError(
                    f"{variable['name']} shape {values.shape} does not match XLAT shape {lats.shape}: {path}"
                )
            fields[variable_key(variable)] = values

    if center_pressure.shape != lats.shape:
        raise ValueError(
            f"{CENTER_PRESSURE_VAR} shape {center_pressure.shape} does not match XLAT shape {lats.shape}: {path}"
        )
    center_idx = np.unravel_index(np.nanargmin(center_pressure), center_pressure.shape)
    center_lat = float(lats[center_idx])
    center_lon = float(lons[center_idx])
    center_pressure_min = float(center_pressure[center_idx])

    nr_lats = thin_2d(lats, NR_THIN_FACTOR, NR_THIN_METHOD)
    nr_lons = thin_2d(lons, NR_THIN_FACTOR, NR_THIN_METHOD)
    nr_fields = {
        key: thin_2d(values, NR_THIN_FACTOR, NR_THIN_METHOD)
        for key, values in fields.items()
    }
    tc_mask = latlon_distance_km(nr_lats, nr_lons, center_lat, center_lon) <= TC_RADIUS_KM
    if not np.any(tc_mask):
        raise ValueError(
            f"No thinned NR points inside TC_RADIUS_KM={TC_RADIUS_KM:g} km for {path}; "
            f"center lat={center_lat:.4f}, lon={center_lon:.4f}"
        )
    return nr_lats, nr_lons, nr_fields, tc_mask, center_lat, center_lon, center_pressure_min


def read_nr_time_data(
    path: Path,
    variables: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, float, float, float]:
    return read_nr_time_data_cached(str(path), variable_signatures(variables))


def build_interpolation_geometry(member_lats: np.ndarray, member_lons: np.ndarray) -> tuple[Delaunay, np.ndarray]:
    points = np.column_stack((member_lons.ravel(), member_lats.ravel()))
    coord_valid = np.isfinite(points).all(axis=1)
    if coord_valid.sum() < 3:
        raise ValueError("Not enough valid member grid points for interpolation")
    triangulation = Delaunay(points[coord_valid])
    return triangulation, coord_valid


def interp_member_to_nr_grid(
    triangulation: Delaunay,
    coord_valid: np.ndarray,
    member_values: np.ndarray,
    nr_lats: np.ndarray,
    nr_lons: np.ndarray,
) -> np.ndarray:
    values_flat = member_values.ravel()[coord_valid]
    values_valid = np.isfinite(values_flat)
    if values_valid.sum() < 3:
        raise ValueError("Not enough valid member points for interpolation")

    if np.all(values_valid):
        interpolator = LinearNDInterpolator(triangulation, values_flat, fill_value=np.nan)
    else:
        interpolator = LinearNDInterpolator(
            triangulation.points[values_valid],
            values_flat[values_valid],
            fill_value=np.nan,
        )
    return np.asarray(interpolator(nr_lons, nr_lats), dtype=float)


def spatial_rmse(test_values: np.ndarray, truth_values: np.ndarray, region_mask: np.ndarray | None = None) -> float:
    mask = np.isfinite(test_values) & np.isfinite(truth_values)
    if region_mask is not None:
        mask &= region_mask
    if not np.any(mask):
        return np.nan
    diff = test_values[mask] - truth_values[mask]
    return float(np.sqrt(np.nanmean(diff ** 2)))


def valid_rmse_points(test_values: np.ndarray, truth_values: np.ndarray, region_mask: np.ndarray) -> int:
    return int((np.isfinite(test_values) & np.isfinite(truth_values) & region_mask).sum())


def ensemble_mean_no_warning(fields: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(fields, axis=0)
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)
    sums = np.where(valid, stack, 0.0).sum(axis=0)
    mean = np.full(stack.shape[1:], np.nan, dtype=float)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return mean


def variable_key(variable: dict) -> str:
    if variable["level_index"] is None:
        return variable["name"]
    return f"{variable['name']}_lev{variable['level_index']}"


def variable_title(variable: dict) -> str:
    if variable["level_index"] is None:
        return variable["label"]
    return f"{variable['label']} level {variable['level_index']}"


# =============================================================================
# Calculation
# =============================================================================

def calculate_for_filter(filter_kind: str, times: list[datetime]) -> pd.DataFrame:
    records = []

    for time_obj in times:
        time_name = wrf_time_name(time_obj)
        print(f"\n>>> Processing {filter_kind} {time_name}")
        nr_file = find_nr_file(time_obj)
        nr_lats, nr_lons, nr_fields, tc_mask, center_lat, center_lon, center_psfc = read_nr_time_data(
            nr_file,
            TARGET_VARIABLES,
        )
        tc_region_points = int(tc_mask.sum())
        print(
            f"  NR center from {CENTER_PRESSURE_VAR}: lat={center_lat:.4f}, "
            f"lon={center_lon:.4f}, min={center_psfc:.2f}; "
            f"TC points={tc_region_points}"
        )

        for exp_name, exp_base_dir in EXP_DIRS.items():
            member_fields_by_var = {variable_key(variable): [] for variable in TARGET_VARIABLES}

            for member in MEMBERS:
                try:
                    member_file = find_member_file(exp_base_dir, filter_kind, member, time_obj)
                    member_lats, member_lons, member_fields = read_fields_and_grid(member_file, TARGET_VARIABLES)
                    triangulation, coord_valid = build_interpolation_geometry(member_lats, member_lons)
                except Exception as exc:
                    print(f"  [skip] {exp_name}/{filter_kind}/{member}: {exc}")
                    continue

                for variable in TARGET_VARIABLES:
                    var_key = variable_key(variable)
                    nr_values = nr_fields[var_key]
                    try:
                        member_on_nr = interp_member_to_nr_grid(
                            triangulation,
                            coord_valid,
                            member_fields[var_key],
                            nr_lats,
                            nr_lons,
                        )
                        rmse = spatial_rmse(member_on_nr, nr_values, tc_mask)
                        rmse_points = valid_rmse_points(member_on_nr, nr_values, tc_mask)
                        member_fields_by_var[var_key].append(member_on_nr)
                    except Exception as exc:
                        print(f"  [skip] {exp_name}/{filter_kind}/{member} {var_key}: {exc}")
                        continue

                    records.append(
                        {
                            "time": time_name,
                            "time_obj": time_obj,
                            "filter_kind": filter_kind,
                            "experiment": exp_name,
                            "variable": var_key,
                            "level_index": variable["level_index"],
                            "member": member,
                            "metric": "member_rmse",
                            "rmse": rmse,
                            "unit": variable["unit"],
                            "tc_region_points": tc_region_points,
                            "rmse_points": rmse_points,
                            "center_lat": center_lat,
                            "center_lon": center_lon,
                            "center_psfc": center_psfc,
                        }
                    )

            for variable in TARGET_VARIABLES:
                var_key = variable_key(variable)
                member_fields = member_fields_by_var[var_key]
                if not member_fields:
                    continue

                nr_values = nr_fields[var_key]
                ens_mean = ensemble_mean_no_warning(member_fields)
                ens_mean_rmse = spatial_rmse(ens_mean, nr_values, tc_mask)
                rmse_points = valid_rmse_points(ens_mean, nr_values, tc_mask)
                records.append(
                    {
                        "time": time_name,
                        "time_obj": time_obj,
                        "filter_kind": filter_kind,
                        "experiment": exp_name,
                        "variable": var_key,
                        "level_index": variable["level_index"],
                        "member": "ensmean",
                        "metric": "ensemble_mean_rmse",
                        "rmse": ens_mean_rmse,
                        "unit": variable["unit"],
                        "tc_region_points": tc_region_points,
                        "rmse_points": rmse_points,
                        "center_lat": center_lat,
                        "center_lon": center_lon,
                        "center_psfc": center_psfc,
                    }
                )

    return pd.DataFrame.from_records(records)


# =============================================================================
# Plotting
# =============================================================================

def plot_filter(df: pd.DataFrame, filter_kind: str, save_path: Path) -> None:
    if df.empty:
        print(f"No records to plot for {filter_kind}")
        return

    fig, axes = plt.subplots(
        len(TARGET_VARIABLES),
        1,
        figsize=(11, 3.3 * len(TARGET_VARIABLES)),
        dpi=150,
        sharex=True,
        constrained_layout=True,
    )
    if len(TARGET_VARIABLES) == 1:
        axes = [axes]

    for ax, variable in zip(axes, TARGET_VARIABLES):
        var_key = variable_key(variable)
        var_df = df[df["variable"] == var_key].copy()
        var_df["time_dt"] = pd.to_datetime(var_df["time"], format="%Y-%m-%d_%H:%M:%S")

        for exp_name in EXP_DIRS:
            exp_member = var_df[
                (var_df["experiment"] == exp_name)
                & (var_df["metric"] == "member_rmse")
            ]
            for _, member_df in exp_member.groupby("member"):
                member_df = member_df.sort_values("time_dt")
                ax.plot(
                    member_df["time_dt"],
                    member_df["rmse"],
                    color="0.75",
                    linestyle="-",
                    linewidth=0.8,
                    alpha=0.65,
                    zorder=1,
                )

            ens_df = var_df[
                (var_df["experiment"] == exp_name)
                & (var_df["metric"] == "ensemble_mean_rmse")
            ].sort_values("time_dt")
            ax.plot(
                ens_df["time_dt"],
                ens_df["rmse"],
                color=EXP_COLORS[exp_name],
                linestyle="-",
                linewidth=2.4,
                alpha=0.95,
                label=exp_name,
                zorder=3,
            )

        ax.set_ylabel(f"RMSE ({variable['unit']})")
        ax.set_title(variable_title(variable), fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.75)

    legend_handles = [
        mlines.Line2D([], [], color=color, linewidth=2.4, label=exp_name)
        for exp_name, color in EXP_COLORS.items()
        if exp_name in EXP_DIRS
    ]
    member_handle = mlines.Line2D([], [], color="0.75", linewidth=0.9, label="Members")
    axes[0].legend(
        handles=[member_handle] + legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
    )

    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=35)
    fig.suptitle(
        f"{filter_kind}: TC-region variable RMSE vs NR d03 thinned to d02 spacing",
        fontsize=14,
        fontweight="bold",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    times = build_times(START_TIME, END_TIME, INTERVAL)
    all_frames = []

    for filter_kind in FILTER_KINDS:
        df = calculate_for_filter(filter_kind, times)
        all_frames.append(df)

        if SAVE_CSV:
            csv_path = OUTPUT_DIR / f"{filter_kind}_tc_3d_rmse_times.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")

        png_path = OUTPUT_DIR / f"{filter_kind}_tc_3d_rmse_times.png"
        plot_filter(df, filter_kind, png_path)

    if SAVE_CSV and all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = OUTPUT_DIR / "all_filters_tc_3d_rmse_times.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Saved {combined_path}")


if __name__ == "__main__":
    main()
