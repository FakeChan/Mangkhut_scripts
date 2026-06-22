"""
Plot time series of TC-region 2-D/3-D variable RMSE against the NR run.

The TC region is treated as the full NR d03 footprint. NR d03 is thinned from
300 m to about the d02 1.5 km spacing, then each ensemble-member d02 field is
linearly interpolated to that thinned NR grid. The script computes

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

FILTER_KINDS = ["EAKF", "QCF_RHF"]

EXP_DIRS = {
    "Exp_oceanAssim0Run0": Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim0Run0"),
    "Exp_oceanAssim0Run1": Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim0Run1"),
    "Exp_oceanAssim1Run1": Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim1Run1"),
}

MEMBERS = ["006", "015", "029", "037", "043", "044"]

# Select the model-level index here. Use None for 2-D variables.
VERTICAL_LEVEL_INDEX = 10

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
def read_field_and_grid_cached(
    path_str: str,
    name: str,
    level_index: int | None,
    add_offset: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path_str)
    variable = variable_from_signature(name, level_index, add_offset, scale)
    with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
        lats, lons = read_lat_lon(ds)
        values = select_2d_field(ds, variable, path)

    if values.shape != lats.shape:
        raise ValueError(f"{name} shape {values.shape} does not match XLAT shape {lats.shape}: {path}")
    return lats, lons, values


def read_field_and_grid(path: Path, variable: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return read_field_and_grid_cached(str(path), *variable_signature(variable))


def thin_2d(values: np.ndarray, factor: int, method: str) -> np.ndarray:
    if factor <= 1:
        return values
    if method == "stride":
        return values[::factor, ::factor]
    if method == "coarsen_mean":
        da = xr.DataArray(values, dims=("south_north", "west_east"))
        return da.coarsen(south_north=factor, west_east=factor, boundary="trim").mean().values
    raise ValueError(f"Unsupported NR_THIN_METHOD={method!r}")


def read_thinned_nr(path: Path, variable: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats, lons, values = read_field_and_grid(path, variable)
    return (
        thin_2d(lats, NR_THIN_FACTOR, NR_THIN_METHOD),
        thin_2d(lons, NR_THIN_FACTOR, NR_THIN_METHOD),
        thin_2d(values, NR_THIN_FACTOR, NR_THIN_METHOD),
    )


def interp_member_to_nr_grid(
    member_lats: np.ndarray,
    member_lons: np.ndarray,
    member_values: np.ndarray,
    nr_lats: np.ndarray,
    nr_lons: np.ndarray,
) -> np.ndarray:
    points = np.column_stack((member_lons.ravel(), member_lats.ravel()))
    values_flat = member_values.ravel()
    valid = np.isfinite(points).all(axis=1) & np.isfinite(values_flat)
    if valid.sum() < 3:
        raise ValueError("Not enough valid member points for interpolation")

    interpolator = LinearNDInterpolator(points[valid], values_flat[valid], fill_value=np.nan)
    return np.asarray(interpolator(nr_lons, nr_lats), dtype=float)


def spatial_rmse(test_values: np.ndarray, truth_values: np.ndarray) -> float:
    mask = np.isfinite(test_values) & np.isfinite(truth_values)
    if not np.any(mask):
        return np.nan
    diff = test_values[mask] - truth_values[mask]
    return float(np.sqrt(np.nanmean(diff ** 2)))


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

        for variable in TARGET_VARIABLES:
            nr_lats, nr_lons, nr_values = read_thinned_nr(nr_file, variable)
            var_key = variable_key(variable)

            for exp_name, exp_base_dir in EXP_DIRS.items():
                member_fields = []

                for member in MEMBERS:
                    try:
                        member_file = find_member_file(exp_base_dir, filter_kind, member, time_obj)
                        member_lats, member_lons, member_values = read_field_and_grid(member_file, variable)
                        member_on_nr = interp_member_to_nr_grid(
                            member_lats,
                            member_lons,
                            member_values,
                            nr_lats,
                            nr_lons,
                        )
                        rmse = spatial_rmse(member_on_nr, nr_values)
                        member_fields.append(member_on_nr)
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
                                "nr_points": int(np.isfinite(nr_values).sum()),
                            }
                        )
                    except Exception as exc:
                        print(f"  [skip] {exp_name}/{filter_kind}/{member} {var_key}: {exc}")

                if member_fields:
                    ens_mean = np.nanmean(np.stack(member_fields, axis=0), axis=0)
                    ens_mean_rmse = spatial_rmse(ens_mean, nr_values)
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
                            "nr_points": int(np.isfinite(nr_values).sum()),
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
