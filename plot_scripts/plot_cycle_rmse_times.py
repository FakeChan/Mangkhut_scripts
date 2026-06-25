"""
Plot domain RMSE for a configured WRF variable relative to NR.

NR d03 is finer than, and smaller than, the ensemble member domain. The script
linearly interpolates NR to each member grid and computes RMSE only over member
grid points inside the NR d03 linear-interpolation footprint.

All parameters are configured explicitly in this file. No command-line
arguments are used.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import LinearNDInterpolator


BASE_DIR = Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test")
NR_BASE = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout")
MEMBER_DOMAIN = "d02"
NR_DOMAIN = "d03"
START_TIME = "2018-09-10_00:00:00"
END_TIME = "2018-09-10_06:00:00"
STEP_MINUTES = 30
NR_COARSEN_FACTOR = 5

EXPERIMENTS = ["6mem_oceanAssim0Run0", "6mem_oceanAssim0Run1", "6mem_oceanAssim1Run1"]
FILTERS = ["EAKF"]
MEMBERS = ["006", "015", "029", "037", "043", "044"]

# Configure the target variable here.
#
# vertical_level:
#     None -> variable must be 2-D after selecting Time, e.g. PSFC(Time,y,x).
#     int  -> variable must be 3-D after selecting Time, e.g. U(Time,z,y,x).
#
# scale converts both member and NR values before RMSE is computed.
# power applies an optional exponent after scaling, e.g. power=2.0 makes UST
# become UST**2 before interpolation/RMSE. 
# e.g. UST ==( UST * scale ) **power
# lat_name/lon_name may be None for automatic WRF coordinate selection:
#     mass-grid variables: XLAT/XLONG
#     U-staggered variables: XLAT_U/XLONG_U
#     V-staggered variables: XLAT_V/XLONG_V
# TARGET_VARIABLE = {
#     "name": "OM_TMP",
#     "vertical_level": 0,
#     "scale": 1.0,
#     "unit": "K",
#     "experiments": ["6mem_oceanAssim0Run1", "6mem_oceanAssim1Run1"],
#     "lat_name": None,
#     "lon_name": None,
#     "out_csv": Path("./figs/omtmp_rmse_vs_nr_timeseries.csv"),
#     "out_png": Path("./figs/omtmp_rmse_vs_nr_timeseries.png"),
# }

# Example for a 2-D variable:
# TARGET_VARIABLE = {
#     "name": "MU",
#     "vertical_level": None,
#     "scale": 1,
#     "power": 1.0,
#     "unit": "hPa",
#     "experiments": EXPERIMENTS,
#     "lat_name": None,
#     "lon_name": None,
#     "out_csv": Path("./figs/mu_rmse_timeseries.csv"),
#     "out_png": Path("./figs/mu_rmse_timeseries.png"),
# }
#
# Example for a 3-D atmospheric variable:
TARGET_VARIABLE = {
    "name": "P",
    "vertical_level": 14,
    "scale": 1.0,
    "unit": "m s-1",
    "experiments": EXPERIMENTS,
    "lat_name": None,
    "lon_name": None,
    "out_csv": Path("./figs/u_level10_rmse_vs_nr_timeseries.csv"),
    "out_png": Path("./figs/u_level10_rmse_vs_nr_timeseries.png"),
}

# Okabe-Ito / colorblind-safe scientific palette.
EXP_COLORS = {
    "6mem_oceanAssim0Run0": "#0072B2",  # blue
    "6mem_oceanAssim0Run1": "#D55E00",  # vermillion
    "6mem_oceanAssim1Run1": "#009E73",  # bluish green
}
FILTER_LINESTYLES = {
    "EAKF": "-",
    "QCF_RHF": (0, (5, 2)),
}
FILTER_MARKERS = {
    "EAKF": "o",
    "QCF_RHF": "s",
}


def build_times(start: str, end: str, step_minutes: int) -> list[datetime]:
    t0 = datetime.strptime(start, "%Y-%m-%d_%H:%M:%S")
    t1 = datetime.strptime(end, "%Y-%m-%d_%H:%M:%S")
    out = []
    t = t0
    while t <= t1:
        out.append(t)
        t += timedelta(minutes=step_minutes)
    return out


def wrf_time_name(t: datetime) -> str:
    return t.strftime("%Y-%m-%d_%H:%M:%S")


def find_member_file(base: Path, exp: str, filt: str, member: str, t: datetime) -> Path:
    member_dir = base / exp / filt / member
    if not member_dir.exists():
        raise FileNotFoundError(f"Missing member directory: {member_dir}")

    pattern = f"wrfout_{MEMBER_DOMAIN}_{wrf_time_name(t)}"
    matches = sorted(member_dir.rglob(pattern))
    if not matches:
        matches = sorted(member_dir.rglob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(f"No wrfout file matching {pattern} under {member_dir}")
    return matches[0]


def find_nr_file(nr_base: Path, t: datetime) -> Path:
    if nr_base.is_file():
        return nr_base

    pattern = f"wrfout_{NR_DOMAIN}_{wrf_time_name(t)}"
    matches = sorted(nr_base.rglob(pattern))
    if not matches:
        matches = sorted(nr_base.rglob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(f"No NR file matching {pattern} under {nr_base}")
    return matches[0]


def isel_time0(da: xr.DataArray) -> xr.DataArray:
    for dim in da.dims:
        if dim.lower() == "time":
            return da.isel({dim: 0})
    return da


def infer_lat_lon_names(ds: xr.Dataset, data: xr.DataArray, variable: dict) -> tuple[str, str]:
    if variable["lat_name"] is not None and variable["lon_name"] is not None:
        return variable["lat_name"], variable["lon_name"]

    dims = set(data.dims)
    if "west_east_stag" in dims and {"XLAT_U", "XLONG_U"} <= set(ds.variables):
        return "XLAT_U", "XLONG_U"
    if "south_north_stag" in dims and {"XLAT_V", "XLONG_V"} <= set(ds.variables):
        return "XLAT_V", "XLONG_V"
    return "XLAT", "XLONG"


def read_lat_lon(ds: xr.Dataset, data: xr.DataArray, variable: dict) -> tuple[np.ndarray, np.ndarray]:
    lat_name, lon_name = infer_lat_lon_names(ds, data, variable)
    lats = isel_time0(ds[lat_name]).values.astype(float)
    lons = isel_time0(ds[lon_name]).values.astype(float)
    return lats, lons


def read_variable_2d(ds: xr.Dataset, variable: dict) -> xr.DataArray:
    name = variable["name"]
    vertical_level = variable["vertical_level"]
    if name not in ds:
        raise KeyError(f"{name} not found")

    data = isel_time0(ds[name])
    if data.ndim == 2:
        if vertical_level is not None:
            raise ValueError(
                f"{name} is 2-D after Time selection, but vertical_level={vertical_level} was configured."
            )
        return data
    if data.ndim == 3:
        if vertical_level is None:
            raise ValueError(f"{name} is 3-D after Time selection. Set vertical_level.")
        vertical_dim = data.dims[0]
        return data.isel({vertical_dim: vertical_level})

    raise ValueError(f"{name} has unsupported dimensions after Time selection: {data.dims}")


def variable_signature(variable: dict) -> tuple[str, int | None, float, float, str | None, str | None]:
    return (
        variable["name"],
        variable["vertical_level"],
        variable["scale"],
        variable.get("power", 1.0),
        variable["lat_name"],
        variable["lon_name"],
    )


def variable_from_signature(
    name: str,
    vertical_level: int | None,
    scale: float,
    power: float,
    lat_name: str | None,
    lon_name: str | None,
) -> dict:
    return {
        "name": name,
        "vertical_level": vertical_level,
        "scale": scale,
        "power": power,
        "lat_name": lat_name,
        "lon_name": lon_name,
    }


@lru_cache(maxsize=None)
def read_field_and_grid_cached(
    path_str: str,
    name: str,
    vertical_level: int | None,
    scale: float,
    power: float,
    lat_name: str | None,
    lon_name: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path_str)
    variable = variable_from_signature(name, vertical_level, scale, power, lat_name, lon_name)
    with xr.open_dataset(path, decode_times=False) as ds:
        data = read_variable_2d(ds, variable)
        lats, lons = read_lat_lon(ds, data, variable)
        values = (data.values.astype(float) * scale) ** power
    return lats, lons, values


def read_field_and_grid(path: Path, variable: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return read_field_and_grid_cached(str(path), *variable_signature(variable))


def coarsen_2d(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return values
    da = xr.DataArray(values, dims=("south_north", "west_east"))
    return da.coarsen(south_north=factor, west_east=factor, boundary="trim").mean().values


@lru_cache(maxsize=None)
def build_nr_interpolator_cached(
    path_str: str,
    name: str,
    vertical_level: int | None,
    scale: float,
    power: float,
    lat_name: str | None,
    lon_name: str | None,
    coarsen_factor: int,
) -> LinearNDInterpolator:
    lats, lons, values = read_field_and_grid_cached(
        path_str,
        name,
        vertical_level,
        scale,
        power,
        lat_name,
        lon_name,
    )

    lats = coarsen_2d(lats, coarsen_factor)
    lons = coarsen_2d(lons, coarsen_factor)
    values = coarsen_2d(values, coarsen_factor)

    points = np.column_stack((lons.ravel(), lats.ravel()))
    values_flat = values.ravel()
    valid = np.isfinite(points).all(axis=1) & np.isfinite(values_flat)
    if valid.sum() < 3:
        raise ValueError(f"Not enough valid NR {name} points in {path_str}")
    return LinearNDInterpolator(points[valid], values_flat[valid], fill_value=np.nan)


def interp_nr_to_member_grid(
    nr_file: Path,
    ens_lats: np.ndarray,
    ens_lons: np.ndarray,
    variable: dict,
) -> np.ndarray:
    interpolator = build_nr_interpolator_cached(
        str(nr_file),
        *variable_signature(variable),
        NR_COARSEN_FACTOR,
    )
    return np.asarray(interpolator(ens_lons, ens_lats), dtype=float)


def spatial_rmse(member_values: np.ndarray, nr_values: np.ndarray, mask: np.ndarray) -> float:
    diff = member_values[mask] - nr_values[mask]
    return float(np.sqrt(np.nanmean(diff**2)))


def unit_slug(variable: dict) -> str:
    return variable["unit"].replace(" ", "_").replace("/", "_")


def mean_member_rmse_column(variable: dict) -> str:
    return f"mean_member_rmse_{unit_slug(variable)}"


def median_member_rmse_column(variable: dict) -> str:
    return f"median_member_rmse_{unit_slug(variable)}"


def ensemble_mean_rmse_column(variable: dict) -> str:
    return f"ensemble_mean_rmse_{unit_slug(variable)}"


def calculate(times: list[datetime], variable: dict) -> pd.DataFrame:
    records = []
    nr_file_by_time = {wrf_time_name(t): find_nr_file(NR_BASE, t) for t in times}

    for exp in variable["experiments"]:
        for filt in FILTERS:
            print(f"Processing {exp}/{filt}")
            for t in times:
                time_name = wrf_time_name(t)
                nr_file = nr_file_by_time[time_name]

                member_rmses = []
                member_fields = []
                nr_fields = []
                overlap_points = []
                overlap_fraction = []

                for mem in MEMBERS:
                    member_file = find_member_file(BASE_DIR, exp, filt, mem, t)
                    ens_lats, ens_lons, ens_values = read_field_and_grid(member_file, variable)
                    nr_on_ens = interp_nr_to_member_grid(nr_file, ens_lats, ens_lons, variable)
                    mask = np.isfinite(nr_on_ens) & np.isfinite(ens_values)
                    n_overlap = int(mask.sum())
                    if n_overlap == 0:
                        raise ValueError(
                            f"No NR/member overlap after interpolation: NR={nr_file}, member={member_file}"
                        )

                    member_rmses.append(spatial_rmse(ens_values, nr_on_ens, mask))
                    member_fields.append(np.where(mask, ens_values, np.nan))
                    nr_fields.append(np.where(mask, nr_on_ens, np.nan))
                    overlap_points.append(n_overlap)
                    overlap_fraction.append(float(n_overlap / mask.size))

                ens_mean_field = np.nanmean(np.stack(member_fields, axis=0), axis=0)
                nr_mean_field = np.nanmean(np.stack(nr_fields, axis=0), axis=0)
                mean_mask = np.isfinite(ens_mean_field) & np.isfinite(nr_mean_field)

                records.append(
                    {
                        "time": time_name,
                        "experiment": exp,
                        "filter": filt,
                        mean_member_rmse_column(variable): float(np.nanmean(member_rmses)),
                        median_member_rmse_column(variable): float(np.nanmedian(member_rmses)),
                        ensemble_mean_rmse_column(variable): spatial_rmse(
                            ens_mean_field,
                            nr_mean_field,
                            mean_mask,
                        ),
                        "mean_overlap_points": float(np.nanmean(overlap_points)),
                        "mean_overlap_fraction": float(np.nanmean(overlap_fraction)),
                    }
                )

    return pd.DataFrame.from_records(records)


def variable_label(variable: dict) -> str:
    level = variable["vertical_level"]
    if level is None:
        return variable["name"]
    return f"{variable['name']} level {level}"


def rmse_column(variable: dict) -> str:
    return mean_member_rmse_column(variable)


def plot(df: pd.DataFrame, variable: dict) -> None:
    times = pd.to_datetime(df["time"], format="%Y-%m-%d_%H:%M:%S")
    df = df.assign(time_dt=times)
    metric = rmse_column(variable)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for exp in variable["experiments"]:
        for filt in FILTERS:
            sub = df[(df["experiment"] == exp) & (df["filter"] == filt)].sort_values("time_dt")
            ax.plot(
                sub["time_dt"],
                sub[metric],
                color=EXP_COLORS[exp],
                linestyle=FILTER_LINESTYLES[filt],
                linewidth=2.1,
                marker=FILTER_MARKERS[filt],
                markersize=3.5,
                label=f"{exp} / {filt}",
                dash_capstyle="butt",
            )

    ax.set_xlabel("Forecast time")
    ax.set_ylabel(f"Mean member RMSE of {variable_label(variable)} ({variable['unit']})")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="best", frameon=False, fontsize=9, handlelength=4.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    variable["out_png"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(variable["out_png"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    times = build_times(START_TIME, END_TIME, STEP_MINUTES)
    df = calculate(times, TARGET_VARIABLE)

    TARGET_VARIABLE["out_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TARGET_VARIABLE["out_csv"], index=False)
    plot(df, TARGET_VARIABLE)

    print(f"Saved {TARGET_VARIABLE['out_csv']}")
    print(f"Saved {TARGET_VARIABLE['out_png']}")


if __name__ == "__main__":
    main()
