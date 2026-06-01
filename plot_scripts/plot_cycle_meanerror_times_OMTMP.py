"""
Plot domain-average OM_TMP(0, 0, :, :) error relative to NR.

Metric:
    ensemble mean error =
        mean_i(mean_over_overlap(OM_TMP_i - interp(NR OM_TMP to member d02 grid)))

The CSV also includes mean absolute error and RMSE across ensemble members.
The experiment 6mem_oceanAssim0Run0 is skipped because it does not run the
ocean model and does not contain OM_TMP.

NR d03 is finer than, and smaller than, the ensemble d02 domain. Therefore the
script linearly interpolates NR OM_TMP to each member's d02 grid and computes
the error only over d02 points inside the NR d03 linear-interpolation footprint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import LinearNDInterpolator


DEFAULT_BASE = Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test")
DEFAULT_NR_BASE = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout")
DEFAULT_NR_DOMAIN = "d03"
DOMAIN = "d02"
EXPERIMENTS = ["6mem_oceanAssim0Run1", "6mem_oceanAssim1Run1"]
FILTERS = ["EAKF", "QCF_RHF"]
MEMBERS = ["006", "015", "029", "037", "043", "044"]

# Okabe-Ito / colorblind-safe scientific palette.
EXP_COLORS = {
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


def find_member_file(base: Path, exp: str, filt: str, member: str, domain: str, t: datetime) -> Path:
    member_dir = base / exp / filt / member
    if not member_dir.exists():
        raise FileNotFoundError(f"Missing member directory: {member_dir}")

    pattern = f"wrfout_{domain}_{wrf_time_name(t)}"
    matches = sorted(member_dir.rglob(pattern))
    if not matches:
        matches = sorted(member_dir.rglob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(f"No wrfout file matching {pattern} under {member_dir}")
    return matches[0]


def find_nr_file(nr_base: Path, domain: str, t: datetime) -> Path:
    if nr_base.is_file():
        return nr_base

    pattern = f"wrfout_{domain}_{wrf_time_name(t)}"
    matches = sorted(nr_base.rglob(pattern))
    if not matches:
        matches = sorted(nr_base.rglob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(f"No NR file matching {pattern} under {nr_base}")
    return matches[0]


def _isel_time0(da: xr.DataArray) -> xr.DataArray:
    if "Time" in da.dims:
        return da.isel(Time=0)
    return da


def _read_lat_lon(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lats = _isel_time0(ds["XLAT"]).values.astype(float)
    lons = _isel_time0(ds["XLONG"]).values.astype(float)
    return lats, lons


def _read_omtmp_surface(ds: xr.Dataset) -> np.ndarray:
    if "OM_TMP" not in ds:
        raise KeyError("OM_TMP not found")

    arr = ds["OM_TMP"]
    indexers = {}
    if "Time" in arr.dims:
        indexers["Time"] = 0
    for dim in arr.dims:
        if dim != "Time" and ("ocean_layer" in dim or dim in {"bottom_top", "bottom_top_stag"}):
            indexers[dim] = 0
            break
    if len(indexers) < (2 if "Time" in arr.dims else 1):
        # Fall back to the original requested position OM_TMP(0, 0, :, :).
        values = arr.isel({arr.dims[0]: 0, arr.dims[1]: 0}).values.astype(float)
    else:
        values = arr.isel(indexers).values.astype(float)
    return values


@lru_cache(maxsize=None)
def read_member_omtmp_and_grid_cached(path_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path_str)
    with xr.open_dataset(path, decode_times=False) as ds:
        lats, lons = _read_lat_lon(ds)
        values = _read_omtmp_surface(ds)
    return lats, lons, values


def read_member_omtmp_and_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return read_member_omtmp_and_grid_cached(str(path))


def _coarsen_2d(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return values
    da = xr.DataArray(values, dims=("south_north", "west_east"))
    return da.coarsen(south_north=factor, west_east=factor, boundary="trim").mean().values


@lru_cache(maxsize=None)
def build_nr_interpolator_cached(path_str: str, coarsen_factor: int) -> LinearNDInterpolator:
    path = Path(path_str)
    with xr.open_dataset(path, decode_times=False) as ds:
        nr_lats, nr_lons = _read_lat_lon(ds)
        nr_values = _read_omtmp_surface(ds)

    nr_lats = _coarsen_2d(nr_lats, coarsen_factor)
    nr_lons = _coarsen_2d(nr_lons, coarsen_factor)
    nr_values = _coarsen_2d(nr_values, coarsen_factor)

    points = np.column_stack((nr_lons.ravel(), nr_lats.ravel()))
    values = nr_values.ravel()
    valid = np.isfinite(points).all(axis=1) & np.isfinite(values)
    if valid.sum() < 3:
        raise ValueError(f"Not enough valid NR OM_TMP points in {path}")
    return LinearNDInterpolator(points[valid], values[valid], fill_value=np.nan)


def interp_nr_to_member_grid(
    nr_file: Path,
    ens_lats: np.ndarray,
    ens_lons: np.ndarray,
    nr_coarsen: int,
) -> np.ndarray:
    interpolator = build_nr_interpolator_cached(str(nr_file), nr_coarsen)
    return np.asarray(interpolator(ens_lons, ens_lats), dtype=float)


def calculate(base: Path, nr_base: Path, nr_domain: str, times: list[datetime], nr_coarsen: int) -> pd.DataFrame:
    records = []

    nr_file_by_time = {}
    for t in times:
        time_name = wrf_time_name(t)
        nr_file_by_time[time_name] = find_nr_file(nr_base, nr_domain, t)

    for exp in EXPERIMENTS:
        for filt in FILTERS:
            print(f"Processing {exp}/{filt}")
            for t in times:
                time_name = wrf_time_name(t)
                nr_file = nr_file_by_time[time_name]

                member_ens_mean = []
                member_nr_mean = []
                member_error = []
                overlap_points = []
                overlap_fraction = []
                for mem in MEMBERS:
                    fpath = find_member_file(base, exp, filt, mem, DOMAIN, t)
                    ens_lats, ens_lons, ens_values = read_member_omtmp_and_grid(fpath)
                    nr_on_ens = interp_nr_to_member_grid(nr_file, ens_lats, ens_lons, nr_coarsen)
                    mask = np.isfinite(nr_on_ens) & np.isfinite(ens_values)
                    n_overlap = int(mask.sum())
                    if n_overlap == 0:
                        raise ValueError(
                            f"No NR/member overlap after interpolation: NR={nr_file}, member={fpath}"
                        )

                    member_ens_mean.append(float(np.nanmean(ens_values[mask])))
                    member_nr_mean.append(float(np.nanmean(nr_on_ens[mask])))
                    member_error.append(float(np.nanmean(ens_values[mask] - nr_on_ens[mask])))
                    overlap_points.append(n_overlap)
                    overlap_fraction.append(float(n_overlap / mask.size))

                member_ens_mean = np.asarray(member_ens_mean, dtype=float)
                member_nr_mean = np.asarray(member_nr_mean, dtype=float)
                member_error = np.asarray(member_error, dtype=float)
                records.append(
                    {
                        "time": time_name,
                        "experiment": exp,
                        "filter": filt,
                        "nr_overlap_mean_omtmp_K": float(np.nanmean(member_nr_mean)),
                        "ens_overlap_mean_omtmp_K": float(np.nanmean(member_ens_mean)),
                        "mean_error_K": float(np.nanmean(member_error)),
                        "mean_abs_error_K": float(np.nanmean(np.abs(member_error))),
                        "rmse_K": float(np.sqrt(np.nanmean(member_error**2))),
                        "mean_overlap_points": float(np.nanmean(overlap_points)),
                        "mean_overlap_fraction": float(np.nanmean(overlap_fraction)),
                    }
                )

    return pd.DataFrame.from_records(records)


def plot(df: pd.DataFrame, out_png: Path, metric: str) -> None:
    times = pd.to_datetime(df["time"], format="%Y-%m-%d_%H:%M:%S")
    df = df.assign(time_dt=times)

    label_map = {
        "mean_error_K": "Ensemble mean domain-average OM_TMP error (K)",
        "mean_abs_error_K": "Ensemble mean absolute domain-average OM_TMP error (K)",
        "rmse_K": "RMSE of domain-average OM_TMP (K)",
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for exp in EXPERIMENTS:
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

    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xlabel("Forecast time")
    ax.set_ylabel(label_map[metric])
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="best", frameon=False, fontsize=9, handlelength=4.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--nr-base", type=Path, default=DEFAULT_NR_BASE)
    parser.add_argument("--nr-domain", default=DEFAULT_NR_DOMAIN, help="NR WRF domain, e.g. d02 or d03")
    parser.add_argument("--start", default="2018-09-10_00:00:00")
    parser.add_argument("--end", default="2018-09-10_06:00:00")
    parser.add_argument("--step-minutes", type=int, default=30)
    parser.add_argument(
        "--nr-coarsen",
        type=int,
        default=5,
        help="Optionally coarsen NR d03 by this grid factor before interpolation, e.g. 5 for 300 m -> 1.5 km",
    )
    parser.add_argument(
        "--metric",
        choices=["mean_error_K", "mean_abs_error_K", "rmse_K"],
        default="mean_error_K",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("./figs/omtmp_domain_mean_error_vs_nr_timeseries.csv"))
    parser.add_argument("--out-png", type=Path, default=Path("./figs/omtmp_domain_mean_error_vs_nr_timeseries.png"))
    args = parser.parse_args()

    times = build_times(args.start, args.end, args.step_minutes)
    df = calculate(args.base, args.nr_base, args.nr_domain, times, args.nr_coarsen)
    df.to_csv(args.out_csv, index=False)
    plot(df, args.out_png, args.metric)
    print(f"Saved {args.out_csv}")
    print(f"Saved {args.out_png}")


if __name__ == "__main__":
    main()