"""
Plot ensemble-mean typhoon minimum sea level pressure error relative to NR.

By default the plotted metric is ensemble mean absolute error:
    mean_i | MSLP_i - MSLP_NR |

The CSV also includes ensemble mean signed bias:
    mean_i ( MSLP_i - MSLP_NR )

MSLP source priority:
    1. SLP variable in the wrfout file, assumed hPa unless values look like Pa.
    2. wrf-python getvar("slp"), if wrf-python is installed.
    3. Fallback to min(PSFC) / 100, with a warning.

For speed, use --mslp-source psfc. This avoids the expensive wrf-python SLP
diagnostic and is usually close to MSLP for an oceanic TC core.

The NR path and NR domain are set by DEFAULT_NR_BASE and DEFAULT_NR_DOMAIN
below, so the script can run without interactive/path arguments. Optional
command-line arguments remain available only for temporary overrides.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_BASE = Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test")
DEFAULT_NR_BASE = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout")
DEFAULT_NR_DOMAIN = "d03"
EXPERIMENTS = ["6mem_oceanAssim0Run0", "6mem_oceanAssim0Run1", "6mem_oceanAssim1Run1"]
FILTERS = ["EAKF", "QCF_RHF"]
MEMBERS = ["006", "015", "029", "037", "043", "044"]

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


def _slp_from_wrf_python(path: Path) -> np.ndarray | None:
    try:
        from netCDF4 import Dataset
        from wrf import getvar
    except Exception:
        return None

    try:
        with Dataset(str(path)) as nc:
            slp = getvar(nc, "slp", timeidx=0, meta=False)
        return np.asarray(slp, dtype=float)
    except Exception as exc:
        warnings.warn(f"wrf-python failed to diagnose SLP for {path}: {exc}")
        return None


def _read_slp_variable_hpa(path: Path) -> np.ndarray | None:
    with xr.open_dataset(path, decode_times=False) as ds:
        for name in ["SLP", "slp"]:
            if name in ds:
                arr = ds[name]
                vals = arr.isel({arr.dims[0]: 0}).values.astype(float) if arr.ndim == 3 else arr.values.astype(float)
                # Some files store pressure-like variables in Pa. Convert if needed.
                if np.nanmedian(vals) > 2000:
                    vals = vals / 100.0
                return vals
    return None


def _read_psfc_hpa(path: Path) -> np.ndarray:
    with xr.open_dataset(path, decode_times=False) as ds:
        if "PSFC" not in ds:
            raise KeyError(f"PSFC not found in {path}")
        arr = ds["PSFC"]
        vals = arr.isel({arr.dims[0]: 0}).values.astype(float) / 100.0
        return vals


def read_slp_hpa(path: Path, source: str) -> np.ndarray:
    if source in {"auto", "slp"}:
        slp = _read_slp_variable_hpa(path)
        if slp is not None:
            return slp
        if source == "slp":
            raise KeyError(f"SLP variable not found in {path}")

    if source in {"auto", "wrf"}:
        slp = _slp_from_wrf_python(path)
        if slp is not None:
            return slp
        if source == "wrf":
            raise RuntimeError(f"wrf-python could not diagnose SLP for {path}")

    if source in {"auto", "psfc"}:
        if source == "auto":
            warnings.warn(f"Using min(PSFC)/100 as MSLP fallback for {path}")
        return _read_psfc_hpa(path)

    raise ValueError(f"Unknown MSLP source: {source}")


@lru_cache(maxsize=None)
def min_mslp_hpa_cached(path_str: str, source: str) -> float:
    slp = read_slp_hpa(Path(path_str), source)
    return float(np.nanmin(slp))


def min_mslp_hpa(path: Path, source: str) -> float:
    return min_mslp_hpa_cached(str(path), source)


def calculate(
    base: Path,
    nr_base: Path,
    domain: str,
    nr_domain: str,
    times: list[datetime],
    mslp_source: str,
) -> pd.DataFrame:
    records = []
    nr_mslp_by_time = {}
    for t in times:
        nr_file = find_nr_file(nr_base, nr_domain, t)
        nr_mslp_by_time[wrf_time_name(t)] = min_mslp_hpa(nr_file, mslp_source)

    for exp in EXPERIMENTS:
        for filt in FILTERS:
            print(f"Processing {exp}/{filt}")
            for t in times:
                time_name = wrf_time_name(t)
                nr_mslp = nr_mslp_by_time[time_name]

                member_mslp = []
                for mem in MEMBERS:
                    fpath = find_member_file(base, exp, filt, mem, domain, t)
                    member_mslp.append(min_mslp_hpa(fpath, mslp_source))

                member_mslp = np.asarray(member_mslp, dtype=float)
                errors = member_mslp - nr_mslp
                records.append(
                    {
                        "time": time_name,
                        "experiment": exp,
                        "filter": filt,
                        "nr_mslp_hPa": nr_mslp,
                        "ens_mean_mslp_hPa": float(np.nanmean(member_mslp)),
                        "mean_error_hPa": float(np.nanmean(errors)),
                        "mean_abs_error_hPa": float(np.nanmean(np.abs(errors))),
                        "rmse_hPa": float(np.sqrt(np.nanmean(errors**2))),
                    }
                )
    return pd.DataFrame.from_records(records)


def plot(df: pd.DataFrame, out_png: Path, metric: str) -> None:
    times = pd.to_datetime(df["time"], format="%Y-%m-%d_%H:%M:%S")
    df = df.assign(time_dt=times)

    label_map = {
        "mean_abs_error_hPa": "Ensemble mean absolute error of minimum SLP (hPa)",
        "mean_error_hPa": "Ensemble mean signed error of minimum SLP (hPa)",
        "rmse_hPa": "RMSE of minimum SLP (hPa)",
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
    parser.add_argument(
        "--nr-base",
        type=Path,
        default=DEFAULT_NR_BASE,
        help="NR wrfout directory or a single NR wrfout file",
    )
    parser.add_argument("--domain", default="d02", help="Forecast member WRF domain")
    parser.add_argument("--nr-domain", default=DEFAULT_NR_DOMAIN, help="NR WRF domain, e.g. d02 or d03")
    parser.add_argument("--start", default="2018-09-10_00:00:00")
    parser.add_argument("--end", default="2018-09-10_06:00:00")
    parser.add_argument("--step-minutes", type=int, default=30)
    parser.add_argument(
        "--mslp-source",
        choices=["auto", "slp", "wrf", "psfc"],
        default="psfc",
        help="Source for minimum SLP: auto, existing SLP variable, wrf-python diagnostic, or fast PSFC/100 fallback",
    )
    parser.add_argument(
        "--metric",
        choices=["mean_abs_error_hPa", "mean_error_hPa", "rmse_hPa"],
        default="mean_abs_error_hPa",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("./figs/mslp_error_vs_nr_timeseries.csv"))
    parser.add_argument("--out-png", type=Path, default=Path("./figs/mslp_error_vs_nr_timeseries.png"))
    args = parser.parse_args()

    times = build_times(args.start, args.end, args.step_minutes)
    df = calculate(args.base, args.nr_base, args.domain, args.nr_domain, times, args.mslp_source)
    df.to_csv(args.out_csv, index=False)
    plot(df, args.out_png, args.metric)
    print(f"Saved {args.out_csv}")
    print(f"Saved {args.out_png}")


if __name__ == "__main__":
    main()
