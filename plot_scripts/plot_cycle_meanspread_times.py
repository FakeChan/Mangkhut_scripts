"""
Plot ensemble spread of domain-mean OM_TMP(0, 0, :, :) and PSFC over d02.

Experiments:
    6mem_oceanAssim0Run0  Assim=0, Run=0
    6mem_oceanAssim0Run1  Assim=0, Run=1
    6mem_oceanAssim1Run1  Assim=1, Run=1

Filters:
    EAKF, QCF_RHF

The script recursively searches each member directory for wrfout_d02 files.
It saves a CSV and a PNG figure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_BASE = Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test")
EXPERIMENTS = ["6mem_oceanAssim0Run0", "6mem_oceanAssim0Run1", "6mem_oceanAssim1Run1"]
OCEAN_EXPERIMENTS = ["6mem_oceanAssim0Run1", "6mem_oceanAssim1Run1"]
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


def find_member_file(base: Path, exp: str, filt: str, member: str, t: datetime) -> Path:
    member_dir = base / exp / filt / member
    if not member_dir.exists():
        raise FileNotFoundError(f"Missing member directory: {member_dir}")

    pattern = f"wrfout_d02_{wrf_time_name(t)}"
    matches = sorted(member_dir.rglob(pattern))
    if not matches:
        matches = sorted(member_dir.rglob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(f"No wrfout file matching {pattern} under {member_dir}")
    return matches[0]


def read_om_tmp_surface(path: Path) -> np.ndarray:
    with xr.open_dataset(path, decode_times=False) as ds:
        if "OM_TMP" not in ds:
            raise KeyError(f"OM_TMP not found in {path}")
        arr = ds["OM_TMP"]
        # Requested variable: OM_TMP(0, 0, :, :).
        # Works for common WRF ocean dims: Time, ocean_layer_stag/ocean_layer, y, x.
        values = arr.isel({arr.dims[0]: 0, arr.dims[1]: 0}).values.astype(float)
    return values


def read_psfc(path: Path) -> np.ndarray:
    with xr.open_dataset(path, decode_times=False) as ds:
        if "PSFC" not in ds:
            raise KeyError(f"PSFC not found in {path}")
        arr = ds["PSFC"]
        values = arr.isel({arr.dims[0]: 0}).values.astype(float)
    return values


def domain_mean_ensemble_spread(fields: list[np.ndarray]) -> float:
    stack = np.stack(fields, axis=0)
    grid_spread = np.nanstd(stack, axis=0, ddof=1)
    return float(np.nanmean(grid_spread))


def calculate(base: Path, times: list[datetime]) -> pd.DataFrame:
    records = []
    for exp in EXPERIMENTS:
        for filt in FILTERS:
            print(f"Processing {exp}/{filt}")
            for t in times:
                om_fields = []
                psfc_fields = []
                for mem in MEMBERS:
                    fpath = find_member_file(base, exp, filt, mem, t)
                    if exp in OCEAN_EXPERIMENTS:
                        om_fields.append(read_om_tmp_surface(fpath))
                    psfc_fields.append(read_psfc(fpath))

                records.append(
                    {
                        "time": wrf_time_name(t),
                        "experiment": exp,
                        "filter": filt,
                        "om_tmp_spread_K": (
                            domain_mean_ensemble_spread(om_fields)
                            if exp in OCEAN_EXPERIMENTS
                            else np.nan
                        ),
                        "psfc_spread_Pa": domain_mean_ensemble_spread(psfc_fields),
                        "psfc_spread_hPa": domain_mean_ensemble_spread(psfc_fields) / 100.0,
                    }
                )
    return pd.DataFrame.from_records(records)


def plot_one(
    df: pd.DataFrame,
    out_png: Path,
    metric: str,
    ylabel: str,
    experiments: list[str],
) -> None:
    times = pd.to_datetime(df["time"], format="%Y-%m-%d_%H:%M:%S")
    df = df.assign(time_dt=times)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for exp in experiments:
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
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot(df: pd.DataFrame, out_om_png: Path, out_psfc_png: Path) -> None:
    plot_one(
        df=df,
        out_png=out_om_png,
        metric="om_tmp_spread_K",
        ylabel="Domain-mean ensemble spread of OM_TMP(0,0,:,:) (K)",
        experiments=OCEAN_EXPERIMENTS,
    )
    plot_one(
        df=df,
        out_png=out_psfc_png,
        metric="psfc_spread_hPa",
        ylabel="Domain-mean ensemble spread of PSFC (hPa)",
        experiments=EXPERIMENTS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--start", default="2018-09-10_00:00:00")
    parser.add_argument("--end", default="2018-09-10_06:00:00")
    parser.add_argument("--step-minutes", type=int, default=30)
    parser.add_argument("--out-csv", type=Path, default=Path("./figs/spread_omtmp_psfc_timeseries.csv"))
    parser.add_argument("--out-om-png", type=Path, default=Path("./figs/spread_omtmp_timeseries.png"))
    parser.add_argument("--out-psfc-png", type=Path, default=Path("./figs/spread_psfc_timeseries.png"))
    args = parser.parse_args()

    times = build_times(args.start, args.end, args.step_minutes)
    df = calculate(args.base, times)
    df.to_csv(args.out_csv, index=False)
    plot(df, args.out_om_png, args.out_psfc_png)
    print(f"Saved {args.out_csv}")
    print(f"Saved {args.out_om_png}")
    print(f"Saved {args.out_psfc_png}")


if __name__ == "__main__":
    main()