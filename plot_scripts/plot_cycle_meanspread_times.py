"""
Plot domain-mean ensemble spread for configured WRF variables over d02.

Experiments:
    6mem_oceanAssim0Run0  Assim=0, Run=0
    6mem_oceanAssim0Run1  Assim=0, Run=1
    6mem_oceanAssim1Run1  Assim=1, Run=1

Filters:
    EAKF, QCF_RHF

The script recursively searches each member directory for wrfout_DOMAIN files.
It saves one CSV and one PNG figure per configured variable.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_BASE = Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test")
START_TIME = "2018-09-10_00:00:00"
END_TIME = "2018-09-10_06:00:00"
STEP_MINUTES = 30

EXPERIMENTS = ["6mem_oceanAssim0Run0", "6mem_oceanAssim0Run1", "6mem_oceanAssim1Run1"]
OCEAN_EXPERIMENTS = ["6mem_oceanAssim0Run1", "6mem_oceanAssim1Run1"]
FILTERS = ["EAKF", "QCF_RHF"]
MEMBERS = ["006", "015", "029", "037", "043", "044"]
DOMAIN = "d02"

OUT_CSV = Path("./figs/spread_timeseries.csv")

# Configure all variables here. No command-line arguments are used.
#
# vertical_level:
#     None -> variable must be 2-D after selecting Time, e.g. PSFC(Time,y,x).
#     int  -> variable must be 3-D after selecting Time, e.g. U(Time,z,y,x).
#
# scale converts the computed spread to the plotted/output unit. e.g. 100000Pa -> 1000hPa
VARIABLES = [
    # {
    #     "name": "OM_TMP",
    #     "vertical_level": 0,
    #     "scale": 1.0,
    #     "unit": "K",
    #     "experiments": OCEAN_EXPERIMENTS,
    #     "out_png": Path("./figs/spread_omtmp_timeseries.png"),
    # },
    # {
    #     "name": "PSFC",
    #     "vertical_level": None,
    #     "scale": 0.01,
    #     "unit": "hPa",
    #     "experiments": EXPERIMENTS,
    #     "out_png": Path("./figs/spread_psfc_timeseries.png"),
    # },
    {
        "name": "HFX",
        "vertical_level": None,
        "scale": 1,
        "unit": " ",
        "experiments": EXPERIMENTS,
        "out_png": Path("./figs/spread_hfx_timeseries.png"),
    },
    # Example for a 3-D atmospheric variable:
    # {
    #     "name": "U",
    #     "vertical_level": 10,
    #     "scale": 1.0,
    #     "unit": "m s-1",
    #     "experiments": EXPERIMENTS,
    #     "out_png": Path("./figs/spread_u_level10_timeseries.png"),
    # },
]

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

    pattern = f"wrfout_{DOMAIN}_{wrf_time_name(t)}"
    matches = sorted(member_dir.rglob(pattern))
    if not matches:
        matches = sorted(member_dir.rglob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(f"No wrfout file matching {pattern} under {member_dir}")
    return matches[0]


def time_dim_name(arr: xr.DataArray) -> str | None:
    for dim in arr.dims:
        if dim.lower() == "time":
            return dim
    return None


def read_variable_field(path: Path, variable: dict) -> np.ndarray:
    var_name = variable["name"]
    vertical_level = variable["vertical_level"]

    with xr.open_dataset(path, decode_times=False) as ds:
        if var_name not in ds:
            raise KeyError(f"{var_name} not found in {path}")
        arr = ds[var_name]
        selectors = {}
        tdim = time_dim_name(arr)
        if tdim is not None:
            selectors[tdim] = 0
        arr = arr.isel(selectors) if selectors else arr

        if arr.ndim == 2:
            if vertical_level is not None:
                raise ValueError(
                    f"{var_name} in {path} is 2-D after Time selection, "
                    f"but vertical_level={vertical_level} was configured."
                )
            values = arr.values.astype(float)
        elif arr.ndim == 3:
            if vertical_level is None:
                raise ValueError(
                    f"{var_name} in {path} is 3-D after Time selection. "
                    "Set vertical_level in VARIABLES."
                )
            vertical_dim = arr.dims[0]
            values = arr.isel({vertical_dim: vertical_level}).values.astype(float)
        else:
            raise ValueError(
                f"{var_name} in {path} has unsupported dimensions after Time selection: "
                f"{arr.dims}"
            )
    return values


def domain_mean_ensemble_spread(fields: list[np.ndarray]) -> float:
    stack = np.stack(fields, axis=0)
    grid_spread = np.nanstd(stack, axis=0, ddof=1)
    return float(np.nanmean(grid_spread))


def spread_column(variable: dict) -> str:
    suffix = variable["unit"].replace(" ", "_").replace("/", "_")
    level = variable["vertical_level"]
    level_part = "" if level is None else f"_lev{level}"
    return f"{variable['name'].lower()}{level_part}_spread_{suffix}"


def variable_label(variable: dict) -> str:
    level = variable["vertical_level"]
    if level is None:
        return variable["name"]
    return f"{variable['name']} level {level}"


def calculate(base: Path, times: list[datetime]) -> pd.DataFrame:
    records = []
    for exp in EXPERIMENTS:
        for filt in FILTERS:
            print(f"Processing {exp}/{filt}")
            for t in times:
                fields_by_metric = {spread_column(variable): [] for variable in VARIABLES}
                for mem in MEMBERS:
                    fpath = find_member_file(base, exp, filt, mem, t)
                    for variable in VARIABLES:
                        if exp not in variable["experiments"]:
                            continue
                        fields_by_metric[spread_column(variable)].append(
                            read_variable_field(fpath, variable)
                        )

                record = {
                    "time": wrf_time_name(t),
                    "experiment": exp,
                    "filter": filt,
                }
                for variable in VARIABLES:
                    column = spread_column(variable)
                    fields = fields_by_metric[column]
                    record[column] = (
                        domain_mean_ensemble_spread(fields) * variable["scale"]
                        if fields
                        else np.nan
                    )
                records.append(record)
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
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot(df: pd.DataFrame) -> None:
    for variable in VARIABLES:
        plot_one(
            df=df,
            out_png=variable["out_png"],
            metric=spread_column(variable),
            ylabel=f"Domain-mean ensemble spread of {variable_label(variable)} ({variable['unit']})",
            experiments=variable["experiments"],
        )


def main() -> None:
    times = build_times(START_TIME, END_TIME, STEP_MINUTES)
    df = calculate(DEFAULT_BASE, times)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    plot(df)

    print(f"Saved {OUT_CSV}")
    for variable in VARIABLES:
        print(f"Saved {variable['out_png']}")


if __name__ == "__main__":
    main()
