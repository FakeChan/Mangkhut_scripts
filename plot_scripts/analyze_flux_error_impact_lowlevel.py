"""
Analyze how surface-flux RMSE relates to low-level atmospheric RMSE growth.

This script is built for the same experiment/member layout as
plot_cycle_rmse_times.py. It computes RMSE against the NR for one selected
surface flux, one selected low-level atmospheric variable, then diagnoses:

1. flux RMSE and low-level atmospheric RMSE time series;
2. lead-lag correlations: flux_error(t) vs atm_error(t + lag);
3. growth correlations: flux_error(t) vs atm_error(t + lag) - atm_error(t);
4. scatter plots for the configured lags.

Change FLUX_NAME to "QFX" or "HFX" to choose the surface-flux object.
Change ATM_VARIABLE and ATM_LEVEL to choose the low-level atmospheric response.
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


# =============================================================================
# User settings
# =============================================================================

BASE_DIR = Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test")
NR_BASE = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout")
MEMBER_DOMAIN = "d02"
NR_DOMAIN = "d03"
START_TIME = "2018-09-10_00:30:00"
END_TIME = "2018-09-10_06:00:00"
STEP_MINUTES = 30
NR_COARSEN_FACTOR = 5

EXPERIMENTS = ["6mem_oceanAssim0Run0", "6mem_oceanAssim0Run1", "6mem_oceanAssim1Run1"]
FILTERS = ["EAKF", "QCF_RHF"]
MEMBERS = ["006", "015", "029", "037", "043", "044"]

# Select the flux error source here: "QFX" or "HFX".
FLUX_NAME = "QFX"

# Select the low-level atmospheric response here.
# Examples:
#   ATM_VARIABLE = "QVAPOR"; ATM_LEVEL = 10
#   ATM_VARIABLE = "T";      ATM_LEVEL = 10
#   ATM_VARIABLE = "U";      ATM_LEVEL = 10
#   ATM_VARIABLE = "V";      ATM_LEVEL = 10
ATM_VARIABLE = "QVAPOR"
ATM_LEVEL = 10

# Lags are in output time steps. With STEP_MINUTES = 30:
#   lag=0 -> same time
#   lag=1 -> flux RMSE leads atmosphere RMSE by 30 min
#   lag=2 -> flux RMSE leads atmosphere RMSE by 60 min
LAGS = [0, 1, 2, 3]

# Use "mean_member_rmse" to ask whether individual-member flux errors relate to
# individual-member atmospheric errors. Use "ensemble_mean_rmse" for the error of
# the ensemble-mean field.
RMSE_METRIC = "mean_member_rmse"

OUT_DIR = Path("./figs/flux_error_impact_lowlevel")


FLUX_CONFIGS = {
    "QFX": {
        "name": "QFX",
        "vertical_level": None,
        "scale": 1.0,
        "power": 1.0,
        "post_scale": 1.0,
        "unit": "kg m-2 s-1",
        "label": "QFX",
    },
    "HFX": {
        "name": "HFX",
        "vertical_level": None,
        "scale": 1.0,
        "power": 1.0,
        "post_scale": 1.0,
        "unit": "W m-2",
        "label": "HFX",
    },
    "LH": {
        # Optional convenience target: latent heat flux derived from QFX.
        "name": "QFX",
        "vertical_level": None,
        "scale": 1.0,
        "power": 1.0,
        "post_scale": 2.5e6,
        "unit": "W m-2",
        "label": "LH = Lv * QFX",
    },
}

ATM_CONFIG = {
    "name": ATM_VARIABLE,
    "vertical_level": ATM_LEVEL,
    "scale": 1.0,
    "power": 1.0,
    "post_scale": 1.0,
    "unit": "native",
    "label": f"{ATM_VARIABLE} level {ATM_LEVEL}",
}

EXP_COLORS = {
    "6mem_oceanAssim0Run0": "#0072B2",
    "6mem_oceanAssim0Run1": "#D55E00",
    "6mem_oceanAssim1Run1": "#009E73",
}
FILTER_LINESTYLES = {
    "EAKF": "-",
    "QCF_RHF": (0, (5, 2)),
}
FILTER_MARKERS = {
    "EAKF": "o",
    "QCF_RHF": "s",
}


# =============================================================================
# File/time utilities
# =============================================================================

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


# =============================================================================
# WRF field reading and interpolation
# =============================================================================

def isel_time0(da: xr.DataArray) -> xr.DataArray:
    for dim in da.dims:
        if dim.lower() == "time":
            return da.isel({dim: 0})
    return da


def infer_lat_lon_names(ds: xr.Dataset, data: xr.DataArray, variable: dict) -> tuple[str, str]:
    if variable.get("lat_name") is not None and variable.get("lon_name") is not None:
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
            raise ValueError(f"{name} is 2-D after Time selection, but vertical_level={vertical_level}.")
        return data
    if data.ndim == 3:
        if vertical_level is None:
            raise ValueError(f"{name} is 3-D after Time selection. Set vertical_level.")
        vertical_dim = data.dims[0]
        return data.isel({vertical_dim: vertical_level})

    raise ValueError(f"{name} has unsupported dimensions after Time selection: {data.dims}")


def variable_signature(variable: dict) -> tuple[str, int | None, float, float, float, str | None, str | None]:
    return (
        variable["name"],
        variable["vertical_level"],
        variable.get("scale", 1.0),
        variable.get("power", 1.0),
        variable.get("post_scale", 1.0),
        variable.get("lat_name"),
        variable.get("lon_name"),
    )


def variable_from_signature(
    name: str,
    vertical_level: int | None,
    scale: float,
    power: float,
    post_scale: float,
    lat_name: str | None,
    lon_name: str | None,
) -> dict:
    return {
        "name": name,
        "vertical_level": vertical_level,
        "scale": scale,
        "power": power,
        "post_scale": post_scale,
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
    post_scale: float,
    lat_name: str | None,
    lon_name: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path_str)
    variable = variable_from_signature(name, vertical_level, scale, power, post_scale, lat_name, lon_name)
    with xr.open_dataset(path, decode_times=False) as ds:
        data = read_variable_2d(ds, variable)
        lats, lons = read_lat_lon(ds, data, variable)
        values = ((data.values.astype(float) * scale) ** power) * post_scale
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
    post_scale: float,
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
        post_scale,
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


def interp_nr_to_member_grid(nr_file: Path, ens_lats: np.ndarray, ens_lons: np.ndarray, variable: dict) -> np.ndarray:
    interpolator = build_nr_interpolator_cached(
        str(nr_file),
        *variable_signature(variable),
        NR_COARSEN_FACTOR,
    )
    return np.asarray(interpolator(ens_lons, ens_lats), dtype=float)


# =============================================================================
# RMSE and relationship diagnostics
# =============================================================================

def spatial_rmse(member_values: np.ndarray, nr_values: np.ndarray, mask: np.ndarray) -> float:
    diff = member_values[mask] - nr_values[mask]
    return float(np.sqrt(np.nanmean(diff**2)))


def nanmean_no_warning(stack: np.ndarray) -> np.ndarray:
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    total = np.nansum(stack, axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=float)
    np.divide(total, count, out=out, where=count > 0)
    return out


def rmse_for_variable(exp: str, filt: str, t: datetime, variable: dict, nr_file: Path) -> dict:
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
            raise ValueError(f"No NR/member overlap after interpolation: NR={nr_file}, member={member_file}")

        member_rmses.append(spatial_rmse(ens_values, nr_on_ens, mask))
        member_fields.append(np.where(mask, ens_values, np.nan))
        nr_fields.append(np.where(mask, nr_on_ens, np.nan))
        overlap_points.append(n_overlap)
        overlap_fraction.append(float(n_overlap / mask.size))

    ens_mean_field = nanmean_no_warning(np.stack(member_fields, axis=0))
    nr_mean_field = nanmean_no_warning(np.stack(nr_fields, axis=0))
    mean_mask = np.isfinite(ens_mean_field) & np.isfinite(nr_mean_field)

    return {
        "mean_member_rmse": float(np.nanmean(member_rmses)),
        "median_member_rmse": float(np.nanmedian(member_rmses)),
        "ensemble_mean_rmse": spatial_rmse(ens_mean_field, nr_mean_field, mean_mask),
        "mean_overlap_points": float(np.nanmean(overlap_points)),
        "mean_overlap_fraction": float(np.nanmean(overlap_fraction)),
    }


def calculate_rmse_timeseries(times: list[datetime], flux_variable: dict, atm_variable: dict) -> pd.DataFrame:
    nr_file_by_time = {wrf_time_name(t): find_nr_file(NR_BASE, t) for t in times}
    records = []

    for exp in EXPERIMENTS:
        for filt in FILTERS:
            print(f"Processing {exp}/{filt}")
            for t in times:
                time_name = wrf_time_name(t)
                nr_file = nr_file_by_time[time_name]
                flux_rmse = rmse_for_variable(exp, filt, t, flux_variable, nr_file)
                atm_rmse = rmse_for_variable(exp, filt, t, atm_variable, nr_file)

                records.append(
                    {
                        "time": time_name,
                        "experiment": exp,
                        "filter": filt,
                        "flux_name": flux_variable["label"],
                        "atm_name": atm_variable["label"],
                        "flux_mean_member_rmse": flux_rmse["mean_member_rmse"],
                        "flux_median_member_rmse": flux_rmse["median_member_rmse"],
                        "flux_ensemble_mean_rmse": flux_rmse["ensemble_mean_rmse"],
                        "atm_mean_member_rmse": atm_rmse["mean_member_rmse"],
                        "atm_median_member_rmse": atm_rmse["median_member_rmse"],
                        "atm_ensemble_mean_rmse": atm_rmse["ensemble_mean_rmse"],
                        "flux_mean_overlap_fraction": flux_rmse["mean_overlap_fraction"],
                        "atm_mean_overlap_fraction": atm_rmse["mean_overlap_fraction"],
                    }
                )

    df = pd.DataFrame.from_records(records)
    df["time_dt"] = pd.to_datetime(df["time"], format="%Y-%m-%d_%H:%M:%S")
    return df


def metric_columns() -> tuple[str, str]:
    if RMSE_METRIC not in ["mean_member_rmse", "ensemble_mean_rmse"]:
        raise ValueError("RMSE_METRIC must be 'mean_member_rmse' or 'ensemble_mean_rmse'")
    return f"flux_{RMSE_METRIC}", f"atm_{RMSE_METRIC}"


def corr_or_nan(x: pd.Series, y: pd.Series) -> float:
    valid = np.isfinite(x.values) & np.isfinite(y.values)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(x.values[valid], y.values[valid])[0, 1])


def calculate_lag_relationships(df: pd.DataFrame) -> pd.DataFrame:
    flux_col, atm_col = metric_columns()
    rows = []

    for exp in EXPERIMENTS:
        for filt in FILTERS:
            sub = df[(df["experiment"] == exp) & (df["filter"] == filt)].sort_values("time_dt").reset_index(drop=True)
            for lag in LAGS:
                shifted_atm = sub[atm_col].shift(-lag)
                atm_growth = shifted_atm - sub[atm_col]
                rows.append(
                    {
                        "experiment": exp,
                        "filter": filt,
                        "lag_steps": lag,
                        "lag_minutes": lag * STEP_MINUTES,
                        "flux_metric": flux_col,
                        "atm_metric": atm_col,
                        "corr_flux_vs_atm_lag": corr_or_nan(sub[flux_col], shifted_atm),
                        "corr_flux_vs_atm_growth": corr_or_nan(sub[flux_col], atm_growth),
                        "n_pairs": int((np.isfinite(sub[flux_col].values) & np.isfinite(shifted_atm.values)).sum()),
                    }
                )

    return pd.DataFrame.from_records(rows)


# =============================================================================
# Plotting
# =============================================================================

def plot_rmse_timeseries(df: pd.DataFrame, flux_variable: dict, atm_variable: dict, out_dir: Path) -> None:
    flux_col, atm_col = metric_columns()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)

    for exp in EXPERIMENTS:
        for filt in FILTERS:
            sub = df[(df["experiment"] == exp) & (df["filter"] == filt)].sort_values("time_dt")
            label = f"{exp} / {filt}"
            axes[0].plot(
                sub["time_dt"],
                sub[flux_col],
                color=EXP_COLORS[exp],
                linestyle=FILTER_LINESTYLES[filt],
                marker=FILTER_MARKERS[filt],
                linewidth=2.0,
                markersize=3.4,
                label=label,
            )
            axes[1].plot(
                sub["time_dt"],
                sub[atm_col],
                color=EXP_COLORS[exp],
                linestyle=FILTER_LINESTYLES[filt],
                marker=FILTER_MARKERS[filt],
                linewidth=2.0,
                markersize=3.4,
                label=label,
            )

    axes[0].set_ylabel(f"{flux_variable['label']} RMSE\n({flux_variable['unit']})")
    axes[1].set_ylabel(f"{atm_variable['label']} RMSE\n({atm_variable['unit']})")
    axes[1].set_xlabel("Forecast time")
    axes[0].set_title(f"Flux error and low-level atmospheric error ({RMSE_METRIC})")
    for ax in axes:
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    axes[0].legend(loc="best", frameon=False, fontsize=8, handlelength=4.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_timeseries_flux_and_atm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lag_correlations(lag_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)

    for exp in EXPERIMENTS:
        for filt in FILTERS:
            sub = lag_df[(lag_df["experiment"] == exp) & (lag_df["filter"] == filt)].sort_values("lag_minutes")
            label = f"{exp} / {filt}"
            axes[0].plot(
                sub["lag_minutes"],
                sub["corr_flux_vs_atm_lag"],
                color=EXP_COLORS[exp],
                linestyle=FILTER_LINESTYLES[filt],
                marker=FILTER_MARKERS[filt],
                linewidth=2.0,
                markersize=3.5,
                label=label,
            )
            axes[1].plot(
                sub["lag_minutes"],
                sub["corr_flux_vs_atm_growth"],
                color=EXP_COLORS[exp],
                linestyle=FILTER_LINESTYLES[filt],
                marker=FILTER_MARKERS[filt],
                linewidth=2.0,
                markersize=3.5,
                label=label,
            )

    axes[0].axhline(0.0, color="k", linewidth=0.8)
    axes[1].axhline(0.0, color="k", linewidth=0.8)
    axes[0].set_ylabel("corr flux(t), atm(t+lag)")
    axes[1].set_ylabel("corr flux(t), atm growth")
    axes[1].set_xlabel("Flux lead time (minutes)")
    axes[0].set_title("Lead-lag relationship between flux error and low-level atmospheric error")
    for ax in axes:
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    axes[0].legend(loc="best", frameon=False, fontsize=8, handlelength=4.0)
    fig.tight_layout()
    fig.savefig(out_dir / "leadlag_correlations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_for_lags(df: pd.DataFrame, out_dir: Path) -> None:
    flux_col, atm_col = metric_columns()

    for lag in LAGS:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
        for exp in EXPERIMENTS:
            for filt in FILTERS:
                sub = df[(df["experiment"] == exp) & (df["filter"] == filt)].sort_values("time_dt").reset_index(drop=True)
                shifted_atm = sub[atm_col].shift(-lag)
                atm_growth = shifted_atm - sub[atm_col]
                label = f"{exp} / {filt}"
                axes[0].scatter(
                    sub[flux_col],
                    shifted_atm,
                    s=28,
                    color=EXP_COLORS[exp],
                    marker=FILTER_MARKERS[filt],
                    alpha=0.75,
                    label=label,
                )
                axes[1].scatter(
                    sub[flux_col],
                    atm_growth,
                    s=28,
                    color=EXP_COLORS[exp],
                    marker=FILTER_MARKERS[filt],
                    alpha=0.75,
                    label=label,
                )

        axes[0].set_xlabel(f"{flux_col} at t")
        axes[0].set_ylabel(f"{atm_col} at t+{lag}")
        axes[1].set_xlabel(f"{flux_col} at t")
        axes[1].set_ylabel(f"{atm_col} growth to t+{lag}")
        axes[0].set_title(f"Flux error vs atmospheric error, lag={lag} steps")
        axes[1].set_title(f"Flux error vs atmospheric error growth, lag={lag} steps")
        for ax in axes:
            ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
        axes[0].legend(loc="best", frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_flux_vs_atm_lag{lag}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_summary(lag_df: pd.DataFrame, out_dir: Path) -> None:
    lines = []
    lines.append("Flux-error impact diagnostics")
    lines.append("=" * 72)
    lines.append(f"Flux target: {FLUX_NAME}")
    lines.append(f"Atmospheric target: {ATM_VARIABLE} level {ATM_LEVEL}")
    lines.append(f"RMSE metric: {RMSE_METRIC}")
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("  corr flux(t), atm(t+lag) > 0 means larger flux error tends to accompany larger later atmospheric error.")
    lines.append("  corr flux(t), atm growth > 0 means larger flux error tends to accompany stronger atmospheric error growth.")
    lines.append("")

    for _, row in lag_df.sort_values(["experiment", "filter", "lag_steps"]).iterrows():
        lines.append(
            f"{row['experiment']} / {row['filter']} / lag={int(row['lag_steps'])} "
            f"({int(row['lag_minutes'])} min): "
            f"corr_atm={row['corr_flux_vs_atm_lag']:.3f}, "
            f"corr_growth={row['corr_flux_vs_atm_growth']:.3f}, "
            f"n={int(row['n_pairs'])}"
        )

    path = out_dir / "diagnosis_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    if FLUX_NAME not in FLUX_CONFIGS:
        raise ValueError(f"FLUX_NAME must be one of {sorted(FLUX_CONFIGS)}")
    flux_variable = FLUX_CONFIGS[FLUX_NAME]
    atm_variable = ATM_CONFIG
    out_dir = OUT_DIR / f"{FLUX_NAME.lower()}_to_{ATM_VARIABLE.lower()}_lev{ATM_LEVEL}"
    out_dir.mkdir(parents=True, exist_ok=True)

    times = build_times(START_TIME, END_TIME, STEP_MINUTES)
    df = calculate_rmse_timeseries(times, flux_variable, atm_variable)
    lag_df = calculate_lag_relationships(df)

    rmse_csv = out_dir / "rmse_timeseries.csv"
    lag_csv = out_dir / "leadlag_correlations.csv"
    df.to_csv(rmse_csv, index=False)
    lag_df.to_csv(lag_csv, index=False)

    plot_rmse_timeseries(df, flux_variable, atm_variable, out_dir)
    plot_lag_correlations(lag_df, out_dir)
    plot_scatter_for_lags(df, out_dir)
    write_summary(lag_df, out_dir)

    print(f"Saved {rmse_csv}")
    print(f"Saved {lag_csv}")
    print(f"Saved figures under {out_dir}")


if __name__ == "__main__":
    main()
