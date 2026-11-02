"""
Plot 6-member ensemble spread inside a 150-km radius of the NR TC centre.

Ensemble members (one directory per experiment / filter / member):
    /scratch/lililei1/kcfu/tc_mangkhut/cycle_test/
        {experiment}/{filter}/{member}/wrfout_d01_{YYYY-MM-DD_HH:MM:SS}

NR (truth):
    /share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain/
        wrfout_d01_{YYYY-MM-DD_HH:MM:SS}

Method
------
1. Locate the NR TC centre as the grid point of minimum PSFC in the NR d01
   file (same convention as diag_tc_3d_rmse_times.py).
2. Extract a d01 (7.5 km) nest around that centre that is slightly wider than
   TC_RADIUS_KM, so the 150-km circle is never clipped by the panel edge.
3. Read every ensemble member on d01 and interpolate its field linearly to the
   NR nest grid (direct slicing is used when the member grid is identical to
   the NR grid).
4. Compute the ensemble standard deviation at every grid point with ddof=1,
   then mask points farther than TC_RADIUS_KM from the NR centre.

Output (one figure per variable; rows = times, columns = experiment x filter):
    ./figs/tc150km_spread/{var}_tc150km_ens_spread.png
    ./figs/tc150km_spread/tc150km_ens_spread_area_mean.csv

All settings are configured at the top of this file; no command-line
arguments are parsed.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata

# =============================================================================
# User settings
# =============================================================================

START_TIME = datetime(2018, 9, 10, 0, 0)
END_TIME = datetime(2018, 9, 10, 6, 0)
TIME_STEP = timedelta(hours=1)

FILTERS = ["EAKF", "QCF_RHF"]

MEMBER_DOMAIN = "d01"
NR_DOMAIN = "d01"

CYCLE_BASE = Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test")
NR_DIR = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain")

MEMBERS = ["006", "015", "029", "037", "043", "044"]

# Column order in the figure follows this list.
EXPERIMENTS = [
    {"dir": "6mem_oceanAssim1Run1", "label": "Strong coupling"},
    {"dir": "6mem_oceanAssim0Run1", "label": "Weak coupling"},
    {"dir": "6mem_oceanAssim0Run0", "label": "No coupling"},
]

# Add or remove variables here.  Only 2-D mass-grid wrfout variables are
# supported by this script (TSK, HFX, SST, PSFC, ...).  Use "scale" for unit
# conversion, e.g. PSFC: scale=0.01 -> hPa.
TARGET_VARIABLES = [
    {"name": "TSK", "label": "TSK", "scale": 1.0, "unit": "K"},
    {"name": "HFX", "label": "HFX", "scale": 1.0, "unit": "W m-2"},
]

# Radius defining the plotted/statistical region.
TC_RADIUS_KM = 150.0
# Half-width of the extracted d01 nest.  Keep it a little larger than
# TC_RADIUS_KM so the circle is fully inside the panel.
HALF_WIDTH_KM = 165.0

# NR TC centre is the minimum of this 2-D field.
CENTER_PRESSURE_VAR = "PSFC"

SPREAD_PERCENTILE = 99.5  # upper colour-limit percentile over all panels
N_COLOR_LEVELS = 21
OUTPUT_DIR = Path("./figs/tc150km_spread")


# =============================================================================
# Time and file helpers
# =============================================================================

def build_times(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    times = []
    current = start
    while current <= end:
        times.append(current)
        current += step
    return times


def wrf_time_name(time_obj: datetime) -> str:
    return time_obj.strftime("%Y-%m-%d_%H:%M:%S")


def find_file(directory: Path, filename: str) -> Path:
    direct = directory / filename
    if direct.exists():
        return direct

    matches = sorted(directory.rglob(filename))
    if not matches:
        matches = sorted(directory.rglob(f"{filename}*"))
    if not matches:
        raise FileNotFoundError(f"No file matching {filename!r} under {directory}")
    return matches[0]


def find_nr_file(time_obj: datetime) -> Path:
    filename = f"wrfout_{NR_DOMAIN}_{wrf_time_name(time_obj)}"
    return find_file(NR_DIR, filename)


def find_member_file(exp_dir: str, filter_kind: str, member: str, time_obj: datetime) -> Path:
    directory = CYCLE_BASE / exp_dir / filter_kind / member
    if not directory.exists():
        raise FileNotFoundError(f"Missing member directory: {directory}")
    filename = f"wrfout_{MEMBER_DOMAIN}_{wrf_time_name(time_obj)}"
    return find_file(directory, filename)


# =============================================================================
# NetCDF readers
# =============================================================================

def isel_time0(data_array: xr.DataArray) -> xr.DataArray:
    for dim in data_array.dims:
        if dim.lower() == "time":
            return data_array.isel({dim: 0})
    return data_array


def read_lat_lon(ds: xr.Dataset, path: Path) -> tuple[np.ndarray, np.ndarray]:
    for name in ("XLAT", "XLONG"):
        if name not in ds:
            raise KeyError(f"{name} not found in {path}")

    lats = np.asarray(isel_time0(ds["XLAT"]).values, dtype=float)
    lons = np.asarray(isel_time0(ds["XLONG"]).values, dtype=float)
    if lats.ndim != 2 or lons.ndim != 2:
        raise ValueError(f"XLAT/XLONG should be 2-D after time selection: {path}")
    return lats, lons


def read_2d_field(ds: xr.Dataset, name: str, path: Path) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"{name} not found in {path}")

    data = isel_time0(ds[name])
    if data.ndim != 2:
        raise ValueError(
            f"{name} should be 2-D after time selection, got dims={data.dims} in {path}. "
            "This script only supports surface/2-D variables."
        )
    return np.asarray(data.values, dtype=float)


def read_nr_state(path: Path) -> dict:
    """Read the NR grid, the 150-km centre, and the d01 grid spacing."""
    with xr.open_dataset(path, decode_times=False) as ds:
        lats, lons = read_lat_lon(ds, path)
        psfc = read_2d_field(ds, CENTER_PRESSURE_VAR, path)
        dx_m = ds.attrs.get("DX", None)

    if psfc.shape != lats.shape:
        raise ValueError(
            f"{CENTER_PRESSURE_VAR} shape {psfc.shape} does not match XLAT shape "
            f"{lats.shape} in {path}"
        )

    try:
        dx_m = float(dx_m)
    except (TypeError, ValueError):
        dx_m = np.nan
    if not np.isfinite(dx_m) or dx_m <= 0:
        print(f"  [warn] DX not found or invalid in {path}; assuming 7500 m")
        dx_m = 7500.0
    elif not np.isclose(dx_m, 7500.0, atol=0.1):
        print(f"  [warn] NR DX={dx_m:.1f} m (expected 7500 m for d01)")

    centre_idx = np.unravel_index(np.nanargmin(psfc), psfc.shape)
    return {
        "lats": lats,
        "lons": lons,
        "centre_idx": centre_idx,
        "centre_lat": float(lats[centre_idx]),
        "centre_lon": float(lons[centre_idx]),
        "centre_psfc": float(psfc[centre_idx]),
        "dx_m": dx_m,
    }


# =============================================================================
# Geometry helpers
# =============================================================================

def nest_slices(shape: tuple[int, int], centre_idx: tuple[int, int], dx_m: float) -> tuple[int, int, int, int]:
    """Return i0, i1, j0, j1 for a nest of +-HALF_WIDTH_KM around the centre."""
    ny, nx = shape
    jc, ic = centre_idx
    half = int(np.ceil(HALF_WIDTH_KM * 1000.0 / dx_m))

    i0 = max(0, ic - half)
    i1 = min(nx, ic + half + 1)
    j0 = max(0, jc - half)
    j1 = min(ny, jc + half + 1)

    if (i0, i1, j0, j1) != (ic - half, ic + half + 1, jc - half, jc + half + 1):
        print(
            f"  [warn] nest clipped by domain boundary: "
            f"i=[{i0},{i1}) j=[{j0},{j1}) centre=({jc},{ic})"
        )
    return i0, i1, j0, j1


def latlon_distance_km(
    lats: np.ndarray, lons: np.ndarray, centre_lat: float, centre_lon: float
) -> np.ndarray:
    """Great-circle distance in km (Haversine)."""
    earth_radius_km = 6371.0
    lat1 = np.deg2rad(lats)
    lon1 = np.deg2rad(lons)
    lat2 = np.deg2rad(centre_lat)
    lon2 = np.deg2rad(centre_lon)

    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


# =============================================================================
# Interpolation and spread
# =============================================================================

def interp_member_to_nest(
    member_lats: np.ndarray,
    member_lons: np.ndarray,
    member_values: np.ndarray,
    nr_lats: np.ndarray,
    nr_lons: np.ndarray,
    nest_lats: np.ndarray,
    nest_lons: np.ndarray,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
) -> np.ndarray:
    """
    Interpolate a member field to the NR nest.

    When the member grid is identical to the NR d01 grid, the nest is simply a
    slice of the member array, which avoids an expensive Delaunay triangulation.
    """
    same_shape = member_lats.shape == nr_lats.shape
    if (
        same_shape
        and np.allclose(member_lats, nr_lats, rtol=0.0, atol=1e-5, equal_nan=True)
        and np.allclose(member_lons, nr_lons, rtol=0.0, atol=1e-5, equal_nan=True)
    ):
        return member_values[j0:j1, i0:i1]

    points = np.column_stack((member_lons.ravel(), member_lats.ravel()))
    values = member_values.ravel()
    valid = np.isfinite(points).all(axis=1) & np.isfinite(values)
    if valid.sum() < 3:
        raise ValueError("Not enough valid member points for interpolation")

    result = griddata(
        points[valid],
        values[valid],
        (nest_lons, nest_lats),
        method="linear",
    )
    if not np.isfinite(result).any():
        result = griddata(
            points[valid],
            values[valid],
            (nest_lons, nest_lats),
            method="nearest",
        )
    return np.asarray(result, dtype=float)


def ensemble_spread(fields: list[np.ndarray]) -> np.ndarray:
    """Ensemble standard deviation with ddof=1, without all-NaN warnings."""
    stack = np.stack(fields, axis=0)
    valid = np.isfinite(stack)
    counts = valid.sum(axis=0)

    safe_counts = np.maximum(counts, 1)
    mean = np.where(valid, stack, 0.0).sum(axis=0) / safe_counts
    deviation = np.where(valid, stack - mean, 0.0)
    variance = (deviation * deviation).sum(axis=0) / np.maximum(counts - 1, 1)

    spread = np.sqrt(variance)
    spread[counts < 2] = np.nan
    return spread


# =============================================================================
# Data collection
# =============================================================================

def collect(times: list[datetime]) -> tuple[dict, dict, list[dict]]:
    """
    Returns
    -------
    spread_data : {var_name: {time: {(exp_dir, filter): 2-D spread}}}
    time_grids  : {time: {"x_km", "y_km", "mask", ...}}
    records     : area-mean spread rows for the CSV
    """
    spread_data = {variable["name"]: {t: {} for t in times} for variable in TARGET_VARIABLES}
    time_grids: dict[datetime, dict] = {}
    records: list[dict] = []

    for time_obj in times:
        time_name = wrf_time_name(time_obj)
        print(f"\n>>> {time_name}")

        nr_file = find_nr_file(time_obj)
        nr = read_nr_state(nr_file)
        i0, i1, j0, j1 = nest_slices(nr["lats"].shape, nr["centre_idx"], nr["dx_m"])

        nest_lats = nr["lats"][j0:j1, i0:i1]
        nest_lons = nr["lons"][j0:j1, i0:i1]
        distance_km = latlon_distance_km(
            nest_lats,
            nest_lons,
            nr["centre_lat"],
            nr["centre_lon"],
        )
        tc_mask = distance_km <= TC_RADIUS_KM

        if not np.any(tc_mask):
            raise ValueError(
                f"No nest point within {TC_RADIUS_KM:g} km of the NR centre at {time_name}"
            )

        jc, ic = nr["centre_idx"]
        x_km = (np.arange(i0, i1) - ic) * nr["dx_m"] / 1000.0
        y_km = (np.arange(j0, j1) - jc) * nr["dx_m"] / 1000.0

        time_grids[time_obj] = {
            "x_km": x_km,
            "y_km": y_km,
            "mask": tc_mask,
            "distance_km": distance_km,
            "centre_lat": nr["centre_lat"],
            "centre_lon": nr["centre_lon"],
            "centre_psfc": nr["centre_psfc"],
            "nr_file": str(nr_file),
            "nest_shape": nest_lats.shape,
        }

        print(
            f"  NR centre: lat={nr['centre_lat']:.4f}, lon={nr['centre_lon']:.4f}, "
            f"min {CENTER_PRESSURE_VAR}={nr['centre_psfc']:.2f}; "
            f"nest={nest_lats.shape}, {int(tc_mask.sum())} points inside "
            f"{TC_RADIUS_KM:g} km"
        )

        for exp in EXPERIMENTS:
            for filter_kind in FILTERS:
                fields_by_var: dict[str, list[np.ndarray]] = {
                    variable["name"]: [] for variable in TARGET_VARIABLES
                }
                used_members: list[str] = []

                for member in MEMBERS:
                    try:
                        member_file = find_member_file(
                            exp["dir"], filter_kind, member, time_obj
                        )
                        with xr.open_dataset(member_file, decode_times=False) as ds:
                            member_lats, member_lons = read_lat_lon(ds, member_file)
                            member_fields = {}
                            for variable in TARGET_VARIABLES:
                                try:
                                    member_fields[variable["name"]] = read_2d_field(
                                        ds, variable["name"], member_file
                                    )
                                except Exception as exc:
                                    print(
                                        f"  [skip var] {exp['dir']}/{filter_kind}/{member} "
                                        f"{variable['name']}: {exc}"
                                    )
                    except Exception as exc:
                        print(
                            f"  [skip member] {exp['dir']}/{filter_kind}/{member}: {exc}"
                        )
                        continue

                    member_used = False
                    for variable in TARGET_VARIABLES:
                        name = variable["name"]
                        if name not in member_fields:
                            continue
                        try:
                            nest_values = interp_member_to_nest(
                                member_lats=member_lats,
                                member_lons=member_lons,
                                member_values=member_fields[name] * variable["scale"],
                                nr_lats=nr["lats"],
                                nr_lons=nr["lons"],
                                nest_lats=nest_lats,
                                nest_lons=nest_lons,
                                i0=i0,
                                i1=i1,
                                j0=j0,
                                j1=j1,
                            )
                        except Exception as exc:
                            print(
                                f"  [skip interp] {exp['dir']}/{filter_kind}/{member} "
                                f"{name}: {exc}"
                            )
                            continue

                        fields_by_var[name].append(nest_values)
                        member_used = True

                    if member_used:
                        used_members.append(member)

                print(
                    f"  {exp['label']}/{filter_kind}: "
                    f"{len(used_members)} member(s) usable"
                )

                for variable in TARGET_VARIABLES:
                    name = variable["name"]
                    fields = fields_by_var[name]
                    if len(fields) < 2:
                        print(
                            f"    [skip spread] {name}: fewer than 2 members "
                            f"({len(fields)})"
                        )
                        continue

                    spread = ensemble_spread(fields)
                    spread = np.where(tc_mask, spread, np.nan)
                    spread_data[name][time_obj][(exp["dir"], filter_kind)] = spread

                    finite_spread = np.isfinite(spread)
                    area_mean = (
                        float(np.mean(spread[finite_spread]))
                        if np.any(finite_spread)
                        else np.nan
                    )
                    records.append(
                        {
                            "time": time_name,
                            "experiment": exp["dir"],
                            "experiment_label": exp["label"],
                            "filter": filter_kind,
                            "variable": name,
                            "unit": variable["unit"],
                            "n_members": len(fields),
                            "tc_radius_km": TC_RADIUS_KM,
                            "tc_points": int(tc_mask.sum()),
                            "area_mean_spread": area_mean,
                            "centre_lat": nr["centre_lat"],
                            "centre_lon": nr["centre_lon"],
                            "centre_psfc": nr["centre_psfc"],
                        }
                    )
                    print(
                        f"    {name}: area-mean spread={area_mean:.5g} "
                        f"{variable['unit']} ({len(fields)} members)"
                    )

    return spread_data, time_grids, records


# =============================================================================
# Plotting
# =============================================================================

def spread_colormap() -> LinearSegmentedColormap:
    """Sequential, non-negative colormap suitable for spread fields."""
    colors = [
        "#ffffff",
        "#d7f0ff",
        "#74add1",
        "#4575b4",
        "#1a9850",
        "#91cf60",
        "#ffffbf",
        "#fdae61",
        "#f46d43",
        "#d73027",
        "#7f0000",
    ]
    return LinearSegmentedColormap.from_list("tc_spread", colors, N=256)


def plot_variable(
    variable: dict,
    spread_data: dict,
    time_grids: dict,
    times: list[datetime],
    output_dir: Path,
) -> Path | None:
    name = variable["name"]
    panels = [
        (exp, filter_kind)
        for exp in EXPERIMENTS
        for filter_kind in FILTERS
    ]

    finite_values = []
    for time_obj in times:
        for exp, filter_kind in panels:
            field = spread_data[name][time_obj].get((exp["dir"], filter_kind))
            if field is not None:
                finite_values.append(field[np.isfinite(field)])
    if not finite_values:
        print(f"[skip plot] no spread data for {name}")
        return None

    finite_values = np.concatenate(finite_values)
    vmax = float(np.nanpercentile(finite_values, SPREAD_PERCENTILE))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(finite_values))
    if not np.isfinite(vmax) or vmax <= 0:
        print(f"[skip plot] non-positive colour range for {name}")
        return None

    cmap = spread_colormap()
    cmap.set_bad("white")

    nrows = len(times)
    ncols = len(panels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.3 * nrows),
        squeeze=False,
    )

    mappable = None
    circle_theta = np.linspace(0.0, 2.0 * np.pi, 361)

    for row, time_obj in enumerate(times):
        grid = time_grids.get(time_obj)
        for col, (exp, filter_kind) in enumerate(panels):
            ax = axes[row, col]
            field = spread_data[name][time_obj].get((exp["dir"], filter_kind))

            if field is None or grid is None:
                ax.axis("off")
                if row == 0:
                    ax.set_title(f"{exp['label']}\n{filter_kind}", fontsize=11)
                continue

            mappable = ax.pcolormesh(
                grid["x_km"],
                grid["y_km"],
                field,
                shading="auto",
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
            )

            ax.plot(
                TC_RADIUS_KM * np.cos(circle_theta),
                TC_RADIUS_KM * np.sin(circle_theta),
                color="black",
                linewidth=0.9,
            )
            ax.plot(0.0, 0.0, marker="+", color="black", markersize=9, markeredgewidth=1.2)

            half = max(HALF_WIDTH_KM, TC_RADIUS_KM)
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_aspect("equal", adjustable="box")

            if row == 0:
                ax.set_title(f"{exp['label']}\n{filter_kind}", fontsize=11)
            if col == 0:
                ax.set_ylabel(
                    time_obj.strftime("%H:%M"),
                    rotation=0,
                    labelpad=34,
                    va="center",
                    fontsize=11,
                )
            if row == nrows - 1:
                ax.set_xlabel("distance east of TC centre (km)", fontsize=9)

            ax.tick_params(labelsize=8)

    if mappable is None:
        plt.close(fig)
        print(f"[skip plot] no panel could be drawn for {name}")
        return None

    fig.subplots_adjust(
        left=0.065,
        right=0.90,
        bottom=0.045,
        top=0.92,
        wspace=0.12,
        hspace=0.14,
    )
    fig.text(
        0.012,
        0.5,
        "distance north of TC centre (km)",
        rotation=90,
        va="center",
        fontsize=11,
    )

    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        shrink=0.55,
        pad=0.012,
    )
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(
        f"{variable['label']} ensemble spread ({variable['unit']})",
        fontsize=13,
    )

    fig.suptitle(
        f"{variable['label']} ensemble spread within {TC_RADIUS_KM:g} km of the "
        f"NR TC centre  |  std across {len(MEMBERS)} members (ddof=1)  |  "
        f"{times[0]:%Y-%m-%d} {times[0]:%H:%M}-{times[-1]:%H:%M}",
        fontsize=14,
        y=0.985,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{name}_tc{int(TC_RADIUS_KM)}km_ens_spread.png"
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")
    return save_path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    times = build_times(START_TIME, END_TIME, TIME_STEP)
    print(
        f"Times ({len(times)}): {times[0]:%Y-%m-%d %H:%M} -> "
        f"{times[-1]:%Y-%m-%d %H:%M} every {TIME_STEP}"
    )
    print(f"Filters: {FILTERS}")
    print(f"Experiments: {[exp['dir'] for exp in EXPERIMENTS]}")
    print(f"Members: {MEMBERS}")

    spread_data, time_grids, records = collect(times)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for variable in TARGET_VARIABLES:
        plot_variable(variable, spread_data, time_grids, times, OUTPUT_DIR)

    records_path = OUTPUT_DIR / f"tc{int(TC_RADIUS_KM)}km_ens_spread_area_mean.csv"
    if records:
        pd.DataFrame.from_records(records).to_csv(records_path, index=False)
        print(f"[saved] {records_path}")
    else:
        print("[warn] no records were written")


if __name__ == "__main__":
    main()
