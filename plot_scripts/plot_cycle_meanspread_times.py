"""
Plot ensemble spread for configured WRF variables over d02 or an NR-centered region.

Experiments:
    6mem_oceanAssim0Run0  Assim=0, Run=0
    6mem_oceanAssim0Run1  Assim=0, Run=1
    6mem_oceanAssim1Run1  Assim=1, Run=1

Filters:
    EAKF, QCF_RHF

The script recursively searches each member directory for wrfout_DOMAIN files.
It saves one CSV and, for each configured variable and region mode, one PNG figure.

Region modes:
    domain        whole d02 grid, land included (legacy behavior)
    nr_each_time  ocean points within RADIUS_KM of the NR minimum-PSFC point at each time
    nr_initial    same radius, centered on the initial NR minimum-PSFC point

In the region modes the sample is the intersection of the initial common ocean
support and the current circle; losing a support point at a later time raises
instead of silently changing the ensemble size. NR files are read only to locate
the center; they are not interpolated.
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

# NR files used only to locate the TC center for the region modes.
NR_BASE = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain")
NR_DOMAIN = "d02"
CENTER_PRESSURE_VAR = "PSFC"
RADIUS_KM = 150.0
REGION_MODES = ("domain", "nr_each_time", "nr_initial")
GRID_TOLERANCE_DEG = 1e-6
FILL_THRESHOLD = 1e30

DOMAIN_MODE = "domain"
VALID_REGION_MODES = ("domain", "nr_each_time", "nr_initial")

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


def find_nr_file(nr_base: Path, t: datetime) -> Path:
    pattern = f"wrfout_{NR_DOMAIN}_{wrf_time_name(t)}"
    matches = sorted(Path(nr_base).rglob(pattern))
    if not matches:
        matches = sorted(Path(nr_base).rglob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(f"No NR file matching {pattern} under {nr_base}")
    return matches[0]


def time_dim_name(arr: xr.DataArray) -> str | None:
    for dim in arr.dims:
        if dim.lower() == "time":
            return dim
    return None


def _select_field(ds: xr.Dataset, variable: dict, path: Path) -> np.ndarray:
    var_name = variable["name"]
    vertical_level = variable["vertical_level"]

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
        return arr.values.astype(float)
    if arr.ndim == 3:
        if vertical_level is None:
            raise ValueError(
                f"{var_name} in {path} is 3-D after Time selection. "
                "Set vertical_level in VARIABLES."
            )
        vertical_dim = arr.dims[0]
        return arr.isel({vertical_dim: vertical_level}).values.astype(float)
    raise ValueError(
        f"{var_name} in {path} has unsupported dimensions after Time selection: "
        f"{arr.dims}"
    )


def _surface(ds: xr.Dataset, name: str, path: Path) -> np.ndarray:
    """Return a 2-D Time-free field such as XLAT, XLONG or LANDMASK."""
    if name not in ds:
        raise KeyError(f"{name} not found in {path}")
    arr = ds[name]
    tdim = time_dim_name(arr)
    if tdim is not None:
        arr = arr.isel({tdim: 0})
    if arr.ndim != 2:
        raise ValueError(
            f"{name} in {path} must be 2-D after Time selection, got {arr.dims}"
        )
    return arr.values.astype(float)


def read_variable_field(path: Path, variable: dict) -> np.ndarray:
    with xr.open_dataset(path, decode_times=False) as ds:
        return _select_field(ds, variable, path)


def read_field_with_grid(path: Path, variable: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path, decode_times=False) as ds:
        values = _select_field(ds, variable, path)
        lat = _surface(ds, "XLAT", path)
        lon = _surface(ds, "XLONG", path)
        land = _surface(ds, "LANDMASK", path)
    return values, lat, lon, land


def read_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path, decode_times=False) as ds:
        return _surface(ds, "XLAT", path), _surface(ds, "XLONG", path)


def read_nr_center(path: Path) -> tuple[float, float]:
    """Locate the NR minimum-pressure grid point."""
    with xr.open_dataset(path, decode_times=False) as ds:
        pressure = _surface(ds, CENTER_PRESSURE_VAR, path)
        lat = _surface(ds, "XLAT", path)
        lon = _surface(ds, "XLONG", path)
    pressure = np.where(_valid(pressure), pressure, np.nan)
    if not np.isfinite(pressure).any():
        raise ValueError(f"No valid {CENTER_PRESSURE_VAR} values in {path}")
    j, i = np.unravel_index(np.nanargmin(pressure), pressure.shape)
    return float(lat[j, i]), float(lon[j, i])


def _valid(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (np.abs(values) < FILL_THRESHOLD)


def _distance(lat: np.ndarray, lon: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    a, b = np.deg2rad(lat), np.deg2rad(center[0])
    dl = np.deg2rad(lon - center[1])
    h = np.sin((a - b) / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dl / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _circle_clipped(disk: np.ndarray) -> bool:
    return bool(
        disk[0, :].any() or disk[-1, :].any()
        or disk[:, 0].any() or disk[:, -1].any()
    )


def _check_grid(path: Path, lat: np.ndarray, lon: np.ndarray,
                reference_lat: np.ndarray, reference_lon: np.ndarray) -> None:
    if lat.shape != reference_lat.shape or not np.allclose(
            lat, reference_lat, atol=GRID_TOLERANCE_DEG, rtol=0):
        raise ValueError(f"Member grid differs from reference grid: {path}")
    if lon.shape != reference_lon.shape or not np.allclose(
            lon, reference_lon, atol=GRID_TOLERANCE_DEG, rtol=0):
        raise ValueError(f"Member grid differs from reference grid: {path}")


def spread_field(fields: list[np.ndarray]) -> np.ndarray:
    """Pointwise ensemble spread (ddof=1) across members."""
    return np.nanstd(np.stack(fields, axis=0), axis=0, ddof=1)


def domain_mean_ensemble_spread(fields: list[np.ndarray]) -> float:
    return float(np.nanmean(spread_field(fields)))


def spread_column(variable: dict) -> str:
    suffix = variable["unit"].replace(" ", "_").replace("/", "_")
    level = variable["vertical_level"]
    level_part = "" if level is None else f"_lev{level}"
    return f"{variable['name'].lower()}{level_part}_spread_{suffix}"


def n_points_column(variable: dict) -> str:
    return f"{spread_column(variable)}n_points"


def variable_label(variable: dict) -> str:
    level = variable["vertical_level"]
    if level is None:
        return variable["name"]
    return f"{variable['name']} level {level}"


def mode_output_path(out_png: Path, mode: str) -> Path:
    if mode == DOMAIN_MODE:
        return out_png
    return out_png.with_name(f"{out_png.stem}_{mode}{out_png.suffix}")


def _validate_config(modes: tuple[str, ...], radius_km: float) -> None:
    if not EXPERIMENTS or not FILTERS or not MEMBERS or not VARIABLES:
        raise ValueError("Experiments, filters, members and variables must be nonempty")
    if not modes:
        raise ValueError("At least one region mode is required")
    unknown = set(modes) - set(VALID_REGION_MODES)
    if unknown:
        raise ValueError(f"Unknown region mode(s): {sorted(unknown)}")
    if any(mode != DOMAIN_MODE for mode in modes) and not radius_km > 0:
        raise ValueError("radius_km must be positive for region modes")


def _initial_support(base: Path, t0: datetime,
                     reference_lat: np.ndarray,
                     reference_lon: np.ndarray) -> dict[str, np.ndarray]:
    """Ocean points valid in every member at t0; frozen across all times."""
    support = {}
    for variable in VARIABLES:
        column = spread_column(variable)
        combined = None
        for exp in EXPERIMENTS:
            if exp not in variable["experiments"]:
                continue
            for filt in FILTERS:
                for member in MEMBERS:
                    path = find_member_file(base, exp, filt, member, t0)
                    field, lat, lon, land = read_field_with_grid(path, variable)
                    _check_grid(path, lat, lon, reference_lat, reference_lon)
                    valid = _valid(field) & _valid(land) & (land < 0.5)
                    combined = valid if combined is None else (combined & valid)
        if combined is None:
            raise ValueError(f"No members configured for {variable['name']}")
        support[column] = combined
    return support


def _load_members(base: Path, exp: str, filt: str, t: datetime,
                  reference_lat: np.ndarray | None,
                  reference_lon: np.ndarray | None,
                  region_enabled: bool
                  ) -> tuple[dict[str, list[np.ndarray]], dict[str, np.ndarray | None]]:
    fields_by_metric = {spread_column(v): [] for v in VARIABLES if exp in v["experiments"]}
    valid_by_metric = {spread_column(v): None for v in VARIABLES if exp in v["experiments"]}
    for member in MEMBERS:
        path = find_member_file(base, exp, filt, member, t)
        for variable in VARIABLES:
            if exp not in variable["experiments"]:
                continue
            column = spread_column(variable)
            if region_enabled:
                field, lat, lon, land = read_field_with_grid(path, variable)
                _check_grid(path, lat, lon, reference_lat, reference_lon)
                valid = _valid(field) & _valid(land) & (land < 0.5)
                previous = valid_by_metric[column]
                valid_by_metric[column] = valid if previous is None else (previous & valid)
            else:
                field = read_variable_field(path, variable)
            fields_by_metric[column].append(field)
    return fields_by_metric, valid_by_metric


def calculate(base: Path, times: list[datetime], *,
              region_modes: tuple[str, ...] | None = None,
              radius_km: float | None = None,
              nr_base: Path | None = None) -> pd.DataFrame:
    modes = tuple(REGION_MODES if region_modes is None else region_modes)
    radius = RADIUS_KM if radius_km is None else float(radius_km)
    nr_root = NR_BASE if nr_base is None else Path(nr_base)
    _validate_config(modes, radius)

    region_enabled = any(mode != DOMAIN_MODE for mode in modes)

    reference_lat = reference_lon = None
    support0: dict[str, np.ndarray] = {}
    centers: dict[datetime, tuple[float, float]] = {}
    clipped: dict[datetime, bool] = {}
    if region_enabled:
        first = find_member_file(base, EXPERIMENTS[0], FILTERS[0], MEMBERS[0], times[0])
        reference_lat, reference_lon = read_grid(first)
        support0 = _initial_support(base, times[0], reference_lat, reference_lon)
        centers = {t: read_nr_center(find_nr_file(nr_root, t)) for t in times}
        clipped = {
            t: _circle_clipped(_distance(reference_lat, reference_lon, centers[t]) <= radius)
            for t in times
        }

    records = []
    for t in times:
        stamp = wrf_time_name(t)
        for exp in EXPERIMENTS:
            for filt in FILTERS:
                fields_by_metric, valid_by_metric = _load_members(
                    base, exp, filt, t, reference_lat, reference_lon, region_enabled)
                if region_enabled:
                    for variable in VARIABLES:
                        column = spread_column(variable)
                        if column not in valid_by_metric:
                            continue
                        support = support0[column]
                        lost = support & ~valid_by_metric[column]
                        if lost.any():
                            raise ValueError(
                                f"Common spatial support lost at {stamp}: "
                                f"{int(lost.sum())} points; inspect missing/fill "
                                "values or changing land masks"
                            )
                spread_by_metric = {
                    column: spread_field(fields)
                    for column, fields in fields_by_metric.items()
                }
                for mode in modes:
                    record = {
                        "time": stamp,
                        "experiment": exp,
                        "filter": filt,
                        "region_mode": mode,
                    }
                    if mode == DOMAIN_MODE:
                        record["center_lat"] = np.nan
                        record["center_lon"] = np.nan
                        record["circle_clipped"] = False
                        disk = None
                    else:
                        center = centers[t] if mode == "nr_each_time" else centers[times[0]]
                        disk = _distance(reference_lat, reference_lon, center) <= radius
                        record["center_lat"], record["center_lon"] = center
                        record["circle_clipped"] = (
                            clipped[t] if mode == "nr_each_time" else clipped[times[0]]
                        )
                    for variable in VARIABLES:
                        column = spread_column(variable)
                        grid_spread = spread_by_metric.get(column)
                        points_column = n_points_column(variable)
                        if grid_spread is None:
                            record[column] = np.nan
                            record[points_column] = 0
                            continue
                        if mode == DOMAIN_MODE:
                            finite = np.isfinite(grid_spread)
                            record[column] = (
                                float(np.nanmean(grid_spread)) if finite.any() else np.nan
                            )
                            record[points_column] = int(finite.sum())
                        else:
                            mask = support0[column] & disk
                            if not mask.any():
                                raise ValueError(f"Empty ocean support: {mode} at {stamp}")
                            record[column] = float(np.nanmean(grid_spread[mask]))
                            record[points_column] = int(mask.sum())
                    records.append(record)
    return pd.DataFrame.from_records(records)


def plot_one(
    df: pd.DataFrame,
    out_png: Path,
    metric: str,
    ylabel: str,
    experiments: list[str],
    region_mode: str,
) -> None:
    data = df[df["region_mode"] == region_mode]
    times = pd.to_datetime(data["time"], format="%Y-%m-%d_%H:%M:%S")
    data = data.assign(time_dt=times)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for exp in experiments:
        for filt in FILTERS:
            sub = data[(data["experiment"] == exp) & (data["filter"] == filt)].sort_values("time_dt")
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


def plot(df: pd.DataFrame, *,
         region_modes: tuple[str, ...] | None = None,
         radius_km: float | None = None) -> None:
    modes = tuple(REGION_MODES if region_modes is None else region_modes)
    radius = RADIUS_KM if radius_km is None else float(radius_km)
    for variable in VARIABLES:
        for mode in modes:
            if mode == DOMAIN_MODE:
                ylabel = (
                    f"Domain-mean ensemble spread of {variable_label(variable)} "
                    f"({variable['unit']})"
                )
            else:
                ylabel = (
                    f"Ocean-mean ensemble spread of {variable_label(variable)} "
                    f"(r ≤ {radius:g} km, {mode}) ({variable['unit']})"
                )
            plot_one(
                df=df,
                out_png=mode_output_path(variable["out_png"], mode),
                metric=spread_column(variable),
                ylabel=ylabel,
                experiments=variable["experiments"],
                region_mode=mode,
            )


def main() -> None:
    times = build_times(START_TIME, END_TIME, STEP_MINUTES)
    df = calculate(DEFAULT_BASE, times)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    plot(df)

    print(f"Saved {OUT_CSV}")
    for variable in VARIABLES:
        for mode in REGION_MODES:
            print(f"Saved {mode_output_path(variable['out_png'], mode)}")


if __name__ == "__main__":
    main()
