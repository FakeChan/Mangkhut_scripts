#!/usr/bin/env python3
"""
Plot single-observation DA responses against an independent 1.5 km NR grid.

The script is adapted for the Mangkhut real-case single-point experiments:

    DART/{EAKF,QCF_RHF}/obs_seq{111,325,640}/

Each run directory is expected to contain DART analysis files such as
``output_d01.mem001`` and ``output_mean_d01.nc`` plus ``test.out``.  The
observation source and first-guess member directory are configured separately
near the top of this file.  The ensemble grid is interpolated to the NR grid
before computing and plotting analysis-minus-NR fields in a TC-centered 150 km
region.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"),
)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figs" / "singleobs_nr_compare"
DEFAULT_NR_BASE = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout")


# =============================================================================
# User configuration
# =============================================================================
# Edit this block before running the script.  Analysis output is expected as:
#   DATA_ROOT / FILTER / obs_seq{point}
DATA_ROOT = None  # None: use PROJECT_ROOT/DART, or /DART if it exists.
FILTERS = ["EAKF", "QCF_RHF"]
OBS_POINTS = [111, 325, 640]
DOMAINS = ["d01", "d02"]
MEMBERS = list(range(1, 51))

# Observation source used only for obs location/value.  This can be:
#   1. a single obs_seq file,
#   2. a directory containing obs_seq files,
#   3. a root directory containing obs_seq111, obs_seq325, obs_seq640,
#   4. a dict such as {111: "/path/to/obs_seq111", 325: "..."}.
OBS_SOURCE_PATH = None

# First-guess member files used for the prior/state scatter panels.  This can be
# a single directory containing firstguess_d01.mem001, firstguess_d02.mem001,
# or a dict such as {"d01": "/path/to/d01/firstguess", "d02": "/path/to/d02/firstguess"}.
FIRSTGUESS_DIR = None

# Field to compare with NR.  QVAPOR is automatically converted kg kg-1 -> g kg-1.
VAR_NAME = "QVAPOR"
LEVEL = 2
SCALE = "auto"

# NR setting.  Set NR_FILE directly when possible.  If NR_FILE is None, the
# script searches NR_BASE for wrfout_{NR_DOMAIN}_{TIME_STRING}.
NR_FILE = None
NR_BASE = DEFAULT_NR_BASE
NR_DOMAIN = "d02"
TIME_STRING = None

TC_RADIUS_KM = 150.0
STATE_SELECTION = "max_abs_error"  # max_abs_error, obs_nearest, or tc_center
STATE_LAT = None
STATE_LON = None

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
FIG_FORMAT = "png"
DPI = 450

# File name prefixes used when searching analysis and first-guess directories.
OUTPUT_PREFIXES = ["output", "postassim", "analysis"]
FIRSTGUESS_PREFIXES = ["firstguess", "input", "preassim", "prior"]

FILTER_LABELS = {
    "EAKF": "EAKF",
    "QCF_RHF": "QCF-RHF",
}

VAR_LABELS = {
    "QVAPOR": r"$q_v$ (g kg$^{-1}$)",
    "THM": "Potential temperature perturbation (K)",
    "P": "Perturbation pressure (Pa)",
    "MU": "Dry-air mass perturbation (Pa)",
    "OM_TMP": "Ocean temperature (K)",
    "OM_S": "Ocean salinity",
    "PSFC": "Surface pressure (hPa)",
}


@dataclass
class ObsSeqInfo:
    obs_id: int | None = None
    lat: float | None = None
    lon: float | None = None
    obs_value: float | None = None
    hx: np.ndarray | None = None
    errvar: float | None = None


@dataclass
class RunResult:
    filt: str
    run_dir: Path
    mean_file: Path
    error_on_nr: np.ndarray
    mean_on_nr: np.ndarray
    rmse: float
    obs_space: dict[str, np.ndarray]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def default_data_root() -> Path:
    project_dart = PROJECT_ROOT / "DART"
    absolute_dart = Path("/DART")
    if (project_dart / "EAKF").exists() or not absolute_dart.exists():
        return project_dart
    return absolute_dart


def read_2d_grid(path: Path, lat_name: str = "XLAT", lon_name: str = "XLONG") -> tuple[np.ndarray, np.ndarray]:
    with netCDF4.Dataset(path) as ds:
        lat = np.asarray(ds.variables[lat_name][:], dtype=float)
        lon = np.asarray(ds.variables[lon_name][:], dtype=float)
    return squeeze_time(lat), squeeze_time(lon)


def squeeze_time(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    return np.squeeze(arr)


def field_grid_names(ds: netCDF4.Dataset, var_name: str) -> tuple[str, str]:
    dims = ds.variables[var_name].dimensions
    if "west_east_stag" in dims and "XLAT_U" in ds.variables:
        return "XLAT_U", "XLONG_U"
    if "south_north_stag" in dims and "XLAT_V" in ds.variables:
        return "XLAT_V", "XLONG_V"
    return "XLAT", "XLONG"


def auto_scale(var_name: str) -> float:
    if var_name.upper() == "QVAPOR":
        return 1000.0
    if var_name.upper() == "PSFC":
        return 0.01
    return 1.0


def read_field(path: Path, var_name: str, level: int | None, scale: float) -> np.ndarray:
    with netCDF4.Dataset(path) as ds:
        if var_name not in ds.variables:
            raise KeyError(f"{var_name} not found in {path}")
        arr = np.asarray(ds.variables[var_name][:], dtype=float)

    arr = squeeze_time(arr)
    if arr.ndim == 3:
        if level is None:
            raise ValueError(f"{var_name} in {path} is 3D after time squeeze; set LEVEL near the top of this script")
        arr = arr[level, :, :]
    elif arr.ndim != 2:
        raise ValueError(f"{var_name} in {path} must become 2D, got shape {arr.shape}")
    return arr * scale


def read_grid_for_field(path: Path, var_name: str) -> tuple[np.ndarray, np.ndarray]:
    with netCDF4.Dataset(path) as ds:
        lat_name, lon_name = field_grid_names(ds, var_name)
        lat = np.asarray(ds.variables[lat_name][:], dtype=float)
        lon = np.asarray(ds.variables[lon_name][:], dtype=float)
    return squeeze_time(lat), squeeze_time(lon)


def read_time_string(path: Path) -> str | None:
    with netCDF4.Dataset(path) as ds:
        if "Times" not in ds.variables:
            return None
        raw = ds.variables["Times"][:]
    if raw.ndim == 2:
        raw = raw[0]
    chars = []
    for item in raw:
        if isinstance(item, bytes):
            chars.append(item.decode("ascii"))
        else:
            chars.append(str(item))
    text = "".join(chars).strip()
    return text or None


def latlon_distance_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    radius_earth_km = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat1_rad - lat2_rad
    dlon = lon1_rad - lon2_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * math.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return radius_earth_km * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def find_pressure_for_center(path: Path) -> tuple[np.ndarray, str]:
    with netCDF4.Dataset(path) as ds:
        for name in ("SLP", "slp", "PSFC"):
            if name in ds.variables:
                arr = squeeze_time(np.asarray(ds.variables[name][:], dtype=float))
                if arr.ndim == 2:
                    return arr, name
    raise KeyError(f"No 2D SLP/slp/PSFC variable found for TC center in {path}")


def tc_center_from_nr(path: Path) -> tuple[float, float, float, str]:
    lat, lon = read_2d_grid(path)
    pressure, pressure_name = find_pressure_for_center(path)
    if pressure.shape != lat.shape:
        raise ValueError(f"{pressure_name} shape {pressure.shape} does not match XLAT shape {lat.shape}")
    j, i = np.unravel_index(np.nanargmin(pressure), pressure.shape)
    return float(lat[j, i]), float(lon[j, i]), float(pressure[j, i]), pressure_name


def find_nr_file(nr_file: Path | None, nr_base: Path, nr_domain: str, time_string: str | None) -> Path:
    if nr_file is not None:
        if not nr_file.exists():
            raise FileNotFoundError(nr_file)
        return nr_file
    if time_string is None:
        raise ValueError("Could not infer Times from DART output; set NR_FILE or TIME_STRING near the top of this script")
    pattern = f"wrfout_{nr_domain}_{time_string}"
    matches = sorted(nr_base.rglob(pattern))
    if not matches:
        matches = sorted(nr_base.rglob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(f"No NR file matching {pattern} under {nr_base}")
    return matches[0]


def parse_obs_space_log(path: Path, members: list[int]) -> dict[str, np.ndarray]:
    if not path.exists():
        warnings.warn(f"test.out not found: {path}")
        return {}

    key_alias = {
        "obs_inc": "obs_increment",
        "obs_increment": "obs_increment",
        "obs_prior": "obs_prior",
        "probit_obs_inc": "probit_obs_increment",
        "probit_obs_increment": "probit_obs_increment",
        "probit_obs_prior": "probit_obs_prior",
    }
    pattern = re.compile(
        r"fkc msg:\s+(?P<key>[A-Za-z_]+):\s+"
        r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
    )
    values: dict[str, list[float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        key = key_alias.get(match.group("key"))
        if key is None:
            continue
        values.setdefault(key, []).append(float(match.group("value")))

    nmem = len(members)
    out: dict[str, np.ndarray] = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size < nmem:
            warnings.warn(f"{path}: found only {arr.size} {key} values for {nmem} members")
            continue
        if arr.size > nmem:
            warnings.warn(f"{path}: found {arr.size} {key} values; using the last {nmem}")
            arr = arr[-nmem:]
        out[key] = arr
    return out


def parse_obs_seq(path: Path, wanted_obs_id: int | None = None) -> ObsSeqInfo | None:
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    best: ObsSeqInfo | None = None
    while i < len(lines):
        if not re.match(r"\s*OBS\s+\d+", lines[i]):
            i += 1
            continue

        obs_id = int(lines[i].split()[1])
        obs_value = try_float_first_token(lines[i + 1]) if i + 1 < len(lines) else None
        info = ObsSeqInfo(obs_id=obs_id, obs_value=obs_value)
        j = i + 3
        while j < len(lines):
            stripped = lines[j].strip()
            if re.match(r"\s*OBS\s+\d+", stripped):
                break
            if stripped == "loc3d" and j + 1 < len(lines):
                parts = lines[j + 1].split()
                if len(parts) >= 2:
                    info.lon = math.degrees(float(parts[0]))
                    info.lat = math.degrees(float(parts[1]))
            elif stripped.startswith("external_FO"):
                parts = stripped.split()
                nmem = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 50
                vals: list[float] = []
                k = j + 1
                while len(vals) < nmem and k < len(lines):
                    if re.match(r"\s*OBS\s+\d+", lines[k]):
                        break
                    try:
                        vals.extend(float(x) for x in lines[k].split())
                    except ValueError:
                        break
                    k += 1
                if len(vals) >= nmem:
                    info.hx = np.asarray(vals[:nmem], dtype=float)
                    if k + 1 < len(lines):
                        info.errvar = try_float_first_token(lines[k + 1])
                j = k
                continue
            j += 1

        if wanted_obs_id is None or obs_id == wanted_obs_id:
            return info
        if best is None:
            best = info
        i = j
    return best


def try_float_first_token(text: str) -> float | None:
    try:
        return float(text.split()[0])
    except (IndexError, ValueError):
        return None


def resolve_obs_source(obs_point: int, fallback_run_dir: Path) -> Path:
    if OBS_SOURCE_PATH is None:
        return fallback_run_dir

    if isinstance(OBS_SOURCE_PATH, dict):
        if obs_point not in OBS_SOURCE_PATH:
            raise KeyError(f"OBS_SOURCE_PATH has no entry for obs point {obs_point}")
        source = Path(OBS_SOURCE_PATH[obs_point])
    else:
        source = Path(OBS_SOURCE_PATH)

    if source.is_file():
        return source

    point_child = source / f"obs_seq{obs_point}"
    if point_child.exists():
        return point_child

    return source


def resolve_firstguess_dir(domain: str) -> Path:
    if FIRSTGUESS_DIR is None:
        raise ValueError("Set FIRSTGUESS_DIR near the top of this script before running.")
    if isinstance(FIRSTGUESS_DIR, dict):
        if domain not in FIRSTGUESS_DIR:
            raise KeyError(f"FIRSTGUESS_DIR has no entry for domain {domain}")
        return Path(FIRSTGUESS_DIR[domain])
    return Path(FIRSTGUESS_DIR)


def find_obs_seq_info(obs_source: Path, obs_point: int) -> ObsSeqInfo | None:
    if obs_source.is_file():
        try:
            return parse_obs_seq(obs_source, wanted_obs_id=None)
        except Exception as exc:
            warnings.warn(f"Could not parse obs_seq metadata from {obs_source}: {exc}")
            return None

    candidates = [
        obs_source / "obs_seq.final",
        obs_source / "obs_seq.out",
        obs_source / f"obs_seq{obs_point}",
        obs_source / f"obs_seq.out{obs_point}",
    ]
    candidates.extend(sorted(p for p in obs_source.glob("obs_seq*") if p.is_file()))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        try:
            info = parse_obs_seq(path, wanted_obs_id=None)
        except Exception as exc:
            warnings.warn(f"Could not parse obs_seq metadata from {path}: {exc}")
            continue
        if info is not None and (info.lat is not None or info.obs_value is not None):
            return info
    return None


def resolve_run_dir(data_root: Path, filt: str, obs_point: int) -> Path:
    run_dir = data_root / filt / f"obs_seq{obs_point}"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    return run_dir


def find_mean_file(run_dir: Path, domain: str) -> Path:
    candidates = [
        f"output_mean_{domain}.nc",
        f"output_mean_{domain}",
        f"postassim_mean_{domain}.nc",
        f"postassim_mean_{domain}",
        f"analysis_mean_{domain}.nc",
        f"analysis_{domain}.ensmean",
        f"output_{domain}.ensmean",
    ]
    for name in candidates:
        path = run_dir / name
        if path.exists():
            return path
    matches = sorted(run_dir.glob(f"*mean*{domain}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No ensemble mean file for {domain} under {run_dir}")


def find_member_file(run_dir: Path, domain: str, member: int, prefixes: list[str]) -> Path | None:
    mem = f"{member:03d}"
    for prefix in prefixes:
        candidates = [
            run_dir / f"{prefix}_{domain}.mem{mem}",
            run_dir / f"{prefix}_{domain}.mem{mem}.nc",
            run_dir / f"{prefix}.{domain}.mem{mem}",
            run_dir / f"{prefix}.mem{mem}",
        ]
        for path in candidates:
            if path.exists():
                return path
    matches = []
    for prefix in prefixes:
        matches.extend(run_dir.glob(f"{prefix}*{domain}*mem{mem}*"))
    return sorted(matches)[0] if matches else None


def subset_source_for_targets(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    pad_deg: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_target = np.isfinite(target_lats) & np.isfinite(target_lons)
    lat_min = float(np.nanmin(target_lats[finite_target])) - pad_deg
    lat_max = float(np.nanmax(target_lats[finite_target])) + pad_deg
    lon_min = float(np.nanmin(target_lons[finite_target])) - pad_deg
    lon_max = float(np.nanmax(target_lons[finite_target])) + pad_deg
    mask = (
        np.isfinite(lats)
        & np.isfinite(lons)
        & np.isfinite(values)
        & (lats >= lat_min)
        & (lats <= lat_max)
        & (lons >= lon_min)
        & (lons <= lon_max)
    )
    if mask.sum() < 3:
        mask = np.isfinite(lats) & np.isfinite(lons) & np.isfinite(values)
    return lats[mask], lons[mask], values[mask]


def interp_to_targets(
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    src_values: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    flat_lats, flat_lons, flat_values = subset_source_for_targets(
        src_lats, src_lons, src_values, target_lats, target_lons
    )
    if flat_values.size < 3:
        raise ValueError("Not enough finite source points for interpolation")
    points = np.column_stack((flat_lons, flat_lats))
    linear = LinearNDInterpolator(points, flat_values, fill_value=np.nan)
    out = np.asarray(linear(target_lons, target_lats), dtype=float)
    if np.isnan(out).any():
        nearest = NearestNDInterpolator(points, flat_values)
        fill = np.asarray(nearest(target_lons, target_lats), dtype=float)
        out = np.where(np.isfinite(out), out, fill)
    return out


def interp_file_to_nr(path: Path, var_name: str, level: int | None, scale: float, nr_lats: np.ndarray, nr_lons: np.ndarray) -> np.ndarray:
    src_lats, src_lons = read_grid_for_field(path, var_name)
    src_values = read_field(path, var_name, level, scale)
    if src_values.shape != src_lats.shape:
        raise ValueError(f"{path}: {var_name} shape {src_values.shape} does not match grid {src_lats.shape}")
    return interp_to_targets(src_lats, src_lons, src_values, nr_lats, nr_lons)


def interp_file_to_point(path: Path, var_name: str, level: int | None, scale: float, lat: float, lon: float) -> float:
    value = interp_file_to_nr(path, var_name, level, scale, np.asarray([[lat]]), np.asarray([[lon]]))
    return float(value[0, 0])


def rmse(values: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(truth) & mask
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.nanmean((values[valid] - truth[valid]) ** 2)))


def choose_state_point(
    nr_lats: np.ndarray,
    nr_lons: np.ndarray,
    truth: np.ndarray,
    score: np.ndarray,
    region_mask: np.ndarray,
) -> tuple[float, float, float]:
    valid = np.isfinite(score) & np.isfinite(truth) & region_mask
    if not np.any(valid):
        raise ValueError("No finite score values are available for state-point selection")
    tmp = np.where(valid, np.abs(score), np.nan)
    j, i = np.unravel_index(np.nanargmax(tmp), tmp.shape)
    return float(nr_lats[j, i]), float(nr_lons[j, i]), float(truth[j, i])


def select_state_point(
    selection: str,
    nr_lats: np.ndarray,
    nr_lons: np.ndarray,
    truth: np.ndarray,
    score: np.ndarray,
    region_mask: np.ndarray,
    obs_info: ObsSeqInfo | None,
    explicit_lat: float | None,
    explicit_lon: float | None,
    tc_lat: float,
    tc_lon: float,
) -> tuple[float, float, float]:
    if explicit_lat is not None and explicit_lon is not None:
        truth_point = interp_to_targets(nr_lats, nr_lons, truth, np.asarray([[explicit_lat]]), np.asarray([[explicit_lon]]))
        return explicit_lat, explicit_lon, float(truth_point[0, 0])
    if selection == "tc_center":
        valid_truth = np.isfinite(truth) & region_mask
        dist = latlon_distance_km(nr_lats, nr_lons, tc_lat, tc_lon)
        j, i = np.unravel_index(np.nanargmin(np.where(valid_truth, dist, np.nan)), truth.shape)
        return float(nr_lats[j, i]), float(nr_lons[j, i]), float(truth[j, i])
    if selection == "obs_nearest":
        if obs_info is None or obs_info.lat is None or obs_info.lon is None:
            warnings.warn("Observation location unavailable; falling back to max_abs_error state point")
        else:
            dist = latlon_distance_km(nr_lats, nr_lons, obs_info.lat, obs_info.lon)
            valid = np.isfinite(truth) & region_mask
            j, i = np.unravel_index(np.nanargmin(np.where(valid, dist, np.nan)), truth.shape)
            return float(nr_lats[j, i]), float(nr_lons[j, i]), float(truth[j, i])
    return choose_state_point(nr_lats, nr_lons, truth, score, region_mask)


def calculate_run(
    filt: str,
    run_dir: Path,
    domain: str,
    members: list[int],
    output_prefixes: list[str],
    var_name: str,
    level: int | None,
    scale: float,
    nr_lats: np.ndarray,
    nr_lons: np.ndarray,
    nr_truth: np.ndarray,
    region_mask: np.ndarray,
) -> RunResult:
    mean_file = find_mean_file(run_dir, domain)
    mean_on_nr = interp_file_to_nr(mean_file, var_name, level, scale, nr_lats, nr_lons)
    error = mean_on_nr - nr_truth
    obs_space = parse_obs_space_log(run_dir / "test.out", members)
    return RunResult(
        filt=filt,
        run_dir=run_dir,
        mean_file=mean_file,
        error_on_nr=np.where(region_mask, error, np.nan),
        mean_on_nr=np.where(region_mask, mean_on_nr, np.nan),
        rmse=rmse(mean_on_nr, nr_truth, region_mask),
        obs_space=obs_space,
    )


def member_values_at_point(
    run_dir: Path,
    domain: str,
    members: list[int],
    prefixes: list[str],
    var_name: str,
    level: int | None,
    scale: float,
    lat: float,
    lon: float,
) -> np.ndarray | None:
    values = []
    missing = []
    for member in members:
        path = find_member_file(run_dir, domain, member, prefixes)
        if path is None:
            missing.append(member)
            continue
        values.append(interp_file_to_point(path, var_name, level, scale, lat, lon))
    if missing:
        warnings.warn(f"{run_dir}: missing {len(missing)} member files for prefixes {prefixes}")
    if not values:
        return None
    return np.asarray(values, dtype=float)


def plot_scatter_panel(
    ax: plt.Axes,
    result: RunResult,
    firstguess_dir: Path,
    domain: str,
    members: list[int],
    firstguess_prefixes: list[str],
    output_prefixes: list[str],
    var_name: str,
    level: int | None,
    scale: float,
    state_lat: float,
    state_lon: float,
    state_truth: float,
    obs_info: ObsSeqInfo | None,
) -> None:
    obs_prior = result.obs_space.get("obs_prior")
    obs_increment = result.obs_space.get("obs_increment")
    if obs_prior is None or obs_increment is None:
        ax.text(0.5, 0.5, "No obs_prior/obs_inc in test.out", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    y_prior = member_values_at_point(
        firstguess_dir, domain, members, firstguess_prefixes, var_name, level, scale, state_lat, state_lon
    )
    y_post = member_values_at_point(
        result.run_dir, domain, members, output_prefixes, var_name, level, scale, state_lat, state_lon
    )
    n = min(len(obs_prior), len(obs_increment), len(y_post) if y_post is not None else 0)
    if n == 0:
        ax.text(0.5, 0.5, "No member values for state point", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x_prior = obs_prior[:n]
    x_post = x_prior + obs_increment[:n]
    y_post = y_post[:n]

    prior_color = "#7b5fc8"
    post_color = "#57a773"
    truth_color = "#d73027"

    if y_prior is not None and len(y_prior) >= n:
        y_prior = y_prior[:n]
        ax.scatter(x_prior, y_prior, s=14, color=prior_color, alpha=0.75, label="prior")
        for xp, xa, yp, ya in zip(x_prior, x_post, y_prior, y_post):
            ax.plot([xp, xa], [yp, ya], color="0.72", lw=0.6, zorder=0)
        ax.scatter(np.nanmean(x_prior), np.nanmean(y_prior), s=36, marker="s", color=prior_color, label="prior mean")

    ax.scatter(x_post, y_post, s=14, color=post_color, alpha=0.75, label="analysis")
    ax.scatter(np.nanmean(x_post), np.nanmean(y_post), s=36, marker="s", color=post_color, label="analysis mean")

    if obs_info is not None and obs_info.obs_value is not None and np.isfinite(state_truth):
        ax.scatter(obs_info.obs_value, state_truth, s=55, marker="v", color=truth_color, label="NR")
        ax.axvline(obs_info.obs_value, color=truth_color, lw=0.8, ls="--", alpha=0.8)
        ax.axhline(state_truth, color=truth_color, lw=0.8, ls="--", alpha=0.8)

    ax.set_title(FILTER_LABELS.get(result.filt, result.filt))
    ax.set_xlabel("Brightness temperature / H(x) (K)")
    ax.set_ylabel(VAR_LABELS.get(var_name, var_name))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(True, color="0.9", lw=0.5)


def plot_error_panel(
    ax: plt.Axes,
    result: RunResult,
    nr_lats: np.ndarray,
    nr_lons: np.ndarray,
    region_mask: np.ndarray,
    tc_lat: float,
    tc_lon: float,
    obs_info: ObsSeqInfo | None,
    state_lat: float,
    state_lon: float,
    vlim: float,
) -> None:
    field = np.where(region_mask, result.error_on_nr, np.nan)
    pcm = ax.pcolormesh(nr_lons, nr_lats, field, shading="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.scatter(tc_lon, tc_lat, marker="+", s=70, lw=1.5, c="black", label="TC center")
    if obs_info is not None and obs_info.lat is not None and obs_info.lon is not None:
        ax.scatter(obs_info.lon, obs_info.lat, marker="x", s=44, lw=1.3, c="black", label="obs")
    ax.scatter(state_lon, state_lat, marker="v", s=42, c="white", edgecolors="black", linewidths=0.8, label="state")
    ax.set_title(f"{FILTER_LABELS.get(result.filt, result.filt)} mean - NR, RMSE={result.rmse:.3g}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.88", lw=0.4)
    return pcm


def make_figure(
    obs_point: int,
    domain: str,
    nr_file: Path,
    members: list[int],
    scale: float,
    data_root: Path,
) -> Path:
    nr_lats, nr_lons = read_grid_for_field(nr_file, VAR_NAME)
    nr_truth = read_field(nr_file, VAR_NAME, LEVEL, scale)
    tc_lat, tc_lon, tc_pressure, center_name = tc_center_from_nr(nr_file)
    region_mask = latlon_distance_km(nr_lats, nr_lons, tc_lat, tc_lon) <= TC_RADIUS_KM
    if not np.any(region_mask):
        raise ValueError(f"No NR grid points inside {TC_RADIUS_KM:g} km of TC center")

    run_dirs = {filt: resolve_run_dir(data_root, filt, obs_point) for filt in FILTERS}
    obs_source = resolve_obs_source(obs_point, next(iter(run_dirs.values())))
    obs_info = find_obs_seq_info(obs_source, obs_point)
    firstguess_dir = resolve_firstguess_dir(domain)

    results = [
        calculate_run(
            filt,
            run_dir,
            domain,
            members,
            OUTPUT_PREFIXES,
            VAR_NAME,
            LEVEL,
            scale,
            nr_lats,
            nr_lons,
            nr_truth,
            region_mask,
        )
        for filt, run_dir in run_dirs.items()
    ]

    reference = results[0]
    state_lat, state_lon, state_truth = select_state_point(
        STATE_SELECTION,
        nr_lats,
        nr_lons,
        nr_truth,
        reference.error_on_nr,
        region_mask,
        obs_info,
        STATE_LAT,
        STATE_LON,
        tc_lat,
        tc_lon,
    )

    finite_errors = np.concatenate([r.error_on_nr[np.isfinite(r.error_on_nr) & region_mask].ravel() for r in results])
    vlim = float(np.nanpercentile(np.abs(finite_errors), 98)) if finite_errors.size else 1.0
    if not np.isfinite(vlim) or vlim == 0:
        vlim = 1.0

    configure_matplotlib()
    ncols = len(results)
    fig, axs = plt.subplots(2, ncols, figsize=(4.3 * ncols, 7.0), squeeze=False, constrained_layout=True)

    for col, result in enumerate(results):
        plot_scatter_panel(
            axs[0, col],
            result,
            firstguess_dir,
            domain,
            members,
            FIRSTGUESS_PREFIXES,
            OUTPUT_PREFIXES,
            VAR_NAME,
            LEVEL,
            scale,
            state_lat,
            state_lon,
            state_truth,
            obs_info,
        )
        pcm = plot_error_panel(
            axs[1, col],
            result,
            nr_lats,
            nr_lons,
            region_mask,
            tc_lat,
            tc_lon,
            obs_info,
            state_lat,
            state_lon,
            vlim,
        )
        if col == ncols - 1:
            fig.colorbar(pcm, ax=axs[1, :], shrink=0.88, label=f"{VAR_LABELS.get(VAR_NAME, VAR_NAME)} difference")

    if axs[0, 0].get_legend_handles_labels()[0]:
        axs[0, 0].legend(loc="best", fontsize=7)
    if axs[1, 0].get_legend_handles_labels()[0]:
        axs[1, 0].legend(loc="best", fontsize=7)

    fig.suptitle(
        (
            f"Single obs {obs_point}, {domain}, {VAR_NAME}"
            f"{'' if LEVEL is None else f' level {LEVEL}'} | "
            f"NR center from {center_name}={tc_pressure:.1f}, radius={TC_RADIUS_KM:g} km"
        ),
        fontsize=10,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"singleobs{obs_point}_{domain}_{VAR_NAME}_lev{LEVEL if LEVEL is not None else '2d'}"
    fig.savefig(f"{stem}.{FIG_FORMAT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(f".{FIG_FORMAT}")


def main() -> None:
    if STATE_SELECTION not in {"max_abs_error", "obs_nearest", "tc_center"}:
        raise ValueError("STATE_SELECTION must be max_abs_error, obs_nearest, or tc_center")

    data_root = default_data_root() if DATA_ROOT is None else Path(DATA_ROOT)
    nr_file_config = None if NR_FILE is None else Path(NR_FILE)
    scale = auto_scale(VAR_NAME) if SCALE == "auto" else float(SCALE)

    first_run = resolve_run_dir(data_root, FILTERS[0], OBS_POINTS[0])
    first_mean = find_mean_file(first_run, DOMAINS[0])
    inferred_time = TIME_STRING or read_time_string(first_mean)
    nr_file = find_nr_file(nr_file_config, Path(NR_BASE), NR_DOMAIN, inferred_time)

    print(f"Using NR file: {nr_file}")
    wrote: list[Path] = []
    for obs_point in OBS_POINTS:
        for domain in DOMAINS:
            print(f"Plotting obs_seq{obs_point}, {domain}")
            wrote.append(make_figure(obs_point, domain, nr_file, MEMBERS, scale, data_root))
    print("Wrote figures:")
    for path in wrote:
        print(f"  {path}")


if __name__ == "__main__":
    main()
