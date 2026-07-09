#!/usr/bin/env python3
"""
Plot prior covariance maps for diagnosing single-observation updates.

For one single-observation experiment, this script compares two covariance
patterns over a TC-centered square region:

1. cov(x_single, x_all): covariance between the state value at the observation
   location and each prior state grid point in the TC square.
2. cov(hx, x_all): covariance between prior ensemble H(x) at the observation
   and each prior state grid point in the TC square.
"""

from __future__ import annotations

import math
import os
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

from plot_singleobs_nr_compare import (
    ObsSeqInfo,
    auto_scale,
    find_mean_file,
    find_member_file,
    find_nr_file,
    find_obs_seq_info,
    latlon_offsets_km,
    read_field,
    read_grid_for_field,
    read_time_string,
    resolve_run_dir,
    tc_center_from_nr,
    tc_square_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figs" / "singleobs_prior_covariance"
DEFAULT_NR_BASE = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain")


# =============================================================================
# User configuration
# =============================================================================
DATA_ROOT = "/scratch/lililei1/kcfu/tc_mangkhut/4assimilation/DART"
FILTER_FOR_TIME = "EAKF"
nobs = 111
OBS_POINTS = [nobs]
DOMAINS = ["d01"]
MEMBERS = list(range(1, 51))

# Observation source used for y, H(x), and observation location.  This can be a
# single obs_seq file, a directory containing obs_seq.out.111-style files, or a
# dict such as {111: "/path/to/obs_seq.out.111", 325: "..."}.
OBS_SOURCE_PATH = "/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/run_dir"

# Prior member files used for x_single and x_all.
FIRSTGUESS_DIR = "/scratch/lililei1/kcfu/tc_mangkhut/4assimilation/DART/EAKF/obs_seq361"

VAR_NAME = "OM_TMP"
LEVEL = 0
SCALE = "auto"

NR_FILE = None
NR_BASE = DEFAULT_NR_BASE
NR_DOMAIN = "d02"
TIME_STRING = "2018-09-10_00:00:00"

TC_HALF_WIDTH_KM = 150.0
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
FIG_FORMAT = "png"
DPI = 300

FIRSTGUESS_PREFIXES = ["preassim", "firstguess", "input", "prior"]

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
class PriorStateData:
    members: list[int]
    x_single: np.ndarray
    x_all: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    region_mask: np.ndarray


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


def resolve_obs_source_local(obs_point: int, fallback_run_dir: Path) -> Path:
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

    for name in (f"obs_seq.out.{obs_point}", f"obs_seq{obs_point}", f"obs_seq.out{obs_point}"):
        point_child = source / name
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


def obs_info_for_point(obs_point: int, fallback_run_dir: Path) -> ObsSeqInfo:
    obs_source = resolve_obs_source_local(obs_point, fallback_run_dir)
    obs_info = find_obs_seq_info(obs_source, obs_point)
    if obs_info is None:
        raise ValueError(f"Could not find observation metadata for obs_seq{obs_point} from {obs_source}")
    if obs_info.obs_value is None:
        raise ValueError(f"obs_seq{obs_point}: observation value y was not found")
    if obs_info.hx is None or obs_info.hx.size == 0:
        raise ValueError(f"obs_seq{obs_point}: external_FO H(x) ensemble was not found")
    if obs_info.lat is None or obs_info.lon is None:
        raise ValueError(f"obs_seq{obs_point}: observation location was not found")
    return obs_info


def nearest_grid_value(values: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat: float, lon: float) -> float:
    valid = np.isfinite(values) & np.isfinite(lats) & np.isfinite(lons)
    if not np.any(valid):
        return float("nan")
    radius_earth_km = 6371.0
    lat1_rad = np.radians(lats)
    lon1_rad = np.radians(lons)
    lat2_rad = math.radians(lat)
    lon2_rad = math.radians(lon)
    dlat = lat1_rad - lat2_rad
    dlon = lon1_rad - lon2_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * math.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    dist = radius_earth_km * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    j, i = np.unravel_index(np.nanargmin(np.where(valid, dist, np.nan)), values.shape)
    return float(values[j, i])


def read_prior_state_data(
    firstguess_dir: Path,
    domain: str,
    members: list[int],
    var_name: str,
    level: int | None,
    scale: float,
    obs_lat: float,
    obs_lon: float,
    tc_lat: float,
    tc_lon: float,
    half_width_km: float,
) -> PriorStateData:
    x_single_values: list[float] = []
    x_all_values: list[np.ndarray] = []
    used_members: list[int] = []
    lats: np.ndarray | None = None
    lons: np.ndarray | None = None
    region_mask: np.ndarray | None = None

    missing: list[int] = []
    for member in members:
        path = find_member_file(firstguess_dir, domain, member, FIRSTGUESS_PREFIXES)
        if path is None:
            missing.append(member)
            continue

        member_lats, member_lons = read_grid_for_field(path, var_name)
        field = read_field(path, var_name, level, scale)
        if field.shape != member_lats.shape:
            raise ValueError(f"{path}: {var_name} shape {field.shape} does not match grid {member_lats.shape}")

        if lats is None or lons is None:
            lats = member_lats
            lons = member_lons
            region_mask = tc_square_mask(lats, lons, tc_lat, tc_lon, half_width_km)
            if not np.any(region_mask):
                raise ValueError(f"No {domain} grid points inside {half_width_km:g} km TC square")
        elif field.shape != lats.shape:
            raise ValueError(f"{path}: member grid shape changed from {lats.shape} to {field.shape}")

        x_single_values.append(nearest_grid_value(field, member_lats, member_lons, obs_lat, obs_lon))
        x_all_values.append(np.where(region_mask, field, np.nan))
        used_members.append(member)

    if missing:
        warnings.warn(f"{firstguess_dir}: missing {len(missing)} prior member files for {domain}: {missing[:8]}")
    if len(used_members) < 2:
        raise ValueError(f"Need at least 2 prior members, found {len(used_members)}")

    assert lats is not None and lons is not None and region_mask is not None
    return PriorStateData(
        members=used_members,
        x_single=np.asarray(x_single_values, dtype=float),
        x_all=np.stack(x_all_values, axis=0),
        lats=lats,
        lons=lons,
        region_mask=region_mask,
    )


def align_hx_to_members(hx: np.ndarray, configured_members: list[int], used_members: list[int]) -> np.ndarray:
    hx = np.asarray(hx, dtype=float)
    if hx.size < len(configured_members):
        warnings.warn(f"H(x) ensemble has {hx.size} values, configured MEMBERS has {len(configured_members)}")
    member_to_hx = {
        member: hx[index]
        for index, member in enumerate(configured_members[: hx.size])
    }
    missing = [member for member in used_members if member not in member_to_hx]
    if missing:
        raise ValueError(f"H(x) is missing values for used members: {missing[:8]}")
    return np.asarray([member_to_hx[member] for member in used_members], dtype=float)


def covariance_map(vector: np.ndarray, field_cube: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    cube = np.asarray(field_cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("field_cube must have shape (member, y, x)")
    if vector.size != cube.shape[0]:
        raise ValueError(f"vector length {vector.size} does not match cube member count {cube.shape[0]}")

    v = vector[:, None, None]
    valid = np.isfinite(v) & np.isfinite(cube)
    count = np.sum(valid, axis=0)
    safe_count = np.where(count > 0, count, 1)
    v_mean = np.sum(np.where(valid, v, 0.0), axis=0) / safe_count
    x_mean = np.sum(np.where(valid, cube, 0.0), axis=0) / safe_count
    anomalies = np.where(valid, (v - v_mean) * (cube - x_mean), 0.0)
    cov = np.sum(anomalies, axis=0) / np.where(count > 1, count - 1, 1)
    cov[count <= 1] = np.nan
    return cov


def symmetric_levels(*fields: np.ndarray, nlevels: int = 21) -> np.ndarray:
    finite = [field[np.isfinite(field)] for field in fields]
    finite = [field for field in finite if field.size > 0]
    if not finite:
        return np.linspace(-1.0, 1.0, nlevels)
    vmax = float(np.nanmax(np.abs(np.concatenate(finite))))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return np.linspace(-vmax, vmax, nlevels)


def plot_covariance_maps(
    obs_point: int,
    domain: str,
    obs_info: ObsSeqInfo,
    prior_data: PriorStateData,
    cov_xsingle_xall: np.ndarray,
    cov_hx_xall: np.ndarray,
    tc_lat: float,
    tc_lon: float,
    tc_pressure: float,
    center_name: str,
) -> Path:
    configure_matplotlib()
    x_km, y_km = latlon_offsets_km(prior_data.lats, prior_data.lons, tc_lat, tc_lon)
    levels = symmetric_levels(cov_xsingle_xall, cov_hx_xall)
    cmap = "RdBu_r"

    fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.1), constrained_layout=True)
    panels = [
        (axs[0], cov_xsingle_xall, r"cov($x_{single}$, $x_{all}$)"),
        (axs[1], cov_hx_xall, r"cov($H(x)$, $x_{all}$)"),
    ]
    contour = None
    for ax, field, title in panels:
        contour = ax.contourf(x_km, y_km, field, levels=levels, cmap=cmap, extend="both")
        ax.contour(x_km, y_km, prior_data.region_mask.astype(float), levels=[0.5], colors="0.25", linewidths=0.7)
        obs_x, obs_y = latlon_offsets_km(
            np.asarray([[obs_info.lat]], dtype=float),
            np.asarray([[obs_info.lon]], dtype=float),
            tc_lat,
            tc_lon,
        )
        ax.scatter(0.0, 0.0, s=44, marker="+", color="black", linewidths=1.2, label="TC center")
        ax.scatter(float(obs_x[0, 0]), float(obs_y[0, 0]), s=38, marker="v", color="#d73027", label="obs")
        ax.set_title(title)
        ax.set_xlabel("x distance from TC center (km)")
        ax.set_ylabel("y distance from TC center (km)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-TC_HALF_WIDTH_KM, TC_HALF_WIDTH_KM)
        ax.set_ylim(-TC_HALF_WIDTH_KM, TC_HALF_WIDTH_KM)
        ax.grid(True, color="0.88", lw=0.5)
        ax.legend(loc="best", fontsize=7)

    assert contour is not None
    cbar = fig.colorbar(contour, ax=axs, shrink=0.88, pad=0.02)
    cbar.set_label(f"sample covariance of {VAR_LABELS.get(VAR_NAME, VAR_NAME)}")
    fig.suptitle(
        (
            f"Single obs {obs_point}, {domain}, {VAR_NAME}"
            f"{'' if LEVEL is None else f' level {LEVEL}'} | "
            f"y={obs_info.obs_value:.3g}; members={len(prior_data.members)}; "
            f"TC center from {center_name}={tc_pressure:.1f}"
        ),
        fontsize=9,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"singleobs{obs_point}_{domain}_{VAR_NAME}_lev{LEVEL if LEVEL is not None else '2d'}_prior_covariance"
    fig.savefig(f"{stem}.{FIG_FORMAT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(f".{FIG_FORMAT}")


def make_figure(obs_point: int, domain: str, nr_file: Path, scale: float, data_root: Path) -> Path:
    run_dir = resolve_run_dir(data_root, FILTER_FOR_TIME, obs_point)
    obs_info = obs_info_for_point(obs_point, run_dir)
    tc_lat, tc_lon, tc_pressure, center_name = tc_center_from_nr(nr_file)
    firstguess_dir = resolve_firstguess_dir(domain)

    prior_data = read_prior_state_data(
        firstguess_dir=firstguess_dir,
        domain=domain,
        members=MEMBERS,
        var_name=VAR_NAME,
        level=LEVEL,
        scale=scale,
        obs_lat=float(obs_info.lat),
        obs_lon=float(obs_info.lon),
        tc_lat=tc_lat,
        tc_lon=tc_lon,
        half_width_km=TC_HALF_WIDTH_KM,
    )
    hx = align_hx_to_members(obs_info.hx, MEMBERS, prior_data.members)
    cov_xsingle_xall = covariance_map(prior_data.x_single, prior_data.x_all)
    cov_hx_xall = covariance_map(hx, prior_data.x_all)
    return plot_covariance_maps(
        obs_point,
        domain,
        obs_info,
        prior_data,
        cov_xsingle_xall,
        cov_hx_xall,
        tc_lat,
        tc_lon,
        tc_pressure,
        center_name,
    )


def main() -> None:
    data_root = default_data_root() if DATA_ROOT is None else Path(DATA_ROOT)
    nr_file_config = None if NR_FILE is None else Path(NR_FILE)
    scale = auto_scale(VAR_NAME) if SCALE == "auto" else float(SCALE)

    first_run = resolve_run_dir(data_root, FILTER_FOR_TIME, OBS_POINTS[0])
    first_mean = find_mean_file(first_run, DOMAINS[0])
    inferred_time = TIME_STRING or read_time_string(first_mean)
    nr_file = find_nr_file(nr_file_config, Path(NR_BASE), NR_DOMAIN, inferred_time)

    print(f"Using NR file: {nr_file}")
    wrote: list[Path] = []
    for obs_point in OBS_POINTS:
        for domain in DOMAINS:
            print(f"Plotting covariance maps for obs_seq{obs_point}, {domain}")
            wrote.append(make_figure(obs_point, domain, nr_file, scale, data_root))
    print("Wrote figures:")
    for path in wrote:
        print(f"  {path}")


if __name__ == "__main__":
    main()
