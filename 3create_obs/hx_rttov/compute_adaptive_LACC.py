"""
compute_adaptive_LACC.py -- per-observation adaptive LACC time weighting.

Called by 4assimilation/1convert_obs/run_single_assimilation_test.sh only when
EXPERIMENT_MODE=LACC and OptimizWeight=1, after the per-lag (un-averaged)
truth BT, member Hx and per-lag clear-sky masks already exist (produced by
run_rttov_TrueObs_LACC_driver.sh / run_rttov_ensBT_LACC_driver.sh), and
before obs2DART_LACC.py / merge_FO_in_one_file.py / text_to_obs.

Pipeline implemented here (position filtering only -- NO time filtering
inside kept positions):

  1. read per-lag truth BT, all-member per-lag Hx, per-lag clear-sky masks;
  2. build the unified LACC position mask with the SAME rule as
     average_LACC_obs.m (keep point if clear in >= floor(frac * T) lag
     times, at least 1);
  3. filter all arrays with one keep-index I in original order;
  4. optimize the time weights against the SST target according to
     cfg.optimization_mode (ADAPTIVE_LACC_OPTIMIZATION_MODE):
       * "joint_sst_field" (default): ONE JOINT optimization of the whole
         T x q weight table against the area-weighted per-gridpoint SST
         squared error over the target region (derivation doc sections
         21-23).  The objective is the ensemble-estimation variance
         reduction J(Omega) = tr[A C W^T (W S W^T)^{-1} W C^T], evaluated in
         ensemble space via B = I + sum_l z_l z_l^T / d_l, maximized with
         projected-gradient (column-wise Euclidean simplex projection,
         Armijo backtracking, multiple fixed-seed initializations).  All
         positions share one objective; cross-covariances between
         positions are kept.  No localization, no serial-EAKF preview, no
         background/SST update during iterations; this is an idealized
         joint linear-compression benchmark, NOT a claimed DART optimum.
       * "independent_sst_mean" (legacy, explicit): area-mean SST scalar
         target, each position solved independently (exact active-set
         enumeration for T <= 8, bounded grid search for T > 8).  Kept for
         comparison; its diagnostics keep the original per-position
         contract.
  5. write weighted obs (with noise), weighted per-member Hx, per-obs
     error VARIANCE, weights (.npz) and diagnostics.  The joint mode adds
     adaptive_lacc_joint_summary.json + adaptive_lacc_optimization_history
     .csv and writes schema_version=2 npz / position-only CSV.

SST target data: analysis-time background that enters DART (default member
naming firstguess_d01.memNNN, configurable via sst_member_pattern /
sst_land_mask_water_above for e.g. DART preassim_member_0001_d01.nc members
with an XLAND land/water field).  Both modes use the same region, variable,
level, ocean mask and cos(lat) area weights (area_weight_method="coslat_proxy":
a latitude proxy for area on a regular lat/lon grid, NOT a verified WRF-grid
true-area model).  The joint mode does NOT average SST beforehand and never
normalizes gridpoints by their own spread.

The weight objective NEVER sees NR errors, real innovations or the applied
noise samples; NR and noise only enter the final weighted-observation
generation.

Error model: nominal diagonal R per time, sigma^2 = OBS_ERR_STD^2,
independent in time, so R_L,l = sigma^2 * sum_n w_n,l^2.  Perturbation
noise follows the average_LACC_obs.m construction epsilon = sigma *
(z - spatial mean of z): standard normal z on the full q_raw grid,
demeaned per lag time over the RAW grid, scaled by sigma, column-major
flattened and THEN filtered by I (not re-centered on q_kept).  The
spatial demeaning realizes sigma^2*(1-1/q_raw) per-point variance and a
spatial error correlation that the diagonal DART representation does NOT
model -- recorded in the .npz.  Noise is generated with numpy
default_rng(PCG64): an identical integer seed does NOT reproduce the legacy
MATLAB randn stream, so this is a NEW noise realization relative to the
old equal-weight experiments; the applied noise (in K) is persisted in the
.npz so this run is exactly reproducible.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np

# =====================
# User configuration
# =====================
@dataclass(frozen=True)
class Config:
    # --- paths (cluster defaults; env vars set by the shell driver override) ---
    # Optional paths use None (never Path("")): Path("") IS Path(".") which is
    # truthy, so it can neither mark "unset" nor trigger the derived defaults.
    hx_dir: Path = Path("/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov")
    obs_bt_dir: Path | None = None   # default: {hx_dir}/3obs_BT_LACC/{sensor}
    ens_bt_dir: Path | None = None   # default: {hx_dir}/4ens_BT_LACC (RAW per-lag Hx; env RAW_ENS_BT_DIR)
    para_file: Path | None = None    # center-time RTTOV profile (lat/lon metadata)
    lacc_times_file: Path | None = None  # center_time=/lag_time= records
    combined_mask_file: Path | None = None  # BT_LACC_<center>/clear_sky_mask.txt (cross-check)
    sst_bg_dir: Path | None = None   # default: analysis-time background that enters DART
    output_dir: Path | None = None   # default: {hx_dir}/adaptive_LACC/{center}_ch{channel}

    # --- experiment shape ---
    domain: str = "d01"
    sensor: str = "AMSUA"
    channel: int = 4
    nobs_raw: int = 676        # q_raw
    npoint: int = 0            # noise grid side; 0 -> derived as isqrt(q_raw)
    ens_size: int = 50
    obs_err_std: float = 0.5   # sigma per time per position (K)

    # --- LACC window ---
    center_time: str = ""      # e.g. "10_00_00"
    lag_times: tuple[str, ...] = ()  # e.g. ("10_00_00", "09_21_00", "09_18_00")
    clear_sky_min_fraction: float = 2.0 / 3.0  # same default as average_LACC_obs.m
    reference_year: int = 2018
    reference_month: int = 9
    mask_crosscheck: bool = True  # recompute the mask, then require it to match
                                  # the existing BT_LACC_<center>/clear_sky_mask.txt

    # --- SST target (version 1: area mean of OM_TMP level 0 over ocean box) ---
    sst_target_id: str = "OM_TMP_LEVEL0"
    sst_var: str = "OM_TMP"
    sst_level: int = 0
    # box as (lat_min, lat_max, lon_min, lon_max); user-confirmed obs-swath box
    sst_region: tuple[float, float, float, float] = (13.2, 15.0, 147.2, 149.1)
    sst_lat_var: str = "XLAT"
    sst_lon_var: str = "XLONG"
    sst_land_mask_var: str = "LANDMASK"  # 1 = land, 0 = water
    # member file NAME pattern inside sst_bg_dir.  {domain} and {mem} are
    # substituted (mem is the raw 1-based member integer); use {mem:04d} for
    # zero padding.  The legacy default reproduces firstguess_d01.memNNN.
    sst_member_pattern: str = "firstguess_{domain}.mem{mem:03d}"
    # water criterion.  None keeps the legacy LANDMASK convention (water where
    # mask < 0.5).  For WRF XLAND (1 = land, 2 = water) set e.g. 1.5: water
    # where mask >= 1.5.
    sst_land_mask_water_above: float | None = None

    # --- optimizer ---
    coarse_step_for_t: tuple[tuple[int, float], ...] = (
        (1, 0.0), (2, 0.01), (3, 0.02), (4, 0.05), (5, 0.1),
    )
    coarse_step_large_t: float = 0.2  # used for T >= 6
    refine_passes: int = 2            # each pass: radius 4*step, finer step

    # --- weight optimization mode ---
    # "joint_sst_field": ONE joint optimization of the whole T x q weight
    #   table against the area-weighted per-gridpoint SST squared error
    #   (derivation doc sections 21-23).  Default when adaptive weighting is
    #   enabled.
    # "independent_sst_mean": legacy mode -- area-mean SST scalar target,
    #   each position solved independently (exact for T <= 8).  Kept for
    #   comparison; diagnostics keep the original per-position contract.
    optimization_mode: str = "joint_sst_field"
    # seed for the joint optimizer's random initializations (independent of
    # the observation-noise seed on purpose)
    optimization_seed: int = 20260911

    # --- perturbation noise ---
    noise_seed: int = 20260910
    noise_generator: str = "numpy.random.default_rng(PCG64)"


# environment-variable overrides used when driven by run_single_assimilation_test.sh
ENV_MAPPING: dict[str, str] = {
    "hx_dir": "HX_DIR",
    "obs_bt_dir": "OBS_BT_DIR",
    # RAW per-lag member Hx root.  Deliberately NOT ENS_BT_DIR: the driver
    # repoints ENS_BT_DIR to the adaptive members tree only AFTER this script
    # has finished, so its meaning changes across stages.
    "ens_bt_dir": "RAW_ENS_BT_DIR",
    "para_file": "PARA_FILE",
    "lacc_times_file": "LACC_TIMES_FILE",
    "combined_mask_file": "CLEAR_SKY_MASK_FILE",
    "sst_bg_dir": "ADAPTIVE_SST_BG_DIR",
    "output_dir": "ADAPTIVE_LACC_DIR",
    "domain": "domain",
    "sensor": "sensor",
    "channel": "assim_channel",
    "nobs_raw": "NOBS",
    "ens_size": "ENS_SIZE",
    "obs_err_std": "OBS_ERR_STD",
    "clear_sky_min_fraction": "CLEAR_SKY_MIN_FRACTION",
    "sst_target_id": "ADAPTIVE_SST_TARGET_ID",
    "sst_member_pattern": "ADAPTIVE_SST_MEMBER_PATTERN",
    "sst_land_mask_var": "ADAPTIVE_SST_LAND_MASK_VAR",
    "sst_land_mask_water_above": "ADAPTIVE_SST_LAND_MASK_WATER_ABOVE",
    "optimization_mode": "ADAPTIVE_LACC_OPTIMIZATION_MODE",
    "optimization_seed": "ADAPTIVE_LACC_OPTIMIZATION_SEED",
    "noise_seed": "ADAPTIVE_NOISE_SEED",
    "mask_crosscheck": "ADAPTIVE_LACC_MASK_CROSSCHECK",
}

VALID_OPTIMIZATION_MODES = ("joint_sst_field", "independent_sst_mean")

# config fields that are paths -- decided by name, NOT by the default value's
# type (optional paths default to None, so isinstance(default, Path) fails)
PATH_FIELDS = frozenset({
    "hx_dir", "obs_bt_dir", "ens_bt_dir", "para_file", "lacc_times_file",
    "combined_mask_file", "sst_bg_dir", "output_dir",
})


def load_config() -> Config:
    values: dict[str, object] = {}
    for f in fields(Config):
        env_name = ENV_MAPPING.get(f.name)
        if not env_name:
            continue
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue  # unset or empty -> stays None -> derived default applies
        try:
            if f.name in PATH_FIELDS:
                values[f.name] = Path(raw)  # an explicit "." stays a valid path
            elif isinstance(f.default, bool):
                values[f.name] = raw == "1"
            elif isinstance(f.default, int):
                values[f.name] = int(raw)
            elif isinstance(f.default, float):
                values[f.name] = float(raw)
            elif f.name == "sst_land_mask_water_above":
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        except ValueError as exc:
            raise ValueError(f"bad value for {env_name}: {raw!r}") from exc

    # SST region box: "lat_min lat_max lon_min lon_max" (explicit config;
    # the area-mean target region never follows the clear-sky filtering)
    region_text = os.environ.get("ADAPTIVE_SST_REGION", "")
    if region_text:
        parts = region_text.split()
        if len(parts) != 4:
            raise ValueError(
                f"ADAPTIVE_SST_REGION must be 'lat_min lat_max lon_min lon_max', "
                f"got {region_text!r}"
            )
        values["sst_region"] = tuple(float(v) for v in parts)

    # resolve the LACC window first: derived default paths depend on center time
    center_time, lag_times = resolve_lacc_window()
    cfg = Config(center_time=center_time, lag_times=lag_times, **values)  # type: ignore[arg-type]

    # derived defaults -- None marks "unset"; every optional path gets a
    # concrete value here BEFORE any file/dir is validated by the loaders
    if cfg.npoint == 0:
        side = math.isqrt(cfg.nobs_raw)
        if side * side != cfg.nobs_raw:
            raise ValueError(
                f"cannot derive the noise grid side: q_raw={cfg.nobs_raw} is "
                f"not a perfect square; set npoint explicitly"
            )
        cfg = _replace(cfg, npoint=side)
    if cfg.obs_bt_dir is None:
        cfg = _replace(cfg, obs_bt_dir=cfg.hx_dir / "3obs_BT_LACC" / cfg.sensor)
    if cfg.ens_bt_dir is None:
        cfg = _replace(cfg, ens_bt_dir=cfg.hx_dir / "4ens_BT_LACC")
    if cfg.sst_bg_dir is None:
        base = os.environ.get("ADAPTIVE_BASE_DIR", "/share/home/lililei1/kcfu/tc_mangkhut")
        cfg = _replace(
            cfg,
            sst_bg_dir=Path(base) / "4assimilation" / "0mem_all_time" / "cyclingDA" / cfg.center_time,
        )
    if cfg.output_dir is None:
        cfg = _replace(
            cfg,
            output_dir=cfg.hx_dir / "adaptive_LACC" / f"{cfg.center_time}_ch{cfg.channel}",
        )
    if cfg.para_file is None:
        prof_dir = os.environ.get("PROFILE_DIR", str(cfg.hx_dir / "profile"))
        profile_subdir = os.environ.get(
            "PROFILE_SUBDIR", f"profile_{cfg.domain}_LACC_{cfg.center_time}"
        )
        day, hour, minute = cfg.center_time.split("_")
        cfg = _replace(
            cfg,
            para_file=Path(prof_dir) / profile_subdir / f"prof{day}_{hour}:{minute}.dat",
        )
    if cfg.combined_mask_file is None:
        cfg = _replace(
            cfg,
            combined_mask_file=(
                cfg.obs_bt_dir / f"BT_LACC_{cfg.center_time}" / "clear_sky_mask.txt"
            ),
        )
    if cfg.optimization_mode not in VALID_OPTIMIZATION_MODES:
        raise ValueError(
            f"optimization_mode must be one of {VALID_OPTIMIZATION_MODES}, "
            f"got {cfg.optimization_mode!r} (env ADAPTIVE_LACC_OPTIMIZATION_MODE)"
        )
    return cfg


def _replace(cfg: Config, **changes) -> Config:
    return replace(cfg, **changes)


# =====================
# input readers (strict: missing / wrong-length / non-finite inputs raise)
# =====================
def read_lacc_times(path: Path) -> tuple[str, tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"LACC times file does not exist: {path}")
    center = ""
    lags: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("center_time="):
                center = line.split("=", 1)[1].strip()
            elif line.startswith("lag_time="):
                lags.append(line.split("=", 1)[1].strip())
    if not center or not lags:
        raise ValueError(f"LACC times file malformed (need center_time= and lag_time=): {path}")
    expected = os.environ.get("EXPECTED_LACC_COUNT")
    if expected is not None and len(lags) != int(expected):
        raise ValueError(
            f"LACC times file has {len(lags)} lag times, EXPECTED_LACC_COUNT={expected}: {path}"
        )
    return center, tuple(lags)


def build_lacc_times_from_env() -> tuple[str, tuple[str, ...]]:
    """Mirror of build_lacc_times() in the shell drivers: only used when no
    LACC times file exists (standalone runs)."""
    day = int(os.environ.get("lacc_center_day", "10"))
    hour = int(os.environ.get("lacc_center_hour", "00"))
    minute = int(os.environ.get("lacc_center_min", "00"))
    center = f"{day:02d}_{hour:02d}_{minute:02d}"
    lags: list[str] = []
    for lag_text in os.environ.get("LACC_LAG_HOURS", "0 3 6").split():
        lag = int(lag_text)
        lag_day, lag_hour = day, hour - lag
        while lag_hour < 0:
            lag_hour += 24
            lag_day -= 1
        lags.append(f"{lag_day:02d}_{lag_hour:02d}_{minute:02d}")
    return center, tuple(lags)


def resolve_lacc_window() -> tuple[str, tuple[str, ...]]:
    """Preferred source: the on-disk LACC_times.txt written by
    average_LACC_obs.m (the same record obs2DART_LACC.py reads)."""
    path_text = os.environ.get("LACC_TIMES_FILE", "")
    if path_text and Path(path_text).is_file():
        return read_lacc_times(Path(path_text))
    if path_text:
        raise FileNotFoundError(
            f"LACC_TIMES_FILE is set but the file does not exist: {path_text} "
            f"(run the truth driver / average_LACC_obs.m first)"
        )
    center, lags = build_lacc_times_from_env()
    print(f"WARNING: LACC_TIMES_FILE not set; built LACC window from env: "
          f"center={center} lags={lags}")
    return center, lags


def lag_hours_of(cfg: Config, lag_time: str) -> float:
    day, hour, minute = (int(v) for v in lag_time.split("_"))
    lag_dt = datetime(cfg.reference_year, cfg.reference_month, day, hour, minute)
    day, hour, minute = (int(v) for v in cfg.center_time.split("_"))
    center_dt = datetime(cfg.reference_year, cfg.reference_month, day, hour, minute)
    return (center_dt - lag_dt).total_seconds() / 3600.0


def load_column(path: Path, expected: int, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    values = np.loadtxt(path, ndmin=1, dtype=float)
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"{label} has {values.size} rows, expected {expected}: {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains non-finite values: {path}")
    return values


def load_truth_obs(cfg: Config) -> np.ndarray:
    """(T, q_raw) un-averaged truth BT per lag time (NOT from BT_LACC_* averages)."""
    rows = []
    for lag in cfg.lag_times:
        path = cfg.obs_bt_dir / f"BT_{lag}" / f"obs_{cfg.domain}_ch{cfg.channel}_totalline.txt"
        rows.append(load_column(path, cfg.nobs_raw, f"truth BT {lag}"))
    return np.asarray(rows, dtype=float)


def load_clear_masks(cfg: Config) -> np.ndarray:
    """(T, q_raw) per-lag-time clear-sky masks (0/1)."""
    rows = []
    for lag in cfg.lag_times:
        path = cfg.obs_bt_dir / f"BT_{lag}" / "clear_sky_mask.txt"
        rows.append(load_column(path, cfg.nobs_raw, f"clear-sky mask {lag}"))
    masks = np.asarray(rows, dtype=float)
    if not np.all(np.isin(masks, [0.0, 1.0])):
        raise ValueError("per-lag clear-sky masks must contain only 0 and 1")
    return masks.astype(np.int8)


def load_ens_hx(cfg: Config) -> np.ndarray:
    """(T, N, q_raw) un-averaged member Hx per lag time."""
    members = member_numbers(cfg)
    out = np.empty((len(cfg.lag_times), cfg.ens_size, cfg.nobs_raw), dtype=float)
    for t, lag in enumerate(cfg.lag_times):
        for k, mem in enumerate(members):
            path = (
                cfg.ens_bt_dir / f"mem{mem:03d}" / cfg.sensor / f"BT_{lag}"
                / f"obs_{cfg.domain}_ch{cfg.channel}_totalline.txt"
            )
            out[t, k, :] = load_column(path, cfg.nobs_raw, f"member Hx mem{mem:03d} {lag}")
    return out


def member_numbers(cfg: Config) -> np.ndarray:
    return np.arange(1, cfg.ens_size + 1, dtype=np.int32)


def read_profile_latlon(para_file: Path, expected: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marker-based parsing: works for both RTTOV-standard and RTTOV-scatt
    profile layouts because it keys on the comment markers instead of
    hard-coded line numbers (obs2DART*.py keeps its own version-specific
    line-number parsing for the DART conversion)."""
    marker = "! Elevation (km), latitude and longitude (degrees)"
    surface_marker = "! Surface type (0=land, 1=sea, 2=sea-ice)"
    if not para_file.is_file():
        raise FileNotFoundError(f"profile file does not exist: {para_file}")
    lines = para_file.read_text(errors="replace").splitlines()
    lats, lons, surface = [], [], []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == marker:
            values = lines[i + 1].split()
            if len(values) < 3:
                raise ValueError(f"bad coordinate line after marker at line {i + 1}: {para_file}")
            lats.append(float(values[1]))
            lons.append(float(values[2]))
        elif stripped == surface_marker:
            values = lines[i + 1].split()
            if not values:
                raise ValueError(f"bad surface-type line at line {i + 1}: {para_file}")
            surface.append(int(round(float(values[0]))))
    if len(lats) != expected:
        raise ValueError(
            f"profile has {len(lats)} coordinates, expected {expected}: {para_file}"
        )
    lat = np.asarray(lats, dtype=float)
    lon = np.asarray(lons, dtype=float)
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        raise ValueError(f"profile coordinates contain non-finite values: {para_file}")
    surf = np.asarray(surface, dtype=np.int8) if len(surface) == expected else np.full(expected, -1, dtype=np.int8)
    return lat, lon, surf


# =====================
# unified LACC position mask (same rule/threshold/rounding as average_LACC_obs.m)
# =====================
def build_keep_mask(cfg: Config, masks_by_time: np.ndarray) -> np.ndarray:
    n_times = masks_by_time.shape[0]
    mask_sum = masks_by_time.sum(axis=0)
    required = int(math.floor(cfg.clear_sky_min_fraction * n_times))
    # guard copied from average_LACC_obs.m: for N=1, floor(2/3)=0 would keep
    # everything; require at least 1 clear time; never exceed n_times.
    required = max(1, min(required, n_times))
    keep = mask_sum >= required
    return keep.astype(np.int8)


def crosscheck_keep_mask(cfg: Config, keep_mask: np.ndarray) -> None:
    if not cfg.mask_crosscheck:
        print(f"Mask cross-check disabled (ADAPTIVE_LACC_MASK_CROSSCHECK=0); "
              f"using recomputed mask only.")
        return
    path = cfg.combined_mask_file
    if not path.is_file():
        raise FileNotFoundError(
            f"existing combined LACC mask not found for cross-check (run the truth "
            f"driver / average_LACC_obs.m first): {path}"
        )
    existing = load_column(path, cfg.nobs_raw, "combined clear-sky mask").astype(np.int8)
    if not np.array_equal(existing, keep_mask):
        diff = int(np.sum(existing != keep_mask))
        raise ValueError(
            f"recomputed LACC mask differs from existing {path} at {diff} points. "
            f"The existing equal-weight products were built with another rule or "
            f"another CLEAR_SKY_MIN_FRACTION; refusing to mix mask definitions. "
            f"Fix the fraction or set ADAPTIVE_LACC_MASK_CROSSCHECK=0 deliberately."
        )


# =====================
# SST target: area-mean of OM_TMP level 0 over the ocean box
# =====================
def _read_sst_member_fields(cfg: Config, mem: int) -> tuple:
    """Read ONE background member and return the raw 2D fields plus the
    region box / ocean / cos(lat) weights / finite support.  Shared by the
    scalar target (legacy mode) and the field target (joint mode); keeps
    the strict variable/shape/missing checks in one place."""
    lat_min, lat_max, lon_min, lon_max = cfg.sst_region
    path = cfg.sst_bg_dir / cfg.sst_member_pattern.format(
        domain=cfg.domain, mem=int(mem)
    )
    if not path.is_file():
        raise FileNotFoundError(f"SST background member file missing: {path}")
    with nc.Dataset(path) as ds:
        missing = [
            name
            for name in (cfg.sst_var, cfg.sst_lat_var, cfg.sst_lon_var, cfg.sst_land_mask_var)
            if name not in ds.variables
        ]
        if missing:
            raise KeyError(f"{path} is missing variables: {', '.join(missing)}")
        grid_lat = np.ma.filled(ds.variables[cfg.sst_lat_var][:], np.nan)
        grid_lon = np.ma.filled(ds.variables[cfg.sst_lon_var][:], np.nan)
        sst = np.ma.filled(ds.variables[cfg.sst_var][:], np.nan)
        land = np.ma.filled(ds.variables[cfg.sst_land_mask_var][:], np.nan)
    grid_lat = grid_lat.reshape(-1, grid_lat.shape[-1]) if grid_lat.ndim == 3 else grid_lat
    grid_lon = grid_lon.reshape(-1, grid_lon.shape[-1]) if grid_lon.ndim == 3 else grid_lon
    land = land.reshape(land.shape[-2:]) if land.ndim == 3 else land
    if sst.ndim == 4:
        sst_field = sst[0, cfg.sst_level]
    elif sst.ndim == 3:
        sst_field = sst[cfg.sst_level]
    else:
        raise ValueError(f"{cfg.sst_var} has unsupported dims {sst.shape}: {path}")
    if not (grid_lat.shape == grid_lon.shape == sst_field.shape == land.shape):
        raise ValueError(f"grid shape mismatch in {path}: lat {grid_lat.shape}, "
                         f"lon {grid_lon.shape}, sst {sst_field.shape}, land {land.shape}")
    box = (
        (grid_lat >= lat_min) & (grid_lat <= lat_max)
        & (grid_lon >= lon_min) & (grid_lon <= lon_max)
    )
    if cfg.sst_land_mask_water_above is None:
        ocean = box & (land < 0.5)          # legacy LANDMASK: 0 = water, 1 = land
    else:
        ocean = box & (land >= cfg.sst_land_mask_water_above)  # XLAND: 1=land, 2=water
    n_box = int(box.sum())
    if n_box < 10:
        raise ValueError(
            f"SST target box contains only {n_box} grid points -- region outside "
            f"the background domain? box=({lat_min},{lat_max},{lon_min},{lon_max})"
        )
    if int(ocean.sum()) == 0:
        raise ValueError("SST target box contains no ocean grid points")
    weights = np.cos(np.deg2rad(grid_lat))
    support = ocean & np.isfinite(sst_field) & np.isfinite(weights)
    if int(support.sum()) == 0:
        raise ValueError(f"SST target box has no finite {cfg.sst_var} values: {path}")
    return grid_lat, grid_lon, sst_field, land, box, ocean, weights, support


def _check_member_consistency(
    mem: int, ref_member: int,
    grid_lat: np.ndarray, grid_lon: np.ndarray,
    support: np.ndarray, ref_grid: tuple, ref_support: np.ndarray,
) -> None:
    """Every member must share the SAME grid and the SAME finite support:
    member-dependent missing values would silently change the target."""
    if grid_lat.shape != ref_grid[0].shape or not (
        np.array_equal(grid_lat, ref_grid[0]) and np.array_equal(grid_lon, ref_grid[1])
    ):
        raise ValueError(
            f"member {mem:03d} background grid differs from member "
            f"{ref_member:03d}; the SST target would not be comparable"
        )
    if not np.array_equal(support, ref_support):
        diff = int(np.sum(support != ref_support))
        raise ValueError(
            f"SST target support differs across members: member "
            f"{mem:03d} differs from member {ref_member:03d} at {diff} "
            f"grid points (member-dependent missing values). Refusing to "
            f"build a target over inconsistent supports; clean the backgrounds "
            f"or restrict the target region first."
        )


def sst_target_metadata(cfg: Config) -> dict:
    rule = (
        f"water where {cfg.sst_land_mask_var} < 0.5"
        if cfg.sst_land_mask_water_above is None
        else f"water where {cfg.sst_land_mask_var} >= {cfg.sst_land_mask_water_above}"
    )
    return {
        "target_id": cfg.sst_target_id,
        "sst_var": cfg.sst_var,
        "sst_level": int(cfg.sst_level),
        "region_lat_min": cfg.sst_region[0],
        "region_lat_max": cfg.sst_region[1],
        "region_lon_min": cfg.sst_region[2],
        "region_lon_max": cfg.sst_region[3],
        "lat_var": cfg.sst_lat_var,
        "lon_var": cfg.sst_lon_var,
        "land_mask_var": cfg.sst_land_mask_var,
        "land_water_rule": rule,
        "member_pattern": cfg.sst_member_pattern,
        "background_dir": str(cfg.sst_bg_dir),
        "area_weight_method": "coslat_proxy",
        "ensemble_size": int(cfg.ens_size),
    }


def load_sst_target(cfg: Config) -> np.ndarray:
    """LEGACY scalar target (independent_sst_mean mode): area-mean SST
    ensemble vector s (N,)."""
    members = member_numbers(cfg)
    s = np.empty(cfg.ens_size, dtype=float)
    ref_support: np.ndarray | None = None
    ref_member = -1
    ref_grid: tuple[np.ndarray, np.ndarray] | None = None
    for k, mem in enumerate(members):
        grid_lat, grid_lon, sst_field, _land, _box, _ocean, weights, support = (
            _read_sst_member_fields(cfg, mem)
        )
        # every member must be averaged over the SAME grid points and weights:
        # member-dependent missing values would silently change the target.
        if ref_support is None:
            ref_support = support
            ref_member = int(mem)
            ref_grid = (grid_lat, grid_lon)
        else:
            _check_member_consistency(mem, ref_member, grid_lat, grid_lon,
                                      support, ref_grid, ref_support)
        s[k] = float(np.sum(weights[support] * sst_field[support]) / np.sum(weights[support]))
    if not np.all(np.isfinite(s)):
        raise ValueError("SST target contains non-finite member values")
    return s


@dataclass(frozen=True)
class SstFieldTarget:
    """Ensemble SST FIELD over the fixed target gridpoints (joint mode)."""
    x: np.ndarray            # (N, p) ensemble members x target gridpoints
    flat_index: np.ndarray   # (p,) index into the C-order raveled model grid
    lat: np.ndarray          # (p,)
    lon: np.ndarray          # (p,)
    area_weights: np.ndarray  # (p,) normalized cos(lat) weights, sum = 1
    metadata: dict


def load_sst_field_target(cfg: Config) -> SstFieldTarget:
    """JOINT-mode target: the SST ensemble FIELD x (N, p) on the fixed
    target gridpoints, plus the gridpoint indices/coordinates and the
    normalized fixed area weights.  No spatial averaging is performed here
    and no gridpoint is normalized by its own spread.  All members share
    one grid, one finite support and one weight vector (strict checks)."""
    members = member_numbers(cfg)
    ref_support: np.ndarray | None = None
    ref_member = -1
    ref_grid: tuple[np.ndarray, np.ndarray] | None = None
    ref_latlon: tuple[np.ndarray, np.ndarray] | None = None
    ref_weights: np.ndarray | None = None
    columns: list[np.ndarray] = []
    for k, mem in enumerate(members):
        grid_lat, grid_lon, sst_field, _land, _box, _ocean, weights, support = (
            _read_sst_member_fields(cfg, mem)
        )
        if ref_support is None:
            ref_support = support
            ref_member = int(mem)
            ref_grid = (grid_lat, grid_lon)
            ref_weights = weights[support]
            ref_weights = ref_weights / ref_weights.sum()
            ref_latlon = (grid_lat[support], grid_lon[support])
            flat_index = np.flatnonzero(support.ravel())  # C-order raveled grid
        else:
            _check_member_consistency(mem, ref_member, grid_lat, grid_lon,
                                      support, ref_grid, ref_support)
        columns.append(sst_field[support].astype(float))
    # each element of `columns` is one member's (p,) row -> stack as ROWS:
    # x[k, j] = member k, target gridpoint j  (shape (N, p))
    x = np.asarray(columns, dtype=np.float64)
    if x.shape != (cfg.ens_size, int(ref_support.sum())):
        raise ValueError(
            f"SST field target assembled with wrong shape {x.shape}, expected "
            f"({cfg.ens_size}, {int(ref_support.sum())})"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("SST field target contains non-finite member values")
    return SstFieldTarget(
        x=x,
        flat_index=flat_index.astype(np.int64),
        lat=ref_latlon[0].astype(float),
        lon=ref_latlon[1].astype(float),
        area_weights=ref_weights.astype(float),
        metadata=sst_target_metadata(cfg),
    )


# =====================
# per-position weight optimization
# =====================
def _simplex_grid(n: int, step: float) -> np.ndarray:
    """All non-negative vectors of length n summing to 1 on a grid of the
    given step (last coordinate absorbs the remainder)."""
    if n == 1:
        return np.ones((1, 1), dtype=float)
    n_steps = int(round(1.0 / step))
    out: list[list[float]] = []

    def recurse(prefix: list[int], remaining: int) -> None:
        depth = len(prefix)
        if depth == n - 1:
            out.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            recurse(prefix + [value], remaining - value)

    recurse([], n_steps)
    grid = np.asarray(out, dtype=float) * step
    # numerical cleanup on the last coordinate
    grid[:, -1] = np.clip(1.0 - grid[:, :-1].sum(axis=1), 0.0, 1.0)
    return grid


def coarse_step(cfg: Config, n_times: int) -> float:
    for threshold, step in cfg.coarse_step_for_t:
        if n_times <= threshold:
            return step
    return cfg.coarse_step_large_t


def _refine_around(best: np.ndarray, step: float, max_candidates: int = 3000) -> np.ndarray:
    """Local simplex grid around best: offsets on the first n-1 coordinates
    within +/- 4*step (larger reach than the coarse step, so the search can
    travel away from a flat coarse optimum), last coordinate closes the sum.
    The per-axis resolution adapts to keep the candidate count bounded."""
    n = best.size
    radius = 4.0 * step
    per_axis = int(round(max_candidates ** (1.0 / max(1, n - 1))))
    per_axis = max(3, min(per_axis, int(round(2 * radius / (step / 25.0))) + 1))
    axis = np.linspace(-radius, radius, per_axis)
    if n == 2:
        first = axis.reshape(-1, 1)
        deltas = np.hstack([first, -first])
    else:
        grids = np.meshgrid(*([axis] * (n - 1)), indexing="ij")
        first = np.stack([g.ravel() for g in grids], axis=1)
        last = -first.sum(axis=1, keepdims=True)
        deltas = np.hstack([first, last])
    candidates = best[None, :] + deltas
    ok = np.all(candidates >= -1.0e-12, axis=1)
    candidates = candidates[ok]
    candidates = np.clip(candidates, 0.0, 1.0)
    candidates = candidates / candidates.sum(axis=1, keepdims=True)
    return candidates


MAX_ENUM_T = 8        # exact active-set enumeration bound (2^T-1 supports)
FEAS_TOL = 1.0e-12    # only tiny negative weights within this tolerance are clipped
TIE_REL_TOL = 1.0e-12  # relative tie tolerance on the max normalized candidate score


def _objective(c: np.ndarray, S: np.ndarray, w: np.ndarray) -> float:
    """J(w) = (c^T w)^2 / (w^T S w) in the ORIGINAL covariance scale."""
    denominator = float(w @ S @ w)
    if denominator <= 0.0 or not np.isfinite(denominator):
        return -np.inf
    return float(c @ w) ** 2 / denominator


def _objective_many(c: np.ndarray, S: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Vectorized _objective over rows of cand (T is small)."""
    quad = np.einsum("ij,jk,ik->i", cand, S, cand)
    lin = cand @ c
    out = np.full(cand.shape[0], -np.inf, dtype=float)
    ok = quad > 0.0
    out[ok] = lin[ok] ** 2 / quad[ok]
    return out


def _validate_position_inputs(Y: np.ndarray, s: np.ndarray, sigma2: float) -> None:
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(s))):
        raise ValueError(
            "non-finite Hx or SST target at a kept position; refusing to optimize"
        )
    if not (np.isfinite(sigma2) and sigma2 > 0.0):
        raise ValueError(
            "sigma^2 must be finite and positive; sigma=0 makes S singular and "
            "needs a dedicated treatment (no implicit regularization is added)"
        )


def _stationary_candidates(S: np.ndarray, chat: np.ndarray, n_times: int) -> np.ndarray:
    """Finite candidate set for the simplex problem.

    For every non-empty support set A solve S_A v_A = c_A (linear solve, no
    explicit inverse); if sum(v_A) is non-zero, w_A = v_A / sum(v_A); keep it
    when all components are non-negative up to FEAS_TOL (tiny negatives are
    clipped to zero and the vector renormalized).  v_A itself may be
    all-negative: normalization then yields the -v direction, which is an
    equally valid candidate because the objective squares the covariance
    term.  Mixed-sign w_A are infeasible; their face boundary is covered by
    the smaller supports.  The T vertices are added explicitly so supports
    whose stationary point vanished (c_i = 0) are still represented, plus
    the exact equal-weight candidate as reference/baseline.
    """
    cands: list[np.ndarray] = []
    for mask in range(1, 1 << n_times):
        idx = [i for i in range(n_times) if (mask >> i) & 1]
        S_A = S[np.ix_(idx, idx)]
        try:
            v = np.linalg.solve(S_A, chat[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(v)):
            continue
        v_scale = float(np.abs(v).max())
        v_sum = float(v.sum())
        if v_scale == 0.0 or abs(v_sum) <= 1.0e-12 * v_scale:
            continue  # stationary direction orthogonal to the simplex normal
        w_A = v / v_sum
        if float(w_A.min()) < -FEAS_TOL:
            continue  # clearly infeasible candidate: discard
        w_A = np.clip(w_A, 0.0, None)
        w_A = w_A / w_A.sum()
        w = np.zeros(n_times, dtype=float)
        w[idx] = w_A
        cands.append(w)
    for i in range(n_times):
        vertex = np.zeros(n_times, dtype=float)
        vertex[i] = 1.0
        cands.append(vertex)
    cands.append(np.full(n_times, 1.0 / n_times, dtype=float))
    return np.asarray(cands, dtype=float)


def _grid_search_candidates(
    S: np.ndarray, chat: np.ndarray, uniform: np.ndarray, n_times: int, cfg: Config
) -> np.ndarray:
    """Fallback for n_times > MAX_ENUM_T: coarse simplex grid + refinement,
    scored with the scale-free normalized objective.  Returns the final
    candidate pool (the exact optimum is NOT guaranteed)."""
    step = coarse_step(cfg, n_times)
    if step <= 0.0:
        return uniform.reshape(1, -1)
    pool = np.vstack([_simplex_grid(n_times, step), uniform.reshape(1, -1)])
    best_w = pool[int(np.argmax(_objective_many(chat, S, pool)))].copy()
    for _ in range(cfg.refine_passes):
        pool = np.vstack([_refine_around(best_w, step), best_w.reshape(1, -1)])
        best_w = pool[int(np.argmax(_objective_many(chat, S, pool)))].copy()
        step = step / 5.0
    return pool


def optimize_one_position(
    Y: np.ndarray, s: np.ndarray, sigma2: float, cfg: Config
) -> tuple[np.ndarray, float, float, bool, str, float, str]:
    """Y: (N, T) Hx of ALL lag times at ONE kept position (same member
    ordering as s).  Maximizes J(w) = (c^T w)^2 / (w^T S w) over the simplex
    w >= 0, sum(w) = 1 with c = Y'^T s'/(N-1), S = Y'^T Y'/(N-1) + sigma^2 I.

    Returns (weights, selected_score, equal_score, fallback, reason,
    max_candidate_score, solver).  selected_score is always recomputed with
    the ORIGINAL c, S and the RETURNED weights; max_candidate_score is the
    best score over the evaluated candidate pool (a separate diagnostic,
    never mixed with selected_score).  Candidate comparisons use the
    scale-free chat = c / max|c| and a purely relative tie tolerance, so an
    overall rescaling of the SST anomalies cannot change the selection;
    exact ties resolve to the candidate closest to equal weight.

    Per-position optimization: each kept observation location is solved
    independently -- this is NOT a joint optimum over all observations.
    """
    n_members, n_times = Y.shape
    _validate_position_inputs(Y, s, sigma2)
    uniform = np.full(n_times, 1.0 / n_times, dtype=float)

    dof = n_members - 1
    s_prime = s - s.mean()
    Y_prime = Y - Y.mean(axis=0)
    c = Y_prime.T @ s_prime / dof
    S = Y_prime.T @ Y_prime / dof + sigma2 * np.eye(n_times)
    S = 0.5 * (S + S.T)  # exact symmetry up to floating point
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(S))):
        raise ValueError(
            "non-finite covariance/cross-covariance at a kept position; "
            "refusing to fall back silently"
        )
    equal_score = _objective(c, S, uniform)

    if float(np.std(s, ddof=1)) == 0.0:
        return (uniform.copy(), equal_score, equal_score, True,
                "zero_target_spread", equal_score, "equal_fallback")
    c_abs_max = float(np.abs(c).max())
    if c_abs_max == 0.0:
        return (uniform.copy(), equal_score, equal_score, True,
                "zero_cross_covariance", equal_score, "equal_fallback")
    chat = c / c_abs_max  # scale-free candidate directions and comparisons

    if n_times == 1:
        w = np.ones(1, dtype=float)
        score = _objective(c, S, w)
        return w, score, equal_score, False, "single_time", score, "t1_direct"

    if n_times <= MAX_ENUM_T:
        cand = _stationary_candidates(S, chat, n_times)
        solver = f"active_set_enum_T{n_times}"
    else:
        cand = _grid_search_candidates(S, chat, uniform, n_times, cfg)
        solver = f"grid_search_T{n_times}"

    j_norm = _objective_many(chat, S, cand)
    j_norm_max = float(np.max(j_norm))
    tie_tol = TIE_REL_TOL * max(j_norm_max, 0.0)  # relative only: no absolute floor
    tied = np.flatnonzero(j_norm >= j_norm_max - tie_tol)
    distances = np.linalg.norm(cand[tied] - uniform[None, :], axis=1)
    w = cand[tied[int(np.argmin(distances))]].copy()

    selected_score = _objective(c, S, w)
    max_candidate_score = float(np.max(_objective_many(c, S, cand)))
    reason = "optimum_at_equal_weight" if np.allclose(w, uniform, atol=1.0e-9) else "ok"
    return w, selected_score, equal_score, False, reason, max_candidate_score, solver


def optimize_weights_all(
    hx_kept: np.ndarray, s: np.ndarray, sigma2: float, cfg: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """hx_kept: (T, N, q_kept), s: (N,).  Every kept position is optimized
    independently against the same SST target s (same member ordering).

    Returns (weights (T,q_kept), selected_scores, equal_scores,
    fallback_flags, reasons, max_candidate_scores, solver_methods)."""
    if not np.all(np.isfinite(hx_kept)):
        raise ValueError("member Hx contains non-finite values; refusing to optimize")
    if not np.all(np.isfinite(s)):
        raise ValueError("SST target contains non-finite values; refusing to optimize")
    if not (np.isfinite(sigma2) and sigma2 > 0.0):
        raise ValueError(
            "sigma^2 must be finite and positive; sigma=0 needs a dedicated "
            "singular-S treatment (no implicit regularization is added)"
        )
    n_times, _, q_kept = hx_kept.shape
    weights = np.empty((n_times, q_kept), dtype=float)
    selected_scores = np.empty(q_kept, dtype=float)
    equal_scores = np.empty(q_kept, dtype=float)
    fallback_flags = np.zeros(q_kept, dtype=np.int8)
    reasons: list[str] = []
    max_candidate_scores = np.empty(q_kept, dtype=float)
    solvers: list[str] = []

    for l in range(q_kept):
        Y = hx_kept[:, :, l].T  # (N, T)
        (w, a_score, e_score, fallback, reason, max_score, solver) = (
            optimize_one_position(Y, s, sigma2, cfg)
        )
        weights[:, l] = w
        selected_scores[l] = a_score
        equal_scores[l] = e_score
        fallback_flags[l] = 1 if fallback else 0
        reasons.append(reason or "ok")
        max_candidate_scores[l] = max_score
        solvers.append(solver)

    return (weights, selected_scores, equal_scores, fallback_flags,
            np.asarray(reasons), max_candidate_scores, np.asarray(solvers))


# =====================
# JOINT SST-field optimization (derivation doc sections 21-23)
# =====================
@dataclass(frozen=True)
class JointOptParams:
    """Projected-gradient / Armijo defaults from derivation doc section 23."""
    initial_step: float = 1.0
    armijo_c: float = 1.0e-4
    backtrack_ratio: float = 0.5
    max_backtracks: int = 40
    max_accepted_iters: int = 500
    alpha_min: float = 1.0e-12
    alpha_max: float = 1.0e6
    rpg_tol: float = 1.0e-7          # projected-gradient residual threshold
    obj_change_tol: float = 1.0e-10  # normalized-objective change threshold
    converged_streak: int = 5        # consecutive accepted iters meeting both
    n_dirichlet_inits: int = 2       # per-column Dirichlet(1) random starts
    comparison_tol: float = 1.0e-12  # machine-rounding-level accept tolerance

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


JOINT_PARAMS = JointOptParams()


@dataclass(frozen=True)
class JointProblem:
    """Fixed background statistics for the joint objective (built ONCE; the
    optimizer never updates SST/Hx -- only candidate weights change)."""
    Y: np.ndarray            # (N, T, q) standardized Hx anomalies (/(sigma*sqrt(N-1)))
    X: np.ndarray            # (N, p) standardized SST anomalies (/sqrt(N-1))
    L: np.ndarray            # (N, N) PSD factor, Ghat = L L^T
    G: np.ndarray            # (N, N) raw target Gram matrix X diag(a) X^T
    Ghat: np.ndarray         # (N, N) G / P0
    P0: float                # tr(G) = area-weighted SST variance, K^2
    sigma2: float            # (K)^2, nominal diagonal observation error variance
    area_weights: np.ndarray  # (p,)
    degenerate_reason: str | None = None  # e.g. "zero_target_spread"
    cross_zero: bool = False  # target field and ALL Hx anomalies uncorrelated


def prepare_joint_problem(
    x: np.ndarray, hx: np.ndarray, sigma2: float, area_weights: np.ndarray
) -> JointProblem:
    """Build the fixed joint-problem statistics (doc section 22).

    x: (N, p) SST ensemble FIELD; hx: (T, N, q) raw member Hx;
    area_weights: (p,) NON-normalized weights (normalized here);
    sigma: finite positive observation error std.

    All demeaning runs along the MEMBER axis.  float64 throughout.
    """
    x = np.asarray(x, dtype=np.float64)
    hx = np.asarray(hx, dtype=np.float64)
    if x.ndim != 2 or hx.ndim != 3:
        raise ValueError(f"expected x (N,p) and hx (T,N,q), got {x.shape} / {hx.shape}")
    n_members = x.shape[0]
    if hx.shape[1] != n_members:
        raise ValueError(
            f"member count mismatch: SST field has N={n_members}, Hx has "
            f"N={hx.shape[1]}; all times and the SST target must share the "
            f"same ensemble member ordering"
        )
    if n_members < 2:
        raise ValueError(f"N={n_members} < 2: anomalies are undefined")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(hx))):
        raise ValueError("non-finite SST field or Hx; refusing to optimize")
    if not (np.isfinite(sigma2) and sigma2 > 0.0):
        raise ValueError(
            "sigma^2 must be finite and positive; sigma=0 needs a dedicated "
            "singular-error treatment (no implicit regularization is added)"
        )
    a = np.asarray(area_weights, dtype=np.float64)
    if a.shape != (x.shape[1],):
        raise ValueError(f"area weights shape {a.shape} does not match p={x.shape[1]}")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("area weights must be finite and non-negative")
    a = a / a.sum()

    dof = n_members - 1
    X = (x - x.mean(axis=0, keepdims=True)) / math.sqrt(dof)          # (N, p)

    # Zero-target-spread detection BEFORE squaring anything: the member
    # spread is compared against the FIELD MAGNITUDE at the float64 rounding
    # level (1e-13 relative).  A field that is constant across members has a
    # mean-rounding residual of order eps*|x| whose square would otherwise
    # look like a tiny-but-"nonzero" P0.  This is a representability test,
    # deliberately NOT a fixed K^2 threshold: a real spread of even 1e-7 K on
    # a 300 K field is ~3e-10 relative, five orders above the cutoff, and is
    # still optimized normally.
    spread_max = float(np.max(np.abs(x - x.mean(axis=0, keepdims=True))))
    field_scale = max(float(np.max(np.abs(x))), 1.0)
    if spread_max == 0.0 or spread_max <= 1.0e-13 * field_scale:
        zero = np.zeros((n_members, n_members), dtype=np.float64)
        return JointProblem(
            Y=np.zeros((hx.shape[1], hx.shape[0], hx.shape[2]), dtype=np.float64),
            X=X, L=zero, G=zero, Ghat=zero,
            P0=0.0, sigma2=sigma2, area_weights=a,
            degenerate_reason="zero_target_spread",
        )

    G = (X * a[None, :]) @ X.T                                        # (N, N)
    G = 0.5 * (G + G.T)
    P0 = float(np.trace(G))

    Ghat = G / P0
    eigvals, eigvecs = np.linalg.eigh(Ghat)
    lam_max = float(eigvals.max())
    neg_tol = 1.0e-10 * max(1.0, lam_max)
    if float(eigvals.min()) < -neg_tol:
        raise ValueError(
            f"target Gram matrix Ghat is not positive semidefinite "
            f"(min eig {eigvals.min():.3e}); refusing to regularize silently"
        )
    lam_clipped = np.clip(eigvals, 0.0, None)
    L = eigvecs * np.sqrt(lam_clipped)[None, :]                       # Ghat = L L^T

    # standardized Hx anomalies: demean along MEMBERS (axis 1 of (T,N,q)),
    # / (sigma sqrt(N-1)), then store member-major (N, T, q) -- the layout
    # every joint-solver consumer expects (Y[:, t, l] is the member vector).
    Y = (hx - hx.mean(axis=1, keepdims=True)) / (math.sqrt(dof) * math.sqrt(sigma2))
    Y = np.ascontiguousarray(Y.transpose(1, 0, 2))

    # Global cross-covariance test at the float64 representability level:
    # a member-constant Hx tree leaves only mean-rounding residuals, whose
    # exact-zero test would fail spuriously.  Only a GLOBAL zero (every time
    # and position) triggers the fallback -- individual positions with zero
    # correlation are kept: they can still contribute jointly.
    cross = X.T @ Y.reshape(Y.shape[0], -1)                            # (p, Tq)
    hx_spread = float(np.max(np.abs(hx - hx.mean(axis=1, keepdims=True))))
    hx_scale = max(float(np.max(np.abs(hx))), 1.0)
    cross_zero = (not np.any(cross)) or (hx_spread <= 1.0e-13 * hx_scale)

    return JointProblem(
        Y=Y, X=X, L=L, G=G, Ghat=Ghat, P0=P0, sigma2=sigma2,
        area_weights=a, degenerate_reason=None, cross_zero=cross_zero,
    )


def joint_objective_and_gradient(
    weights: np.ndarray, problem: JointProblem
) -> tuple[float, np.ndarray | None]:
    """Normalized joint objective Jhat and its (T, q) gradient (doc 22.2-22.3).

    Jhat = 1 - tr(Ghat B^-1) computed in the POSITIVE form
        Jhat = sum_l dot((Z^T L)[l, :], Q[l, :]) / d_l
    with B = I + sum_l z_l z_l^T / d_l, V = B^-1 L, Q = Z^T V -- no
    subtraction of nearly equal quantities, no explicit inverse, no p x p
    matrix.  J (K^2) = P0 * Jhat.  Returns (jhat, grad); grad is None when
    the Cholesky solve fails numerically (the caller treats jhat = -inf as
    an unusable candidate).

    This evaluates the JOINT benefit over ALL positions: cross-covariances
    between observation positions are kept (Z enters through one shared B).
    """
    w = np.asarray(weights, dtype=np.float64)
    Y = problem.Y                       # (N, T, q)
    Z = np.einsum("tl,ntl->ln", w, Y)   # (q, N)
    d = np.einsum("tl,tl->l", w, w)     # (q,) in [1/T, 1] on the simplex
    B = np.eye(Y.shape[0]) + Z.T @ (Z / d[:, None])
    B = 0.5 * (B + B.T)
    try:
        cf = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        return -np.inf, None
    V = _cho_solve(problem.L, cf)       # V = B^-1 L (triangular solves)
    ZtL = Z @ problem.L                 # (q, N)
    Q = Z @ V                           # (q, N)  Q[l,:] = z_l^T V = (V^T z_l)^T
    jhat = float(np.sum(ZtL * Q / d[:, None]))
    if not np.isfinite(jhat):
        return -np.inf, None

    # gradient: F = B^-1 Ghat B^-1 = V V^T, applied without forming F:
    #   F z_l = V (V^T z_l) = V Q[l, :]
    FZ = V @ Q.T                        # (N, q), column l = F z_l
    quad = np.einsum("nl,nl->l", Z.T, FZ)             # z_l^T F z_l
    YtFZ = np.einsum("ntl,nl->tl", Y, FZ)             # (T, q): Y[:,t,l]^T F z_l
    grad = 2.0 * YtFZ / d[None, :] - 2.0 * w * quad[None, :] / (d[None, :] ** 2)
    return jhat, grad


def _cho_solve(rhs: np.ndarray, cf: np.ndarray) -> np.ndarray:
    """Solve (C C^T) X = rhs given the Cholesky factor C (no explicit inverse)."""
    return np.linalg.solve(cf.T, np.linalg.solve(cf, rhs))


def project_columns_simplex(V: np.ndarray) -> np.ndarray:
    """Euclidean projection of each COLUMN onto the probability simplex
    (sorted-threshold method): w_i = max(v_i - tau, 0) with the unique tau
    making the column sum to 1.  NOT clip-and-renormalize, NOT softmax --
    exact zero weights are allowed."""
    V = np.asarray(V, dtype=np.float64)
    out = np.empty_like(V)
    for c in range(V.shape[1]):
        v = V[:, c]
        u = np.sort(v)[::-1]
        css = np.cumsum(u)
        idx = np.arange(1, v.size + 1)
        cond = u * idx > (css - 1.0)
        if not cond.any():
            # only possible for pathological (inf/nan) input; uniform fallback
            out[:, c] = 1.0 / v.size
            continue
        rho = int(np.flatnonzero(cond)[-1])
        tau = (css[rho] - 1.0) / (rho + 1)
        out[:, c] = np.maximum(v - tau, 0.0)
    return out


def _joint_rpg(weights: np.ndarray, grad: np.ndarray) -> float:
    """Projected-gradient residual at FIXED reference step 1 (doc 23.1);
    independent of the backtracking step size."""
    return float(np.max(np.abs(project_columns_simplex(weights + grad) - weights)))


def optimize_joint_weights(
    problem: JointProblem, cfg: Config, params: JointOptParams = JOINT_PARAMS
) -> dict:
    """Multi-start projected-gradient maximization of the joint normalized
    objective over the T x q column-simplex (doc section 23).

    Returns a dict with the final weights and full diagnostics (per-init
    records, accepted-iteration history, statuses, seeds).  The returned
    score is always RECOMPUTED from the returned weights; ties resolve to
    the candidate closest to equal weight; the solution is never allowed
    below the retained equal-weight baseline.  Background statistics stay
    fixed: evaluations never touch SST/Hx, only candidate weights change.
    """
    n_times, q = problem.Y.shape[1], problem.Y.shape[2]
    uniform = np.full((n_times, q), 1.0 / n_times, dtype=np.float64)
    accept_tol = params.comparison_tol

    def evaluate(w):
        return joint_objective_and_gradient(w, problem)

    if q == 0:
        raise RuntimeError("no retained observation positions (q=0)")
    if n_times == 1:
        w = np.ones((1, q), dtype=np.float64)
        jhat, _ = evaluate(w)
        return {
            "weights": w, "jhat": jhat, "j_phys": problem.P0 * jhat,
            "rpg": 0.0, "status": "t1_direct", "selected_source": "t1_direct",
            "equal_jhat": jhat, "equal_j_phys": problem.P0 * jhat,
            "improvement_jhat": 0.0,
            "per_init": [{"id": 0, "type": "t1_direct", "status": "t1_direct",
                          "initial_jhat": jhat, "final_jhat": jhat,
                          "best_jhat": jhat, "accepted_iters": 0,
                          "total_backtracks": 0}],
            "history": [], "params": params,
            "seed": int(cfg.optimization_seed),
            "degenerate_reason": problem.degenerate_reason,
        }

    equal_jhat, _ = evaluate(uniform)

    # degenerate target or globally zero cross-covariance -> equal weights
    if problem.degenerate_reason is not None:
        return _equal_weight_result(problem, cfg, uniform, equal_jhat,
                                    problem.degenerate_reason)
    if problem.cross_zero:
        return _equal_weight_result(problem, cfg, uniform, equal_jhat,
                                    "zero_cross_covariance")

    # ----- initializations (fixed optimizer seed, independent of noise seed)
    rng = np.random.default_rng(cfg.optimization_seed)
    inits: list[tuple[str, np.ndarray]] = [("equal", uniform.copy())]
    for t in range(n_times):
        vertex = np.zeros((n_times, q), dtype=np.float64)
        vertex[t, :] = 1.0
        inits.append((f"vertex_{t}", vertex))
    for k in range(params.n_dirichlet_inits):
        draws = rng.dirichlet(np.ones(n_times), size=q).T  # (T, q)
        inits.append((f"dirichlet_{k + 1}", draws))

    per_init: list[dict] = []
    history: list[dict] = []
    pool: list[tuple[float, np.ndarray, str]] = []  # (jhat, w, source)

    for init_id, (init_type, w) in enumerate(inits):
        jhat, grad = evaluate(w)
        if not np.isfinite(jhat):
            per_init.append({
                "id": init_id, "type": init_type, "status": "evaluate_failed",
                "initial_jhat": None, "final_jhat": None, "best_jhat": None,
                "accepted_iters": 0, "total_backtracks": 0,
            })
            continue
        initial_jhat = jhat
        best_w, best_j = w.copy(), jhat
        rpg = _joint_rpg(w, grad)
        if rpg <= params.rpg_tol:
            per_init.append({
                "id": init_id, "type": init_type, "status": "stationary_initial",
                "initial_jhat": jhat, "final_jhat": jhat, "best_jhat": best_j,
                "accepted_iters": 0, "total_backtracks": 0,
            })
            pool.append((best_j, best_w, f"{init_type}:stationary_initial"))
            continue

        alpha = params.initial_step
        prev_j = jhat
        streak = 0
        accepted = 0
        total_backtracks = 0
        status = "max_iter"
        for _accepted_iter in range(params.max_accepted_iters):
            backtracks = 0
            accepted_step = False
            while True:
                trial = project_columns_simplex(w + alpha * grad)
                delta = trial - w
                j_trial, grad_trial = evaluate(trial)
                if (np.isfinite(j_trial) and
                        j_trial >= jhat + params.armijo_c * float(np.sum(grad * delta))
                        - accept_tol * max(1.0, abs(jhat))):
                    accepted_step = True
                    break
                alpha *= params.backtrack_ratio
                backtracks += 1
                total_backtracks += 1
                if backtracks >= params.max_backtracks or alpha < params.alpha_min:
                    break
            if not accepted_step:
                status = "line_search_failed"
                break

            w, jhat, grad = trial, j_trial, grad_trial
            accepted_alpha = alpha
            alpha = min(2.0 * alpha, params.alpha_max)
            accepted += 1
            if jhat > best_j:
                best_j, best_w = jhat, w.copy()
            rpg = _joint_rpg(w, grad)
            history.append({
                "init_id": init_id, "init_type": init_type,
                "accepted_iter": accepted, "jhat": jhat,
                "j_phys": problem.P0 * jhat, "alpha": accepted_alpha, "rpg": rpg,
                "backtracks": backtracks,
            })

            delta_j = abs(jhat - prev_j)
            prev_j = jhat
            streak = (streak + 1
                      if (rpg <= params.rpg_tol and delta_j <= params.obj_change_tol)
                      else 0)
            if streak >= params.converged_streak:
                status = "converged"
                break
        else:
            status = "max_iter"

        final_j, _ = evaluate(w)
        per_init.append({
            "id": init_id, "type": init_type, "status": status,
            "initial_jhat": initial_jhat, "final_jhat": final_j,
            "best_jhat": best_j, "accepted_iters": accepted,
            "total_backtracks": total_backtracks,
        })
        pool.append((best_j, best_w, f"{init_type}:best"))
        if not np.allclose(w, best_w, atol=0.0):
            pool.append((final_j, w.copy(), f"{init_type}:final"))

    # ----- final selection: recompute scores, ties -> closest to equal weight
    candidates = [(equal_jhat, uniform.copy(), "equal_baseline")] + pool
    rescored = []
    for _j, w_cand, source in candidates:
        j_re, _ = evaluate(w_cand)
        rescored.append((j_re, w_cand, source))
    j_max = max(j for j, _w, _s in rescored)
    tie_tol = 1.0e-12 * max(1.0, abs(j_max))
    tied = [(j, w_cand, s) for j, w_cand, s in rescored if j >= j_max - tie_tol]
    dist_eq = [float(np.linalg.norm(w_cand - uniform)) for _j, w_cand, _s in tied]
    j_sel, w_sel, source_sel = tied[int(np.argmin(dist_eq))]

    if j_sel < equal_jhat - accept_tol * max(1.0, abs(equal_jhat)):
        # never output below the retained equal-weight baseline
        return _equal_weight_result(problem, cfg, uniform, equal_jhat,
                                    "below_equal_baseline")

    final_rpg = _joint_rpg(w_sel, evaluate(w_sel)[1])
    return {
        "weights": w_sel,
        "jhat": j_sel,
        "j_phys": problem.P0 * j_sel,
        "rpg": final_rpg,
        "status": "selected_from_multistart",
        "selected_source": source_sel,
        "equal_jhat": equal_jhat,
        "equal_j_phys": problem.P0 * equal_jhat,
        "improvement_jhat": j_sel - equal_jhat,
        "per_init": per_init,
        "history": history,
        "params": params,
        "seed": int(cfg.optimization_seed),
        "degenerate_reason": None,
    }


def _equal_weight_result(problem, cfg, uniform, equal_jhat, reason):
    grad = joint_objective_and_gradient(uniform, problem)[1] \
        if np.isfinite(equal_jhat) else np.zeros_like(uniform)
    return {
        "weights": uniform.copy(),
        "jhat": equal_jhat,
        "j_phys": problem.P0 * equal_jhat,
        "rpg": _joint_rpg(uniform, grad),
        "status": f"equal_fallback:{reason}",
        "selected_source": "equal_baseline",
        "equal_jhat": equal_jhat,
        "equal_j_phys": problem.P0 * equal_jhat,
        "improvement_jhat": 0.0,
        "per_init": [{"id": 0, "type": "equal", "status": f"equal_fallback:{reason}",
                      "initial_jhat": equal_jhat, "final_jhat": equal_jhat,
                      "best_jhat": equal_jhat, "accepted_iters": 0,
                      "total_backtracks": 0}],
        "history": [],
        "params": JOINT_PARAMS,
        "seed": int(cfg.optimization_seed),
        "degenerate_reason": reason,
    }


# =====================
# outputs
# =====================
def _write_common_products(
    cfg: Config,
    keep_index: np.ndarray,
    keep_mask_raw: np.ndarray,
    obs_kept: np.ndarray,
    noise_kept: np.ndarray,
    hx_kept: np.ndarray,
    weights: np.ndarray,
    error_variance: np.ndarray,
) -> np.ndarray:
    """Products shared by BOTH modes: weighted obs, weighted per-member Hx,
    error variance, original index, raw mask, LACC times.  Returns the
    weighted Hx (N, q_kept).  All files land in the SAME adaptive output
    directory; rows are q_kept everywhere and share one original order."""
    out = cfg.output_dir
    members_dir = out / "members"
    for mem in member_numbers(cfg):
        (members_dir / f"mem{mem:03d}" / cfg.sensor / f"BT_LACC_{cfg.center_time}").mkdir(
            parents=True, exist_ok=True
        )

    # weighted obs with the SHARED per-time noise (same noise for every
    # candidate weight set; generated on the raw grid, filtered by I)
    weighted_obs = np.sum(weights * obs_kept, axis=0) + np.sum(weights * noise_kept, axis=0)
    # per-member weighted Hx: (N, q_kept)
    weighted_hx = np.einsum("tnl,tl->nl", hx_kept, weights)

    np.savetxt(out / f"obs_{cfg.domain}_ch{cfg.channel}_totalline_withpert.txt",
               weighted_obs.reshape(-1, 1), fmt="%.4f")
    np.savetxt(out / "obs_error_variance.txt", error_variance.reshape(-1, 1), fmt="%.8f")
    np.savetxt(out / "original_obs_index.txt", (keep_index + 1).reshape(-1, 1), fmt="%d")
    np.savetxt(out / "clear_sky_mask_raw.txt", keep_mask_raw.reshape(-1, 1), fmt="%d")
    with open(out / "LACC_times.txt", "w", encoding="utf-8") as fh:
        fh.write(f"center_time={cfg.center_time}\n")
        for lag in cfg.lag_times:
            fh.write(f"lag_time={lag}\n")

    for k, mem in enumerate(member_numbers(cfg)):
        path = (
            members_dir / f"mem{mem:03d}" / cfg.sensor / f"BT_LACC_{cfg.center_time}"
            / f"obs_{cfg.domain}_ch{cfg.channel}_totalline.txt"
        )
        np.savetxt(path, weighted_hx[k].reshape(-1, 1), fmt="%.4f")
    return weighted_hx


def write_products(
    cfg: Config,
    keep_index: np.ndarray,          # 0-based, original order
    keep_mask_raw: np.ndarray,       # (q_raw,) 0/1
    clear_mask_kept: np.ndarray,     # (T, q_kept)
    obs_kept: np.ndarray,            # (T, q_kept) truth BT without noise
    noise_kept: np.ndarray,          # (T, q_kept)
    hx_kept: np.ndarray,             # (T, N, q_kept)
    weights: np.ndarray,             # (T, q_kept)
    lat: np.ndarray, lon: np.ndarray, surface: np.ndarray,  # (q_kept,)
    s: np.ndarray,                   # (N,)
    adaptive_scores: np.ndarray,     # J of the RETURNED weights (original scale)
    equal_scores: np.ndarray,
    fallback_flags: np.ndarray,
    reasons: np.ndarray,
    max_candidate_scores: np.ndarray,  # best J over the candidate pool (diagnostic)
    solvers: np.ndarray,               # per-position solver method
    error_variance: np.ndarray,      # (q_kept,) VARIANCE (sigma^2), not std
) -> None:
    """LEGACY-mode diagnostics (independent_sst_mean): keeps the original
    per-position score columns and npz schema."""
    _write_common_products(
        cfg, keep_index, keep_mask_raw, obs_kept, noise_kept, hx_kept,
        weights, error_variance,
    )
    out = cfg.output_dir
    n_times = len(cfg.lag_times)
    lag_hours = np.asarray([lag_hours_of(cfg, lag) for lag in cfg.lag_times], dtype=float)

    # diagnostics CSV
    nonclear_weight = np.sum(weights * (clear_mask_kept == 0), axis=0)
    with open(out / "adaptive_lacc_diagnostics.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        header = (
            ["original_obs_index", "lat", "lon"]
            + [f"w_lag{h:g}h" for h in lag_hours]
            + ["nonclear_time_weight_total", "equal_weight_score", "adaptive_score",
               "max_candidate_score", "score_gain", "fallback", "reason", "solver",
               "obs_error_variance"]
        )
        writer.writerow(header)
        for l in range(weights.shape[1]):
            writer.writerow(
                [int(keep_index[l] + 1), f"{lat[l]:.5f}", f"{lon[l]:.5f}"]
                + [f"{weights[t, l]:.6f}" for t in range(n_times)]
                + [
                    f"{nonclear_weight[l]:.6f}",
                    f"{equal_scores[l]:.8e}",
                    f"{adaptive_scores[l]:.8e}",
                    f"{max_candidate_scores[l]:.8e}",
                    f"{adaptive_scores[l] - equal_scores[l]:.8e}",
                    int(fallback_flags[l]),
                    str(reasons[l]),
                    str(solvers[l]),
                    f"{error_variance[l]:.8f}",
                ]
            )

    # weights archive: plain NumPy arrays only, loadable with allow_pickle=False
    np.savez(
        out / "adaptive_lacc_weights.npz",
        weights=weights,
        lag_times=np.asarray(cfg.lag_times, dtype=np.str_),
        lag_hours=lag_hours,
        center_time=np.asarray(cfg.center_time, dtype=np.str_),
        analysis_time=np.asarray(cfg.center_time, dtype=np.str_),
        original_obs_index=(keep_index + 1).astype(np.int32),  # 1-based
        lat=lat.astype(float),
        lon=lon.astype(float),
        channel=np.int32(cfg.channel),
        member_numbers=member_numbers(cfg),
        q_raw=np.int32(cfg.nobs_raw),
        q_kept=np.int32(weights.shape[1]),
        raw_keep_mask=keep_mask_raw.astype(np.int8),
        clear_mask_by_time=clear_mask_kept.astype(np.int8),
        surface_type=surface.astype(np.int8),
        target_id=np.asarray(cfg.sst_target_id, dtype=np.str_),
        target_lat_min=np.float64(cfg.sst_region[0]),
        target_lat_max=np.float64(cfg.sst_region[1]),
        target_lon_min=np.float64(cfg.sst_region[2]),
        target_lon_max=np.float64(cfg.sst_region[3]),
        sst_var_name=np.asarray(cfg.sst_var, dtype=np.str_),
        sst_background_dir=np.asarray(str(cfg.sst_bg_dir), dtype=np.str_),
        sst_target_members=s.astype(float),
        obs_err_std=np.float64(cfg.obs_err_std),
        equal_weight_scores=equal_scores,
        # adaptive_scores = J recomputed with the RETURNED weights (original
        # scale); max_candidate_scores = best J over the candidate pool.
        adaptive_scores=adaptive_scores,
        max_candidate_scores=max_candidate_scores,
        fallback_flags=fallback_flags.astype(np.int8),
        reasons=np.asarray(reasons, dtype=np.str_),
        solver_method=np.asarray(solvers, dtype=np.str_),
        noise_seed=np.int64(cfg.noise_seed),
        noise_generator=np.asarray(cfg.noise_generator, dtype=np.str_),
        # noise_by_time_kept is the FINAL applied perturbation in K:
        # epsilon = sigma * (z - spatial mean of z), i.e. already scaled --
        # not the unscaled standard-normal z.
        noise_by_time_kept=noise_kept,
        error_model_note=np.asarray(
            "nominal diagonal R: R_L = sigma^2 * sum_n w_n^2 with independent "
            "per-lag epsilon = sigma*(z - spatial mean over the q_raw grid); "
            "the spatial demeaning realizes sigma^2*(1-1/q_raw) per-point "
            "variance and a spatial error correlation that the diagonal DART "
            "error model does NOT represent",
            dtype=np.str_,
        ),
    )


def _json_safe(value):
    """Recursively convert numpy scalars / None so json.dumps(allow_nan=False)
    never sees non-standard NaN/Infinity tokens."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"non-finite value would corrupt JSON: {v}")
        return v
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value would corrupt JSON: {value}")
        return value
    if isinstance(value, (int,)):
        return int(value)
    raise TypeError(f"cannot JSON-serialize {type(value)!r}")


def write_products_joint(
    cfg: Config,
    keep_index: np.ndarray,
    keep_mask_raw: np.ndarray,
    clear_mask_kept: np.ndarray,
    obs_kept: np.ndarray,
    noise_kept: np.ndarray,
    hx_kept: np.ndarray,
    weights: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    surface: np.ndarray,
    sst_field: SstFieldTarget,
    problem: JointProblem,
    joint: dict,
    error_variance: np.ndarray,
) -> None:
    """JOINT-mode diagnostics (joint_sst_field): position-only CSV (no
    per-position score copies), schema_version=2 npz with the joint
    diagnostics, joint summary JSON and optimization history CSV."""
    out = cfg.output_dir
    _write_common_products(
        cfg, keep_index, keep_mask_raw, obs_kept, noise_kept, hx_kept,
        weights, error_variance,
    )
    lag_hours = np.asarray([lag_hours_of(cfg, lag) for lag in cfg.lag_times], dtype=float)
    n_times, q = weights.shape

    # position-only diagnostics CSV: weights / index / location / variance /
    # non-clear-time total weight.  A single joint score is NOT copied onto
    # every row -- per-position contributions are not defined by this
    # objective and must not be implied.
    nonclear_weight = np.sum(weights * (clear_mask_kept == 0), axis=0)
    with open(out / "adaptive_lacc_diagnostics.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["original_obs_index", "lat", "lon"]
            + [f"w_lag{h:g}h" for h in lag_hours]
            + ["nonclear_time_weight_total", "obs_error_variance"]
        )
        for l in range(q):
            writer.writerow(
                [int(keep_index[l] + 1), f"{lat[l]:.5f}", f"{lon[l]:.5f}"]
                + [f"{weights[t, l]:.6f}" for t in range(n_times)]
                + [f"{nonclear_weight[l]:.6f}", f"{error_variance[l]:.8f}"]
            )

    params = joint["params"]
    np.savez(
        out / "adaptive_lacc_weights.npz",
        # schema v2 -- joint mode
        schema_version=np.int32(2),
        optimization_mode=np.asarray(cfg.optimization_mode, dtype=np.str_),
        # ---- legacy interface fields (unchanged meaning) ----
        weights=weights,
        lag_times=np.asarray(cfg.lag_times, dtype=np.str_),
        lag_hours=lag_hours,
        center_time=np.asarray(cfg.center_time, dtype=np.str_),
        analysis_time=np.asarray(cfg.center_time, dtype=np.str_),
        original_obs_index=(keep_index + 1).astype(np.int32),  # 1-based
        lat=lat.astype(float),
        lon=lon.astype(float),
        channel=np.int32(cfg.channel),
        member_numbers=member_numbers(cfg),
        q_raw=np.int32(cfg.nobs_raw),
        q_kept=np.int32(q),
        raw_keep_mask=keep_mask_raw.astype(np.int8),
        clear_mask_by_time=clear_mask_kept.astype(np.int8),
        surface_type=surface.astype(np.int8),
        obs_err_std=np.float64(cfg.obs_err_std),
        noise_seed=np.int64(cfg.noise_seed),
        noise_generator=np.asarray(cfg.noise_generator, dtype=np.str_),
        noise_by_time_kept=noise_kept,
        error_model_note=np.asarray(
            "nominal diagonal R: R_L = sigma^2 * sum_n w_n^2 with independent "
            "per-lag epsilon = sigma*(z - spatial mean over the q_raw grid); "
            "the spatial demeaning realizes sigma^2*(1-1/q_raw) per-point "
            "variance and a spatial error correlation that the diagonal DART "
            "error model does NOT represent",
            dtype=np.str_,
        ),
        # ---- joint-mode target metadata ----
        target_id=np.asarray(cfg.sst_target_id, dtype=np.str_),
        target_config=np.asarray(json.dumps(sst_field.metadata), dtype=np.str_),
        target_grid_flat_index=sst_field.flat_index.astype(np.int64),
        target_lat=sst_field.lat.astype(float),
        target_lon=sst_field.lon.astype(float),
        target_area_weights=sst_field.area_weights.astype(float),
        area_weight_method=np.asarray("coslat_proxy", dtype=np.str_),
        # ---- joint problem + result ----
        G=problem.G,
        P0=np.float64(problem.P0),
        equal_jhat=np.float64(joint["equal_jhat"]),
        equal_j_phys=np.float64(joint["equal_j_phys"]),
        final_jhat=np.float64(joint["jhat"]),
        final_j_phys=np.float64(joint["j_phys"]),
        final_rpg=np.float64(joint["rpg"]),
        final_status=np.asarray(joint["status"], dtype=np.str_),
        selected_source=np.asarray(joint["selected_source"], dtype=np.str_),
        improvement_jhat=np.float64(joint["improvement_jhat"]),
        optimization_seed=np.int64(joint["seed"]),
        opt_params=np.asarray(json.dumps(params.as_dict()), dtype=np.str_),
        per_init=np.asarray([json.dumps(r) for r in joint["per_init"]], dtype=np.str_),
        degenerate_reason=np.asarray(
            joint["degenerate_reason"] if joint["degenerate_reason"] is not None else "",
            dtype=np.str_,
        ),
    )

    # joint summary JSON (strict: no NaN/Infinity tokens)
    summary = {
        "optimization_mode": cfg.optimization_mode,
        "schema_version": 2,
        "center_time": cfg.center_time,
        "lag_times": list(cfg.lag_times),
        "target": sst_field.metadata,
        "error_model": {
            "R": "sigma^2 I (nominal diagonal; spatially-demeaned noise "
                 "correlation NOT represented)",
            "sigma": cfg.obs_err_std,
        },
        "problem": {
            "N": int(problem.Y.shape[0]), "T": n_times, "q": int(q),
            "p": int(problem.X.shape[1]), "P0": problem.P0,
        },
        "optimizer": {**params.as_dict(), "optimization_seed": joint["seed"]},
        "equal_weight": {"jhat": joint["equal_jhat"],
                         "j_phys": joint["equal_j_phys"]},
        "selected": {"source": joint["selected_source"], "status": joint["status"],
                     "jhat": joint["jhat"], "j_phys": joint["j_phys"],
                     "rpg": joint["rpg"]},
        "improvement_vs_equal": {"jhat": joint["improvement_jhat"],
                                 "j_phys": problem.P0 * joint["improvement_jhat"]},
        "inits": joint["per_init"],
        "degenerate_reason": joint["degenerate_reason"],
        "note": "Joint objective gain is an ensemble-estimation benchmark "
                "under R=sigma^2 I without localization; it does NOT imply "
                "improved cold-trail/local SST or OSSE assimilation skill.",
    }
    with open(out / "adaptive_lacc_joint_summary.json", "w", encoding="utf-8") as fh:
        json.dump(_json_safe(summary), fh, indent=2, allow_nan=False)

    # optimization history CSV: one row per ACCEPTED iteration
    with open(out / "adaptive_lacc_optimization_history.csv", "w",
              encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["init_id", "init_type", "accepted_iter",
                         "jhat", "j_phys", "alpha", "rpg", "backtracks"])
        for row in joint["history"]:
            writer.writerow([
                row["init_id"], row["init_type"], row["accepted_iter"],
                f"{row['jhat']:.10e}", f"{row['j_phys']:.8f}",
                f"{row['alpha']:.6e}", f"{row['rpg']:.6e}", row["backtracks"],
            ])


def generate_shared_noise(cfg: Config) -> np.ndarray:
    """(T, q_raw) per-lag-time perturbation noise in K, ONE realization shared
    by all candidate weight sets (no resampling during optimization).

    Construction matches average_LACC_obs.m's `obserr_std * (rand - mean)`:
      1. draw standard normal z on the full (npoint, npoint) grid;
      2. remove each time's spatial mean over the RAW grid (same rule as the
         MATLAB code; NOT re-centered after the keep-index filtering);
      3. scale by the observation error standard deviation:
         epsilon = sigma * (z - mean(z));
      4. flatten column-major (MATLAB totalline order); the caller filters
         with the unified keep index I.

    Because of the spatial demeaning the realized per-point variance is
    sigma^2 * (1 - 1/q_raw) and neighbouring points are correlated; the DART
    representation stays the NOMINAL diagonal model R = sigma^2 I (combined
    R_L = sigma^2 * sum_n w_n^2).  This is a numpy PCG64 stream: an identical
    integer seed does NOT reproduce the legacy MATLAB randn realization.
    """
    rng = np.random.default_rng(cfg.noise_seed)
    rows = []
    for _ in cfg.lag_times:
        field = rng.standard_normal((cfg.npoint, cfg.npoint))
        field = cfg.obs_err_std * (field - field.mean())
        rows.append(field.reshape(-1, order="F"))
    return np.asarray(rows, dtype=float)


def main() -> None:
    cfg = load_config()

    # 1. historical obs, all-member Hx, per-time clear masks
    print(f"Adaptive LACC: center={cfg.center_time} lags={list(cfg.lag_times)}")
    print(f"Output directory: {cfg.output_dir}")
    if cfg.npoint * cfg.npoint != cfg.nobs_raw:
        raise ValueError(
            f"npoint^2 ({cfg.npoint ** 2}) does not equal q_raw ({cfg.nobs_raw}); "
            f"the noise grid must match the raw observation grid"
        )

    obs_raw = load_truth_obs(cfg)                  # (T, q_raw)
    hx_raw = load_ens_hx(cfg)                      # (T, N, q_raw)
    masks_raw = load_clear_masks(cfg)              # (T, q_raw)
    lat_raw, lon_raw, surface_raw = read_profile_latlon(cfg.para_file, cfg.nobs_raw)

    # 2. unified LACC position mask, same rule as average_LACC_obs.m
    keep_mask = build_keep_mask(cfg, masks_raw)
    crosscheck_keep_mask(cfg, keep_mask)
    keep_index = np.flatnonzero(keep_mask)         # original order, 0-based
    q_kept = int(keep_index.size)
    print(f"Unified LACC mask: kept {q_kept} / {cfg.nobs_raw} positions "
          f"(rule: clear in >= {max(1, min(int(math.floor(cfg.clear_sky_min_fraction * len(cfg.lag_times))), len(cfg.lag_times)))} "
          f"of {len(cfg.lag_times)} lag times)")
    if q_kept == 0:
        raise RuntimeError(
            "all raw positions were removed by the unified LACC mask; no usable "
            "observation -- adaptive conversion stopped without writing outputs"
        )

    # 3. filter everything with the same index I
    obs_kept = obs_raw[:, keep_index]              # (T, q_kept)
    hx_kept = hx_raw[:, :, keep_index]             # (T, N, q_kept)
    clear_kept = masks_raw[:, keep_index]          # (T, q_kept)
    lat = lat_raw[keep_index]
    lon = lon_raw[keep_index]
    surface = surface_raw[keep_index]

    # 4. time weights at kept positions (mode-dispatched); the optimization
    #    never sees NR errors, real innovations or the applied noise
    sigma2 = cfg.obs_err_std ** 2
    noise_raw = generate_shared_noise(cfg)         # (T, q_raw), raw-grid centered
    noise_kept = noise_raw[:, keep_index]          # filtered by I, NOT re-centered
    error_variance = None                          # set below with the final weights

    if cfg.optimization_mode == "joint_sst_field":
        sst_field = load_sst_field_target(cfg)     # (N, p) field + fixed support
        print(f"SST field target '{cfg.sst_target_id}' ({cfg.sst_var} level "
              f"{cfg.sst_level}) over box {cfg.sst_region}: p={sst_field.x.shape[1]} "
              f"grid points, area_weight_method={sst_field.metadata['area_weight_method']} "
              f"(background: {cfg.sst_bg_dir})")
        problem = prepare_joint_problem(
            sst_field.x, hx_kept, sigma2, sst_field.area_weights
        )
        print(f"Joint problem: N={problem.Y.shape[0]} T={problem.Y.shape[1]} "
              f"q={problem.Y.shape[2]} P0={problem.P0:.6f} K^2 "
              f"(area-weighted SST variance; JOINT objective over all "
              f"{problem.Y.shape[2]} positions, no localization)")
        t0 = time.time()
        joint = optimize_joint_weights(problem, cfg)
        elapsed = time.time() - t0
        weights = joint["weights"]
        n_iters = sum(r.get("accepted_iters", 0) or 0 for r in joint["per_init"])
        finals = [r["final_jhat"] for r in joint["per_init"]
                  if r.get("final_jhat") is not None]
        finals_text = (
            f"[{min(finals):.8f}, {max(finals):.8f}]" if finals else "n/a"
        )
        print(f"Joint optimization: status={joint['status']} "
              f"source={joint['selected_source']}; Jhat "
              f"{joint['equal_jhat']:.8f} (equal) -> {joint['jhat']:.8f} "
              f"(gain {joint['improvement_jhat']:.3e}); J "
              f"{joint['equal_j_phys']:.6f} -> {joint['j_phys']:.6f} K^2; "
              f"rPG={joint['rpg']:.3e}; accepted iters={n_iters}; "
              f"final-Jhat range over inits={finals_text}")
        print(f"Joint optimization wall time: {elapsed:.2f} s "
              f"(fixed background; multi-start does NOT prove global optimum)")
        error_variance = np.sum(weights ** 2, axis=0) * sigma2

        write_products_joint(
            cfg, keep_index, keep_mask, clear_kept, obs_kept, noise_kept,
            hx_kept, weights, lat, lon, surface, sst_field, problem, joint,
            error_variance,
        )
    else:
        s = load_sst_target(cfg)                   # (N,) legacy scalar target
        print(f"SST target '{cfg.sst_target_id}' ({cfg.sst_var} level {cfg.sst_level}) "
              f"area mean over box {cfg.sst_region}: "
              f"mean={s.mean():.4f} K spread(std)={s.std(ddof=1):.6f} K (background: {cfg.sst_bg_dir})")
        (weights, adaptive_scores, equal_scores, fallback_flags, reasons,
         max_candidate_scores, solvers) = optimize_weights_all(hx_kept, s, sigma2, cfg)

        n_fallback = int(fallback_flags.sum())
        gain = adaptive_scores - equal_scores
        solver_counts: dict[str, int] = {}
        for name in solvers:
            solver_counts[name] = solver_counts.get(name, 0) + 1
        print(f"Weight optimization: {q_kept} positions, fallback to equal weight at "
              f"{n_fallback}; mean score gain {gain.mean():.6e}; "
              f"positions with gain>0: {int((gain > 0).sum())}")
        print(f"Solver methods: {solver_counts} "
              f"(per-position independent optima, not a joint optimum over all obs)")
        error_variance = np.sum(weights ** 2, axis=0) * sigma2

        write_products(
            cfg, keep_index, keep_mask, clear_kept, obs_kept, noise_kept, hx_kept,
            weights, lat, lon, surface, s, adaptive_scores, equal_scores,
            fallback_flags, reasons, max_candidate_scores, solvers, error_variance,
        )

    print(f"Adaptive LACC products written under {cfg.output_dir} "
          f"(mode={cfg.optimization_mode}: weights npz, diagnostics csv, "
          f"obs/variance/index/mask/LACC_times txt, per-member weighted Hx under "
          f"members/memNNN/{cfg.sensor}/BT_LACC_{cfg.center_time}"
          + (", joint summary json + optimization history csv"
             if cfg.optimization_mode == "joint_sst_field" else "") + ")")


if __name__ == "__main__":
    main()
