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
  4. optimize time weights independently at each kept position: for
     T <= 8 lag times an EXACT active-set enumeration over all non-empty
     support sets (stationary direction S_A v_A = c_A per face, normalized
     onto the simplex; vertices and the exact equal-weight candidate added;
     scale-free scoring, relative-only tie tolerance, final score recomputed
     with the original c and S).  For T > 8 a bounded grid+refinement search
     with the same scale-free scoring is used instead and recorded as the
     solver method.  Each position is solved independently -- NOT a joint
     optimum over all observations; the result is a numerical/finite-candidate
     solution, exact up to floating-point precision for T <= 8;
  5. write weighted obs (with noise), weighted per-member Hx, per-obs
     error VARIANCE, weights (.npz) and diagnostics.

Target (version 1): area-mean SST over a user-specified ocean box,
s_i = sum_j a_j SST_{i,j}(t_analysis), from the same firstguess_d01.memNNN
background that enters DART.  Weights maximize
(c_l w)^2 / (w^T S_l w) with
  c_l = s'^T Y'_l / (N-1),   S_l = Y'_l^T Y'_l / (N-1) + R_l
subject to w >= 0, sum(w) = 1.  This is a numerical grid-search
approximation, not a claimed analytic optimum; weights are chosen per
observation location, not jointly; the covariance formula assumes
background and observation errors independent (no complete-LACC historical
reuse correction); NR truth is never used to pick weights.

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
import math
import os
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

    # --- optimizer ---
    coarse_step_for_t: tuple[tuple[int, float], ...] = (
        (1, 0.0), (2, 0.01), (3, 0.02), (4, 0.05), (5, 0.1),
    )
    coarse_step_large_t: float = 0.2  # used for T >= 6
    refine_passes: int = 2            # each pass: radius 4*step, finer step

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
    "noise_seed": "ADAPTIVE_NOISE_SEED",
    "mask_crosscheck": "ADAPTIVE_LACC_MASK_CROSSCHECK",
}

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
            sst_bg_dir=Path(base) / "4assimilation" / "0mem_all_time" / cfg.center_time,
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
def load_sst_target(cfg: Config) -> np.ndarray:
    lat_min, lat_max, lon_min, lon_max = cfg.sst_region
    members = member_numbers(cfg)
    s = np.empty(cfg.ens_size, dtype=float)
    ref_support: np.ndarray | None = None
    ref_member = -1
    ref_grid: tuple[np.ndarray, np.ndarray] | None = None
    for k, mem in enumerate(members):
        path = cfg.sst_bg_dir / f"firstguess_{cfg.domain}.mem{mem:03d}"
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
        ocean = box & (land < 0.5)
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

        # every member must be averaged over the SAME grid points and weights:
        # member-dependent missing values would silently change the target.
        if ref_support is None:
            ref_support = support
            ref_member = int(mem)
            ref_grid = (grid_lat, grid_lon)
        else:
            if grid_lat.shape != ref_grid[0].shape or not (
                np.array_equal(grid_lat, ref_grid[0]) and np.array_equal(grid_lon, ref_grid[1])
            ):
                raise ValueError(
                    f"member {mem:03d} background grid differs from member "
                    f"{ref_member:03d}; the SST area mean would not be comparable"
                )
            if not np.array_equal(support, ref_support):
                diff = int(np.sum(support != ref_support))
                raise ValueError(
                    f"SST area-mean support differs across members: member "
                    f"{mem:03d} differs from member {ref_member:03d} at {diff} "
                    f"grid points (member-dependent missing values). Refusing to "
                    f"average over inconsistent supports; clean the backgrounds "
                    f"or restrict the target region first."
                )
        s[k] = float(np.sum(weights[support] * sst_field[support]) / np.sum(weights[support]))
    if not np.all(np.isfinite(s)):
        raise ValueError("SST target contains non-finite member values")
    return s


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
# outputs
# =====================
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
    out = cfg.output_dir
    members_dir = out / "members"
    for mem in member_numbers(cfg):
        (members_dir / f"mem{mem:03d}" / cfg.sensor / f"BT_LACC_{cfg.center_time}").mkdir(
            parents=True, exist_ok=True
        )

    n_times = len(cfg.lag_times)
    lag_hours = np.asarray([lag_hours_of(cfg, lag) for lag in cfg.lag_times], dtype=float)

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

    # 4. time weights only at kept positions
    s = load_sst_target(cfg)                       # (N,)
    print(f"SST target '{cfg.sst_target_id}' ({cfg.sst_var} level {cfg.sst_level}) "
          f"area mean over box {cfg.sst_region}: "
          f"mean={s.mean():.4f} K spread(std)={s.std(ddof=1):.6f} K (background: {cfg.sst_bg_dir})")
    sigma2 = cfg.obs_err_std ** 2
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

    # 5. shared per-time noise, weighted products, all on disk
    noise_raw = generate_shared_noise(cfg)         # (T, q_raw), raw-grid centered
    noise_kept = noise_raw[:, keep_index]          # filtered by I, NOT re-centered
    error_variance = np.sum(weights ** 2, axis=0) * sigma2  # R_L = sum w^2 sigma^2

    write_products(
        cfg, keep_index, keep_mask, clear_kept, obs_kept, noise_kept, hx_kept,
        weights, lat, lon, surface, s, adaptive_scores, equal_scores,
        fallback_flags, reasons, max_candidate_scores, solvers, error_variance,
    )
    print(f"Adaptive LACC products written under {cfg.output_dir} "
          f"(weights npz, diagnostics csv, obs/variance/index/mask/LACC_times txt, "
          f"per-member weighted Hx under members/memNNN/{cfg.sensor}/BT_LACC_{cfg.center_time})")


if __name__ == "__main__":
    main()
