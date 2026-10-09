#!/usr/bin/env python3
"""
diag_LACC_lag_cov_innovation.py -- single-observation LACC lead/lag ocean
covariance vs. innovation-direction diagnostic.

WHAT THIS SCRIPT IS FOR
-----------------------
The LACC single-observation experiments assimilate a time-averaged AMSU-A
channel-4 brightness temperature.  After the analysis the ensemble is closer
to the observation in BT space, yet OM_TMP moves *away* from the NR.  This
script tests one concrete explanation of that paradox: the sign of the
ensemble covariance between the CENTER-TIME ocean state and the *lagged* Hx
that the LACC average is built from may be opposite to the sign of the lagged
innovation, so the update is delivered in the wrong direction in ocean space.

For every lag time j (lag tau_j hours behind the center time t) it computes

    c_j    = Cov_m[ T_o,m(t, s) , Hx_m(t - tau_j, k) ]          (ddof = 1)
    d_j_NR = y_j_NR(k) - mean_m[ Hx_m(t - tau_j, k) ]

with these sign conventions:

  * m runs over the ensemble members; the covariance is taken ACROSS MEMBERS
    AT ONE LOCATION.  Members and spatial positions are never pooled into the
    same covariance (that would mix two different ensembles).
  * s is the ocean state location and k is the observation location.  They are
    the same physical point by default but are reported separately.
  * T_o is ALWAYS the center-time ocean state, never the ocean state at the
    lag time.  Correlating T_o(t - tau_j) with Hx(t - tau_j) would answer a
    different (and far less interesting) question and would destroy the lag
    structure this diagnostic exists to measure.
  * y_j_NR is the NOISE-FREE NR brightness temperature at the lag time.  The
    per-lag directories carry no ``*_withpert`` file, so this diagnostic can
    only produce noise-free lagged innovations; it can never recover the
    per-time noise from the final noisy LACC average.

DIRECTION VS. MAGNITUDE -- READ THIS BEFORE QUOTING ANY RESULT
--------------------------------------------------------------
The ensemble update in ocean space at the state point is, for one observation,

    Delta T_o  ~  c_j / (sigma_hx^2 + R) * d_j

so sign(Delta T_o) = sign(c_j * d_j_NR) because the denominator is positive.
Comparing that sign against the correction the prior actually needs,

    e_o = T_o,NR - mean_m[ T_o,m(t, s) ]           (prior ocean error vs NR)

gives the alignment flag of this script,

    alignment = (c_j * d_j_NR) * e_o

    alignment > 0  ->  the ensemble-implied update points TOWARD the NR
    alignment < 0  ->  the update points AWAY from the NR
    alignment near 0 ->  reported separately as UNDEFINED_E_O / WEAK_COUPLING

The classification is on ``alignment``, NOT on ``c_j * d_j_NR`` alone.  The
product only fixes the sign of the increment; whether that increment moves the
state toward the NR depends on which way the prior is wrong, i.e. on the sign
of ``e_o``.  When ``e_o < 0`` (the NR is colder than the prior mean) the same
product means the OPPOSITE direction, so a product-only test inverts the answer
for half of all cases.  ``e_o == 0`` is handled explicitly, so a
``near_zero_frac`` of 0 cannot make a directionless case look directional.

A correct DIRECTION is not the same thing as a smaller FINAL ERROR.  The sign
of c_j*d_j_NR says nothing about magnitude, about overshoot, or about whether
the analysis-error variance actually shrinks -- those depend on |c_j|, on the
Hx spread and on R, none of which this diagnostic constrains.  Never quote an
alignment flag as if it were a skill score.

THE ASSIMILATED WINDOW IS NOT THE DIAGNOSTIC WINDOW
---------------------------------------------------
Two different lag sets appear in this script and they must not be conflated:

  * the ASSIMILATED window -- the lag times recorded in ``LACC_times.txt``,
    i.e. what the experiment actually averaged over;
  * the DIAGNOSED lags -- what ``lag_hours`` selects out of that window, plus
    any extra lags the user explicitly asked for with ``--enable-extra-lags``
    (9 h, 12 h), which lie OUTSIDE the real window.

Every real-window quantity (``Hx_L``, ``y_L_NR``, ``d_L_NR``, ``d_L_actual``,
``noise_L``) is built from the assimilated lags ONLY, and the external_FO
cross-check compares against that same window rather than a hard-coded
``[0, 3, 6]``.  Folding an extra diagnostic lag in would silently redefine the
quantity the assimilation actually saw.  The window may also be diagnosed only
IN PART -- ``lag_hours = (3, 6)`` is a legitimate request -- and then those
quantities are reported ``available: false`` with the missing lag hours named,
never approximated from the diagnostic subset.  In that case the 0 h
CENTER-TIME profile is still read, because it defines the observation position
and the default ocean state point; reading it does NOT put the 0 h lag into the
diagnostic set, into any window average, or into the external_FO comparison.
Correspondingly, ``f_order_numbering_center`` is reported ``available: false``
when the 0 h NR column was never checked, rather than borrowing another lag's
result.

ENSEMBLE SUBSETS AND THE OLD ``external_FO``
--------------------------------------------
``external_FO`` in the single-obs obs_seq stores the FULL DART ensemble, so a
run over, say, members 2..4 must pick entries 1..3 of that array and compare
them member-by-member against the Hx it computed.  The stored count is read and
checked; a count that disagrees with the number of values, a member outside the
stored range, or a block with no reliable count is refused or reported
``unavailable`` -- the correspondence is never inferred from the number of
members that happen to be configured.  The comparison is only ever made over
the real window: a subset run reports it ``unavailable`` instead.

WHY THE WINDOW PRODUCT IS NOT A WEIGHTED SUM OF PRODUCTS
--------------------------------------------------------
For an equal-weight window W the script evaluates

    Hx_w(m) = sum_j w_j Hx_mj          (per member, over the window)
    c_w     = Cov_m[ T_o , Hx_w ]
    d_w_NR  = sum_j w_j d_j_NR
    direction from   c_w * d_w_NR

It must NOT be replaced by ``sum_j w_j * c_j * d_j_NR``: averaging is a
linear operation on Hx but the covariance is bilinear, so the cross terms
between different lags matter and the two expressions are different numbers.
The naive sum is still emitted, clearly labelled ``..._NAIVE_do_not_use``, so
the size of the discrepancy is visible rather than hidden.

INPUT DATA (all shapes/layouts verified against the running experiment)
----------------------------------------------------------------------
    {hx_root}/4ens_BT_LACC/mem{m:03d}/{sensor}/BT_{tag}/obs_{dom}_ch{ch}_totalline.txt
    {hx_root}/3obs_BT_LACC/{sensor}/BT_{tag}/obs_{dom}_ch{ch}_totalline.txt
        676 lines, one column, K.  ``obs_{dom}_ch{ch}.txt`` in the same
        directory is the 26x26 array.  BOTH files are checked, for EVERY member
        and lag that enters a computation, and neither check stands in for the
        other:

          * the shape must be exactly ``(676, 1)``.  ``np.loadtxt`` flattens a
            26x26 matrix into the same 676 numbers a genuine one-column file
            holds, so a count-only test accepts a matrix pasted at the column
            path; the shape is therefore compared with ``ndmin=2``.
          * ``matrix.reshape(-1, order="F") == one_column_file``.  NumPy's
            default C-order flattening is WRONG here and is checked explicitly.

        The member Hx matrices live in the member directories, the NR matrix in
        ``3obs_BT_LACC``.  Swapping two lag columns while leaving the matrices
        in place leaves the window average -- and therefore ``external_FO`` --
        untouched, so only this per-member F-order check can catch it.

    {hx_root}/3obs_BT_LACC/{sensor}/BT_LACC_{center}/LACC_times.txt
        ``center_time=`` / ``lag_time=`` records.  The tag is DD_HH_MM
        (day-of-month, hour, minute), NOT HH_MM_SS: for the 2018-09-10_00:00
        center the three assimilated times are 10_00_00, 09_21_00, 09_18_00,
        i.e. lag hours 0, 3, 6.

    {hx_root}/profile/profile_{dom}_LACC_{center}/prof{DD}_{HH}:{MM}.dat
        RTTOV profiles.  Coordinates are read by locating the comment marker
        ``! Elevation (km), latitude and longitude (degrees)`` and taking the
        next line; the fixed 185-line stride used by the DART converter is NOT
        reused here.

    {hx_root}/3obs_BT_LACC/{sensor}/BT_{tag}/clear_sky_mask.txt
        Per-lag 0/1 masks.  They only label a lag as clear / not clear.  This
        script NEVER drops a lag and NEVER renormalises the existing LACC
        weights because of them.  A missing mask file (the 12 h NR mask is
        currently absent) is reported as ``unknown``, never as clear.

LIMITATIONS THIS SCRIPT DOES NOT PAPER OVER
-------------------------------------------
  * Both evidence batches are single-typhoon, 50-member, one center time.
  * The alignment flag is a DIRECTION statement only (see above).
  * ``r_j`` below is the ensemble CORRELATION at the state/observation pair.
    It is emitted because it is scale-free, but it is a diagnostic of the
    ensemble, not a significance test: the 50 members are one sample, and no
    p-value is attached.
  * No conclusion about the optimal observation-error inflation can be drawn
    from this diagnostic.

USAGE
-----
Check the code first (no experiment data needed, runs anywhere):

    python diag_LACC_lag_cov_innovation.py --self-test

Then run against the experiment.  On the cluster, use the project interpreter
and submit through LSF rather than running on a login node::

    PY=/share/home/lililei1/kcfu/anaconda/envs/wrf/bin/python
    bsub -q serial -oo diag_lacc.out -eo diag_lacc.err \\
         "$PY -B diag_LACC_lag_cov_innovation.py"

Note that ``--self-test`` builds and removes its own temporary dataset; it
never touches the experiment tree, and the main run is read-only as well.

Useful flags:

    --self-test                     run the synthetic code checks and exit
    --project-root DIR              point at a local copy of the data tree
    --hx-root DIR                   override {project_root}/3create_obs/hx_rttov
    --output-dir DIR                where the CSV / JSON / PNG land
    --background {obs_seq111,mem_all_time}
                                    which center-time background to use
    --sample-mode {linear,nearest}  how the ocean state is taken at the point
    --enable-extra-lags --extra-lag-hours 9 12
                                    diagnose lags OUTSIDE the assimilated
                                    window; requires the explicit switch

DEPENDENCIES
------------
    numpy, netCDF4, scipy, matplotlib  (plus the Python standard library)

Nothing else is imported; pandas is deliberately not used.  matplotlib is
forced to the Agg backend and ``MPLCONFIGDIR`` defaults to ``$TMPDIR`` so the
script runs unattended on a compute node.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It never writes into the data tree, never modifies an input, and never
launches an assimilation.  It reads, it computes, it writes figures and tables
into ``--output-dir``.

Author: prepared for kcfu's Mangkhut LACC experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

# matplotlib must see a writable cache dir before it is imported: the compute
# nodes have a small HOME quota.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"),
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import netCDF4 as nc  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import LinearNDInterpolator  # noqa: E402


# =============================================================================
# Verified server defaults
# =============================================================================
SERVER_PROJECT_ROOT = Path("/share/home/lililei1/kcfu/tc_mangkhut")
SERVER_OBS_SEQ111_DIR = Path(
    "/scratch/lililei1/kcfu/tc_mangkhut/4assimilation/DART/EAKF/obs_seq111"
)

PROFILE_COORDINATE_MARKER = "! Elevation (km), latitude and longitude (degrees)"

# Reference numbers verified once by hand against the running experiment.  They
# exist ONLY to answer "did I parse the same files the old experiment used?".
# They are never substituted for a computed result, and every one of them can
# be switched off with Config.check_reference_values.
REFERENCE_EXTERNAL_FO_TOL_K = 1.0e-4
REFERENCE_NOISE_FREE_INNOVATION_K = {
    "66": {0: 0.8254, 3: -0.2882, 6: 0.4390},
    "98": {0: -0.1344, 3: 0.0224, 6: 0.4144},
}
REFERENCE_LACC_OBS_VALUE_K = {"66": 255.7672, "98": 254.7134}
REFERENCE_TOLERANCE_K = 5.0e-4


# =============================================================================
# Errors
# =============================================================================
class DiagnosticError(RuntimeError):
    """Base class for every refusal in this script."""


class MissingInputError(DiagnosticError):
    """An input file that the diagnostic needs does not exist."""


class SamplingError(DiagnosticError):
    """A state point could not be sampled without extrapolating."""


class NonOceanPointError(SamplingError):
    """The requested state point sits on a non-water grid cell."""


class OutsideDomainError(SamplingError):
    """The requested state point is outside the interpolation support."""


class ConsistencyError(DiagnosticError):
    """A cross-check between two inputs disagreed."""


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class ObsTarget:
    """One single-observation case.

    ``filtered_index`` and ``raw_index`` are alternatives, both 1-based:

      * ``filtered_index`` counts inside the clear-sky-KEPT list, i.e. the
        observation numbering that the LACC obs_seq and the single-obs DART
        runs use ("OBS 66", "OBS 98");
      * ``raw_index`` counts rows of the unfiltered 676-row file.

    The two are different numbers and confusing them silently reads the wrong
    pixel, so the script resolves both and reports the mapping it used.
    """

    label: str
    filtered_index: int | None = None
    raw_index: int | None = None
    expected_lat: float | None = None
    expected_lon: float | None = None


@dataclass(frozen=True)
class Config:
    # ---------------------------------------------------------------- roots
    project_root: Path = SERVER_PROJECT_ROOT
    hx_root: Path | None = None  # default: {project_root}/3create_obs/hx_rttov
    output_dir: Path = Path(__file__).resolve().parent / "figs" / "LACC_lag_cov_innovation"

    # ------------------------------------------------------ experiment shape
    center_time: str = "2018-09-10_00:00:00"
    domain: str = "d01"
    sensor: str = "AMSUA"
    channel: int = 4
    member_start: int = 1
    member_end: int = 50
    nobs_raw: int = 676
    matrix_side: int = 26  # obs_{dom}_ch{ch}.txt is matrix_side**2 == nobs_raw

    # --------------------------------------------------------- LACC lag set
    # ONLY the lag times that were actually assimilated.  Anything else must be
    # asked for explicitly through extra_lag_hours + enable_extra_lag_diagnostics.
    lag_hours: tuple[int, ...] = (0, 3, 6)
    extra_lag_hours: tuple[int, ...] = ()
    enable_extra_lag_diagnostics: bool = False
    lacc_times_file: Path | None = None  # default: the BT_LACC_{center} copy

    # ----------------------------------------------------------- observations
    obs_targets: tuple[ObsTarget, ...] = (
        ObsTarget("66", filtered_index=66, expected_lat=14.2751, expected_lon=147.6832),
        ObsTarget("98", filtered_index=98, expected_lat=14.5495, expected_lon=147.2686),
    )
    clear_sky_mask_file: Path | None = None  # default: the unified center-time mask
    single_obs_dir: Path | None = None  # default: {project_root}/4assimilation/1convert_obs/run_dir

    # ------------------------------------------------------------- background
    # "obs_seq111" or "mem_all_time".  Chosen explicitly by the user: if the
    # selected directory has no member files the script REFUSES rather than
    # silently falling back to the other one, because the two backgrounds are
    # not interchangeable and a silent switch would change every number here.
    background_choice: str = "obs_seq111"
    obs_seq111_dir: Path = SERVER_OBS_SEQ111_DIR
    obs_seq111_pattern: str = "preassim_member_{mem:04d}_{domain}.nc"
    mem_all_time_dir: Path | None = None  # default {project_root}/4assimilation/0mem_all_time/10_00_00
    mem_all_time_pattern: str = "firstguess_{domain}.mem{mem:03d}"

    # ------------------------------------------------------------------- NR
    nr_file: Path | None = None  # default {project_root}/NR_wrfout/2domain/wrfout_d02_{center_time}

    # --------------------------------------------------------- state variable
    state_var: str = "OM_TMP"
    state_level: int = 0  # level 0 only; the 30-layer array is never read whole
    state_lat: float | None = None  # None -> use the observation latitude
    state_lon: float | None = None  # None -> use the observation longitude
    lat_var: str = "XLAT"
    lon_var: str = "XLONG"
    # Water test.  LANDMASK (1 = land, 0 = water) is preferred when present;
    # XLAND (1 = land, 2 = water) is used otherwise.  Forcing either is
    # possible through water_mask_source.
    water_mask_source: str = "auto"  # "auto" | "LANDMASK" | "XLAND"
    sample_mode: str = "linear"  # "linear" (default) | "nearest"
    sample_require_water: bool = True  # explicit opt-out only; never silent
    local_window_deg: float = 1.0  # local subset used to build the triangulation
    max_local_window_deg: float = 4.0
    nearest_max_km: float = 25.0  # nearest mode refuses beyond this distance

    # ------------------------------------------------- single-obs cross-check
    # The obs_seq loc3d block stores radians with limited text precision, so this
    # tolerance absorbs the rounding.  It is NOT a licence to accept a different
    # observation location: a mismatch means the file describes another pixel.
    single_obs_coord_tol_km: float = 1.0

    # ------------------------------------------------- time / background gate
    # Every background member and the NR file must carry Times == center_time.
    # Set False only as a deliberate override for files that genuinely have no
    # Times variable; leaving it True is what makes the check a gate.
    require_center_time_match: bool = True

    # ---------------------------------------------------------- presentation
    near_zero_frac: float = 0.1  # near-zero test, in units of the ocean spread
    check_reference_values: bool = True
    require_f_order_check: bool = True
    write_figures: bool = True
    figure_dpi: int = 300

    # ------------------------------------------------------------- derived
    def resolved_hx_root(self) -> Path:
        return self.hx_root if self.hx_root is not None else self.project_root / "3create_obs" / "hx_rttov"

    def resolved_obs_bt_dir(self) -> Path:
        return self.resolved_hx_root() / "3obs_BT_LACC" / self.sensor

    def resolved_ens_bt_dir(self) -> Path:
        return self.resolved_hx_root() / "4ens_BT_LACC"

    def resolved_profile_dir(self) -> Path:
        return self.resolved_hx_root() / "profile"

    def resolved_lacc_times(self) -> Path:
        if self.lacc_times_file is not None:
            return self.lacc_times_file
        return self.resolved_obs_bt_dir() / f"BT_LACC_{self.center_tag()}" / "LACC_times.txt"

    def resolved_unified_mask(self) -> Path:
        if self.clear_sky_mask_file is not None:
            return self.clear_sky_mask_file
        return self.resolved_obs_bt_dir() / f"BT_LACC_{self.center_tag()}" / "clear_sky_mask.txt"

    def resolved_single_obs_dir(self) -> Path:
        if self.single_obs_dir is not None:
            return self.single_obs_dir
        return self.project_root / "4assimilation" / "1convert_obs" / "run_dir"

    def resolved_nr_file(self) -> Path:
        if self.nr_file is not None:
            return self.nr_file
        return self.project_root / "NR_wrfout" / "2domain" / f"wrfout_d02_{self.center_time}"

    def center_datetime(self) -> datetime:
        return datetime.strptime(self.center_time, "%Y-%m-%d_%H:%M:%S")

    def center_tag(self) -> str:
        centre = self.center_datetime()
        return f"{centre.day:02d}_{centre.hour:02d}_{centre.minute:02d}"

    def member_numbers(self) -> list[int]:
        return list(range(self.member_start, self.member_end + 1))


DEFAULT_CONFIG = Config()


# =============================================================================
# Time helpers.  The on-disk tag is DD_HH_MM (day-of-month, hour, minute).
# =============================================================================
def tag_to_datetime(tag: str, center_dt: datetime, search_days: int = 3) -> datetime:
    """Resolve a DD_HH_MM tag to a real datetime near ``center_dt``.

    The tag carries no year or month, so the month/day rollover is resolved by
    scanning a small window of days around the center time and matching the
    day-of-month.  A +-3 day window covers every LACC lag used here.
    """
    parts = tag.split("_")
    if len(parts) != 3:
        raise ValueError(f"lag tag {tag!r} is not DD_HH_MM")
    try:
        day, hour, minute = (int(value) for value in parts)
    except ValueError as exc:
        raise ValueError(f"lag tag {tag!r} is not DD_HH_MM") from exc
    if not (1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"lag tag {tag!r} has out-of-range fields")

    for offset in range(-search_days, search_days + 1):
        candidate = (center_dt + timedelta(days=offset)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if candidate.day == day:
            return candidate
    raise ValueError(
        f"lag tag {tag!r} is more than {search_days} days from the center time "
        f"{center_dt:%Y-%m-%d %H:%M:%S}"
    )


def lag_hours_between(center_dt: datetime, lag_dt: datetime) -> int:
    hours = (center_dt - lag_dt).total_seconds() / 3600.0
    rounded = int(round(hours))
    if abs(hours - rounded) > 1.0e-6:
        raise ValueError(
            f"lag {lag_dt:%Y-%m-%d %H:%M:%S} is {hours} h behind the center "
            f"{center_dt:%Y-%m-%d %H:%M:%S}, which is not a whole number of hours"
        )
    if rounded < 0:
        raise ValueError("lag time is after the center time")
    return rounded


def datetime_to_tag(moment: datetime) -> str:
    return f"{moment.day:02d}_{moment.hour:02d}_{moment.minute:02d}"


def profile_file_name(tag: str) -> str:
    """``10_00_00`` -> ``prof10_00:00.dat`` (the layout the RTTOV driver writes)."""
    day, hour, minute = tag.split("_")
    return f"prof{day}_{hour}:{minute}.dat"


def bt_directory_name(tag: str) -> str:
    return f"BT_{tag}"


# =============================================================================
# Strict text readers
# =============================================================================
def read_column_file(path: Path, expected: int, label: str) -> np.ndarray:
    """Read a ONE-COLUMN numeric file of exactly ``expected`` finite values.

    The SHAPE is checked, not merely the number of values.  ``np.loadtxt``
    flattens a 26x26 matrix into the same 676 numbers that a genuine one-column
    file carries, so a count-only check silently accepts a matrix pasted at the
    one-column path -- and every downstream index then refers to a different
    pixel than the reader believes.  ``ndmin=2`` keeps the row/column structure
    so the file has to be ``(expected, 1)``: a single row of many columns, a
    multi-column file and a square matrix are all refused.  A file holding one
    value is the ``(1, 1)`` case of the same rule.
    """
    if not path.is_file():
        raise MissingInputError(f"{label} file does not exist: {path}")
    try:
        values = np.loadtxt(path, ndmin=2, dtype=float)
    except ValueError as exc:
        raise ConsistencyError(
            f"{label} could not be read as a numeric array: {path} ({exc})"
        ) from exc
    if values.shape != (expected, 1):
        if values.ndim == 2 and values.shape[0] * values.shape[1] == expected:
            raise ConsistencyError(
                f"{label} in {path} has shape {values.shape}, expected "
                f"({expected}, 1).  It holds exactly {expected} values, which is "
                f"what a one-column file of this length holds -- this looks like a "
                f"2D matrix or a multi-column file at the one-column path, where "
                f"flattening would silently renumber every pixel.  Refusing."
            )
        raise ConsistencyError(
            f"{label} in {path} has shape {values.shape}, expected ({expected}, 1): "
            f"a one-column file with exactly {expected} rows"
        )
    column = values[:, 0]
    if not np.all(np.isfinite(column)):
        raise ConsistencyError(f"{label} contains non-finite values: {path}")
    return column


def read_times_file(path: Path) -> tuple[str, tuple[str, ...]]:
    if not path.is_file():
        raise MissingInputError(f"LACC times file does not exist: {path}")
    center = ""
    lags: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("center_time="):
            center = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("lag_time="):
            lags.append(stripped.split("=", 1)[1].strip())
    if not center or not lags:
        raise ConsistencyError(
            f"LACC times file needs centre_time= and at least one lag_time=: {path}"
        )
    if len(set(lags)) != len(lags):
        raise ConsistencyError(f"LACC times file repeats a lag_time: {path}")
    return center, tuple(lags)


def detect_lag_directories(obs_bt_dir: Path) -> tuple[str, ...]:
    """Every ``BT_*`` directory that exists, for reporting what was left out."""
    if not obs_bt_dir.is_dir():
        return ()
    tags = []
    for child in sorted(obs_bt_dir.iterdir()):
        if child.is_dir() and child.name.startswith("BT_") and not child.name.startswith("BT_LACC_"):
            tags.append(child.name[len("BT_"):])
    return tuple(tags)


# =============================================================================
# Profile coordinates
# =============================================================================
@dataclass(frozen=True)
class ProfileCoordinates:
    tag: str
    path: Path
    elevation_km: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


def read_profile_coordinates(path: Path, expected: int, tag: str) -> ProfileCoordinates:
    """Marker-based parse; the fixed 185-line stride is deliberately not reused."""
    if not path.is_file():
        raise MissingInputError(f"profile file does not exist: {path}")
    lines = path.read_text(errors="replace").splitlines()
    elevations: list[float] = []
    lats: list[float] = []
    lons: list[float] = []
    for index, line in enumerate(lines):
        if line.strip() != PROFILE_COORDINATE_MARKER:
            continue
        if index + 1 >= len(lines):
            raise ConsistencyError(
                f"{path}: coordinate marker on line {index + 1} has no following row"
            )
        fields = lines[index + 1].split()
        if len(fields) < 3:
            raise ConsistencyError(
                f"{path}: expected elevation, latitude, longitude on line {index + 2}"
            )
        elevation, lat, lon = (float(value) for value in fields[:3])
        if not all(math.isfinite(value) for value in (elevation, lat, lon)):
            raise ConsistencyError(f"{path}: non-finite coordinate on line {index + 2}")
        elevations.append(elevation)
        lats.append(lat)
        lons.append(lon)

    if len(lats) != expected:
        raise ConsistencyError(
            f"{path}: found {len(lats)} coordinates, expected {expected}"
        )
    return ProfileCoordinates(
        tag=tag,
        path=path,
        elevation_km=np.asarray(elevations, dtype=float),
        lat=np.asarray(lats, dtype=float),
        lon=np.asarray(lons, dtype=float),
    )


def normalize_longitude_deg(value: float) -> float:
    """Wrap a longitude into [0, 360), the convention this script compares in.

    The DART ``loc3d`` block and the RTTOV profiles can disagree about whether
    longitudes run 0..360 or -180..180 while describing the same point.  The
    comparison is therefore made modulo 360 -- explicitly, here, instead of
    relying on a subtraction that would report a 360 deg offset as a 40 000 km
    distance.  Latitude has no such ambiguity and is never wrapped.
    """
    return float(value % 360.0)


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Great-circle distance in km, vectorised over the first argument.

    ``sin^2(dlon/2)`` is periodic in 360 deg, so the result is unchanged by
    adding or subtracting full turns to either longitude.
    """
    radius_km = 6371.0
    lat1_rad = np.radians(np.asarray(lat1, dtype=float))
    lon1_rad = np.radians(np.asarray(lon1, dtype=float))
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat1_rad - lat2_rad
    dlon = lon1_rad - lon2_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * math.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return radius_km * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def compare_profile_coordinates(
    reference: ProfileCoordinates, other: ProfileCoordinates, report_every: int = 1
) -> dict:
    """Per-observation displacement between two lag times' coordinates.

    The 9 h and 12 h profile files are known to move the sounding positions, so
    the lagged Hx cannot be assumed to sit at one fixed location.  This returns
    the displacement explicitly instead of leaving it implicit.
    """
    if reference.lat.shape != other.lat.shape:
        raise ConsistencyError(
            f"profile {reference.path} and {other.path} have different lengths"
        )
    dlat = other.lat - reference.lat
    dlon = other.lon - reference.lon
    # Local flat-earth displacement: resolved at the ~km scale, where the
    # question "did this sounding move?" is asked.
    dx_km = 111.320 * np.cos(np.radians(reference.lat)) * dlon
    dy_km = 110.574 * dlat
    distance = np.hypot(dx_km, dy_km)

    moved = distance > 1.0e-6
    first_moved = np.flatnonzero(moved)
    worst = int(np.argmax(distance)) if distance.size else 0

    rows = []
    for index in range(0, distance.size, max(1, report_every)):
        rows.append(
            {
                "raw_index": index + 1,
                "ref_tag": reference.tag,
                "ref_lat": float(reference.lat[index]),
                "ref_lon": float(reference.lon[index]),
                "other_tag": other.tag,
                "other_lat": float(other.lat[index]),
                "other_lon": float(other.lon[index]),
                "dlat_deg": float(dlat[index]),
                "dlon_deg": float(dlon[index]),
                "distance_km": float(distance[index]),
                "moved": bool(moved[index]),
            }
        )

    return {
        "summary": {
            "reference_tag": reference.tag,
            "other_tag": other.tag,
            "n_points": int(distance.size),
            "n_moved": int(moved.sum()),
            "max_distance_km": float(distance[worst]) if distance.size else 0.0,
            "max_at_raw_index": int(worst) + 1,
            "mean_distance_km": float(np.mean(distance)) if distance.size else 0.0,
            "first_moved_raw_index": int(first_moved[0]) + 1 if first_moved.size else None,
            "same_fixed_location": bool(not moved.any()),
        },
        "rows": rows,
    }


# =============================================================================
# Clear-sky masks and the filtered <-> raw index mapping
# =============================================================================
@dataclass(frozen=True)
class MaskInfo:
    tag: str
    path: Path
    values: np.ndarray  # int8, 0/1
    available: bool

    @property
    def n_kept(self) -> int:
        return int(self.values.sum()) if self.available else 0


def read_mask(path: Path, expected: int, tag: str) -> MaskInfo:
    values = read_column_file(path, expected, f"clear-sky mask {tag}")
    if not np.all(np.isin(values, (0.0, 1.0))):
        raise ConsistencyError(f"clear-sky mask {tag} must contain only 0 and 1: {path}")
    return MaskInfo(tag=tag, path=path, values=values.astype(np.int8), available=True)


def read_mask_or_unknown(path: Path, expected: int, tag: str) -> MaskInfo:
    """A missing per-lag mask is recorded as ``unknown``, never as clear."""
    if not path.is_file():
        return MaskInfo(tag=tag, path=path, values=np.full(expected, -1, dtype=np.int8), available=False)
    return read_mask(path, expected, tag)


def filtered_to_raw(mask: np.ndarray, filtered_index: int) -> int:
    """1-based kept-index -> 1-based raw row index."""
    kept = np.flatnonzero(mask) + 1
    if not (1 <= filtered_index <= kept.size):
        raise ConsistencyError(
            f"filtered index {filtered_index} is outside 1..{kept.size} of the kept list"
        )
    return int(kept[filtered_index - 1])


def raw_to_filtered(mask: np.ndarray, raw_index: int) -> int | None:
    """1-based raw row index -> 1-based kept-index, or None when masked out."""
    kept = np.flatnonzero(mask) + 1
    position = np.flatnonzero(kept == raw_index)
    return int(position[0]) + 1 if position.size else None


def resolve_obs_target(
    target: ObsTarget, unified_mask: np.ndarray, nobs_raw: int
) -> dict:
    """Fill in both index conventions and verify they agree."""
    if target.filtered_index is None and target.raw_index is None:
        raise ConsistencyError(f"observation {target.label} sets neither filtered_index nor raw_index")

    from_filtered = (
        filtered_to_raw(unified_mask, target.filtered_index)
        if target.filtered_index is not None
        else None
    )
    from_raw = (
        raw_to_filtered(unified_mask, target.raw_index) if target.raw_index is not None else None
    )

    if target.filtered_index is not None and target.raw_index is not None:
        if from_filtered != target.raw_index:
            raise ConsistencyError(
                f"observation {target.label}: filtered_index {target.filtered_index} maps to raw row "
                f"{from_filtered}, but raw_index {target.raw_index} was configured"
            )
    if from_filtered is None and from_raw is None:
        raise ConsistencyError(
            f"observation {target.label}: raw row {target.raw_index} is masked out in the "
            f"unified clear-sky mask, so it was never assimilated"
        )

    raw_index = from_filtered if from_filtered is not None else target.raw_index
    filtered_index = from_raw if from_raw is not None else target.filtered_index
    assert raw_index is not None and filtered_index is not None
    if not (1 <= raw_index <= nobs_raw):
        raise ConsistencyError(
            f"observation {target.label}: raw row {raw_index} outside 1..{nobs_raw}"
        )
    return {
        "label": target.label,
        "filtered_index": int(filtered_index),
        "raw_index": int(raw_index),
        "zero_based_raw_index": int(raw_index) - 1,
        "expected_lat": target.expected_lat,
        "expected_lon": target.expected_lon,
    }


# =============================================================================
# F-order numbering check
# =============================================================================
def check_matrix_flattening(
    matrix_path: Path,
    column_values: np.ndarray,
    side: int,
    label: str,
    column_path: Path | None = None,
) -> dict:
    """Confirm ``matrix.reshape(-1, order='F')`` reproduces the one-column file.

    The DART converter flattens the 2D BT array column-major.  A reader that
    silently used NumPy's default C-order would pick the wrong pixel for every
    index while still producing plausible-looking numbers, so this is checked
    rather than assumed.
    """
    if not matrix_path.is_file():
        return {
            "label": label,
            "path": str(matrix_path),
            "column_path": str(column_path) if column_path is not None else None,
            "available": False,
            "f_order_matches": None,
            "c_order_matches": None,
        }
    matrix = np.loadtxt(matrix_path, dtype=float)
    if matrix.shape != (side, side):
        raise ConsistencyError(
            f"{label}: expected a {side}x{side} matrix in {matrix_path}, got {matrix.shape}"
        )
    flat_f = matrix.reshape(-1, order="F")
    flat_c = matrix.reshape(-1, order="C")
    max_diff_f = float(np.max(np.abs(flat_f - column_values)))
    max_diff_c = float(np.max(np.abs(flat_c - column_values)))
    return {
        "label": label,
        "path": str(matrix_path),
        "column_path": str(column_path) if column_path is not None else None,
        "available": True,
        "side": side,
        "f_order_matches": bool(np.array_equal(flat_f, column_values)),
        "c_order_matches": bool(np.array_equal(flat_c, column_values)),
        "max_abs_diff_f_order": max_diff_f,
        "max_abs_diff_c_order": max_diff_c,
    }


# The two F-order suites are INDEPENDENT and must stay visibly separate: a
# passing NR check says nothing about the member Hx files, which are different
# files in different directories and are what the lagged covariance is built
# from.  ``enforce_f_order`` tags every result with the suite it came from so
# the JSON cannot present one suite's result as the other's.
SUITE_NR_BT = "nr_bt"
SUITE_MEMBER_HX = "member_hx"


def enforce_f_order(
    config: Config,
    matrix: Path,
    column_values: np.ndarray,
    label: str,
    column_path: Path | None = None,
    suite: str = SUITE_NR_BT,
) -> dict:
    """Require one lag's one-column BT to be the F-order flattening of its matrix.

    This runs for EVERY file that enters a computation -- every lag of the NR
    reference AND every member's Hx at every lag, not only the center time.  A
    file whose matrix silently disagrees has a misaligned pixel index, and the
    resulting ``d_j_NR`` feeds straight into the direction product -- so a
    mismatch refuses the whole diagnostic instead of being noted in the JSON
    while a direction conclusion is still printed.

    A missing matrix follows ``require_f_order_check`` explicitly: strict by
    default, and an auditable warning when deliberately relaxed.  It is never
    bypassed silently, and a relaxed skip is recorded as *not verified* rather
    than as a pass.
    """
    result = check_matrix_flattening(
        matrix, column_values, config.matrix_side, label, column_path
    )
    result["suite"] = suite
    if not result["available"]:
        if config.require_f_order_check:
            raise MissingInputError(
                f"{label}: the {config.matrix_side}x{config.matrix_side} BT matrix "
                f"{matrix} is missing, so the F-order numbering of "
                f"{result.get('column_path') or 'its one-column file'} could not be "
                f"verified.  Set require_f_order_check=False to proceed without "
                f"that check; the F-order numbering would then be UNVERIFIED, not "
                f"assumed correct."
            )
        result["enforcement"] = "skipped_matrix_absent"
        return result
    if not result["f_order_matches"]:
        raise ConsistencyError(
            f"{label}: the one-column BT file is NOT the F-order flattening of "
            f"{matrix} (max |diff| F-order="
            f"{result['max_abs_diff_f_order']:.6g}, C-order="
            f"{result['max_abs_diff_c_order']:.6g}).  Every index for this lag "
            f"would be misaligned; refusing to continue."
        )
    result["enforcement"] = "passed"
    return result


class HxFileCache:
    """Read and F-order-verify each BT file at most ONCE per run.

    Every observation rebuilds the same per-member Hx series, so without a cache
    a run over N observations would re-read and re-verify the whole file set N
    times (50 members x 3 lags x N).  Keying the verification by member and lag
    keeps the check as strict while paying for it once.

    The cache is created inside ``run()``, so mutating the files between runs
    (as the self-test does) can never be masked by stale values.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._columns: dict[Path, np.ndarray] = {}
        self._nr: dict[str, dict] = {}
        self._member: dict[tuple[int, str], dict] = {}

    # -- raw reads ---------------------------------------------------------
    def column(self, path: Path, label: str) -> np.ndarray:
        cached = self._columns.get(path)
        if cached is None:
            cached = read_column_file(path, self.config.nobs_raw, label)
            self._columns[path] = cached
        return cached

    # -- verification ------------------------------------------------------
    def verify_nr(self, tag: str) -> dict:
        """F-order check for one lag of the NR reference file (its own matrix)."""
        if tag not in self._nr:
            label = f"NR BT {tag}"
            column_path = nr_bt_path(self.config, tag)
            self._nr[tag] = enforce_f_order(
                self.config,
                matrix_path(self.config, tag),
                self.column(column_path, label),
                label,
                column_path,
                suite=SUITE_NR_BT,
            )
        return self._nr[tag]

    def verify_member(self, member: int, tag: str) -> dict:
        """F-order check for one member's Hx at one lag (its own matrix).

        Cached by (member, tag): the second observation to ask for this pair
        reuses the first result instead of re-reading a 676-row file.
        """
        key = (member, tag)
        if key not in self._member:
            label = f"member Hx mem{member:03d} {tag}"
            column_path = member_hx_path(self.config, member, tag)
            self._member[key] = enforce_f_order(
                self.config,
                member_matrix_path(self.config, member, tag),
                self.column(column_path, label),
                label,
                column_path,
                suite=SUITE_MEMBER_HX,
            )
        return self._member[key]

    # -- reads that must have been verified --------------------------------
    def nr_column(self, tag: str) -> np.ndarray:
        self.verify_nr(tag)
        return self.column(nr_bt_path(self.config, tag), f"NR BT {tag}")

    def member_column(self, member: int, tag: str) -> np.ndarray:
        self.verify_member(member, tag)
        return self.column(
            member_hx_path(self.config, member, tag), f"member Hx mem{member:03d} {tag}"
        )

    # -- reporting ---------------------------------------------------------
    def member_report(self, tag_hours: dict[str, int]) -> dict:
        """Summary of the member Hx suite, kept separate from the NR suite."""
        per_lag: dict[str, dict] = {}
        unverified: list[dict] = []
        n_verified = 0
        for (member, tag), entry in self._member.items():
            bucket = per_lag.setdefault(
                tag,
                {
                    "lag_hours": tag_hours.get(tag),
                    "n_members": 0,
                    "n_verified": 0,
                    "n_unverified": 0,
                    "members_unverified": [],
                    "max_abs_diff_c_order": 0.0,
                },
            )
            bucket["n_members"] += 1
            if entry.get("enforcement") == "passed":
                bucket["n_verified"] += 1
                n_verified += 1
                c_diff = entry.get("max_abs_diff_c_order")
                if isinstance(c_diff, float) and math.isfinite(c_diff):
                    bucket["max_abs_diff_c_order"] = max(
                        bucket["max_abs_diff_c_order"], c_diff
                    )
            else:
                bucket["n_unverified"] += 1
                bucket["members_unverified"].append(member)
                unverified.append(
                    {
                        "member": member,
                        "lag_tag": tag,
                        "lag_hours": tag_hours.get(tag),
                        "matrix_path": entry.get("path"),
                        "column_path": entry.get("column_path"),
                        "enforcement": entry.get("enforcement"),
                    }
                )
        n_pairs = len(self._member)
        if n_pairs == 0:
            status = "not_run"
        elif unverified:
            status = "incomplete_unverified_pairs"
        else:
            status = "passed"
        return {
            "scope": (
                "member Hx files under 4ens_BT_LACC/mem<m>/<sensor>/BT_<lag>/ -- "
                "NOT the NR reference files under 3obs_BT_LACC"
            ),
            "suite": SUITE_MEMBER_HX,
            "status": status,
            "all_verified": bool(n_pairs > 0 and not unverified),
            "n_member_lag_pairs": n_pairs,
            "n_verified": n_verified,
            "n_unverified": len(unverified),
            "unverified": unverified,
            "per_lag": per_lag,
            "member_numbers": self.config.member_numbers(),
            "matrix_side": self.config.matrix_side,
            "note": (
                "Every member-and-lag pair that enters the diagnostic is checked "
                "against its own 2D matrix; 'max_abs_diff_c_order' records what "
                "the C-order flattening would have differed by, so a vacuous "
                "check is visible.  A missing matrix under "
                "require_f_order_check=False appears in 'unverified' and is NOT "
                "counted as verified.  The per-lag NR results in "
                "f_order_numbering_per_obs are a different suite and do not "
                "cover these files."
            ),
        }


# =============================================================================
# obs_seq single-observation reader
# =============================================================================
@dataclass(frozen=True)
class ObsSeqRecord:
    path: Path
    obs_block_number: int | None
    obs_value: float | None
    lat: float | None
    lon: float | None
    external_fo: np.ndarray | None  # the WHOLE stored ensemble, DART member order
    external_fo_member_count: int | None = None
    external_fo_unavailable_reason: str | None = None


@dataclass(frozen=True)
class ExternalFO:
    """``external_FO`` cut down to the members this run actually configured.

    The stored array covers the FULL ensemble, so the configured members are
    picked out by DART member number (member ``m`` -> entry ``m - 1``).  The
    provenance is kept so the JSON can show which entries were compared.
    """

    path: Path
    declared_member_count: int
    ensemble_values: np.ndarray
    member_numbers: tuple[int, ...]
    selected_indices: tuple[int, ...]
    selected_values: np.ndarray


def _is_integer_token(token: str) -> bool:
    return token.lstrip("+-").isdigit()


def parse_obs_seq_single(path: Path) -> ObsSeqRecord:
    """Read value, loc3d and the FULL external_FO array from a DART single-obs obs_seq.

    ``loc3d`` stores longitude/latitude in radians.  The ``OBS n`` counter is
    the single-observation file's own block numbering -- it is NOT the raw 676
    row index, and no integer inside the visir block is treated as one.

    ``external_FO`` is returned whole, with the member count it declares, even
    when only a subset of the ensemble is configured: the caller knows the
    member numbers and does the selection.  A block whose count cannot be read
    reliably yields ``external_fo=None`` plus a reason, so the cross-check is
    reported unavailable rather than compared against a guessed
    correspondence -- the values are followed by another numeric line (an
    inflation), so "read numbers until they stop" would silently append it.
    """
    if not path.is_file():
        raise MissingInputError(f"single-observation obs_seq does not exist: {path}")
    lines = path.read_text(errors="replace").splitlines()

    obs_block_number: int | None = None
    obs_value: float | None = None
    lat: float | None = None
    lon: float | None = None
    external_fo: np.ndarray | None = None
    external_fo_member_count: int | None = None
    external_fo_reason: str | None = None

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        tokens = stripped.split()
        if len(tokens) >= 2 and tokens[0] == "OBS" and tokens[1].isdigit():
            obs_block_number = int(tokens[1])
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            if cursor < len(lines):
                head = lines[cursor].split()
                if head:
                    try:
                        obs_value = float(head[0])
                    except ValueError:
                        obs_value = None
                index = cursor
        elif tokens and tokens[0] == "loc3d":
            cursor = index + 1
            if cursor < len(lines):
                fields = lines[cursor].split()
                if len(fields) >= 2:
                    try:
                        lon = math.degrees(float(fields[0]))
                        lat = math.degrees(float(fields[1]))
                    except ValueError as exc:
                        raise ConsistencyError(
                            f"{path}: the loc3d block on line {cursor + 1} is not "
                            f"numeric ({lines[cursor].strip()!r}); the observation "
                            f"location cannot be checked, so this file must not be "
                            f"used as the assimilated observation"
                        ) from exc
                    index = cursor
        elif tokens and tokens[0] == "external_FO":
            cursor = index + 1
            # DART writes the member count on the line AFTER ``external_FO``;
            # some tooling appends it to the keyword line instead.  Either way
            # it must be consumed as a count, never as the first member value,
            # or every member is shifted by one.
            declared: int | None = None
            if len(tokens) >= 2 and _is_integer_token(tokens[1]):
                declared = int(tokens[1])
            elif cursor < len(lines):
                peek = lines[cursor].split()
                if len(peek) == 1 and _is_integer_token(peek[0]):
                    declared = int(peek[0])
                    cursor += 1
            if declared is None:
                external_fo_reason = (
                    f"{path}: the external_FO block declares no member count "
                    f"(neither on the keyword line nor on the line after it), so "
                    f"the stored ensemble size -- and with it which entry belongs "
                    f"to which member -- cannot be established reliably.  The "
                    f"cross-check is reported unavailable rather than compared "
                    f"against a guessed correspondence."
                )
                index = cursor  # the block ends here; skip past it below
            elif declared < 1:
                raise ConsistencyError(
                    f"{path}: external_FO declares {declared} members; a member "
                    f"count must be positive"
                )
            else:
                values: list[float] = []
                while len(values) < declared and cursor < len(lines):
                    fields = lines[cursor].split()
                    if not fields:
                        cursor += 1
                        continue
                    try:
                        line_values = [float(field) for field in fields]
                    except ValueError:
                        break
                    values.extend(line_values)
                    if len(values) > declared:
                        raise ConsistencyError(
                            f"{path}: external_FO declares {declared} members but "
                            f"the value lines carry {len(values)} numbers by line "
                            f"{cursor + 1}; the declared count and the stored values "
                            f"disagree, so no member correspondence can be assumed"
                        )
                    cursor += 1
                if len(values) != declared:
                    raise ConsistencyError(
                        f"{path}: external_FO declares {declared} members but only "
                        f"{len(values)} values were found; the declared count and the "
                        f"stored values disagree, so no member correspondence can be "
                        f"assumed"
                    )
                stored = np.asarray(values, dtype=float)
                if not np.all(np.isfinite(stored)):
                    raise ConsistencyError(
                        f"{path}: external_FO contains non-finite values, so a real "
                        f"disagreement could not be told apart from a missing number"
                    )
                external_fo = stored
                external_fo_member_count = declared
                index = cursor
        index += 1

    if obs_value is None:
        raise ConsistencyError(f"{path}: no OBS block with a value was found")
    return ObsSeqRecord(
        path=path,
        obs_block_number=obs_block_number,
        obs_value=obs_value,
        lat=lat,
        lon=lon,
        external_fo=external_fo,
        external_fo_member_count=external_fo_member_count,
        external_fo_unavailable_reason=external_fo_reason,
    )


def select_external_fo(record: ObsSeqRecord, members: list[int]) -> ExternalFO | None:
    """Pick the stored ``external_FO`` entries that belong to the configured members.

    The array covers the WHOLE ensemble in DART member order, so member ``m`` is
    entry ``m - 1``.  Taking the first ``len(members)`` entries instead would
    compare member 2's Hx against member 1's stored value whenever the run does
    not start at member 1 -- a mismatch that looks like a genuine disagreement
    between the files rather than a bookkeeping error, which is exactly why the
    correspondence is resolved by member NUMBER and not by count.
    """
    if record.external_fo is None:
        return None
    full = np.asarray(record.external_fo, dtype=float).reshape(-1)
    declared = record.external_fo_member_count
    if declared is None:
        raise ConsistencyError(
            f"{record.path}: external_FO carries {full.size} values but no member "
            f"count, so the member correspondence cannot be established"
        )
    if int(declared) != full.size:
        raise ConsistencyError(
            f"{record.path}: external_FO declares {declared} members but stores "
            f"{full.size} values; refusing to guess which entry belongs to which "
            f"member"
        )
    if not members:
        raise ConsistencyError(f"{record.path}: no ensemble members were configured")
    if min(members) < 1 or max(members) > full.size:
        offending = max(members) if max(members) > full.size else min(members)
        raise ConsistencyError(
            f"{record.path}: external_FO stores members 1..{full.size}, but member "
            f"{offending} was requested (member_start={min(members)}, "
            f"member_end={max(members)}).  The stored values cannot be matched to "
            f"the configured members; refusing to compare against the wrong entries."
        )
    indices = tuple(member - 1 for member in members)
    return ExternalFO(
        path=record.path,
        declared_member_count=int(declared),
        ensemble_values=full,
        member_numbers=tuple(members),
        selected_indices=indices,
        selected_values=full[list(indices)],
    )


# =============================================================================
# netCDF readers
# =============================================================================
@dataclass(frozen=True)
class GridField:
    """One horizontal field plus the grid and water mask it lives on."""

    path: Path
    field: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    water: np.ndarray
    water_source: str
    time_string: str | None


def _squeeze_leading_time(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def read_time_string(dataset: nc.Dataset) -> str | None:
    if "Times" not in dataset.variables:
        return None
    raw = dataset.variables["Times"][:]
    if raw.ndim == 2:
        raw = raw[0]
    chars = []
    for item in np.atleast_1d(raw):
        chars.append(item.decode("ascii") if isinstance(item, bytes) else str(item))
    text = "".join(chars).strip()
    return text or None


def select_water_mask(
    dataset: nc.Dataset, path: Path, source: str
) -> tuple[np.ndarray, str]:
    """Return a boolean water mask and the variable it came from."""
    if source not in {"auto", "LANDMASK", "XLAND"}:
        raise ValueError("water_mask_source must be auto, LANDMASK or XLAND")

    order = ["LANDMASK", "XLAND"] if source == "auto" else [source]
    for name in order:
        if name not in dataset.variables:
            continue
        raw = np.ma.filled(dataset.variables[name][:], np.nan)
        mask = _squeeze_leading_time(raw)
        if mask.ndim != 2:
            raise ConsistencyError(f"{path}: {name} is not 2D after squeezing: {mask.shape}")
        if name == "LANDMASK":
            water = mask < 0.5  # 0 = water, 1 = land
        else:
            water = mask >= 1.5  # 1 = land, 2 = water
        return water, name

    if source == "auto":
        raise ConsistencyError(
            f"{path}: neither LANDMASK nor XLAND is present, so the water test "
            f"cannot be applied"
        )
    raise ConsistencyError(f"{path}: requested water-mask variable {source} is absent")


def read_level_zero_field(
    path: Path,
    variable: str,
    level: int,
    lat_var: str,
    lon_var: str,
    water_source: str,
) -> GridField:
    """Read ONE level of a 3D/4D field plus its grid and water mask.

    Only ``variable[0, level, :, :]`` is requested from netCDF4, so the full
    30-layer OM_TMP array is never materialised.
    """
    if not path.is_file():
        raise MissingInputError(f"state file does not exist: {path}")
    with nc.Dataset(path) as dataset:
        missing = [
            name
            for name in (variable, lat_var, lon_var)
            if name not in dataset.variables
        ]
        if missing:
            raise ConsistencyError(f"{path} is missing variables: {', '.join(missing)}")

        variable_object = dataset.variables[variable]
        dimensions = variable_object.dimensions
        if len(dimensions) < 3:
            raise ConsistencyError(
                f"{path}: {variable} has dimensions {dimensions}; expected a level axis"
            )
        # (..., level, y, x): read one time slice, one level, all horizontal.
        raw = np.ma.filled(variable_object[0, level, :, :], np.nan)
        field = np.asarray(raw, dtype=float)
        if field.ndim != 2:
            raise ConsistencyError(
                f"{path}: {variable}[0, {level}] is not 2D: {field.shape}"
            )
        # XLAT/XLONG carry a Time dimension in wrfout but may already be 2D in
        # the DART preassim members; both are supported.
        lat = _squeeze_leading_time(np.ma.filled(dataset.variables[lat_var][:], np.nan))
        lon = _squeeze_leading_time(np.ma.filled(dataset.variables[lon_var][:], np.nan))
        water, water_name = select_water_mask(dataset, path, water_source)
        time_string = read_time_string(dataset)

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if not (lat.shape == lon.shape == field.shape == water.shape):
        raise ConsistencyError(
            f"{path}: grid shapes disagree -- {variable} {field.shape}, {lat_var} "
            f"{lat.shape}, {lon_var} {lon.shape}, water {water.shape}"
        )
    return GridField(
        path=path,
        field=field,
        lat=lat,
        lon=lon,
        water=water,
        water_source=water_name,
        time_string=time_string,
    )


# =============================================================================
# Point sampling.  Never extrapolates, never silently falls back.
# =============================================================================
@dataclass(frozen=True)
class SampleResult:
    value: float
    sample_lat: float
    sample_lon: float
    distance_km: float
    mode: str
    n_support_points: int
    local_window_deg: float


def sample_point(
    grid: GridField,
    target_lat: float,
    target_lon: float,
    mode: str,
    config: Config,
) -> SampleResult:
    """Sample ``grid`` at one physical point.

    Refusals, in order:

      * the nearest grid cell is not water (``sample_require_water``) -> the
        point is not an ocean point, so an ocean analysis value there would be
        meaningless;
      * ``mode == "linear"``: the point falls outside the triangulation of the
        water support -> ``OutsideDomainError``.  No nearest-neighbour fill-in
        happens, because a silent fallback is exactly the "extrapolation by
        accident" this script must not perform;
      * ``mode == "nearest"``: the nearest water point is further than
        ``nearest_max_km`` -> ``OutsideDomainError``.

    The actually-sampled coordinates and the distance are always returned so
    the caller can report where the value really came from.
    """
    if mode not in {"linear", "nearest"}:
        raise ValueError("sample_mode must be 'linear' or 'nearest'")
    if not (math.isfinite(target_lat) and math.isfinite(target_lon)):
        raise SamplingError("target latitude/longitude must be finite")

    finite = np.isfinite(grid.field) & np.isfinite(grid.lat) & np.isfinite(grid.lon)
    support = finite & grid.water

    distance_to_cells = haversine_km(grid.lat, grid.lon, target_lat, target_lon)
    nearest_flat = int(np.argmin(np.where(np.isfinite(distance_to_cells), distance_to_cells, np.inf)))
    nearest_cell = np.unravel_index(nearest_flat, grid.field.shape)
    if config.sample_require_water and not bool(grid.water[nearest_cell]):
        raise NonOceanPointError(
            f"{grid.path}: nearest cell to ({target_lat:.4f}, {target_lon:.4f}) is "
            f"{grid.water_source}={grid.water[nearest_cell]} (not water); refusing to "
            f"sample an ocean variable there.  Set sample_require_water=False only "
            f"as a deliberate override."
        )

    if int(support.sum()) < 3:
        raise OutsideDomainError(
            f"{grid.path}: only {int(support.sum())} finite water points; "
            f"cannot interpolate"
        )

    if mode == "linear":
        window = config.local_window_deg
        while True:
            subset = (
                support
                & (np.abs(grid.lat - target_lat) <= window)
                & (np.abs(grid.lon - target_lon) <= window)
            )
            if int(subset.sum()) >= 3:
                break
            if window >= config.max_local_window_deg:
                raise OutsideDomainError(
                    f"{grid.path}: fewer than 3 water points within "
                    f"{config.max_local_window_deg} deg of ({target_lat:.4f}, "
                    f"{target_lon:.4f}); refusing to extrapolate"
                )
            window = min(window * 2.0, config.max_local_window_deg)

        source_lat = grid.lat[subset]
        source_lon = grid.lon[subset]
        source_value = grid.field[subset]
        interpolator = LinearNDInterpolator(
            np.column_stack((source_lon, source_lat)), source_value, fill_value=np.nan
        )
        value = float(np.asarray(interpolator(target_lon, target_lat)).reshape(-1)[0])
        if not math.isfinite(value):
            raise OutsideDomainError(
                f"{grid.path}: ({target_lat:.4f}, {target_lon:.4f}) is outside the "
                f"triangulation of the water support; refusing to extrapolate.  Use "
                f"sample_mode='nearest' explicitly if a nearest water point is "
                f"acceptable."
            )
        return SampleResult(
            value=value,
            sample_lat=float(target_lat),
            sample_lon=float(target_lon),
            distance_km=0.0,
            mode=mode,
            n_support_points=int(subset.sum()),
            local_window_deg=float(window),
        )

    source_lat = grid.lat[support]
    source_lon = grid.lon[support]
    source_value = grid.field[support]
    # ONE index decides the value, the reported coordinates AND the distance.
    # Selecting the value with one metric (e.g. a plane-distance interpolator)
    # and reporting the coordinates with another (great-circle) can return the
    # value of one cell while naming a different one -- a mismatch that is
    # invisible in linear mode and would silently misreport where the state
    # came from.
    support_distance = haversine_km(source_lat, source_lon, target_lat, target_lon)
    nearest_support = int(np.argmin(support_distance))
    distance = float(support_distance[nearest_support])
    if distance > config.nearest_max_km:
        # The limit applies to the point that was actually selected.
        raise OutsideDomainError(
            f"{grid.path}: nearest water point to ({target_lat:.4f}, {target_lon:.4f}) "
            f"is ({source_lat[nearest_support]:.4f}, {source_lon[nearest_support]:.4f}), "
            f"{distance:.2f} km away, beyond nearest_max_km="
            f"{config.nearest_max_km:g}; refusing the snap"
        )
    return SampleResult(
        value=float(source_value[nearest_support]),
        sample_lat=float(source_lat[nearest_support]),
        sample_lon=float(source_lon[nearest_support]),
        distance_km=distance,
        mode=mode,
        n_support_points=int(support.sum()),
        local_window_deg=float("nan"),
    )


# =============================================================================
# Ensemble moments (ddof = 1)
# =============================================================================
@dataclass(frozen=True)
class PairMoments:
    n_used: int
    mean_x: float
    mean_y: float
    std_x: float
    std_y: float
    cov: float
    corr: float
    degenerate: bool

    def as_row(self, prefix_x: str, prefix_y: str) -> dict:
        return {
            f"{prefix_x}_mean": self.mean_x,
            f"{prefix_x}_std": self.std_x,
            f"{prefix_y}_mean": self.mean_y,
            f"{prefix_y}_std": self.std_y,
            f"cov_{prefix_x}_{prefix_y}": self.cov,
            f"corr_{prefix_x}_{prefix_y}": self.corr,
        }


def pair_moments(x: np.ndarray, y: np.ndarray, min_members: int = 2) -> PairMoments:
    """Mean / std / covariance / correlation over members, ddof = 1.

    Non-finite members are dropped pairwise.  A zero-variance ensemble is
    reported as ``degenerate`` with NaN covariance and correlation -- it is
    never quietly turned into a zero covariance, because a zero covariance and
    an undefined covariance mean very different things when the answer is read
    as a direction.
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"pair_moments needs equal-length inputs, got {x_arr.shape} and {y_arr.shape}"
        )
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    n_used = int(valid.sum())
    if n_used < min_members:
        return PairMoments(
            n_used=n_used,
            mean_x=float("nan"),
            mean_y=float("nan"),
            std_x=float("nan"),
            std_y=float("nan"),
            cov=float("nan"),
            corr=float("nan"),
            degenerate=True,
        )
    xv = x_arr[valid]
    yv = y_arr[valid]
    mean_x = float(np.mean(xv))
    mean_y = float(np.mean(yv))
    std_x = float(np.std(xv, ddof=1))
    std_y = float(np.std(yv, ddof=1))
    cov = float(np.sum((xv - mean_x) * (yv - mean_y)) / (n_used - 1))
    degenerate = std_x == 0.0 or std_y == 0.0
    corr = float("nan") if degenerate else float(cov / (std_x * std_y))
    return PairMoments(
        n_used=n_used,
        mean_x=mean_x,
        mean_y=mean_y,
        std_x=std_x,
        std_y=std_y,
        cov=cov,
        corr=corr,
        degenerate=degenerate,
    )


def mean_over_members(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


# =============================================================================
# Alignment classification
# =============================================================================
def classify_alignment(
    product: float, e_ocean: float, corr: float, ocean_std: float, near_zero_frac: float
) -> str:
    """Direction flag for one lag or window -- from ``product * e_ocean``.

    The decision variable is::

        alignment = product * e_ocean,   product = c_j * d_j_NR

    ``product`` alone only fixes the sign of the ocean increment.  Whether that
    increment moves the state TOWARD the NR depends on which way the prior is
    wrong, i.e. on the sign of ``e_ocean``: with ``e_ocean < 0`` the same
    ``product`` means the opposite direction.  A product-only test therefore
    inverts the answer whenever the NR is colder than the prior mean, which is
    half of all cases.  Both the per-lag rows and the window rows call this
    function, so they cannot drift apart.

    The near-zero tests sit on quantities that carry a physical scale --
    ``e_ocean`` against the prior ocean spread, and the correlation against a
    bare threshold -- rather than on ``alignment`` itself, whose units are a
    product of four others and therefore have no absolute scale to compare to.
    ``e_ocean == 0`` is handled explicitly so that ``near_zero_frac = 0`` (which
    disables the relative test) still cannot classify a directionless case.
    """
    if not math.isfinite(e_ocean):
        return "UNDEFINED_E_O"
    if not math.isfinite(ocean_std) or ocean_std == 0.0:
        return "UNDEFINED_E_O"
    if e_ocean == 0.0:
        # The NR coincides with the prior mean: there is no correction to aim
        # at, so "toward"/"away" is meaningless regardless of near_zero_frac.
        return "UNDEFINED_E_O"
    if abs(e_ocean) < near_zero_frac * ocean_std:
        return "UNDEFINED_E_O"
    if not math.isfinite(product):
        return "UNDEFINED_PRODUCT"
    if not math.isfinite(corr) or abs(corr) < near_zero_frac:
        return "WEAK_COUPLING"
    alignment = product * e_ocean
    if alignment > 0.0:
        return "TOWARD_NR"
    if alignment < 0.0:
        return "AWAY_FROM_NR"
    return "EXACTLY_ZERO"


# =============================================================================
# Path construction
# =============================================================================
def member_hx_dir(config: Config, member: int, tag: str) -> Path:
    return (
        config.resolved_ens_bt_dir()
        / f"mem{member:03d}"
        / config.sensor
        / bt_directory_name(tag)
    )


def member_hx_path(config: Config, member: int, tag: str) -> Path:
    return member_hx_dir(config, member, tag) / f"obs_{config.domain}_ch{config.channel}_totalline.txt"


def member_matrix_path(config: Config, member: int, tag: str) -> Path:
    """The member's own 2D BT matrix, i.e. the reference its column must match.

    It sits NEXT TO the member's one-column file (same directory, same domain
    and channel), so the F-order check compares a lag against its own matrix
    instead of against another file that merely happens to parse.
    """
    return member_hx_dir(config, member, tag) / f"obs_{config.domain}_ch{config.channel}.txt"


def nr_bt_path(config: Config, tag: str) -> Path:
    return (
        config.resolved_obs_bt_dir()
        / bt_directory_name(tag)
        / f"obs_{config.domain}_ch{config.channel}_totalline.txt"
    )


def per_lag_mask_path(config: Config, tag: str) -> Path:
    return config.resolved_obs_bt_dir() / bt_directory_name(tag) / "clear_sky_mask.txt"


def matrix_path(config: Config, tag: str) -> Path:
    return (
        config.resolved_obs_bt_dir()
        / bt_directory_name(tag)
        / f"obs_{config.domain}_ch{config.channel}.txt"
    )


def noisy_bt_path(config: Config, tag: str) -> Path:
    return (
        config.resolved_obs_bt_dir()
        / bt_directory_name(tag)
        / f"obs_{config.domain}_ch{config.channel}_totalline_withpert.txt"
    )


def unified_noisy_bt_path(config: Config) -> Path:
    return (
        config.resolved_obs_bt_dir()
        / f"BT_LACC_{config.center_tag()}"
        / f"obs_{config.domain}_ch{config.channel}_totalline_withpert.txt"
    )


def profile_path(config: Config, tag: str) -> Path:
    return (
        config.resolved_profile_dir()
        / f"profile_{config.domain}_LACC_{config.center_tag()}"
        / profile_file_name(tag)
    )


def single_obs_path(config: Config, label: str) -> Path:
    return config.resolved_single_obs_dir() / f"obs_seq.out_LACC_single_{label}"


def background_member_path(config: Config, member: int) -> Path:
    if config.background_choice == "obs_seq111":
        return config.obs_seq111_dir / config.obs_seq111_pattern.format(
            mem=member, domain=config.domain
        )
    if config.background_choice == "mem_all_time":
        directory = (
            config.mem_all_time_dir
            if config.mem_all_time_dir is not None
            else config.project_root / "4assimilation" / "0mem_all_time" / config.center_tag()
        )
        return directory / config.mem_all_time_pattern.format(mem=member, domain=config.domain)
    raise ValueError(
        f"background_choice must be 'obs_seq111' or 'mem_all_time', got "
        f"{config.background_choice!r}"
    )


# =============================================================================
# Configuration validation
# =============================================================================
def validate_config(config: Config) -> None:
    config.center_datetime()  # raises on a malformed center_time
    if config.member_start < 1 or config.member_end < config.member_start:
        raise ValueError("members must satisfy 1 <= member_start <= member_end")
    if config.nobs_raw < 1:
        raise ValueError("nobs_raw must be positive")
    if config.matrix_side**2 != config.nobs_raw:
        raise ValueError(
            f"matrix_side**2 = {config.matrix_side ** 2} must equal nobs_raw = {config.nobs_raw}"
        )
    if config.background_choice not in {"obs_seq111", "mem_all_time"}:
        raise ValueError("background_choice must be 'obs_seq111' or 'mem_all_time'")
    if config.extra_lag_hours and not config.enable_extra_lag_diagnostics:
        raise ValueError(
            f"extra_lag_hours={config.extra_lag_hours} were requested but "
            f"enable_extra_lag_diagnostics is False.  Lags outside the assimilated "
            f"window are never added automatically: set "
            f"enable_extra_lag_diagnostics=True deliberately."
        )
    if config.sample_mode not in {"linear", "nearest"}:
        raise ValueError("sample_mode must be 'linear' or 'nearest'")
    if not (0.0 <= config.near_zero_frac < 1.0):
        raise ValueError("near_zero_frac must lie in [0, 1)")
    if config.state_level < 0:
        raise ValueError("state_level must be nonnegative")
    if not config.lag_hours:
        raise ValueError(
            "lag_hours must name at least one lag time to diagnose"
        )
    # A tolerance that cannot decide anything must not silently accept every
    # location: NaN compares False against everything, so the gate would pass.
    tolerance = float(config.single_obs_coord_tol_km)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"single_obs_coord_tol_km={config.single_obs_coord_tol_km!r} must be "
            f"finite and non-negative"
        )


def resolve_lag_schedule(
    config: Config, times_file_center: str, times_file_lags: tuple[str, ...]
) -> dict:
    """Reconcile the assimilated LACC window with the requested lag set."""
    center_dt = config.center_datetime()
    if times_file_center != config.center_tag():
        raise ConsistencyError(
            f"LACC times file centre_time={times_file_center} disagrees with the "
            f"configured center tag {config.center_tag()}"
        )

    assimilated: dict[int, str] = {}
    for tag in times_file_lags:
        hours = lag_hours_between(center_dt, tag_to_datetime(tag, center_dt))
        if hours in assimilated:
            raise ConsistencyError(f"two LACC lag times map to lag {hours} h")
        assimilated[hours] = tag

    requested = sorted(set(config.lag_hours))
    missing = [hours for hours in requested if hours not in assimilated]
    if missing:
        raise ConsistencyError(
            f"requested lag hours {missing} are not in the assimilated LACC window "
            f"{sorted(assimilated)}; refusing to diagnose a time the experiment "
            f"never assimilated"
        )

    schedule = [(hours, assimilated[hours]) for hours in requested]
    for hours in sorted(set(config.extra_lag_hours)):
        if hours in assimilated:
            raise ConsistencyError(
                f"lag {hours} h is already in the assimilated window; list it in "
                f"lag_hours instead of extra_lag_hours"
            )
        schedule.append((hours, datetime_to_tag(center_dt - timedelta(hours=hours))))
    schedule.sort(key=lambda item: item[0])

    return {
        "assimilated_lag_hours": sorted(assimilated),
        "assimilated_tags": {hours: assimilated[hours] for hours in sorted(assimilated)},
        "diagnosed_lag_hours": [hours for hours, _ in schedule],
        "schedule": schedule,
        "extra_lag_hours_enabled": bool(config.enable_extra_lag_diagnostics),
        "extra_lag_hours_used": sorted(set(config.extra_lag_hours))
        if config.enable_extra_lag_diagnostics
        else [],
    }


def prefix_windows(lag_hours: list[int]) -> list[tuple[int, ...]]:
    """Equal-weight prefix windows [0], [0, 3], [0, 3, 6] over the diagnosed set."""
    ordered = sorted(lag_hours)
    return [tuple(ordered[: size]) for size in range(1, len(ordered) + 1)]


# =============================================================================
# Core diagnostic for one observation
# =============================================================================
@dataclass
class ObservationContext:
    """Everything about one observation that does not depend on the lag."""

    label: str
    filtered_index: int
    raw_index: int
    zero_based_raw_index: int
    obs_lat: float
    obs_lon: float
    state_lat: float
    state_lon: float
    ocean_prior: np.ndarray  # (member,) center-time OM_TMP at the state point
    ocean_nr: float
    ocean_sample: SampleResult
    nr_sample: SampleResult
    background_files: list[str]
    background_times: list[str | None]
    single_obs: ObsSeqRecord | None
    single_obs_location: dict | None
    external_fo: ExternalFO | None  # this run's members, selected by member number


def build_observation_context(
    config: Config,
    target_info: dict,
    coordinates: ProfileCoordinates,
    background_grids: list[GridField],
    nr_grid: GridField,
) -> ObservationContext:
    """Everything about one observation that does not depend on the lag.

    ``coordinates`` MUST be the CENTER-TIME profile.  It defines which raw row
    is being diagnosed, the default ocean state point, and the reference the
    single-obs ``loc3d`` is checked against -- all of which belong to
    ``config.center_time`` and none of which may drift with whichever lag
    happens to be first in the requested diagnostic set.
    """
    members = config.member_numbers()
    raw_index = int(target_info["raw_index"])
    position = raw_index - 1
    obs_lat = float(coordinates.lat[position])
    obs_lon = float(coordinates.lon[position])

    if target_info["expected_lat"] is not None and target_info["expected_lon"] is not None:
        dlat = abs(obs_lat - float(target_info["expected_lat"]))
        dlon = abs(obs_lon - float(target_info["expected_lon"]))
        if dlat > 1.0e-3 or dlon > 1.0e-3:
            raise ConsistencyError(
                f"observation {target_info['label']}: profile coordinates "
                f"({obs_lat:.4f}, {obs_lon:.4f}) disagree with the verified values "
                f"({target_info['expected_lat']}, {target_info['expected_lon']})"
            )

    state_lat = obs_lat if config.state_lat is None else float(config.state_lat)
    state_lon = obs_lon if config.state_lon is None else float(config.state_lon)

    # Ocean prior: one sampled value per member, all at the SAME physical point.
    ocean_values: list[float] = []
    samples: list[SampleResult] = []
    for member_index, grid in enumerate(background_grids):
        sample = sample_point(grid, state_lat, state_lon, config.sample_mode, config)
        samples.append(sample)
        ocean_values.append(sample.value)
    ocean_prior = np.asarray(ocean_values, dtype=float)

    nr_sample = sample_point(nr_grid, state_lat, state_lon, config.sample_mode, config)

    if config.sample_mode == "nearest":
        for member_index, sample in enumerate(samples):
            if sample.sample_lat != samples[0].sample_lat or sample.sample_lon != samples[0].sample_lon:
                raise ConsistencyError(
                    f"nearest sampling picked a different grid point for member "
                    f"{members[member_index]} than for {members[0]}; the members are "
                    f"not on a common grid"
                )

    single_obs: ObsSeqRecord | None = None
    single_obs_location: dict | None = None
    external_fo: ExternalFO | None = None
    path = single_obs_path(config, target_info["label"])
    if path.is_file():
        single_obs = parse_obs_seq_single(path)
        # The obs_seq loc3d must describe THIS observation, not merely carry a
        # matching external_FO.
        single_obs_location = check_single_obs_location(config, single_obs, obs_lat, obs_lon)
        # external_FO covers the whole ensemble; the entries compared here are
        # the configured members, picked by DART member number.
        external_fo = select_external_fo(single_obs, members)

    return ObservationContext(
        label=str(target_info["label"]),
        filtered_index=int(target_info["filtered_index"]),
        raw_index=raw_index,
        zero_based_raw_index=int(target_info["zero_based_raw_index"]),
        obs_lat=obs_lat,
        obs_lon=obs_lon,
        state_lat=float(state_lat),
        state_lon=float(state_lon),
        ocean_prior=ocean_prior,
        ocean_nr=nr_sample.value,
        ocean_sample=samples[0],
        nr_sample=nr_sample,
        background_files=[str(grid.path) for grid in background_grids],
        background_times=[grid.time_string for grid in background_grids],
        single_obs=single_obs,
        single_obs_location=single_obs_location,
        external_fo=external_fo,
    )


@dataclass
class LagSeries:
    """The per-lag Hx ensemble and NR BT for one observation."""

    lag_hours: list[int]
    tags: list[str]
    valid_times: list[str]
    clear_sky_flags: list[str]
    hx: np.ndarray  # (n_lag, n_member) at the observation pixel
    y_nr: np.ndarray  # (n_lag,)
    hx_file_count: int
    obs_lat_at_lag: list[float]  # the coordinate row each lag's Hx was built on
    obs_lon_at_lag: list[float]
    obs_displacement_km: list[float]  # distance from the center-time position


def load_lag_series(
    config: Config,
    context: ObservationContext,
    schedule: list[tuple[int, str]],
    profiles: dict[str, ProfileCoordinates],
    files: HxFileCache,
) -> LagSeries:
    """Read the per-lag Hx / NR series for one observation.

    Every value comes out of ``files``, which F-order-verifies each member and
    lag against ITS OWN 2D matrix before returning it and then caches it, so the
    numbers used here are the numbers that were checked -- not a second read of
    the same path.
    """
    members = config.member_numbers()
    n_member = len(members)
    positions = context.zero_based_raw_index

    hx = np.full((len(schedule), n_member), np.nan, dtype=float)
    y_nr = np.full(len(schedule), np.nan, dtype=float)
    clear_sky_flags: list[str] = []
    valid_times: list[str] = []
    tags: list[str] = []
    lag_hours: list[int] = []
    obs_lat_at_lag: list[float] = []
    obs_lon_at_lag: list[float] = []
    obs_displacement_km: list[float] = []
    file_count = 0

    center_dt = config.center_datetime()
    for lag_index, (hours, tag) in enumerate(schedule):
        lag_hours.append(hours)
        tags.append(tag)
        valid_times.append(tag_to_datetime(tag, center_dt).strftime("%Y-%m-%d_%H:%M:%S"))

        for member_index, member in enumerate(members):
            hx[lag_index, member_index] = files.member_column(member, tag)[positions]
            file_count += 1

        y_nr[lag_index] = files.nr_column(tag)[positions]

        mask = read_mask_or_unknown(per_lag_mask_path(config, tag), config.nobs_raw, tag)
        if not mask.available:
            clear_sky_flags.append("unknown")
        else:
            clear_sky_flags.append("clear" if int(mask.values[positions]) == 1 else "not_clear")

        # The raw row index identifies the observation, but the COORDINATES of
        # that row come from the lag time's own profile and are known to move
        # for the 9 h / 12 h files.  Recording them per lag keeps the
        # "one fixed location" assumption visible instead of implied.
        lag_lat = float(profiles[tag].lat[positions])
        lag_lon = float(profiles[tag].lon[positions])
        obs_lat_at_lag.append(lag_lat)
        obs_lon_at_lag.append(lag_lon)
        obs_displacement_km.append(
            float(haversine_km(np.array([lag_lat]), np.array([lag_lon]),
                               context.obs_lat, context.obs_lon)[0])
        )

    return LagSeries(
        lag_hours=lag_hours,
        tags=tags,
        valid_times=valid_times,
        clear_sky_flags=clear_sky_flags,
        hx=hx,
        y_nr=y_nr,
        hx_file_count=file_count,
        obs_lat_at_lag=obs_lat_at_lag,
        obs_lon_at_lag=obs_lon_at_lag,
        obs_displacement_km=obs_displacement_km,
    )


def compute_lag_rows(
    config: Config, context: ObservationContext, series: LagSeries
) -> tuple[list[dict], list[dict]]:
    """Per-lag rows and the equal-weight window rows for one observation."""
    ocean_mean = mean_over_members(context.ocean_prior)
    ocean_std = float(np.nanstd(context.ocean_prior, ddof=1)) if np.isfinite(context.ocean_prior).sum() > 1 else float("nan")
    e_ocean = context.ocean_nr - ocean_mean

    lag_rows: list[dict] = []
    d_by_lag: list[float] = []

    for lag_index, hours in enumerate(series.lag_hours):
        hx_members = series.hx[lag_index]
        moments = pair_moments(context.ocean_prior, hx_members)
        hx_mean = mean_over_members(hx_members)
        d_j = series.y_nr[lag_index] - hx_mean
        product = moments.cov * d_j
        d_by_lag.append(d_j)
        alignment = product * e_ocean

        lag_rows.append(
            {
                # --- fields named in the diagnostic contract ---
                "obs_id": context.label,
                "raw_index": context.raw_index,
                "lat": context.obs_lat,
                "lon": context.obs_lon,
                "state_lat": context.state_lat,
                "state_lon": context.state_lon,
                "lag_hours": hours,
                "valid_time": series.valid_times[lag_index],
                "clear_sky_flag": series.clear_sky_flags[lag_index],
                "hx_mean": hx_mean,
                "hx_std": moments.std_y,
                "y_NR": series.y_nr[lag_index],
                "c_j": moments.cov,
                "r_j": moments.corr,
                "d_j_NR": d_j,
                "c_j_times_d_j_NR": product,
                "ocean_prior_mean": ocean_mean,
                "ocean_prior_std": ocean_std,
                "ocean_NR": context.ocean_nr,
                "e_o": e_ocean,
                "alignment": alignment,
                # --- provenance and diagnostics ---
                "filtered_index": context.filtered_index,
                "lag_tag": series.tags[lag_index],
                # The coordinate row that this lag's Hx was actually built on.
                # It is NOT guaranteed to equal the center-time position, so it
                # is reported per lag instead of assumed fixed.
                "obs_lat_at_lag": series.obs_lat_at_lag[lag_index],
                "obs_lon_at_lag": series.obs_lon_at_lag[lag_index],
                "obs_displacement_from_center_km": series.obs_displacement_km[lag_index],
                "alignment_flag": classify_alignment(
                    product, e_ocean, moments.corr, ocean_std, config.near_zero_frac
                ),
                "n_members_used": moments.n_used,
                "degenerate_ensemble": moments.degenerate,
                "state_sample_mode": context.ocean_sample.mode,
                # Taken from the SampleResult, never from the requested point:
                # in nearest mode the two differ.
                "state_sample_lat": context.ocean_sample.sample_lat,
                "state_sample_lon": context.ocean_sample.sample_lon,
                "state_sample_distance_km": context.ocean_sample.distance_km,
            }
        )

    window_rows = build_window_rows(
        config, context, series, ocean_mean, ocean_std, e_ocean, d_by_lag
    )
    return lag_rows, window_rows


def build_window_rows(
    config: Config,
    context: ObservationContext,
    series: LagSeries,
    ocean_mean: float,
    ocean_std: float,
    e_ocean: float,
    d_by_lag: list[float],
) -> list[dict]:
    """Equal-weight prefix windows, evaluated as Cov(T_o, mean_j Hx) * sum_j w_j d_j.

    The covariance is recomputed from the WINDOW-MEAN Hx ensemble, not summed
    from the per-lag covariances.  ``sum_j w_j c_j d_j`` is emitted alongside
    under a name that says not to use it, so the discrepancy stays visible.
    """
    ordered = sorted(range(len(series.lag_hours)), key=lambda index: series.lag_hours[index])
    rows: list[dict] = []

    for size in range(1, len(ordered) + 1):
        chosen = ordered[:size]
        weight = 1.0 / size
        hx_window = np.mean(series.hx[chosen, :], axis=0)
        # The window innovation is the weighted mean of the per-lag innovations,
        # NOT the weighted mean of the raw NR brightness temperatures.
        d_window = float(np.sum([d_by_lag[index] for index in chosen]) * weight)

        moments = pair_moments(context.ocean_prior, hx_window)
        product = moments.cov * d_window
        alignment = product * e_ocean

        naive = float(
            np.sum([weight * (pair_moments(context.ocean_prior, series.hx[index]).cov
                              * d_by_lag[index])
                    for index in chosen])
        )

        rows.append(
            {
                "obs_id": context.label,
                "raw_index": context.raw_index,
                "lat": context.obs_lat,
                "lon": context.obs_lon,
                "state_lat": context.state_lat,
                "state_lon": context.state_lon,
                "window_label": "[" + ",".join(str(series.lag_hours[index]) for index in chosen) + "]",
                "window_size": size,
                "included_lag_hours": ",".join(str(series.lag_hours[index]) for index in chosen),
                "window_weight": weight,
                "hx_window_mean": mean_over_members(hx_window),
                "hx_window_std": moments.std_y,
                "c_w": moments.cov,
                "r_w": moments.corr,
                "d_w_NR": d_window,
                "c_w_times_d_w_NR": product,
                "sum_j_wj_cj_dj_NAIVE_do_not_use": naive,
                "ocean_prior_mean": ocean_mean,
                "ocean_prior_std": ocean_std,
                "ocean_NR": context.ocean_nr,
                "e_o": e_ocean,
                "alignment": alignment,
                "alignment_flag": classify_alignment(
                    product, e_ocean, moments.corr, ocean_std, config.near_zero_frac
                ),
                "naive_minus_product": naive - product,
                "n_members_used": moments.n_used,
                "degenerate_ensemble": moments.degenerate,
                "state_sample_mode": context.ocean_sample.mode,
                "state_sample_lat": context.ocean_sample.sample_lat,
                "state_sample_lon": context.ocean_sample.sample_lon,
                "state_sample_distance_km": context.ocean_sample.distance_km,
            }
        )
    return rows


def compute_noise_free_averages(
    config: Config,
    context: ObservationContext,
    series: LagSeries,
    assimilated_lag_hours: set[int],
) -> dict:
    """The assimilated-window innovations, kept separate from the per-lag ones.

    The averaging runs over the ACTUAL assimilation window -- the lag times
    recorded in ``LACC_times.txt`` -- and nothing else.  Extra diagnostic lags
    (9 h, 12 h) lie OUTSIDE that window; folding them in would silently
    redefine the quantity the assimilation really saw::

        Hx_L(m)    = mean_j_assimilated Hx_mj
        y_L_NR     = mean_j_assimilated y_j_NR
        d_L_NR     = y_L_NR - mean_m(Hx_L)          (NOISE-FREE diagnostic)
        d_L_actual = y_L_actual - mean_m(Hx_L)
        noise_L    = y_L_actual - y_L_NR

    If any assimilated lag is not in the diagnosed series the window quantities
    are reported ``unavailable`` rather than approximated from the diagnostic
    subset -- a partial window is a different number, not a noisier one.

    ``d_L_NR`` uses the noise-free lagged NR brightness temperatures, so it is a
    NOISE-FREE diagnostic.  ``d_L_actual`` uses the observation that was
    actually assimilated and ``noise_L`` is the difference between them.  They
    are deliberately not collapsed into one number: quoting ``d_L_NR`` as "the
    innovation" would understate the noise the assimilation really saw, and the
    per-lag directories carry no ``*_withpert`` file, so the per-time noise
    cannot be recovered from the final noisy average either.
    """
    tag_index = {hours: index for index, hours in enumerate(series.lag_hours)}
    window = sorted(assimilated_lag_hours)
    missing = [hours for hours in window if hours not in tag_index]

    hx_mean_by_lag = np.asarray(
        [mean_over_members(series.hx[index]) for index in range(len(series.lag_hours))],
        dtype=float,
    )

    result = {
        "assimilated_lag_hours": window,
        "diagnosed_lag_hours": list(series.lag_hours),
        "excluded_extra_lag_hours": sorted(set(series.lag_hours) - set(window)),
        "available": False,
        "unavailable_reason": None,
        "hx_mean_per_lag": hx_mean_by_lag.tolist(),
        "hx_mean_over_members_and_lags": None,
        "y_NR_mean_over_lags": None,
        "d_L_NR": None,
        "y_L_actual": None,
        "d_L_actual": None,
        "noise_L": None,
        "single_obs_path": None,
        "single_obs_block_number": None,
        "single_obs_lat": None,
        "single_obs_lon": None,
    }

    if context.single_obs is not None:
        result["single_obs_path"] = str(context.single_obs.path)
        result["single_obs_block_number"] = context.single_obs.obs_block_number
        result["single_obs_lat"] = context.single_obs.lat
        result["single_obs_lon"] = context.single_obs.lon
        if context.single_obs.obs_value is not None:
            result["y_L_actual"] = float(context.single_obs.obs_value)

    if missing:
        result["unavailable_reason"] = (
            f"the assimilated window {window} is not fully diagnosed "
            f"(missing lag(s) {missing}); the real-window quantities are "
            f"reported unavailable rather than approximated from the "
            f"diagnostic subset"
        )
        return result

    indices = [tag_index[hours] for hours in window]
    hx_window_mean = mean_over_members(np.mean(series.hx[indices, :], axis=0))
    y_nr_mean = float(np.mean(series.y_nr[indices]))

    result["available"] = True
    result["hx_mean_over_members_and_lags"] = hx_window_mean
    result["y_NR_mean_over_lags"] = y_nr_mean
    result["d_L_NR"] = y_nr_mean - hx_window_mean
    if result["y_L_actual"] is not None:
        result["d_L_actual"] = result["y_L_actual"] - hx_window_mean
        result["noise_L"] = result["y_L_actual"] - y_nr_mean
    return result


# =============================================================================
# Cross-checks
# =============================================================================
def check_external_fo(
    context: ObservationContext,
    series: LagSeries,
    assimilated_lag_hours: set[int],
) -> dict:
    """Compare the ASSIMILATED-window Hx against the old obs_seq external_FO.

    The window is the real assimilation window from ``LACC_times.txt``, not a
    hard-coded ``[0, 3, 6]``: the old obs_seq was averaged over exactly those
    lag times, so comparing against any other set -- a diagnostic subset, or a
    subset plus extra lags -- is not a like-for-like check.

    The stored array covers the whole ensemble, so the comparison uses the
    entries belonging to ``config.member_numbers()`` (member ``m`` -> entry
    ``m - 1``), paired with the same members' Hx.  Pairing by position in the
    two arrays would compare different members whenever the run starts above
    member 1.
    """
    if context.single_obs is None:
        return {
            "status": "unavailable",
            "reason": "no single-observation obs_seq file for this observation",
        }
    if context.external_fo is None:
        return {
            "status": "unavailable",
            "reason": (
                context.single_obs.external_fo_unavailable_reason
                or "the single-obs file carries no external_FO block"
            ),
        }

    tag_index = {hours: index for index, hours in enumerate(series.lag_hours)}
    window = sorted(assimilated_lag_hours)
    missing = [hours for hours in window if hours not in tag_index]
    if missing:
        return {
            "status": "unavailable",
            "reason": (
                f"the assimilated window {window} is not fully diagnosed "
                f"(missing lag(s) {missing}); the diagnostic subset is NOT "
                f"substituted for the real assimilation window"
            ),
            "assimilated_lag_hours": window,
        }

    hx_window = np.mean(series.hx[[tag_index[hours] for hours in window], :], axis=0)
    reference = np.asarray(context.external_fo.selected_values, dtype=float)
    member_numbers = list(context.external_fo.member_numbers)
    if reference.size != hx_window.size:
        return {
            "status": "failed",
            "reason": (
                f"external_FO was reduced to {reference.size} member value(s) "
                f"{member_numbers}, but Hx carries {hx_window.size}"
            ),
            "assimilated_lag_hours": window,
        }
    difference = np.abs(hx_window - reference)
    if not np.all(np.isfinite(difference)):
        return {
            "status": "failed",
            "reason": "the Hx / external_FO difference contains non-finite values",
            "assimilated_lag_hours": window,
        }
    max_difference = float(np.max(difference))
    return {
        "status": "passed" if max_difference <= REFERENCE_EXTERNAL_FO_TOL_K else "failed",
        "max_abs_difference_K": max_difference,
        "tolerance_K": REFERENCE_EXTERNAL_FO_TOL_K,
        # Index into the COMPARED members, then mapped back to the DART member
        # number so the report names the member, not a position in a subset.
        "worst_member": member_numbers[int(np.argmax(difference))],
        "n_members": int(reference.size),
        "compared_member_numbers": member_numbers,
        "selected_indices_0based": list(context.external_fo.selected_indices),
        "declared_ensemble_size": context.external_fo.declared_member_count,
        "assimilated_lag_hours": window,
        "note": (
            "The residual is four-decimal rounding in the stored files; it is "
            "reported rather than assumed.  The stored external_FO covers the "
            "whole ensemble, so the compared entries are the configured members "
            "picked by member number, not the first N values."
        ),
    }


def check_reference_values(
    config: Config, context: ObservationContext, series: LagSeries, averages: dict
) -> dict:
    """Compare against the numbers verified by hand, reporting rather than asserting.

    A failure here means the inputs are NOT the ones the reference numbers came
    from, so the script refuses to claim it reproduced the old experiment.
    """
    if not config.check_reference_values:
        return {"status": "skipped"}

    report: dict = {"status": "passed", "checks": []}

    expected_innovation = REFERENCE_NOISE_FREE_INNOVATION_K.get(context.label, {})
    for lag_index, hours in enumerate(series.lag_hours):
        if hours not in expected_innovation:
            continue
        expected = expected_innovation[hours]
        actual = float(series.y_nr[lag_index] - mean_over_members(series.hx[lag_index]))
        difference = abs(actual - expected)
        ok = difference <= REFERENCE_TOLERANCE_K
        report["checks"].append(
            {
                "name": f"innovation_lag_{hours}h",
                "expected_K": expected,
                "computed_K": actual,
                "abs_difference_K": difference,
                "tolerance_K": REFERENCE_TOLERANCE_K,
                "passed": ok,
            }
        )
        if not ok:
            report["status"] = "failed"

    expected_value = REFERENCE_LACC_OBS_VALUE_K.get(context.label)
    if expected_value is not None and averages.get("y_L_actual") is not None:
        actual = float(averages["y_L_actual"])
        difference = abs(actual - expected_value)
        ok = difference <= REFERENCE_TOLERANCE_K
        report["checks"].append(
            {
                "name": "lacc_obs_value",
                "expected_K": expected_value,
                "computed_K": actual,
                "abs_difference_K": difference,
                "tolerance_K": REFERENCE_TOLERANCE_K,
                "passed": ok,
            }
        )
        if not ok:
            report["status"] = "failed"
    return report


def validate_background_consistency(
    config: Config,
    grids: list[GridField],
    members: list[int],
    nr_grid: GridField,
) -> dict:
    """Hard gate on the valid times and on the common member grid.

    Raises ``ConsistencyError`` when

      * any background member's ``Times`` is not exactly ``config.center_time``
        (or is absent, while ``require_center_time_match`` is set);
      * the NR file's ``Times`` is not ``config.center_time``;
      * two members disagree on ``XLAT``, ``XLONG`` or the water mask.

    "Every member agrees with every other member" is NOT the same as "they are
    the center time": a directory holding one wrong time would satisfy the
    former, so the expected value is compared explicitly.  Longitude is checked
    alongside latitude -- a run whose members differ only in longitude is just
    as unusable, and a latitude-only test would pass it.
    """
    problems: list[str] = []

    def check_time(label: str, path: Path, value: str | None) -> None:
        if value is None:
            if config.require_center_time_match:
                problems.append(
                    f"{label}: {path} has no Times variable, so its valid time "
                    f"could not be confirmed against the expected center time "
                    f"{config.center_time}"
                )
            return
        if value != config.center_time:
            problems.append(
                f"{label}: {path} has Times={value}, expected {config.center_time}"
            )

    for member, grid in zip(members, grids):
        check_time(f"background member {member}", grid.path, grid.time_string)
    check_time("NR", nr_grid.path, nr_grid.time_string)

    reference = grids[0]
    reference_member = members[0]
    for member, grid in zip(members[1:], grids[1:]):
        if grid.lat.shape != reference.lat.shape or not np.array_equal(grid.lat, reference.lat):
            problems.append(
                f"background member {member}: {config.lat_var} grid differs from "
                f"member {reference_member} (shape {grid.lat.shape} vs "
                f"{reference.lat.shape})"
            )
        if grid.lon.shape != reference.lon.shape or not np.array_equal(grid.lon, reference.lon):
            problems.append(
                f"background member {member}: {config.lon_var} grid differs from "
                f"member {reference_member} (shape {grid.lon.shape} vs "
                f"{reference.lon.shape})"
            )
        if grid.water.shape != reference.water.shape or not np.array_equal(grid.water, reference.water):
            problems.append(
                f"background member {member}: water mask from {grid.water_source} "
                f"differs from member {reference_member}"
            )

    if problems:
        raise ConsistencyError(
            "background / NR consistency check failed:\n  - " + "\n  - ".join(problems)
        )

    times = [grid.time_string for grid in grids]
    distinct_times = sorted({value for value in times if value})
    return {
        "status": "passed",
        "n_members": len(grids),
        "expected_center_time": config.center_time,
        "member_valid_times": {
            int(member): grid.time_string for member, grid in zip(members, grids)
        },
        "nr_valid_time": nr_grid.time_string,
        "grid_latitude_identical": True,
        "grid_longitude_identical": True,
        "water_mask_identical": True,
        "water_mask_source": reference.water_source,
        "require_center_time_match": config.require_center_time_match,
        # Keys kept from the earlier report shape so existing consumers of the
        # JSON do not break.  They are now always the "everything agreed"
        # values, because anything else raises above instead of being reported.
        "grid_identical": True,
        "distinct_valid_times": distinct_times,
        "single_valid_time": len(distinct_times) <= 1,
        "problems": [],
        "note": (
            "Times is read and compared against the configured CENTER time, "
            "which is stricter than checking that the members agree with each "
            "other; the directory name alone guarantees nothing.  A mismatch "
            "raises instead of being reported here."
        ),
    }


def check_single_obs_location(
    config: Config, record: ObsSeqRecord, obs_lat: float, obs_lon: float
) -> dict:
    """Confirm the old single-obs file really describes the selected Hx pixel.

    ``loc3d`` stores RADIANS and is converted to degrees by the parser before it
    reaches here.  A mismatch means this obs_seq is not this observation, so its
    value must not be used as "the observation that was assimilated" -- even
    when its external_FO happens to line up, which it can do whenever the crop
    picked a neighbouring block.

    Every input to that decision is checked for finiteness first.  A NaN
    coordinate would make the great-circle distance NaN, and ``NaN > tol`` is
    False, so an unchecked comparison would wave a locationless file straight
    through and report an innovation computed from it.  Latitudes must lie in
    [-90, 90] and longitudes are compared modulo 360 deg (see
    :func:`normalize_longitude_deg`); the tolerance must be a finite,
    non-negative number of kilometres.
    """
    tolerance = float(config.single_obs_coord_tol_km)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ConsistencyError(
            f"single_obs_coord_tol_km={config.single_obs_coord_tol_km!r} must be a "
            f"finite, non-negative number of kilometres; it cannot decide whether "
            f"{record.path} describes the selected pixel."
        )

    if record.lat is None or record.lon is None:
        raise ConsistencyError(
            f"{record.path}: no usable loc3d block, so the observation location "
            f"could not be checked against the selected profile row "
            f"({obs_lat:.4f}, {obs_lon:.4f})"
        )
    if not (math.isfinite(record.lat) and math.isfinite(record.lon)):
        raise ConsistencyError(
            f"{record.path}: loc3d is ({record.lat!r}, {record.lon!r}) in degrees "
            f"after the radians->degrees conversion.  A non-finite location cannot "
            f"be checked against the selected profile row and must not be accepted "
            f"as the assimilated observation."
        )
    if not (math.isfinite(obs_lat) and math.isfinite(obs_lon)):
        raise ConsistencyError(
            f"{record.path}: the reference profile coordinates are "
            f"({obs_lat!r}, {obs_lon!r}); refusing to compare against a non-finite "
            f"reference."
        )
    if not -90.0 <= record.lat <= 90.0:
        raise ConsistencyError(
            f"{record.path}: loc3d latitude {record.lat!r} deg is outside [-90, 90].  "
            f"The value is read as radians and converted, so a latitude outside that "
            f"range means the stored units are not radians and the resulting offset "
            f"would be meaningless."
        )
    if not -90.0 <= obs_lat <= 90.0:
        raise ConsistencyError(
            f"{record.path}: the reference profile latitude {obs_lat!r} deg is "
            f"outside [-90, 90]"
        )
    if not -360.0 <= record.lon <= 360.0:
        raise ConsistencyError(
            f"{record.path}: loc3d longitude {record.lon!r} deg is outside "
            f"[-360, 360].  The value is read as radians and converted, so a "
            f"longitude outside that range means the stored units are not radians."
        )
    if not -360.0 <= obs_lon <= 360.0:
        raise ConsistencyError(
            f"{record.path}: the reference profile longitude {obs_lon!r} deg is "
            f"outside [-360, 360]"
        )
    if record.obs_value is not None and not math.isfinite(record.obs_value):
        raise ConsistencyError(
            f"{record.path}: the OBS block value is {record.obs_value!r}, which is "
            f"not finite.  The actual-innovation quantities (d_L_actual, noise_L) "
            f"are built from it, so a non-finite value would be reported as if it "
            f"were a measurement."
        )

    record_lon = normalize_longitude_deg(record.lon)
    profile_lon = normalize_longitude_deg(obs_lon)
    distance = float(
        haversine_km(
            np.array([record.lat]), np.array([record_lon]), obs_lat, profile_lon
        )[0]
    )
    if not math.isfinite(distance):
        raise ConsistencyError(
            f"{record.path}: the great-circle distance between loc3d "
            f"({record.lat:.4f}, {record_lon:.4f}) and the selected profile row "
            f"({obs_lat:.4f}, {profile_lon:.4f}) is not finite, so it cannot be "
            f"compared against single_obs_coord_tol_km={tolerance:g} km."
        )
    if distance > tolerance:
        raise ConsistencyError(
            f"{record.path}: loc3d says ({record.lat:.4f}, {record_lon:.4f}) but the "
            f"selected profile row is ({obs_lat:.4f}, {profile_lon:.4f}); the offset "
            f"is {distance:.3f} km, beyond single_obs_coord_tol_km={tolerance:g} km.  "
            f"This observation file does not describe the selected Hx pixel; "
            f"refusing to use its value."
        )
    return {
        "single_obs_lat": record.lat,
        "single_obs_lon": record.lon,
        "single_obs_lon_normalised_0_360": record_lon,
        "profile_lat": obs_lat,
        "profile_lon": obs_lon,
        "profile_lon_normalised_0_360": profile_lon,
        "distance_km": distance,
        "tolerance_km": tolerance,
        "obs_value": record.obs_value,
        "loc3d_units": "radians in the file, converted to degrees for comparison",
        "longitude_convention": (
            "longitudes are compared modulo 360 deg, so a 0..360 file and a "
            "180..180 file describing the same point agree"
        ),
    }


def check_withpert_availability(config: Config, schedule: list[tuple[int, str]]) -> dict:
    """Record which per-lag noisy files exist.  None are used by this diagnostic."""
    per_lag = {
        tag: noisy_bt_path(config, tag).is_file() for _, tag in schedule
    }
    unified = unified_noisy_bt_path(config)
    return {
        "per_lag_withpert": {tag: present for tag, present in per_lag.items()},
        "per_lag_withpert_present": any(per_lag.values()),
        "unified_withpert": str(unified),
        "unified_withpert_present": unified.is_file(),
        "note": (
            "The per-lag directories carry no *_withpert file, so only the "
            "noise-free per-lag innovation d_j_NR can be formed.  The per-time "
            "noise cannot be recovered from the final noisy LACC average, and "
            "this script does not attempt it."
        ),
    }


# =============================================================================
# Output writers
# =============================================================================
def write_csv(path: Path, rows: list[dict]) -> Path:
    if not rows:
        raise DiagnosticError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_safe)
    return path


def _json_safe(value):
    """JSON encoder hook: numpy scalars, Paths and non-finite floats -> plain JSON.

    NaN and +/-inf are written as ``null`` rather than the non-standard ``NaN``
    token, so the JSON stays parseable by strict readers.
    """
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (np.bool_, bool)):  # before int: bool is an int subclass
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, int):
        return value
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    raise TypeError(f"cannot serialise {type(value)!r}")


# =============================================================================
# Figure
# =============================================================================
FIGURE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 110,
}

COLOR_TOWARD = "#1F7A4D"
COLOR_AWAY = "#B03A2E"
COLOR_NEUTRAL = "#7B8A93"
COLOR_REQUIRED = "#2464AA"

CLEAR_SKY_GLYPH = {"clear": "", "not_clear": "▲", "unknown": "×"}


def _mark_lag_status(axis, x_values, y_values, flags) -> None:
    """Stamp non-clear / unknown lags so a flagged time can never be read as clear.

    Each glyph rides on its own bar, so a flagged lag stays identifiable even
    when the bars have wildly different magnitudes.
    """
    for x, y, flag in zip(x_values, y_values, flags):
        glyph = CLEAR_SKY_GLYPH.get(flag, "")
        if not glyph or not math.isfinite(y):
            continue
        axis.annotate(
            glyph,
            xy=(x, y),
            xytext=(0, 6 if y >= 0 else -15),
            textcoords="offset points",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=12,
            color=COLOR_NEUTRAL,
            weight="bold",
            annotation_clip=False,
        )


def pad_ylim(axis, fraction: float = 0.12) -> None:
    """Leave room for the value labels so none of them lands on the axis spine."""
    low, high = axis.get_ylim()
    span = high - low if high > low else 1.0
    axis.set_ylim(low - fraction * span, high + fraction * span)


def make_figure(
    config: Config,
    context: ObservationContext,
    series: LagSeries,
    lag_rows: list[dict],
    window_rows: list[dict],
    averages: dict,
    output_path: Path,
) -> Path:
    lag_x = np.asarray(series.lag_hours, dtype=float)
    c_values = np.asarray([row["c_j"] for row in lag_rows], dtype=float)
    d_values = np.asarray([row["d_j_NR"] for row in lag_rows], dtype=float)
    product_values = np.asarray([row["c_j_times_d_j_NR"] for row in lag_rows], dtype=float)
    e_ocean = float(lag_rows[0]["e_o"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), constrained_layout=True)

        # --- panel A: c_j -------------------------------------------------
        ax = axes[0, 0]
        ax.bar(lag_x, c_values, width=0.5, color=COLOR_REQUIRED, alpha=0.85)
        ax.axhline(0.0, color="0.3", lw=0.9, ls="--")
        ax.set_title(r"A  $c_j=\mathrm{Cov}_m[T_o(t,s),\,H_x(t-\tau_j,k)]$  (ddof = 1)")
        ax.set_xlabel("Lag of Hx (h)")
        ax.set_ylabel(r"$c_j$  (K$^2$)")
        ax.set_xticks(lag_x)
        ax.grid(True, alpha=0.25, axis="y")
        for x, value in zip(lag_x, c_values):
            if not math.isfinite(value):
                continue
            ax.annotate(
                f"{value:.3g}",
                xy=(x, value),
                xytext=(0, 5 if value >= 0 else -16),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=COLOR_NEUTRAL,
            )
        pad_ylim(ax)
        _mark_lag_status(ax, lag_x, c_values, series.clear_sky_flags)

        # --- panel B: d_j_NR ----------------------------------------------
        ax = axes[0, 1]
        colors = [COLOR_AWAY if value < 0 else "#2464AA" for value in np.nan_to_num(d_values)]
        ax.bar(lag_x, d_values, width=0.5, color=colors, alpha=0.85)
        ax.axhline(0.0, color="0.3", lw=0.9, ls="--")
        ax.set_title(r"B  $d_{j,\mathrm{NR}}=y_{j,\mathrm{NR}}(k)-\overline{H_x}_{m}(t-\tau_j,k)$")
        ax.set_xlabel("Lag of Hx (h)")
        ax.set_ylabel(r"$d_{j,\mathrm{NR}}$  (K)")
        ax.set_xticks(lag_x)
        ax.grid(True, alpha=0.25, axis="y")
        for x, value in zip(lag_x, d_values):
            if not math.isfinite(value):
                continue
            ax.annotate(
                f"{value:+.3f}",
                xy=(x, value),
                xytext=(0, 5 if value >= 0 else -16),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=COLOR_NEUTRAL,
            )
        pad_ylim(ax)

        # --- panel C: the direction product -------------------------------
        # Bar HEIGHT and sign show c_j*d_j, i.e. the direction of the ocean
        # increment.  Bar COLOUR shows alignment = product * e_o, i.e. whether
        # that increment moves toward or away from the NR.  The two disagree
        # whenever e_o < 0, which is exactly the case a product-only colouring
        # gets backwards.
        ax = axes[1, 0]
        alignment_values = np.asarray(
            [row["alignment"] for row in lag_rows], dtype=float
        )
        product_colors = [
            COLOR_TOWARD if value > 0 else COLOR_AWAY if value < 0 else COLOR_NEUTRAL
            for value in np.nan_to_num(alignment_values)
        ]
        ax.bar(lag_x, product_values, width=0.5, color=product_colors, alpha=0.85)
        ax.axhline(0.0, color="0.3", lw=0.9, ls="--")
        ax.set_title(
            r"C  height $c_j\cdot d_{j,\mathrm{NR}}$  /  colour "
            r"$c_j d_{j,\mathrm{NR}}\cdot e_o$"
        )
        ax.set_xlabel("Lag of Hx (h)")
        ax.set_ylabel(r"$c_j\cdot d_{j,\mathrm{NR}}$")
        ax.set_xticks(lag_x)
        ax.grid(True, alpha=0.25, axis="y")

        # Headroom so the required-direction box never lands on a bar or the title.
        low, high = ax.get_ylim()
        span = high - low if high > low else 1.0
        ax.set_ylim(low - 0.10 * span, high + 0.42 * span)

        if math.isfinite(e_ocean):
            ax.text(
                0.02,
                0.97,
                f"NR needs $e_o$ = {e_ocean:+.4f} K\n"
                f"({'warmer' if e_ocean > 0 else 'cooler'} than the prior mean)\n"
                f"-> toward NR means $c_j d_j$ sign "
                f"{'+' if e_ocean > 0 else '−'}\n"
                f"bar sign = increment, colour = toward/away NR",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color=COLOR_REQUIRED,
                bbox=dict(
                    boxstyle="round,pad=0.32",
                    facecolor="white",
                    edgecolor=COLOR_REQUIRED,
                    linewidth=0.8,
                    alpha=0.92,
                ),
            )
        _mark_lag_status(ax, lag_x, product_values, series.clear_sky_flags)

        # --- panel D: window direction ------------------------------------
        ax = axes[1, 1]
        window_x = np.arange(len(window_rows), dtype=float)
        window_product = np.asarray([row["c_w_times_d_w_NR"] for row in window_rows], dtype=float)
        window_alignment = np.asarray([row["alignment"] for row in window_rows], dtype=float)
        window_colors = [
            COLOR_TOWARD if value > 0 else COLOR_AWAY if value < 0 else COLOR_NEUTRAL
            for value in np.nan_to_num(window_alignment)
        ]
        ax.bar(window_x, window_product, width=0.55, color=window_colors, alpha=0.85)
        ax.axhline(0.0, color="0.3", lw=0.9, ls="--")
        ax.set_title(r"D  Window $c_w\cdot d_{w,\mathrm{NR}}$   (equal-weight prefix windows)")
        ax.set_xlabel("Window")
        ax.set_ylabel(r"$c_w\cdot d_{w,\mathrm{NR}}$")
        ax.set_xticks(window_x)
        ax.set_xticklabels([row["window_label"] for row in window_rows])
        ax.grid(True, alpha=0.25, axis="y")
        for x, row in zip(window_x, window_rows):
            ax.annotate(
                row["alignment_flag"],
                xy=(x, 0.0),
                xytext=(0, -22),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=COLOR_NEUTRAL,
            )

        # --- shared header -------------------------------------------------
        flag_summary = ", ".join(
            f"{hours}h={row['alignment_flag']}" for hours, row in zip(series.lag_hours, lag_rows)
        )
        clear_summary = ", ".join(
            f"{hours}h={flag}" for hours, flag in zip(series.lag_hours, series.clear_sky_flags)
        )
        fig.suptitle(
            f"LACC lag covariance vs. innovation direction -- observation "
            f"{context.label} (raw row {context.raw_index})\n"
            f"obs ({context.obs_lat:.4f}, {context.obs_lon:.4f})   "
            f"state ({context.state_lat:.4f}, {context.state_lon:.4f})   "
            f"ocean prior {lag_rows[0]['ocean_prior_mean']:.4f} +/- "
            f"{lag_rows[0]['ocean_prior_std']:.4f} K, NR {context.ocean_nr:.4f} K\n"
            f"lag direction: {flag_summary}   |   clear-sky: {clear_summary}"
            f"   (▲ = not clear, × = unknown)\n"
            f"DIRECTION ONLY: a correctly signed increment can still overshoot; "
            f"this is not a skill score.",
            fontsize=9.5,
        )
        fig.savefig(output_path, dpi=config.figure_dpi, bbox_inches="tight")
        plt.close(fig)
    return output_path


# =============================================================================
# Orchestration
# =============================================================================
def run(config: Config) -> dict:
    validate_config(config)

    warnings: list[str] = []
    output_dir = Path(config.output_dir)

    # ---- LACC window -------------------------------------------------------
    times_path = config.resolved_lacc_times()
    file_center, file_lags = read_times_file(times_path)
    schedule_info = resolve_lag_schedule(config, file_center, file_lags)
    schedule: list[tuple[int, str]] = schedule_info["schedule"]

    discovered = detect_lag_directories(config.resolved_obs_bt_dir())
    undiagnosed = [
        tag for tag in discovered if tag not in {tag for _, tag in schedule}
    ]
    if undiagnosed:
        warnings.append(
            f"{len(undiagnosed)} lag directory/directories under "
            f"{config.resolved_obs_bt_dir()} are NOT diagnosed: "
            f"{', '.join(undiagnosed)}.  Lags outside the assimilated window are "
            f"never added automatically."
        )

    # The CENTER time is defined by config.center_time -- whose tag the LACC
    # times file was just checked against -- NOT by whichever lag happens to
    # come first in the requested diagnostic set.  It fixes the observation
    # position, the default state point and the reference for the coordinate
    # displacement.  ``lag_hours = (3, 6)`` is a legitimate subset: it does not
    # move the center profile, and reading that profile does not fold the 0 h
    # lag into the diagnostics or into any window average.
    center_tag = config.center_tag()
    center_lag_hours = {
        tag: hours for hours, tag in schedule_info["assimilated_tags"].items()
    }.get(center_tag)

    assimilated_lag_hours = set(schedule_info["assimilated_lag_hours"])
    diagnosed_lag_hours = {hours for hours, _ in schedule}
    missing_window_hours = sorted(assimilated_lag_hours - diagnosed_lag_hours)
    if missing_window_hours:
        warnings.append(
            f"the assimilated window {sorted(assimilated_lag_hours)} is NOT fully "
            f"diagnosed: lag(s) {missing_window_hours} are absent from the requested "
            f"set {sorted(diagnosed_lag_hours)}.  The real-window quantities (Hx_L, "
            f"y_L_NR, d_L_NR, d_L_actual, noise_L) and the external_FO cross-check "
            f"are reported unavailable; the diagnostic subset is never substituted "
            f"for the real window."
        )

    # ---- masks and observation targets ------------------------------------
    unified_tag = config.center_tag()
    unified_mask = read_mask(config.resolved_unified_mask(), config.nobs_raw, unified_tag)
    targets = [resolve_obs_target(target, unified_mask.values, config.nobs_raw)
               for target in config.obs_targets]

    # ---- F-order numbering: enforced for EVERY file that enters a computation
    # TWO INDEPENDENT SUITES, both checked against each file's OWN 2D matrix:
    #
    #   * the per-lag NR reference column under 3obs_BT_LACC,
    #   * every member's Hx column under 4ens_BT_LACC/mem*/.
    #
    # A passing NR check says nothing about the member files: they are different
    # files in different directories, and it is the MEMBER Hx that the lagged
    # covariance is built from.  Swapping two lags' member columns while leaving
    # the matrices in place leaves the time average -- and therefore external_FO
    # -- untouched, so only the per-member check can catch it.
    #
    # The checks are cached by member and lag, so N observations read and verify
    # each file once rather than N times.
    files = HxFileCache(config)
    members = config.member_numbers()

    per_tag_flattening: dict[str, dict] = {}
    for _, tag in schedule:
        entry = files.verify_nr(tag)
        if entry.get("enforcement") == "skipped_matrix_absent":
            warnings.append(
                f"NR BT {tag}: the 2D BT matrix is absent and "
                f"require_f_order_check=False, so the F-order numbering for this "
                f"lag was NOT verified."
            )
        per_tag_flattening[tag] = entry

    unverified_member_pairs: list[tuple[int, str]] = []
    for _, tag in schedule:
        for member in members:
            entry = files.verify_member(member, tag)
            if entry.get("enforcement") == "skipped_matrix_absent":
                unverified_member_pairs.append((member, tag))
    if unverified_member_pairs:
        shown = ", ".join(
            f"mem{member:03d} {tag}" for member, tag in unverified_member_pairs[:5]
        )
        more = (
            ""
            if len(unverified_member_pairs) <= 5
            else f" (and {len(unverified_member_pairs) - 5} more)"
        )
        warnings.append(
            f"{len(unverified_member_pairs)} member Hx matrix file(s) are absent and "
            f"require_f_order_check=False, so the F-order numbering of those member "
            f"columns was NOT verified: {shown}{more}.  The NR check passing does NOT "
            f"cover this."
        )
    member_flattening = files.member_report({tag: hours for hours, tag in schedule})

    # The center-time NR column is only checked when the 0 h lag is part of the
    # diagnostic set.  When it is not, the result is reported unavailable rather
    # than borrowed from another lag -- a different lag's pass would say nothing
    # about the center file, and the per-lag entries stay where they belong.
    center_flattening = per_tag_flattening.get(center_tag)
    if center_flattening is None:
        center_flattening = {
            "label": f"NR BT {center_tag}",
            "path": str(matrix_path(config, center_tag)),
            "column_path": str(nr_bt_path(config, center_tag)),
            "available": False,
            "f_order_matches": None,
            "c_order_matches": None,
            "suite": SUITE_NR_BT,
            "enforcement": "not_computed_center_lag_not_diagnosed",
            "reason": (
                f"the center-time lag ({center_tag}"
                + (f", {center_lag_hours} h" if center_lag_hours is not None else "")
                + f") is not part of the requested diagnostic set "
                f"{sorted(diagnosed_lag_hours)}, so its NR F-order numbering was never "
                f"checked.  No other lag's result is substituted here; see "
                f"f_order_numbering_per_obs for the lags that WERE checked."
            ),
        }

    # ---- profiles ----------------------------------------------------------
    # The CENTER-time profile defines where the observation is and (unless it is
    # overridden) the ocean state point, so it is required even when the 0 h lag
    # is not diagnosed.  The per-lag profiles are only used for the per-lag
    # observation coordinates and for the displacement report.
    profiles: dict[str, ProfileCoordinates] = {}
    center_profile_path = profile_path(config, center_tag)
    if not center_profile_path.is_file():
        raise MissingInputError(
            f"the center-time profile {center_profile_path} is missing.  The "
            f"observation position and the default ocean state point belong to the "
            f"center time {config.center_time}, so this file is required even when "
            f"the diagnostic set ({sorted(diagnosed_lag_hours)} h) excludes the 0 h "
            f"lag."
        )
    profiles[center_tag] = read_profile_coordinates(
        center_profile_path, config.nobs_raw, center_tag
    )
    for _, tag in schedule:
        if tag == center_tag:
            continue  # already read above, from the same path
        profiles[tag] = read_profile_coordinates(
            profile_path(config, tag), config.nobs_raw, tag
        )

    # The displacement reference is the CENTER profile, not the first diagnosed
    # lag: with a (3, 6) subset the reference would otherwise silently become the
    # 3 h file and the reported displacements would change meaning.
    coordinate_report: dict[str, dict] = {}
    center_profile = profiles[center_tag]
    for _, tag in schedule:
        if tag == center_tag:
            continue
        comparison = compare_profile_coordinates(center_profile, profiles[tag])
        coordinate_report[f"{center_tag}->{tag}"] = comparison
        if not comparison["summary"]["same_fixed_location"]:
            warnings.append(
                f"profile coordinates for lag {tag} move relative to the center time "
                f"{center_tag} (max {comparison['summary']['max_distance_km']:.3f} km "
                f"at raw row {comparison['summary']['max_at_raw_index']}); the lagged "
                f"Hx must NOT be treated as sitting at one fixed location."
            )

    # ---- background and NR -------------------------------------------------
    background_grids: list[GridField] = []
    for member in members:
        path = background_member_path(config, member)
        if not path.is_file():
            raise MissingInputError(
                f"background member {member} is missing: {path}.  background_choice="
                f"{config.background_choice!r} was selected explicitly; this script "
                f"never falls back to a different background, because that would "
                f"silently change every number in the output."
            )
        background_grids.append(
            read_level_zero_field(
                path, config.state_var, config.state_level,
                config.lat_var, config.lon_var, config.water_mask_source,
            )
        )
    background_check = None

    nr_grid = read_level_zero_field(
        config.resolved_nr_file(), config.state_var, config.state_level,
        config.lat_var, config.lon_var, config.water_mask_source,
    )

    # ---- gate: every valid time and the common member grid -----------------
    background_check = validate_background_consistency(
        config, background_grids, members, nr_grid
    )

    # ---- diagnostics -------------------------------------------------------
    all_lag_rows: list[dict] = []
    all_window_rows: list[dict] = []
    context_reports: dict[str, dict] = {}
    f_order_checks: dict[str, dict] = {}
    internal_fo_checks: dict[str, dict] = {}
    reference_checks: dict[str, dict] = {}

    for target_info in targets:
        # ALWAYS the center-time profile: it defines the diagnosed raw row, the
        # single-obs loc3d reference and the default state point, none of which
        # may follow the first diagnosed lag.
        context = build_observation_context(
            config, target_info, center_profile, background_grids, nr_grid
        )

        # The state is always the CENTER-time ocean state; the per-lag profiles
        # enter only through the innovation and through the per-lag observation
        # coordinates recorded on each row.
        series = load_lag_series(config, context, schedule, profiles, files)

        lag_rows, window_rows = compute_lag_rows(config, context, series)
        averages = compute_noise_free_averages(
            config, context, series, assimilated_lag_hours
        )

        all_lag_rows.extend(lag_rows)
        all_window_rows.extend(window_rows)

        internal_fo_checks[context.label] = check_external_fo(
            context, series, assimilated_lag_hours
        )
        reference_checks[context.label] = check_reference_values(
            config, context, series, averages
        )

        # The per-tag NR F-order results are identical for every observation, so
        # they are computed once above and mirrored here to keep the JSON shape.
        # This entry covers the NR reference files ONLY; the member Hx files are
        # reported separately under f_order_numbering_member_hx, and a pass here
        # must never be read as covering them.
        f_order_checks[context.label] = per_tag_flattening

        context_reports[context.label] = {
            "obs_id": context.label,
            "filtered_index": context.filtered_index,
            "raw_index": context.raw_index,
            "zero_based_raw_index": context.zero_based_raw_index,
            "obs_lat": context.obs_lat,
            "obs_lon": context.obs_lon,
            "state_lat": context.state_lat,
            "state_lon": context.state_lon,
            "state_sample_mode": context.ocean_sample.mode,
            "state_sample_support_points": context.ocean_sample.n_support_points,
            "state_sample_local_window_deg": context.ocean_sample.local_window_deg,
            # The coordinates the value ACTUALLY came from, straight out of the
            # SampleResult.  In nearest mode these differ from the requested
            # point, and reporting the request instead would hide the snap.
            "state_sample_lat": context.ocean_sample.sample_lat,
            "state_sample_lon": context.ocean_sample.sample_lon,
            "state_sample_distance_km": context.ocean_sample.distance_km,
            "nr_sample_lat": context.nr_sample.sample_lat,
            "nr_sample_lon": context.nr_sample.sample_lon,
            "ocean_prior_mean": float(np.mean(context.ocean_prior)),
            "ocean_prior_std": float(np.std(context.ocean_prior, ddof=1)),
            "ocean_NR": context.ocean_nr,
            "e_o": context.ocean_nr - float(np.mean(context.ocean_prior)),
            "nr_sample_distance_km": context.nr_sample.distance_km,
            "background_files_used": context.background_files,
            "background_valid_times": context.background_times,
            "single_obs": {
                "path": str(context.single_obs.path) if context.single_obs else None,
                "block_number": context.single_obs.obs_block_number if context.single_obs else None,
                "lat": context.single_obs.lat if context.single_obs else None,
                "lon": context.single_obs.lon if context.single_obs else None,
                "obs_value": context.single_obs.obs_value if context.single_obs else None,
                "location_check": context.single_obs_location,
                # The stored external_FO covers the WHOLE ensemble; these are the
                # entries this run compared, picked by DART member number.
                "external_fo_members": {
                    "declared_ensemble_size": context.external_fo.declared_member_count,
                    "compared_member_numbers": list(context.external_fo.member_numbers),
                    "selected_indices_0based": list(context.external_fo.selected_indices),
                }
                if context.external_fo is not None
                else None,
                "external_fo_unavailable_reason": (
                    context.single_obs.external_fo_unavailable_reason
                    if context.single_obs is not None
                    else None
                ),
                "note": (
                    "The OBS block number inside the single-obs file is that "
                    "file's own numbering, not a raw 676-row index."
                ),
            },
            "lag_mean_hx": series.hx.mean(axis=1).tolist(),
            "lag_y_NR": series.y_nr.tolist(),
            "lag_clear_sky": series.clear_sky_flags,
            "lag_obs_coordinates": [
                {
                    "lag_hours": hours,
                    "lag_tag": tag,
                    "obs_lat": lat,
                    "obs_lon": lon,
                    "displacement_from_center_km": distance,
                }
                for hours, tag, lat, lon, distance in zip(
                    series.lag_hours,
                    series.tags,
                    series.obs_lat_at_lag,
                    series.obs_lon_at_lag,
                    series.obs_displacement_km,
                )
            ],
            "hx_files_read": series.hx_file_count,
            "averages": averages,
        }

        if config.write_figures:
            figure_path = output_dir / f"lacc_lag_cov_innovation_obs{context.label}_{config.center_tag()}.png"
            make_figure(
                config, context, series, lag_rows, window_rows, averages, figure_path
            )
            context_reports[context.label]["figure"] = str(figure_path)

    # ---- outputs -----------------------------------------------------------
    lag_csv = write_csv(
        output_dir / f"lacc_lag_diagnostics_{config.center_tag()}.csv", all_lag_rows
    )
    window_csv = write_csv(
        output_dir / f"lacc_window_diagnostics_{config.center_tag()}.csv", all_window_rows
    )

    coordinate_csv = None
    coordinate_rows: list[dict] = []
    for comparison in coordinate_report.values():
        coordinate_rows.extend(comparison["rows"])
    if coordinate_rows:
        coordinate_csv = write_csv(
            output_dir / f"lacc_profile_coordinate_displacement_{config.center_tag()}.csv",
            coordinate_rows,
        )

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": str(Path(__file__).resolve()),
        "purpose": (
            "Diagnose why single-observation LACC assimilation approaches the "
            "observation in BT space while pushing OM_TMP away from the NR."
        ),
        "config": {
            key: _json_safe(value)
            for key, value in {
                "project_root": config.project_root,
                "hx_root": config.resolved_hx_root(),
                "output_dir": config.output_dir,
                "center_time": config.center_time,
                "center_tag": config.center_tag(),
                "domain": config.domain,
                "sensor": config.sensor,
                "channel": config.channel,
                "member_start": config.member_start,
                "member_end": config.member_end,
                "nobs_raw": config.nobs_raw,
                "lag_hours": config.lag_hours,
                "extra_lag_hours": config.extra_lag_hours,
                "enable_extra_lag_diagnostics": config.enable_extra_lag_diagnostics,
                "background_choice": config.background_choice,
                "state_var": config.state_var,
                "state_level": config.state_level,
                "state_lat": config.state_lat,
                "state_lon": config.state_lon,
                "sample_mode": config.sample_mode,
                "sample_require_water": config.sample_require_water,
                "water_mask_source": config.water_mask_source,
                "near_zero_frac": config.near_zero_frac,
                "single_obs_coord_tol_km": config.single_obs_coord_tol_km,
                "require_center_time_match": config.require_center_time_match,
            }.items()
        },
        "resolved_paths": {
            "lacc_times_file": str(times_path),
            "unified_clear_sky_mask": str(config.resolved_unified_mask()),
            "nr_file": str(config.resolved_nr_file()),
            "nr_file_valid_time": nr_grid.time_string,
            "nr_water_mask_source": nr_grid.water_source,
            "center_profile": str(center_profile_path),
            "profiles": {tag: str(profile_path(config, tag)) for _, tag in schedule},
            "per_lag_masks": {tag: str(per_lag_mask_path(config, tag)) for _, tag in schedule},
            "single_obs_files": {
                target["label"]: str(single_obs_path(config, target["label"]))
                for target in targets
            },
            "background_choice": config.background_choice,
            "background_member_files": [
                str(background_member_path(config, member)) for member in members
            ],
            "background_member_numbers": members,
        },
        "lag_schedule": {
            "assimilated_lag_hours": schedule_info["assimilated_lag_hours"],
            "assimilated_tags": schedule_info["assimilated_tags"],
            "diagnosed_lag_hours": schedule_info["diagnosed_lag_hours"],
            "center_tag": center_tag,
            "center_lag_hours": center_lag_hours,
            "assimilated_window_fully_diagnosed": not missing_window_hours,
            "missing_assimilated_lag_hours": missing_window_hours,
            "diagnostic_set_is_a_subset_of_the_window": bool(missing_window_hours),
            "undiagnosed_lag_directories": undiagnosed,
            "extra_lag_hours_enabled": schedule_info["extra_lag_hours_enabled"],
            "extra_lag_hours_used": schedule_info["extra_lag_hours_used"],
            "windows": [list(window) for window in prefix_windows(schedule_info["diagnosed_lag_hours"])],
        },
        "checks": {
            # NR reference files, center lag: unavailable (not borrowed) when the
            # 0 h lag is outside the requested diagnostic set.
            "f_order_numbering_center": center_flattening,
            # NR reference files, every diagnosed lag, mirrored per observation.
            "f_order_numbering_per_obs": f_order_checks,
            # Member Hx files: a SEPARATE suite.  A pass in the two entries above
            # says nothing about these files, and this entry says nothing about
            # the NR files.
            "f_order_numbering_member_hx": member_flattening,
            "observation_index_mapping": {
                target["label"]: {
                    "filtered_index": target["filtered_index"],
                    "raw_index": target["raw_index"],
                    "zero_based_raw_index": target["zero_based_raw_index"],
                    "note": (
                        "filtered_index counts the clear-sky-KEPT list used by the "
                        "LACC obs_seq; raw_index counts the unfiltered 676-row file."
                    ),
                }
                for target in targets
            },
            "unified_mask": {
                "path": str(config.resolved_unified_mask()),
                "n_raw": int(unified_mask.values.size),
                "n_kept": unified_mask.n_kept,
            },
            "profile_coordinate_displacement": {
                key: value["summary"] for key, value in coordinate_report.items()
            },
            "background_consistency": background_check,
            "external_fo": internal_fo_checks,
            "reference_values": reference_checks,
            "withpert_availability": check_withpert_availability(config, schedule),
        },
        "observations": context_reports,
        "outputs": {
            "lag_csv": str(lag_csv),
            "window_csv": str(window_csv),
            "coordinate_csv": str(coordinate_csv) if coordinate_csv else None,
        },
        "warnings": warnings,
    }
    json_path = write_json(
        output_dir / f"lacc_lag_cov_innovation_summary_{config.center_tag()}.json", payload
    )
    payload["outputs"]["json"] = str(json_path)
    # Convenience read-back for the console summary and for the self-test.  It
    # is attached AFTER the JSON is written, so these keys are never part of
    # the on-disk payload contract.
    payload["_lag_rows"] = _rows_by_obs(lag_csv)
    payload["_window_rows"] = _rows_by_obs(window_csv)
    return payload


def _rows_by_obs(path: Path) -> dict[str, list[dict]]:
    """Read a CSV written by ``write_csv``, grouped by obs_id with numerics cast."""
    grouped: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict = {}
            for key, value in raw.items():
                if key == "obs_id":
                    row[key] = value  # keep the label as text, it is a grouping key
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            grouped.setdefault(str(row["obs_id"]), []).append(row)
    return grouped


def print_summary(payload: dict) -> None:
    print("=" * 78)
    print("LACC lag covariance / innovation-direction diagnostic")
    print("=" * 78)
    print(f"center time       : {payload['config']['center_time']}")
    print(f"diagnosed lags    : {payload['lag_schedule']['diagnosed_lag_hours']} h")
    print(f"assimilated lags  : {payload['lag_schedule']['assimilated_lag_hours']} h")
    if payload["lag_schedule"]["undiagnosed_lag_directories"]:
        print(
            "NOT diagnosed     : "
            + ", ".join(payload["lag_schedule"]["undiagnosed_lag_directories"])
        )
    print(f"background        : {payload['config']['background_choice']}")
    print(f"state sampling    : {payload['config']['sample_mode']}")

    # The two F-order suites are reported separately: a passing NR check must
    # never be read as covering the member Hx files.
    center_fo = payload["checks"]["f_order_numbering_center"]
    member_fo = payload["checks"]["f_order_numbering_member_hx"]
    print(
        f"F-order, NR files : center {center_fo.get('enforcement')} "
        f"({center_fo.get('f_order_matches')})"
    )
    print(
        f"F-order, member Hx: {member_fo.get('status')} -- "
        f"{member_fo.get('n_verified')}/{member_fo.get('n_member_lag_pairs')} "
        f"member/lag pairs verified"
    )
    if member_fo.get("n_unverified"):
        print(
            f"                    {member_fo['n_unverified']} pair(s) NOT verified; "
            f"see checks.f_order_numbering_member_hx.unverified in the JSON"
        )
    print()
    for label, report in payload["observations"].items():
        print(f"--- observation {label} (raw row {report['raw_index']}) ---")
        print(
            f"    ocean prior {report['ocean_prior_mean']:.4f} +/- "
            f"{report['ocean_prior_std']:.4f} K, NR {report['ocean_NR']:.4f} K, "
            f"e_o = {report['e_o']:+.4f} K"
        )
        averages = report["averages"]
        window_text = ",".join(str(hours) for hours in averages["assimilated_lag_hours"])
        if averages["available"]:
            print(
                f"    d_L_NR     = {averages['d_L_NR']:+.4f} K   "
                f"(noise-free, assimilated window [{window_text}])"
            )
            if averages["d_L_actual"] is not None:
                print(f"    d_L_actual = {averages['d_L_actual']:+.4f} K")
                print(f"    noise_L    = {averages['noise_L']:+.4f} K")
        else:
            print(
                f"    assimilated-window quantities UNAVAILABLE "
                f"(window [{window_text}]): {averages['unavailable_reason']}"
            )
        excluded = averages["excluded_extra_lag_hours"]
        if excluded:
            print(
                f"    excluded from the assimilated window: "
                f"{','.join(str(hours) for hours in excluded)} h "
                f"(diagnostic-only lags)"
            )
        for row in payload["_lag_rows"][label]:
            print(
                f"    lag {int(row['lag_hours'])}h [{row['clear_sky_flag']:>9s}] "
                f"c_j={row['c_j']:+.5g}  d_j={row['d_j_NR']:+.4f}  "
                f"c_j*d_j={row['c_j_times_d_j_NR']:+.5g}  -> {row['alignment_flag']}"
            )
        for row in payload["_window_rows"][label]:
            print(
                f"    window {row['window_label']:<9s} "
                f"c_w={row['c_w']:+.5g}  d_w={row['d_w_NR']:+.4f}  "
                f"c_w*d_w={row['c_w_times_d_w_NR']:+.5g}  -> {row['alignment_flag']}"
            )
        external = payload["checks"]["external_fo"].get(label, {})
        if external.get("status") != "unavailable":
            print(
                f"    external_FO check : {external.get('status')} "
                f"(max |diff| = {external.get('max_abs_difference_K', float('nan')):.3e} K, "
                f"tol {REFERENCE_EXTERNAL_FO_TOL_K:g} K)"
            )
        reference = payload["checks"]["reference_values"].get(label, {})
        for check in reference.get("checks", []):
            print(
                f"    {check['name']:<20s}: expected {check['expected_K']:+.4f}, "
                f"computed {check['computed_K']:+.4f} "
                f"({'OK' if check['passed'] else 'MISMATCH'})"
            )
        print()
    for warning in payload["warnings"]:
        print(f"WARNING: {warning}")
    print()
    print("outputs:")
    for name, path in payload["outputs"].items():
        print(f"  {name}: {path}")


# =============================================================================
# Synthetic self-test
# =============================================================================
# Fixture values shared by the writer and the checker, so every expected number
# in the self-test is derived from exactly what was written to disk.
#
# The geometry deliberately matches the REAL experiment -- 676 raw rows, a
# 26x26 BT matrix, 186 clear-sky-kept rows, and observations 66 / 98 sitting at
# raw rows 397 / 495 with their verified coordinates.  That way the self-test
# exercises the documented index mapping instead of a toy that happens to work.
SYNTHETIC_SIDE = 26
SYNTHETIC_NOBS = SYNTHETIC_SIDE * SYNTHETIC_SIDE  # 676
SYNTHETIC_MEMBERS = 4
SYNTHETIC_TAGS = ("10_00_00", "09_21_00", "09_18_00")
SYNTHETIC_N_KEPT = 186
SYNTHETIC_OBS_ANCHORS = ((66, 397), (98, 495))
SYNTHETIC_OBS_LATLON = {
    "66": (14.2751, 147.6832),
    "98": (14.5495, 147.2686),
}
SYNTHETIC_OCEAN_BY_MEMBER = np.array([0.10, -0.30, 0.55, 0.90])
SYNTHETIC_HX_BY_LAG_MEMBER = np.array(
    [
        [250.0, 250.4, 249.8, 250.9],
        [251.0, 250.2, 251.3, 249.9],
        [249.5, 250.8, 249.9, 250.4],
    ]
)
SYNTHETIC_NR_BT_BY_LAG = np.array([250.5, 251.4, 250.9])


def build_synthetic_mask(
    nobs: int = SYNTHETIC_NOBS,
    n_kept: int = SYNTHETIC_N_KEPT,
    anchors: tuple[tuple[int, int], ...] = SYNTHETIC_OBS_ANCHORS,
) -> np.ndarray:
    """A 0/1 mask with ``n_kept`` kept rows whose anchor ranks land exactly.

    Reproduces the documented layout: 186 kept of 676, the 66th kept row at raw
    row 397 and the 98th at raw row 495.
    """
    (rank_a, row_a), (rank_b, row_b) = anchors
    mask = np.zeros(nobs, dtype=int)
    first = np.unique(np.round(np.linspace(1, row_a - 1, rank_a - 1)).astype(int))
    second = np.unique(np.round(np.linspace(row_a + 1, row_b - 1, rank_b - rank_a - 1)).astype(int))
    third = np.unique(np.round(np.linspace(row_b + 1, nobs, n_kept - rank_b)).astype(int))
    for rows in (first, [row_a], second, [row_b], third):
        mask[np.asarray(rows, dtype=int) - 1] = 1
    kept = np.flatnonzero(mask) + 1
    if kept.size != n_kept:
        raise AssertionError(f"fixture mask keeps {kept.size} rows, wanted {n_kept}")
    for rank, row in anchors:
        if kept[rank - 1] != row:
            raise AssertionError(
                f"fixture mask puts kept rank {rank} at raw row {kept[rank - 1]}, wanted {row}"
            )
    return mask


def build_synthetic_profile_coordinates(tags: tuple[str, ...] = SYNTHETIC_TAGS):
    """Coordinate table anchored on the verified 66 / 98 positions."""
    # Anchor on the RAW ROWS (397, 495 -> 0-based 396, 494), not on the
    # clear-sky-kept ranks: the profile file is indexed by raw row.
    anchor_rows = [row - 1 for _, row in SYNTHETIC_OBS_ANCHORS]
    lat_anchor = [SYNTHETIC_OBS_LATLON[label][0] for label in ("66", "98")]
    lon_anchor = [SYNTHETIC_OBS_LATLON[label][1] for label in ("66", "98")]
    rows = [0] + anchor_rows + [SYNTHETIC_NOBS - 1]
    lat = np.interp(np.arange(SYNTHETIC_NOBS), rows, [14.05] + lat_anchor + [14.80])
    lon = np.interp(np.arange(SYNTHETIC_NOBS), rows, [147.90] + lon_anchor + [147.00])
    return {tag: (lat.copy(), lon.copy()) for tag in tags}


SYNTHETIC_UNIFIED_MASK = build_synthetic_mask()


def _build_synthetic_dataset(root: Path) -> Config:
    """Write a compact on-disk dataset shaped exactly like the server layout."""
    hx_root = root / "3create_obs" / "hx_rttov"
    obs_bt = hx_root / "3obs_BT_LACC" / "AMSUA"
    ens_bt = hx_root / "4ens_BT_LACC"
    profile_dir = hx_root / "profile" / "profile_d01_LACC_10_00_00"
    single_dir = root / "4assimilation" / "1convert_obs" / "run_dir"
    background_dir = root / "4assimilation" / "0mem_all_time" / "10_00_00"
    nr_dir = root / "NR_wrfout" / "2domain"
    for directory in (obs_bt, ens_bt, profile_dir, single_dir, background_dir, nr_dir):
        directory.mkdir(parents=True, exist_ok=True)

    side = SYNTHETIC_SIDE
    nobs = SYNTHETIC_NOBS
    n_member = SYNTHETIC_MEMBERS
    center_tag = "10_00_00"
    tags = list(SYNTHETIC_TAGS)
    ocean_by_member = SYNTHETIC_OCEAN_BY_MEMBER
    hx_by_lag_member = SYNTHETIC_HX_BY_LAG_MEMBER
    nr_bt_by_lag = SYNTHETIC_NR_BT_BY_LAG
    unified_mask = SYNTHETIC_UNIFIED_MASK
    labels = tuple(SYNTHETIC_OBS_LATLON)
    coordinates = build_synthetic_profile_coordinates()

    # --- LACC times file ---------------------------------------------------
    lines = [f"center_time={center_tag}"] + [f"lag_time={tag}" for tag in tags]
    (obs_bt / f"BT_LACC_{center_tag}").mkdir(parents=True, exist_ok=True)
    (obs_bt / f"BT_LACC_{center_tag}" / "LACC_times.txt").write_text("\n".join(lines) + "\n")

    # --- per-lag files -----------------------------------------------------
    # Each file carries a ramp so that the C-order flattening is genuinely a
    # different vector from the F-order one.  The two observed pixels are then
    # OVERWRITTEN with the fixture values, so what the diagnostic reads there
    # is exactly SYNTHETIC_HX_BY_LAG_MEMBER / SYNTHETIC_NR_BT_BY_LAG and the
    # expected covariance can be written in closed form.
    base_column = np.arange(1, nobs + 1, dtype=float)
    obs_positions = [
        int(np.flatnonzero(unified_mask)[rank - 1]) for rank, _ in SYNTHETIC_OBS_ANCHORS
    ]

    for lag_index, tag in enumerate(tags):
        directory = obs_bt / f"BT_{tag}"
        directory.mkdir(parents=True, exist_ok=True)

        column = base_column + 100.0 * lag_index
        column[obs_positions] = nr_bt_by_lag[lag_index]
        np.savetxt(directory / "obs_d01_ch4_totalline.txt", column, fmt="%.10f")
        # Every lag gets its OWN matrix, built as the F-order flattening of its
        # own column, because the reader now enforces that check per lag.
        matrix = column.reshape(side, side, order="F")
        assert not np.array_equal(matrix.reshape(-1, order="C"), column)
        np.savetxt(directory / "obs_d01_ch4.txt", matrix, fmt="%.10f")
        # Every lag shares the same kept pattern, so the unified (center-time)
        # mask and the per-lag masks agree by construction.
        np.savetxt(directory / "clear_sky_mask.txt", unified_mask, fmt="%d")

        for member in range(1, n_member + 1):
            member_dir = ens_bt / f"mem{member:03d}" / "AMSUA" / f"BT_{tag}"
            member_dir.mkdir(parents=True, exist_ok=True)
            member_column = column.copy()
            member_column[obs_positions] = hx_by_lag_member[lag_index, member - 1]
            np.savetxt(
                member_dir / "obs_d01_ch4_totalline.txt", member_column, fmt="%.10f"
            )
            # Each member's OWN matrix, the F-order flattening of its own column,
            # next to that column: the member Hx F-order check compares a file
            # against its own directory, so a fixture without these would leave
            # the check untested (and, under the strict default, unpassable).
            member_matrix = member_column.reshape(side, side, order="F")
            assert not np.array_equal(
                member_matrix.reshape(-1, order="C"), member_column
            )
            np.savetxt(member_dir / "obs_d01_ch4.txt", member_matrix, fmt="%.10f")

    # The unified mask lives under the center tag.  The 12 h lag has no mask on
    # the server, which the "unknown" path below reproduces for an extra lag.
    np.savetxt(obs_bt / f"BT_LACC_{center_tag}" / "clear_sky_mask.txt", unified_mask, fmt="%d")
    np.savetxt(
        obs_bt / f"BT_LACC_{center_tag}" / "obs_d01_ch4_totalline_withpert.txt",
        base_column,
        fmt="%.6f",
    )

    # --- profiles ----------------------------------------------------------
    # Coordinates are anchored so the raw rows the anchors point at carry
    # exactly the verified latitudes and longitudes.
    for tag in tags:
        day, hour, minute = tag.split("_")
        tag_lat, tag_lon = coordinates[tag]
        body = []
        for index in range(nobs):
            body.append("! filler line keeping the stride non-uniform")
            body.append(PROFILE_COORDINATE_MARKER)
            body.append(f"  0.0100  {tag_lat[index]:.4f}  {tag_lon[index]:.4f}")
        (profile_dir / f"prof{day}_{hour}:{minute}.dat").write_text("\n".join(body) + "\n")

    # --- background and NR netCDF -----------------------------------------
    grid_side = 6
    lat = np.linspace(13.5, 14.6, grid_side)
    lon = np.linspace(146.5, 147.9, grid_side)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    def write_state_file(
        path: Path, level0: float | np.ndarray, extra: float = 0.0
    ) -> None:
        # f8 rather than WRF's f4: the synthetic case exists to check the
        # FORMULA, and float32 rounding would put a 1e-7 floor on every
        # comparison and blur a real sign error into a tolerance.
        with nc.Dataset(path, "w", format="NETCDF4") as dataset:
            dataset.createDimension("Time", 1)
            dataset.createDimension("DateStrLen", 19)
            dataset.createDimension("ocean_layer_stag", 30)
            dataset.createDimension("south_north", grid_side)
            dataset.createDimension("west_east", grid_side)
            times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
            stamp = "2018-09-10_00:00:00"
            times[:] = np.array([[char.encode("ascii") for char in stamp]], dtype="S1")
            xlat = dataset.createVariable("XLAT", "f8", ("Time", "south_north", "west_east"))
            xlon = dataset.createVariable("XLONG", "f8", ("Time", "south_north", "west_east"))
            xlat[:] = lat2d
            xlon[:] = lon2d
            landmask = dataset.createVariable("LANDMASK", "f8", ("Time", "south_north", "west_east"))
            landmask[:] = 0.0  # all water
            omtmp = dataset.createVariable(
                "OM_TMP", "f8", ("Time", "ocean_layer_stag", "south_north", "west_east")
            )
            field = np.full((30, grid_side, grid_side), 290.0 + extra, dtype="f8")
            # Level 0 gets the member/NR specific value; other levels are filler.
            field[0, :, :] = level0
            omtmp[0, :, :, :] = field

    for member in range(1, n_member + 1):
        write_state_file(
            background_dir / f"firstguess_d01.mem{member:03d}",
            290.0 + ocean_by_member[member - 1],
        )
    # The NR is placed well away from the prior mean (e_o ~ +1.19 K) so that
    # the TOWARD_NR / AWAY_FROM_NR branches are actually exercised rather than
    # short-circuited by the near-zero guard.
    write_state_file(nr_dir / "wrfout_d02_2018-09-10_00:00:00", 290.0 + 1.5)

    # --- single-observation obs_seq ---------------------------------------
    for (filtered_index, raw_index), label in zip(SYNTHETIC_OBS_ANCHORS, labels):
        position = raw_index - 1
        obs_lat, obs_lon = SYNTHETIC_OBS_LATLON[label]
        window = np.mean(hx_by_lag_member[[0, 1, 2], :], axis=0)
        actual_value = float(np.mean(nr_bt_by_lag)) + 0.11
        body = [
            " obs_sequence",
            "obs_kind_definitions",
            "           1",
            "         170",
            "num_copies",
            "          1",
            " OBS        1",
            f"  {actual_value:.10f}",
            "   -1",
            "obdef",
            "loc3d",
            f"  {math.radians(obs_lon):.10f}  {math.radians(obs_lat):.10f}  0.0  1",
            "kind",
            "        170",
            "external_FO",
            f"  {n_member}",
            "  " + "  ".join(f"{value:.10f}" for value in window),
            "  0.250",
        ]
        (single_dir / f"obs_seq.out_LACC_single_{label}").write_text("\n".join(body) + "\n")

    return Config(
        project_root=root,
        hx_root=hx_root,
        output_dir=root / "out",
        background_choice="mem_all_time",
        mem_all_time_dir=background_dir,
        nobs_raw=nobs,
        matrix_side=side,
        member_start=1,
        member_end=n_member,
        lag_hours=(0, 3, 6),
        obs_targets=tuple(
            ObsTarget(
                label,
                filtered_index=filtered_index,
                expected_lat=SYNTHETIC_OBS_LATLON[label][0],
                expected_lon=SYNTHETIC_OBS_LATLON[label][1],
            )
            for (filtered_index, _), label in zip(SYNTHETIC_OBS_ANCHORS, labels)
        ),
        check_reference_values=False,
        write_figures=True,
        local_window_deg=2.0,
        # Tightened so the deliberately weak 3 h coupling (~0.08) is reported
        # as a direction rather than swallowed by the near-zero guard.
        near_zero_frac=0.05,
    )


def _make_series(lag_hours, hx_rows, y_nr, tags=None) -> LagSeries:
    """Minimal LagSeries for the pure window / external_FO tests."""
    hours = list(lag_hours)
    return LagSeries(
        lag_hours=hours,
        tags=list(tags) if tags is not None else [f"tag{index}" for index in range(len(hours))],
        valid_times=[f"valid{index}" for index in range(len(hours))],
        clear_sky_flags=["clear"] * len(hours),
        hx=np.asarray(hx_rows, dtype=float),
        y_nr=np.asarray(y_nr, dtype=float),
        hx_file_count=0,
        obs_lat_at_lag=[14.0] * len(hours),
        obs_lon_at_lag=[147.0] * len(hours),
        obs_displacement_km=[0.0] * len(hours),
    )


def _make_context(obs_value: float | None = 251.0, label: str = "66") -> ObservationContext:
    """Minimal ObservationContext for the pure window / external_FO tests.

    The stored external_FO goes through the real selection path, so these tests
    exercise the same member correspondence the full run does.
    """
    sample = SampleResult(
        value=290.0,
        sample_lat=14.0,
        sample_lon=147.0,
        distance_km=0.0,
        mode="linear",
        n_support_points=4,
        local_window_deg=1.0,
    )
    record = ObsSeqRecord(
        path=Path(f"obs_seq.out_LACC_single_{label}"),
        obs_block_number=1,
        obs_value=obs_value,
        lat=14.0,
        lon=147.0,
        external_fo=np.zeros(4, dtype=float),
        external_fo_member_count=4,
    )
    return ObservationContext(
        label=label,
        filtered_index=1,
        raw_index=397,
        zero_based_raw_index=396,
        obs_lat=14.0,
        obs_lon=147.0,
        state_lat=14.0,
        state_lon=147.0,
        ocean_prior=np.array([289.0, 290.0, 291.0, 290.0]),
        ocean_nr=290.0,
        ocean_sample=sample,
        nr_sample=sample,
        background_files=[],
        background_times=[],
        single_obs=record,
        single_obs_location=None,
        external_fo=select_external_fo(record, [1, 2, 3, 4]),
    )


def _with_external_fo(
    context: ObservationContext, values, members: tuple[int, ...] = (1, 2, 3, 4)
) -> ObservationContext:
    """Attach a stored external_FO array to a synthetic context.

    Both the record and the context's selection are rebuilt, because the check
    reads the SELECTED entries -- updating only the raw record would leave the
    old values in place and the test would pass for the wrong reason.
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    record = replace(
        context.single_obs,
        external_fo=array,
        external_fo_member_count=int(array.size),
        external_fo_unavailable_reason=None,
    )
    return replace(
        context, single_obs=record, external_fo=select_external_fo(record, list(members))
    )


def _make_grid(
    path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    field: np.ndarray,
    water: np.ndarray | None = None,
    time_string: str | None = "2018-09-10_00:00:00",
) -> GridField:
    lat_arr = np.asarray(lat, dtype=float)
    return GridField(
        path=path,
        field=np.asarray(field, dtype=float),
        lat=lat_arr,
        lon=np.asarray(lon, dtype=float),
        water=np.ones_like(lat_arr, dtype=bool) if water is None else np.asarray(water, dtype=bool),
        water_source="LANDMASK",
        time_string=time_string,
    )


def _replace_loc3d(path: Path, lat_deg: float, lon_deg: float) -> None:
    """Rewrite a single-obs file's loc3d to another location, keeping its value."""
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.strip().split()[:1] == ["loc3d"] and index + 1 < len(lines):
            lines[index + 1] = (
                f"  {math.radians(lon_deg):.10f}  {math.radians(lat_deg):.10f}  0.0  1"
            )
            path.write_text("\n".join(lines) + "\n")
            return
    raise AssertionError(f"no loc3d block found in {path}")


def _rewrite_loc3d_text(path: Path, lon_text: str, lat_text: str) -> None:
    """Put RAW text into the loc3d line, to build NaN / Inf inputs."""
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.strip().split()[:1] == ["loc3d"] and index + 1 < len(lines):
            lines[index + 1] = f"  {lon_text}  {lat_text}  0.0  1"
            path.write_text("\n".join(lines) + "\n")
            return
    raise AssertionError(f"no loc3d block found in {path}")


def _rewrite_obs_value_text(path: Path, text: str) -> None:
    """Put RAW text into the OBS block's value line, to build NaN inputs."""
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0] == "OBS" and tokens[1].isdigit():
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            lines[cursor] = f"  {text}"
            path.write_text("\n".join(lines) + "\n")
            return
    raise AssertionError(f"no OBS block found in {path}")


def _rewrite_external_fo(path: Path, values, declared: int | None) -> None:
    """Rebuild a fixture single-obs file's external_FO tail.

    The synthetic writer puts the block last and closes it with a bare ``0.250``
    line, so replacing everything from the keyword onwards reproduces exactly
    the layout the writer produces -- with a count line that may be absent or
    wrong.  (That trailing numeric line is also why the reader must trust a
    declared count instead of scanning for numbers until they stop.)
    """
    lines = path.read_text().splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().split()[:1] == ["external_FO"]:
            start = index
            break
    if start is None:
        raise AssertionError(f"no external_FO block found in {path}")
    tail = ["external_FO"]
    if declared is not None:
        tail.append(f"  {declared}")
    if values is not None and len(np.asarray(values).reshape(-1)):
        tail.append("  " + "  ".join(f"{value:.10f}" for value in np.asarray(values).reshape(-1)))
    tail.append("  0.250")
    path.write_text("\n".join(lines[:start] + tail) + "\n")


def _remove_external_fo(path: Path) -> None:
    """Delete the external_FO block entirely (the optional-block case)."""
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.strip().split()[:1] == ["external_FO"]:
            path.write_text("\n".join(lines[:index]) + "\n")
            return
    raise AssertionError(f"no external_FO block found in {path}")


def _shift_profile_coordinates(path: Path, dlat: float, dlon: float) -> None:
    """Move every sounding in a profile file by a constant offset."""
    lines = path.read_text().splitlines()
    moved = 0
    for index, line in enumerate(lines):
        if line.strip() != PROFILE_COORDINATE_MARKER or index + 1 >= len(lines):
            continue
        fields = lines[index + 1].split()
        elevation, lat, lon = (float(value) for value in fields[:3])
        lines[index + 1] = f"  {elevation:.4f}  {lat + dlat:.4f}  {lon + dlon:.4f}"
        moved += 1
    if not moved:
        raise AssertionError(f"no coordinate rows found in {path}")
    path.write_text("\n".join(lines) + "\n")


def _swap_file_contents(first: Path, second: Path) -> None:
    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()
    first.write_bytes(second_bytes)
    second.write_bytes(first_bytes)


def _set_times(path: Path, stamp: str) -> None:
    """Overwrite a netCDF file's Times variable in place."""
    if len(stamp) != 19:
        raise AssertionError(f"Times stamp must be 19 characters, got {len(stamp)}")
    with nc.Dataset(path, "r+") as dataset:
        times = dataset.variables["Times"]
        times[:] = np.array([[char.encode("ascii") for char in stamp]], dtype="S1")


def run_self_test() -> int:
    """Synthetic checks for the covariance, F-order numbering, the window
    product, and the missing / zero-variance paths.

    This validates the CODE, not the server data.  A green run here says
    nothing about whether the real LACC numbers are correct.
    """
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        status = "ok  " if condition else "FAIL"
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
        if not condition:
            failures.append(name)

    print("=" * 78)
    print("SELF-TEST (synthetic data; this validates the code, not the server data)")
    print("=" * 78)

    # -- 1. covariance and correlation, ddof = 1 ----------------------------
    print("\n[1] covariance / correlation (ddof = 1)")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 1.0, 4.0, 3.0, 7.0])
    moments = pair_moments(x, y)
    expected_cov = float(np.cov(x, y, ddof=1)[0, 1])
    expected_corr = float(np.corrcoef(x, y)[0, 1])
    check("covariance matches np.cov(ddof=1)", math.isclose(moments.cov, expected_cov, rel_tol=1e-12),
          f"{moments.cov!r} vs {expected_cov!r}")
    check("correlation matches np.corrcoef", math.isclose(moments.corr, expected_corr, rel_tol=1e-12),
          f"{moments.corr!r} vs {expected_corr!r}")
    check("n_used counts finite pairs", moments.n_used == 5)

    # ddof=1 differs from ddof=0 by exactly sqrt(n/(n-1)) on the std
    population_std = float(np.std(x))
    check(
        "ddof=1 (not ddof=0)",
        math.isclose(moments.std_x, population_std * math.sqrt(5 / 4), rel_tol=1e-12),
        f"{moments.std_x:.10f}",
    )

    # -- 2. missing members and zero variance ------------------------------
    print("\n[2] missing members and zero variance")
    x_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y_nan = np.array([2.0, np.nan, 4.0, 3.0, 7.0])
    moments_nan = pair_moments(x_nan, y_nan)
    check("non-finite members dropped pairwise", moments_nan.n_used == 3,
          f"n_used={moments_nan.n_used}")
    check("dropped members still give a finite covariance",
          math.isfinite(moments_nan.cov))

    zero = np.full(5, 3.0)
    moments_zero = pair_moments(x, zero)
    check("zero-variance ensemble is flagged, not zeroed",
          moments_zero.degenerate and math.isnan(moments_zero.corr) and moments_zero.cov == 0.0,
          f"cov={moments_zero.cov}, corr={moments_zero.corr}")

    too_few = pair_moments(np.array([1.0, np.nan, np.nan]), np.array([1.0, 2.0, 3.0]))
    check("too few valid members -> NaN, not a fabricated number",
          too_few.degenerate and math.isnan(too_few.cov))

    check(
        "zero variance is classified as undefined, never as TOWARD_NR",
        classify_alignment(0.0, 1.0, float("nan"), 0.5, 0.1) == "WEAK_COUPLING",
    )
    check(
        "a near-zero e_o is reported as UNDEFINED_E_O",
        classify_alignment(1.0, 1.0e-9, 0.9, 0.5, 0.1) == "UNDEFINED_E_O",
    )

    # -- 3. window product is not the weighted sum of products -------------
    print("\n[3] window product vs. the naive weighted sum of products")
    ocean = np.array([0.1, -0.3, 0.55, 0.9])
    hx = np.array([[250.0, 250.4, 249.8, 250.9],
                   [251.0, 250.2, 251.3, 249.9],
                   [249.5, 250.8, 249.9, 250.4]])
    weights = np.full(3, 1.0 / 3.0)
    hx_window = np.mean(hx, axis=0)
    c_window = pair_moments(ocean, hx_window).cov
    d_window = float(np.mean(np.array([250.5, 251.4, 250.9])
                             - hx.mean(axis=1)))
    product = c_window * d_window

    per_lag_c = [pair_moments(ocean, hx[index]).cov for index in range(3)]
    per_lag_d = [250.5 - hx[0].mean(), 251.4 - hx[1].mean(), 250.9 - hx[2].mean()]
    naive = float(np.sum([weights[index] * per_lag_c[index] * per_lag_d[index]
                          for index in range(3)]))
    check("window covariance uses the window-mean Hx ensemble",
          math.isclose(c_window, pair_moments(ocean, hx.mean(axis=0)).cov, rel_tol=1e-12))
    check("the two expressions really are different (so the test is meaningful)",
          not math.isclose(product, naive, rel_tol=1e-12),
          f"product={product:.8g}, naive={naive:.8g}")

    # -- 4. end-to-end on the synthetic dataset ----------------------------
    print("\n[4] coordinate displacement between lag times")
    base = ProfileCoordinates(
        tag="10_00_00",
        path=Path("base.dat"),
        elevation_km=np.zeros(3),
        lat=np.array([14.0, 14.1, 14.2]),
        lon=np.array([147.0, 147.1, 147.2]),
    )
    moved = ProfileCoordinates(
        tag="09_12_00",
        path=Path("moved.dat"),
        elevation_km=np.zeros(3),
        lat=base.lat + np.array([0.0, 0.09, 0.0]),   # ~10 km north at row 2
        lon=base.lon.copy(),
    )
    identical = ProfileCoordinates(
        tag="09_21_00",
        path=Path("same.dat"),
        elevation_km=np.zeros(3),
        lat=base.lat.copy(),
        lon=base.lon.copy(),
    )
    same_report = compare_profile_coordinates(base, identical)
    moved_report = compare_profile_coordinates(base, moved)
    check(
        "identical coordinates report same_fixed_location",
        same_report["summary"]["same_fixed_location"] is True
        and same_report["summary"]["n_moved"] == 0,
    )
    check(
        "a 0.09 deg shift is detected and located",
        moved_report["summary"]["same_fixed_location"] is False
        and moved_report["summary"]["n_moved"] == 1
        and moved_report["summary"]["max_at_raw_index"] == 2,
        f"n_moved={moved_report['summary']['n_moved']}, "
        f"at={moved_report['summary']['max_at_raw_index']}",
    )
    check(
        "0.09 deg of latitude is reported as roughly 10 km",
        9.0 < moved_report["summary"]["max_distance_km"] < 11.0,
        f"{moved_report['summary']['max_distance_km']:.3f} km",
    )

    # -- 5. end-to-end on the synthetic dataset ----------------------------
    print("\n[5] end-to-end run on a synthetic dataset")

    # The fixture reproduces the documented layout, so the index mapping the
    # real run depends on is checked directly rather than assumed.
    kept = np.flatnonzero(SYNTHETIC_UNIFIED_MASK) + 1
    check("fixture mask keeps 186 of 676", kept.size == 186, f"{kept.size} kept")
    check(
        "filtered 66 -> raw 397 and filtered 98 -> raw 495",
        kept[65] == 397 and kept[97] == 495,
        f"kept[66]={kept[65]}, kept[98]={kept[97]}",
    )
    check(
        "raw 397 is NOT row 66 of the raw file",
        kept[65] != 66 and kept[97] != 98,
        "the kept rank and the raw row are different numbering systems",
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = _build_synthetic_dataset(root)

        # F-order check first: the writer used order="F", so the reader must see it.
        column = read_column_file(nr_bt_path(config, config.center_tag()), config.nobs_raw, "nr")
        flattening = check_matrix_flattening(
            matrix_path(config, config.center_tag()), column, config.matrix_side, "synthetic"
        )
        check("F-order flattening detected", flattening["f_order_matches"] is True)
        check("C-order flattening correctly rejected", flattening["c_order_matches"] is False)

        payload = run(config)

        check("lag CSV written", Path(payload["outputs"]["lag_csv"]).is_file())
        check("window CSV written", Path(payload["outputs"]["window_csv"]).is_file())
        check("JSON written", Path(payload["outputs"]["json"]).is_file())

        for label, report in payload["observations"].items():
            rows = payload["_lag_rows"][label]
            check(f"obs {label}: one row per lag", len(rows) == 3)

            reference_rows = {
                row["lag_hours"]: row for row in rows
            }
            for hours in (0, 3, 6):
                row = reference_rows[hours]
                lag_index = {0: 0, 3: 1, 6: 2}[hours]
                check(
                    f"obs {label}: c_j at {hours}h matches np.cov(ddof=1)",
                    math.isclose(
                        row["c_j"],
                        float(np.cov(SYNTHETIC_OCEAN_BY_MEMBER, SYNTHETIC_HX_BY_LAG_MEMBER[lag_index], ddof=1)[0, 1]),
                        rel_tol=1e-10,
                    ),
                    f"c={row['c_j']!r}",
                )
                check(
                    f"obs {label}: d_j_NR at {hours}h matches y_NR - mean(Hx)",
                    math.isclose(
                        row["d_j_NR"],
                        float(SYNTHETIC_NR_BT_BY_LAG[lag_index] - SYNTHETIC_HX_BY_LAG_MEMBER[lag_index].mean()),
                        rel_tol=1e-10,
                    ),
                    f"d={row['d_j_NR']!r}",
                )
                check(
                    f"obs {label}: clear_sky_flag at {hours}h is read, not assumed",
                    row["clear_sky_flag"] in {"clear", "not_clear", "unknown"},
                    row["clear_sky_flag"],
                )

            windows = {row["window_label"]: row for row in payload["_window_rows"][label]}
            check(f"obs {label}: three prefix windows", len(windows) == 3)
            full = windows["[0,3,6]"]
            expected_c = float(
                np.cov(SYNTHETIC_OCEAN_BY_MEMBER, SYNTHETIC_HX_BY_LAG_MEMBER.mean(axis=0), ddof=1)[0, 1]
            )
            check(
                f"obs {label}: [0,3,6] window c_w uses the window-mean Hx",
                math.isclose(full["c_w"], expected_c, rel_tol=1e-10),
                f"{full['c_w']!r}",
            )
            check(
                f"obs {label}: [0,3,6] window product is c_w * d_w, not the naive sum",
                math.isclose(
                    full["c_w_times_d_w_NR"], full["c_w"] * full["d_w_NR"], rel_tol=1e-12
                )
                and not math.isclose(
                    full["c_w_times_d_w_NR"],
                    full["sum_j_wj_cj_dj_NAIVE_do_not_use"],
                    rel_tol=1e-12,
                ),
                f"product={full['c_w_times_d_w_NR']:.6g}, "
                f"naive={full['sum_j_wj_cj_dj_NAIVE_do_not_use']:.6g}",
            )

            # d_w must be the weighted mean of the per-lag INNOVATIONS, not the
            # weighted mean of the raw NR brightness temperatures.  Using y_NR
            # would inflate every window product by roughly the 250 K offset,
            # which is exactly the failure the [0] window would hide.
            expected_d = float(
                np.mean(
                    [
                        SYNTHETIC_NR_BT_BY_LAG[index]
                        - SYNTHETIC_HX_BY_LAG_MEMBER[index].mean()
                        for index in range(3)
                    ]
                )
            )
            check(
                f"obs {label}: [0,3,6] d_w is the mean of d_j, not of y_NR",
                math.isclose(full["d_w_NR"], expected_d, rel_tol=1e-10)
                and not math.isclose(
                    full["d_w_NR"], float(SYNTHETIC_NR_BT_BY_LAG.mean()), rel_tol=1e-3
                ),
                f"d_w={full['d_w_NR']:.6f}, expected={expected_d:.6f}, "
                f"mean(y_NR)={float(SYNTHETIC_NR_BT_BY_LAG.mean()):.3f}",
            )

            # The alignment flags must follow sign(c_j*d_j)*sign(e_o) whenever
            # neither the coupling nor e_o is near zero.
            flags = {int(row["lag_hours"]): row["alignment_flag"] for row in rows}
            check(
                f"obs {label}: 0 h is reported TOWARD_NR",
                flags[0] == "TOWARD_NR",
                flags[0],
            )
            check(
                f"obs {label}: 3 h and 6 h are reported AWAY_FROM_NR",
                flags[3] == "AWAY_FROM_NR" and flags[6] == "AWAY_FROM_NR",
                f"{flags[3]}, {flags[6]}",
            )
            signs = {
                int(row["lag_hours"]): math.copysign(1.0, row["c_j_times_d_j_NR"])
                for row in rows
            }
            e_ocean = rows[0]["e_o"]
            check(
                f"obs {label}: alignment sign equals sign(c_j*d_j)*sign(e_o)",
                all(
                    (flags[hours] == "TOWARD_NR") == (signs[hours] * e_ocean > 0)
                    for hours in (0, 3, 6)
                ),
                f"signs={[signs[hours] for hours in (0, 3, 6)]}, e_o={e_ocean:+.4f}",
            )

            check(
                f"obs {label}: index mapping resolved",
                report["raw_index"]
                == int(np.flatnonzero(SYNTHETIC_UNIFIED_MASK)[report["filtered_index"] - 1] + 1),
            )
            check(
                f"obs {label}: coordinate displacement check ran",
                "profile_coordinate_displacement" in payload["checks"],
            )
            check(f"obs {label}: figure written", Path(report["figure"]).is_file())

        # The external_FO cross-check must actually pass on consistent inputs;
        # a parser that swallowed the member-count line would shift every value
        # and land here instead.
        for label in payload["observations"]:
            external = payload["checks"]["external_fo"][label]
            check(
                f"obs {label}: external_FO cross-check passes",
                external.get("status") == "passed",
                f"status={external.get('status')}, "
                f"max|diff|={external.get('max_abs_difference_K')}",
            )

        # Missing per-lag mask must be recorded as unknown, never as clear.
        withpert = payload["checks"]["withpert_availability"]
        check(
            "per-lag *_withpert absence recorded",
            withpert["per_lag_withpert_present"] is False,
        )

        # A missing member file must refuse rather than fall back.
        missing_member = background_member_path(config, config.member_end)
        missing_member.rename(missing_member.with_suffix(".moved"))
        refused = False
        try:
            run(config)
        except MissingInputError:
            refused = True
        except Exception:  # noqa: BLE001 - any other failure is a test failure
            refused = False
        finally:
            missing_member.with_suffix(".moved").rename(missing_member)
        check("a missing background member refuses instead of falling back", refused)

        # A non-water state point must be refused.
        waterless = replace(config, sample_require_water=True, state_lat=90.0, state_lon=0.0)
        refused = False
        try:
            run(waterless)
        except SamplingError:
            refused = True
        except Exception:  # noqa: BLE001
            refused = False
        check("a point outside the domain refuses instead of extrapolating", refused)

        # Extra lags cannot sneak in without the explicit switch.
        refused = False
        try:
            validate_config(replace(config, extra_lag_hours=(9,)))
        except ValueError:
            refused = True
        check("extra lags require the explicit enable flag", refused)

    # -- 6. direction classification uses alignment = product * e_o ---------
    print("\n[6] alignment classification uses product * e_o, not product alone")
    corr_ok, std_ok, frac = 0.9, 0.5, 0.1
    check(
        "product>0, e_o>0 -> TOWARD_NR",
        classify_alignment(1.0, 1.0, corr_ok, std_ok, frac) == "TOWARD_NR",
    )
    check(
        "product<0, e_o>0 -> AWAY_FROM_NR",
        classify_alignment(-1.0, 1.0, corr_ok, std_ok, frac) == "AWAY_FROM_NR",
    )
    check(
        "product>0, e_o<0 -> AWAY_FROM_NR (a product-only test would say TOWARD)",
        classify_alignment(1.0, -1.0, corr_ok, std_ok, frac) == "AWAY_FROM_NR",
    )
    check(
        "product<0, e_o<0 -> TOWARD_NR (a product-only test would say AWAY)",
        classify_alignment(-1.0, -1.0, corr_ok, std_ok, frac) == "TOWARD_NR",
    )
    check(
        "e_o == 0 is explicitly undefined even when near_zero_frac == 0",
        classify_alignment(1.0, 0.0, corr_ok, std_ok, 0.0) == "UNDEFINED_E_O",
        classify_alignment(1.0, 0.0, corr_ok, std_ok, 0.0),
    )
    check(
        "near_zero_frac == 0 still never classifies a zero alignment as TOWARD",
        classify_alignment(0.0, 1.0, corr_ok, std_ok, 0.0) == "EXACTLY_ZERO",
        classify_alignment(0.0, 1.0, corr_ok, std_ok, 0.0),
    )
    check(
        "a degenerate ensemble stays undefined regardless of e_o sign",
        classify_alignment(1.0, -1.0, float("nan"), 0.5, frac) == "WEAK_COUPLING",
    )

    # -- 7. the assimilated window is not the diagnostic window -------------
    print("\n[7] assimilated window vs. diagnostic window")
    # Assimilated window [0,3,6]: Hx mean 250 K, NR BT mean 250 K, obs 251 K.
    # Extra 9 h diagnostic: Hx 254 K, NR BT 258 K -- deliberately far away, so
    # folding it in changes the answer and the test can tell.
    window_series = _make_series(
        [0, 3, 6, 9],
        [[250.0] * 4, [250.0] * 4, [250.0] * 4, [254.0] * 4],
        [250.0, 250.0, 250.0, 258.0],
    )
    context = _make_context(obs_value=251.0)
    averages = compute_noise_free_averages(config, context, window_series, {0, 3, 6})
    check(
        "assimilated-window quantities are available",
        averages["available"] is True,
        str(averages["unavailable_reason"]),
    )
    check(
        "the 9 h diagnostic lag is excluded from the window",
        averages["excluded_extra_lag_hours"] == [9],
        f"{averages['excluded_extra_lag_hours']}",
    )
    check(
        "d_L_actual == +1 K with the extra 9 h lag present",
        math.isclose(averages["d_L_actual"], 1.0, abs_tol=1e-12),
        f"{averages['d_L_actual']!r}",
    )
    check(
        "noise_L == +1 K with the extra 9 h lag present",
        math.isclose(averages["noise_L"], 1.0, abs_tol=1e-12),
        f"{averages['noise_L']!r}",
    )
    check(
        "d_L_NR == 0 K (noise-free window innovation)",
        math.isclose(averages["d_L_NR"], 0.0, abs_tol=1e-12),
        f"{averages['d_L_NR']!r}",
    )
    # Demonstrate the test discriminates: averaging ALL diagnosed lags gives
    # d_L_actual = 251 - 251 = 0 and noise_L = 251 - 252 = -1.
    all_lag_averages = compute_noise_free_averages(
        config, context, window_series, {0, 3, 6, 9}
    )
    check(
        "the excluded lag really would change the answer (test is meaningful)",
        not math.isclose(all_lag_averages["d_L_actual"], 1.0, abs_tol=1e-12)
        and not math.isclose(all_lag_averages["noise_L"], 1.0, abs_tol=1e-12),
        f"all-lag d_L_actual={all_lag_averages['d_L_actual']!r}, "
        f"noise_L={all_lag_averages['noise_L']!r}",
    )

    partial = _make_series([0, 3], [[250.0] * 4, [250.0] * 4], [250.0, 250.0])
    partial_averages = compute_noise_free_averages(config, context, partial, {0, 3, 6})
    check(
        "an incomplete assimilated window is UNAVAILABLE, not approximated",
        partial_averages["available"] is False
        and partial_averages["d_L_actual"] is None
        and partial_averages["noise_L"] is None,
        str(partial_averages["unavailable_reason"]),
    )

    # external_FO must be compared over the assimilated window, not [0,3,6].
    fo_context = _make_context(obs_value=251.0)
    fo_window = np.mean(window_series.hx[[0, 1, 2], :], axis=0)
    fo_context = _with_external_fo(fo_context, fo_window)
    fo_ok = check_external_fo(fo_context, window_series, {0, 3, 6})
    check(
        "external_FO check passes over the assimilated window",
        fo_ok["status"] == "passed",
        f"status={fo_ok['status']}, max|diff|={fo_ok.get('max_abs_difference_K')}",
    )
    fo_wrong = check_external_fo(fo_context, window_series, {0, 3, 6, 9})
    check(
        "external_FO check is not hard-coded to [0,3,6]",
        fo_wrong["status"] == "failed",
        f"status={fo_wrong['status']}",
    )
    fo_partial = check_external_fo(fo_context, partial, {0, 3, 6})
    check(
        "external_FO on an incomplete window is unavailable, not substituted",
        fo_partial["status"] == "unavailable",
        f"status={fo_partial['status']}",
    )

    # -- 8. nearest mode: one index for value, coordinates and distance -----
    print("\n[8] nearest sampling uses a single index")
    # A is nearer in the (lon, lat) PLANE (0.099 vs 0.100) but farther on the
    # sphere (11.01 km vs 10.79 km), so a plane-distance value lookup would
    # return A's value while reporting B's coordinates.
    nearest_grid = _make_grid(
        Path("nearest_grid"),
        lat=np.array([[14.099, 14.0], [13.0, 15.0]]),
        lon=np.array([[0.0, 0.1], [0.0, 0.0]]),
        field=np.array([[10.0, 20.0], [30.0, 40.0]]),
    )
    nearest_config = replace(config, sample_mode="nearest")
    snapped = sample_point(nearest_grid, 14.0, 0.0, "nearest", nearest_config)
    check(
        "the value comes from the great-circle nearest point (B, value 20)",
        math.isclose(snapped.value, 20.0, abs_tol=1e-9),
        f"value={snapped.value!r} (10 would be the plane-distance point A)",
    )
    check(
        "the reported coordinates are that same point",
        math.isclose(snapped.sample_lat, 14.0, abs_tol=1e-12)
        and math.isclose(snapped.sample_lon, 0.1, abs_tol=1e-12),
        f"({snapped.sample_lat}, {snapped.sample_lon})",
    )
    check(
        "the reported distance is that same point's distance",
        10.7 < snapped.distance_km < 10.9,
        f"{snapped.distance_km:.3f} km",
    )
    check(
        "the plane-distance point really is a different cell (test is meaningful)",
        float(
            haversine_km(np.array([14.099]), np.array([0.0]), 14.0, 0.0)[0]
        )
        > snapped.distance_km,
    )
    refused = False
    try:
        sample_point(
            nearest_grid, 14.0, 0.0, "nearest", replace(nearest_config, nearest_max_km=5.0)
        )
    except OutsideDomainError:
        refused = True
    check("nearest_max_km applies to the point actually selected", refused)

    # -- 9. background / NR valid-time and grid gate ------------------------
    print("\n[9] background valid-time and common-grid gate")
    good_time = "2018-09-10_00:00:00"
    lat2d = np.array([[13.0, 13.0], [14.0, 14.0]])
    lon2d = np.array([[147.0, 148.0], [147.0, 148.0]])
    field2d = np.array([[290.0, 290.0], [290.0, 290.0]])
    members = [1, 2]
    nr_grid = _make_grid(Path("nr.nc"), lat2d, lon2d, field2d, time_string=good_time)

    def member_grid(member, time_string, lat=lat2d, lon=lon2d):
        return _make_grid(
            Path(f"preassim_member_{member:04d}_d01.nc"), lat, lon, field2d,
            time_string=time_string,
        )

    def gate(member_times, nr_time, lats=None, lons=None):
        member_lats = [lat2d, lat2d] if lats is None else lats
        member_lons = [lon2d, lon2d] if lons is None else lons
        grids = [
            member_grid(member, value, member_lats[index], member_lons[index])
            for index, (member, value) in enumerate(zip(members, member_times))
        ]
        return validate_background_consistency(
            config, grids, members,
            _make_grid(Path("nr.nc"), lat2d, lon2d, field2d, time_string=nr_time),
        )

    check("a consistent set passes the gate", gate([good_time, good_time], good_time)["status"] == "passed")

    def gate_refuses(name, *args, **kwargs) -> None:
        refused = False
        try:
            gate(*args, **kwargs)
        except ConsistencyError:
            refused = True
        check(name, refused)

    gate_refuses(
        "all members sharing ONE WRONG time is refused",
        [good_time.replace("00:00:00", "03:00:00")] * 2, good_time,
    )
    gate_refuses(
        "a single member with the wrong time is refused",
        [good_time, good_time.replace("00:00:00", "03:00:00")], good_time,
    )
    gate_refuses("a wrong NR time is refused", [good_time, good_time], "2018-09-10_03:00:00")
    gate_refuses(
        "a longitude-only grid difference is refused",
        [good_time, good_time], good_time,
        [lat2d, lat2d], [lon2d, lon2d + 0.5],
    )
    gate_refuses(
        "a latitude-only grid difference is refused",
        [good_time, good_time], good_time,
        [lat2d, lat2d + 0.5], [lon2d, lon2d],
    )
    # A MISSING Times is the documented opt-out; a present-but-WRONG one is a
    # real inconsistency and is refused either way.
    relaxed = replace(config, require_center_time_match=False)
    missing_report = validate_background_consistency(
        relaxed, [member_grid(1, None), member_grid(2, None)], members, nr_grid
    )
    check(
        "require_center_time_match=False tolerates a MISSING Times",
        missing_report["status"] == "passed",
    )
    refused = False
    try:
        validate_background_consistency(
            relaxed,
            [member_grid(1, None), member_grid(2, "2018-09-10_03:00:00")],
            members,
            nr_grid,
        )
    except ConsistencyError:
        refused = True
    check(
        "require_center_time_match=False still refuses a WRONG time",
        refused,
    )

    # -- 10. end-to-end: the gates must stop the run, not just log ----------
    print("\n[10] end-to-end refusals and reported sample coordinates")

    def diagnostic_refusal(call) -> str | None:
        """The message if ``call`` refuses with the script's own error, else None.

        Only ``DiagnosticError`` counts: any other exception means the check
        itself blew up, which must not be reported as the gate working.
        """
        try:
            call()
        except DiagnosticError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - any other failure is a test failure
            print(f"       (a non-diagnostic {type(exc).__name__} escaped: {exc})")
            return None
        return None

    def any_refusal(call) -> str | None:
        """The message if ``call`` refuses the way ``main()`` reports it.

        ``main()`` turns ``DiagnosticError`` and ``(ValueError, KeyError)`` alike
        into an ERROR line with exit code 2, so configuration-level refusals
        count here too.  A silent success still does not.
        """
        try:
            call()
        except (DiagnosticError, ValueError, KeyError) as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - any other failure is a test failure
            print(f"       (a non-diagnostic {type(exc).__name__} escaped: {exc})")
            return None
        return None

    def refuses(call) -> bool:
        return diagnostic_refusal(call) is not None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = _build_synthetic_dataset(root)

        payload = run(config)
        check("baseline synthetic run succeeds", bool(payload["observations"]))

        # (a) a NON-CENTER lag whose matrix is flattened in C-order.
        tag_3h = "09_21_00"
        matrix_file = matrix_path(config, tag_3h)
        column_3h = read_column_file(nr_bt_path(config, tag_3h), config.nobs_raw, "nr")
        backup_matrix = matrix_file.read_bytes()
        np.savetxt(
            matrix_file,
            column_3h.reshape(SYNTHETIC_SIDE, SYNTHETIC_SIDE, order="C"),
            fmt="%.10f",
        )
        check("a 3 h matrix flattened in C-order refuses the run",
              refuses(lambda: run(config)))
        matrix_file.write_bytes(backup_matrix)
        check("the run succeeds again once the 3 h matrix is restored",
              bool(run(config)["observations"]))

        # (b) a single-obs file whose loc3d points somewhere else.  Its
        # external_FO still matches, so only the coordinate check can catch it.
        obs_file = single_obs_path(config, "66")
        backup_obs = obs_file.read_text()
        _replace_loc3d(obs_file, 0.0, 0.0)
        check("a single-obs loc3d of (0,0) is refused despite matching external_FO",
              refuses(lambda: run(config)))
        obs_file.write_text(backup_obs)

        # (c) one background member on a different valid time.
        member_file = background_member_path(config, 2)
        _set_times(member_file, "2018-09-09_21:00:00")
        check("one member on the wrong valid time refuses the run",
              refuses(lambda: run(config)))
        _set_times(member_file, config.center_time)

        # (d) the NR on a different valid time.
        nr_file = config.resolved_nr_file()
        _set_times(nr_file, "2018-09-09_21:00:00")
        check("a wrong NR valid time refuses the run", refuses(lambda: run(config)))
        _set_times(nr_file, config.center_time)

        # (e) nearest mode must report the point it actually sampled.
        nearest = run(replace(config, sample_mode="nearest", write_figures=False))
        for label, report in nearest["observations"].items():
            row = nearest["_lag_rows"][label][0]
            check(
                f"obs {label}: nearest mode reports the sampled point, not the request",
                math.isclose(row["state_sample_lat"], report["state_sample_lat"])
                and math.isclose(row["state_sample_lon"], report["state_sample_lon"])
                and not math.isclose(row["state_sample_lat"], report["state_lat"])
                and row["state_sample_distance_km"] > 0.0,
                f"requested=({report['state_lat']}, {report['state_lon']}) "
                f"sampled=({row['state_sample_lat']}, {row['state_sample_lon']}) "
                f"d={row['state_sample_distance_km']:.3f} km",
            )

    # -- 11. the member Hx F-order suite ------------------------------------
    print("\n[11] member Hx F-order numbering (a suite separate from the NR one)")
    tag_0h, tag_3h, tag_6h = SYNTHETIC_TAGS
    expected_member_pairs = SYNTHETIC_MEMBERS * len(SYNTHETIC_TAGS)
    with tempfile.TemporaryDirectory() as tmp:
        config = _build_synthetic_dataset(Path(tmp))
        payload = run(replace(config, write_figures=False))
        member_suite = payload["checks"]["f_order_numbering_member_hx"]

        check(
            f"all {expected_member_pairs} member/lag pairs are checked",
            member_suite["n_member_lag_pairs"] == expected_member_pairs
            and member_suite["n_verified"] == expected_member_pairs
            and member_suite["all_verified"] is True,
            f"pairs={member_suite['n_member_lag_pairs']}, "
            f"verified={member_suite['n_verified']}, status={member_suite['status']}",
        )
        check(
            "the NR suite and the member suite are separate JSON entries",
            member_suite["suite"] == "member_hx"
            and member_suite["status"] == "passed"
            and member_suite["n_member_lag_pairs"] == expected_member_pairs
            and payload["checks"]["f_order_numbering_center"]["suite"] == "nr_bt"
            and all(
                entry["suite"] == "nr_bt" and entry["label"].startswith("NR BT")
                for entry in payload["checks"]["f_order_numbering_per_obs"]["66"].values()
            ),
            f"member suite={member_suite['suite']}, "
            f"NR suite={payload['checks']['f_order_numbering_center']['suite']}",
        )
        check(
            "the member check is not vacuous (C-order would differ on every lag)",
            len(member_suite["per_lag"]) == len(SYNTHETIC_TAGS)
            and all(
                bucket["max_abs_diff_c_order"] > 0.0
                for bucket in member_suite["per_lag"].values()
            ),
            str(
                {
                    tag: round(bucket["max_abs_diff_c_order"], 3)
                    for tag, bucket in member_suite["per_lag"].items()
                }
            ),
        )

        # (a) a NON-CENTER lag, ONE member, whose one-column file is the
        # C-order flattening of its own matrix.
        member_column_file = member_hx_path(config, 3, tag_3h)
        backup_column = member_column_file.read_bytes()
        column_3h_member3 = read_column_file(member_column_file, config.nobs_raw, "mem003 3h")
        np.savetxt(
            member_column_file,
            column_3h_member3.reshape(SYNTHETIC_SIDE, SYNTHETIC_SIDE, order="C"),
            fmt="%.10f",
        )
        message = diagnostic_refusal(lambda: run(replace(config, write_figures=False)))
        check(
            "a non-center lag, one member, C-order column file refuses the run",
            message is not None and "mem003" in message and tag_3h in message,
            message,
        )
        member_column_file.write_bytes(backup_column)
        check(
            "the run succeeds again once that member column is restored",
            bool(run(replace(config, write_figures=False))["observations"]),
        )

        # (b) swap EVERY member's 0 h and 6 h COLUMN files, leaving all matrices
        # alone.  The NR files are untouched and the window average is symmetric
        # in the two swapped lags, so external_FO still matches -- yet every 0 h
        # Hx is now the 6 h one, which flips c_0h to the 6 h value.  That is the
        # reported symptom, and only the per-member F-order check can see it.
        position = int(np.flatnonzero(SYNTHETIC_UNIFIED_MASK)[66 - 1])

        def member_columns(tag: str) -> dict[int, np.ndarray]:
            return {
                member: read_column_file(
                    member_hx_path(config, member, tag), config.nobs_raw, tag
                )
                for member in config.member_numbers()
            }

        before = {tag: member_columns(tag) for tag in SYNTHETIC_TAGS}
        for member in config.member_numbers():
            _swap_file_contents(
                member_hx_path(config, member, tag_0h),
                member_hx_path(config, member, tag_6h),
            )
        after = {tag: member_columns(tag) for tag in SYNTHETIC_TAGS}

        # Averaging is symmetric in the two swapped lags, so the window mean at
        # the observed pixel -- the value external_FO stores -- is unchanged.
        stored_member1 = float(SYNTHETIC_HX_BY_LAG_MEMBER.mean(axis=0)[0])
        mean_before = float(np.mean([before[tag][1][position] for tag in SYNTHETIC_TAGS]))
        mean_after = float(np.mean([after[tag][1][position] for tag in SYNTHETIC_TAGS]))
        check(
            "the swap really changed the 0 h member Hx (test is meaningful)",
            not np.array_equal(before[tag_0h][1], after[tag_0h][1]),
        )
        check(
            "the window mean at the observed pixel is UNCHANGED, so external_FO still agrees",
            mean_before == mean_after
            and abs(mean_after - stored_member1) <= REFERENCE_EXTERNAL_FO_TOL_K,
            f"before={mean_before!r}, after={mean_after!r}, "
            f"stored external_FO[member 1]={stored_member1!r}",
        )
        # Without the member check this is what the run would have reported.
        c_0h_original = float(
            np.cov(SYNTHETIC_OCEAN_BY_MEMBER, SYNTHETIC_HX_BY_LAG_MEMBER[0], ddof=1)[0, 1]
        )
        c_6h_original = float(
            np.cov(SYNTHETIC_OCEAN_BY_MEMBER, SYNTHETIC_HX_BY_LAG_MEMBER[2], ddof=1)[0, 1]
        )
        c_0h_swapped = pair_moments(
            SYNTHETIC_OCEAN_BY_MEMBER,
            np.array([after[tag_0h][member][position] for member in config.member_numbers()]),
        ).cov
        check(
            "the unchecked swap would flip c_0h to the 6 h value (the reported symptom)",
            math.isclose(c_0h_swapped, c_6h_original, rel_tol=1e-12)
            and math.copysign(1.0, c_0h_swapped) != math.copysign(1.0, c_0h_original),
            f"c_0h {c_0h_original:+.5f} -> {c_0h_swapped:+.5f} "
            f"(the 6 h value is {c_6h_original:+.5f})",
        )
        message = diagnostic_refusal(lambda: run(replace(config, write_figures=False)))
        check(
            "swapping the members' 0 h / 6 h columns is refused despite matching external_FO",
            message is not None and "mem001" in message and tag_0h in message,
            message,
        )
        for member in config.member_numbers():
            _swap_file_contents(
                member_hx_path(config, member, tag_0h),
                member_hx_path(config, member, tag_6h),
            )
        check(
            "the run succeeds again once the member columns are swapped back",
            bool(run(replace(config, write_figures=False))["observations"]),
        )

        # (c) a missing member matrix: strict by default, an auditable skip when
        # deliberately relaxed -- never a silent pass.
        member_matrix = member_matrix_path(config, 1, tag_6h)
        backup_matrix = member_matrix.read_bytes()
        member_matrix.unlink()
        message = diagnostic_refusal(lambda: run(replace(config, write_figures=False)))
        check(
            "a missing member Hx matrix refuses the run by default",
            message is not None and "require_f_order_check" in message,
            message,
        )
        relaxed = replace(config, require_f_order_check=False, write_figures=False)
        relaxed_payload = run(relaxed)
        relaxed_suite = relaxed_payload["checks"]["f_order_numbering_member_hx"]
        check(
            "require_f_order_check=False proceeds but marks the pair UNVERIFIED",
            relaxed_suite["status"] == "incomplete_unverified_pairs"
            and relaxed_suite["all_verified"] is False
            and relaxed_suite["n_unverified"] == 1
            and relaxed_suite["n_verified"] == expected_member_pairs - 1
            and relaxed_suite["unverified"][0]["member"] == 1
            and relaxed_suite["unverified"][0]["lag_tag"] == tag_6h
            and relaxed_suite["per_lag"][tag_6h]["members_unverified"] == [1],
            str(relaxed_suite["unverified"]),
        )
        check(
            "the relaxed run warns that the member column was not verified",
            any(
                "member Hx matrix file(s) are absent" in warning
                and "require_f_order_check=False" in warning
                and "mem001" in warning
                for warning in relaxed_payload["warnings"]
            ),
            "; ".join(relaxed_payload["warnings"]),
        )
        check(
            "an unverified member pair is never counted as a member pass",
            relaxed_suite["all_verified"] is False,
        )
        member_matrix.write_bytes(backup_matrix)
        check(
            "restoring the matrix clears the unverified pair",
            run(replace(config, write_figures=False))["checks"][
                "f_order_numbering_member_hx"
            ]["all_verified"]
            is True,
        )

    # -- 12. a diagnostic subset that excludes the 0 h lag -------------------
    print("\n[12] a diagnostic subset without the 0 h lag: lag_hours = (3, 6)")
    with tempfile.TemporaryDirectory() as tmp:
        config = _build_synthetic_dataset(Path(tmp))
        center_tag = config.center_tag()
        # Move the 3 h soundings so the 3 h lag coordinate is NOT the center
        # coordinate.  If any position were taken from schedule[0] instead of the
        # center profile, the state point would move with it.
        _shift_profile_coordinates(profile_path(config, tag_3h), 0.30, 0.20)

        full = run(config)
        subset = run(replace(config, lag_hours=(3, 6), write_figures=False))
        center_lat, center_lon = SYNTHETIC_OBS_LATLON["66"]
        report = subset["observations"]["66"]
        rows = subset["_lag_rows"]["66"]
        row_3h = next(row for row in rows if int(row["lag_hours"]) == 3)

        check(
            "the (3, 6) subset runs and diagnoses exactly those two lags",
            bool(subset["observations"])
            and sorted(int(row["lag_hours"]) for row in rows) == [3, 6],
            str(sorted(int(row["lag_hours"]) for row in rows)),
        )
        check(
            "the state point still uses the CENTER-time position",
            math.isclose(report["state_lat"], center_lat, abs_tol=1e-9)
            and math.isclose(report["state_lon"], center_lon, abs_tol=1e-9)
            and not math.isclose(row_3h["obs_lat_at_lag"], center_lat, abs_tol=1e-6),
            f"state=({report['state_lat']}, {report['state_lon']}), "
            f"3 h lag coordinate=({row_3h['obs_lat_at_lag']}, {row_3h['obs_lon_at_lag']})",
        )
        check(
            "the sampled ocean prior is identical to the full-window run's",
            math.isclose(
                report["ocean_prior_mean"],
                full["observations"]["66"]["ocean_prior_mean"],
                rel_tol=0,
                abs_tol=1e-12,
            )
            and report["state_sample_lat"] == full["observations"]["66"]["state_sample_lat"],
            f"subset={report['ocean_prior_mean']!r}, full="
            f"{full['observations']['66']['ocean_prior_mean']!r}",
        )
        check(
            "the 3 h displacement is measured against the CENTER time",
            row_3h["obs_displacement_from_center_km"] > 1.0
            and f"{center_tag}->{tag_3h}" in subset["checks"]["profile_coordinate_displacement"],
            f"{row_3h['obs_displacement_from_center_km']:.3f} km",
        )
        check(
            "the single-obs loc3d is still checked against the CENTER profile",
            report["single_obs"]["location_check"] is not None
            and math.isclose(
                report["single_obs"]["location_check"]["profile_lat"],
                center_lat,
                abs_tol=1e-9,
            )
            and report["single_obs"]["location_check"]["distance_km"]
            <= report["single_obs"]["location_check"]["tolerance_km"],
        )
        averages = report["averages"]
        check(
            "the real-window quantities are UNAVAILABLE, naming the missing lag",
            averages["available"] is False
            and averages["d_L_NR"] is None
            and averages["d_L_actual"] is None
            and averages["noise_L"] is None
            and "missing lag(s) [0]" in averages["unavailable_reason"],
            str(averages["unavailable_reason"]),
        )
        check(
            "external_FO is unavailable, not compared against the subset",
            subset["checks"]["external_fo"]["66"]["status"] == "unavailable"
            and "missing lag(s) [0]" in subset["checks"]["external_fo"]["66"]["reason"],
            str(subset["checks"]["external_fo"]["66"].get("reason")),
        )
        center_check = subset["checks"]["f_order_numbering_center"]
        check(
            "the center F-order result is unavailable, not borrowed from 3 h",
            center_check["available"] is False
            and center_check["f_order_matches"] is None
            and center_check["enforcement"] == "not_computed_center_lag_not_diagnosed"
            and center_check["label"] == f"NR BT {center_tag}",
            str(center_check.get("enforcement")),
        )
        check(
            "the diagnosed lags keep their own NR entries",
            subset["checks"]["f_order_numbering_per_obs"]["66"][tag_3h]["label"]
            == f"NR BT {tag_3h}"
            and subset["checks"]["f_order_numbering_per_obs"]["66"][tag_3h]["enforcement"]
            == "passed",
        )
        check(
            "the member suite covers only the diagnosed lags",
            subset["checks"]["f_order_numbering_member_hx"]["n_member_lag_pairs"]
            == SYNTHETIC_MEMBERS * 2,
            str(subset["checks"]["f_order_numbering_member_hx"]["n_member_lag_pairs"]),
        )
        check(
            "the window rows cover only the diagnosed lags (0 h is not folded in)",
            [row["window_label"] for row in subset["_window_rows"]["66"]] == ["[3]", "[3,6]"]
            and all("0" not in row["window_label"] for row in subset["_window_rows"]["66"]),
            str([row["window_label"] for row in subset["_window_rows"]["66"]]),
        )
        check(
            "the schedule reports the window as only partly diagnosed",
            subset["lag_schedule"]["assimilated_window_fully_diagnosed"] is False
            and subset["lag_schedule"]["missing_assimilated_lag_hours"] == [0]
            and subset["lag_schedule"]["center_lag_hours"] == 0,
        )
        check(
            "the partial window is warned about, not silently accepted",
            any("NOT fully diagnosed" in warning for warning in subset["warnings"]),
            "; ".join(subset["warnings"]),
        )
        check(
            "the 3 h profile move is warned about, against the CENTER reference",
            any(
                f"relative to the center time {center_tag}" in warning
                for warning in subset["warnings"]
            ),
            "; ".join(subset["warnings"]),
        )

        # The center profile is required even though the 0 h lag is not diagnosed.
        center_profile = profile_path(config, center_tag)
        backup_profile = center_profile.read_bytes()
        center_profile.unlink()
        message = diagnostic_refusal(
            lambda: run(replace(config, lag_hours=(3, 6), write_figures=False))
        )
        check(
            "a missing CENTER profile is a clear MissingInputError for a (3,6) subset",
            message is not None
            and "center-time profile" in message
            and str(center_profile) in message,
            message,
        )
        center_profile.write_bytes(backup_profile)

    # -- 13. a member subset against the full stored external_FO -------------
    print("\n[13] a member subset is matched to external_FO by member number")
    with tempfile.TemporaryDirectory() as tmp:
        config = _build_synthetic_dataset(Path(tmp))
        subset_config = replace(
            config, member_start=2, member_end=4, write_figures=False
        )
        payload = run(subset_config)
        external = payload["checks"]["external_fo"]["66"]
        true_window_mean = SYNTHETIC_HX_BY_LAG_MEMBER.mean(axis=0)  # members 1..4

        check(
            "the run over members 2..4 succeeds",
            sorted(int(key) for key in payload["observations"]) == [66, 98]
            and payload["observations"]["66"]["hx_files_read"] == 9,
            f"hx_files_read={payload['observations']['66']['hx_files_read']}",
        )
        check(
            "external_FO passes and reports the members it compared",
            external["status"] == "passed"
            and external["compared_member_numbers"] == [2, 3, 4]
            and external["selected_indices_0based"] == [1, 2, 3]
            and external["declared_ensemble_size"] == 4
            and external["n_members"] == 3,
            str({key: external.get(key) for key in (
                "status", "compared_member_numbers", "selected_indices_0based",
                "declared_ensemble_size", "max_abs_difference_K")}),
        )
        check(
            "taking the FIRST three entries instead would fail (test is meaningful)",
            float(np.max(np.abs(true_window_mean[1:4] - true_window_mean[0:3])))
            > REFERENCE_EXTERNAL_FO_TOL_K,
            f"max|members 2..4 - members 1..3| = "
            f"{float(np.max(np.abs(true_window_mean[1:4] - true_window_mean[0:3]))):.4f} K",
        )
        row_0h = next(
            row for row in payload["_lag_rows"]["66"] if int(row["lag_hours"]) == 0
        )
        check(
            "the Hx really is members 2..4's, not members 1..3's",
            math.isclose(
                row_0h["hx_mean"],
                float(SYNTHETIC_HX_BY_LAG_MEMBER[0][1:4].mean()),
                rel_tol=1e-12,
            )
            and not math.isclose(
                row_0h["hx_mean"],
                float(SYNTHETIC_HX_BY_LAG_MEMBER[0][0:3].mean()),
                rel_tol=1e-12,
            ),
            f"hx_mean={row_0h['hx_mean']:.9f}, members 2..4 mean="
            f"{float(SYNTHETIC_HX_BY_LAG_MEMBER[0][1:4].mean()):.9f}",
        )
        # Re-read the file through the real parser: the entries compared must be
        # the stored 2nd..4th values, not the first three.
        reread = select_external_fo(
            parse_obs_seq_single(single_obs_path(config, "66")), [2, 3, 4]
        )
        check(
            "the compared entries are the stored 2nd..4th values",
            reread is not None
            and np.array_equal(reread.selected_indices, (1, 2, 3))
            and np.allclose(reread.selected_values, true_window_mean[1:4], rtol=0, atol=1e-9)
            and not np.allclose(reread.selected_values, true_window_mean[0:3], rtol=0, atol=1e-9),
            f"selected={np.round(reread.selected_values, 6).tolist()}",
        )

        obs_file = single_obs_path(config, "66")
        backup_obs = obs_file.read_text()

        # A requested member that the stored array does not contain.
        _rewrite_external_fo(obs_file, true_window_mean[:3], declared=3)
        message = diagnostic_refusal(lambda: run(subset_config))
        check(
            "a member outside the stored ensemble is refused with a clear error",
            message is not None and "member 4" in message,
            message,
        )

        # A declared count that disagrees with the stored values.
        _rewrite_external_fo(obs_file, true_window_mean, declared=3)
        message = diagnostic_refusal(lambda: run(subset_config))
        check(
            "a declared count smaller than the stored values is refused",
            message is not None and "disagree" in message,
            message,
        )
        _rewrite_external_fo(obs_file, true_window_mean, declared=50)
        message = diagnostic_refusal(lambda: run(subset_config))
        check(
            "a declared count larger than the stored values is refused",
            message is not None and "declares 50 members" in message,
            message,
        )

        # No reliable member count: reported unavailable rather than guessed.
        _rewrite_external_fo(obs_file, true_window_mean, declared=None)
        no_count = run(subset_config)
        no_count_check = no_count["checks"]["external_fo"]["66"]
        check(
            "a block with no member count is UNAVAILABLE, never guessed at",
            no_count_check["status"] == "unavailable"
            and "declares no member count" in no_count_check["reason"]
            and no_count["observations"]["66"]["single_obs"]["external_fo_members"] is None,
            str(no_count_check.get("reason")),
        )
        check(
            "the unavailable count is not quietly compared against the configured size",
            no_count_check.get("max_abs_difference_K") is None
            and no_count_check.get("compared_member_numbers") is None,
        )

        # No external_FO block at all: the documented optional-block handling.
        _remove_external_fo(obs_file)
        absent = run(subset_config)
        absent_check = absent["checks"]["external_fo"]["66"]
        check(
            "a missing external_FO block stays a non-fatal 'unavailable'",
            absent_check["status"] == "unavailable"
            and "no external_FO block" in absent_check["reason"]
            and bool(absent["observations"]),
            str(absent_check.get("reason")),
        )
        obs_file.write_text(backup_obs)
        check(
            "the run succeeds again once the obs_seq file is restored",
            run(subset_config)["checks"]["external_fo"]["66"]["status"] == "passed",
        )

    # -- 14. one-column files are checked by shape, not value count ----------
    print("\n[14] one-column files are checked by SHAPE, not just value count")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        column = np.arange(1.0, SYNTHETIC_NOBS + 1)
        good = root / "good.txt"
        np.savetxt(good, column, fmt="%.6f")
        scalar = root / "scalar.txt"
        np.savetxt(scalar, np.array([3.5]), fmt="%.6f")
        matrix_at_column_path = root / "matrix.txt"
        np.savetxt(
            matrix_at_column_path,
            column.reshape(SYNTHETIC_SIDE, SYNTHETIC_SIDE, order="F"),
            fmt="%.6f",
        )
        one_row = root / "one_row.txt"
        np.savetxt(one_row, column.reshape(1, -1), fmt="%.6f")
        two_columns = root / "two_columns.txt"
        np.savetxt(two_columns, np.column_stack([column, column]), fmt="%.6f")
        short = root / "short.txt"
        np.savetxt(short, column[:-1], fmt="%.6f")
        nonfinite = root / "nonfinite.txt"
        nonfinite.write_text(
            "\n".join(["1.0"] * 5 + ["nan"] + ["2.0"] * (SYNTHETIC_NOBS - 6)) + "\n"
        )

        def refusal_type(call) -> str:
            try:
                call()
            except DiagnosticError as exc:
                return type(exc).__name__
            except Exception as exc:  # noqa: BLE001
                return f"non-diagnostic {type(exc).__name__}"
            return "no exception"

        check(
            "a genuine one-column file reads back as 1D",
            read_column_file(good, SYNTHETIC_NOBS, "good").shape == (SYNTHETIC_NOBS,),
        )
        check(
            "a single-value file works with expected = 1",
            read_column_file(scalar, 1, "scalar").tolist() == [3.5],
        )
        matrix_message = diagnostic_refusal(
            lambda: read_column_file(matrix_at_column_path, SYNTHETIC_NOBS, "matrix")
        )
        check(
            "a 26x26 matrix holding the SAME 676 values is refused at the column path",
            matrix_message is not None and "2D matrix" in matrix_message,
            matrix_message,
        )
        check(
            "a single row of many columns is refused",
            refusal_type(lambda: read_column_file(one_row, SYNTHETIC_NOBS, "row"))
            == "ConsistencyError",
        )
        check(
            "a two-column file is refused",
            refusal_type(lambda: read_column_file(two_columns, SYNTHETIC_NOBS, "two"))
            == "ConsistencyError",
        )
        check(
            "a file with the wrong number of rows is refused",
            refusal_type(lambda: read_column_file(short, SYNTHETIC_NOBS, "short"))
            == "ConsistencyError",
        )
        check(
            "a non-finite value is still refused",
            refusal_type(lambda: read_column_file(nonfinite, SYNTHETIC_NOBS, "nonfinite"))
            == "ConsistencyError",
        )
        check(
            "a missing file is a MissingInputError",
            refusal_type(lambda: read_column_file(root / "absent.txt", SYNTHETIC_NOBS, "absent"))
            == "MissingInputError",
        )

        # End to end: the same 676 numbers, arranged as a matrix, hide the
        # F-order mismatch completely -- the flattening the reader would do in
        # C-order reproduces the column exactly, so only the shape check can
        # refuse it.
        config = _build_synthetic_dataset(root / "dataset")
        member_file = member_hx_path(config, 2, tag_3h)
        member_column = read_column_file(member_file, config.nobs_raw, "mem002 3h")
        backup_member = member_file.read_bytes()
        np.savetxt(
            member_file,
            member_column.reshape(SYNTHETIC_SIDE, SYNTHETIC_SIDE, order="C"),
            fmt="%.10f",
        )
        check(
            "the C-order reading of that matrix really is the expected column",
            np.array_equal(
                np.loadtxt(member_file, ndmin=2).reshape(-1), member_column
            ),
        )
        message = diagnostic_refusal(lambda: run(replace(config, write_figures=False)))
        check(
            "a 26x26 matrix at a member's one-column path refuses the run",
            message is not None and "has shape" in message and "mem002" in message,
            message,
        )
        member_file.write_bytes(backup_member)
        check(
            "the run succeeds again once the member column file is restored",
            bool(run(replace(config, write_figures=False))["observations"]),
        )

    # -- 15. non-finite locations, values and tolerances ---------------------
    print("\n[15] non-finite loc3d / obs value / tolerance are refused")
    location_record = ObsSeqRecord(
        path=Path("obs_seq.out_LACC_single_66"),
        obs_block_number=1,
        obs_value=251.0,
        lat=14.2751,
        lon=147.6832,
        external_fo=np.zeros(SYNTHETIC_MEMBERS),
        external_fo_member_count=SYNTHETIC_MEMBERS,
    )
    location_config = replace(DEFAULT_CONFIG, single_obs_coord_tol_km=1.0)
    check(
        "a well-formed location still passes",
        check_single_obs_location(
            location_config, location_record, 14.2751, 147.6832
        )["distance_km"]
        == 0.0,
    )
    check(
        "a longitude 360 deg away is the same point (documented convention)",
        check_single_obs_location(
            location_config, location_record, 14.2751, 147.6832 - 360.0
        )["distance_km"]
        <= location_config.single_obs_coord_tol_km,
    )
    # The expected fragment matters: a NaN also fails an ordered range test, so
    # without pinning the wording a lost finiteness guard would still "refuse"
    # -- with a message that blames the wrong thing.
    for name, bad_lat, bad_lon, expected in (
        ("a NaN latitude", float("nan"), 147.6832, "non-finite location"),
        ("a NaN longitude", 14.2751, float("nan"), "non-finite location"),
        ("an infinite latitude", float("inf"), 147.6832, "non-finite location"),
        ("an infinite longitude", 14.2751, float("-inf"), "non-finite location"),
        (
            "a latitude outside [-90, 90] (radians read as degrees)",
            147.6832,
            147.6832,
            "outside [-90, 90]",
        ),
        ("a longitude outside [-360, 360]", 14.2751, 4000.0, "outside [-360, 360]"),
    ):
        message = diagnostic_refusal(
            lambda lat=bad_lat, lon=bad_lon: check_single_obs_location(
                location_config, replace(location_record, lat=lat, lon=lon), 14.2751, 147.6832
            )
        )
        check(
            f"{name} is refused as such, with the file path in the message",
            message is not None
            and expected in message
            and str(location_record.path) in message,
            message,
        )
    message = diagnostic_refusal(
        lambda: check_single_obs_location(
            location_config,
            replace(location_record, obs_value=float("nan")),
            14.2751,
            147.6832,
        )
    )
    check(
        "a non-finite obs value is refused rather than reported as an innovation",
        message is not None and "not finite" in message,
        message,
    )
    for bad_tolerance in (float("nan"), float("inf"), -1.0):
        bad_config = replace(location_config, single_obs_coord_tol_km=bad_tolerance)
        check(
            f"a {bad_tolerance!r} km tolerance is refused by the location check",
            diagnostic_refusal(
                lambda cfg=bad_config: check_single_obs_location(
                    cfg, location_record, 14.2751, 147.6832
                )
            )
            is not None,
        )
        check(
            f"a {bad_tolerance!r} km tolerance is refused by validate_config",
            any_refusal(lambda cfg=bad_config: validate_config(cfg)) is not None,
        )

    with tempfile.TemporaryDirectory() as tmp:
        config = _build_synthetic_dataset(Path(tmp))
        obs_file = single_obs_path(config, "66")
        backup_obs = obs_file.read_text()

        _rewrite_loc3d_text(obs_file, "nan", "nan")
        message = diagnostic_refusal(lambda: run(replace(config, write_figures=False)))
        check(
            "a NaN loc3d refuses the FULL run, with the file path named",
            message is not None
            and "loc3d" in message
            and str(obs_file) in message,
            message,
        )
        obs_file.write_text(backup_obs)

        _rewrite_loc3d_text(obs_file, "1e400", "1e400")  # parses as +inf
        check(
            "an infinite loc3d refuses the full run",
            diagnostic_refusal(lambda: run(replace(config, write_figures=False))) is not None,
        )
        obs_file.write_text(backup_obs)

        _rewrite_obs_value_text(obs_file, "nan")
        check(
            "a NaN obs value refuses the full run",
            diagnostic_refusal(lambda: run(replace(config, write_figures=False))) is not None,
        )
        obs_file.write_text(backup_obs)

        check(
            "a NaN coordinate tolerance refuses the full run",
            any_refusal(
                lambda: run(
                    replace(
                        config,
                        single_obs_coord_tol_km=float("nan"),
                        write_figures=False,
                    )
                )
            )
            is not None,
        )
        check(
            "the run succeeds again once the obs_seq file is restored",
            bool(run(replace(config, write_figures=False))["observations"]),
        )

    print("\n" + "=" * 78)
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s)")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("SELF-TEST PASSED (code-level only; the server data was NOT exercised)")
    print("=" * 78)
    return 0


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LACC lag covariance vs. innovation-direction diagnostic.  Read-only: "
            "this script never writes into the experiment tree and never launches "
            "an assimilation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true",
                        help="run the synthetic code checks and exit")
    parser.add_argument("--project-root", type=Path, default=None,
                        help="override PROJECT_SERVER_ROOT (for a local data copy)")
    parser.add_argument("--hx-root", type=Path, default=None,
                        help="override {project_root}/3create_obs/hx_rttov")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--nr-file", type=Path, default=None)
    parser.add_argument("--background", choices=["obs_seq111", "mem_all_time"], default=None)
    parser.add_argument("--obs-seq111-dir", type=Path, default=None)
    parser.add_argument("--mem-all-time-dir", type=Path, default=None)
    parser.add_argument("--sample-mode", choices=["linear", "nearest"], default=None)
    parser.add_argument("--single-obs-coord-tol-km", type=float, default=None,
                        help="tolerance for matching the single-obs loc3d to the "
                             "selected profile row (default 1.0 km)")
    parser.add_argument("--allow-missing-times", action="store_true",
                        help="do not require background/NR Times to equal the center "
                             "time; use ONLY for files that genuinely have no Times "
                             "variable.  A PRESENT but wrong time is still refused.")
    parser.add_argument("--member-start", type=int, default=None,
                        help="first ensemble member (default 1)")
    parser.add_argument("--member-end", type=int, default=None,
                        help="last ensemble member (default 50); lower it when "
                             "running against a reduced local copy")
    parser.add_argument("--nobs-raw", type=int, default=None,
                        help="rows per one-column BT file (default 676)")
    parser.add_argument("--matrix-side", type=int, default=None,
                        help="side of the 2D BT matrix, must satisfy side**2 == nobs_raw")
    parser.add_argument("--lag-hours", type=int, nargs="*", default=None,
                        help="which ASSIMILATED lags to diagnose, e.g. '--lag-hours 3 6' "
                             "for a subset that excludes the 0 h center lag (default "
                             "0 3 6).  The real-window quantities and the external_FO "
                             "check are then reported unavailable; the center-time "
                             "profile is still read for the positions")
    parser.add_argument("--enable-extra-lags", action="store_true",
                        help="allow extra_lag_hours to be diagnosed")
    parser.add_argument("--extra-lag-hours", type=int, nargs="*", default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-reference-checks", action="store_true",
                        help="skip the hand-verified reference numbers; use ONLY "
                             "for synthetic or non-server data, never to silence a "
                             "mismatch on the real inputs")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict = {}
    if args.project_root is not None:
        overrides["project_root"] = args.project_root
    if args.hx_root is not None:
        overrides["hx_root"] = args.hx_root
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.nr_file is not None:
        overrides["nr_file"] = args.nr_file
    if args.background is not None:
        overrides["background_choice"] = args.background
    if args.obs_seq111_dir is not None:
        overrides["obs_seq111_dir"] = args.obs_seq111_dir
    if args.mem_all_time_dir is not None:
        overrides["mem_all_time_dir"] = args.mem_all_time_dir
    if args.sample_mode is not None:
        overrides["sample_mode"] = args.sample_mode
    if args.single_obs_coord_tol_km is not None:
        overrides["single_obs_coord_tol_km"] = args.single_obs_coord_tol_km
    if args.allow_missing_times:
        overrides["require_center_time_match"] = False
    if args.member_start is not None:
        overrides["member_start"] = args.member_start
    if args.member_end is not None:
        overrides["member_end"] = args.member_end
    if args.nobs_raw is not None:
        overrides["nobs_raw"] = args.nobs_raw
    if args.matrix_side is not None:
        overrides["matrix_side"] = args.matrix_side
    if args.lag_hours is not None:
        overrides["lag_hours"] = tuple(sorted(set(args.lag_hours)))
    if args.enable_extra_lags:
        overrides["enable_extra_lag_diagnostics"] = True
    if args.extra_lag_hours is not None:
        overrides["extra_lag_hours"] = tuple(sorted(set(args.extra_lag_hours)))
    if args.no_figures:
        overrides["write_figures"] = False
    if args.no_reference_checks:
        overrides["check_reference_values"] = False
    return replace(DEFAULT_CONFIG, **overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()

    config = config_from_args(args)
    try:
        payload = run(config)
    except DiagnosticError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as exc:
        # Configuration and parse errors carry an actionable message already;
        # a traceback would only bury it.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
