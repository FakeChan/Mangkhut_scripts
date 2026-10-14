#!/usr/bin/env python3
"""
filter_LACC_sst_direction.py -- NR-assisted SST update-direction screen for the
equal-weight LACC observation set.

WHAT THIS SCRIPT DOES
---------------------
The equal-weight LACC experiment assimilates a time-averaged AMSU-A channel
brightness temperature.  For every raw observation i this script asks one
question, using the background ensemble that the Hx came from and the nature
run:

    does the ensemble-implied update of the background SST MEAN point the same
    way as the NR prior error at that same location?

With

    x_m   = center-time OM_TMP level 0 of background member m at the obs point
    h_m   = member m's final equal-weight LACC Hx value (the generated file)
    y     = the final LACC observation (already perturbed with noise)
    x_NR  = center-time NR OM_TMP level 0 at the same obs point

the criterion is

    x_bar = mean_m(x_m)
    h_bar = mean_m(h_m)
    c     = sum_m( (x_m - x_bar) * (h_m - h_bar) ) / (M - 1)
    d     = y - h_bar
    e     = x_NR - x_bar
    score = c * d * e

    score <  0  ->  the update direction is opposite to the NR error: DELETE
    score >= 0  ->  not opposite: KEEP

and the published mask is

    final_mask = original_clear_sky_mask AND direction_keep

so the original clear-sky screen stays in force and is never weakened.

THIS IS A DIRECTION TEST, NOT AN ERROR-REDUCTION TEST
-----------------------------------------------------
The single-observation ensemble update in ocean space is approximately

    Delta T_o  ~  c / (sigma_hx^2 + R) * d

so sign(Delta T_o) = sign(c * d) because the denominator is positive.  Comparing
that sign against the correction the prior actually needs (e = x_NR - x_bar)
gives the direction proxy c * d * e.  It says NOTHING about magnitude:

  * an overshooting but correctly directed update is KEPT -- no overshoot,
    amplitude, correlation-threshold or significance test is applied here;
  * this is a single-observation proxy for the LINEAR MEAN increment of the
    background ensemble.  It does not guarantee the direction a serial
    multi-observation assimilation actually produces, and it is NOT claimed to
    be equivalent to the update of a nonlinear or quantile/adaptive filter;
  * it is a STATIC screen.  It is computed once, from one common prior, before
    the assimilation.  Nothing is recomputed when the background would change
    inside a serial assimilation (that is deliberately out of scope);
  * SST here is strictly OM_TMP[0, 0, :, :] (first time, level 0) in BOTH the
    background members and the NR file.  It is never TSK and never another SST
    variable from the same file.

WHAT IS REUSED, AND WHAT IS NOT
-------------------------------
Path construction, the strict text readers, the F-order matrix gate, the netCDF
level-0 reader, the point sampler and the background/NR consistency checks all
come from ``plot_scripts/diag_LACC_lag_cov_innovation.py``, which is loaded by
ABSOLUTE script path (the shell driver runs from an unrelated working
directory).  Nothing in that diagnostic is modified.  The observation
coordinates come from ``read_profile_coordinates()``, so the profile file is
parsed by the existing marker-based reader and the raw observation order is
preserved.

The observation file is the SAME file the downstream converter reads: the
driver resolves ``OBS_INPUT_FILE`` once and hands it to both, so the screen can
never certify one file while the converter runs on another.

INPUT DISCIPLINE
----------------
* Every background member file is opened once for its level-0 field; all
  selected observations are then sampled from that in-memory grid.  The units
  attribute costs one additional, metadata-only open per file (the shared
  reader owns its own file handle).  Files are never re-opened per observation.
* The NR file is read once as well.
* Background members must agree with each other on the grid and the water mask;
  the NR grid is explicitly allowed to differ (it is sampled at the same
  longitudes/latitudes, never at the same array indices).
* Every member and the NR must carry Times == center_time.
* Sampling is linear, water points only, local window 1 deg, never more than
  4 deg, and it NEVER extrapolates or falls back to a nearest neighbour.
* For a raw observation that the original clear-sky mask keeps, a non-water
  point, an impossible interpolation, a missing member file/matrix, a wrong
  valid time, a grid mismatch, a non-Kelvin unit or an F-order mismatch is a
  hard refusal naming the path, the member number and the raw observation
  number.  Nothing is silently kept, dropped or replaced.

OUTPUTS (all in one directory, written atomically)
--------------------------------------------------
    clear_sky_and_nr_sst_direction_mask.txt   fixed nobs rows of 0/1
    nr_sst_direction_audit.csv                one row per RAW observation
    nr_sst_direction_summary.json            strict JSON (allow_nan=False)

The inputs are checked and the whole screen is computed BEFORE anything is
published: each file is written to a temporary name in the target directory and
then moved into place, so a failure cannot leave a half-written mask that a
later stage would happily consume.  The original observation file, the member
Hx files, the original clear-sky mask and every netCDF input are only ever read.

If EVERY observation is deleted by the direction test, the audit, the summary
and the (all-zero) mask are still written -- that is the record of what happened
-- and ``run()`` then raises ``NoObservationSurvivesError``, so the driver stops
before text_to_obs instead of converting an empty observation set.  The mask is
published in that case on purpose: it is the truthful result for those inputs,
and leaving the previous run's usable mask in place would be worse.

STANDALONE USE
--------------
Edit ``CONFIG`` below and run the file; there are no command-line arguments.
The shell driver does NOT use this entry point: it builds a ``Config`` in a
short bridge and calls ``run(config)``, so the module itself never depends on
implicit environment variables.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import netCDF4 as nc  # noqa: E402
import numpy as np

# matplotlib must see a writable cache dir before it is imported: the companion
# diagnostic imports it, and the compute nodes have a small HOME quota.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"),
)


# =============================================================================
# The companion diagnostic, loaded by ABSOLUTE path (never through the CWD)
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_MODULE_PATH = REPO_ROOT / "plot_scripts" / "diag_LACC_lag_cov_innovation.py"

# Private module name for the by-path import.  It is NOT the name the
# diagnostic uses for itself, so this import can never be mistaken for the
# ``import diag_LACC_lag_cov_innovation`` a user might have done.
DIAGNOSTIC_MODULE_NAME = "_lacc_sst_direction_companion_diagnostic"


def _load_diagnostic_module():
    """Import the companion diagnostic by path, reusing an existing import.

    Every reused reader/sampler/validator lives in that file.  Loading it by
    absolute path keeps this script independent of the working directory, and
    an already-imported copy of the SAME file is reused so that a test (or a
    caller) importing both modules still sees a single definition of every
    class -- in particular exactly one ``Config``.
    """
    target = DIAGNOSTIC_MODULE_PATH.resolve()
    if not target.is_file():
        raise ImportError(
            f"the companion diagnostic is missing: {target}.  This filter reuses "
            f"its readers, sampler and consistency checks and does not carry a "
            f"second copy of them."
        )
    for name in (
        "plot_scripts.diag_LACC_lag_cov_innovation",
        "diag_LACC_lag_cov_innovation",
    ):
        existing = sys.modules.get(name)
        existing_path = getattr(existing, "__file__", None) if existing is not None else None
        if existing_path and Path(existing_path).resolve() == target:
            return existing

    spec = importlib.util.spec_from_file_location(DIAGNOSTIC_MODULE_NAME, target)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot import the companion diagnostic: {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[DIAGNOSTIC_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


lag = _load_diagnostic_module()


# =============================================================================
# Refusals
# =============================================================================
class DirectionFilterError(lag.DiagnosticError):
    """Every refusal raised by this filter.

    It derives from the companion diagnostic's ``DiagnosticError`` so that a
    caller (the driver's bridge) can catch the shared readers' refusals and this
    script's refusals with ONE exception class.
    """


class NoObservationSurvivesError(DirectionFilterError):
    """The direction test deleted every observation.

    Carries the published ``payload`` so the caller can report the audit paths
    and the counts even though the run must stop.
    """

    def __init__(self, message: str, payload: dict) -> None:
        super().__init__(message)
        self.payload = payload


# =============================================================================
# Configuration
# =============================================================================
# Fixed output names (the shell driver refers to the mask by this exact name).
MASK_FILENAME = "clear_sky_and_nr_sst_direction_mask.txt"
AUDIT_FILENAME = "nr_sst_direction_audit.csv"
SUMMARY_FILENAME = "nr_sst_direction_summary.json"

# Status vocabulary of the audit file.  These four strings are the only values
# the ``status`` column ever carries.
STATUS_EXCLUDED_BY_ORIGINAL_MASK = "excluded_by_original_mask"
STATUS_REJECTED_REVERSE = "rejected_reverse"
STATUS_KEPT_NONREVERSE = "kept_nonreverse"
STATUS_KEPT_ZERO_DIRECTION = "kept_zero_direction"

# The audit columns, in the published order.  Listed explicitly so the header
# is identical for every run and no column can go missing.
AUDIT_COLUMNS = (
    "raw_obs_id_1based",
    "base_filtered_id_1based",
    "final_filtered_id_1based",
    "latitude",
    "longitude",
    "original_mask",
    "status",
    "sst_background_mean",
    "sst_nr",
    "sst_error_nr_minus_background",
    "hx_mean",
    "actual_observation",
    "innovation",
    "cov_sst_hx",
    "direction_score",
    "keep_final",
)

# Units strings accepted as "temperature in kelvin".  Both verified files
# (firstguess_d01.mem001 and the NR wrfout) declare OM_TMP:units = "k".
KELVIN_UNIT_TOKENS = frozenset(
    {
        "k",
        "kelvin",
        "kelvins",
        "degk",
        "deg_k",
        "deg_kelvin",
        "deg_kelvins",
        "degree_k",
        "degree_kelvin",
        "degree_kelvins",
        "degrees_k",
        "degrees_kelvin",
        "degrees_kelvins",
    }
)

SERVER_PROJECT_ROOT = lag.SERVER_PROJECT_ROOT


@dataclass(frozen=True)
class Config:
    """Every input this filter reads.  Nothing here is implicit.

    The defaults are the deployed server layout.  ``project_root`` plus the
    member pattern and the NR path are the ONLY places the background and the
    nature run are named: the script never searches for another firstguess or
    preassim file on its own, because a silent substitution would change every
    number the screen produces.
    """

    # ---------------------------------------------------------------- roots
    project_root: Path = SERVER_PROJECT_ROOT
    hx_root: Path | None = None  # default: {project_root}/3create_obs/hx_rttov
    output_dir: Path = (
        Path(__file__).resolve().parent / "single_assim_test" / "nr_sst_direction_filter"
    )

    # ------------------------------------------------------ experiment shape
    center_time: str = "2018-09-10_00:00:00"
    domain: str = "d01"
    sensor: str = "AMSUA"
    channel: int = 4
    member_start: int = 1
    member_end: int = 50
    nobs_raw: int = 676
    matrix_side: int = 26  # obs_{domain}_ch{channel}.txt is matrix_side**2 == nobs_raw
    # The assimilated window the DRIVER was configured with (LACC_LAG_HOURS).
    # The times file must agree with it; the comparison does not assume any
    # particular number of times.
    expected_lag_hours: tuple[int, ...] = (0, 3, 6)

    # ---------------------------------------------------------- observations
    obs_input_file: Path | None = None  # default: the unified *_withpert file
    clear_sky_mask_file: Path | None = None  # default: the unified center-time mask
    profile_file: Path | None = None  # default: the center-time LACC profile
    lacc_times_file: Path | None = None  # default: the center-time LACC times

    # ----------------------------------------------------------- background
    background_dir: Path | None = None  # default: {project_root}/4assimilation/0mem_all_time/cyclingDA/{tag}
    background_pattern: str = "firstguess_{domain}.mem{mem:03d}"

    # ------------------------------------------------------------------- NR
    nr_file: Path | None = None  # default: {project_root}/NR_wrfout/2domain/wrfout_d02_{center_time}

    # --------------------------------------------------------- state variable
    # SST is DEFINED here: first time index, level 0 of this variable.  The
    # 30-layer variable is never read whole.
    state_var: str = "OM_TMP"
    state_level: int = 0
    lat_var: str = "XLAT"
    lon_var: str = "XLONG"
    water_mask_source: str = "auto"  # LANDMASK (0 = water) preferred, XLAND (2 = water) otherwise
    sample_mode: str = "linear"  # no nearest-neighbour fallback exists
    sample_require_water: bool = True
    local_window_deg: float = 1.0
    max_local_window_deg: float = 4.0
    require_center_time_match: bool = True

    # ------------------------------------------------------------- derived
    def diagnostic_inputs(self) -> "lag.Config":
        """The companion diagnostic's Config, for every shared derivation.

        Path construction, the F-order gate, the netCDF reader, the point
        sampler and the background/NR consistency check all take this object,
        so this filter cannot drift from the validated implementations.  The
        F-order gate is deliberately hard-wired ON: this task adds no switch
        that would relax it.
        """
        return lag.Config(
            project_root=Path(self.project_root),
            hx_root=None if self.hx_root is None else Path(self.hx_root),
            center_time=self.center_time,
            domain=self.domain,
            sensor=self.sensor,
            channel=self.channel,
            member_start=self.member_start,
            member_end=self.member_end,
            nobs_raw=self.nobs_raw,
            matrix_side=self.matrix_side,
            state_var=self.state_var,
            state_level=self.state_level,
            lat_var=self.lat_var,
            lon_var=self.lon_var,
            water_mask_source=self.water_mask_source,
            sample_mode=self.sample_mode,
            sample_require_water=self.sample_require_water,
            local_window_deg=self.local_window_deg,
            max_local_window_deg=self.max_local_window_deg,
            require_center_time_match=self.require_center_time_match,
            require_f_order_check=True,
            write_figures=False,
        )

    def center_tag(self) -> str:
        return self.diagnostic_inputs().center_tag()

    def center_datetime(self):
        return self.diagnostic_inputs().center_datetime()

    def member_numbers(self) -> list[int]:
        return list(range(self.member_start, self.member_end + 1))

    def resolved_obs_input_file(self) -> Path:
        if self.obs_input_file is not None:
            return Path(self.obs_input_file)
        return lag.unified_noisy_bt_path(self.diagnostic_inputs())

    def resolved_clear_sky_mask_file(self) -> Path:
        if self.clear_sky_mask_file is not None:
            return Path(self.clear_sky_mask_file)
        return self.diagnostic_inputs().resolved_unified_mask()

    def resolved_profile_file(self) -> Path:
        if self.profile_file is not None:
            return Path(self.profile_file)
        return lag.profile_path(self.diagnostic_inputs(), self.center_tag())

    def resolved_lacc_times_file(self) -> Path:
        if self.lacc_times_file is not None:
            return Path(self.lacc_times_file)
        return self.diagnostic_inputs().resolved_lacc_times()

    def resolved_background_dir(self) -> Path:
        if self.background_dir is not None:
            return Path(self.background_dir)
        return (
            Path(self.project_root)
            / "4assimilation"
            / "0mem_all_time"
            / "cyclingDA"
            / self.center_tag()
        )

    def background_member_path(self, member: int) -> Path:
        return self.resolved_background_dir() / self.background_pattern.format(
            domain=self.domain, mem=member
        )

    def resolved_nr_file(self) -> Path:
        if self.nr_file is not None:
            return Path(self.nr_file)
        return self.diagnostic_inputs().resolved_nr_file()

    def member_hx_dir(self, member: int) -> Path:
        """The member directory holding the FINAL equal-weight LACC Hx.

        Note the ``BT_LACC_<centre>`` directory name: that is the time-averaged
        Hx the merge converter reads (``ENS_BT_SUBDIR=BT_LACC_<centre>`` in the
        driver).  The companion diagnostic's ``member_hx_path`` deliberately
        points at the per-lag ``BT_<lag>`` directories instead, which are a
        DIFFERENT file set; re-averaging those here would produce a different
        Hx from the one the assimilation actually consumes.
        """
        return (
            self.diagnostic_inputs().resolved_ens_bt_dir()
            / f"mem{member:03d}"
            / self.sensor
            / f"BT_LACC_{self.center_tag()}"
        )

    def member_hx_path(self, member: int) -> Path:
        return self.member_hx_dir(member) / f"obs_{self.domain}_ch{self.channel}_totalline.txt"

    def member_matrix_path(self, member: int) -> Path:
        """The member's own 2D Hx matrix, next to its own one-column file."""
        return self.member_hx_dir(member) / f"obs_{self.domain}_ch{self.channel}.txt"

    def output_paths(self) -> dict[str, Path]:
        directory = Path(self.output_dir)
        return {
            "mask": directory / MASK_FILENAME,
            "audit_csv": directory / AUDIT_FILENAME,
            "summary_json": directory / SUMMARY_FILENAME,
        }


# ===================== User configuration =====================
# Edit this block for a standalone run; the shell driver builds its own Config.
CONFIG = Config()
# =================== End user configuration ===================


def validate_config(config: Config) -> None:
    """Refuse anything that cannot produce an unambiguous screen."""
    fixed_settings = {
        "state_var": "OM_TMP", "state_level": 0, "sample_mode": "linear",
        "sample_require_water": True, "require_center_time_match": True,
    }
    for name, expected in fixed_settings.items():
        if getattr(config, name) != expected:
            raise DirectionFilterError(
                f"{name} must be {expected!r} for this SST direction filter, "
                f"got {getattr(config, name)!r}"
            )
    lag.validate_config(config.diagnostic_inputs())  # centre time, nobs, matrix side, members, mode

    if config.member_end - config.member_start + 1 < 2:
        raise DirectionFilterError(
            f"the direction criterion needs at least 2 members (ddof = 1); "
            f"members {config.member_start}..{config.member_end} give "
            f"{config.member_end - config.member_start + 1}"
        )
    if not config.expected_lag_hours:
        raise DirectionFilterError("expected_lag_hours must name the assimilated window")
    bad = [value for value in config.expected_lag_hours if int(value) < 0]
    if bad:
        raise DirectionFilterError(f"expected_lag_hours must be non-negative, got {bad}")
    if len(set(int(value) for value in config.expected_lag_hours)) != len(config.expected_lag_hours):
        raise DirectionFilterError(
            f"expected_lag_hours repeats a lag: {config.expected_lag_hours}"
        )
    if int(config.channel) < 1:
        raise DirectionFilterError(f"channel must be positive, got {config.channel}")
    if not str(config.domain).strip() or not str(config.sensor).strip():
        raise DirectionFilterError("domain and sensor must be non-empty")
    # A pattern that cannot be formatted would fail much later, on the first
    # member file, with a KeyError that names neither the member nor the cause.
    try:
        config.background_pattern.format(domain=config.domain, mem=config.member_start)
    except (KeyError, IndexError, ValueError) as exc:
        raise DirectionFilterError(
            f"background_pattern {config.background_pattern!r} cannot be formatted "
            f"with domain/mem: {exc}"
        ) from exc

    resolved_paths: dict[Path, int] = {}
    file_identities: dict[tuple[int, int], int] = {}
    for member in config.member_numbers():
        path = config.background_member_path(member)
        resolved = path.resolve()
        previous = resolved_paths.get(resolved)
        identity = None
        if path.is_file():
            metadata = path.stat()
            identity = (metadata.st_dev, metadata.st_ino)
            if previous is None:
                previous = file_identities.get(identity)
        if previous is not None:
            raise DirectionFilterError(
                f"duplicate background file for members {previous:03d} and "
                f"{member:03d}: {config.background_member_path(previous)} and {path}"
            )
        resolved_paths[resolved] = member
        if identity is not None:
            file_identities[identity] = member


# =============================================================================
# The direction criterion (a pure function, so the formula is testable alone)
# =============================================================================
@dataclass(frozen=True)
class DirectionResult:
    n_members: int
    sst_background_mean: float
    hx_mean: float
    cov_sst_hx: float
    innovation: float
    sst_error: float
    score: float
    reverse: bool
    status: str
    keep: bool

    def as_dict(self) -> dict:
        return {
            "n_members": self.n_members,
            "sst_background_mean": self.sst_background_mean,
            "hx_mean": self.hx_mean,
            "cov_sst_hx": self.cov_sst_hx,
            "innovation": self.innovation,
            "sst_error": self.sst_error,
            "score": self.score,
            "reverse": self.reverse,
            "status": self.status,
            "keep": self.keep,
        }


def _sign(value: float) -> int:
    """-1, 0 or +1 -- exact, and never a thresholded comparison."""
    return int(value > 0.0) - int(value < 0.0)


def evaluate_direction(
    sst_background: np.ndarray,
    hx: np.ndarray,
    observation: float,
    sst_nr: float,
) -> DirectionResult:
    """The static direction criterion for ONE observation.

    ``sst_background`` and ``hx`` are MEMBER vectors, paired strictly by member
    number (entry k of both belongs to the same member).  Members are used as
    given: a non-finite value is refused here rather than dropped, because
    dropping one member from x but not from h would break the pairing that the
    covariance is built on.

    The decision is taken from the SIGNS of c, d and e after a finiteness check,
    not from the product: c, d and e can all be tiny but nonzero (e.g. three
    values near 1e-200), in which case ``c * d * e`` underflows to exactly 0.0
    and a "score < 0" test would silently KEEP a reverse-direction observation.
    The product is still computed and reported as ``score``, as a diagnostic.

    A zero factor keeps the observation, as configured: c = 0, d = 0 and e = 0
    all mean "no direction is defined", and this screen deletes only what points
    the wrong way.
    """
    x = np.asarray(sst_background, dtype=np.float64).reshape(-1)
    h = np.asarray(hx, dtype=np.float64).reshape(-1)
    if x.shape != h.shape:
        raise DirectionFilterError(
            f"the SST background vector {x.shape} and the Hx vector {h.shape} must "
            f"have one entry per member"
        )
    n_members = int(x.size)
    if n_members < 2:
        raise DirectionFilterError(
            f"the direction criterion needs at least 2 members, got {n_members}"
        )
    if not np.all(np.isfinite(x)):
        raise DirectionFilterError("the SST background member values must all be finite")
    if not np.all(np.isfinite(h)):
        raise DirectionFilterError("the member Hx values must all be finite")
    y = float(observation)
    x_nr = float(sst_nr)
    if not math.isfinite(y):
        raise DirectionFilterError(f"the observation {y!r} is not finite")
    if not math.isfinite(x_nr):
        raise DirectionFilterError(f"the NR SST {x_nr!r} is not finite")

    # Identical members must give EXACTLY zero deviations.  The mean of k copies
    # of a decimal Kelvin value is not always that value in binary floating
    # point (e.g. 250.0001 is not representable), so subtracting the computed
    # mean would leave a rounding residual and manufacture a covariance that the
    # ensemble does not have.
    x_constant = bool(np.all(x == x[0]))
    h_constant = bool(np.all(h == h[0]))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            x_bar = float(x[0]) if x_constant else float(np.mean(x))
            h_bar = float(h[0]) if h_constant else float(np.mean(h))
            dx = np.zeros(n_members, dtype=np.float64) if x_constant else x - x_bar
            dh = np.zeros(n_members, dtype=np.float64) if h_constant else h - h_bar
            cov = float(np.sum(dx * dh) / (n_members - 1))
    except FloatingPointError as exc:
        raise DirectionFilterError(f"non-finite mean/covariance calculation: {exc}") from exc

    innovation = y - h_bar
    sst_error = x_nr - x_bar
    score = cov * innovation * sst_error

    for name, value in (
        ("sst_background_mean", x_bar), ("hx_mean", h_bar),
        ("cov_sst_hx", cov), ("innovation", innovation),
        ("sst_error", sst_error), ("direction_score", score),
    ):
        if not math.isfinite(value):
            raise DirectionFilterError(f"computed {name} is non-finite: {value!r}")

    signs = (_sign(cov), _sign(innovation), _sign(sst_error))
    n_negative = sum(1 for sign in signs if sign < 0)
    has_zero = any(sign == 0 for sign in signs)
    # c * d * e < 0 requires an ODD number of negative factors AND no zero
    # factor.  Spelled out here because the product itself can underflow.
    reverse = (not has_zero) and (n_negative % 2 == 1)

    if has_zero:
        status = STATUS_KEPT_ZERO_DIRECTION
    elif reverse:
        status = STATUS_REJECTED_REVERSE
    else:
        status = STATUS_KEPT_NONREVERSE

    return DirectionResult(
        n_members=n_members,
        sst_background_mean=x_bar,
        hx_mean=h_bar,
        cov_sst_hx=cov,
        innovation=innovation,
        sst_error=sst_error,
        score=score,
        reverse=reverse,
        status=status,
        keep=not reverse,
    )


# =============================================================================
# Strict readers used by this filter
# =============================================================================
def read_temperature_units(path: Path, variable: str, label: str) -> str:
    """Confirm the state variable is a temperature in kelvin.

    Only the attribute is read here -- no data array is touched.  The shared
    level-0 reader opens the file itself, so this is one additional,
    metadata-only open per file; the FIELD is read exactly once and every
    observation is sampled from that in-memory grid, never by re-opening the
    file per observation.

    A missing attribute is a refusal, not an assumption: "no units" cannot be
    distinguished from "not kelvin", and a Celsius field would make every
    temperature difference in this screen wrong.
    """
    if not path.is_file():
        raise DirectionFilterError(f"{label} state file does not exist: {path}")
    with nc.Dataset(path) as dataset:
        if variable not in dataset.variables:
            raise DirectionFilterError(f"{path} is missing the variable {variable}")
        raw_units = getattr(dataset.variables[variable], "units", None)
    if raw_units is None:
        raise DirectionFilterError(
            f"{label}: {path} declares no units attribute on {variable}.  This "
            f"screen needs a temperature in kelvin and refuses to assume one; "
            f"the verified background and NR files declare units = 'k'."
        )
    text = str(raw_units)
    token = text.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in token:
        token = token.replace("__", "_")
    if token not in KELVIN_UNIT_TOKENS:
        raise DirectionFilterError(
            f"{label}: {path} declares {variable}:units = {text!r}, which this "
            f"screen does not recognise as kelvin (accepted: "
            f"{', '.join(sorted(KELVIN_UNIT_TOKENS))}).  Temperature differences "
            f"in a non-kelvin unit would make the direction test meaningless."
        )
    return text


def check_coordinates(coordinates: "lag.ProfileCoordinates") -> dict:
    """Finiteness and range checks on the center-time observation locations."""
    lat = np.asarray(coordinates.lat, dtype=float)
    lon = np.asarray(coordinates.lon, dtype=float)
    if lat.shape != lon.shape:
        raise DirectionFilterError(
            f"{coordinates.path}: latitude {lat.shape} and longitude {lon.shape} "
            f"disagree"
        )
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        bad = int(np.flatnonzero(~(np.isfinite(lat) & np.isfinite(lon)))[0]) + 1
        raise DirectionFilterError(
            f"{coordinates.path}: raw observation {bad} has a non-finite "
            f"coordinate (lat {lat[bad - 1]!r}, lon {lon[bad - 1]!r})"
        )
    outside = np.flatnonzero((lat < -90.0) | (lat > 90.0))
    if outside.size:
        index = int(outside[0])
        raise DirectionFilterError(
            f"{coordinates.path}: raw observation {index + 1} has latitude "
            f"{lat[index]!r} deg, outside [-90, 90]"
        )
    far = np.flatnonzero((lon < -360.0) | (lon > 360.0))
    if far.size:
        index = int(far[0])
        raise DirectionFilterError(
            f"{coordinates.path}: raw observation {index + 1} has longitude "
            f"{lon[index]!r} deg, outside [-360, 360]"
        )
    return {
        "file": str(coordinates.path),
        "n_points": int(lat.size),
        "latitude_min": float(lat.min()),
        "latitude_max": float(lat.max()),
        "longitude_min": float(lon.min()),
        "longitude_max": float(lon.max()),
        "reader": "diag_LACC_lag_cov_innovation.read_profile_coordinates (marker based)",
        "order": "raw observation order, never sorted",
    }


def check_window(config: Config, source: str, lags: tuple[str, ...]) -> dict:
    """Reconcile the LACC times file with the driver's window.

    The centre time, the absence of a repeated lag and the exact agreement with
    the window the driver was configured for are all checked.  Nothing assumes
    three times: any number of distinct, non-negative lag hours is accepted as
    long as the FILE and the CONFIGURATION agree.
    """
    times_path = config.resolved_lacc_times_file()
    expected_tag = config.center_tag()
    if source != expected_tag:
        raise DirectionFilterError(
            f"{times_path} declares center_time={source}, which disagrees with the "
            f"configured centre {expected_tag} ({config.center_time})"
        )

    center_dt = config.center_datetime()
    hhmm = []
    for tag in lags:
        try:
            moment = lag.tag_to_datetime(tag, center_dt)
            hours = lag.lag_hours_between(center_dt, moment)
        except ValueError as exc:
            raise DirectionFilterError(f"{times_path}: lag_time={tag}: {exc}") from exc
        hhmm.append((hours, tag))
    hours_only = [hours for hours, _ in hhmm]
    if len(set(hours_only)) != len(hours_only):
        raise DirectionFilterError(
            f"{times_path}: two lag times resolve to the same lag hour: {hhmm}"
        )

    configured = sorted(int(value) for value in config.expected_lag_hours)
    if sorted(hours_only) != configured:
        raise DirectionFilterError(
            f"{times_path} describes the window {sorted(hours_only)} h but the driver "
            f"was configured for {configured} h (LACC_LAG_HOURS).  The SAME window must "
            f"produce the Hx and be screened here; refusing to screen against a "
            f"different set of times."
        )
    return {
        "file": str(times_path),
        "center_time_file": source,
        "center_time_configured": expected_tag,
        "lag_tags": list(lags),
        "lag_hours": sorted(hours_only),
        "expected_lag_hours": configured,
        "n_lag_times": len(hours_only),
        "lag_tags_unique": True,
        "reader": "diag_LACC_lag_cov_innovation.read_times_file",
    }


def sample_or_refuse(
    grid: "lag.GridField",
    latitude: float,
    longitude: float,
    origin: str,
    raw_index: int,
    config: Config,
) -> float:
    """Sample one point, or refuse with the path, the origin and the raw obs id."""
    try:
        # The shared sampler drops NaN cells before triangulating.  Retain the
        # geometry here and first interpolate a missing-value indicator: its
        # positive value means a missing water vertex has a positive weight at
        # this observation.  Remote missing cells do not invalidate the point.
        missing_water = grid.water & ~np.isfinite(grid.field)
        sampling_grid = grid
        if np.any(missing_water):
            distances = lag.haversine_km(grid.lat, grid.lon, latitude, longitude)
            nearest = int(np.argmin(np.where(np.isfinite(distances), distances, np.inf)))
            if missing_water.flat[nearest]:
                raise DirectionFilterError(
                    f"raw observation {raw_index}: {origin} ({grid.path}) has a "
                    f"non-finite temperature at its nearest water cell"
                )
            indicator_grid = replace(grid, field=missing_water.astype(np.float64))
            indicator = lag.sample_point(
                indicator_grid, latitude, longitude, config.sample_mode,
                config.diagnostic_inputs(),
            )
            if indicator.value > 0.0:
                raise DirectionFilterError(
                    f"raw observation {raw_index}: {origin} ({grid.path}) has a "
                    f"non-finite temperature in the required interpolation support"
                )
            # Missing vertices have been proved to carry zero weight here.
            # Keeping them in the triangulation prevents a different stencil
            # from being constructed after their exclusion.
            sampling_grid = replace(grid, field=np.where(missing_water, 0.0, grid.field))
        sample = lag.sample_point(
            sampling_grid, latitude, longitude, config.sample_mode, config.diagnostic_inputs()
        )
    except lag.SamplingError as exc:
        raise DirectionFilterError(
            f"raw observation {raw_index} at ({latitude:.4f}, {longitude:.4f}): "
            f"{origin} could not be sampled: {exc}"
        ) from exc
    if not math.isfinite(sample.value):
        raise DirectionFilterError(
            f"raw observation {raw_index} at ({latitude:.4f}, {longitude:.4f}): "
            f"{origin} ({grid.path}) sampled to a non-finite value"
        )
    return float(sample.value)


# =============================================================================
# Inputs
# =============================================================================
@dataclass
class OriginalMask:
    path: Path
    values: np.ndarray  # int8, 0/1, raw observation order
    kept_raw_ids: np.ndarray  # 1-based raw indices of the rows the ORIGINAL mask keeps


def read_original_mask(config: Config) -> OriginalMask:
    """The original clear-sky mask -- read only, never written, never replaced.

    Every run starts from THIS file.  A previously merged mask is never used as
    an input, so a second run cannot compound the first one's deletions.
    """
    path = config.resolved_clear_sky_mask_file()
    info = lag.read_mask(path, config.nobs_raw, config.center_tag())
    values = np.asarray(info.values, dtype=np.int8)
    kept = np.flatnonzero(values) + 1
    if kept.size == 0:
        raise DirectionFilterError(
            f"the original clear-sky mask {path} keeps no observation at all, so "
            f"there is nothing to screen"
        )
    return OriginalMask(path=path, values=values, kept_raw_ids=kept)


@dataclass
class BackgroundSamples:
    values: np.ndarray  # (n_members, n_selected) float64
    member_times: dict[int, "str | None"]
    reference_member: int
    grid_report: dict
    units: dict[str, str]


def sample_background(
    config: Config,
    nr_grid: "lag.GridField",
    places: list[tuple[int, float, float]],
) -> BackgroundSamples:
    """Sample every background member at every selected observation point.

    ONE open per member file: the level-0 field is read once and all selected
    observations are sampled from that in-memory grid.  Member grids are
    compared against the first member's grid with the companion diagnostic's
    ``validate_background_consistency`` (which also checks that the member and
    the NR valid times are exactly the centre time); the NR grid is NOT required
    to match the members, because it lives on a different grid.

    The comparison is done pairwise against a retained reference grid rather
    than over a list of every grid, so the memory cost does not grow with the
    ensemble size while the checks stay exactly the shared implementation's.
    """
    inputs = config.diagnostic_inputs()
    members = config.member_numbers()
    values = np.full((len(members), len(places)), np.nan, dtype=np.float64)
    member_times: dict[int, "str | None"] = {}
    compared: list[int] = []
    reference_grid = None
    reference_member = None
    unit_report: dict[str, str] = {}

    for index, member in enumerate(members):
        path = config.background_member_path(member)
        label = f"background member {member:03d}"
        units = read_temperature_units(path, config.state_var, label)
        unit_report[f"member_{member:03d}"] = units

        try:
            grid = lag.read_level_zero_field(
                path,
                config.state_var,
                config.state_level,
                config.lat_var,
                config.lon_var,
                config.water_mask_source,
            )
        except lag.DiagnosticError as exc:
            raise DirectionFilterError(
                f"{label}: {config.state_var} level {config.state_level} could not be "
                f"read: {exc}"
            ) from exc

        if reference_grid is None:
            reference_grid = grid
            reference_member = member
            lag.validate_background_consistency(inputs, [grid], [member], nr_grid)
        else:
            try:
                lag.validate_background_consistency(
                    inputs, [reference_grid, grid], [reference_member, member], nr_grid
                )
            except lag.ConsistencyError as exc:
                raise DirectionFilterError(
                    f"background member {member:03d} ({path}) disagrees with "
                    f"member {reference_member:03d} "
                    f"({reference_grid.path}) or with the NR valid time: {exc}"
                ) from exc
            compared.append(member)

        member_times[member] = grid.time_string
        for position, (raw_index, latitude, longitude) in enumerate(places):
            values[index, position] = sample_or_refuse(
                grid,
                latitude,
                longitude,
                f"background member {member:03d} ({path})",
                raw_index,
                config,
            )
        # The field is not needed any more: only the reference grid survives this
        # iteration, so the memory cost does not grow with the ensemble size.

    if reference_grid is None:  # pragma: no cover - members is never empty
        raise DirectionFilterError("no background member was loaded")

    assert reference_member is not None
    grid_report = {
        "validator": (
            "diag_LACC_lag_cov_innovation.validate_background_consistency "
            "(pairwise against the reference member)"
        ),
        "status": "passed",
        "expected_center_time": config.center_time,
        "require_center_time_match": config.require_center_time_match,
        "reference_member": int(reference_member),
        "n_members": len(members),
        "member_valid_times": {str(member): member_times[member] for member in members},
        "members_compared_to_reference": [int(member) for member in compared],
        "nr_valid_time": nr_grid.time_string,
        "water_mask_source": reference_grid.water_source,
        "latitude_grid": list(reference_grid.lat.shape),
        "note": (
            "member grids, water masks and valid times must all agree; the NR is "
            "only required to carry the centre time, because NR and background "
            "live on different grids and are sampled at the same physical point, "
            "never at the same array index"
        ),
    }
    return BackgroundSamples(
        values=values,
        member_times=member_times,
        reference_member=int(reference_member),
        grid_report=grid_report,
        units=unit_report,
    )


@dataclass
class MemberHx:
    values: np.ndarray  # (n_members, n_selected) float64
    entries: list[dict]


def read_member_hx(
    config: Config, places: list[tuple[int, float, float]]
) -> MemberHx:
    """Read every member's FINAL equal-weight LACC Hx and verify its numbering.

    The values are taken from the generated per-member file, never recomputed
    from the per-lag data: re-averaging the lags here could disagree with the Hx
    that the assimilation actually uses.  Every file must be the F-order
    flattening of its own 2x2... its own matrix; a matrix that is missing, that
    has the wrong shape, or that does not match is a hard refusal.  This task
    adds NO switch that relaxes that gate.

    Members are paired by number: the loop index is the member number, so
    member m's Hx row is combined with member m's SST row and never with a row
    produced by sorting one of the two vectors.
    """
    inputs = config.diagnostic_inputs()
    positions = [raw_index - 1 for raw_index, _, _ in places]
    members = config.member_numbers()
    values = np.full((len(members), len(places)), np.nan, dtype=np.float64)
    entries: list[dict] = []

    for index, member in enumerate(members):
        column_path = config.member_hx_path(member)
        matrix_path = config.member_matrix_path(member)
        label = f"member {member:03d} centre-time LACC Hx"
        try:
            column = lag.read_column_file(column_path, config.nobs_raw, label)
        except lag.DiagnosticError as exc:
            raise DirectionFilterError(f"member {member:03d}: {exc}") from exc
        try:
            entry = lag.enforce_f_order(
                inputs,
                matrix_path,
                column,
                label,
                column_path,
                suite=lag.SUITE_MEMBER_HX,
            )
        except lag.DiagnosticError as exc:
            raise DirectionFilterError(
                f"member {member:03d}: {exc}  The F-order / matrix gate is enforced "
                f"unconditionally by this filter."
            ) from exc
        entries.append(
            {
                "member": int(member),
                "column_path": str(column_path),
                "matrix_path": str(matrix_path),
                "f_order_matches": bool(entry.get("f_order_matches")),
                "max_abs_diff_f_order": entry.get("max_abs_diff_f_order"),
                "enforcement": entry.get("enforcement"),
            }
        )
        values[index, :] = column[positions]
    return MemberHx(values=values, entries=entries)


# =============================================================================
# Writers (atomic: nothing is published until every input has been checked)
# =============================================================================
def _number(value: "float | None", digits: int = 12) -> str:
    if value is None:
        return ""
    return format(float(value), f".{digits}g")


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_mask(path: Path, mask: np.ndarray) -> Path:
    """Fixed nobs rows, one 0/1 per row, trailing newline."""
    text = "".join(f"{int(value)}\n" for value in np.asarray(mask).reshape(-1))
    return _write_text_atomic(path, text)


def write_audit_csv(path: Path, rows: list[dict]) -> Path:
    if not rows:
        raise DirectionFilterError(f"refusing to write an empty audit: {path}")
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(AUDIT_COLUMNS), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return _write_text_atomic(path, buffer.getvalue())


def _json_safe(value):
    """Plain JSON for numpy scalars, Paths and non-finite floats.

    NaN and +/-inf become ``null``: the payload is written with
    ``allow_nan=False``, so a non-standard token cannot slip through, and no
    audit field is ever reported as the string "NaN".
    """
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (np.bool_, bool)):
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


def write_summary_json(path: Path, payload: dict) -> Path:
    text = json.dumps(
        _json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False
    )
    return _write_text_atomic(path, text + "\n")


# =============================================================================
# Reports
# =============================================================================
STATIC_FILTER_DEFINITION = {
    "name": "NR-assisted SST update-direction screen (equal-weight LACC)",
    "sst_definition": (
        "centre-time OM_TMP first time index, level 0, in BOTH the background "
        "members and the NR file; never TSK and never another SST variable"
    ),
    "symbols": {
        "x_m": "background member m centre-time OM_TMP[0, 0, :, :] interpolated at the observation",
        "h_m": "member m's final equal-weight LACC Hx value for the same observation",
        "y": "the final LACC observation, already perturbed with noise",
        "x_NR": "centre-time NR OM_TMP[0, 0, :, :] interpolated at the same point",
        "M": "number of ensemble members (at least 2)",
    },
    "formula": {
        "x_bar": "mean_m(x_m)",
        "h_bar": "mean_m(h_m)",
        "c": "sum_m((x_m - x_bar) * (h_m - h_bar)) / (M - 1)",
        "d": "y - h_bar",
        "e": "x_NR - x_bar",
        "score": "c * d * e",
    },
    "rule": (
        "score < 0 -> reject the whole observation; score >= 0 -> keep.  c = 0, "
        "d = 0 and e = 0 are all kept."
    ),
    "decision_implementation": (
        "after a finiteness check the decision is taken from the signs of c, d "
        "and e (odd number of negative factors and no zero factor), not from the "
        "product, because three small non-zero factors make c*d*e underflow to "
        "0.0 and a product test would then keep a reverse-direction observation. "
        "The product is still reported as the direction_score diagnostic."
    ),
    "deviations": (
        "member vectors that are exactly equal give deviations and covariance of "
        "EXACTLY zero: identical decimal Kelvin values do not average exactly in "
        "binary floating point, and a rounding residual would otherwise "
        "manufacture a covariance the ensemble does not have."
    ),
    "member_pairing": (
        "background member m is paired with Hx member m by member NUMBER; the "
        "two vectors are never sorted independently"
    ),
    "final_mask": "original_clear_sky_mask AND direction_keep",
    "single_static_pass": (
        "computed ONCE against one common prior, before the assimilation.  "
        "Nothing is recomputed inside a serial assimilation, and no increment "
        "magnitude, R estimate or localisation parameter is used."
    ),
    "scope_limits": [
        "a single-observation proxy for the LINEAR MEAN increment of the "
        "background ensemble at that point; the denominator of the real update is "
        "positive, so only the sign is claimed",
        "it does NOT guarantee the direction of the increment that a serial "
        "multi-observation assimilation actually produces",
        "it is NOT claimed to be equivalent to the update of a nonlinear or "
        "quantile/adaptive filter",
        "a correctly directed update that overshoots is KEPT: no overshoot, "
        "amplitude, correlation-threshold or significance test is applied",
        "a correct direction is not a smaller analysis error; this screen makes "
        "no skill claim",
    ],
}


def config_report(config: Config) -> dict:
    return {
        "project_root": str(config.project_root),
        "hx_root": str(config.diagnostic_inputs().resolved_hx_root()),
        "output_dir": str(config.output_dir),
        "center_time": config.center_time,
        "center_tag": config.center_tag(),
        "domain": config.domain,
        "sensor": config.sensor,
        "channel": int(config.channel),
        "member_start": int(config.member_start),
        "member_end": int(config.member_end),
        "n_members": len(config.member_numbers()),
        "nobs_raw": int(config.nobs_raw),
        "matrix_side": int(config.matrix_side),
        "expected_lag_hours": [int(value) for value in config.expected_lag_hours],
        "state_var": config.state_var,
        "state_level": int(config.state_level),
        "lat_var": config.lat_var,
        "lon_var": config.lon_var,
        "water_mask_source": config.water_mask_source,
        "sample_mode": config.sample_mode,
        "sample_require_water": bool(config.sample_require_water),
        "local_window_deg": float(config.local_window_deg),
        "max_local_window_deg": float(config.max_local_window_deg),
        "require_center_time_match": bool(config.require_center_time_match),
        "background_pattern": config.background_pattern,
    }


def input_report(config: Config) -> dict:
    return {
        "observation_file": str(config.resolved_obs_input_file()),
        "original_clear_sky_mask_file": str(config.resolved_clear_sky_mask_file()),
        "profile_file": str(config.resolved_profile_file()),
        "lacc_times_file": str(config.resolved_lacc_times_file()),
        "background_dir": str(config.resolved_background_dir()),
        "background_member_example": str(config.background_member_path(config.member_start)),
        "nr_file": str(config.resolved_nr_file()),
        "member_hx_example": str(config.member_hx_path(config.member_start)),
        "member_matrix_example": str(config.member_matrix_path(config.member_start)),
    }


# =============================================================================
# The run
# =============================================================================
def run(config: Config) -> dict:
    """Screen the observation set and publish the merged mask.

    Returns the payload that is also written to the summary JSON: the output
    paths and the counts.  Raises on every refusal, and raises
    ``NoObservationSurvivesError`` (after writing the audit) when the direction
    test leaves nothing to assimilate.
    """
    validate_config(config)
    members = config.member_numbers()

    # ---- 1. the assimilated window ----------------------------------------
    times_path = config.resolved_lacc_times_file()
    file_center, file_lags = lag.read_times_file(times_path)  # rejects a repeated lag
    window_report = check_window(config, file_center, file_lags)

    # ---- 2. the original clear-sky mask -----------------------------------
    original = read_original_mask(config)
    selected_positions = [int(raw) - 1 for raw in original.kept_raw_ids]

    # ---- 3. observations and their locations ------------------------------
    obs_path = config.resolved_obs_input_file()
    observation = lag.read_column_file(obs_path, config.nobs_raw, "final LACC observation")
    coordinates = lag.read_profile_coordinates(
        config.resolved_profile_file(), config.nobs_raw, config.center_tag()
    )
    coordinate_report = check_coordinates(coordinates)
    places = [
        (raw, float(coordinates.lat[position]), float(coordinates.lon[position]))
        for raw, position in zip(original.kept_raw_ids.tolist(), selected_positions)
    ]

    # ---- 4. background members (one open each) and the NR (one open) ------
    nr_path = config.resolved_nr_file()
    nr_units = read_temperature_units(nr_path, config.state_var, "NR")
    try:
        nr_grid = lag.read_level_zero_field(
            nr_path,
            config.state_var,
            config.state_level,
            config.lat_var,
            config.lon_var,
            config.water_mask_source,
        )
    except lag.DiagnosticError as exc:
        raise DirectionFilterError(
            f"NR: {config.state_var} level {config.state_level} could not be read: {exc}"
        ) from exc
    nr_values = np.asarray(
        [
            sample_or_refuse(nr_grid, latitude, longitude, f"NR ({nr_path})", raw, config)
            for raw, latitude, longitude in places
        ],
        dtype=np.float64,
    )

    background = sample_background(config, nr_grid, places)
    hx = read_member_hx(config, places)

    # The NR grid is not needed any more.
    del nr_grid

    # ---- 5. the static direction screen -----------------------------------
    final_mask = np.zeros(config.nobs_raw, dtype=np.int8)
    rows: list[dict] = []
    counts = {
        "n_raw": int(config.nobs_raw),
        "n_original_clear_sky": int(original.kept_raw_ids.size),
        "excluded_by_original_mask": 0,
        "rejected_reverse": 0,
        "kept_nonreverse": 0,
        "kept_zero_direction": 0,
        "final_kept": 0,
    }
    base_position_of_raw = {
        int(raw): position for position, raw in enumerate(original.kept_raw_ids.tolist())
    }
    final_ids: dict[int, int] = {}

    for raw_position in range(config.nobs_raw):
        raw_id = raw_position + 1
        latitude = float(coordinates.lat[raw_position])
        longitude = float(coordinates.lon[raw_position])
        if int(original.values[raw_position]) == 0:
            counts["excluded_by_original_mask"] += 1
            rows.append(
                {
                    "raw_obs_id_1based": raw_id,
                    "base_filtered_id_1based": "",
                    "final_filtered_id_1based": "",
                    "latitude": _number(latitude),
                    "longitude": _number(longitude),
                    "original_mask": 0,
                    "status": STATUS_EXCLUDED_BY_ORIGINAL_MASK,
                    "sst_background_mean": "",
                    "sst_nr": "",
                    "sst_error_nr_minus_background": "",
                    "hx_mean": "",
                    "actual_observation": "",
                    "innovation": "",
                    "cov_sst_hx": "",
                    "direction_score": "",
                    "keep_final": 0,
                }
            )
            continue

        position = base_position_of_raw[raw_id]
        try:
            result = evaluate_direction(
                background.values[:, position],
                hx.values[:, position],
                float(observation[raw_position]),
                float(nr_values[position]),
            )
        except DirectionFilterError as exc:
            raise DirectionFilterError(
                f"raw observation {raw_id}, observation file {obs_path}: {exc}"
            ) from exc
        if result.keep:
            final_mask[raw_position] = 1
            final_ids[raw_id] = len(final_ids) + 1
            counts["final_kept"] += 1
        else:
            counts["rejected_reverse"] += 1
        if result.status == STATUS_KEPT_NONREVERSE:
            counts["kept_nonreverse"] += 1
        elif result.status == STATUS_KEPT_ZERO_DIRECTION:
            counts["kept_zero_direction"] += 1

        rows.append(
            {
                "raw_obs_id_1based": raw_id,
                "base_filtered_id_1based": position + 1,
                "final_filtered_id_1based": final_ids.get(raw_id, ""),
                "latitude": _number(latitude),
                "longitude": _number(longitude),
                "original_mask": 1,
                "status": result.status,
                "sst_background_mean": _number(result.sst_background_mean),
                "sst_nr": _number(float(nr_values[position])),
                "sst_error_nr_minus_background": _number(result.sst_error),
                "hx_mean": _number(result.hx_mean),
                "actual_observation": _number(float(observation[raw_position])),
                "innovation": _number(result.innovation),
                "cov_sst_hx": _number(result.cov_sst_hx),
                "direction_score": _number(result.score),
                "keep_final": int(result.keep),
            }
        )

    # ---- 6. publish (only now: every input was read and every refusal has
    #         already happened) ---------------------------------------------
    paths = config.output_paths()
    payload = {
        "script": Path(__file__).name,
        "static_filter": STATIC_FILTER_DEFINITION,
        "config": config_report(config),
        "inputs": input_report(config),
        "members": {
            "numbers": [int(member) for member in members],
            "n_members": len(members),
            "paired_by_member_number": True,
        },
        "checks": {
            "window": window_report,
            "coordinates": coordinate_report,
            "times_and_grid": background.grid_report,
            "temperature_units": {
                "background": background.units,
                "nr": {"file": str(nr_path), "value": nr_units},
                "accepted_tokens": sorted(KELVIN_UNIT_TOKENS),
            },
            "member_hx_f_order": {
                "suite": lag.SUITE_MEMBER_HX,
                "status": "passed",
                "n_verified": len(hx.entries),
                "gate": "enforced unconditionally (no switch relaxes it)",
                "entries": hx.entries,
            },
            "observation_file": {
                "path": str(obs_path),
                "shape": [int(config.nobs_raw), 1],
                "reader": "diag_LACC_lag_cov_innovation.read_column_file",
                "note": (
                    "exactly the file the downstream converter reads: the driver "
                    "resolves OBS_INPUT_FILE once and exports it for both"
                ),
            },
        },
        "counts": counts,
        "outputs": {
            "mask": str(paths["mask"]),
            "audit_csv": str(paths["audit_csv"]),
            "summary_json": str(paths["summary_json"]),
        },
    }

    write_audit_csv(paths["audit_csv"], rows)
    write_mask(paths["mask"], final_mask)
    write_summary_json(paths["summary_json"], payload)

    if counts["final_kept"] == 0:
        raise NoObservationSurvivesError(
            f"the NR SST direction screen deleted every observation "
            f"({counts['n_original_clear_sky']} were clear-sky of "
            f"{counts['n_raw']} raw).  The audit was written to "
            f"{paths['audit_csv']} and the mask to {paths['mask']}, but the "
            f"assimilation must NOT continue: an empty observation set is a "
            f"failure of the screen, not a result.",
            payload,
        )
    return payload


def print_summary(payload: dict) -> None:
    counts = payload["counts"]
    inputs = payload["inputs"]
    outputs = payload["outputs"]
    print("NR-assisted SST update-direction screen (equal-weight LACC)")
    print(f"  observation file:          {inputs['observation_file']}")
    print(f"  original clear-sky mask:   {inputs['original_clear_sky_mask_file']}")
    print(f"  background directory:      {inputs['background_dir']}")
    print(f"  NR file:                   {inputs['nr_file']}")
    print(f"  raw observations:          {counts['n_raw']}")
    print(f"  original clear-sky kept:   {counts['n_original_clear_sky']}")
    print(f"  excluded by original mask: {counts['excluded_by_original_mask']}")
    print(f"  rejected (reverse):        {counts['rejected_reverse']}")
    print(f"  final kept:                {counts['final_kept']}")
    print(f"    kept, non-reverse:       {counts['kept_nonreverse']}")
    print(f"    kept, zero direction:    {counts['kept_zero_direction']}")
    print(f"  merged mask:               {outputs['mask']}")
    print(f"  audit CSV:                 {outputs['audit_csv']}")
    print(f"  summary JSON:              {outputs['summary_json']}")


def main() -> int:
    """Run the top-of-file CONFIG (the shell driver uses its own bridge)."""
    try:
        payload = run(CONFIG)
    except NoObservationSurvivesError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print_summary(exc.payload)
        return 3
    except (DirectionFilterError, lag.DiagnosticError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
