#!/usr/bin/env bash
#BSUB -J cycle_50mem_hourly
#BSUB -q serial
#BSUB -n 1
#BSUB -oo cycle_50mem_hourly.out
#BSUB -eo cycle_50mem_hourly.err
#
# One overall driver: observation production -> DART -> WRF, 50 members.
# Edit the configuration below, then: bash run_driver_50mem_cyclingDA.sh
# MODE=DRY_RUN prints the exact plan without creating files; PREFLIGHT only
# reads prerequisites; RUN performs preflight before creating/submitting work.
# Requires the existing production scripts, MATLAB/RTTOV, DART, WRF, NCO and
# LSF environment. Python needs netCDF4/numpy (the deployed wrf environment).
# No installation, server deployment or modification of source data is done.
#
# 00/01/02/03 UTC analyses; 1h/1h/1h/6h forecasts, finishing at09 UTC.
# LACC NR and H(x) windows: 0,3,6 / 0,4,7 / 0,5,8 / 0,3,6 hours.
# At03 the historical00 is the preserved00 PRIOR, not the00 analysis.
# Strong coupling follows the six-member driver: two independent DA passes
# from the same prior, followed by transplantation of the first-pass ocean.
#
# Runtime products and generated job scripts live in RUN_ROOT, separated by
# group/filter/cycle/stage. Original production scripts and inputs stay intact.
# IMPORTANT: the server BC checked on 2026-11-10 ends at06. Prepare member BCs
# covering09 (including correct tendencies) and set BC_ROOT/BC_PATTERN first.
# The driver refuses insufficient BCs BEFORE submitting any production job.
# Existing complete stages are reusable only with RESUME=1 and matching inputs.
# Resume fingerprints include source code, namelists, symlink targets, WRF
# static tables and RTTOV coefficients/binaries; inputs are rechecked per cycle.
# To inspect before running, change MODE above/below to DRY_RUN or PREFLIGHT.
# Set RUN_GROUP_1/2/3 and RUN_EAKF/RUN_QCF_RHF individually (0/1); all default1.
# Directory layout: RUN_ROOT/oceanAssimXRunY/FILTER/{prior,analysis,wrfout,
# run_wrf,dart,obs,jobs,logs}/TIME. Segment-start analysis output and the prior
# at the same valid time are stored separately. Final output reaches09 UTC.
# With ocean physics OFF, WRF omits OM_*: next priors retain the unchanged
# previous ocean in an annotated COPY, while original wrfout stays archived.
# All editable settings are here; the embedded Python is orchestration code.
set -Eeuo pipefail

# ======================= user configuration ================================
export MODE="${MODE:-RUN}"                          # RUN / PREFLIGHT / DRY_RUN
export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/lililei1/kcfu/tc_mangkhut}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/lililei1/kcfu/tc_mangkhut}"
export RUN_ROOT="${RUN_ROOT:-${SCRATCH_ROOT}/5cyclingDA/50mem_hourly}"
export INITIAL_PRIOR_DIR="${INITIAL_PRIOR_DIR:-${PROJECT_ROOT}/4assimilation/0mem_all_time/cyclingDA/10_00_00}"
export HISTORY_ROOT="${HISTORY_ROOT:-${SCRATCH_ROOT}/2ens_free_fcst}"
export BC_ROOT="${BC_ROOT:-${PROJECT_ROOT}/1icbc}"
export IC_ROOT="${IC_ROOT:-${PROJECT_ROOT}/1icbc}"
# Patterns are Python format strings; keep literal braces as written.
BC_PATTERN_DEFAULT='mem{mem:03d}/BC/wrfbdy_d01_mem{mem:03d}'
IC_PATTERN_DEFAULT='mem{mem:03d}/IC/wrfinput_2018091000_mem{mem:03d}'
export BC_PATTERN="${BC_PATTERN:-${BC_PATTERN_DEFAULT}}"
export IC_PATTERN="${IC_PATTERN:-${IC_PATTERN_DEFAULT}}"
export NR_ROOT="${NR_ROOT:-${PROJECT_ROOT}/NR_wrfout/2domain}"
export DART_TEMPLATE_DIR="${DART_TEMPLATE_DIR:-${PROJECT_ROOT}/4assimilation/2DART/run_dir}"
export DART_WORK="${DART_WORK:-/share/home/lililei1/kcfu/models/DART_main/models/wrf/work}"
export WRF_EM_REAL="${WRF_EM_REAL:-/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/WRF-4.1/test/em_real}"
export RTTOV_ROOT="${RTTOV_ROOT:-/share/home/lililei1/kcfu/models/rttov123}"
export PYTHON_EXE="${PYTHON_EXE:-/share/home/lililei1/kcfu/anaconda/envs/wrf/bin/python}"

export RUN_GROUP_1="${RUN_GROUP_1:-1}"              # no ocean update/model
export RUN_GROUP_2="${RUN_GROUP_2:-1}"              # ocean model, no update
export RUN_GROUP_3="${RUN_GROUP_3:-1}"              # ocean update + model
export RUN_EAKF="${RUN_EAKF:-1}"
export RUN_QCF_RHF="${RUN_QCF_RHF:-1}"
export HX_STATE_DOMAIN="${HX_STATE_DOMAIN:-d02}"    # physical grid; obs label=d01
export ASSIM_CHANNEL="${ASSIM_CHANNEL:-4}"
export OBS_ERR_STD="${OBS_ERR_STD:-0.5}"             # single-time K; LACC /sqrt(3)
export CLEAR_SKY_THRESHOLD="${CLEAR_SKY_THRESHOLD:-0.2}"
export HGT_OBS="${HGT_OBS:-85000}"                  # Pa
export RESUME="${RESUME:-0}"
export POLL_SECONDS="${POLL_SECONDS:-30}"
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-172800}"
export MAX_WRF_ATTEMPTS="${MAX_WRF_ATTEMPTS:-3}"
export DART_QUEUE="${DART_QUEUE:-largemem}"
export DART_TASKS="${DART_TASKS:-120}"
export WRF_QUEUE="${WRF_QUEUE:-mpi}"
export WRF_TASKS="${WRF_TASKS:-144}"
export TASKS_PER_NODE="${TASKS_PER_NODE:-24}"
export DART_MPI="${DART_MPI:-mpirun}"
export WRF_MPI="${WRF_MPI:-mpiexec.hydra}"
export CYCLE_DRIVER_SOURCE="${BASH_SOURCE[0]}"
# ===========================================================================
exec "${PYTHON_EXE}" - <<'CYCLE_PYTHON'
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

START = datetime(2018, 9, 10, tzinfo=timezone.utc)
MEMBERS = tuple(range(1, 51))
DOMAINS = ("d01", "d02")
ATM_VARS = "MU,P,PH,QVAPOR,THM,U,V,W"
OCEAN_VARS = "OM_TMP,OM_S,OM_U,OM_V"


class DriverError(RuntimeError):
    pass


class JobFailed(DriverError):
    """The scheduler confirmed failure/termination, so a later retry is safe."""


def stamp(moment):
    return moment.strftime("%Y-%m-%d_%H:%M:%S")


def tag(moment):
    return moment.strftime("%d_%H_%M")


@dataclass(frozen=True)
class Cycle:
    time: datetime
    hours: int
    lags: tuple[int, ...]

    @property
    def tag(self):
        return tag(self.time)

    @property
    def end(self):
        return self.time + timedelta(hours=self.hours)

    @property
    def end_tag(self):
        return tag(self.end)

    @property
    def lacc_times(self):
        return tuple(self.time - timedelta(hours=h) for h in self.lags)

    @property
    def lacc_tags(self):
        return tuple(tag(t) for t in self.lacc_times)


def cycles():
    return tuple(Cycle(START + timedelta(hours=i), 6 if i == 3 else 1, lags)
                 for i, lags in enumerate(((0, 3, 6), (0, 4, 7), (0, 5, 8), (0, 3, 6))))


@dataclass(frozen=True)
class Experiment:
    group: int
    filter: str

    @property
    def update_ocean(self):
        return int(self.group == 3)

    @property
    def run_ocean(self):
        return int(self.group != 1)

    @property
    def name(self):
        return f"oceanAssim{self.update_ocean}Run{self.run_ocean}/{self.filter}"


@dataclass(frozen=True)
class Config:
    project: Path
    scratch: Path
    run_root: Path | None = None
    initial: Path | None = None
    history: Path | None = None
    bc_root: Path | None = None
    ic_root: Path | None = None
    nr_root: Path | None = None
    dart_templates: Path | None = None
    dart_work: Path = Path("/share/home/lililei1/kcfu/models/DART_main/models/wrf/work")
    wrf_static: Path = Path("/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/WRF-4.1/test/em_real")
    rttov_root: Path = Path("/share/home/lililei1/kcfu/models/rttov123")
    python: str = sys.executable
    groups: tuple[int, ...] = (1, 2, 3)
    filters: tuple[str, ...] = ("EAKF", "QCF_RHF")
    hx_domain: str = "d02"
    channel: int = 4
    obs_error: float = 0.5
    clear_threshold: float = 0.2
    obs_pressure: int = 85000
    resume: bool = False
    poll: int = 30
    max_wait: int = 172800
    wrf_attempts: int = 3
    dart_queue: str = "largemem"
    dart_tasks: int = 120
    wrf_queue: str = "mpi"
    wrf_tasks: int = 144
    ptile: int = 24
    dart_mpi: str = "mpirun"
    wrf_mpi: str = "mpiexec.hydra"
    bc_pattern: str = "mem{mem:03d}/BC/wrfbdy_d01_mem{mem:03d}"
    ic_pattern: str = "mem{mem:03d}/IC/wrfinput_2018091000_mem{mem:03d}"

    @property
    def root(self):
        return self.run_root or self.scratch / "5cyclingDA/50mem_hourly"

    @property
    def initial_dir(self):
        return self.initial or self.project / "4assimilation/0mem_all_time/cyclingDA/10_00_00"

    @property
    def history_dir(self):
        return self.history or self.scratch / "2ens_free_fcst"

    @property
    def nr_dir(self):
        return self.nr_root or self.project / "NR_wrfout/2domain"

    @property
    def templates(self):
        return self.dart_templates or self.project / "4assimilation/2DART/run_dir"

    def bc(self, member):
        return (self.bc_root or self.project / "1icbc") / self.bc_pattern.format(mem=member)

    def ic(self, member):
        return (self.ic_root or self.project / "1icbc") / self.ic_pattern.format(mem=member)

    @classmethod
    def from_env(cls):
        def enabled(name):
            value = os.environ.get(name, "1")
            if value not in ("0", "1"):
                raise DriverError(f"{name} must be 0 or 1")
            return value == "1"
        opts = dict(project=Path(os.environ["PROJECT_ROOT"]), scratch=Path(os.environ["SCRATCH_ROOT"]))
        for name, key in (("RUN_ROOT", "run_root"), ("INITIAL_PRIOR_DIR", "initial"),
                          ("HISTORY_ROOT", "history"), ("BC_ROOT", "bc_root"),
                          ("IC_ROOT", "ic_root"), ("NR_ROOT", "nr_root"),
                          ("DART_TEMPLATE_DIR", "dart_templates"), ("DART_WORK", "dart_work"),
                          ("WRF_EM_REAL", "wrf_static"), ("RTTOV_ROOT", "rttov_root")):
            opts[key] = Path(os.environ[name])
        for name, key in (("ASSIM_CHANNEL", "channel"), ("HGT_OBS", "obs_pressure"),
                          ("POLL_SECONDS", "poll"), ("MAX_WAIT_SECONDS", "max_wait"),
                          ("MAX_WRF_ATTEMPTS", "wrf_attempts"), ("DART_TASKS", "dart_tasks"),
                          ("WRF_TASKS", "wrf_tasks"), ("TASKS_PER_NODE", "ptile")):
            opts[key] = int(os.environ[name])
        for name, key in (("OBS_ERR_STD", "obs_error"), ("CLEAR_SKY_THRESHOLD", "clear_threshold")):
            opts[key] = float(os.environ[name])
        for name, key in (("PYTHON_EXE", "python"), ("HX_STATE_DOMAIN", "hx_domain"),
                          ("BC_PATTERN", "bc_pattern"), ("IC_PATTERN", "ic_pattern"),
                          ("DART_QUEUE", "dart_queue"), ("WRF_QUEUE", "wrf_queue"),
                          ("DART_MPI", "dart_mpi"), ("WRF_MPI", "wrf_mpi")):
            opts[key] = os.environ[name]
        opts["groups"] = tuple(i for i in (1, 2, 3) if enabled(f"RUN_GROUP_{i}"))
        opts["filters"] = tuple(f for f in ("EAKF", "QCF_RHF") if enabled(f"RUN_{f}"))
        opts["resume"] = enabled("RESUME")
        cfg = cls(**opts)
        if not cfg.groups or not cfg.filters:
            raise DriverError("select at least one coupling group and one filter")
        if cfg.hx_domain not in DOMAINS or cfg.channel not in range(1, 7):
            raise DriverError("HX_STATE_DOMAIN must be d01/d02; channel must be 1..6")
        if not all(math.isfinite(v) and v > 0 for v in (cfg.obs_error, cfg.clear_threshold)):
            raise DriverError("observation error and clear-sky threshold must be finite and positive")
        if min(cfg.poll, cfg.max_wait, cfg.wrf_attempts, cfg.dart_tasks, cfg.wrf_tasks, cfg.ptile) <= 0:
            raise DriverError("polling, retry and LSF resource values must be positive")
        return cfg


def experiments(cfg):
    return tuple(Experiment(g, f) for g in cfg.groups for f in cfg.filters)


def prior_file(root, moment, domain, member):
    return root / "prior" / tag(moment) / f"firstguess_{domain}.mem{member:03d}"


def input_source(cfg, root, cycle, moment, domain, member):
    if moment < START:
        return cfg.history_dir / f"mem{member:03d}" / f"wrfout_{domain}_{stamp(moment)}"
    if moment == START and cycle.time == START:
        return cfg.initial_dir / f"firstguess_{domain}.mem{member:03d}"
    return prior_file(root, moment, domain, member)


def plan(cfg):
    return {"members": 50, "domains": list(DOMAINS), "hx_state_domain": cfg.hx_domain,
            "boundary_mem001": str(cfg.bc(1)), "initialization_mem001": str(cfg.ic(1)),
            "lacc_nr_sst_direction_filter": True, "experiments": [
        {"group": exp.group, "filter": exp.filter, "root": str(cfg.root / exp.name),
         "update_ocean": exp.update_ocean, "run_ocean": exp.run_ocean,
         "cycles": [{"analysis": stamp(c.time), "end": stamp(c.end), "forecast_hours": c.hours,
                     "lag_hours": list(c.lags), "lacc_times": [stamp(t) for t in c.lacc_times],
                     "hx_sources_mem001": [str(input_source(cfg, cfg.root / exp.name, c, t,
                                                              cfg.hx_domain, 1)) for t in c.lacc_times]}
                    for c in cycles()]} for exp in experiments(cfg)]}


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise DriverError(f"missing or empty input: {path}")
    return path


def read_nc_times(dataset, name="Times"):
    import netCDF4 as nc
    if name not in dataset.variables:
        raise DriverError(f"{dataset.filepath()}: missing {name}")
    strings = nc.chartostring(dataset.variables[name][:]).reshape(-1)
    try:
        return tuple(datetime.strptime(str(s).strip(), "%Y-%m-%d_%H:%M:%S")
                     .replace(tzinfo=timezone.utc) for s in strings)
    except ValueError as exc:
        raise DriverError(f"{dataset.filepath()}: invalid {name}: {exc}") from exc


def validate_state(path, moment, required=()):
    import netCDF4 as nc
    require_file(path)
    try:
        with nc.Dataset(path, "r") as ds:
            if read_nc_times(ds) != (moment,):
                raise DriverError(f"{path}: valid time must be exactly {stamp(moment)}")
            missing = set(required) - set(ds.variables)
            if missing:
                raise DriverError(f"{path}: missing variables {','.join(sorted(missing))}")
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, DriverError):
            raise
        raise DriverError(f"cannot read state {path}: {exc}") from exc


def validate_boundary(path):
    import netCDF4 as nc
    require_file(path)
    with nc.Dataset(path, "r") as ds:
        begins = read_nc_times(ds, "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_")
        ends = read_nc_times(ds, "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_")
        records = read_nc_times(ds)
    if not begins or len(begins) != len(ends) or records != begins:
        raise DriverError(f"{path}: boundary Times/thisbdytime/nextbdytime disagree")
    cursor = START
    for begin, end in zip(begins, ends):
        if end <= begin:
            raise DriverError(f"{path}: nonpositive boundary interval")
        if end <= cursor:
            continue
        if begin > cursor:
            raise DriverError(f"{path}: boundary gap at {stamp(cursor)} -> {stamp(begin)}")
        cursor = end
        if cursor >= cycles()[-1].end:
            return
    raise DriverError(f"{path}: boundary effective end {stamp(cursor)} < required {stamp(cycles()[-1].end)}; "
                      "prepare extended MEMBER boundaries; no extrapolation is performed")


def producer_sources(cfg):
    hx = cfg.project / "3create_obs/hx_rttov"
    files = [hx / name for name in (
        "run_rttov_TrueObs_driver.sh", "run_rttov_ensBT_driver.sh",
        "run_rttov_TrueObs_LACC_driver.sh", "run_rttov_ensBT_LACC_driver.sh")]
    for part in ("profile/1run_matlab", "profile/2ens_matlab", "2call_rttov",
                 "3obs_BT/merge", "3obs_BT/add_pert", "4ens_BT/merge"):
        files += sorted(p for p in (hx / part).rglob("*")
                        if p.is_file() and p.suffix in (".sh", ".m", ".py"))
    convert = cfg.project / "4assimilation/1convert_obs"
    files += [convert / name for name in (
        "run_single_assimilation_test.sh", "obs2DART_test.py", "obs2DART_LACC.py",
        "merge_FO_in_one_file.py", "filter_LACC_sst_direction.py")]
    files += [cfg.project / "plot_scripts/diag_LACC_lag_cov_innovation.py"]
    return sorted(set(files))


def validate_producer_layout(cfg):
    hx = cfg.project / "3create_obs/hx_rttov"
    def assignment(path, key, allowed):
        text = require_file(path).read_text(encoding="utf-8", errors="surrogateescape")
        values = re.findall(r"(?m)^\s*(?:export\s+)?" + re.escape(key) + r"=(.+)$", text)
        if not values or any(value.strip().strip("\"'") not in allowed for value in values):
            raise DriverError(f"unsupported mutable/private-path assignment {key}: {path}")
    for name in ("run_rttov_TrueObs_driver.sh", "run_rttov_ensBT_driver.sh",
                 "run_rttov_TrueObs_LACC_driver.sh", "run_rttov_ensBT_LACC_driver.sh"):
        file = hx / name
        lacc = "LACC" in name
        obs_part = "3obs_BT" if "TrueObs" in name else "4ens_BT"
        assignment(file, "work_dir", {str(cfg.project / "3create_obs/hx_rttov"),
                                     "/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov"})
        assignment(file, "run_matlab_dir", {"${work_dir}/profile"})
        assignment(file, "prof_dir", {"${work_dir}/profile/profile_${domain}" +
                                       ("_LACC_${lacc_center_time}" if lacc else "")})
        assignment(file, "obs_dir", {"${work_dir}/" + obs_part + ("_LACC" if lacc else "")})
        assignment(file, "merge_dir", {"${work_dir}/" + obs_part + "/merge" if lacc else "${obs_dir}/merge"})
        assignment(file, "add_pert_dir", {"${obs_dir}/add_pert"})
    coordinator = cfg.project / "4assimilation/1convert_obs/run_single_assimilation_test.sh"
    expected = {
        "HX_DIR": {"${BASE_DIR}/3create_obs/hx_rttov"},
        "CONVERT_DIR": {"${BASE_DIR}/4assimilation/1convert_obs"},
        "TEXT_TO_OBS_RUN_DIR": {"${CONVERT_DIR}/run_dir"},
        "OBS_SEQ_OUT_FILE": {"${TEXT_TO_OBS_RUN_DIR}/${OBS_SEQ_OUT_NAME}"},
        "SINGLE_TEST_WORK_DIR": {"${CONVERT_DIR}/single_assim_test/${EXPERIMENT_MODE}_${CURRENT_TIME}_ch${ASSIM_CHANNEL}${NR_SST_DIRECTION_TAG}"},
        "LOG_DIR": {"${SINGLE_TEST_WORK_DIR}/logs"},
        "OBS_DART_INPUT_FILE": {"${SINGLE_TEST_WORK_DIR}/obs_input.txt"},
        "MERGED_FO_FILE": {"${SINGLE_TEST_WORK_DIR}/merged_FO_data.txt"},
        "TEST_INPUT_NML": {"${SINGLE_TEST_WORK_DIR}/input.nml"},
        "RTTOV_SENSOR_DB_LINK": {"${SINGLE_TEST_WORK_DIR}/rttov_sensor_db.csv"},
        "OBS_BT_DIR": {"${HX_DIR}/3obs_BT/AMSUA", "${HX_DIR}/3obs_BT_LACC/AMSUA"},
        "PROFILE_DIR": {"${HX_DIR}/profile"},
        "ENS_BT_DIR": {"${HX_DIR}/4ens_BT", "${HX_DIR}/4ens_BT_LACC", "${ADAPTIVE_LACC_DIR}/members"},
        "NR_SST_DIRECTION_FILTER_DIR": {"${SINGLE_TEST_WORK_DIR}/nr_sst_direction_filter"},
    }
    for key, allowed in expected.items():
        assignment(coordinator, key, allowed)
    matlab = hx / "profile/2ens_matlab/wrf_rttov_d02_75km_ens_interp.m"
    text = require_file(matlab).read_text()
    if len(re.findall(r"wrfout_d0[12]_2018-09-", text)) < 2:
        raise DriverError(f"unsupported physical H(x) filename layout: {matlab}")
    wrapper = require_file(hx / "profile/2ens_matlab/d01_run_matlab.sh").read_text()
    if not re.search(r"(?m)^\s*running_script_file=wrf_rttov_d02_75km_ens_interp\.m\s*$", wrapper):
        raise DriverError("producer clear-sky MATLAB wrapper selects an unsupported profile implementation")


def preflight(cfg):
    """Read metadata only; run BEFORE creating directories or submitting jobs."""
    import importlib.util
    issues = []
    def check(fn, *args):
        try:
            fn(*args)
        except (DriverError, OSError, ValueError, RuntimeError) as exc:
            issues.append(str(exc))
    if cfg.scratch.resolve() not in cfg.root.resolve().parents:
        issues.append(f"RUN_ROOT must be strictly underneath SCRATCH_ROOT: {cfg.root}")
    for path in (cfg.project, cfg.root, cfg.initial_dir, cfg.history_dir, cfg.nr_dir,
                 cfg.templates, cfg.dart_work, cfg.wrf_static, cfg.rttov_root, Path(cfg.python)):
        if re.search(r"[\s'\"`$]", str(path)):
            issues.append(f"legacy producer requires paths without whitespace/shell metacharacters: {path}")
    for module in ("netCDF4", "numpy", "scipy", "matplotlib"):
        if importlib.util.find_spec(module) is None:
            issues.append(f"{cfg.python}: missing Python dependency {module}")
    commands = ["bsub", "bjobs", "bhist", "bkill", "matlab", "ncks", "perl", "bash"]
    commands += [shlex.split(cfg.dart_mpi)[0], shlex.split(cfg.wrf_mpi)[0]]
    for command in commands:
        if shutil.which(command) is None:
            issues.append(f"command is not on PATH: {command}")
    for file in producer_sources(cfg):
        check(require_file, file)
    check(validate_producer_layout, cfg)
    runtime = cfg.project / "4assimilation/1convert_obs/run_dir"
    for name in ("input.nml", "text_to_obs", "rttov_sensor_db.csv"):
        check(require_file, runtime / name)
    for path in (cfg.dart_work / "filter", cfg.dart_work / "update_wrf_bc",
                 cfg.wrf_static / "wrf.exe", cfg.project / "0necessay_files/input.nml",
                 cfg.project / "0necessay_files/namelist.input.ens",
                 cfg.project / "0necessay_files/ad_omini_d01.py",
                 cfg.project / "0necessay_files/ad_omini_d02.py"):
        check(require_file, path)
    for path in rttov_files(cfg):
        check(require_file, path)
    needed_templates = {"input.nml.all", "input.nml.inflateOcean"} if 3 in cfg.groups else set()
    if any(g in cfg.groups for g in (1, 2)):
        needed_templates.add("input.nml.air")
    for name in needed_templates:
        check(require_file, cfg.templates / name)
    if "QCF_RHF" in cfg.filters:
        check(require_file, cfg.templates / "qceff_table_fkc.csv")
    for name in ("rttov_sensor_db.csv", "rtcoef_noaa_18_amsua.dat"):
        check(require_file, cfg.templates / name)
    # Reading Times and checking variable names does not read the large fields.
    state_vars = (ATM_VARS + "," + OCEAN_VARS + ",TSK,XLAT,XLONG").split(",")
    hx_vars = ("T", "P", "PB", "PH", "PHB", "TSK", "T2", "Q2", "U10", "V10",
               "QVAPOR", "QCLOUD", "QICE", "PSFC", "HGT", "LANDMASK", "CLDFRA", "XLAT", "XLONG")
    for member in MEMBERS:
        for domain in DOMAINS:
            check(validate_state, cfg.initial_dir / f"firstguess_{domain}.mem{member:03d}", START,
                  tuple(set(state_vars) | (set(hx_vars) if domain == cfg.hx_domain else set())))
        for hour in (3, 6):
            moment = START - timedelta(hours=hour)
            check(validate_state, cfg.history_dir / f"mem{member:03d}" /
                  f"wrfout_{cfg.hx_domain}_{stamp(moment)}", moment, hx_vars)
        check(validate_state, cfg.ic(member), START, ("OM_TINI", "OM_SINI"))
        check(validate_boundary, cfg.bc(member))
    nr_vars = tuple(set(hx_vars) | {"QRAIN", "QSNOW", "QGRAUP", "OM_TMP"})
    nr_times = sorted({t for c in cycles() for t in c.lacc_times})
    for moment in nr_times:
        check(validate_state, cfg.nr_dir / f"wrfout_d02_{stamp(moment)}", moment, nr_vars)
    if issues:
        raise DriverError("preflight failed; NO jobs submitted and NO runtime directories created:\n  - "
                          + "\n  - ".join(issues))
    log("Preflight passed: all 50 members, both domains, LACC times and BC coverage through09")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def link_readonly(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == Path(source).absolute().resolve():
            return
        target.unlink()
    elif target.exists():
        raise DriverError(f"refusing to replace a regular file with an input link: {target}")
    target.symlink_to(Path(source).absolute())


def prepare_producer(cfg, work_root):
    """Copy code, never large data; all sed-mutated scripts are private copies."""
    target_root = work_root / "producer"
    if target_root.exists():
        raise DriverError(f"producer workspace already exists: {target_root}")
    for source in producer_sources(cfg):
        require_file(source)
        target = target_root / source.relative_to(cfg.project)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8", errors="surrogateescape")
        # The old drivers have unconditional cluster paths. Rewrite only their
        # private copies, including default paths inside Python converters.
        for old_root in dict.fromkeys((str(cfg.project), "/share/home/lililei1/kcfu/tc_mangkhut")):
            text = text.replace(old_root, str(target_root))
        if source.suffix == ".sh":
            text = re.sub(r"(?m)^export python_bin=.*$",
                          "export python_bin=" + shlex.quote(str(Path(cfg.python).parent)), text)
            text = re.sub(r"(?m)^export rttov_dir=.*$",
                          "export rttov_dir=" + shlex.quote(str(cfg.rttov_root)), text)
        if source.name == "wrf_rttov_d02_75km_ens_interp.m":
            text, count = re.subn(r"wrfout_d0[12]_2018-09-", f"wrfout_{cfg.hx_domain}_2018-09-", text)
            if count < 2:
                raise DriverError(f"unsupported physical H(x) filename layout: {source}")
            text = re.sub(r"firstguess_d0[12]\.", f"firstguess_{cfg.hx_domain}.", text)
        target.write_text(text, encoding="utf-8", errors="surrogateescape")
        target.chmod(0o755 if source.suffix == ".sh" else 0o644)
    run_dir = target_root / "4assimilation/1convert_obs/run_dir"
    for name in ("text_to_obs", "input.nml", "rttov_sensor_db.csv"):
        source = cfg.project / "4assimilation/1convert_obs/run_dir" / name
        require_file(source)
        link_readonly(source, run_dir / name)
    link_readonly(cfg.nr_dir, target_root / "NR_wrfout/2domain")
    return target_root


def observation_environment(cfg, root, cycle, mode):
    env = dict(os.environ)
    # Neutralize inherited overrides that could point at another experiment.
    for key in ("OBS_INPUT_FILE", "OBS_DART_INPUT_FILE", "MERGED_FO_FILE", "OBS_BT_DIR",
                "ENS_BT_DIR", "ADAPTIVE_ACTIVE", "CLEAR_SKY_MASK_FILE", "PARA_FILE",
                "PROFILE_DIR", "PROFILE_SUBDIR", "RAW_ENS_BT_DIR", "ADAPTIVE_LACC_DIR"):
        env.pop(key, None)
    producer = root / "obs" / cycle.tag / "producer"
    env.update({
        "BASE_DIR": str(producer), "PROJECT_ROOT": str(producer),
        "CONVERT_TOP_DIR": str(producer / "4assimilation/1convert_obs"),
        "CONVERT_OUTPUT_DIR": str(producer / "4assimilation/1convert_obs"),
        "EXPERIMENT_MODE": mode, "CENTER_DAY": cycle.time.strftime("%d"),
        "CENTER_HOUR": cycle.time.strftime("%H"), "CENTER_MIN": cycle.time.strftime("%M"),
        "obs_day": cycle.time.strftime("%d"), "obs_hour": cycle.time.strftime("%H"),
        "obs_min": cycle.time.strftime("%M"), "cycle_flag": "1", "profile_matlab_flag": "1",
        "LACC_LAG_HOURS": " ".join(map(str, cycle.lags)), "LACC_TIMES": " ".join(cycle.lacc_tags),
        "ENS_SIZE": "50", "ENS_WRFOUT_DIR": str(root / "hx_inputs" / cycle.tag),
        "PYTHON_BIN": cfg.python, "PYTHONDONTWRITEBYTECODE": "1", "RTTOV_SCATT": "0",
        "USE_TOTAL_ICE": "0", "USE_CLEAR_SKY_MASK": "1", "CLEAR_SKY_MIN_FRACTION": "",
        "CLEAR_SKY_THRESHOLD": str(cfg.clear_threshold), "OBS_ERR_STD": str(cfg.obs_error),
        "HGT_OBS": str(cfg.obs_pressure), "ASSIM_CHANNEL": str(cfg.channel), "OptimizWeight": "0",
        "USE_NR_SST_DIRECTION_FILTER": "1" if mode == "LACC" else "0",
        "NR_SST_BG_DIR": str(root / "prior" / cycle.tag),
        "NR_SST_BG_MEMBER_PATTERN": f"firstguess_{cfg.hx_domain}.mem{{mem:03d}}",
        "NR_SST_NR_FILE": str(cfg.nr_dir / f"wrfout_d02_{stamp(cycle.time)}"),
        "NR_SST_CENTER_TIME": stamp(cycle.time), "OVERWRITE_OBS_SEQ": "1",
        "OBS_SEQ_OUT_NAME": f"obs_seq.{mode}.{cycle.tag}",
        "MPLCONFIGDIR": str(root / "obs" / cycle.tag / "matplotlib"),
        "PATH": str(Path(cfg.python).parent) + os.pathsep + os.environ.get("PATH", ""),
    })
    for key in ("RUN_TRUTH", "RUN_ENSEMBLE", "RUN_OBS_CONVERT", "RUN_HX_MERGE",
                "RUN_TEXT_TO_OBS", "RUN_VALIDATION"):
        env[key] = "1"
    return env


def checked(command, cwd=None, env=None, logfile=None):
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("w") as output:
            result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                                    stdout=output, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise DriverError(f"command failed ({result.returncode}): {shlex.join(list(map(str, command)))}; "
                          + (f"log: {logfile}" if logfile else str(result.stdout)[-3000:]))
    return result


def file_identity(path):
    path = require_file(path)
    info = path.stat()
    return {"path": str(path.absolute()), "source": str(path.resolve()),
            "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def save_checkpoint(path, signature, files):
    atomic_json(path, {"signature": signature,
                       "files": [file_identity(p) for p in sorted(set(map(Path, files)))]})


def checkpoint_ok(path, signature, resume):
    if not path.exists():
        return False
    if not resume:
        raise DriverError(f"stage already exists: {path}; use RESUME=1 to verify and reuse")
    record = json.loads(path.read_text())
    if record.get("signature") != signature:
        raise DriverError(f"checkpoint configuration changed: {path}")
    for saved in record.get("files", []):
        if not Path(saved["path"]).is_file() or file_identity(saved["path"]) != saved:
            raise DriverError(f"checkpoint output changed or missing: {saved['path']}")
    if not record.get("files"):
        raise DriverError(f"checkpoint has no verified outputs: {path}")
    return True


def write_namelist(source, target, overrides):
    """Edit only explicit section.key assignments, leaving science templates intact."""
    wanted = {key.lower(): value for key, value in overrides.items()}
    counts = dict.fromkeys(wanted, 0)
    group = None
    lines = []
    for line in require_file(source).read_text().splitlines(keepends=True):
        opening = re.match(r"^\s*&([A-Za-z_]\w*)\s*$", line)
        if opening:
            group = opening.group(1).lower()
        match = re.match(r"^(\s*)([A-Za-z_]\w*)\s*=", line)
        key = f"{group}.{match.group(2).lower()}" if match else ""
        if key in wanted:
            counts[key] += 1
            line = f"{match.group(1)}{match.group(2)} = {wanted[key]},\n"
        lines.append(line)
        if line.strip() == "/":
            group = None
    bad = [key for key, count in counts.items() if count != 1]
    if bad:
        raise DriverError(f"{source}: missing or repeated namelist assignments: {', '.join(bad)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text("".join(lines))
    temporary.replace(target)


def dart_job_script(cfg, work):
    command = shlex.join(shlex.split(cfg.dart_mpi) + ["./filter"])
    return ("#!/usr/bin/env bash\nset -Eeuo pipefail\n"
            f"cd {shlex.quote(str(work))}\n"
            "ulimit -s unlimited\n"
            f"{command}\n"
            "for kind in prior post; do\n"
            '  file="${kind}_forward_ope_errors000000"\n'
            '  if [[ -f "$file" ]]; then mv "$file" "${kind}_forward_ope_errors"; fi\n'
            "done\n"
            "touch dart_done\n")


def wrf_job_script(cfg, work_root, identity):
    command = shlex.join(shlex.split(cfg.wrf_mpi) + ["./wrf.exe"])
    return ("#!/usr/bin/env bash\nset -Eeuo pipefail\n"
            'mem=$(printf "%03d" "${LSB_JOBINDEX:?missing array index}")\n'
            f"cd {shlex.quote(str(work_root))}/\"${{mem}}\"\n"
            "ulimit -s unlimited\nulimit -d unlimited\n"
            "export I_MPI_DEVICE=rdssm WRFIO_NCD_LARGE_FILE_SUPPORT=1\n"
            f"{command}\n"
            "if ! grep -q 'SUCCESS COMPLETE WRF' rsl.error.0000 2>/dev/null && "
            "! grep -q 'SUCCESS COMPLETE WRF' rsl.out.0000 2>/dev/null; then\n"
            "  echo 'WRF success evidence is absent' >&2; exit 1\nfi\n"
            f"printf '%s\\n' {shlex.quote(identity)} > wrf_done\n")


class LSF:
    def __init__(self, cfg):
        self.cfg = cfg

    def submit(self, work, script, name, tasks, queue, members=()):
        record = work / "lsf_job.json"
        if record.exists():
            existing = json.loads(record.read_text())
            if existing.get("state") == "SUBMITTING":
                raise DriverError(f"submission interrupted without a confirmed job ID: {record}; "
                                  "reconcile with LSF before retrying")
            return str(existing["job_id"])
        work.mkdir(parents=True, exist_ok=True)
        (work / "job.sh").write_text(script)
        atomic_json(record, {"state": "SUBMITTING", "members": list(members), "name": name})
        job_name = name + ("[" + ",".join(map(str, members)) + "]" if members else "")
        command = ["bsub", "-J", job_name, "-n", str(tasks), "-q", queue,
                   "-R", f"span[ptile={self.cfg.ptile}]",
                   "-oo", str(work / "job_%I.out"), "-eo", str(work / "job_%I.err")]
        result = subprocess.run(command, input=script, cwd=work,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = result.stdout or ""
        (work / "submission.log").write_text(output)
        matches = re.findall(r"Job\s*<(\d+)>", output)
        if result.returncode or len(matches) != 1:
            raise DriverError(f"unconfirmed LSF submission; reconcile {record} before retrying: {output}")
        atomic_json(record, {"state": "SUBMITTED", "job_id": matches[0],
                             "members": list(members), "name": name})
        log(f"LSF submitted {job_name}: job {matches[0]}")
        return matches[0]

    def states(self, job_id, members=()):
        expected = [f"{job_id}[{m}]" for m in members] if members else [str(job_id)]
        result = subprocess.run(["bjobs", "-a", "-noheader", "-o", "jobid stat", str(job_id)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        found = {}
        for line in (result.stdout or "").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] in expected:
                found[fields[0]] = fields[1]
        for identifier in expected:
            if identifier not in found:
                hist = subprocess.run(["bhist", "-l", identifier], stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
                events = []
                for line in (hist.stdout or "").splitlines():
                    if "Done successfully" in line:
                        events.append("DONE")
                    elif re.search(r"\bExited\b", line):
                        events.append("EXIT")
                if events:
                    found[identifier] = events[-1]
        return [found.get(identifier) for identifier in expected]

    def wait(self, job_id, members=(), allow_failure=False):
        started, unknown = time.monotonic(), 0
        while True:
            states = self.states(job_id, members)
            if any(state is None for state in states):
                unknown += 1
                if unknown >= 5:
                    raise DriverError(f"cannot determine LSF job {job_id} state; refusing unknown success")
            else:
                unknown = 0
                if all(state in ("DONE", "EXIT") for state in states):
                    success = all(state == "DONE" for state in states)
                    if not success and not allow_failure:
                        raise JobFailed(f"LSF job {job_id} reached EXIT")
                    return success
                if any(state not in ("DONE", "EXIT", "PEND", "RUN", "PSUSP", "USUSP", "SSUSP", "WAIT")
                       for state in states):
                    raise DriverError(f"unexpected LSF state for {job_id}: {states}")
            if time.monotonic() - started >= self.cfg.max_wait:
                # Cancel exactly this submitted ID, then prove it has stopped
                # before any caller could reuse or clean its working files.
                self.cancel(job_id)
                for _ in range(12):
                    if all(s in ("DONE", "EXIT") for s in self.states(job_id, members)):
                        if allow_failure:
                            return False
                        raise JobFailed(f"LSF job {job_id} timed out and was stopped")
                    time.sleep(min(5, self.cfg.poll))
                raise DriverError(f"LSF job {job_id} timed out; cannot prove termination; keep work intact")
            time.sleep(self.cfg.poll)

    def cancel(self, job_id):
        if not re.fullmatch(r"\d+", str(job_id)):
            raise DriverError(f"invalid numeric LSF job ID: {job_id}")
        subprocess.run(["bkill", "-r", str(job_id)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def validate_dart_outputs(work, cycle, ocean):
    import netCDF4 as nc
    required = (ATM_VARS + ("," + OCEAN_VARS if ocean else "")).split(",")
    paths = []
    for domain in DOMAINS:
        for member in MEMBERS:
            path = require_file(work / f"output_{domain}.mem{member:03d}")
            with nc.Dataset(path) as ds:
                missing = set(required) - set(ds.variables)
                if missing:
                    raise DriverError(f"{path}: missing DART output variables {sorted(missing)}")
                # DART restart/state files may omit Times; full WRF files do not.
                if "Times" in ds.variables and read_nc_times(ds) != (cycle.time,):
                    raise DriverError(f"{path}: DART output valid time mismatch")
            paths.append(path)
    return paths


def experiment_record(root):
    record = json.loads(require_file(root / "experiment.json").read_text())
    return record, Experiment(record["group"], record["filter"])


def job_name(root, cycle, stage):
    token = hashlib.sha256(str(root).encode()).hexdigest()[:10]
    return f"c50_{token}_{cycle.tag}_{stage}"


def run_dart_stage(cfg, root, cycle, stage, template, obs_file):
    record, exp = experiment_record(root)
    area = root / "dart" / cycle.tag / stage
    work = area / "work"
    checkpoint = area / "done.json"
    signature = record["signature"] + ":" + cycle.tag + ":" + stage
    ocean = template != "input.nml.air"
    if checkpoint_ok(checkpoint, signature, cfg.resume):
        log(f"{exp.name} {cycle.tag}: verified DART {stage} checkpoint")
        validate_dart_outputs(work, cycle, ocean)
        return work
    if (work / "failed").exists() and cfg.resume:
        work.rename(area / f"failed_{time.time_ns()}")
    job_record = work / "lsf_job.json"
    if not job_record.exists():
        if work.exists():
            shutil.rmtree(work)  # preparation was interrupted before submission
        work.mkdir(parents=True)
        for name in ("filter",):
            link_readonly(cfg.dart_work / name, work / name)
        for name in ("rttov_sensor_db.csv", "rtcoef_noaa_18_amsua.dat"):
            link_readonly(cfg.templates / name, work / name)
        if exp.filter == "QCF_RHF":
            shutil.copy2(cfg.templates / "qceff_table_fkc.csv", work / "qceff_table_fkc.csv")
        for domain in DOMAINS:
            inputs = [prior_file(root, cycle.time, domain, m) for m in MEMBERS]
            for path in inputs:
                validate_state(path, cycle.time)
            (work / f"input_list_{domain}.txt").write_text("".join(str(p) + "\n" for p in inputs))
            (work / f"output_list_{domain}.txt").write_text(
                "".join(f"output_{domain}.mem{m:03d}\n" for m in MEMBERS))
            link_readonly(inputs[0], work / f"wrfinput_{domain}")
        # A real copy fixes this stage's observation input even if a producer is
        # retried later; preserve the untouched template as well as the edited one.
        shutil.copy2(obs_file, work / "obs_seq.out")
        shutil.copy2(cfg.templates / template, work / "template.nml")
        write_namelist(cfg.templates / template, work / "input.nml", {
            "filter_nml.ens_size": "50",
            "filter_nml.input_state_file_list": "'input_list_d01.txt', 'input_list_d02.txt'",
            "filter_nml.output_state_file_list": "'output_list_d01.txt', 'output_list_d02.txt'",
            "model_nml.num_domains": "2", "model_nml.assimilation_period_seconds": "3600",
            "algorithm_info_nml.qceff_table_filename": "'qceff_table_fkc.csv'" if exp.filter == "QCF_RHF" else "''",
        })
    scheduler = LSF(cfg)
    job_id = scheduler.submit(work, dart_job_script(cfg, work), job_name(root, cycle, stage),
                              cfg.dart_tasks, cfg.dart_queue)
    try:
        scheduler.wait(job_id)
    except JobFailed as exc:
        (work / "failed").write_text(str(exc))
        raise
    try:
        if not (work / "dart_done").is_file():
            raise DriverError(f"DART job {job_id}: missing successful-completion marker")
        outputs = validate_dart_outputs(work, cycle, ocean)
    except DriverError as exc:
        (work / "failed").write_text(str(exc))
        raise
    save_checkpoint(checkpoint, signature, outputs + [work / "input.nml", work / "template.nml",
                                                     work / "obs_seq.out", obs_file])
    return work


def assimilate(cfg, root, cycle, obs_files):
    record, exp = experiment_record(root)
    destination = root / "analysis" / cycle.tag
    checkpoint = destination / "done.json"
    signature = record["signature"] + ":" + cycle.tag + ":analysis"
    if checkpoint_ok(checkpoint, signature, cfg.resume):
        log(f"{exp.name} {cycle.tag}: verified full WRF analysis checkpoint")
        return
    ocean_work = None
    if exp.update_ocean:
        ocean_work = run_dart_stage(cfg, root, cycle, "ocean", "input.nml.inflateOcean", obs_files["LACC"])
    atmosphere_work = run_dart_stage(cfg, root, cycle, "atmosphere",
                                    "input.nml.all" if exp.update_ocean else "input.nml.air", obs_files["STANDARD"])
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for domain in DOMAINS:
        for member in MEMBERS:
            name = f"{domain}.mem{member:03d}"
            path = destination / f"analysis_{name}"
            temporary = destination / f"analysis_{name}.tmp"
            shutil.copyfile(prior_file(root, cycle.time, domain, member), temporary)
            checked(["ncks", "-A", "-v", ATM_VARS, atmosphere_work / f"output_{name}", temporary])
            if ocean_work:
                # Discard the second-pass ocean, exactly as the six-member
                # workflow does. Transplant only the four first-pass fields.
                checked(["ncks", "-A", "-v", OCEAN_VARS, ocean_work / f"output_{name}", temporary])
            validate_state(temporary, cycle.time, (ATM_VARS + "," + OCEAN_VARS).split(","))
            temporary.replace(path)
            outputs.append(path)
    save_checkpoint(checkpoint, signature, outputs)
    log(f"{exp.name} {cycle.tag}: complete 50-member, two-domain analysis")


def prepare_wrf_member(cfg, root, cycle, member):
    _, exp = experiment_record(root)
    work = root / "run_wrf" / cycle.tag / f"{member:03d}"
    if work.exists():
        old_logs = root / "logs" / cycle.tag / f"{member:03d}_{time.time_ns()}"
        old_logs.mkdir(parents=True)
        for path in work.iterdir():
            if path.name.startswith("rsl.") and path.is_file():
                shutil.copy2(path, old_logs / path.name)
        shutil.rmtree(work)
    # A crash during archiving can leave only some files here. Remove that
    # uncommitted partial archive before starting a fresh member attempt.
    archive = root / "wrfout" / cycle.tag / f"{member:03d}"
    if archive.exists():
        shutil.rmtree(archive)
    work.mkdir(parents=True)
    for source in wrf_static_entries(cfg):
        link_readonly(source, work / source.name)
    for domain in DOMAINS:
        source = root / "analysis" / cycle.tag / f"analysis_{domain}.mem{member:03d}"
        validate_state(source, cycle.time)
        shutil.copyfile(source, work / f"wrfinput_{domain}")
    shutil.copyfile(cfg.bc(member), work / "wrfbdy_d01")
    link_readonly(cfg.bc(member), work / "wrfbdy_d01_orig")
    link_readonly(cfg.dart_work / "update_wrf_bc", work / "update_wrf_bc")
    shutil.copy2(cfg.project / "0necessay_files/input.nml", work / "input.nml")
    overrides = {"time_control.run_days": "0", "time_control.run_hours": str(cycle.hours),
                 "time_control.run_minutes": "0", "time_control.run_seconds": "0",
                 "time_control.history_interval": "30, 30", "time_control.frames_per_outfile": "1, 1",
                 "time_control.restart": ".false.", "time_control.input_from_file": ".true., .true.",
                 "domains.max_dom": "2", "physics.sf_ocean_physics": str(2 if exp.run_ocean else 0)}
    for prefix, moment in (("start", cycle.time), ("end", cycle.end)):
        for part, value in (("year", moment.year), ("month", moment.month), ("day", moment.day),
                            ("hour", moment.hour), ("minute", moment.minute), ("second", moment.second)):
            overrides[f"time_control.{prefix}_{part}"] = f"{value}, {value}"
    write_namelist(cfg.project / "0necessay_files/namelist.input.ens", work / "namelist.input", overrides)
    checked([work / "update_wrf_bc"], cwd=work, logfile=work / "update_bc.log")
    link_readonly(cfg.ic(member), work / "wrfinput_d01_gfs")
    for domain in DOMAINS:
        helper = cfg.project / f"0necessay_files/ad_omini_{domain}.py"
        checked([cfg.python, helper], cwd=work, logfile=work / f"ad_omini_{domain}.log")
        validate_state(work / f"wrfinput_{domain}", cycle.time, ("OM_TINI", "OM_SINI"))


def expected_wrf_files(directory, cycle):
    for minute in range(0, cycle.hours * 60 + 1, 30):
        moment = cycle.time + timedelta(minutes=minute)
        for domain in DOMAINS:
            yield directory / f"wrfout_{domain}_{stamp(moment)}", moment


def member_ready(root, cycle, member, identity):
    work = root / "run_wrf" / cycle.tag / f"{member:03d}"
    archive = root / "wrfout" / cycle.tag / f"{member:03d}"
    checkpoint = archive / "member_done.json"
    if checkpoint_ok(checkpoint, identity, True):
        return True
    marker = work / "wrf_done"
    if not marker.is_file() or marker.read_text().strip() != identity:
        return False
    try:
        for target, moment in expected_wrf_files(archive, cycle):
            source = target if target.exists() else work / target.name
            validate_state(source, moment, ("TSK",))
    except (DriverError, OSError):
        return False
    return True


def publish_next_prior(root, cycle, domain, member, archived):
    """Ocean-off WRF omits OM_*: carry its unadvanced state in a prior COPY."""
    _, exp = experiment_record(root)
    target = prior_file(root, cycle.end, domain, member)
    if exp.run_ocean:
        validate_state(archived, cycle.end, OCEAN_VARS.split(","))
        link_readonly(archived, target)
        return
    import netCDF4 as nc
    import numpy as np
    previous = prior_file(root, cycle.time, domain, member)
    validate_state(previous, cycle.time, OCEAN_VARS.split(","))
    validate_state(archived, cycle.end)
    source_identity = json.dumps(file_identity(previous), sort_keys=True)
    if target.is_file() and not target.is_symlink():
        validate_state(target, cycle.end, OCEAN_VARS.split(","))
        with nc.Dataset(target) as ds:
            if getattr(ds, "CYCLE_FROZEN_OCEAN_SOURCE", "") != source_identity:
                raise DriverError(f"frozen-ocean prior has a different source: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(archived, temporary)
    with nc.Dataset(previous) as source, nc.Dataset(temporary, "a") as destination:
        for coordinate in ("XLAT", "XLONG"):
            if coordinate not in source.variables or coordinate not in destination.variables:
                raise DriverError(f"cannot carry frozen ocean without {coordinate}: {previous}, {archived}")
            if not np.array_equal(source[coordinate][:], destination[coordinate][:]):
                raise DriverError(f"cannot carry frozen ocean across a changed grid: {domain} mem{member:03d}")
        fields = []
        for name in OCEAN_VARS.split(","):
            original = source[name]
            for dimension in original.dimensions:
                if dimension not in destination.dimensions:
                    destination.createDimension(dimension, len(source.dimensions[dimension]))
                elif len(destination.dimensions[dimension]) != len(source.dimensions[dimension]):
                    raise DriverError(f"incompatible frozen-ocean dimension {dimension}: {archived}")
            if name not in destination.variables:
                options = {"fill_value": original.getncattr("_FillValue")} if "_FillValue" in original.ncattrs() else {}
                copied = destination.createVariable(name, original.datatype, original.dimensions, **options)
                copied.setncatts({key: original.getncattr(key) for key in original.ncattrs() if key != "_FillValue"})
            else:
                copied = destination[name]
            fields.append((original, copied))
        # Finish all definitions before data I/O: classic NetCDF otherwise
        # repeatedly shifts a gigabyte-scale file as its header grows.
        destination.CYCLE_FROZEN_OCEAN_SOURCE = source_identity
        destination.CYCLE_OCEAN_STATE = "unadvanced ocean: update_ocean=0, run_ocean=0"
        for original, copied in fields:
            # Bound peak memory to one ocean level, not a full 30-level field.
            if original.ndim >= 2:
                for level in range(original.shape[1]):
                    copied[:, level, ...] = original[:, level, ...]
            else:
                copied[:] = original[:]
    validate_state(temporary, cycle.end, OCEAN_VARS.split(","))
    if target.is_symlink():
        target.unlink()
    temporary.replace(target)


def archive_member(root, cycle, member, identity):
    work = root / "run_wrf" / cycle.tag / f"{member:03d}"
    archive = root / "wrfout" / cycle.tag / f"{member:03d}"
    checkpoint = archive / "member_done.json"
    if not checkpoint_ok(checkpoint, identity, True):
        archive.mkdir(parents=True, exist_ok=True)
        for source in sorted(work.glob("wrfout_d0*")):
            target = archive / source.name
            if target.exists():
                raise DriverError(f"conflicting WRF archive file: {target}")
            source.replace(target)
        files = []
        for path, moment in expected_wrf_files(archive, cycle):
            validate_state(path, moment, ("TSK",))
            files.append(path)
        for domain in DOMAINS:
            publish_next_prior(root, cycle, domain, member, archive / f"wrfout_{domain}_{stamp(cycle.end)}")
        files += [prior_file(root, cycle.end, domain, member) for domain in DOMAINS]
        save_checkpoint(checkpoint, identity, files)
    return [archive / path.name for path, _ in expected_wrf_files(archive, cycle)]


def forecast(cfg, root, cycle):
    record, exp = experiment_record(root)
    identity = record["signature"] + ":" + cycle.tag
    checkpoint = root / "wrfout" / cycle.tag / "done.json"
    if checkpoint_ok(checkpoint, identity + ":forecast", cfg.resume):
        log(f"{exp.name} {cycle.tag}: verified archived WRF forecast checkpoint")
        return
    scheduler = LSF(cfg)
    jobs = root / "jobs" / cycle.tag
    # On resume, wait for every previously submitted array to be definitively
    # terminal before inspecting, deleting or retrying any member files.
    previous = sorted(jobs.glob("attempt_*/lsf_job.json"))
    previous_states = {}
    for path in previous:
        job = json.loads(path.read_text())
        if job.get("state") == "SUBMITTING":
            raise DriverError(f"unconfirmed array submission; reconcile with LSF: {path}")
        members = tuple(job["members"])
        succeeded = scheduler.wait(str(job["job_id"]), members, allow_failure=True)
        states = ["DONE"] * len(members) if succeeded else scheduler.states(str(job["job_id"]), members)
        previous_states.update(zip(members, states))
    ready = {m for m in MEMBERS if previous_states.get(m, "DONE") == "DONE"
             and member_ready(root, cycle, m, identity)}
    for attempt in range(len(list(jobs.glob("attempt_*"))) + 1, cfg.wrf_attempts + 1):
        pending = tuple(m for m in MEMBERS if m not in ready)
        if not pending:
            break
        for member in pending:
            prepare_wrf_member(cfg, root, cycle, member)
        area = jobs / f"attempt_{attempt:02d}"
        script = wrf_job_script(cfg, root / "run_wrf" / cycle.tag, identity)
        job_id = scheduler.submit(area, script, job_name(root, cycle, f"wrf{attempt}"),
                                  cfg.wrf_tasks, cfg.wrf_queue, pending)
        succeeded = scheduler.wait(job_id, pending, allow_failure=True)
        states = ["DONE"] * len(pending) if succeeded else scheduler.states(job_id, pending)
        for member, state in zip(pending, states):
            if state == "DONE" and member_ready(root, cycle, member, identity):
                archive_member(root, cycle, member, identity)
                ready.add(member)
        log(f"{exp.name} {cycle.tag}: WRF attempt {attempt}, {len(ready)}/{len(MEMBERS)} members complete")
    if ready != set(MEMBERS):
        raise DriverError(f"{exp.name} {cycle.tag}: WRF failed after {cfg.wrf_attempts} attempts: "
                          + ", ".join(f"mem{m:03d}" for m in MEMBERS if m not in ready))
    outputs = []
    for member in MEMBERS:
        outputs += archive_member(root, cycle, member, identity)
        outputs += [prior_file(root, cycle.end, domain, member) for domain in DOMAINS]
    save_checkpoint(checkpoint, identity + ":forecast", outputs)


def produce_observations(cfg, root, cycle):
    work = root / "obs" / cycle.tag
    if work.exists():
        shutil.rmtree(work)  # only an unfinished cycle's private production tree
    producer = prepare_producer(cfg, work)
    for member in MEMBERS:
        for moment in cycle.lacc_times:
            source = input_source(cfg, root, cycle, moment, cfg.hx_domain, member)
            validate_state(source, moment, ("TSK", "OM_TMP"))
            target = root / "hx_inputs" / cycle.tag / f"mem{member:03d}" / f"wrfout_{cfg.hx_domain}_{stamp(moment)}"
            link_readonly(source, target)
    files = {}
    for mode in ("STANDARD", "LACC"):
        log(f"{root.name} {cycle.tag}: generating {mode} truth, H(x), mask and obs_seq")
        env = observation_environment(cfg, root, cycle, mode)
        checked(["bash", producer / "4assimilation/1convert_obs/run_single_assimilation_test.sh"],
                env=env, logfile=work / f"{mode}.log")
        name = env["OBS_SEQ_OUT_NAME"] + ("_nr_sst_direction" if mode == "LACC" else "")
        files[mode] = producer / "4assimilation/1convert_obs/run_dir" / name
        require_file(files[mode])
    return files


def experiment_signature(cfg, exp):
    parameters = dataclasses.asdict(cfg)
    for name in ("resume", "groups", "filters", "poll", "max_wait", "wrf_attempts"):
        parameters.pop(name, None)
    # Large fields are fingerprinted by resolved filename, length and mtime;
    # source code/templates additionally use content hashes. Never hash TB of
    # input fields just to resume a serial driver.
    inputs = [cfg.bc(m) for m in MEMBERS] + [cfg.ic(m) for m in MEMBERS]
    inputs += [cfg.initial_dir / f"firstguess_{dom}.mem{m:03d}" for dom in DOMAINS for m in MEMBERS]
    inputs += [cfg.history_dir / f"mem{m:03d}" / f"wrfout_{cfg.hx_domain}_{stamp(START - timedelta(hours=h))}"
               for m in MEMBERS for h in (3, 6)]
    inputs += [cfg.nr_dir / f"wrfout_d02_{stamp(t)}" for t in sorted({t for c in cycles() for t in c.lacc_times})]
    code = producer_sources(cfg)
    code += [cfg.project / f"0necessay_files/{name}" for name in
             ("input.nml", "namelist.input.ens", "ad_omini_d01.py", "ad_omini_d02.py")]
    code += [cfg.templates / name for name in
             (("input.nml.inflateOcean", "input.nml.all") if exp.update_ocean else ("input.nml.air",))]
    code += [cfg.project / "4assimilation/1convert_obs/run_dir/input.nml",
             cfg.project / "4assimilation/1convert_obs/run_dir/rttov_sensor_db.csv"]
    code += [cfg.templates / name for name in ("rttov_sensor_db.csv", "rtcoef_noaa_18_amsua.dat")]
    # These are the actual physics tables/binaries linked or invoked at runtime,
    # including the targets of symlinks. Content hashes detect even changes that
    # preserve a table's length and modification timestamp.
    for entry in wrf_static_entries(cfg):
        code += [entry] if entry.is_file() else [p for p in entry.rglob("*") if p.is_file()]
    code += list(rttov_files(cfg))
    if exp.filter == "QCF_RHF":
        code.append(cfg.templates / "qceff_table_fkc.csv")
    inputs += [cfg.dart_work / "filter", cfg.dart_work / "update_wrf_bc", cfg.wrf_static / "wrf.exe",
               cfg.project / "4assimilation/1convert_obs/run_dir/text_to_obs"]
    for command in ("matlab", "ncks", shlex.split(cfg.dart_mpi)[0], shlex.split(cfg.wrf_mpi)[0]):
        executable = shutil.which(command)
        if executable and Path(executable).is_file():
            code.append(Path(executable))
    driver = Path(os.environ.get("CYCLE_DRIVER_SOURCE", __file__)).absolute()
    if driver.is_file():
        code.append(driver)
    snapshot = {"config": parameters, "experiment": dataclasses.asdict(exp),
                "inputs": [file_identity(p) for p in inputs],
                "code": [{**file_identity(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in sorted(set(code))]}
    signature = hashlib.sha256(json.dumps(snapshot, default=str, sort_keys=True).encode()).hexdigest()
    return signature, snapshot


def wrf_static_entries(cfg):
    excluded = ("wrfinput", "wrfbdy", "wrfout", "wrfrst", "rsl.", "namelist.", "met_em")
    return sorted(p for p in cfg.wrf_static.iterdir() if not p.name.startswith(excluded))


def rttov_files(cfg):
    return (cfg.rttov_root / "bin/example_fwd.exe",
            cfg.rttov_root / "rtcoef_rttov12/rttov7pred54L/rtcoef_noaa_18_amsua.dat")


def initialize_experiment(cfg, exp):
    root = cfg.root / exp.name
    signature, snapshot = experiment_signature(cfg, exp)
    record_path = root / "experiment.json"
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if not cfg.resume:
            raise DriverError(f"experiment exists: {root}; use RESUME=1")
        if old.get("signature") != signature:
            raise DriverError(f"experiment inputs/configuration changed: {root}; choose a new RUN_ROOT")
    else:
        if root.exists() and any(root.iterdir()):
            raise DriverError(f"unowned nonempty experiment directory: {root}")
        # Convert Path objects before persisting the reviewable snapshot.
        snapshot = json.loads(json.dumps(snapshot, default=str))
        atomic_json(record_path, {"signature": signature, "group": exp.group,
                                  "filter": exp.filter, "snapshot": snapshot})
    for domain in DOMAINS:
        for member in MEMBERS:
            link_readonly(cfg.initial_dir / f"firstguess_{domain}.mem{member:03d}",
                          prior_file(root, START, domain, member))
    return root, signature


def run(cfg):
    import fcntl
    preflight(cfg)  # all external inputs and BCs, before ANY runtime writes/jobs
    cfg.root.mkdir(parents=True, exist_ok=True)
    with (cfg.root / ".driver.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DriverError(f"another driver owns RUN_ROOT: {cfg.root}") from exc
        for exp in experiments(cfg):
            root, signature = initialize_experiment(cfg, exp)
            log(f"Starting {exp.name}")
            for cycle in cycles():
                if experiment_signature(cfg, exp)[0] != signature:
                    raise DriverError(f"external inputs/runtime dependencies changed during the experiment: {root}")
                obs_checkpoint = root / "obs" / cycle.tag / "done.json"
                obs_signature = signature + ":" + cycle.tag + ":observations"
                if checkpoint_ok(obs_checkpoint, obs_signature, cfg.resume):
                    base = root / "obs" / cycle.tag / "producer/4assimilation/1convert_obs/run_dir"
                    obs_files = {"STANDARD": base / f"obs_seq.STANDARD.{cycle.tag}",
                                 "LACC": base / f"obs_seq.LACC.{cycle.tag}_nr_sst_direction"}
                    # Saved paths are canonical, also for producer implementations
                    # returning files elsewhere in the private work area.
                    meta = json.loads(obs_checkpoint.read_text())
                    if "obs_files" in meta:
                        obs_files = {key: Path(value) for key, value in meta["obs_files"].items()}
                else:
                    obs_files = produce_observations(cfg, root, cycle)
                    outputs = [p for p in (root / "obs" / cycle.tag).rglob("*") if p.is_file() and p.stat().st_size]
                    save_checkpoint(obs_checkpoint, obs_signature, outputs)
                    saved = json.loads(obs_checkpoint.read_text())
                    saved["obs_files"] = {key: str(value) for key, value in obs_files.items()}
                    atomic_json(obs_checkpoint, saved)
                assimilate(cfg, root, cycle, obs_files)
                forecast(cfg, root, cycle)
            atomic_json(root / "complete.json", {"signature": signature, "last_forecast": stamp(cycles()[-1].end)})
            log(f"Completed {exp.name}: final free forecast reaches {stamp(cycles()[-1].end)}")


def main():
    cfg = Config.from_env()
    mode = os.environ.get("MODE", "RUN")
    if mode == "DRY_RUN":
        print(json.dumps(plan(cfg), indent=2))
    elif mode == "PREFLIGHT":
        preflight(cfg)
    elif mode == "RUN":
        run(cfg)
    else:
        raise DriverError("MODE must be RUN, PREFLIGHT or DRY_RUN")


if __name__ == "__main__":
    try:
        main()
    except (DriverError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
CYCLE_PYTHON
