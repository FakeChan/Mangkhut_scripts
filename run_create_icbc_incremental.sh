#!/usr/bin/env bash
#BSUB -q mpi
#BSUB -n 48
#BSUB -J icbc_extend_06_12
#BSUB -oo icbc_incremental.out
#BSUB -eo icbc_incremental.err
#
# Incremental 50-member IC/BC extension, default 2018-09-10 06 -> 12 UTC.
# Edit settings here. Inspect with MODE=DRY_RUN / PREFLIGHT; production:
#     MODE=PREFLIGHT bash run_create_icbc_incremental.sh
#     bsub < run_create_icbc_incremental.sh
# RUN requires an LSF allocation for mpirun. No nested jobs are submitted.
# Existing 06 ICs and old BC fragments are never regenerated/deleted.
# A valid existing 12 IC or 06 fragment is reused; an already-covered BC is skipped.
# New ICs: 1icbc/memNNN/IC/wrfinput_2018091012_memNNN
# New BCs: 1icbc/memNNN/BC/wrfbdy_d01_memNNN_2018091006
# Extended BC: 1icbc/memNNN/BC/wrfbdy_d01_memNNN; backups: BC/backups/
# All candidates are built in fresh scratch directories and validated BEFORE
# any canonical files are added/replaced. Combined BCs receive byte-exact
# backups before atomic replacement. Failed runs retain their private logs.
# Rerunning detects published outputs by timestamps/coverage, never appends twice.
set -Eeuo pipefail

# =========================== user configuration =============================
export MODE="${MODE:-RUN}"                         # DRY_RUN / PREFLIGHT / RUN
export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/lililei1/kcfu/tc_mangkhut}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/lililei1/kcfu/tc_mangkhut}"
export ICBC_ROOT="${ICBC_ROOT:-${PROJECT_ROOT}/1icbc}"
export STAGE_ROOT="${STAGE_ROOT:-${SCRATCH_ROOT}/1icbc_incremental}"
export START_TIME_UTC="${START_TIME_UTC:-2018-09-10_06:00:00}"
export END_TIME_UTC="${END_TIME_UTC:-2018-09-10_12:00:00}"
export RAW_BC="${RAW_BC:-${PROJECT_ROOT}/0necessay_files/wrfbdy_d01_trfollow}"
# Empty means derive wrfinput_d01_YYYYMMDDHH from END_TIME_UTC.
export END_REFERENCE_IC="${END_REFERENCE_IC:-}"
export WRFDA_TEMPLATE="${WRFDA_TEMPLATE:-${PROJECT_ROOT}/0necessay_files/namelist.WRFDA.VAR}"
export WRFDA_EXE="${WRFDA_EXE:-/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/DA/WRFDAV4/var/build/da_wrfvar.exe}"
export WRFDA_SUPPORT="${WRFDA_SUPPORT:-/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/DA/WRFDAV4/var/run}"
export DART_WORK="${DART_WORK:-/share/home/lililei1/kcfu/models/DART_main/models/wrf/work}"
export DART_INPUT_NML="${DART_INPUT_NML:-${DART_WORK}/input.nml}"
export MPI_LAUNCH="${MPI_LAUNCH:-mpirun}"
export PYTHON_EXE="${PYTHON_EXE:-/share/home/lililei1/kcfu/anaconda/envs/wrf/bin/python}"
export ICBC_INCREMENTAL_SOURCE="${BASH_SOURCE[0]}"
# ===========================================================================
exec "${PYTHON_EXE}" - <<'ICBC_PYTHON'
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import netCDF4 as nc
import numpy as np

START = datetime(2018, 9, 10, 6, tzinfo=timezone.utc)
END = datetime(2018, 9, 10, 12, tzinfo=timezone.utc)
MEMBERS = tuple(range(1, 51))
THIS = "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
NEXT = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
IC_VARIABLES = ("U", "V", "W", "T", "PH", "MU", "MUB", "MAPFAC_U", "MAPFAC_V", "MAPFAC_M", "XLAT", "XLONG")
# pert_wrf_bc skips moisture species absent from the input; validate every
# species present and require the same set in both endpoint ICs.
IC_OPTIONAL = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE")


class IncrementalError(RuntimeError):
    pass


def stamp(moment):
    return moment.strftime("%Y-%m-%d_%H:%M:%S")


def time_tag(moment):
    return moment.strftime("%Y%m%d%H")


def parse_time(value):
    return datetime.strptime(value, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise IncrementalError(f"missing or empty input: {path}")
    return path


def nc_times(dataset, name="Times"):
    if name not in dataset.variables:
        raise IncrementalError(f"{dataset.filepath()}: missing {name}")
    return tuple(parse_time(str(s).strip().strip("\0"))
                 for s in nc.chartostring(dataset[name][:]).reshape(-1))


def records(path):
    with nc.Dataset(require_file(path)) as ds:
        times, begins, ends = nc_times(ds), nc_times(ds, THIS), nc_times(ds, NEXT)
    if not times or times != begins or len(begins) != len(ends):
        raise IncrementalError(f"{path}: Times/THISBDYTIME/NEXTBDYTIME disagree")
    pairs = tuple(zip(begins, ends))
    for i, (begin, end) in enumerate(pairs):
        if end <= begin:
            raise IncrementalError(f"{path}: nonpositive boundary interval")
        if i:
            previous_begin, previous_end = pairs[i - 1]
            if begin <= previous_begin:
                raise IncrementalError(f"{path}: duplicate/out-of-order boundary records")
            if begin != previous_end:
                raise IncrementalError(f"{path}: boundary gap/overlap before {stamp(begin)}")
    return pairs


def find_record(path, start, end):
    pairs = records(path)
    found = [i for i, pair in enumerate(pairs) if pair == (start, end)]
    if len(found) != 1:
        raise IncrementalError(f"{path}: require one exact {stamp(start)} -> {stamp(end)} record")
    return found[0]  # ncks defaults to zero-based indexing; do not add -F.


def validate_ic(path, moment):
    with nc.Dataset(require_file(path)) as ds:
        if nc_times(ds) != (moment,):
            raise IncrementalError(f"{path}: IC valid time must be {stamp(moment)}")
        missing = set(IC_VARIABLES) - set(ds.variables)
        if missing:
            raise IncrementalError(f"{path}: missing IC fields {sorted(missing)}")
        fields = IC_VARIABLES + tuple(name for name in IC_OPTIONAL if name in ds.variables)
        for name in fields:
            variable = ds[name]
            if not variable.dimensions or variable.dimensions[0] != "Time" or any(n == 0 for n in variable.shape):
                raise IncrementalError(f"{path}: invalid IC dimensions for {name}")
        validate_numeric(ds, fields)


def validate_numeric(dataset, names, indices=None):
    """Read one time record at a time; reject NaN/Inf and masked fill values."""
    for name in names:
        variable = dataset[name]
        if np.dtype(variable.dtype).kind not in "iuf":
            raise IncrementalError(f"{dataset.filepath()}: nonnumeric field {name}")
        if "Time" in variable.dimensions:
            axis = variable.dimensions.index("Time")
            selections = []
            for index in indices if indices is not None else range(variable.shape[axis]):
                selection = [slice(None)] * variable.ndim
                selection[axis] = index
                selections.append(tuple(selection))
        else:
            selections = [Ellipsis]
        for selection in selections:
            values = variable[selection]
            if np.ma.is_masked(values) or not np.all(np.isfinite(np.asarray(values))):
                raise IncrementalError(f"{dataset.filepath()}: missing/nonfinite data in {name}")


def validate_bc(path, reference=None, indices=None):
    with nc.Dataset(require_file(path)) as ds:
        fields = [name for name in ds.variables if re.search(r"_BT?[XY][SE]$", name)]
        if not fields:
            raise IncrementalError(f"{path}: no boundary numerical fields")
        if reference is not None:
            with nc.Dataset(reference) as old:
                if set(ds.variables) != set(old.variables):
                    raise IncrementalError(f"{path}: boundary variable set differs from {reference}")
                for name, variable in old.variables.items():
                    other = ds[name]
                    if variable.dimensions != other.dimensions or variable.dtype != other.dtype:
                        raise IncrementalError(f"{path}: incompatible boundary dimensions/type for {name}")
                    for dim in variable.dimensions:
                        if dim != "Time" and len(ds.dimensions[dim]) != len(old.dimensions[dim]):
                            raise IncrementalError(f"{path}: incompatible boundary dimension {dim}")
        for name in fields:
            if not ds[name].dimensions or ds[name].dimensions[0] != "Time" or any(n == 0 for n in ds[name].shape):
                raise IncrementalError(f"{path}: invalid boundary dimensions for {name}")
        validate_numeric(ds, fields, indices)


def same_grid(first, second):
    with nc.Dataset(first) as a, nc.Dataset(second) as b:
        for name in IC_VARIABLES + IC_OPTIONAL:
            if (name in a.variables) != (name in b.variables):
                raise IncrementalError(f"IC variable sets disagree ({name}): {first}, {second}")
            if name not in a.variables:
                continue
            if a[name].dimensions != b[name].dimensions or a[name].shape != b[name].shape:
                raise IncrementalError(f"IC dimensions disagree ({name}): {first}, {second}")
        for name in ("XLAT", "XLONG"):
            if not np.array_equal(a[name][:], b[name][:]):
                raise IncrementalError(f"IC grids disagree ({name}): {first}, {second}")


@dataclass(frozen=True)
class Config:
    project: Path
    scratch: Path
    icbc_root: Path | None = None
    stage_root: Path | None = None
    start: datetime = START
    end: datetime = END
    raw_bc: Path | None = None
    reference_ic: Path | None = None
    wrfda_template: Path | None = None
    wrfda_exe: Path = Path("/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/DA/WRFDAV4/var/build/da_wrfvar.exe")
    wrfda_support: Path = Path("/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/DA/WRFDAV4/var/run")
    dart_work: Path = Path("/share/home/lililei1/kcfu/models/DART_main/models/wrf/work")
    dart_nml: Path | None = None
    mpi: str = "mpirun"

    @property
    def icbc(self):
        return self.icbc_root or self.project / "1icbc"

    @property
    def stage(self):
        return self.stage_root or self.scratch / "1icbc_incremental"

    @property
    def source_bc(self):
        return self.raw_bc or self.project / "0necessay_files/wrfbdy_d01_trfollow"

    @property
    def source_end_ic(self):
        return self.reference_ic or self.project / f"0necessay_files/wrfinput_d01_{time_tag(self.end)}"

    @property
    def template(self):
        return self.wrfda_template or self.project / "0necessay_files/namelist.WRFDA.VAR"

    def ic(self, member, moment):
        return self.icbc / f"mem{member:03d}/IC/wrfinput_{time_tag(moment)}_mem{member:03d}"

    def bc(self, member):
        return self.icbc / f"mem{member:03d}/BC/wrfbdy_d01_mem{member:03d}"

    def segment(self, member):
        return self.bc(member).with_name(self.bc(member).name + "_" + time_tag(self.start))

    @classmethod
    def from_env(cls):
        fields = {"project": Path(os.environ["PROJECT_ROOT"]), "scratch": Path(os.environ["SCRATCH_ROOT"]),
                  "start": parse_time(os.environ["START_TIME_UTC"]), "end": parse_time(os.environ["END_TIME_UTC"]),
                  "mpi": os.environ["MPI_LAUNCH"]}
        for key, env in (("icbc_root", "ICBC_ROOT"), ("stage_root", "STAGE_ROOT"),
                         ("raw_bc", "RAW_BC"), ("reference_ic", "END_REFERENCE_IC"),
                         ("wrfda_template", "WRFDA_TEMPLATE"), ("wrfda_exe", "WRFDA_EXE"),
                         ("wrfda_support", "WRFDA_SUPPORT"), ("dart_work", "DART_WORK"), ("dart_nml", "DART_INPUT_NML")):
            if os.environ.get(env):
                fields[key] = Path(os.environ[env])
        cfg = cls(**fields)
        if cfg.end - cfg.start != timedelta(hours=6) or cfg.start.minute or cfg.start.second:
            raise IncrementalError("this incremental entry extends exactly one6-hour interval on the hour")
        return cfg


def write_wrfda_namelist(source, target, moment, member):
    values = {"seed_array1": time_tag(moment), "seed_array2": str(member),
              "analysis_date": "'" + stamp(moment) + "'"}
    text = require_file(source).read_text()
    for key, value in values.items():
        text, count = re.subn(r"(?im)^([ \t]*)" + key + r"[ \t]*=.*$",
                             lambda match: f"{match.group(1)}{key} = {value},", text)
        if count != 1:
            raise IncrementalError(f"{source}: expected one {key} assignment, found {count}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def inspect_inputs(cfg):
    index = find_record(cfg.source_bc, cfg.start, cfg.end)
    validate_bc(cfg.source_bc, indices=[index])
    members = []
    reference_validated = False
    for member in MEMBERS:
        combined = cfg.bc(member)
        existing = records(combined)
        covered = existing[0][0] <= cfg.start and existing[-1][1] >= cfg.end
        if not covered and existing[-1][1] != cfg.start:
            raise IncrementalError(f"{combined}: existing end {stamp(existing[-1][1])} must equal extension start {stamp(cfg.start)}")
        item = {"member": member, "combined": combined, "covered": covered, "record_index": index,
                "start_ic": cfg.ic(member, cfg.start), "end_ic": cfg.ic(member, cfg.end),
                "segment": cfg.segment(member), "old_records": existing}
        if not covered:
            validate_ic(item["start_ic"], cfg.start)
            if item["end_ic"].exists():
                validate_ic(item["end_ic"], cfg.end)
                same_grid(item["start_ic"], item["end_ic"])
            else:
                if not reference_validated:
                    validate_ic(cfg.source_end_ic, cfg.end)
                    reference_validated = True
                same_grid(item["start_ic"], cfg.source_end_ic)
            if item["segment"].exists():
                if records(item["segment"]) != ((cfg.start, cfg.end),):
                    raise IncrementalError(f"existing extension segment has wrong interval: {item['segment']}")
                validate_bc(item["segment"], reference=combined)
        members.append(item)
    return members


def runtime_files(cfg):
    return [cfg.template, cfg.wrfda_exe, cfg.wrfda_support / "be.dat.cv3",
            cfg.dart_work / "pert_wrf_bc", cfg.dart_nml or cfg.dart_work / "input.nml",
            cfg.project / "0necessay_files/gribmap.txt", cfg.project / "0necessay_files/LANDUSE.TBL"]


def preflight(cfg):
    items = inspect_inputs(cfg)
    if not any(not item["covered"] for item in items):
        return items
    if cfg.scratch.resolve() not in cfg.stage.resolve().parents:
        raise IncrementalError("STAGE_ROOT must be strictly underneath SCRATCH_ROOT")
    for path in runtime_files(cfg):
        require_file(path)
    for path in (cfg.wrfda_exe, cfg.dart_work / "pert_wrf_bc"):
        if not os.access(path, os.X_OK):
            raise IncrementalError(f"not executable: {path}")
    for command in ("ncks", "ncrcat", shlex.split(cfg.mpi)[0]):
        if shutil.which(command) is None:
            raise IncrementalError(f"command is not on PATH: {command}")
    text = cfg.template.read_text()
    for name, expected in (("cv_options", "3"), ("analysis_type", "RANDOMCV")):
        values = re.findall(r"(?im)^[ \t]*" + name + r"[ \t]*=[ \t]*([^,\n]+)", text)
        if len(values) != 1 or values[0].strip().strip("\"'").upper() != expected:
            raise IncrementalError(f"{cfg.template}: require {name}={expected}, as in the original perturbation workflow")
    return items


def digest(path):
    result = hashlib.sha256()
    with require_file(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def signatures(paths, content=False, allow_missing=False):
    answer = {}
    for path in sorted(set(map(Path, paths))):
        if allow_missing and not path.exists() and not path.is_symlink():
            answer[str(path)] = None
            continue
        stat = require_file(path).stat()
        answer[str(path)] = {"source": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if content:
            answer[str(path)]["sha256"] = digest(path)
    return answer


def unchanged(saved):
    for path, old in saved.items():
        if old is None:
            if os.path.lexists(path):
                raise IncrementalError(f"input/destination appeared during staging: {path}")
            continue
        current = signatures([path], content="sha256" in old)[path]
        if current != old:
            raise IncrementalError(f"input changed during staging; refusing publication: {path}")


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, default=str, indent=2) + "\n")
    temporary.replace(path)


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def execute(argv, cwd, log):
    argv = list(map(str, argv))
    Path(log).with_suffix(".command.json").write_text(json.dumps(argv) + "\n")
    with Path(log).open("w") as output:
        result = subprocess.run(argv, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise IncrementalError(f"command exited {result.returncode}: {shlex.join(argv)}; log: {log}")


def validate_prefix(old, merged):
    """Every pre-existing record/value must survive concatenation exactly."""
    before, after = records(old), records(merged)
    if after[:len(before)] != before:
        raise IncrementalError(f"merged boundary changed the old time prefix: {merged}")
    with nc.Dataset(old) as source, nc.Dataset(merged) as target:
        source.set_auto_maskandscale(False)
        target.set_auto_maskandscale(False)
        for name, variable in source.variables.items():
            if name not in target.variables or variable.dimensions != target[name].dimensions:
                raise IncrementalError(f"merged boundary did not preserve variable {name}")
            if "Time" in variable.dimensions:
                axis = variable.dimensions.index("Time")
                slices = []
                for index in range(len(before)):
                    selection = [slice(None)] * variable.ndim
                    selection[axis] = index
                    slices.append(tuple(selection))
            else:
                slices = [Ellipsis]
            for selection in slices:
                a, b = np.asarray(variable[selection]), np.asarray(target[name][selection])
                equal = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)
                if not equal:
                    raise IncrementalError(f"merged boundary changed old prefix data: {name}")


def stage_member(cfg, item, workspace):
    source_guard = item["source_guard"]
    unchanged(source_guard)
    member = item["member"]
    area = workspace / f"mem{member:03d}"
    area.mkdir()
    original = area / "original_combined.nc"
    shutil.copyfile(item["combined"], original)
    old_hash = digest(original)
    if digest(item["combined"]) != old_hash:
        raise IncrementalError(f"combined BC changed while copying: {item['combined']}")
    end_ic = item["end_ic"]
    if not end_ic.exists():
        ic_work = area / "ic"
        ic_work.mkdir()
        # Private fg protects the shared unperturbed reference even if a solver
        # happens to open fg in write mode. The source namelist is never sed-ed.
        shutil.copyfile(cfg.source_end_ic, ic_work / "fg")
        for name, source in (("be.dat", cfg.wrfda_support / "be.dat.cv3"),
                             ("gribmap.txt", cfg.project / "0necessay_files/gribmap.txt"),
                             ("LANDUSE.TBL", cfg.project / "0necessay_files/LANDUSE.TBL")):
            (ic_work / name).symlink_to(source.absolute())
        write_wrfda_namelist(cfg.template, ic_work / "namelist.input", cfg.end, member)
        log(f"mem{member:03d}: generating only missing {time_tag(cfg.end)} IC")
        execute([*shlex.split(cfg.mpi), str(cfg.wrfda_exe)], ic_work, ic_work / "wrfda.log")
        logs = [ic_work / name for name in ("rsl.out.0000", "rsl.error.0000", "wrfda.log")]
        if not any(p.is_file() and "WRF-Var completed successfully" in p.read_text(errors="replace") for p in logs):
            raise IncrementalError(f"mem{member:03d}: WRFDA success marker absent; see {ic_work}")
        end_ic = ic_work / "wrfvar_output"
        validate_ic(end_ic, cfg.end)
        same_grid(item["start_ic"], end_ic)
    segment = item["segment"]
    if not segment.exists():
        bc_work = area / "bc"
        bc_work.mkdir()
        segment = bc_work / "wrfbdy_this"
        index = item["record_index"]
        execute(["ncks", "-O", "-d", f"Time,{index},{index}", str(cfg.source_bc), str(segment)],
                bc_work, bc_work / "extract.log")
        if records(segment) != ((cfg.start, cfg.end),):
            raise IncrementalError(f"wrong extracted BC record: {segment}")
        validate_bc(segment, reference=original)
        (bc_work / "wrfinput_this").symlink_to(item["start_ic"].absolute())
        (bc_work / "wrfinput_next").symlink_to(end_ic.absolute())
        shutil.copy2(cfg.dart_nml or cfg.dart_work / "input.nml", bc_work / "input.nml")
        execute([str(cfg.dart_work / "pert_wrf_bc")], bc_work, bc_work / "pert.log")
        logs = [bc_work / "pert.log", bc_work / "dart_log.out"]
        if not any(p.is_file() and "FINISHED pert_wrf_bc." in p.read_text(errors="replace") for p in logs):
            raise IncrementalError(f"mem{member:03d}: pert_wrf_bc success marker absent; see {bc_work}")
        if records(segment) != ((cfg.start, cfg.end),):
            raise IncrementalError(f"perturbed BC interval changed unexpectedly: {segment}")
    validate_bc(segment, reference=original)
    merged = area / "extended_combined.nc"
    execute(["ncrcat", "-O", str(original), str(segment), str(merged)], area, area / "merge.log")
    wanted = item["old_records"] + ((cfg.start, cfg.end),)
    if records(merged) != wanted:
        raise IncrementalError(f"extended BC has duplicate/missing/unexpected records: {merged}")
    validate_prefix(original, merged)
    validate_bc(merged, reference=original, indices=[len(item["old_records"])])
    unchanged(source_guard)
    return {"item": item, "end_ic": end_ic, "segment": segment, "merged": merged,
            "original": original, "old_hash": old_hash, "source_guard": source_guard}


def copy_temporary(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def install_new(source, target):
    if source.absolute() == target.absolute():
        return
    if target.exists() or target.is_symlink():
        if not target.is_file() or digest(target) != digest(source):
            raise IncrementalError(f"refusing to overwrite an existing different file: {target}")
        return
    temporary = copy_temporary(source, target)
    try:
        # Same-filesystem hard-link publication gives atomic no-clobber install.
        # Removing the temporary name leaves a regular, independently owned file.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if digest(target) != digest(source):
                raise IncrementalError(f"another writer created a different file: {target}")
    finally:
        temporary.unlink(missing_ok=True)


def publish_member(cfg, staged):
    item = staged["item"]
    unchanged(staged["source_guard"])
    if digest(item["combined"]) != staged["old_hash"]:
        raise IncrementalError(f"combined BC changed before publication: {item['combined']}")
    install_new(staged["end_ic"], item["end_ic"])
    install_new(staged["segment"], item["segment"])
    backup = item["combined"].parent / "backups" / (
        item["combined"].name + ".before_" + time_tag(cfg.end) + "." + staged["old_hash"][:12])
    install_new(staged["original"], backup)
    temporary = copy_temporary(staged["merged"], item["combined"])
    try:
        if digest(item["combined"]) != staged["old_hash"]:
            raise IncrementalError(f"combined BC changed during publication: {item['combined']}")
        os.replace(temporary, item["combined"])
    finally:
        temporary.unlink(missing_ok=True)
    log(f"mem{item['member']:03d}: extended through {stamp(cfg.end)}; backup {backup}")


def run(cfg):
    import fcntl
    items = preflight(cfg)
    if all(item["covered"] for item in items):
        log(f"All {len(items)} members already cover {stamp(cfg.end)}; nothing changed")
        return
    if not os.environ.get("LSB_JOBID"):
        raise IncrementalError("RUN requires an LSF MPI allocation: submit with bsub < run_create_icbc_incremental.sh")
    with (cfg.icbc / ".icbc_incremental.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IncrementalError(f"another incremental writer owns {cfg.icbc}") from exc
        # Snapshot ALL member inputs and expected-absent destinations before
        # validating/consuming any member. A later member cannot acquire a new
        # baseline while earlier members' MPI jobs are running.
        canonical = [path for member in MEMBERS for path in (
            cfg.ic(member, cfg.start), cfg.ic(member, cfg.end), cfg.segment(member), cfg.bc(member))]
        member_guard = signatures(canonical, content=True, allow_missing=True)
        # Guard dependencies before validation too, including a currently absent
        # optional reference IC (allowed when every endpoint IC already exists).
        dependency_guard = signatures(runtime_files(cfg) + [cfg.source_bc, cfg.source_end_ic],
                                      content=True, allow_missing=True)
        items = preflight(cfg)  # recheck canonical state after obtaining the lock
        unchanged(member_guard)
        unchanged(dependency_guard)
        pending = [item for item in items if not item["covered"]]
        if not pending:
            return
        for item in pending:
            item["source_guard"] = {str(item[key]): member_guard[str(item[key])]
                                    for key in ("start_ic", "end_ic", "segment", "combined")}
        cfg.stage.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix=time_tag(cfg.start) + "_" + time_tag(cfg.end) + "_", dir=cfg.stage))
        manifest = {"start": stamp(cfg.start), "end": stamp(cfg.end), "inputs": dependency_guard,
                    "canonical_inputs": member_guard,
                    "members": [item["member"] for item in pending], "published": []}
        write_json(workspace / "manifest.json", manifest)
        log(f"Private workspace: {workspace}; staging {len(pending)} members before publication")
        staged = [stage_member(cfg, item, workspace) for item in pending]
        unchanged(dependency_guard)
        unchanged(member_guard)
        # Check every member once more before the first canonical addition.
        for candidate in staged:
            unchanged(candidate["source_guard"])
            if digest(candidate["item"]["combined"]) != candidate["old_hash"]:
                raise IncrementalError("a combined BC changed; no staged data will be published")
        for candidate in staged:
            publish_member(cfg, candidate)
            manifest["published"].append(candidate["item"]["member"])
            write_json(workspace / "manifest.json", manifest)
        manifest["complete"] = True
        write_json(workspace / "manifest.json", manifest)
        log(f"Increment complete; all members cover {stamp(cfg.end)}; old ICs/fragments retained")


def main():
    cfg = Config.from_env()
    mode = os.environ.get("MODE", "RUN")
    if mode not in ("RUN", "PREFLIGHT", "DRY_RUN"):
        raise IncrementalError("MODE must be RUN/PREFLIGHT/DRY_RUN")
    if mode == "DRY_RUN":
        items = inspect_inputs(cfg)
        print(json.dumps({"start": stamp(cfg.start), "end": stamp(cfg.end), "members": items}, default=str, indent=2))
    elif mode == "PREFLIGHT":
        items = preflight(cfg)
        count = sum(not item["covered"] for item in items)
        print(f"Preflight passed: {count}/{len(items)} members need extension; source record index={items[0]['record_index']} (zero-based)")
    else:
        run(cfg)


if __name__ == "__main__":
    try:
        main()
    except (IncrementalError, ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
ICBC_PYTHON
