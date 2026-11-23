"""Shared helpers for the bjobs/bhist wrappers.

The LSF on this cluster drops the "[index]" suffix when `bjobs` is asked for
the "jobid stat" fields, and completed array elements disappear from `bjobs`.
The driver therefore cannot map array elements to states with its built-in
query.  The wrappers reconstruct per-element states from
`bhist -l <base_job_id>` and cache them so the driver's per-element `bhist`
fallback does not hang on this LSF.

These scripts are only placed in PATH for the 50-member controller job; they
do not modify run_driver_50mem_cyclingDA.sh or its experiment signature.
"""
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

HEADER_RE = re.compile(r"^Job <(\d+)(?:\[(\d+)\])?>")
ARRAY_QUERY_RE = re.compile(r"^\d+$")
ELEMENT_RE = re.compile(r"^(\d+)\[(\d+)\]$")

LSF_BIN_DEFAULT = "/share/apps/lsf/10.1/linux2.6-glibc2.3-x86_64/bin"


def real_tool(name):
    """Return the real LSF executable, skipping this wrapper directory."""
    override = os.environ.get("REAL_" + name.upper())
    if override:
        return override
    own = os.path.dirname(os.path.realpath(__file__))
    search = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and os.path.realpath(entry) != own:
            search.append(entry)
    found = shutil.which(name, path=os.pathsep.join(search))
    if found:
        return found
    return os.path.join(LSF_BIN_DEFAULT, name)


def pass_through(name, args):
    tool = real_tool(name)
    os.execv(tool, [tool] + list(args))


def is_array_query(args):
    return (len(args) == 5
            and args[0] == "-a"
            and args[1] == "-noheader"
            and args[2] == "-o"
            and args[3] == "jobid stat"
            and ARRAY_QUERY_RE.match(args[4]) is not None)


ACTIVE_STATES = ("RUN", "PEND", "PSUSP", "USUSP", "SSUSP", "WAIT")
TERMINAL_STATES = ("DONE", "EXIT")
KNOWN_STATES = set(ACTIVE_STATES) | set(TERMINAL_STATES)


def find_job(job_id):
    """Return (lsf_job.json path, members) for the driver's array job."""
    root = os.environ.get("RUN_ROOT")
    if not root:
        return None
    candidates = []
    for path in Path(root).glob("*/*/jobs/*/attempt_*/lsf_job.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if str(record.get("job_id")) != str(job_id):
            continue
        members = [int(m) for m in record.get("members", [])]
        if members:
            candidates.append((path.stat().st_mtime_ns, path, members))
    if not candidates:
        return None
    candidates.sort()
    _, path, members = candidates[-1]
    return path, members


def member_dirs(record_path, member):
    """Return (run_wrf/<cycle>/<member>, wrfout/<cycle>/<member>)."""
    experiment_root = record_path.parents[3]
    cycle_tag = record_path.parents[1].name
    work = experiment_root / "run_wrf" / cycle_tag / ("%03d" % member)
    archive = experiment_root / "wrfout" / cycle_tag / ("%03d" % member)
    return work, archive


def filesystem_done(record_path, member):
    """A member that wrote wrf_done or was archived has definitely finished."""
    work, archive = member_dirs(record_path, member)
    return (archive / "member_done.json").is_file() or (work / "wrf_done").is_file()


def parse_bhist(text):
    """Map array member -> latest LSF state from `bhist -l <base_id>`.

    A single (non-array) job produces the key None.
    """
    states = {}
    current = None
    started = False
    for raw in text.splitlines():
        match = HEADER_RE.match(raw.strip())
        if match:
            current = int(match.group(2)) if match.group(2) is not None else None
            states.setdefault(current, None)
            started = True
            continue
        if not started:
            continue
        if "Done successfully" in raw:
            states[current] = "DONE"
        elif re.search(r"\bExited\b", raw) or "Killed" in raw or "Terminated" in raw:
            states[current] = "EXIT"
        elif ("Running with" in raw or "Starting (" in raw
              or "Dispatched" in raw):
            if states.get(current) not in ("DONE", "EXIT"):
                states[current] = "RUN"
        elif "Submitted" in raw or "PENDING" in raw:
            if states.get(current) not in ("DONE", "EXIT", "RUN"):
                states[current] = "PEND"
    return states


def cache_path(job_id):
    return Path(tempfile.gettempdir()) / (
        "lsf_state_%s_%s.json" % (os.getuid(), job_id))


def read_cache(job_id, ttl=20.0):
    try:
        data = json.loads(cache_path(job_id).read_text())
        if time.time() - float(data.get("ts", 0)) <= ttl:
            return data.get("states")
    except (OSError, ValueError, TypeError):
        pass
    return None


def write_cache(job_id, states):
    path = cache_path(job_id)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps({"ts": time.time(), "states": states},
                                    sort_keys=True))
    temporary.replace(path)
