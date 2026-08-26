#!/usr/bin/env bash
set -Eeuo pipefail

#================================================================================
# run_single_obs_dart.sh
#
# Single-observation DART assimilation orchestrated by LSF/bsub.
#
# MAINTENANCE / RUNTIME NOTE:
#   * This file is maintained in the LOCAL git repository:
#         /Users/kcfu/works/nju/tc_mangkhut/Mangkhut_scripts/4assimilation/2DART/run_dir/
#   * Every RUNTIME path inside this script is the CLUSTER (LSF) path set for
#     the DART experiment on the cluster.  Do not use the local Mac layout
#     (`/Users/...`, `Mangkhut_scripts/...`) as runtime paths here.
#   * PROJECT_ROOT and SOURCE_OBS_FILE are independent configuration items.
#     SOURCE_OBS_FILE may live under DART_SOURCE_ROOT (the DART source tree)
#     and can be overridden entirely via the SOURCE_OBS_FILE environment
#     variable.
#
# Flow:
#   1) validate_config                      - read-only: validate all inputs/paths
#   2) extract_observation_location         - read-only: parse "OBS <idx>" block
#   3) check_stale_outputs                  - read-only: refuse prior outputs BEFORE
#                                             any file is modified
#   4) compute_crop_bounds                  - pure math (no file writes): build a
#                                             tiny lat/lon window around the obs
#   5) update_obs_sequence_tool_nml         - write six &obs_sequence_tool_nml fields
#   6) run_obs_sequence_tool                - crop exactly one observation
#   7) validate_single_observation          - confirm the crop is one obs at target
#   8) install_dart_obs_seq                 - install the single obs into DART run dir
#   9) configure_dart_filter                - set qceff_table_filename per FILTER_TYPE
#  10) submit_dart_job                      - bsub < sub_dart.sh, parse job id
#  11) wait_for_lsf_job                     - poll bjobs/bjobs -a/bhist until a
#                                             definitive final state
#  12) check_dart_outputs                   - verify fkc_dart, test.out, post_assim_me*
#  13) archive_outputs                      - move outputs; honest partial-failure report
#
# One FILTER_TYPE and one OBS_INDEX per run; nothing is submitted twice.
# All python is invoked with ${PYTHON_EXE}; no bare python/python3.
# No eval, no unconstrained rm wildcards; post_assim_me* is handled with
# nullglob + bash arrays.
#================================================================================

#==============================================================================
# (A) parameter configuration region
#==============================================================================

# DART filter: EAKF or QCF_RHF
FILTER_TYPE="${FILTER_TYPE:-EAKF}"

# observation block number to crop (the "OBS <idx>" block, not a line number)
OBS_INDEX="${OBS_INDEX:-98}"

# cluster experiment root (LSF runtime paths are derived from this)
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/lililei1/kcfu/tc_mangkhut}"

# cluster DART source tree (independent of PROJECT_ROOT)
DART_SOURCE_ROOT="${DART_SOURCE_ROOT:-/share/home/lililei1/kcfu/models/DART_main}"

# original/complete observation file.
# NOTE: this is an independent configuration item (not bound to PROJECT_ROOT);
# it may be overridden entirely via the SOURCE_OBS_FILE environment variable.
# Default: the clear-sky-filtered LACC ch4 obs_seq produced by
# run_single_assimilation_test.sh (LACC_LAG_HOURS="0 3 6", ch4, clear02).
SOURCE_OBS_FILE="${SOURCE_OBS_FILE:-/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir/obs_seq.out_kctest1_d01_10_00_00_LACC_ch4_clear02}"

# python interpreter used to parse obs_seq, compute crop bounds and edit namelists
PYTHON_EXE="${PYTHON_EXE:-/share/home/lililei1/kcfu/anaconda/envs/wrf/bin/python}"

# LSF job-status polling interval (seconds) and maximum wait (seconds)
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_POLL_SECONDS="${MAX_POLL_SECONDS:-172800}"

# bounded retries when bjobs/bjobs -a cannot yet see a freshly submitted job
LSF_QUERY_RETRIES="${LSF_QUERY_RETRIES:-5}"
LSF_QUERY_RETRY_DELAY="${LSF_QUERY_RETRY_DELAY:-5}"

# half-width (degrees) of the tiny lat/lon window around the target observation.
# obs_sequence_tool rejects min_lat >= max_lat and min_lon == max_lon, so the
# crop window is target +/- CROP_EPSILON_DEG instead of min == max.
CROP_EPSILON_DEG="${CROP_EPSILON_DEG:-1e-8}"

# tolerance (degrees) for comparing crop results to the target location
LOC_TOLERANCE_DEG="${LOC_TOLERANCE_DEG:-1e-6}"

#==============================================================================
# (B) fixed paths derived from the cluster roots
#==============================================================================

# obs_sequence_tool work directory + its namelist + executable
OBS_TOOL_WORK_DIR="${PROJECT_ROOT}/4assimilation/1convert_obs/run_dir"
OBS_TOOL_INPUT_NML="${OBS_TOOL_WORK_DIR}/input.nml"
OBS_TOOL_EXE="${OBS_TOOL_WORK_DIR}/obs_sequence_tool"

# DART run directory + its configuration and submit script
DART_RUN_DIR="${PROJECT_ROOT}/4assimilation/2DART/run_dir"
DART_INPUT_NML="${DART_RUN_DIR}/input.nml"
DART_SUBMIT_SCRIPT="${DART_RUN_DIR}/sub_dart.sh"
DART_OBS_FILE="${DART_RUN_DIR}/obs_seq.out"

# qceff table required only when FILTER_TYPE=QCF_RHF
QCEFF_TABLE_FILE="${DART_RUN_DIR}/qceff_table_fkc.csv"

# single-observation file produced by obs_sequence_tool.
# Defined AFTER OBS_INDEX and in DOUBLE quotes so ${OBS_INDEX} expands.
SINGLE_OBS_FILE="${OBS_TOOL_WORK_DIR}/obs_seq.out_LACC_single_${OBS_INDEX}"

# archive root; results land in ${ARCHIVE_ROOT}/${FILTER_TYPE}/obs_seq${OBS_INDEX}
ARCHIVE_ROOT="/scratch/lililei1/kcfu/tc_mangkhut/4assimilation/DART"

#==============================================================================
# (C) logging + unified error handling (timestamps everywhere)
#==============================================================================

CURRENT_STEP="validate_config"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log_line() { printf '%s [%s] %s\n' "$(ts)" "$CURRENT_STEP" "$*"; }

fatal() {  # fatal <message> [exit_status]
    local msg="${1:-unspecified error}"
    local rc="${2:-1}"
    printf '%s ERROR [%s] at %s():%s (exit status %s): %s\n' \
        "$(ts)" "$CURRENT_STEP" "${FUNCNAME[1]:-main}" "${BASH_LINENO[0]:-?}" "$rc" "$msg" >&2
    exit "${rc}"
}

on_error() {  # unexpected failure trap
    local rc=$?
    printf '%s FATAL [%s] at %s():%s (exit status %s) - unexpected failure\n' \
        "$(ts)" "$CURRENT_STEP" "${FUNCNAME[1]:-main}" "${BASH_LINENO[0]:-?}" "$rc" >&2
    exit "$rc"
}
trap 'on_error' ERR

# numeric helpers (floating point, no bc)
float_eq() {  # float_eq <a> <b>  -> 0 if numerically equal
    awk -v a="${1}" -v b="${2}" 'BEGIN{exit !(a==b)}'
}
within_tol() {  # within_tol <val> <ref> <tol>  -> 0 if |val-ref| <= tol
    awk -v a="${1}" -v b="${2}" -v t="${3}" 'BEGIN{exit !((a-b)*(a-b) <= t*t)}'
}

#==============================================================================
# (D) log file setup (all output copied into the log).
#     Installed inside main() AFTER validate_config(), so DART_RUN_DIR is
#     known to exist first; writing the log does not touch experiment files.
#=============================================================================
SCRIPT_LOG="${DART_RUN_DIR}/run_single_obs_dart_${FILTER_TYPE}_obs${OBS_INDEX}.log"

#==============================================================================
# (E) single embedded python helper (py command)
#     commands:
#       extract <file> <obs_index>
#           -> "lon_rad lat_rad lon_deg lat_deg vert vtype"
#       inspect <file>
#           -> "lon_deg lat_deg vert vtype"   (requires exactly one obs block)
#       crop_bounds <lon_deg> <lat_deg> <epsilon_deg>
#           -> "min_lat max_lat min_lon max_lon"
#       check_nums <epsilon_deg> <tolerance_deg>
#           -> "ok"   (both must be finite positive numbers)
#       edit_obs_tool <nml> <seq_in> <seq_out> <min_lat> <max_lat> <min_lon> <max_lon>
#           -> readback lines "field value" for the six edited fields
#       edit_qceff <nml> <quoted_value>
#           -> readback line "qceff_table_filename <unquoted value>"
#==============================================================================

py() {
    "${PYTHON_EXE}" - "$@" <<'PYEOF'
import sys, re, math

RAD2DEG = 180.0 / math.pi
OBS_RE = re.compile(r'^\s*OBS\s+\d+')
NUMOBS_RE = re.compile(r'^\s*num_obs\s*:\s*(\d+)')
ASSIGN_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=')
TOOL_FIELDS = ['filename_seq', 'filename_out', 'min_lat', 'max_lat', 'min_lon', 'max_lon']
QCEFF_VAR = 'qceff_table_filename'

def die(msg):
    sys.stderr.write('ERROR: %s\n' % msg)
    sys.exit(1)

def read_lines(path):
    with open(path, 'rb') as fh:
        data = fh.read().decode('utf-8', errors='replace')
    return data.splitlines()

def scan_blocks(lines):
    n = len(lines); i = 0; blocks = []
    while i < n:
        m = OBS_RE.match(lines[i])
        if m:
            no = int(m.group(0).split()[1])
            j = i + 1; loc = None
            while j < n and not OBS_RE.match(lines[j]):
                if lines[j].strip() == 'loc3d':
                    if loc is not None:
                        die('multiple loc3d entries inside one OBS block (near OBS %d)' % no)
                    loc = j
                j += 1
            if loc is None:
                die('OBS %d block has no loc3d marker' % no)
            if loc + 1 >= n:
                die('OBS %d loc3d marker has no coordinate line' % no)
            fields = lines[loc + 1].split()
            if len(fields) < 2:
                die('OBS %d loc3d coordinate line is malformed' % no)
            blocks.append({'no': no, 'fields': fields})
            i = j
        else:
            i += 1
    return blocks

def parse_location(fields):
    try:
        lon_rad = float(fields[0]); lat_rad = float(fields[1])
        vert = float(fields[2]) if len(fields) > 2 else None
        vtype = fields[3].strip() if len(fields) > 3 else 'NA'
    except ValueError:
        die('non-numeric location values in loc3d line: %s' % ' '.join(fields))
    if not (math.isfinite(lon_rad) and math.isfinite(lat_rad)):
        die('non-finite location values in loc3d line: %s' % ' '.join(fields))
    lon_deg = lon_rad * RAD2DEG
    lat_deg = lat_rad * RAD2DEG
    return lon_rad, lat_rad, lon_deg, lat_deg, vert, vtype

def fmt_vert(vert):
    if vert is None:
        return 'NA'
    if float(vert).is_integer():
        return '%.1f' % vert
    return '%.17g' % vert

def cmd_extract(args):
    path, idx = args[0], int(args[1])
    blocks = scan_blocks(read_lines(path))
    hit = [b for b in blocks if b['no'] == idx]
    if len(hit) == 0:
        die('obs block OBS %d not found in %s (file holds %d obs blocks)' % (idx, path, len(blocks)))
    if len(hit) > 1:
        die('duplicate obs block OBS %d found %d times in %s' % (idx, len(hit), path))
    lon_rad, lat_rad, lon_deg, lat_deg, vert, vtype = parse_location(hit[0]['fields'])
    print('%.17g %.17g %.15f %.15f %s %s' % (lon_rad, lat_rad, lon_deg, lat_deg, fmt_vert(vert), vtype.strip()))

def cmd_inspect(args):
    path = args[0]
    lines = read_lines(path)
    blocks = scan_blocks(lines)
    if len(blocks) == 0:
        die('%s contains no obs blocks (crop produced zero observations)' % path)
    if len(blocks) > 1:
        die('%s contains %d obs blocks; expected exactly 1' % (path, len(blocks)))
    nm = None
    for ln in lines:
        mm = NUMOBS_RE.match(ln)
        if mm:
            nm = int(mm.group(1)); break
    if nm is None:
        die('num_obs header missing in %s' % path)
    if nm != 1:
        die('%s num_obs=%d; expected 1' % (path, nm))
    lon_rad, lat_rad, lon_deg, lat_deg, vert, vtype = parse_location(blocks[0]['fields'])
    print('%.15f %.15f %s %s' % (lon_deg, lat_deg, fmt_vert(vert), vtype.strip()))

def cmd_crop_bounds(args):
    try:
        lon = float(args[0]); lat = float(args[1]); eps = float(args[2])
    except ValueError:
        die('crop_bounds inputs must be numbers: %r' % ' '.join(args))
    if not (math.isfinite(eps) and eps > 0.0):
        die('CROP_EPSILON_DEG must be a finite positive number, got %r' % args[2])
    if not (math.isfinite(lon) and math.isfinite(lat)):
        die('target longitude/latitude must be finite, got %r %r' % (args[0], args[1]))
    # latitude: clamp to the globe; if the target is near a pole the window
    # becomes asymmetric but min_lat must stay strictly below max_lat.
    min_lat = max(-90.0, lat - eps)
    max_lat = min(90.0, lat + eps)
    if not (min_lat < max_lat):
        die('cannot construct a valid latitude window for lat=%r eps=%r' % (lat, eps))
    # longitude: keep the 0..360 representation; eps is tiny so at most one
    # bound can cross 360.  If it does, wrap max_lon back by 360 to form a
    # window that crosses the 0-meridian (min_lon may then exceed max_lon).
    min_lon = lon - eps
    max_lon = lon + eps
    if max_lon > 360.0:
        max_lon -= 360.0
    if min_lon == max_lon:
        die('min_lon would equal max_lon; cannot construct a longitude window')
    if not all(map(math.isfinite, (min_lat, max_lat, min_lon, max_lon))):
        die('non-finite crop bounds')
    print('%.15f %.15f %.15f %.15f' % (min_lat, max_lat, min_lon, max_lon))

def cmd_check_nums(args):
    for name, s in [('CROP_EPSILON_DEG', args[0]), ('LOC_TOLERANCE_DEG', args[1])]:
        try:
            v = float(s)
        except ValueError:
            die('%s must be a finite positive number, got %r' % (name, s))
        if not (math.isfinite(v) and v > 0.0):
            die('%s must be a finite positive number, got %r' % (name, s))
    print('ok')

def find_module(lines, name):
    count = 0; s = None
    for i, l in enumerate(lines):
        if l.strip() == '&%s' % name:
            s = i; count += 1
    if count == 0:
        die('namelist &%s not found in input.nml' % name)
    if count > 1:
        die('namelist &%s appears %d times; cannot decide which to edit' % (name, count))
    e = None
    for j in range(s + 1, len(lines)):
        t = lines[j].strip()
        if not t or t.startswith('!') or t.startswith('#'):
            continue
        if t == '/' or t.endswith('/'):
            e = j; break
    if e is None:
        die('no closing / found for &%s' % name)
    return s, e

def module_assign_lines(lines, s, e, var):
    out = []
    for k in range(s + 1, e + 1):
        t = lines[k].strip()
        if not t or t.startswith('!') or t.startswith('#'):
            continue
        m = ASSIGN_RE.match(t)
        if m and m.group(1) == var:
            out.append(k)
    return out

def parse_value(snippet):
    v = snippet.split('=', 1)[1].strip()
    while v.endswith(','):
        v = v[:-1].strip()
    if v.endswith('/'):
        v = v[:-1].strip()
    return v.strip("'\"")

def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write('\n'.join(lines) + '\n')

def cmd_edit_obs_tool(args):
    path, seq_in, seq_out, min_lat, max_lat, min_lon, max_lon = args
    lines = read_lines(path)
    s, e = find_module(lines, 'obs_sequence_tool_nml')
    newvals = {
        'filename_seq': "'%s'" % seq_in,
        'filename_out': "'%s'" % seq_out,
        'min_lat': min_lat, 'max_lat': max_lat,
        'min_lon': min_lon, 'max_lon': max_lon,
    }
    for var, nv in newvals.items():
        ks = module_assign_lines(lines, s, e, var)
        if len(ks) == 0:
            die('field %s not found in &obs_sequence_tool_nml' % var)
        if len(ks) > 1:
            die('field %s duplicated %d times in &obs_sequence_tool_nml; ambiguous' % (var, len(ks)))
        k = ks[0]
        indent = lines[k][:len(lines[k]) - len(lines[k].lstrip())]
        lines[k] = '%s%s = %s,' % (indent, var, nv)
    write_lines(path, lines)
    # read-back verification
    rb = {}
    for var in TOOL_FIELDS:
        ks = module_assign_lines(lines, s, e, var)
        if len(ks) == 0:
            die('read-back: field %s missing in &obs_sequence_tool_nml' % var)
        if len(ks) > 1:
            die('read-back: field %s duplicated in &obs_sequence_tool_nml' % var)
        rb[var] = parse_value(lines[ks[0]])
    for var in TOOL_FIELDS:
        print('%s %s' % (var, rb[var]))

def cmd_edit_qceff(args):
    path, quoted_new = args
    original = read_lines(path)
    hits = []
    i = 0; n = len(original)
    while i < n:
        m = re.match(r'^&([A-Za-z_][A-Za-z0-9_]*)\s*$', original[i].strip())
        if m:
            mod = m.group(1)
            s = i; e = None
            for j in range(s + 1, n):
                t = original[j].strip()
                if not t or t.startswith('!') or t.startswith('#'):
                    continue
                if t == '/' or t.endswith('/'):
                    e = j; break
            if e is None:
                die('no closing / for &%s in %s' % (mod, path))
            for k in range(s + 1, e + 1):
                t = original[k].strip()
                if not t or t.startswith('!') or t.startswith('#'):
                    continue
                m2 = ASSIGN_RE.match(t)
                if m2 and m2.group(1) == QCEFF_VAR:
                    hits.append(k)
            i = e + 1
        else:
            i += 1
    if len(hits) == 0:
        die('%s not found in %s' % (QCEFF_VAR, path))
    if len(hits) > 1:
        die('%s appears %d times in %s; ambiguous' % (QCEFF_VAR, len(hits), path))
    k = hits[0]
    indent = original[k][:len(original[k]) - len(original[k].lstrip())]
    original[k] = '%s%s = %s,' % (indent, QCEFF_VAR, quoted_new)
    write_lines(path, original)
    # read-back verification
    lines2 = read_lines(path)
    hits2 = []
    i = 0; n = len(lines2)
    while i < n:
        m = re.match(r'^&([A-Za-z_][A-Za-z0-9_]*)\s*$', lines2[i].strip())
        if m:
            mod = m.group(1); s = i; e = None
            for j in range(s + 1, n):
                t = lines2[j].strip()
                if not t or t.startswith('!') or t.startswith('#'):
                    continue
                if t == '/' or t.endswith('/'):
                    e = j; break
            if e is None:
                die('no closing / for &%s in %s (read-back)' % (mod, path))
            for k in range(s + 1, e + 1):
                t = lines2[k].strip()
                if not t or t.startswith('!') or t.startswith('#'):
                    continue
                m2 = ASSIGN_RE.match(t)
                if m2 and m2.group(1) == QCEFF_VAR:
                    hits2.append(k)
            i = e + 1
        else:
            i += 1
    if len(hits2) != 1:
        die('read-back: %s not uniquely present after edit' % QCEFF_VAR)
    print('%s %s' % (QCEFF_VAR, parse_value(lines2[hits2[0]])))

def main():
    if len(sys.argv) < 3:
        die('usage: py <cmd> ...')
    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == 'extract':
        cmd_extract(args)
    elif cmd == 'inspect':
        cmd_inspect(args)
    elif cmd == 'crop_bounds':
        cmd_crop_bounds(args)
    elif cmd == 'check_nums':
        cmd_check_nums(args)
    elif cmd == 'edit_obs_tool':
        cmd_edit_obs_tool(args)
    elif cmd == 'edit_qceff':
        cmd_edit_qceff(args)
    else:
        die('unknown py command: %s' % cmd)

main()
PYEOF
}

#==============================================================================
# (F) configuration/read-only validation
#==============================================================================

validate_config() {
    CURRENT_STEP="validate_config"
    if [[ "${FILTER_TYPE}" != "EAKF" && "${FILTER_TYPE}" != "QCF_RHF" ]]; then
        fatal "FILTER_TYPE must be EAKF or QCF_RHF, got '${FILTER_TYPE}'"
    fi
    if [[ ! "${OBS_INDEX}" =~ ^[0-9]+$ ]] || (( 10#${OBS_INDEX} <= 0 )); then
        fatal "OBS_INDEX must be a positive integer, got '${OBS_INDEX}'"
    fi
    if [[ ! "${POLL_INTERVAL}" =~ ^[0-9]+$ ]] || (( 10#${POLL_INTERVAL} <= 0 )); then
        fatal "POLL_INTERVAL must be a positive integer, got '${POLL_INTERVAL}'"
    fi
    if [[ ! "${MAX_POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( 10#${MAX_POLL_SECONDS} <= 0 )); then
        fatal "MAX_POLL_SECONDS must be a positive integer, got '${MAX_POLL_SECONDS}'"
    fi
    if [[ ! "${LSF_QUERY_RETRIES}" =~ ^[0-9]+$ ]] || (( 10#${LSF_QUERY_RETRIES} <= 0 )); then
        fatal "LSF_QUERY_RETRIES must be a positive integer, got '${LSF_QUERY_RETRIES}'"
    fi
    if [[ ! "${LSF_QUERY_RETRY_DELAY}" =~ ^[0-9]+$ ]] || (( 10#${LSF_QUERY_RETRY_DELAY} <= 0 )); then
        fatal "LSF_QUERY_RETRY_DELAY must be a positive integer, got '${LSF_QUERY_RETRY_DELAY}'"
    fi
    # interpreter first (it is needed for all numeric validation below)
    if [[ ! -e "${PYTHON_EXE}" ]]; then
        fatal "PYTHON_EXE does not exist: ${PYTHON_EXE}"
    fi
    if [[ ! -x "${PYTHON_EXE}" ]]; then
        fatal "PYTHON_EXE is not executable: ${PYTHON_EXE}"
    fi
    # finite-positive epsilon/tolerance via ${PYTHON_EXE}
    if ! out="$(py check_nums "${CROP_EPSILON_DEG}" "${LOC_TOLERANCE_DEG}")"; then
        fatal "invalid numeric config (${CROP_EPSILON_DEG} / ${LOC_TOLERANCE_DEG}); see message above"
    fi
    # cluster runtime paths / files (read-only)
    if [[ ! -f "${SOURCE_OBS_FILE}" ]]; then
        fatal "SOURCE_OBS_FILE does not exist: ${SOURCE_OBS_FILE} - override SOURCE_OBS_FILE or edit the config region"
    fi
    if [[ ! -s "${SOURCE_OBS_FILE}" ]]; then
        fatal "SOURCE_OBS_FILE is empty: ${SOURCE_OBS_FILE}"
    fi
    if [[ ! -r "${SOURCE_OBS_FILE}" ]]; then
        fatal "SOURCE_OBS_FILE is not readable: ${SOURCE_OBS_FILE}"
    fi
    if [[ ! -d "${OBS_TOOL_WORK_DIR}" ]]; then
        fatal "obs_sequence_tool work dir does not exist: ${OBS_TOOL_WORK_DIR}"
    fi
    if [[ ! -f "${OBS_TOOL_INPUT_NML}" || ! -r "${OBS_TOOL_INPUT_NML}" ]]; then
        fatal "obs_sequence_tool input.nml missing/unreadable: ${OBS_TOOL_INPUT_NML}"
    fi
    if [[ ! -e "${OBS_TOOL_EXE}" ]]; then
        fatal "obs_sequence_tool not found: ${OBS_TOOL_EXE}"
    fi
    if [[ ! -x "${OBS_TOOL_EXE}" ]]; then
        fatal "obs_sequence_tool is not executable: ${OBS_TOOL_EXE}"
    fi
    if [[ ! -d "${DART_RUN_DIR}" ]]; then
        fatal "DART run dir does not exist: ${DART_RUN_DIR}"
    fi
    if [[ ! -f "${DART_INPUT_NML}" || ! -r "${DART_INPUT_NML}" ]]; then
        fatal "DART input.nml missing/unreadable: ${DART_INPUT_NML}"
    fi
    if [[ ! -f "${DART_SUBMIT_SCRIPT}" || ! -r "${DART_SUBMIT_SCRIPT}" ]]; then
        fatal "DART submit script missing/unreadable: ${DART_SUBMIT_SCRIPT}"
    fi
    if [[ "${FILTER_TYPE}" == "QCF_RHF" ]]; then
        if [[ ! -f "${QCEFF_TABLE_FILE}" ]]; then
            fatal "QCF_RHF requires qceff table but it does not exist: ${QCEFF_TABLE_FILE}"
        fi
        if [[ ! -s "${QCEFF_TABLE_FILE}" ]]; then
            fatal "QCF_RHF qceff table is empty: ${QCEFF_TABLE_FILE}"
        fi
        if [[ ! -r "${QCEFF_TABLE_FILE}" ]]; then
            fatal "QCF_RHF qceff table is not readable: ${QCEFF_TABLE_FILE}"
        fi
    fi
    if [[ -e "${SINGLE_OBS_FILE}" && ( -d "${SINGLE_OBS_FILE}" || -L "${SINGLE_OBS_FILE}" ) ]]; then
        fatal "SINGLE_OBS_FILE unexpectedly is a directory/symlink: ${SINGLE_OBS_FILE}"
    fi
}

#==============================================================================
# (G) business functions
#==============================================================================

#------------------------------------------------------------------------------
# read-only: extract the target observation location from SOURCE_OBS_FILE
#------------------------------------------------------------------------------
extract_observation_location() {
    CURRENT_STEP="extract_observation_location"
    local loc_line=""
    loc_line="$(py extract "${SOURCE_OBS_FILE}" "${OBS_INDEX}")" || \
        fatal "failed to extract OBS ${OBS_INDEX} location from ${SOURCE_OBS_FILE}"
    local lon_rad lat_rad lon_deg lat_deg vert vtype=""
    read -r lon_rad lat_rad lon_deg lat_deg vert vtype <<<"${loc_line}"
    if [[ -z "${lon_deg}" || -z "${lat_deg}" ]]; then
        fatal "extract returned empty location fields"
    fi
    TARGET_LON_RAD="${lon_rad}"
    TARGET_LAT_RAD="${lat_rad}"
    TARGET_LON_DEG="${lon_deg}"
    TARGET_LAT_DEG="${lat_deg}"
    TARGET_VERT="${vert}"
    TARGET_VTYPE="${vtype}"
    log_line "target observation number : ${OBS_INDEX}"
    log_line "raw longitude (radians)   : ${TARGET_LON_RAD}"
    log_line "raw latitude  (radians)   : ${TARGET_LAT_RAD}"
    log_line "longitude (degrees)       : ${TARGET_LON_DEG}"
    log_line "latitude  (degrees)       : ${TARGET_LAT_DEG}"
    log_line "vertical coord            : ${TARGET_VERT}"
    log_line "vertical coord type       : ${TARGET_VTYPE}"
}

#------------------------------------------------------------------------------
# read-only: refuse to run on top of a previous experiment's outputs
# (must run BEFORE any experiment file is modified)
#------------------------------------------------------------------------------
check_stale_outputs() {
    CURRENT_STEP="check_stale_outputs"
    local -a stale=()
    # Only nullglob's `post_assim_me*` may be glob-expanded; a bare literal
    # path (no glob metacharacters) is always kept verbatim by the shell, so
    # test.out/fkc_dart must be checked for actual existence with -e, not
    # stuffed into a nullglob array (that made them "stale" unconditionally).
    shopt -s nullglob
    stale=( "${DART_RUN_DIR}"/post_assim_me* )
    shopt -u nullglob
    if [[ -e "${DART_RUN_DIR}/test.out" ]]; then
        stale+=( "${DART_RUN_DIR}/test.out" )
    fi
    if [[ -e "${DART_RUN_DIR}/fkc_dart" ]]; then
        stale+=( "${DART_RUN_DIR}/fkc_dart" )
    fi
    if (( ${#stale[@]} > 0 )); then
        local f=""
        for f in "${stale[@]}"; do
            log_line "stale output blocking this run: ${f}"
        done
        fatal "${#stale[@]} stale output(s) found in DART run dir (post_assim_me*/test.out/fkc_dart); remove or archive them first"
    fi
    log_line "no stale post_assim_me*/test.out/fkc_dart in DART run dir"
}

#------------------------------------------------------------------------------
# pure math (no file writes): tiny lat/lon window around the target obs
#------------------------------------------------------------------------------
compute_crop_bounds() {
    CURRENT_STEP="compute_crop_bounds"
    local out=""
    out="$(py crop_bounds "${TARGET_LON_DEG}" "${TARGET_LAT_DEG}" "${CROP_EPSILON_DEG}")" || \
        fatal "failed to compute the longitude/latitude crop window"
    local min_lat max_lat min_lon max_lon=""
    read -r min_lat max_lat min_lon max_lon <<<"${out}"
    if [[ -z "${min_lat}" || -z "${max_lat}" || -z "${min_lon}" || -z "${max_lon}" ]]; then
        fatal "crop bounds computation returned empty values"
    fi
    CROP_MIN_LAT="${min_lat}"
    CROP_MAX_LAT="${max_lat}"
    CROP_MIN_LON="${min_lon}"
    CROP_MAX_LON="${max_lon}"
    log_line "target longitude (deg)   : ${TARGET_LON_DEG}"
    log_line "target latitude  (deg)   : ${TARGET_LAT_DEG}"
    log_line "CROP_EPSILON_DEG         : ${CROP_EPSILON_DEG}"
    log_line "min_lat                  : ${CROP_MIN_LAT}"
    log_line "max_lat                  : ${CROP_MAX_LAT}"
    log_line "min_lon                  : ${CROP_MIN_LON}"
    log_line "max_lon                  : ${CROP_MAX_LON}"
}

#------------------------------------------------------------------------------
# edit the six &obs_sequence_tool_nml fields (crop window bounds + filenames)
#------------------------------------------------------------------------------
update_obs_sequence_tool_nml() {
    CURRENT_STEP="update_obs_sequence_tool_nml"
    local rb=""
    rb="$(py edit_obs_tool \
            "${OBS_TOOL_INPUT_NML}" \
            "${SOURCE_OBS_FILE}" \
            "${SINGLE_OBS_FILE}" \
            "${CROP_MIN_LAT}" "${CROP_MAX_LAT}" \
            "${CROP_MIN_LON}" "${CROP_MAX_LON}")" || \
        fatal "failed to edit &obs_sequence_tool_nml in ${OBS_TOOL_INPUT_NML}"
    local k v=""
    while read -r k v; do
        [[ -n "${k}" ]] || continue
        log_line "obs_sequence_tool nml read-back: ${k} = ${v}"
        case "${k}" in
            filename_seq)
                if [[ "${v}" != "${SOURCE_OBS_FILE}" ]]; then
                    fatal "read-back mismatch for filename_seq: got '${v}'"
                fi ;;
            filename_out)
                if [[ "${v}" != "${SINGLE_OBS_FILE}" ]]; then
                    fatal "read-back mismatch for filename_out: got '${v}'"
                fi ;;
            min_lat)
                if ! float_eq "${v}" "${CROP_MIN_LAT}"; then
                    fatal "read-back mismatch for min_lat: got '${v}'"
                fi ;;
            max_lat)
                if ! float_eq "${v}" "${CROP_MAX_LAT}"; then
                    fatal "read-back mismatch for max_lat: got '${v}'"
                fi ;;
            min_lon)
                if ! float_eq "${v}" "${CROP_MIN_LON}"; then
                    fatal "read-back mismatch for min_lon: got '${v}'"
                fi ;;
            max_lon)
                if ! float_eq "${v}" "${CROP_MAX_LON}"; then
                    fatal "read-back mismatch for max_lon: got '${v}'"
                fi ;;
        esac
    done <<<"${rb}"
}

#------------------------------------------------------------------------------
# run obs_sequence_tool from its own work directory
#------------------------------------------------------------------------------
run_obs_sequence_tool() {
    CURRENT_STEP="run_obs_sequence_tool"
    # remove only the exact target file so the crop result is provably fresh
    rm -f "${SINGLE_OBS_FILE}"
    if [[ -e "${SINGLE_OBS_FILE}" ]]; then
        fatal "unable to remove stale SINGLE_OBS_FILE: ${SINGLE_OBS_FILE}"
    fi
    local tool_out=""
    local tool_rc=0
    if tool_out="$(cd -- "${OBS_TOOL_WORK_DIR}" && "${OBS_TOOL_EXE}" 2>&1)"; then
        tool_rc=0
    else
        tool_rc=$?
    fi
    if (( tool_rc != 0 )); then
        log_line "obs_sequence_tool failed (exit ${tool_rc}); last lines:"
        printf '%s\n' "${tool_out}" | tail -n 30 | sed 's/^/    /'
        fatal "obs_sequence_tool returned non-zero status ${tool_rc}"
    fi
    log_line "obs_sequence_tool completed (exit 0)"
    if [[ ! -f "${SINGLE_OBS_FILE}" ]]; then
        fatal "obs_sequence_tool did not produce SINGLE_OBS_FILE: ${SINGLE_OBS_FILE}"
    fi
    if [[ ! -s "${SINGLE_OBS_FILE}" ]]; then
        fatal "SINGLE_OBS_FILE is empty: ${SINGLE_OBS_FILE}"
    fi
    if [[ ! -r "${SINGLE_OBS_FILE}" ]]; then
        fatal "SINGLE_OBS_FILE is not readable: ${SINGLE_OBS_FILE}"
    fi
    log_line "SINGLE_OBS_FILE created: ${SINGLE_OBS_FILE}"
}

#------------------------------------------------------------------------------
# validate the cropped file: exactly one obs, matching location
#------------------------------------------------------------------------------
validate_single_observation() {
    CURRENT_STEP="validate_single_observation"
    local insp=""
    insp="$(py inspect "${SINGLE_OBS_FILE}")" || \
        fatal "failed to validate SINGLE_OBS_FILE: ${SINGLE_OBS_FILE}"
    local s_lon s_lat s_vert s_vtype=""
    read -r s_lon s_lat s_vert s_vtype <<<"${insp}"
    if [[ -z "${s_lon}" || -z "${s_lat}" ]]; then
        fatal "single obs location is empty"
    fi
    if ! within_tol "${s_lon}" "${TARGET_LON_DEG}" "${LOC_TOLERANCE_DEG}"; then
        fatal "single obs longitude ${s_lon} != target ${TARGET_LON_DEG}"
    fi
    if ! within_tol "${s_lat}" "${TARGET_LAT_DEG}" "${LOC_TOLERANCE_DEG}"; then
        fatal "single obs latitude ${s_lat} != target ${TARGET_LAT_DEG}"
    fi
    log_line "single obs file validated: exactly 1 obs at lon=${s_lon} lat=${s_lat}"
}

#------------------------------------------------------------------------------
# install the single obs into the DART run dir as the exact file obs_seq.out
#------------------------------------------------------------------------------
install_dart_obs_seq() {
    CURRENT_STEP="install_dart_obs_seq"
    rm -f "${DART_OBS_FILE}"
    if [[ -e "${DART_OBS_FILE}" ]]; then
        fatal "unable to remove DART obs_seq.out: ${DART_OBS_FILE}"
    fi
    cp "${SINGLE_OBS_FILE}" "${DART_OBS_FILE}"
    if [[ ! -f "${DART_OBS_FILE}" || ! -s "${DART_OBS_FILE}" || ! -r "${DART_OBS_FILE}" ]]; then
        fatal "DART obs_seq.out missing/empty/unreadable after copy: ${DART_OBS_FILE}"
    fi
    local insp=""
    insp="$(py inspect "${DART_OBS_FILE}")" || \
        fatal "DART obs_seq.out failed the single-observation check"
    local d_lon d_lat d_vert d_vtype=""
    read -r d_lon d_lat d_vert d_vtype <<<"${insp}"
    if ! within_tol "${d_lon}" "${TARGET_LON_DEG}" "${LOC_TOLERANCE_DEG}"; then
        fatal "DART obs_seq.out longitude mismatch ${d_lon} vs ${TARGET_LON_DEG}"
    fi
    if ! within_tol "${d_lat}" "${TARGET_LAT_DEG}" "${LOC_TOLERANCE_DEG}"; then
        fatal "DART obs_seq.out latitude mismatch ${d_lat} vs ${TARGET_LAT_DEG}"
    fi
    log_line "DART obs_seq.out installed from SINGLE_OBS_FILE (lon=${d_lon} lat=${d_lat})"
}

#------------------------------------------------------------------------------
# set qceff_table_filename according to FILTER_TYPE
#------------------------------------------------------------------------------
configure_dart_filter() {
    CURRENT_STEP="configure_dart_filter"
    local quoted_val="" expect_val=""
    if [[ "${FILTER_TYPE}" == "EAKF" ]]; then
        quoted_val="''"
        expect_val=""
    else
        quoted_val="'qceff_table_fkc.csv'"
        expect_val="qceff_table_fkc.csv"
        # defensive re-check (also validated in validate_config)
        if [[ ! -s "${QCEFF_TABLE_FILE}" ]]; then
            fatal "QCF_RHF qceff table missing/empty/unreadable: ${QCEFF_TABLE_FILE}"
        fi
    fi
    local rb=""
    rb="$(py edit_qceff "${DART_INPUT_NML}" "${quoted_val}")" || \
        fatal "failed to edit qceff_table_filename in ${DART_INPUT_NML}"
    local var="" val=""
    read -r var val <<<"${rb}"
    if [[ "${var}" != "qceff_table_filename" ]]; then
        fatal "unexpected qceff read-back: ${rb}"
    fi
    if [[ "${val}" != "${expect_val}" ]]; then
        fatal "qceff_table_filename read-back '${val}' != expected '${expect_val}'"
    fi
    log_line "${FILTER_TYPE}: qceff_table_filename = '${val}'"
}

#------------------------------------------------------------------------------
# submit sub_dart.sh via bsub from the DART run dir
#------------------------------------------------------------------------------
submit_dart_job() {
    CURRENT_STEP="submit_dart_job"
    local bsub_out="" bsub_rc=0
    if bsub_out="$(cd -- "${DART_RUN_DIR}" && bsub < ./sub_dart.sh 2>&1)"; then
        bsub_rc=0
    else
        bsub_rc=$?
    fi
    if (( bsub_rc != 0 )); then
        log_line "bsub failed (exit ${bsub_rc}); output:"
        printf '%s\n' "${bsub_out}" | sed 's/^/    /'
        fatal "bsub returned non-zero status ${bsub_rc}"
    fi
    local n_matches=0
    n_matches="$(printf '%s\n' "${bsub_out}" | grep -Eo 'Job[[:space:]]*<[0-9]+>' | wc -l | tr -d ' ' || true)"
    if [[ ! "${n_matches}" =~ ^[0-9]+$ ]] || (( ${n_matches} != 1 )); then
        log_line "could not parse a unique LSF job id from bsub output: ${bsub_out}"
        fatal "unable to parse unique LSF job id from bsub output"
    fi
    LSF_JOB_ID="$(printf '%s\n' "${bsub_out}" | grep -oE 'Job[[:space:]]*<[0-9]+>' | head -n1 | grep -oE '[0-9]+' | head -n1)"
    if [[ ! "${LSF_JOB_ID}" =~ ^[0-9]+$ ]]; then
        fatal "parsed LSF job id is invalid: '${LSF_JOB_ID}'"
    fi
    log_line "submitted one LSF job: ${LSF_JOB_ID}"
    log_line "bsub output: ${bsub_out}"
}

#------------------------------------------------------------------------------
# helpers for LSF final-state resolution
#------------------------------------------------------------------------------
get_lsf_status() {  # echo the status column from bjobs / bjobs -a; empty if unknown
    local job_id="$1" out="" st=""
    if out="$(bjobs -noheader -o 'jobid stat' "${job_id}" 2>/dev/null)"; then
        st="$(printf '%s\n' "${out}" | awk -v jid="${job_id}" '$1==jid{print $2; exit}')"
    fi
    if [[ -z "${st}" ]]; then
        if out="$(bjobs -a -noheader -o 'jobid stat' "${job_id}" 2>/dev/null)"; then
            st="$(printf '%s\n' "${out}" | awk -v jid="${job_id}" '$1==jid{print $2; exit}')"
        fi
    fi
    printf '%s' "${st}"
}

parse_bhist_final_status() {  # echo the LAST explicit DONE/EXIT event from bhist -l
    local job_id="$1" hist="" line="" final=""
    hist="$(bhist -l "${job_id}" 2>/dev/null)" || hist=""
    [[ -n "${hist}" ]] || { printf ''; return 0; }
    while IFS= read -r line; do
        case "${line}" in
            *"Done successfully"*|*"Done"*|*"DONE"*)
                final="DONE" ;;
            *"Exited"*|*"EXIT"*)
                final="EXIT" ;;
        esac
    done <<<"${hist}"
    printf '%s' "${final}"
}

#------------------------------------------------------------------------------
# poll LSF until the job reaches a definitive final state (DONE or EXIT)
#------------------------------------------------------------------------------
wait_for_lsf_job() {
    CURRENT_STEP="wait_for_lsf_job"
    local job_id="$1"
    local total=0
    while :; do
        local status="" attempt=0
        # 1) current status via bjobs, then bjobs -a for finished jobs
        status="$(get_lsf_status "${job_id}")"
        # 2) freshly submitted jobs may not be visible yet: bounded retries
        if [[ -z "${status}" ]]; then
            while (( attempt < LSF_QUERY_RETRIES )); do
                log_line "LSF job ${job_id} not found via bjobs; retry $((attempt + 1))/${LSF_QUERY_RETRIES}"
                sleep "${LSF_QUERY_RETRY_DELAY}"
                status="$(get_lsf_status "${job_id}")"
                [[ -n "${status}" ]] && break
                attempt=$(( attempt + 1 ))
            done
        fi
        # 3) last resort: parse the final explicit state from bhist
        if [[ -z "${status}" ]]; then
            status="$(parse_bhist_final_status "${job_id}")"
        fi
        if [[ -z "${status}" ]]; then
            fatal "unable to determine final LSF state of job ${job_id} (bjobs/bjobs -a/bhist all failed)"
        fi
        case "${status}" in
            DONE)
                log_line "LSF job ${job_id} reached definitive final state DONE"
                return 0
                ;;
            EXIT)
                fatal "LSF job ${job_id} reached definitive final state EXIT"
                ;;
            PEND|RUN|PSUSP|SSUSP|USUSP)
                log_line "LSF job ${job_id} status=${status} (waited ${total}s); polling again"
                ;;
            *)
                fatal "LSF job ${job_id} in unexpected status '${status}'"
                ;;
        esac
        sleep "${POLL_INTERVAL}"
        total=$(( total + POLL_INTERVAL ))
        if (( total >= MAX_POLL_SECONDS )); then
            log_line "reached MAX_POLL_SECONDS=${MAX_POLL_SECONDS} for job ${job_id}; not issuing bkill, aborting"
            fatal "LSF job ${job_id} did not finish within ${MAX_POLL_SECONDS}s"
        fi
    done
}

#------------------------------------------------------------------------------
# verify DART completion: fkc_dart marker + test.out + post_assim_me*
#------------------------------------------------------------------------------
check_dart_outputs() {
    CURRENT_STEP="check_dart_outputs"
    local -a pa=()
    shopt -s nullglob
    pa=( "${DART_RUN_DIR}"/post_assim_me* )
    shopt -u nullglob
    if (( ${#pa[@]} == 0 )); then
        fatal "no post_assim_me* files found in ${DART_RUN_DIR}; DART did not produce ensemble analysis output"
    fi
    if [[ ! -f "${DART_RUN_DIR}/test.out" ]]; then
        fatal "test.out does not exist in ${DART_RUN_DIR}"
    fi
    if [[ ! -s "${DART_RUN_DIR}/test.out" ]]; then
        fatal "test.out is empty in ${DART_RUN_DIR}"
    fi
    if [[ ! -f "${DART_RUN_DIR}/fkc_dart" ]]; then
        fatal "fkc_dart marker missing in ${DART_RUN_DIR}; sub_dart.sh did not complete the whole DART flow"
    fi
    # test.err is informational only (programs may legitimately write warnings
    # to stderr).  Warn if non-empty but never mark the run failed on this alone.
    if [[ -s "${DART_RUN_DIR}/test.err" ]]; then
        log_line "WARNING: test.err is non-empty (warnings/notes may be written to stderr):"
        tail -n 20 "${DART_RUN_DIR}/test.err" | sed 's/^/    /'
    fi
    log_line "DART outputs present: ${#pa[@]} post_assim_me* file(s), test.out (non-empty), fkc_dart"
}

#------------------------------------------------------------------------------
# archive post_assim_me* and test.out, never overwriting existing archives
#------------------------------------------------------------------------------
archive_outputs() {
    CURRENT_STEP="archive_outputs"
    local archive_dir="${ARCHIVE_ROOT}/${FILTER_TYPE}/obs_seq${OBS_INDEX}"
    local -a pa=()
    shopt -s nullglob
    pa=( "${DART_RUN_DIR}"/post_assim_me* )
    shopt -u nullglob
    if (( ${#pa[@]} == 0 )); then
        fatal "no post_assim_me* to archive in ${DART_RUN_DIR}"
    fi
    mkdir -p "$(dirname "${archive_dir}")" "${archive_dir}"
    if [[ -d "${archive_dir}" && -n "$(ls -A "${archive_dir}" 2>/dev/null)" ]]; then
        fatal "archive directory exists and is not empty; refusing to overwrite: ${archive_dir}"
    fi

    local -a move_targets=("${pa[@]}" "${DART_RUN_DIR}/test.out")
    local -a moved_files=()
    local -a failed_files=()
    local f=""
    for f in "${move_targets[@]}"; do
        if mv -f "${f}" "${archive_dir}/"; then
            moved_files+=("${f}")
            log_line "archived ${f}"
        else
            failed_files+=("${f}")
            log_line "FAILED to archive ${f}"
        fi
    done

    if (( ${#failed_files[@]} > 0 )); then
        log_line "successfully moved files:"
        printf '    %s\n' "${moved_files[@]+"${moved_files[@]}"}"
        log_line "files that failed to move (still in DART run dir):"
        printf '    %s\n' "${failed_files[@]+"${failed_files[@]}"}"
        local -a leftover=()
        shopt -s nullglob
        leftover=( "${DART_RUN_DIR}"/post_assim_me* "${DART_RUN_DIR}"/test.out )
        shopt -u nullglob
        if (( ${#leftover[@]} > 0 )); then
            log_line "files still present in DART run dir:"
            printf '    %s\n' "${leftover[@]+"${leftover[@]}"}"
        fi
        fatal "partial archive: ${#failed_files[@]} file(s) failed to move"
    fi

    # verify what landed in the archive
    local -a archived_pa=()
    shopt -s nullglob
    archived_pa=( "${archive_dir}"/post_assim_me* )
    shopt -u nullglob
    if (( ${#archived_pa[@]} != ${#pa[@]} )); then
        fatal "archive incomplete: expected ${#pa[@]} post_assim_me* files, found ${#archived_pa[@]} in ${archive_dir}"
    fi
    if [[ ! -s "${archive_dir}/test.out" ]]; then
        fatal "archived test.out missing or empty in ${archive_dir}"
    fi

    # verify originals are gone
    local -a leftover=()
    shopt -s nullglob
    leftover=( "${DART_RUN_DIR}"/post_assim_me* "${DART_RUN_DIR}"/test.out )
    shopt -u nullglob
    if (( ${#leftover[@]} > 0 )); then
        local lf=""
        for lf in "${leftover[@]}"; do
            log_line "still present in DART run dir: ${lf}"
        done
        fatal "archived originals still present (${#leftover[@]} file(s))"
    fi
    log_line "archive complete: ${archive_dir}"
}

#------------------------------------------------------------------------------
# main flow (check_stale_outputs runs before ANY experiment file is modified)
#------------------------------------------------------------------------------
main() {
    CURRENT_STEP="main"
    validate_config

    # install log redirection now that DART_RUN_DIR is known to exist;
    # writing this log does not touch experiment files
    SCRIPT_LOG="${DART_RUN_DIR}/run_single_obs_dart_${FILTER_TYPE}_obs${OBS_INDEX}.log"
    : > "${SCRIPT_LOG}" || fatal "cannot create/truncate log file: ${SCRIPT_LOG}"
    exec > >(tee -a "${SCRIPT_LOG}") 2>&1

    log_line "run_single_obs_dart.sh starting"
    log_line "FILTER_TYPE=${FILTER_TYPE}  OBS_INDEX=${OBS_INDEX}"
    log_line "PROJECT_ROOT=${PROJECT_ROOT}"
    log_line "DART_SOURCE_ROOT=${DART_SOURCE_ROOT}"
    log_line "SOURCE_OBS_FILE=${SOURCE_OBS_FILE}"
    log_line "PYTHON_EXE=${PYTHON_EXE}  POLL_INTERVAL=${POLL_INTERVAL}s"
    log_line "CROP_EPSILON_DEG=${CROP_EPSILON_DEG}  LOC_TOLERANCE_DEG=${LOC_TOLERANCE_DEG}"
    log_line "SINGLE_OBS_FILE=${SINGLE_OBS_FILE}"

    extract_observation_location
    check_stale_outputs
    compute_crop_bounds
    update_obs_sequence_tool_nml
    run_obs_sequence_tool
    validate_single_observation
    install_dart_obs_seq
    configure_dart_filter
    submit_dart_job
    wait_for_lsf_job "${LSF_JOB_ID}"
    check_dart_outputs
    archive_outputs

    log_line "all done: FILTER_TYPE=${FILTER_TYPE} OBS_INDEX=${OBS_INDEX} archived to ${ARCHIVE_ROOT}/${FILTER_TYPE}/obs_seq${OBS_INDEX}"
    log_line "run log: ${SCRIPT_LOG}"
    return 0
}

main "$@"