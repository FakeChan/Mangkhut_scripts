#!/usr/bin/env bash
#BSUB -J cycle_50mem
#BSUB -q serial
#BSUB -n 1
#BSUB -oo /share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA/cycle_50mem_%J.out
#BSUB -eo /share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA/cycle_50mem_%J.err
#
# LSF wrapper for run_driver_50mem_cyclingDA.sh.
#
# Usage:
#   bsub < submit_50mem_cyclingDA.sh
#
# The fkc environment (HDF5 / Intel / netcdf / MPI / MATLAB / NCO) is loaded
# INSIDE the job, so it does not matter which shell submitted the job or
# whether `fkc` was run there. The driver itself submits the DART and WRF
# jobs with bsub; those inherit this environment in turn.
#
# Override defaults from the submitting shell (bsub propagates exported vars):
#   RESUME=0 bsub < submit_50mem_cyclingDA.sh
#   RUN_GROUP_1=1 RUN_GROUP_2=1 RUN_GROUP_3=1 bsub < submit_50mem_cyclingDA.sh
# Or use the LSF -env option:
#   bsub -env "MODE=PREFLIGHT" < submit_50mem_cyclingDA.sh
#
# Interactive checks (no LSF, no runtime directories, no jobs):
#   MODE=DRY_RUN  bash submit_50mem_cyclingDA.sh
#   MODE=PREFLIGHT bash submit_50mem_cyclingDA.sh
#
# Full production preset (all groups, both filters, d02 physical grid):
#   RUN_GROUP_1=1 RUN_GROUP_2=1 RUN_GROUP_3=1 \
#   RUN_EAKF=1 RUN_QCF_RHF=1 HX_STATE_DOMAIN=d02 \
#   bsub < submit_50mem_cyclingDA.sh
#
# The exit status of the LSF job is the driver's exit status.

set -Eeo pipefail

# ---------------------------------------------------------------------------
# run configuration (defaults = the group-3 / EAKF / d01 test configuration)
# ---------------------------------------------------------------------------
export MODE="${MODE:-RUN}"                          # RUN / PREFLIGHT / DRY_RUN
export RESUME="${RESUME:-1}"                        # 1 = reuse verified stages
export HX_STATE_DOMAIN="${HX_STATE_DOMAIN:-d01}"    # physical grid for H(x)
export RUN_ROOT="${RUN_ROOT:-/scratch/lililei1/kcfu/tc_mangkhut/5cyclingDA/50mem_hourly_d01hx}"
export RUN_GROUP_1="${RUN_GROUP_1:-0}"              # 1 = no ocean update/model
export RUN_GROUP_2="${RUN_GROUP_2:-0}"              # 1 = ocean model, no update
export RUN_GROUP_3="${RUN_GROUP_3:-1}"              # 1 = ocean update + model
export RUN_EAKF="${RUN_EAKF:-1}"
export RUN_QCF_RHF="${RUN_QCF_RHF:-0}"

# ---------------------------------------------------------------------------
# cluster paths
# ---------------------------------------------------------------------------
FKC_PROFILE=/share/home/lililei1/kcfu/bashrc
PROJECT=/share/home/lililei1/kcfu/tc_mangkhut
SCRIPT_DIR="${PROJECT}/5cyclingDA"
DRIVER="${SCRIPT_DIR}/run_driver_50mem_cyclingDA.sh"
RTTOV_ROOT="${RTTOV_ROOT:-/share/home/lililei1/kcfu/models/rttov123}"

# ---------------------------------------------------------------------------
# load the same environment used by all previous working runs
# ---------------------------------------------------------------------------
# The fkc profile is written for interactive use: it does not tolerate
# `set -u` (PS1 is treated as unset) and may trip `set -e` internally, so
# load it with errexit temporarily disabled and validate the result below.
set +e
source "${FKC_PROFILE}"
fkc_rc=$?
set -e
if (( fkc_rc != 0 )); then
    echo "ERROR: sourcing ${FKC_PROFILE} failed (rc=${fkc_rc})" >&2
    exit 1
fi

# Fail before touching RUN_ROOT or submitting anything if the RTTOV/HDF5
# environment is incomplete. This is the check that the failed 2026-11-22
# run was missing.
if [[ -z "${H5PATH:-}" || ! -d "${H5PATH}/lib" ]]; then
    echo "ERROR: H5PATH does not point at a valid HDF5 installation: ${H5PATH:-<unset>}" >&2
    exit 1
fi
case ":${LD_LIBRARY_PATH:-}:" in
    *":${H5PATH}/lib:"*) ;;
    *) echo "ERROR: ${H5PATH}/lib is missing from LD_LIBRARY_PATH" >&2; exit 1 ;;
esac
if [[ ! -f "${RTTOV_ROOT}/bin/example_fwd.exe" ]]; then
    echo "ERROR: RTTOV executable not found: ${RTTOV_ROOT}/bin/example_fwd.exe" >&2
    exit 1
fi
if ldd "${RTTOV_ROOT}/bin/example_fwd.exe" 2>/dev/null | grep -q "not found"; then
    echo "ERROR: RTTOV example_fwd.exe has unresolved shared libraries:" >&2
    ldd "${RTTOV_ROOT}/bin/example_fwd.exe" | grep "not found" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# launch the driver
# ---------------------------------------------------------------------------
[[ -f "${DRIVER}" ]] || { echo "ERROR: driver not found: ${DRIVER}" >&2; exit 1; }
cd "${SCRIPT_DIR}"

echo "=============================================================="
echo " LSF job         : ${LSB_JOBID:-<not under LSF>}"
echo " execution host  : $(hostname)"
echo " start (UTC)     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo " driver          : ${DRIVER}"
echo " MODE            : ${MODE}"
echo " RESUME          : ${RESUME}"
echo " HX_STATE_DOMAIN : ${HX_STATE_DOMAIN}"
echo " RUN_ROOT        : ${RUN_ROOT}"
echo " groups 1/2/3    : ${RUN_GROUP_1}/${RUN_GROUP_2}/${RUN_GROUP_3}"
echo " filters EAKF/QCF: ${RUN_EAKF}/${RUN_QCF_RHF}"
echo " HDF5 library    : ${H5PATH}/lib"
echo "=============================================================="

# This cluster's bjobs omits the [member] suffix for array elements and
# purges completed elements, so the driver cannot map array states.  The
# wrappers in bin/ reconstruct them from bhist; they only affect this job's
# PATH and do not change the driver or its experiment signature.
export PATH="${SCRIPT_DIR}/bin:${PATH}"

exec bash "${DRIVER}"
