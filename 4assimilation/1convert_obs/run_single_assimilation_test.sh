#!/bin/bash
#================================================================================
# run_single_assimilation_test.sh
#
# Run a single "offline" AMSUA assimilation observation test (truth + ensemble
# Hx + obs text conversion + Hx merge + text_to_obs) WITHOUT touching the
# fixed cyclingDA production workflow (run_driver_cyclingDA.sh stays untouched,
# nothing here is integrated into it).
#
# Supports both LACC and STANDARD single-observation experiments via the
# EXPERIMENT_MODE variable. Both modes apply the SAME clear-sky mask (from the
# truth run) to every ensemble member.
#================================================================================
set -euo pipefail

#==============================================================================
# centralized experiment configuration
#==============================================================================
EXPERIMENT_MODE="${EXPERIMENT_MODE:-LACC}"

CENTER_DAY="${CENTER_DAY:-10}"
CENTER_HOUR="${CENTER_HOUR:-00}"
CENTER_MIN="${CENTER_MIN:-00}"

ASSIM_CHANNEL="${ASSIM_CHANNEL:-4}"
LACC_LAG_HOURS="${LACC_LAG_HOURS:-0 3 6 9}"

ENS_SIZE="${ENS_SIZE:-50}"
ENS_WRFOUT_DIR="${ENS_WRFOUT_DIR:-/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst}"

RTTOV_SCATT="${RTTOV_SCATT:-0}"
USE_TOTAL_ICE="${USE_TOTAL_ICE:-0}"

USE_CLEAR_SKY_MASK="${USE_CLEAR_SKY_MASK:-1}"
CLEAR_SKY_THRESHOLD="${CLEAR_SKY_THRESHOLD:-0.2}"

OBS_ERR_STD="${OBS_ERR_STD:-0.5}"

RUN_TRUTH="${RUN_TRUTH:-1}"
RUN_ENSEMBLE="${RUN_ENSEMBLE:-1}"
RUN_OBS_CONVERT="${RUN_OBS_CONVERT:-1}"
RUN_HX_MERGE="${RUN_HX_MERGE:-1}"
RUN_TEXT_TO_OBS="${RUN_TEXT_TO_OBS:-1}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"

OBS_SEQ_OUT_NAME="${OBS_SEQ_OUT_NAME:-obs_seq.out_kctest1_d01_10_00_00_LACC_ch4_clear02}"
OVERWRITE_OBS_SEQ="${OVERWRITE_OBS_SEQ:-0}"

#==============================================================================
# fixed paths
#==============================================================================
BASE_DIR="/share/home/lililei1/kcfu/tc_mangkhut"
HX_DIR="${BASE_DIR}/3create_obs/hx_rttov"
CONVERT_DIR="${BASE_DIR}/4assimilation/1convert_obs"
TEXT_TO_OBS_RUN_DIR="${CONVERT_DIR}/run_dir"

PYTHON_BIN="/share/home/lililei1/kcfu/anaconda/envs/wrf/bin/python"

#==============================================================================
# central time
#==============================================================================
CURRENT_TIME="${CENTER_DAY}_${CENTER_HOUR}_${CENTER_MIN}"

#-----------------------------------------------------------------------------
# guards
#-----------------------------------------------------------------------------
if [[ "${EXPERIMENT_MODE}" != "LACC" && "${EXPERIMENT_MODE}" != "STANDARD" ]]; then
    echo "ERROR: EXPERIMENT_MODE must be LACC or STANDARD, got '${EXPERIMENT_MODE}'" >&2
    exit 1
fi

if [[ "${RUN_VALIDATION}" != "0" && "${RUN_VALIDATION}" != "1" ]]; then
    echo "ERROR: RUN_VALIDATION must be 0 or 1, got '${RUN_VALIDATION}'" >&2
    exit 1
fi

if [[ "${ENS_SIZE}" != "50" ]]; then
    echo "ERROR: this workflow requires ENS_SIZE=50 because the existing text_to_obs executable is compiled with ens_size=50." >&2
    exit 1
fi

if [[ "${USE_CLEAR_SKY_MASK}" == "1" && "${RTTOV_SCATT}" != "0" ]]; then
    echo "ERROR: USE_CLEAR_SKY_MASK=1 requires RTTOV_SCATT=0 in this single-assimilation clear-sky workflow." >&2
    exit 1
fi

if [[ "${OBS_SEQ_OUT_NAME}" == */* ]]; then
    echo "ERROR: OBS_SEQ_OUT_NAME must not contain '/': ${OBS_SEQ_OUT_NAME}" >&2
    exit 1
fi

OBS_SEQ_OUT_FILE="${TEXT_TO_OBS_RUN_DIR}/${OBS_SEQ_OUT_NAME}"

if [[ "${RUN_TEXT_TO_OBS}" == "1" &&
      -e "${OBS_SEQ_OUT_FILE}" &&
      "${OVERWRITE_OBS_SEQ}" != "1" ]]; then
    echo "ERROR: OBS_SEQ_OUT_FILE already exists (set OVERWRITE_OBS_SEQ=1 to replace):" >&2
    echo "       ${OBS_SEQ_OUT_FILE}" >&2
    exit 1
fi

#==============================================================================
# environment export (shared)
#==============================================================================
export current_day="${CENTER_DAY}"
export current_hour="${CENTER_HOUR}"
export current_min="${CENTER_MIN}"
export current_time="${CURRENT_TIME}"

export lacc_center_day="${CENTER_DAY}"
export lacc_center_hour="${CENTER_HOUR}"
export lacc_center_min="${CENTER_MIN}"
export LACC_LAG_HOURS

export domain="d01"
export sensor="AMSUA"
export instrument="AMSUA"
export assim_channel="${ASSIM_CHANNEL}"

export NOBS="676"
export ENS_SIZE
export ens_wrfout_dir="${ENS_WRFOUT_DIR}"

export rttov_scatt="${RTTOV_SCATT}"
export use_total_ice="${USE_TOTAL_ICE}"

export USE_CLEAR_SKY_MASK
export CLEAR_SKY_THRESHOLD
export OBS_ERR_STD
export obs_err_std="${OBS_ERR_STD}"

# number of distinct LACC lag hours -> EXPECTED_LACC_COUNT
# every lag must be a non-negative integer; at least one must be 0 (center time)
EXPECTED_LACC_COUNT=0
LACC_HAS_ZERO="0"

for _lag in ${LACC_LAG_HOURS}; do
    if ! [[ "${_lag}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: every value in LACC_LAG_HOURS must be a non-negative integer, got '${_lag}'." >&2
        exit 1
    fi

    EXPECTED_LACC_COUNT=$((EXPECTED_LACC_COUNT + 1))

    if (( 10#${_lag} == 0 )); then
        LACC_HAS_ZERO="1"
    fi
done

if [[ "${EXPECTED_LACC_COUNT}" -le 0 ]]; then
    echo "ERROR: LACC_LAG_HOURS must contain at least one time." >&2
    exit 1
fi

# STANDARD mode may compute the count too, but only LACC requires a zero lag
if [[ "${EXPERIMENT_MODE}" == "LACC" && "${LACC_HAS_ZERO}" != "1" ]]; then
    echo "ERROR: LACC_LAG_HOURS must contain 0 because PARA_FILE uses the center-time truth profile." >&2
    exit 1
fi

export EXPECTED_LACC_COUNT

#==============================================================================
# mode-dependent paths
#==============================================================================
if [[ "${EXPERIMENT_MODE}" == "LACC" ]]; then
    OBS_BT_DIR="${HX_DIR}/3obs_BT_LACC/AMSUA"
    OBS_BT_SUBDIR="BT_LACC_${CURRENT_TIME}"

    PROFILE_DIR="${HX_DIR}/profile"
    PROFILE_SUBDIR="profile_d01_LACC_${CURRENT_TIME}"
    PARA_FILE="${PROFILE_DIR}/${PROFILE_SUBDIR}/prof${CENTER_DAY}_${CENTER_HOUR}:${CENTER_MIN}.dat"

    ENS_BT_DIR="${HX_DIR}/4ens_BT_LACC"
    ENS_BT_SUBDIR="BT_LACC_${CURRENT_TIME}"

    CLEAR_SKY_MASK_FILE="${OBS_BT_DIR}/${OBS_BT_SUBDIR}/clear_sky_mask.txt"
    LACC_TIMES_FILE="${OBS_BT_DIR}/${OBS_BT_SUBDIR}/LACC_times.txt"

    TRUTH_DRIVER="${HX_DIR}/run_rttov_TrueObs_LACC_driver.sh"
    ENSEMBLE_DRIVER="${HX_DIR}/run_rttov_ensBT_LACC_driver.sh"
    OBS_CONVERT_SCRIPT="${CONVERT_DIR}/obs2DART_LACC.py"
else
    OBS_BT_DIR="${HX_DIR}/3obs_BT/AMSUA"
    OBS_BT_SUBDIR="BT_${CURRENT_TIME}"

    PROFILE_DIR="${HX_DIR}/profile"
    PROFILE_SUBDIR="profile_d01"
    PARA_FILE="${PROFILE_DIR}/${PROFILE_SUBDIR}/prof${CENTER_DAY}_${CENTER_HOUR}:${CENTER_MIN}.dat"

    ENS_BT_DIR="${HX_DIR}/4ens_BT"
    ENS_BT_SUBDIR="BT_${CURRENT_TIME}"

    CLEAR_SKY_MASK_FILE="${OBS_BT_DIR}/${OBS_BT_SUBDIR}/clear_sky_mask.txt"

    TRUTH_DRIVER="${HX_DIR}/run_rttov_TrueObs_driver.sh"
    ENSEMBLE_DRIVER="${HX_DIR}/run_rttov_ensBT_driver.sh"
    OBS_CONVERT_SCRIPT="${CONVERT_DIR}/obs2DART_test.py"
fi

#==============================================================================
# single-test work area and fixed intermediate files
#==============================================================================
SINGLE_TEST_WORK_DIR="${CONVERT_DIR}/single_assim_test/${EXPERIMENT_MODE}_${CURRENT_TIME}_ch${ASSIM_CHANNEL}"
LOG_DIR="${SINGLE_TEST_WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"

OBS_DART_INPUT_FILE="${SINGLE_TEST_WORK_DIR}/obs_input.txt"
MERGED_FO_FILE="${SINGLE_TEST_WORK_DIR}/merged_FO_data.txt"
TEST_INPUT_NML="${SINGLE_TEST_WORK_DIR}/input.nml"

# The production text_to_obs input.nml references rttov_sensor_db_file =
# 'rttov_sensor_db.csv' relative to the working directory, so text_to_obs needs
# a symlink to the sensor database inside the single-test work directory.
RTTOV_SENSOR_DB_SOURCE="${TEXT_TO_OBS_RUN_DIR}/rttov_sensor_db.csv"
RTTOV_SENSOR_DB_LINK="${SINGLE_TEST_WORK_DIR}/rttov_sensor_db.csv"

export OBS_BT_DIR
export OBS_BT_SUBDIR
export PROFILE_DIR
export PROFILE_SUBDIR
export PARA_FILE
export ENS_BT_DIR
export ENS_BT_SUBDIR
export CLEAR_SKY_MASK_FILE
export OBS_DART_INPUT_FILE
export MERGED_FO_FILE

if [[ "${EXPERIMENT_MODE}" == "LACC" ]]; then
    export LACC_TIMES_FILE
fi

#==============================================================================
# summary counters (defined up front so later stages stay consistent)
#==============================================================================
n_mask_rows=0
n_mask_one=0
n_mask_zero=0
n_mask_invalid=0
n_obs_rows=0
n_merged_rows=0
expected_rows=""

log_stage() {
    echo ""
    echo "======================================================================"
    echo "== $1"
    echo "======================================================================"
}

#==============================================================================
# validation helpers (used by RUN_VALIDATION stage and post-text_to_obs check)
#==============================================================================

#-----------------------------------------------------------------------------
# set expected_rows and mask statistics based on USE_CLEAR_SKY_MASK
#-----------------------------------------------------------------------------
validate_mask_and_set_expected_rows() {
    if [[ "${USE_CLEAR_SKY_MASK}" == "1" ]]; then
        if [[ ! -f "${CLEAR_SKY_MASK_FILE}" ]]; then
            echo "ERROR: clear-sky mask does not exist: ${CLEAR_SKY_MASK_FILE}" >&2
            exit 1
        fi

        n_mask_rows=$(wc -l < "${CLEAR_SKY_MASK_FILE}")

        if [[ "${n_mask_rows}" -ne "${NOBS}" ]]; then
            echo "ERROR: clear-sky mask has ${n_mask_rows} rows, expected ${NOBS}: ${CLEAR_SKY_MASK_FILE}" >&2
            exit 1
        fi

        n_mask_one=$(awk '
            NF == 1 && $1 == 1 { count++ }
            END { print count + 0 }
        ' "${CLEAR_SKY_MASK_FILE}")

        n_mask_zero=$(awk '
            NF == 1 && $1 == 0 { count++ }
            END { print count + 0 }
        ' "${CLEAR_SKY_MASK_FILE}")

        n_mask_invalid=$(awk '
            NF != 1 || ($1 != 0 && $1 != 1) { count++ }
            END { print count + 0 }
        ' "${CLEAR_SKY_MASK_FILE}")

        if [[ "${n_mask_invalid}" -ne 0 ]]; then
            echo "ERROR: clear-sky mask contains ${n_mask_invalid} invalid rows; only one value of 0 or 1 is allowed per row: ${CLEAR_SKY_MASK_FILE}" >&2
            exit 1
        fi

        if [[ $((n_mask_one + n_mask_zero)) -ne "${NOBS}" ]]; then
            echo "ERROR: clear-sky mask counts do not sum to NOBS." >&2
            exit 1
        fi

        expected_rows="${n_mask_one}"
    elif [[ "${USE_CLEAR_SKY_MASK}" == "0" ]]; then
        n_mask_rows="${NOBS}"
        n_mask_one="${NOBS}"
        n_mask_zero="0"
        n_mask_invalid="0"
        expected_rows="${NOBS}"
    else
        echo "ERROR: USE_CLEAR_SKY_MASK must be 0 or 1, got ${USE_CLEAR_SKY_MASK}" >&2
        exit 1
    fi
}

#-----------------------------------------------------------------------------
# full validation of mask + filtered observations + merged Hx
#-----------------------------------------------------------------------------
validate_intermediate_files() {
    validate_mask_and_set_expected_rows

    if [[ ! -s "${OBS_DART_INPUT_FILE}" ]]; then
        echo "ERROR: observation input file is missing or empty: ${OBS_DART_INPUT_FILE}" >&2
        exit 1
    fi

    if [[ ! -s "${MERGED_FO_FILE}" ]]; then
        echo "ERROR: merged Hx file is missing or empty: ${MERGED_FO_FILE}" >&2
        exit 1
    fi

    n_obs_rows=$(wc -l < "${OBS_DART_INPUT_FILE}")
    n_merged_rows=$(wc -l < "${MERGED_FO_FILE}")

    if [[ "${n_obs_rows}" -ne "${expected_rows}" ]]; then
        echo "ERROR: observation input has ${n_obs_rows} rows, expected ${expected_rows}." >&2
        exit 1
    fi

    if [[ "${n_merged_rows}" -ne "${expected_rows}" ]]; then
        echo "ERROR: merged Hx has ${n_merged_rows} rows, expected ${expected_rows}." >&2
        exit 1
    fi

    if ! awk -v expected_cols="${ENS_SIZE}" '
        {
            if (NF != expected_cols) {
                print "bad NF=" NF " at line " NR \
                      ", expected " expected_cols > "/dev/stderr"
                exit 1
            }
        }
    ' "${MERGED_FO_FILE}"; then
        echo "ERROR: merged Hx column count does not match ENS_SIZE=${ENS_SIZE}: ${MERGED_FO_FILE}" >&2
        exit 1
    fi

    echo "Intermediate validation passed:"
    echo "  expected rows: ${expected_rows}"
    echo "  observation rows: ${n_obs_rows}"
    echo "  merged Hx shape: ${n_merged_rows} x ${ENS_SIZE}"
}

#-----------------------------------------------------------------------------
# validation of the written obs_seq output
#-----------------------------------------------------------------------------
validate_obs_seq_file() {
    if [[ ! -s "${OBS_SEQ_OUT_FILE}" ]]; then
        echo "ERROR: obs_seq output is missing or empty: ${OBS_SEQ_OUT_FILE}" >&2
        exit 1
    fi

    validate_mask_and_set_expected_rows

    obs_seq_num_obs=$(awk '
        /num_obs:/ {
            for (i = 1; i <= NF; i++) {
                if ($i == "num_obs:") {
                    print $(i + 1)
                    exit
                }
            }
        }
    ' "${OBS_SEQ_OUT_FILE}")

    if [[ -z "${obs_seq_num_obs}" ]]; then
        echo "ERROR: cannot find num_obs in obs_seq output: ${OBS_SEQ_OUT_FILE}" >&2
        exit 1
    fi

    if ! [[ "${obs_seq_num_obs}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid num_obs value '${obs_seq_num_obs}' in ${OBS_SEQ_OUT_FILE}" >&2
        exit 1
    fi

    if [[ "${obs_seq_num_obs}" -ne "${expected_rows}" ]]; then
        echo "ERROR: obs_seq contains ${obs_seq_num_obs} observations, expected ${expected_rows}." >&2
        exit 1
    fi

    echo "obs_seq validation passed:"
    echo "  num_obs: ${obs_seq_num_obs}"
    echo "  expected: ${expected_rows}"
}

#==============================================================================
# stage 1: truth profiles + clear-sky mask (+ truth Hx)
#==============================================================================
if [[ "${RUN_TRUTH}" == "1" ]]; then
    log_stage "Running truth driver: ${TRUTH_DRIVER}"
    bash "${TRUTH_DRIVER}" > "${LOG_DIR}/truth_driver.log" 2>&1

    if [[ ! -f "${CLEAR_SKY_MASK_FILE}" ]]; then
        echo "ERROR: clear-sky mask missing after truth driver: ${CLEAR_SKY_MASK_FILE}" >&2
        exit 1
    fi
    n_mask_rows=$(wc -l < "${CLEAR_SKY_MASK_FILE}")
    if [[ "${n_mask_rows}" -ne "${NOBS}" ]]; then
        echo "ERROR: clear-sky mask has ${n_mask_rows} rows, expected ${NOBS}: ${CLEAR_SKY_MASK_FILE}" >&2
        exit 1
    fi
    n_mask_one=$(awk 'NF == 1 && $1 == 1 { c++ } END { print c + 0 }' "${CLEAR_SKY_MASK_FILE}")
    n_mask_zero=$(awk 'NF == 1 && $1 == 0 { c++ } END { print c + 0 }' "${CLEAR_SKY_MASK_FILE}")
    echo "Clear-sky mask: ${CLEAR_SKY_MASK_FILE}"
    echo "  rows total=${n_mask_rows}  mask=1: ${n_mask_one}  mask=0: ${n_mask_zero}"
else
    log_stage "Skipping truth driver (RUN_TRUTH=${RUN_TRUTH})"
fi

#==============================================================================
# stage 2: ensemble Hx
#==============================================================================
if [[ "${RUN_ENSEMBLE}" == "1" ]]; then
    log_stage "Running ensemble driver: ${ENSEMBLE_DRIVER}"
    if [[ "${EXPERIMENT_MODE}" == "LACC" ]]; then
        export lacc_skip_rttov=0
    fi
    bash "${ENSEMBLE_DRIVER}" > "${LOG_DIR}/ensemble_driver.log" 2>&1
else
    log_stage "Skipping ensemble driver (RUN_ENSEMBLE=${RUN_ENSEMBLE})"
fi

#==============================================================================
# stage 3: observation text conversion
#==============================================================================
if [[ "${RUN_OBS_CONVERT}" == "1" ]]; then
    log_stage "Running observation conversion: ${OBS_CONVERT_SCRIPT}"
    "${PYTHON_BIN}" "${OBS_CONVERT_SCRIPT}" > "${LOG_DIR}/obs_convert.log" 2>&1

    if [[ ! -s "${OBS_DART_INPUT_FILE}" ]]; then
        echo "ERROR: OBS_DART_INPUT_FILE missing or empty: ${OBS_DART_INPUT_FILE}" >&2
        exit 1
    fi
else
    log_stage "Skipping observation conversion (RUN_OBS_CONVERT=${RUN_OBS_CONVERT})"
fi

#==============================================================================
# stage 4: merge ensemble Hx into one file
#==============================================================================
if [[ "${RUN_HX_MERGE}" == "1" ]]; then
    log_stage "Running Hx merge: ${CONVERT_DIR}/merge_FO_in_one_file.py"
    "${PYTHON_BIN}" "${CONVERT_DIR}/merge_FO_in_one_file.py" > "${LOG_DIR}/hx_merge.log" 2>&1

    if [[ ! -s "${MERGED_FO_FILE}" ]]; then
        echo "ERROR: MERGED_FO_FILE missing or empty: ${MERGED_FO_FILE}" >&2
        exit 1
    fi
else
    log_stage "Skipping Hx merge (RUN_HX_MERGE=${RUN_HX_MERGE})"
fi

#==============================================================================
# stage 5: intermediate validation (before text_to_obs)
#==============================================================================
if [[ "${RUN_VALIDATION}" == "1" ]]; then
    log_stage "Validating mask, filtered observations, and merged Hx"
    validate_intermediate_files
else
    log_stage "Skipping intermediate validation (RUN_VALIDATION=${RUN_VALIDATION})"
fi

#==============================================================================
# stage 6: text_to_obs
#==============================================================================
if [[ "${RUN_TEXT_TO_OBS}" == "1" ]]; then

    #--------------------------------------------------------------
    # minimal pre-call checks (kept even when RUN_VALIDATION=0)
    #--------------------------------------------------------------
    if [[ ! -s "${OBS_DART_INPUT_FILE}" ]]; then
        echo "ERROR: observation input file missing or empty: ${OBS_DART_INPUT_FILE}" >&2
        exit 1
    fi

    if [[ ! -s "${MERGED_FO_FILE}" ]]; then
        echo "ERROR: merged Hx file missing or empty: ${MERGED_FO_FILE}" >&2
        exit 1
    fi

    #--------------------------------------------------------------
    # RTTOV sensor database symlink (required by the production
    # input.nml's rttov_sensor_db_file = 'rttov_sensor_db.csv')
    #--------------------------------------------------------------
    if [[ ! -e "${RTTOV_SENSOR_DB_SOURCE}" ]]; then
        echo "ERROR: RTTOV sensor database does not exist: ${RTTOV_SENSOR_DB_SOURCE}" >&2
        exit 1
    fi

    ln -sfn "${RTTOV_SENSOR_DB_SOURCE}" "${RTTOV_SENSOR_DB_LINK}"

    if [[ ! -e "${RTTOV_SENSOR_DB_LINK}" ]]; then
        echo "ERROR: failed to create RTTOV sensor database symlink: ${RTTOV_SENSOR_DB_LINK}" >&2
        exit 1
    fi

    echo "RTTOV sensor database link:"
    echo "  ${RTTOV_SENSOR_DB_LINK} -> ${RTTOV_SENSOR_DB_SOURCE}"

    #--------------------------------------------------------------
    # overwrite handling + build test input.nml from the production one
    #--------------------------------------------------------------
    if [[ "${OVERWRITE_OBS_SEQ}" == "1" && -e "${OBS_SEQ_OUT_FILE}" ]]; then
        rm -f "${OBS_SEQ_OUT_FILE}"
    fi

    log_stage "Preparing ${TEST_INPUT_NML} and calling text_to_obs"
    rm -f "${TEST_INPUT_NML}"
    cp "${TEXT_TO_OBS_RUN_DIR}/input.nml" "${TEST_INPUT_NML}"

    awk -v ti="${OBS_DART_INPUT_FILE}" \
        -v fo="${MERGED_FO_FILE}" \
        -v oo="${OBS_SEQ_OUT_FILE}" '
        /^[[:space:]]*text_input_file[[:space:]]*=/ { print "    text_input_file = '\''" ti "'\'',"; next }
        /^[[:space:]]*FO_input_file[[:space:]]*=/   { print "    FO_input_file   = '\''" fo "'\'',"; next }
        /^[[:space:]]*obs_out_file[[:space:]]*=/    { print "    obs_out_file    = '\''" oo "'\'',"; next }
        { print }
    ' "${TEXT_TO_OBS_RUN_DIR}/input.nml" > "${TEST_INPUT_NML}.tmp"
    mv "${TEST_INPUT_NML}.tmp" "${TEST_INPUT_NML}"

    # text_to_obs reads input.nml from its working directory
    (
        cd "${SINGLE_TEST_WORK_DIR}"
        "${TEXT_TO_OBS_RUN_DIR}/text_to_obs" > "${LOG_DIR}/text_to_obs.log" 2>&1
    )

    if [[ "${RUN_VALIDATION}" == "1" ]]; then
        validate_obs_seq_file
    else
        if [[ ! -f "${OBS_SEQ_OUT_FILE}" ]]; then
            echo "ERROR: OBS_SEQ_OUT_FILE missing after text_to_obs: ${OBS_SEQ_OUT_FILE}" >&2
            exit 1
        fi
        if [[ ! -s "${OBS_SEQ_OUT_FILE}" ]]; then
            echo "ERROR: OBS_SEQ_OUT_FILE is empty: ${OBS_SEQ_OUT_FILE}" >&2
            exit 1
        fi
    fi
else
    log_stage "Skipping text_to_obs (RUN_TEXT_TO_OBS=${RUN_TEXT_TO_OBS})"
fi

#==============================================================================
# final summary
#==============================================================================
LACC_FINAL_ERR_STD=""
if [[ "${EXPERIMENT_MODE}" == "LACC" ]]; then
    LACC_FINAL_ERR_STD=$(awk -v s="${OBS_ERR_STD}" -v n="${EXPECTED_LACC_COUNT}" \
        'BEGIN { printf "%.8f", s/sqrt(n) }')
fi

if [[ "${RUN_VALIDATION}" == "1" ]]; then
    # counts obtained by the validation functions
    orig_shown="${n_mask_rows}"
    retained_shown="${n_mask_one}"
    cloud_shown=$(( n_mask_rows - n_mask_one ))
    obs_rows_shown="${n_obs_rows}"
    merged_rows_shown="${n_merged_rows}"
else
    # best-effort display only (not claimed as validated); files may be missing
    if [[ "${USE_CLEAR_SKY_MASK}" == "1" ]]; then
        if [[ -f "${CLEAR_SKY_MASK_FILE}" ]]; then
            orig_shown=$(wc -l < "${CLEAR_SKY_MASK_FILE}")
            _one=$(awk 'NF == 1 && $1 == 1 { c++ } END { print c + 0 }' "${CLEAR_SKY_MASK_FILE}")
            retained_shown="${_one}"
            cloud_shown=$(( orig_shown - _one ))
        else
            orig_shown="not checked"
            retained_shown="not checked"
            cloud_shown="not checked"
        fi
    else
        # USE_CLEAR_SKY_MASK=0: no clear-sky filtering was applied
        orig_shown="${NOBS}"
        retained_shown="${NOBS}"
        cloud_shown="0"
    fi

    if [[ -f "${OBS_DART_INPUT_FILE}" ]]; then
        obs_rows_shown=$(wc -l < "${OBS_DART_INPUT_FILE}")
    else
        obs_rows_shown="not checked"
    fi

    if [[ -f "${MERGED_FO_FILE}" ]]; then
        merged_rows_shown=$(wc -l < "${MERGED_FO_FILE}")
        merged_shape_shown="${merged_rows_shown} x ${ENS_SIZE}"
    else
        merged_shape_shown="not checked"
    fi
fi

if [[ "${RUN_VALIDATION}" == "1" ]]; then
    merged_shape_shown="${n_merged_rows} x ${ENS_SIZE}"
fi

if [[ "${USE_CLEAR_SKY_MASK}" == "1" ]]; then
    mask_applied_shown="yes"
else
    mask_applied_shown="no"
fi

echo ""
echo "======================================================================"
echo " single assimilation observation test summary"
echo "======================================================================"
echo "Experiment mode:             ${EXPERIMENT_MODE}"
echo "Original grid points:        ${orig_shown}"
echo "Clear-sky retained:          ${retained_shown}"
echo "Cloud-affected:              ${cloud_shown}"
echo "Clear-sky mask applied:      ${mask_applied_shown}"
if [[ "${EXPERIMENT_MODE}" == "LACC" ]]; then
    echo "LACC time count:             ${EXPECTED_LACC_COUNT}"
    echo "Single-time obs error std:   ${OBS_ERR_STD} K"
    echo "Final LACC obs error std:    ${LACC_FINAL_ERR_STD} K"
else
    echo "Single-time obs error std:   ${OBS_ERR_STD} K"
fi
echo "Ensemble member count:       ${ENS_SIZE}"
echo "Observation input row count: ${obs_rows_shown}"
echo "Merged Hx row x col count:   ${merged_shape_shown}"
echo "Final obs_seq.out path:      ${OBS_SEQ_OUT_FILE}"
echo "======================================================================"