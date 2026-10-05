#!/bin/sh
# =============================================================================
# run_verify_diag.sh — 一次性提交 verify_diag 诊断的 LSF 作业脚本
#
# 提交方式(在服务器上):
#     cd /share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/verify_diag
#     bsub < run_verify_diag.sh
#
# 选择要运行的诊断:修改下方 RUN_VERIFY01/02/03/04(true=运行,false=跳过)。
# 各诊断按 01 -> 02 -> 03 -> 04 顺序在同一作业内串行执行;每个诊断独立记录
# 成功/失败,某个诊断失败不会中断其余诊断,作业结尾输出汇总并以非零
# 状态退出(若有失败)。
#
# 注意:
# 1. 各入口脚本实际使用合成模式还是真实模式,由脚本自身的 CONFIG
#    决定(mode="synthetic"/"real");本作业脚本不改写配置,只在日志中
#    打印每个脚本当前的 mode,提交前请据此确认。
# 2. verify_04(初值传递核验)的真实模式需要先在其配置区确认
#    STAGE_SOURCE_DEFAULTS 的 B/A/I 阶段路径(默认未配置,会按
#    source_not_configured 跳过相应比较);合成模式可直接运行。
# =============================================================================

#BSUB -J verify_diag
#BSUB -q serial
#BSUB -n 1
#BSUB -oo verify_diag.out
#BSUB -eo verify_diag.err
## 如集群要求限时,取消下一行注释(按需调整):
##BSUB -W 06:00

# =====================
# 可编辑配置区
# =====================

# --- 选择要运行的诊断(true=运行,false=跳过) ---
RUN_VERIFY01=true
RUN_VERIFY02=true
RUN_VERIFY03=true
RUN_VERIFY04=true

# --- 运行环境 ---
PYTHON=/share/home/lililei1/kcfu/anaconda/envs/wrf/bin/python
VERIFY_DIR=/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/verify_diag

# =====================
# 以下一般无需修改
# =====================

# 计算节点上 matplotlib 缓存目录(避免 HOME 配额/权限问题)
MPLCONFIGDIR=/tmp/matplotlib
export MPLCONFIGDIR

LOG_DIR=${VERIFY_DIR}/logs
JOB_TAG=${LSB_JOBID:-manual}
mkdir -p "${LOG_DIR}" || exit 1

cd "${VERIFY_DIR}" || { echo "ERROR: cannot cd to ${VERIFY_DIR}"; exit 1; }

echo "==============================================================="
echo "[verify_diag] job ${JOB_TAG} start: $(date)"
echo "[verify_diag] python : ${PYTHON}"
echo "[verify_diag] workdir: ${VERIFY_DIR}"
echo "==============================================================="

FAILED=""

run_one() {
    # $1 = 脚本名
    script="$1"
    log="${LOG_DIR}/${script%.py}_${JOB_TAG}.log"
    echo ""
    echo "---------------------------------------------------------------"
    echo "[verify_diag] running ${script}"
    # 打印脚本当前配置的运行模式,供提交前确认(synthetic/real)
    grep -m1 'mode="' "${script}" | sed 's/^[[:space:]]*//;s/^mode=/    mode = /'
    echo "[verify_diag] log: ${log}"
    if "${PYTHON}" -B "${script}" > "${log}" 2>&1; then
        echo "[verify_diag] DONE  : ${script}"
        tail -3 "${log}"
    else
        echo "[verify_diag] FAILED: ${script} (详见 ${log},末尾如下)"
        tail -10 "${log}"
        FAILED="${FAILED} ${script}"
    fi
}

if [ "${RUN_VERIFY01}" = "true" ]; then
    run_one verify_01_skill_timeseries.py
else
    echo "[verify_diag] skip  : verify_01_skill_timeseries.py (RUN_VERIFY01=false)"
fi

if [ "${RUN_VERIFY02}" = "true" ]; then
    run_one verify_02_flux_error_budget.py
else
    echo "[verify_diag] skip  : verify_02_flux_error_budget.py (RUN_VERIFY02=false)"
fi

if [ "${RUN_VERIFY03}" = "true" ]; then
    run_one verify_03_fixed_atmosphere_flux.py
else
    echo "[verify_diag] skip  : verify_03_fixed_atmosphere_flux.py (RUN_VERIFY03=false)"
fi

if [ "${RUN_VERIFY04}" = "true" ]; then
    run_one verify_04_initial_handoff.py
else
    echo "[verify_diag] skip  : verify_04_initial_handoff.py (RUN_VERIFY04=false)"
fi

echo ""
echo "==============================================================="
if [ -z "${FAILED}" ]; then
    echo "[verify_diag] all requested diagnostics finished: $(date)"
    echo "==============================================================="
    exit 0
fi
echo "[verify_diag] finished with failures:${FAILED} ($(date))"
echo "==============================================================="
exit 1
