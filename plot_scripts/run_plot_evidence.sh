#!/bin/sh
# =============================================================================
# run_plot_evidence.sh — 提交 OM_TMP 证据图(图 1/2/3)计算的 LSF 作业脚本
#
# 提交方式(在服务器上):
#     cd /share/home/lililei1/kcfu/tc_mangkhut/plot_scripts
#     bsub < run_plot_evidence.sh
#
# 作业内容:python -B plot_omtmp_pathway_evidence.py(默认 EVIDENCE_SOURCE=wrf,
# RAW_EVIDENCE_MODE=real)——从原始 wrfout 依次计算 verify_01/02/03 三个诊断
# 并汇总出图。真实模式完整计算约 2.5 小时(诊断三离线通量重建占绝大部分)。
#
# 断点续算:每个诊断算完立即在 omtmp_raw_evidence_cache/ 写阶段检查点;
# 作业若再次中断(节点故障/超时/被清),直接重新 bsub 即可,已完成的
# 阶段自动复用,只从缺失的阶段继续。全部完成后重跑本作业为秒级缓存命中,
# 仅重绘图件。
#
# 注意:
# 1. 登录节点不要直接 nohup 跑本脚本(登录节点会清理长占 CPU 的进程);
#    请一律通过 bsub 提交到 serial 队列。
# 2. 修改过 verify_diag 任何配置或源码后,缓存指纹自动失效,作业会重算
#    全部阶段,属预期行为。
# 3. 诊断配置的真实模式确认开关已在各脚本配置区打开;本作业脚本不改写配置。
# =============================================================================

#BSUB -J plot_evidence
#BSUB -q serial
#BSUB -n 1
#BSUB -oo plot_evidence.out
#BSUB -eo plot_evidence.err
## 如集群要求限时,取消下一行注释(按需调整):
##BSUB -W 06:00

# =====================
# 可编辑配置区
# =====================
PYTHON=/share/home/lililei1/kcfu/anaconda/envs/wrf/bin/python
SCRIPT_DIR=/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts
#: 传给绘图脚本的环境变量(留空 = 用脚本内置默认 wrf/real/auto)
EXTRA_ENV=""

# =====================
# 以下一般无需修改
# =====================

# 计算节点上 matplotlib 缓存目录(避免 HOME 配额/权限问题)
MPLCONFIGDIR=/tmp/matplotlib
export MPLCONFIGDIR

LOG_DIR=${SCRIPT_DIR}/logs
JOB_TAG=${LSB_JOBID:-manual}
mkdir -p "${LOG_DIR}" || exit 1

cd "${SCRIPT_DIR}" || { echo "ERROR: cannot cd to ${SCRIPT_DIR}"; exit 1; }

echo "==============================================================="
echo "[plot_evidence] job ${JOB_TAG} start: $(date)"
echo "[plot_evidence] python : ${PYTHON}"
echo "[plot_evidence] workdir: ${SCRIPT_DIR}"
echo "[plot_evidence] source : ${EXTRA_ENV}"
echo "==============================================================="

LOG="${LOG_DIR}/plot_omtmp_pathway_evidence_${JOB_TAG}.log"

# 提交前在作业日志里确认绘图脚本当前的数据源配置
grep -m2 'EVIDENCE_SOURCE\|RAW_EVIDENCE_MODE' plot_omtmp_pathway_evidence.py \
    | grep '^EVIDENCE_SOURCE\|^RAW_EVIDENCE_MODE' | sed 's/^/    /'

echo "[plot_evidence] running plot_omtmp_pathway_evidence.py"
echo "[plot_evidence] log: ${LOG}"

if env ${EXTRA_ENV} "${PYTHON}" -B plot_omtmp_pathway_evidence.py > "${LOG}" 2>&1; then
    echo "[plot_evidence] DONE: $(date)"
    echo "[plot_evidence] figures:"
    ls -l figs/verify_diag_updated_evidence/ 2>/dev/null | tail -n +2
    echo "[plot_evidence] python log tail:"
    tail -5 "${LOG}"
    exit 0
fi

echo "[plot_evidence] FAILED: $(date) (详见 ${LOG},末尾如下)"
tail -20 "${LOG}"
exit 1
