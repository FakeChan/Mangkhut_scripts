#!/bin/bash
#BSUB -J cycle_50mem_hourly
#BSUB -q serial
#BSUB -n 1
#BSUB -oo cycle_50mem_hourly.out
#BSUB -eo cycle_50mem_hourly.err
#
# ============================================================================
# run_driver_50mem_cyclingDA.sh（Python 编排）控制流程的纯 bash 直译版
# ============================================================================
# 目的：把"谁在什么时候调用谁"摊平成 run_driver_cyclingDA.sh 那种顺序脚本，
#       便于逐行核对。函数顺序、目录布局、循环次序、门控条件与 Python 版一致。
#
# 整个流程（自上而下读）：
#
#   main
#    ├─ DRY_RUN   -> plan()                 只打印计划
#    ├─ PREFLIGHT -> preflight()            只读检查，不建目录、不投作业
#    └─ RUN       -> run()
#                    ├─ preflight()          先检查，再动手
#                    ├─ flock RUN_ROOT/.driver.lock
#                    └─ 对每个 group(1,2,3) × filter(EAKF,QCF_RHF)：
#                         run_experiment()
#                         ├─ init_experiment()      写 experiment 指纹 + 50×2 初始场软链成 prior/10_00_00
#                         └─ 对 4 个 cycle(00,01,02,03)，预报时长 1,1,1,6 h：
#                              produce_observations()  造真值/H(x)/掩膜/obs_seq（STANDARD + LACC 两套）
#                              assimilate()            group3 先 inflateOcean，再 air/all；ncks 拼回 prior
#                              forecast()              50 成员 LSF 数组作业 -> 校验 -> 归档 -> publish 下一时次 prior
#
# 与 Python 版的取舍（为可读性刻意简化，已在对应函数上注明）：
#   * netCDF 校验：Python 用 netCDF4 读 Times/变量齐全性/NaN；这里改用 ncdump 抓时间字符串（+可选变量名检查）。
#   * checkpoint：Python 用 done.json（signature + 每个文件的 size/mtime/sha256）；这里用同名 marker + stat 行。
#   * namelist 改写：Python 按 &section 作用域精确替换并要求"恰好一条"；这里退化为 sed 行匹配。
#   * experiment 指纹：Python 逐文件 sha256；这里 input 用 size+mtime，code 用 sha256sum。
#   * 其余（循环、目录、门控、LSF 数组、等待/超时/中止、resume 判定）与 Python 一一对应。
#
# 依赖：bash>=4（mapfile/进程替换）、GNU date/stat、NCO(ncks/ncrcat/ncatted/ncdump)、LSF、sha256sum。
# 本文件仅供阅读核对，未在集群上实跑验证。
# ============================================================================
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
BC_PATTERN_DEFAULT='mem{mem:03d}/BC/wrfbdy_d01_mem{mem:03d}'   # Python format 串，bash 里做字符串替换
IC_PATTERN_DEFAULT='mem{mem:03d}/IC/wrfinput_2018091000_mem{mem:03d}'
export BC_PATTERN="${BC_PATTERN:-${BC_PATTERN_DEFAULT}}"
export IC_PATTERN="${IC_PATTERN:-${IC_PATTERN_DEFAULT}}"
export NR_ROOT="${NR_ROOT:-${PROJECT_ROOT}/NR_wrfout/2domain}"
export DART_TEMPLATE_DIR="${DART_TEMPLATE_DIR:-${PROJECT_ROOT}/4assimilation/2DART/run_dir}"
export DART_WORK="${DART_WORK:-/share/home/lililei1/kcfu/models/DART_main/models/wrf/work}"
export WRF_EM_REAL="${WRF_EM_REAL:-/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/WRF-4.1/test/em_real}"
export RTTOV_ROOT="${RTTOV_ROOT:-/share/home/lililei1/kcfu/models/rttov123}"
export PYTHON_EXE="${PYTHON_EXE:-/share/home/lililei1/kcfu/anaconda/envs/wrf/bin/python}"

export RUN_GROUP_1="${RUN_GROUP_1:-1}"              # 无海洋更新、无海洋模式
export RUN_GROUP_2="${RUN_GROUP_2:-1}"              # 跑海洋模式、不更新
export RUN_GROUP_3="${RUN_GROUP_3:-1}"              # 海洋更新 + 海洋模式
export RUN_EAKF="${RUN_EAKF:-1}"
export RUN_QCF_RHF="${RUN_QCF_RHF:-1}"
export HX_STATE_DOMAIN="${HX_STATE_DOMAIN:-d02}"    # 物理网格；观测标签仍写 d01
export ASSIM_CHANNEL="${ASSIM_CHANNEL:-4}"
export OBS_ERR_STD="${OBS_ERR_STD:-0.5}"             # 单时刻 K；LACC 用 /sqrt(3)
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
# ===========================================================================

# ======================= 常量与时间轴（Python: START/MEMBERS/cycles()）=====
START_ISO="2018-09-10 00:00:00"

# 时间工具：加减一律走 epoch。不要用 date -d "<完整时刻> +N hours" —— GNU date 会把
# 结尾的 +N 当成时区偏移（实测 "2018-09-10 00:00:00 +6 hours" 得到 09-09 19:00）。
to_epoch()       { date -u -d "$1" '+%s'; }
from_epoch()     { date -u -d "@$1" '+%Y-%m-%d %H:%M:%S'; }
stamp_of()       { date -u -d "$1" '+%Y-%m-%d_%H:%M:%S'; }   # 2018-09-10_00:00:00
tag_of()         { date -u -d "$1" '+%d_%H_%M'; }            # 10_00_00
tag_from_stamp() { local s=$1; printf '%s_%s_%s' "${s:8:2}" "${s:11:2}" "${s:14:2}"; }
shift_hours()    { local ts; ts=$(to_epoch "$1"); from_epoch $((ts + $2 * 3600)); }
shift_minutes()  { local ts; ts=$(to_epoch "$1"); from_epoch $((ts + $2 * 60)); }

START_STAMP=$(stamp_of "$START_ISO")
DOMAINS=(d01 d02)
ATM_VARS="MU,P,PH,QVAPOR,THM,U,V,W"
OCEAN_VARS="OM_TMP,OM_S,OM_U,OM_V"
STATE_EXTRA="TSK,XLAT,XLONG"
HX_VARS="T,P,PB,PH,PHB,TSK,T2,Q2,U10,V10,QVAPOR,QCLOUD,QICE,PSFC,HGT,LANDMASK,CLDFRA,XLAT,XLONG"

# 4 个分析时次：00/01/02/03 UTC；前三个预报 1 h，最后一个 6 h（结束于 09 UTC）
CYCLE_COUNT=4
CYCLE_HOURS=(1 1 1 6)
CYCLE_LAGS=("0 3 6" "0 4 7" "0 5 8" "0 3 6")
CYCLE_ISOS=()
for ((i = 0; i < CYCLE_COUNT; i++)); do
  CYCLE_ISOS+=("$(shift_hours "$START_ISO" "$i")")
done
FINAL_END_STAMP=$(stamp_of "$(shift_hours "${CYCLE_ISOS[CYCLE_COUNT-1]}" "${CYCLE_HOURS[CYCLE_COUNT-1]}")")

# ============================== 小工具 =====================================
log()  { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require_file() { [[ -s $1 ]] || die "missing or empty input: $1"; }

# Python: read_nc_times —— 取 NetCDF 里的时间字符串（简化：只用 ncdump 抓）
nc_times() {
  ncdump -v "${2:-Times}" -p 9,18 "$1" 2>/dev/null \
    | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}:[0-9]{2}:[0-9]{2}' || true
}

# Python: validate_state —— 有效时刻 + 必需变量（简化：变量名用 ncdump -h 粗查）
check_vars() {
  local file=$1; shift
  local name header
  header=$(ncdump -h "$file" 2>/dev/null) || die "cannot read state $file"
  for name in "$@"; do
    grep -qE "(^|[[:space:]])${name}\\(" <<<"$header" || die "$file: missing variable $name"
  done
}

check_state() {
  local file=$1 want=$2; shift 2
  require_file "$file"
  local got
  got=$(nc_times "$file" | head -1)
  [[ $got == "$want" ]] || die "$file: valid time ${got:-<none>} must be $want"
  check_vars "$file" "$@"
}

# Python: validate_boundary —— 记录连续、且覆盖到 required_end。
# 时间串定宽零填充，所以字符串比较等价于时间比较。
check_boundary() {
  local file=$1 required_end=$2
  require_file "$file"
  local -a begins ends times
  mapfile -t begins < <(nc_times "$file" md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_)
  mapfile -t ends   < <(nc_times "$file" md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_)
  mapfile -t times  < <(nc_times "$file")
  ((${#begins[@]})) || die "$file: no boundary records"
  (( ${#begins[@]} == ${#ends[@]} )) || die "$file: boundary Times/thisbdytime/nextbdytime disagree"
  ((${#times[@]} == ${#begins[@]})) || die "$file: boundary Times/thisbdytime disagree"
  local i b e cursor=$START_STAMP
  for ((i = 0; i < ${#begins[@]}; i++)); do
    b=${begins[i]} e=${ends[i]}
    [[ $e > $b ]] || die "$file: nonpositive boundary interval"
    [[ $e > $cursor ]] || continue
    [[ $b > $cursor ]] && die "$file: boundary gap at $cursor -> $b"
    cursor=$e
    [[ $cursor > $required_end || $cursor == $required_end ]] && return 0
  done
  die "$file: boundary effective end $cursor < required $required_end; 先补齐成员边界，脚本不外推"
}

# Python: link_readonly
link_readonly() {
  local source=$1 target=$2
  mkdir -p "$(dirname "$target")"
  if [[ -L $target ]]; then
    [[ $(readlink -f "$target") == "$(readlink -f "$source")" ]] && return 0
    rm -f "$target"
  elif [[ -e $target ]]; then
    die "refusing to replace a regular file with an input link: $target"
  fi
  ln -s "$(readlink -f "$source")" "$target"
}

# ---------------------------------------------------------------- checkpoint
# Python: save_checkpoint / checkpoint_ok。marker 第 1 行是指纹，后面每行 "size mtime path"。
write_done() {
  local marker=$1 sig=$2; shift 2
  local f
  {
    printf '%s\n' "$sig"
    for f in "$@"; do
      [[ -f $f ]] && printf '%s %s %s\n' "$(stat -c '%s' "$f")" "$(stat -c '%Y' "$f")" "$f"
    done
  } > "$marker.tmp"
  mv "$marker.tmp" "$marker"
}

have_done() {
  local marker=$1 sig=$2 resume=$3; shift 3
  [[ -e $marker ]] || return 1
  [[ $resume -eq 1 ]] || die "stage already exists: $marker; use RESUME=1"
  [[ $(head -1 "$marker") == "$sig" ]] || die "checkpoint configuration changed: $marker"
  local size mtime path
  while read -r size mtime path; do
    [[ -f $path ]] || die "checkpoint output changed or missing: $path"
    [[ "$(stat -c '%s %Y' "$path")" == "$size $mtime" ]] || die "checkpoint output changed: $path"
  done < <(tail -n +2 "$marker")
  return 0
}

# Python: file_identity / experiment_signature（input 用 size+mtime，code 额外 sha256）
fingerprint_paths() { local p; for p in "$@"; do [[ -f $p ]] && printf 'IN %s %s\n' "$(stat -c '%s %Y' "$p")" "$p"; done; }
fingerprint_code()  { local p; for p in "$@"; do [[ -f $p ]] && printf 'CODE %s %s\n' "$(sha256sum "$p" | cut -d' ' -f1)" "$p"; done; }

# ============================== LSF 封装 ===================================
# Python: class LSF
# 提交；stdout 只输出 job id，便于 jobid=$(lsf_submit ...)
lsf_submit() {
  local work=$1 name=$2 tasks=$3 queue=$4; shift 4
  local record=$work/lsf_job.id
  if [[ -f $record ]]; then
    grep -q '^SUBMITTING$' "$record" \
      && die "submission interrupted without a confirmed job ID: $record; 先用 LSF 命令核对再重试"
    cat "$record"; return 0                      # 复用作已提交的 id（resume 路径）
  fi
  mkdir -p "$work"
  local label=$name
  if (($#)); then
    local joined="" m
    for m in "$@"; do joined+="${joined:+,}$m"; done
    label="$name[$joined]"
  fi
  printf 'SUBMITTING\n' > "$record"
  local out jobid
  out=$(bsub -J "$label" -n "$tasks" -q "$queue" -R "span[ptile=$TASKS_PER_NODE]" \
             -oo "$work/job_%I.out" -eo "$work/job_%I.err" < "$work/job.sh" 2>&1) || true
  printf '%s\n' "$out" > "$work/submission.log"
  jobid=$(printf '%s\n' "$out" | grep -o 'Job <[0-9]*>' | grep -o '[0-9]*' | head -1 || true)
  [[ -n $jobid ]] || die "unconfirmed LSF submission; 先核对 $record 再重试: $out"
  printf '%s\n' "$jobid" > "$record"
  log "LSF submitted $label: job $jobid" >&2
  printf '%s\n' "$jobid"
}

# 逐成员/逐作业查状态：DONE / EXIT / PEND / RUN / ... ; 查不到则打 NONE
lsf_states() {
  local job=$1; shift
  local -a expected=()
  if (($#)); then local m; for m in "$@"; do expected+=("$job[$m]"); done
  else expected=("$job"); fi
  local listing id st out=()
  listing=$(bjobs -a -noheader -o 'jobid stat' "$job" 2>/dev/null || true)
  for id in "${expected[@]}"; do
    st=$(awk -v want="$id" '$1 == want {print $2}' <<<"$listing")
    if [[ -z $st ]]; then                      # 已不在 bjobs：查 bhist 的最终事件
      if bhist -l "$id" 2>/dev/null | grep -q 'Done successfully'; then st=DONE
      elif bhist -l "$id" 2>/dev/null | grep -qE '\bExited\b'; then st=EXIT
      fi
    fi
    out+=("${st:-NONE}")
  done
  printf '%s\n' "${out[@]}"
}

lsf_cancel() {
  [[ $1 =~ ^[0-9]+$ ]] || die "invalid numeric LSF job ID: $1"
  bkill -r "$1" >/dev/null 2>&1 || true
}

# 等待终止；allow=0 时任何 EXIT 即失败；allow=1 时返回真实成败（1=有失败）
lsf_wait() {
  local job=$1 allow=$2; shift 2
  local started=$SECONDS unknown=0
  local -a states=()
  while :; do
    mapfile -t states < <(lsf_states "$job" "$@")
    local st pending=0 failed=0 unknown_now=0
    for st in "${states[@]}"; do
      case $st in
        DONE) ;;
        EXIT) failed=1 ;;
        PEND|RUN|PSUSP|USUSP|SSUSP|WAIT) pending=1 ;;
        *) unknown_now=1 ;;
      esac
    done
    if [[ $pending -eq 0 && $unknown_now -eq 0 ]]; then
      if [[ $failed -eq 1 ]]; then
        [[ $allow -eq 1 ]] && return 1
        die "LSF job $job reached EXIT"
      fi
      return 0
    fi
    if [[ $unknown_now -eq 1 ]]; then
      unknown=$((unknown + 1))
      [[ $unknown -ge 5 ]] && die "cannot determine LSF job $job state; refusing unknown success"
    else
      unknown=0
    fi
    if (( SECONDS - started >= MAX_WAIT_SECONDS )); then
      lsf_cancel "$job"                          # 只取消本 id，并证明它确实停了再走
      local n
      for n in {1..12}; do
        mapfile -t states < <(lsf_states "$job" "$@")
        local all_terminal=1
        for st in "${states[@]}"; do
          [[ $st == DONE || $st == EXIT ]] || all_terminal=0
        done
        if [[ $all_terminal -eq 1 ]]; then
          [[ $allow -eq 1 ]] && return 1
          die "LSF job $job timed out and was stopped"
        fi
        sleep 5
      done
      die "LSF job $job timed out; cannot prove termination; keep work intact"
    fi
    sleep "$POLL_SECONDS"
  done
}

job_name() {  # Python: job_name(root, cycle, stage)
  local token
  token=$(printf '%s' "$ROOT" | sha256sum | cut -c1-10)
  printf 'c50_%s_%s_%s' "$token" "$CTAG" "$1"
}

# ============================== 路径构造 ===================================
# Python: cfg.bc()/cfg.ic() —— 把 BC_PATTERN/IC_PATTERN 里的 {mem:03d} 换成成员号
member_path() { local pat=$1 m=$2; pat=${pat//\{mem:03d\}/$(printf '%03d' "$m")}; printf '%s' "$pat"; }
bc_path() { printf '%s/%s' "$BC_ROOT" "$(member_path "$BC_PATTERN" "$1")"; }
ic_path() { printf '%s/%s' "$IC_ROOT" "$(member_path "$IC_PATTERN" "$1")"; }
prior_path() {  # <stamp> <domain> <member>
  printf '%s/prior/%s/firstguess_%s.mem%03d' "$ROOT" "$(tag_from_stamp "$1")" "$2" "$3"
}
# Python: input_source —— 先验从哪来：历史自由预报 / 初始场 / 本实验上一时次 prior
input_source() {
  local moment=$1 dom=$2 m=$3
  if [[ $moment < $START_STAMP ]]; then
    printf '%s/mem%03d/wrfout_%s_%s' "$HISTORY_ROOT" "$m" "$dom" "$moment"
  elif [[ $moment == "$START_STAMP" && $CSTAMP == "$START_STAMP" ]]; then
    printf '%s/firstguess_%s.mem%03d' "$INITIAL_PRIOR_DIR" "$dom" "$m"
  else
    prior_path "$moment" "$dom" "$m"
  fi
}

# Python: wrf_static_entries
wrf_static_entries() {
  find "$WRF_EM_REAL" -maxdepth 1 -mindepth 1 \
       ! -name 'wrfinput*' ! -name 'wrfbdy*' ! -name 'wrfout*' ! -name 'wrfrst*' \
       ! -name 'rsl.*' ! -name 'namelist.*' ! -name 'met_em*' | sort
}

# Python: rttov_files
rttov_files() {
  printf '%s\n' "$RTTOV_ROOT/bin/example_fwd.exe" \
                "$RTTOV_ROOT/rtcoef_rttov12/rttov7pred54L/rtcoef_noaa_18_amsua.dat"
}

# Python: producer_sources —— 需要私有化（会 rewrite 路径/sed 改）的代码文件
producer_sources() {
  local hx=$PROJECT_ROOT/3create_obs/hx_rttov conv=$PROJECT_ROOT/4assimilation/1convert_obs part
  printf '%s\n' "$hx/run_rttov_TrueObs_driver.sh" "$hx/run_rttov_ensBT_driver.sh" \
                "$hx/run_rttov_TrueObs_LACC_driver.sh" "$hx/run_rttov_ensBT_LACC_driver.sh"
  for part in profile/1run_matlab profile/2ens_matlab 2call_rttov \
              3obs_BT/merge 3obs_BT/add_pert 4ens_BT/merge; do
    find "$hx/$part" -type f \( -name '*.sh' -o -name '*.m' -o -name '*.py' \) | sort
  done
  printf '%s\n' "$conv/run_single_assimilation_test.sh" "$conv/obs2DART_test.py" \
                "$conv/obs2DART_LACC.py" "$conv/merge_FO_in_one_file.py" \
                "$conv/filter_LACC_sst_direction.py" \
                "$PROJECT_ROOT/plot_scripts/diag_LACC_lag_cov_innovation.py"
}

# ============================== preflight ==================================
# Python: preflight() —— 只读；失败时一次性列出所有问题，绝不建目录、不投作业
preflight() {
  ISSUES=()
  [[ $RUN_ROOT == "$SCRATCH_ROOT"/* ]] || ISSUES+=("RUN_ROOT must be strictly underneath SCRATCH_ROOT: $RUN_ROOT")
  local p
  for p in "$PROJECT_ROOT" "$RUN_ROOT" "$INITIAL_PRIOR_DIR" "$HISTORY_ROOT" "$NR_ROOT" \
           "$DART_TEMPLATE_DIR" "$DART_WORK" "$WRF_EM_REAL" "$RTTOV_ROOT" "$PYTHON_EXE"; do
    printf '%s' "$p" | grep -qE "[[:space:]'\"\`\$]" \
      && ISSUES+=("producer 不支持含空白/shell 元字符的路径: $p")
  done
  local cmd
  for cmd in bsub bjobs bhist bkill matlab ncks ncrcat ncatted ncdump perl bash sha256sum flock \
             "$(awk '{print $1}' <<<"$DART_MPI")" "$(awk '{print $1}' <<<"$WRF_MPI")"; do
    command -v "$cmd" >/dev/null || ISSUES+=("command is not on PATH: $cmd")
  done
  "$PYTHON_EXE" -c 'import netCDF4, numpy, scipy, matplotlib' 2>/dev/null \
    || ISSUES+=("$PYTHON_EXE: missing Python dependency (netCDF4/numpy/scipy/matplotlib)")

  # 生产脚本、转换器、运行时文件
  local f
  while read -r f; do [[ -s $f ]] || ISSUES+=("missing or empty input: $f"); done < <(producer_sources)
  local run_dir=$PROJECT_ROOT/4assimilation/1convert_obs/run_dir
  for f in "$run_dir/input.nml" "$run_dir/text_to_obs" "$run_dir/rttov_sensor_db.csv" \
           "$DART_WORK/filter" "$DART_WORK/update_wrf_bc" "$WRF_EM_REAL/wrf.exe" \
           "$PROJECT_ROOT/0necessay_files/input.nml" \
           "$PROJECT_ROOT/0necessay_files/namelist.input.ens" \
           "$PROJECT_ROOT/0necessay_files/ad_omini_d01.py" \
           "$PROJECT_ROOT/0necessay_files/ad_omini_d02.py" \
           "$DART_TEMPLATE_DIR/rttov_sensor_db.csv" \
           "$DART_TEMPLATE_DIR/rtcoef_noaa_18_amsua.dat"; do
    [[ -s $f ]] || ISSUES+=("missing or empty input: $f")
  done
  while read -r f; do [[ -s $f ]] || ISSUES+=("missing or empty input: $f"); done < <(rttov_files)
  # DA 模板只检查本次实际会用到的那几个
  [[ " ${SELECTED_GROUPS[*]} " == *" 3 "* ]] && { [[ -s $DART_TEMPLATE_DIR/input.nml.inflateOcean ]] || ISSUES+=("missing template: input.nml.inflateOcean"); }
  [[ " ${SELECTED_GROUPS[*]} " == *" 3 "* ]] && { [[ -s $DART_TEMPLATE_DIR/input.nml.all ]] || ISSUES+=("missing template: input.nml.all"); }
  [[ " ${SELECTED_GROUPS[*]} " == *" 1 "* || " ${SELECTED_GROUPS[*]} " == *" 2 "* ]] && { [[ -s $DART_TEMPLATE_DIR/input.nml.air ]] || ISSUES+=("missing template: input.nml.air"); }
  [[ " ${FILTERS[*]} " == *" QCF_RHF "* ]] && { [[ -s $DART_TEMPLATE_DIR/qceff_table_fkc.csv ]] || ISSUES+=("missing qceff_table_fkc.csv"); }

  # 50 成员 × 2 域 初始场、ICBC、历史 LACC 源、NR 场
  local m dom moment
  local -a state_vars hx_vars
  IFS=, read -r -a state_vars <<<"$ATM_VARS,$OCEAN_VARS,$STATE_EXTRA"
  IFS=, read -r -a hx_vars <<<"$HX_VARS"
  for ((m = 1; m <= 50; m++)); do
    for dom in "${DOMAINS[@]}"; do
      if [[ $dom == "$HX_STATE_DOMAIN" ]]; then
        need_state "$INITIAL_PRIOR_DIR/firstguess_${dom}.mem$(printf '%03d' "$m")" "$START_STAMP" \
                   "${state_vars[@]}" "${hx_vars[@]}"
      else
        need_state "$INITIAL_PRIOR_DIR/firstguess_${dom}.mem$(printf '%03d' "$m")" "$START_STAMP" \
                   "${state_vars[@]}"
      fi
    done
    for moment in 3 6; do
      local hist
      hist=$(stamp_of "$(shift_hours "$START_ISO" "-$moment")")
      need_state "$HISTORY_ROOT/mem$(printf '%03d' "$m")/wrfout_${HX_STATE_DOMAIN}_${hist}" "$hist" "${hx_vars[@]}"
    done
    need_state "$(ic_path "$m")" "$START_STAMP" OM_TINI OM_SINI
    need_boundary "$(bc_path "$m")" "$FINAL_END_STAMP"
  done
  local t
  for t in $(nr_times); do
    need_state "$NR_ROOT/wrfout_d02_${t}" "$t" "${hx_vars[@]}" QRAIN QSNOW QGRAUP OM_TMP
  done

  ((${#ISSUES[@]} == 0)) || die "preflight failed; NO jobs submitted and NO runtime directories created:
  - $(printf '%s\n  - ' "${ISSUES[@]}")"
  log "Preflight passed: 50 members, 2 domains, LACC times and BC coverage through 09"
}

# preflight 内用：把 validate 的失败信息收集起来（不直接退出脚本）
need_state() {
  local msg
  if ! msg=$(check_state "$@" 2>&1); then ISSUES+=("$msg"); fi
}

need_boundary() {
  local msg
  if ! msg=$(check_boundary "$@" 2>&1); then ISSUES+=("$msg"); fi
}

# 所有 cycle 用到的 LACC 时刻（去重排序）
nr_times() {
  local c lag t
  for ((c = 0; c < CYCLE_COUNT; c++)); do
    for lag in ${CYCLE_LAGS[$c]}; do
      t=$(stamp_of "$(shift_hours "${CYCLE_ISOS[$c]}" "-$lag")")
      printf '%s\n' "$t"
    done
  done | sort -u
}

# ============================== 时次上下文 =================================
# Python: Cycle 对象 + 循环变量
set_cycle() {
  local i=$1 lag t
  CYCLE_ISO=${CYCLE_ISOS[$i]}
  CYCLE_HOURS_H=${CYCLE_HOURS[$i]}
  CEND_ISO=$(shift_hours "$CYCLE_ISO" "$CYCLE_HOURS_H")
  CSTAMP=$(stamp_of "$CYCLE_ISO")
  CEND_STAMP=$(stamp_of "$CEND_ISO")
  CTAG=$(tag_of "$CYCLE_ISO")
  LACC_STAMPS=(); LACC_TAGS=()
  for lag in ${CYCLE_LAGS[$i]}; do
    t=$(shift_hours "$CYCLE_ISO" "-$lag")
    LACC_STAMPS+=("$(stamp_of "$t")")
    LACC_TAGS+=("$(tag_of "$t")")
  done
}

# Python: Experiment
set_experiment() {
  GROUP=$1 FILTER=$2
  UPDATE_OCEAN=$([[ $GROUP -eq 3 ]] && echo 1 || echo 0)
  RUN_OCEAN=$([[ $GROUP -ne 1 ]] && echo 1 || echo 0)
  EXP_NAME="oceanAssim${UPDATE_OCEAN}Run${RUN_OCEAN}/${FILTER}"
  ROOT="$RUN_ROOT/$EXP_NAME"
}

# ============================== 实验指纹 ===================================
# Python: experiment_signature —— 大文件用 size+mtime，代码用 sha256
experiment_signature() {
  local m dom h t
  local -a inputs=() code=()
  for ((m = 1; m <= 50; m++)); do
    inputs+=("$(bc_path "$m")" "$(ic_path "$m")")
    for dom in "${DOMAINS[@]}"; do inputs+=("$INITIAL_PRIOR_DIR/firstguess_${dom}.mem$(printf '%03d' "$m")"); done
    for h in 3 6; do
      inputs+=("$HISTORY_ROOT/mem$(printf '%03d' "$m")/wrfout_${HX_STATE_DOMAIN}_$(stamp_of "$(shift_hours "$START_ISO" "-$h")")")
    done
  done
  while read -r t; do inputs+=("$NR_ROOT/wrfout_d02_${t}"); done < <(nr_times)
  code+=("$DART_WORK/filter" "$DART_WORK/update_wrf_bc" "$WRF_EM_REAL/wrf.exe" \
         "$PROJECT_ROOT/4assimilation/1convert_obs/run_dir/text_to_obs" \
         "$PROJECT_ROOT/0necessay_files/input.nml" "$PROJECT_ROOT/0necessay_files/namelist.input.ens" \
         "$PROJECT_ROOT/0necessay_files/ad_omini_d01.py" "$PROJECT_ROOT/0necessay_files/ad_omini_d02.py" \
         "$DART_TEMPLATE_DIR/rttov_sensor_db.csv" "$DART_TEMPLATE_DIR/rtcoef_noaa_18_amsua.dat" \
         "$PROJECT_ROOT/4assimilation/1convert_obs/run_dir/input.nml" \
         "$PROJECT_ROOT/4assimilation/1convert_obs/run_dir/rttov_sensor_db.csv")
  [[ $UPDATE_OCEAN -eq 1 ]] && code+=("$DART_TEMPLATE_DIR/input.nml.inflateOcean" "$DART_TEMPLATE_DIR/input.nml.all") \
                            || code+=("$DART_TEMPLATE_DIR/input.nml.air")
  [[ $FILTER == QCF_RHF ]] && code+=("$DART_TEMPLATE_DIR/qceff_table_fkc.csv")
  while read -r f; do code+=("$f"); done < <(producer_sources)
  while read -r f; do code+=("$f"); done < <(rttov_files)
  while read -r f; do code+=("$f"); done < <(wrf_static_entries)
  {
    printf 'CONFIG %s %s %s %s %s %s %s %s\n' "$GROUP" "$FILTER" "$HX_STATE_DOMAIN" "$ASSIM_CHANNEL" \
      "$OBS_ERR_STD" "$CLEAR_SKY_THRESHOLD" "$HGT_OBS" "$RUN_ROOT"
    fingerprint_paths "${inputs[@]}"
    fingerprint_code "${code[@]}"
  } | sha256sum | cut -d' ' -f1
}

# Python: initialize_experiment
init_experiment() {
  local dom m record=$ROOT/experiment.meta
  SIG=$(experiment_signature)
  if [[ -e $record ]]; then
    [[ $RESUME -eq 1 ]] || die "experiment exists: $ROOT; use RESUME=1"
    [[ $(sed -n 's/^signature=//p' "$record") == "$SIG" ]] \
      || die "experiment inputs/configuration changed: $ROOT; choose a new RUN_ROOT"
  else
    [[ -e $ROOT ]] && [[ -n $(ls -A "$ROOT" 2>/dev/null) ]] && die "unowned nonempty experiment directory: $ROOT"
    mkdir -p "$ROOT"
    printf 'signature=%s\ngroup=%s\nfilter=%s\n' "$SIG" "$GROUP" "$FILTER" > "$record.tmp"
    mv "$record.tmp" "$record"
  fi
  # 初始时刻的 prior：软链到已有集合初始场（不复制大文件）
  for dom in "${DOMAINS[@]}"; do
    for ((m = 1; m <= 50; m++)); do
      link_readonly "$INITIAL_PRIOR_DIR/firstguess_${dom}.mem$(printf '%03d' "$m")" \
                    "$(prior_path "$START_STAMP" "$dom" "$m")"
    done
  done
}

# ============================== 观测生产 ===================================
# Python: prepare_producer —— 只复制代码、不复制大文件；所有被 sed 改的都是私有副本
prepare_producer() {
  local work=$1
  local target=$work/producer
  [[ -e $target ]] && die "producer workspace already exists: $target"
  local src rel
  while read -r src; do
    require_file "$src"
    rel=${src#"$PROJECT_ROOT"/}
    mkdir -p "$target/$(dirname "$rel")"
    sed -e "s#${PROJECT_ROOT}#${target}#g" \
        -e "s#/share/home/lililei1/kcfu/tc_mangkhut#${target}#g" "$src" > "$target/$rel"
    case $src in
      *.sh) sed -i -e "s#^export python_bin=.*#export python_bin=$(dirname "$PYTHON_EXE")#" \
                     -e "s#^export rttov_dir=.*#export rttov_dir=${RTTOV_ROOT}#" "$target/$rel" ;;
    esac
    if [[ $(basename "$src") == wrf_rttov_d02_75km_ens_interp.m ]]; then
      sed -i -e "s#wrfout_d0[12]_2018-09-#wrfout_${HX_STATE_DOMAIN}_2018-09-#g" \
             -e "s#firstguess_d0[12]\.#firstguess_${HX_STATE_DOMAIN}.#g" "$target/$rel"
    fi
    [[ $src == *.sh ]] && chmod 755 "$target/$rel" || chmod 644 "$target/$rel"
  done < <(producer_sources)
  # text_to_obs / input.nml / 传感器库只读软链；NR 目录整体软链
  for rel in text_to_obs input.nml rttov_sensor_db.csv; do
    link_readonly "$PROJECT_ROOT/4assimilation/1convert_obs/run_dir/$rel" \
                  "$target/4assimilation/1convert_obs/run_dir/$rel"
  done
  link_readonly "$NR_ROOT" "$target/NR_wrfout/2domain"
  PRODUCER=$target
}

# Python: observation_environment —— 显式给出，并清掉可能污染的外部同名变量
run_producer() {
  local mode=$1 logfile=$2
  (
    unset OBS_INPUT_FILE OBS_DART_INPUT_FILE MERGED_FO_FILE OBS_BT_DIR ENS_BT_DIR ADAPTIVE_ACTIVE \
          CLEAR_SKY_MASK_FILE PARA_FILE PROFILE_DIR PROFILE_SUBDIR RAW_ENS_BT_DIR ADAPTIVE_LACC_DIR
    export BASE_DIR=$PRODUCER PROJECT_ROOT=$PRODUCER
    export CONVERT_TOP_DIR=$PRODUCER/4assimilation/1convert_obs
    export CONVERT_OUTPUT_DIR=$PRODUCER/4assimilation/1convert_obs
    export EXPERIMENT_MODE=$mode
    export CENTER_DAY=${CTAG:0:2} CENTER_HOUR=${CTAG:3:2} CENTER_MIN=${CTAG:6:2}
    export obs_day=${CTAG:0:2} obs_hour=${CTAG:3:2} obs_min=${CTAG:6:2}
    export cycle_flag=1 profile_matlab_flag=1
    export LACC_LAG_HOURS="${CYCLE_LAGS[$CYCLE_INDEX]}" LACC_TIMES="${LACC_TAGS[*]}"
    export ENS_SIZE=50 ENS_WRFOUT_DIR=$ROOT/hx_inputs/$CTAG
    export PYTHON_BIN=$PYTHON_EXE PYTHONDONTWRITEBYTECODE=1
    export RTTOV_SCATT=0 USE_TOTAL_ICE=0 USE_CLEAR_SKY_MASK=1 CLEAR_SKY_MIN_FRACTION=
    export CLEAR_SKY_THRESHOLD=$CLEAR_SKY_THRESHOLD OBS_ERR_STD=$OBS_ERR_STD HGT_OBS=$HGT_OBS
    export ASSIM_CHANNEL=$ASSIM_CHANNEL OptimizWeight=0
    export USE_NR_SST_DIRECTION_FILTER=$([[ $mode == LACC ]] && echo 1 || echo 0)
    export NR_SST_BG_DIR=$ROOT/prior/$CTAG
    export NR_SST_BG_MEMBER_PATTERN="firstguess_${HX_STATE_DOMAIN}.mem{mem:03d}"
    export NR_SST_NR_FILE=$NR_ROOT/wrfout_d02_$CSTAMP
    export NR_SST_CENTER_TIME=$CSTAMP OVERWRITE_OBS_SEQ=1
    export OBS_SEQ_OUT_NAME="obs_seq.${mode}.${CTAG}"
    export MPLCONFIGDIR=$ROOT/obs/$CTAG/matplotlib
    export PATH="$(dirname "$PYTHON_EXE"):$PATH"
    export RUN_TRUTH=1 RUN_ENSEMBLE=1 RUN_OBS_CONVERT=1 RUN_HX_MERGE=1 RUN_TEXT_TO_OBS=1 RUN_VALIDATION=1
    bash "$PRODUCER/4assimilation/1convert_obs/run_single_assimilation_test.sh"
  ) > "$logfile" 2>&1 || die "observation producer failed ($mode); log: $logfile"
}

# Python: produce_observations
produce_observations() {
  local work=$ROOT/obs/$CTAG
  rm -rf "$work"                                    # 只清本时次未完成的私有产物树
  prepare_producer "$work"
  local m t source target
  for ((m = 1; m <= 50; m++)); do
    for t in "${LACC_STAMPS[@]}"; do
      source=$(input_source "$t" "$HX_STATE_DOMAIN" "$m")
      check_state "$source" "$t" TSK OM_TMP
      target=$ROOT/hx_inputs/$CTAG/mem$(printf '%03d' "$m")/wrfout_${HX_STATE_DOMAIN}_${t}
      link_readonly "$source" "$target"
    done
  done
  local mode
  for mode in STANDARD LACC; do
    log "$ROOT ${CTAG}: generating ${mode} truth, H(x), mask and obs_seq"
    run_producer "$mode" "$work/${mode}.log"
  done
  local base=$PRODUCER/4assimilation/1convert_obs/run_dir
  OBS_STANDARD=$base/obs_seq.STANDARD.${CTAG}
  OBS_LACC=$base/obs_seq.LACC.${CTAG}_nr_sst_direction
  require_file "$OBS_STANDARD"
  require_file "$OBS_LACC"
}

# ============================== DART 单阶段 ================================
# Python: dart_job_script
dart_job_script() {  # <work> <out>
  cat > "$2" <<EOF
#!/bin/bash
set -Eeuo pipefail
cd $(printf '%q' "$1")
ulimit -s unlimited
$DART_MPI ./filter
for kind in prior post; do
  f="\${kind}_forward_ope_errors000000"
  if [[ -f "\$f" ]]; then mv "\$f" "\${kind}_forward_ope_errors"; fi
done
touch dart_done
EOF
}

# Python: write_namelist —— 这里用 sed 行替换（Python 版按 &section 作用域并要求恰好一条）
render_dart_namelist() {  # <template> <target> <stage-ocean?>
  local src=$1 target=$2
  cp "$src" "$target"
  sed -i -e "s#^[[:space:]]*ens_size[[:space:]]*=.*#ens_size = 50,#" \
         -e "s#^[[:space:]]*input_state_file_list[[:space:]]*=.*#input_state_file_list = 'input_list_d01.txt', 'input_list_d02.txt',#" \
         -e "s#^[[:space:]]*output_state_file_list[[:space:]]*=.*#output_state_file_list = 'output_list_d01.txt', 'output_list_d02.txt',#" \
         -e "s#^[[:space:]]*num_domains[[:space:]]*=.*#num_domains = 2,#" \
         -e "s#^[[:space:]]*assimilation_period_seconds[[:space:]]*=.*#assimilation_period_seconds = 3600,#" \
         "$target"
  if [[ $FILTER == QCF_RHF ]]; then
    sed -i "s#^[[:space:]]*qceff_table_filename[[:space:]]*=.*#qceff_table_filename = 'qceff_table_fkc.csv',#" "$target"
  else
    sed -i "s#^[[:space:]]*qceff_table_filename[[:space:]]*=.*#qceff_table_filename = '',#" "$target"
  fi
}

# Python: validate_dart_outputs —— DART 输出可能不带 Times，带的话必须匹配
validate_dart_outputs() {  # <work> <ocean 0|1> ; 打印所有输出路径
  local work=$1 ocean=$2 dom m path got
  local -a vars
  if [[ $ocean -eq 1 ]]; then IFS=, read -r -a vars <<<"$ATM_VARS,$OCEAN_VARS"
  else IFS=, read -r -a vars <<<"$ATM_VARS"; fi
  for dom in "${DOMAINS[@]}"; do
    for ((m = 1; m <= 50; m++)); do
      path=$work/output_${dom}.mem$(printf '%03d' "$m")
      require_file "$path"
      got=$(nc_times "$path" | head -1)
      [[ -z $got || $got == "$CSTAMP" ]] || die "$path: DART output valid time mismatch"
      check_vars "$path" "${vars[@]}"
      printf '%s\n' "$path"
    done
  done
}

# Python: run_dart_stage —— 两步 DA 里的"一步"
run_dart_stage() {  # <stage> <template> <obs_file>
  local stage=$1 template=$2 obs=$3
  local area=$ROOT/dart/$CTAG/$stage work=$area/work
  local sig="$SIG:$CTAG:$stage" ocean=1
  [[ $template == input.nml.air ]] && ocean=0
  if have_done "$area/done.json" "$sig" "$RESUME"; then
    log "$EXP_NAME ${CTAG}: verified DART $stage checkpoint"
    validate_dart_outputs "$work" "$ocean" >/dev/null
    DART_WORKDIR=$work; return
  fi
  if [[ -e $work/failed && $RESUME -eq 1 ]]; then
    mv "$work" "$area/failed_$(date +%s%N)"
  fi
  if [[ ! -e $work/lsf_job.id ]]; then           # 尚未提交过：准备私有工作目录
    rm -rf "$work"                               # 上次在提交前中断
    mkdir -p "$work"
    link_readonly "$DART_WORK/filter" "$work/filter"
    link_readonly "$DART_TEMPLATE_DIR/rttov_sensor_db.csv" "$work/rttov_sensor_db.csv"
    link_readonly "$DART_TEMPLATE_DIR/rtcoef_noaa_18_amsua.dat" "$work/rtcoef_noaa_18_amsua.dat"
    [[ $FILTER == QCF_RHF ]] && cp "$DART_TEMPLATE_DIR/qceff_table_fkc.csv" "$work/qceff_table_fkc.csv"
    local dom m
    for dom in "${DOMAINS[@]}"; do
      : > "$work/input_list_${dom}.txt"
      : > "$work/output_list_${dom}.txt"
      for ((m = 1; m <= 50; m++)); do
        local p; p=$(prior_path "$CSTAMP" "$dom" "$m")
        check_state "$p" "$CSTAMP"
        printf '%s\n' "$p" >> "$work/input_list_${dom}.txt"
        printf 'output_%s.mem%03d\n' "$dom" "$m" >> "$work/output_list_${dom}.txt"
      done
      link_readonly "$(prior_path "$CSTAMP" "$dom" 1)" "$work/wrfinput_${dom}"
    done
    cp "$obs" "$work/obs_seq.out"                # 真拷贝：把本阶段的观测输入钉住
    cp "$DART_TEMPLATE_DIR/$template" "$work/template.nml"
    render_dart_namelist "$DART_TEMPLATE_DIR/$template" "$work/input.nml"
  fi
  dart_job_script "$work" "$work/job.sh"
  local jobid
  jobid=$(lsf_submit "$work" "$(job_name "$stage")" "$DART_TASKS" "$DART_QUEUE")
  lsf_wait "$jobid" 0                             # allow=0：EXIT 即失败
  if [[ ! -f $work/dart_done ]]; then
    printf 'missing marker\n' > "$work/failed"
    die "DART job $jobid: missing successful-completion marker"
  fi
  local listing
  if ! listing=$(validate_dart_outputs "$work" "$ocean" 2>&1); then
    printf '%s\n' "$listing" > "$work/failed"
    die "DART job $jobid: output validation failed; see $work/failed"
  fi
  mapfile -t DART_OUTPUTS <<<"$listing"
  write_done "$area/done.json" "$sig" "${DART_OUTPUTS[@]}" "$work/input.nml" "$work/template.nml" \
             "$work/obs_seq.out" "$obs"
  DART_WORKDIR=$work
}

# Python: assimilate —— 两步 DA + ncks 拼回 prior，得到 analysis
assimilate() {
  local dest=$ROOT/analysis/$CTAG sig="$SIG:$CTAG:analysis"
  if have_done "$dest/done.json" "$sig" "$RESUME"; then
    log "$EXP_NAME ${CTAG}: verified full WRF analysis checkpoint"; return
  fi
  local ocean_work="" atm_template
  if [[ $UPDATE_OCEAN -eq 1 ]]; then
    # 第一次：只做海洋膨胀/更新（用 LACC 观测）
    run_dart_stage ocean input.nml.inflateOcean "$OBS_LACC"
    ocean_work=$DART_WORKDIR
    atm_template=input.nml.all
  else
    atm_template=input.nml.air
  fi
  # 第二次：大气（group3 用 all，否则 air）；两者的海洋部分按 6 成员版做法丢弃
  run_dart_stage atmosphere "$atm_template" "$OBS_STANDARD"
  local atm_work=$DART_WORKDIR
  mkdir -p "$dest"
  local dom m name tmp
  local -a outputs=()
  for dom in "${DOMAINS[@]}"; do
    for ((m = 1; m <= 50; m++)); do
      name=${dom}.mem$(printf '%03d' "$m")
      tmp=$dest/analysis_${name}.tmp
      cp "$(prior_path "$CSTAMP" "$dom" "$m")" "$tmp"
      ncks -A -v "$ATM_VARS" "$atm_work/output_${name}" "$tmp"
      if [[ -n $ocean_work ]]; then
        ncks -A -v "$OCEAN_VARS" "$ocean_work/output_${name}" "$tmp"
      fi
      check_state "$tmp" "$CSTAMP" ${ATM_VARS//,/ } ${OCEAN_VARS//,/ }
      mv "$tmp" "$dest/analysis_${name}"
      outputs+=("$dest/analysis_${name}")
    done
  done
  write_done "$dest/done.json" "$sig" "${outputs[@]}"
  log "$EXP_NAME ${CTAG}: complete 50-member, two-domain analysis"
}

# ============================== WRF 预报 ===================================
# Python: expected_wrf_files —— 每 30 min 一个输出，两个域
expected_wrf_files() {  # <dir> ; 打印 "路径 时刻"
  local dir=$1 minute moment dom
  for ((minute = 0; minute <= CYCLE_HOURS_H * 60; minute += 30)); do
    moment=$(stamp_of "$(shift_minutes "$CYCLE_ISO" "$minute")")
    for dom in "${DOMAINS[@]}"; do
      printf '%s %s\n' "$dir/wrfout_${dom}_${moment}" "$moment"
    done
  done
}

# Python: wrf_job_script —— 数组作业的一个元素跑一个成员
wrf_job_script() {  # <work_root> <identity> <out>
  cat > "$3" <<EOF
#!/bin/bash
set -Eeuo pipefail
mem=\$(printf "%03d" "\${LSB_JOBINDEX:?missing array index}")
cd $(printf '%q' "$1")/"\${mem}"
ulimit -s unlimited
ulimit -d unlimited
export I_MPI_DEVICE=rdssm WRFIO_NCD_LARGE_FILE_SUPPORT=1
$WRF_MPI ./wrf.exe
if ! grep -q 'SUCCESS COMPLETE WRF' rsl.error.0000 2>/dev/null && ! grep -q 'SUCCESS COMPLETE WRF' rsl.out.0000 2>/dev/null; then
  echo 'WRF success evidence is absent' >&2; exit 1
fi
printf '%s\n' $(printf '%q' "$2") > wrf_done
EOF
}

# Python: write_namelist(overrides) —— 同样退化为 sed
render_wrf_namelist() {  # <work>
  local work=$1
  cp "$PROJECT_ROOT/0necessay_files/namelist.input.ens" "$work/namelist.input"
  local f=$work/namelist.input
  sed -i -e "s#^[[:space:]]*run_days[[:space:]]*=.*#run_days = 0,#" \
         -e "s#^[[:space:]]*run_hours[[:space:]]*=.*#run_hours = ${CYCLE_HOURS_H},#" \
         -e "s#^[[:space:]]*run_minutes[[:space:]]*=.*#run_minutes = 0,#" \
         -e "s#^[[:space:]]*run_seconds[[:space:]]*=.*#run_seconds = 0,#" \
         -e "s#^[[:space:]]*start_year[[:space:]]*=.*#start_year = ${CYCLE_ISO:0:4}, ${CYCLE_ISO:0:4},#" \
         -e "s#^[[:space:]]*start_month[[:space:]]*=.*#start_month = ${CYCLE_ISO:5:2}, ${CYCLE_ISO:5:2},#" \
         -e "s#^[[:space:]]*start_day[[:space:]]*=.*#start_day = ${CYCLE_ISO:8:2}, ${CYCLE_ISO:8:2},#" \
         -e "s#^[[:space:]]*start_hour[[:space:]]*=.*#start_hour = ${CYCLE_ISO:11:2}, ${CYCLE_ISO:11:2},#" \
         -e "s#^[[:space:]]*start_minute[[:space:]]*=.*#start_minute = ${CYCLE_ISO:14:2}, ${CYCLE_ISO:14:2},#" \
         -e "s#^[[:space:]]*start_second[[:space:]]*=.*#start_second = ${CYCLE_ISO:17:2}, ${CYCLE_ISO:17:2},#" \
         -e "s#^[[:space:]]*end_year[[:space:]]*=.*#end_year = ${CEND_ISO:0:4}, ${CEND_ISO:0:4},#" \
         -e "s#^[[:space:]]*end_month[[:space:]]*=.*#end_month = ${CEND_ISO:5:2}, ${CEND_ISO:5:2},#" \
         -e "s#^[[:space:]]*end_day[[:space:]]*=.*#end_day = ${CEND_ISO:8:2}, ${CEND_ISO:8:2},#" \
         -e "s#^[[:space:]]*end_hour[[:space:]]*=.*#end_hour = ${CEND_ISO:11:2}, ${CEND_ISO:11:2},#" \
         -e "s#^[[:space:]]*end_minute[[:space:]]*=.*#end_minute = ${CEND_ISO:14:2}, ${CEND_ISO:14:2},#" \
         -e "s#^[[:space:]]*end_second[[:space:]]*=.*#end_second = ${CEND_ISO:17:2}, ${CEND_ISO:17:2},#" \
         -e "s#^[[:space:]]*history_interval[[:space:]]*=.*#history_interval = 30, 30,#" \
         -e "s#^[[:space:]]*frames_per_outfile[[:space:]]*=.*#frames_per_outfile = 1, 1,#" \
         -e "s#^[[:space:]]*restart[[:space:]]*=.*#restart = .false.,#" \
         -e "s#^[[:space:]]*input_from_file[[:space:]]*=.*#input_from_file = .true., .true.,#" \
         -e "s#^[[:space:]]*max_dom[[:space:]]*=.*#max_dom = 2,#" \
         -e "s#^[[:space:]]*sf_ocean_physics[[:space:]]*=.*#sf_ocean_physics = $([[ $RUN_OCEAN -eq 1 ]] && echo 2 || echo 0),#" \
         "$f"
}

# Python: prepare_wrf_member
prepare_wrf_member() {
  local m=$1 mem work archive dom
  mem=$(printf '%03d' "$m")
  work=$ROOT/run_wrf/$CTAG/$mem
  archive=$ROOT/wrfout/$CTAG/$mem
  if [[ -e $work ]]; then
    local keep=$ROOT/logs/$CTAG/${mem}_$(date +%s%N) f
    mkdir -p "$keep"
    for f in "$work"/rsl.*; do [[ -f $f ]] && cp -p "$f" "$keep/"; done
    rm -rf "$work"
  fi
  [[ -e $archive ]] && rm -rf "$archive"          # 上次崩溃留下的半成品归档
  mkdir -p "$work"
  local entry
  while read -r entry; do link_readonly "$entry" "$work/$(basename "$entry")"; done < <(wrf_static_entries)
  for dom in "${DOMAINS[@]}"; do
    local src=$ROOT/analysis/$CTAG/analysis_${dom}.mem${mem}
    check_state "$src" "$CSTAMP"
    cp "$src" "$work/wrfinput_${dom}"
  done
  cp "$(bc_path "$m")" "$work/wrfbdy_d01"
  link_readonly "$(bc_path "$m")" "$work/wrfbdy_d01_orig"
  link_readonly "$DART_WORK/update_wrf_bc" "$work/update_wrf_bc"
  cp "$PROJECT_ROOT/0necessay_files/input.nml" "$work/input.nml"
  render_wrf_namelist "$work"
  ( cd "$work" && ./update_wrf_bc ) > "$work/update_bc.log" 2>&1 \
    || die "update_wrf_bc failed for mem$mem; see $work/update_bc.log"
  link_readonly "$(ic_path "$m")" "$work/wrfinput_d01_gfs"
  for dom in "${DOMAINS[@]}"; do
    ( cd "$work" && "$PYTHON_EXE" "$PROJECT_ROOT/0necessay_files/ad_omini_${dom}.py" ) \
      > "$work/ad_omini_${dom}.log" 2>&1 || die "ad_omini_${dom}.py failed for mem$mem"
    check_state "$work/wrfinput_${dom}" "$CSTAMP" OM_TINI OM_SINI
  done
}

# Python: member_ready —— 归档 checkpoint 有效，或"wrf_done 标记 + 所有期望 wrfout 都在且时刻正确"
member_ready() {
  local m=$1 identity=$2 mem work archive marker
  mem=$(printf '%03d' "$m")
  work=$ROOT/run_wrf/$CTAG/$mem
  archive=$ROOT/wrfout/$CTAG/$mem
  have_done "$archive/member_done.json" "$identity" 1 && return 0
  marker=$work/wrf_done
  [[ -f $marker && $(<"$marker") == "$identity" ]] || return 1
  local path moment
  while read -r path moment; do
    [[ -f $path ]] || path=$work/$(basename "$path")
    check_state "$path" "$moment" TSK >/dev/null 2>&1 || return 1
  done < <(expected_wrf_files "$archive")
  return 0
}

# Python: publish_next_prior —— 把本时次末端 wrfout 变成下一时次 prior
publish_next_prior() {  # <domain> <member> <archived>
  local dom=$1 m=$2 archived=$3
  local target; target=$(prior_path "$CEND_STAMP" "$dom" "$m")
  if [[ $RUN_OCEAN -eq 1 ]]; then
    check_state "$archived" "$CEND_STAMP" OM_TMP OM_S OM_U OM_V
    link_readonly "$archived" "$target"
    return
  fi
  # run_ocean=0 时 WRF 不输出 OM_*：把上一时次未演变的海洋原样带过去（COPY，不改原 wrfout）
  local previous; previous=$(prior_path "$CSTAMP" "$dom" "$m")
  check_state "$previous" "$CSTAMP" OM_TMP OM_S OM_U OM_V
  check_state "$archived" "$CEND_STAMP"
  local identity
  identity=$(printf '%s %s %s' "$previous" "$(stat -c '%s' "$previous")" "$(stat -c '%Y' "$previous")")
  if [[ -f $target && ! -L $target ]]; then
    check_state "$target" "$CEND_STAMP" OM_TMP OM_S OM_U OM_V
    ncdump -h "$target" | grep -qF "$identity" || die "frozen-ocean prior has a different source: $target"
    return
  fi
  mkdir -p "$(dirname "$target")"
  cp "$archived" "$target.tmp"
  [[ $(ncks -H -C -v XLAT,XLONG "$previous" | md5sum) == $(ncks -H -C -v XLAT,XLONG "$target.tmp" | md5sum) ]] \
    || die "cannot carry frozen ocean across a changed grid: $dom mem$(printf '%03d' "$m")"
  ncks -A -v "$OCEAN_VARS" "$previous" "$target.tmp"
  ncatted -O -a CYCLE_FROZEN_OCEAN_SOURCE,global,o,c,"$identity" \
               -a CYCLE_OCEAN_STATE,global,o,c,"unadvanced ocean: update_ocean=0, run_ocean=0" "$target.tmp"
  check_state "$target.tmp" "$CEND_STAMP" OM_TMP OM_S OM_U OM_V
  [[ -L $target ]] && rm -f "$target"
  mv "$target.tmp" "$target"
}

# Python: archive_member —— 校验通过后把 wrfout 移进归档并 publish 下一时次 prior
archive_member() {
  local m=$1 identity=$2 mem work archive
  mem=$(printf '%03d' "$m")
  work=$ROOT/run_wrf/$CTAG/$mem
  archive=$ROOT/wrfout/$CTAG/$mem
  if ! have_done "$archive/member_done.json" "$identity" 1; then
    mkdir -p "$archive"
    local f target
    for f in "$work"/wrfout_d0*; do
      target=$archive/$(basename "$f")
      [[ -e $target ]] && die "conflicting WRF archive file: $target"
      mv "$f" "$target"
    done
    local -a files=()
    local path moment
    while read -r path moment; do
      check_state "$path" "$moment" TSK
      files+=("$path")
    done < <(expected_wrf_files "$archive")
    local dom
    for dom in "${DOMAINS[@]}"; do
      publish_next_prior "$dom" "$m" "$archive/wrfout_${dom}_${CEND_STAMP}"
      files+=("$(prior_path "$CEND_STAMP" "$dom" "$m")")
    done
    write_done "$archive/member_done.json" "$identity" "${files[@]}"
  fi
}

# Python: forecast —— 尝试-提交-监控-校验-归档 的主循环
forecast() {
  local identity="$SIG:$CTAG" checkpoint=$ROOT/wrfout/$CTAG/done.json
  if have_done "$checkpoint" "$identity:forecast" "$RESUME"; then
    log "$EXP_NAME ${CTAG}: verified archived WRF forecast checkpoint"; return
  fi
  local jobs=$ROOT/jobs/$CTAG
  local -a ready=()
  local m
  for ((m = 1; m <= 50; m++)); do ready[$m]=0; done

  # resume：先确认历史上提交过的数组作业都已终结，才允许检查/删除/重试成员文件
  # （仍在 RUN 的作业若被 rmtree 工作目录，会导致灾难性后果）
  local jf area job
  local -a prev_state=()
  for ((m = 1; m <= 50; m++)); do prev_state[$m]=DONE; done   # 从未提交过 = 视为可继续
  for jf in "$jobs"/attempt_*/lsf_job.id; do
    [[ -f $jf ]] || continue
    area=$(dirname "$jf")
    job=$(cat "$jf")
    local -a members=() states=()
    mapfile -t members < "$area/members"
    lsf_wait "$job" 1 "${members[@]}" || true
    mapfile -t states < <(lsf_states "$job" "${members[@]}")
    local i=0
    for m in "${members[@]}"; do
      prev_state[$m]=${states[$i]:-NONE}
      i=$((i + 1))
    done
  done
  # 就绪 = 上次作业 DONE（或从未提交）+ 归档 checkpoint / wrf_done + 所有期望 wrfout 校验通过
  for ((m = 1; m <= 50; m++)); do
    if [[ ${prev_state[$m]} == DONE ]] && member_ready "$m" "$identity"; then ready[$m]=1; fi
  done

  local attempt existing=0
  existing=$(find "$jobs" -maxdepth 1 -name 'attempt_*' -type d 2>/dev/null | wc -l)
  for ((attempt = existing + 1; attempt <= MAX_WRF_ATTEMPTS; attempt++)); do
    local -a pending=()
    local count=0
    for ((m = 1; m <= 50; m++)); do
      if [[ ${ready[$m]} -ne 1 ]]; then pending+=("$m"); count=$((count + 1)); fi
    done
    ((count)) || break
    for m in "${pending[@]}"; do prepare_wrf_member "$m"; done
    area=$jobs/$(printf 'attempt_%02d' "$attempt")
    mkdir -p "$area"
    wrf_job_script "$ROOT/run_wrf/$CTAG" "$identity" "$area/job.sh"
    printf '%s\n' "${pending[@]}" > "$area/members"
    local jobid
    jobid=$(lsf_submit "$area" "$(job_name "wrf$attempt")" "$WRF_TASKS" "$WRF_QUEUE" "${pending[@]}")
    lsf_wait "$jobid" 1 "${pending[@]}" || true
    local -a states=(); mapfile -t states < <(lsf_states "$jobid" "${pending[@]}")
    local i=0
    for m in "${pending[@]}"; do
      if [[ ${states[$i]:-NONE} == DONE ]] && member_ready "$m" "$identity"; then
        archive_member "$m" "$identity"
        ready[$m]=1
      fi
      i=$((i + 1))
    done
    local done_count=0
    for ((m = 1; m <= 50; m++)); do [[ ${ready[$m]} -eq 1 ]] && done_count=$((done_count + 1)); done
    log "$EXP_NAME ${CTAG}: WRF attempt ${attempt}, ${done_count}/50 members complete"
  done
  local done_count=0 missing=""
  for ((m = 1; m <= 50; m++)); do
    if [[ ${ready[$m]} -eq 1 ]]; then done_count=$((done_count + 1)); else missing+=" mem$(printf '%03d' "$m")"; fi
  done
  [[ $done_count -eq 50 ]] || die "$EXP_NAME ${CTAG}: WRF failed after ${MAX_WRF_ATTEMPTS} attempts:${missing}"

  # 收尾：对全部成员跑一遍 archive_member（幂等），保证
  #   * resume 时"只有 wrf_done 标记、尚未归档"的成员也被归档
  #   * 每个成员的下一时次 prior 都被 publish
  for ((m = 1; m <= 50; m++)); do archive_member "$m" "$identity"; done

  local -a outputs=()
  local path moment dom
  for ((m = 1; m <= 50; m++)); do
    while read -r path moment; do outputs+=("$path"); done < <(expected_wrf_files "$ROOT/wrfout/$CTAG/$(printf '%03d' "$m")")
    for dom in "${DOMAINS[@]}"; do outputs+=("$(prior_path "$CEND_STAMP" "$dom" "$m")"); done
  done
  write_done "$checkpoint" "$identity:forecast" "${outputs[@]}"
}

# ============================== 主流程 =====================================
# Python: run() —— preflight -> flock -> experiment x cycle
run() {
  preflight
  mkdir -p "$RUN_ROOT"
  exec 9>"$RUN_ROOT/.driver.lock"
  flock -n 9 || die "another driver owns RUN_ROOT: $RUN_ROOT"
  local g f
  for g in "${SELECTED_GROUPS[@]}"; do
    for f in "${FILTERS[@]}"; do
      run_experiment "$g" "$f"
    done
  done
}

# 单个 experiment：初始化 + 4 个时次的 obs -> DA -> WRF
run_experiment() {
  local group=$1 filter=$2 c
  set_experiment "$group" "$filter"
  init_experiment
  log "Starting $EXP_NAME"
  for ((c = 0; c < CYCLE_COUNT; c++)); do
    CYCLE_INDEX=$c
    set_cycle "$c"
    # 每个时次都复核一次指纹：外部输入/依赖在实验中途变化就拒绝继续
    [[ $(experiment_signature) == "$SIG" ]] \
      || die "external inputs/runtime dependencies changed during the experiment: $ROOT"
    if have_done "$ROOT/obs/$CTAG/done.json" "$SIG:$CTAG:observations" "$RESUME"; then
      mapfile -t OBS_FILES < "$ROOT/obs/$CTAG/.obs_files"
      OBS_STANDARD=${OBS_FILES[0]} OBS_LACC=${OBS_FILES[1]}
    else
      produce_observations
      printf '%s\n%s\n' "$OBS_STANDARD" "$OBS_LACC" > "$ROOT/obs/$CTAG/.obs_files"
      mapfile -t OBS_ALL < <(find "$ROOT/obs/$CTAG" -type f -size +0 | sort)
      write_done "$ROOT/obs/$CTAG/done.json" "$SIG:$CTAG:observations" "${OBS_ALL[@]}"
    fi
    assimilate
    forecast
  done
  printf 'signature=%s\nlast_forecast=%s\n' "$SIG" "$FINAL_END_STAMP" > "$ROOT/complete.meta"
  log "Completed $EXP_NAME: final free forecast reaches $FINAL_END_STAMP"
}

# Python: plan()
plan() {
  local g f i first
  printf '{\n  "members": 50,\n  "domains": ["d01","d02"],\n  "hx_state_domain": "%s",\n' "$HX_STATE_DOMAIN"
  printf '  "boundary_mem001": "%s",\n  "initialization_mem001": "%s",\n' "$(bc_path 1)" "$(ic_path 1)"
  printf '  "experiments": [\n'
  for g in "${SELECTED_GROUPS[@]}"; do
    for f in "${FILTERS[@]}"; do
      set_experiment "$g" "$f"
      printf '    {"group": %d, "filter": "%s", "root": "%s", "update_ocean": %d, "run_ocean": %d,\n' \
        "$GROUP" "$FILTER" "$ROOT" "$UPDATE_OCEAN" "$RUN_OCEAN"
      printf '     "cycles": ['
      first=1
      for ((i = 0; i < CYCLE_COUNT; i++)); do
        set_cycle "$i"
        ((first)) || printf ', '
        first=0
        local lacc="" lag
        for lag in "${LACC_STAMPS[@]}"; do lacc+="${lacc:+, }\"$lag\""; done
        printf '{"analysis": "%s", "end": "%s", "forecast_hours": %d, "lag_hours": [%s], "lacc_times": [%s]}' \
          "$CSTAMP" "$CEND_STAMP" "$CYCLE_HOURS_H" "${CYCLE_LAGS[$i]// /, }" "$lacc"
      done
      printf ']},\n'
    done
  done
  printf '  ]\n}\n'
}

main() {
  # 从环境变量组装实验清单（顺序固定：group 1,2,3 × filter EAKF,QCF_RHF）
  # 注意：变量名不能叫 GROUPS —— 那是 bash 内建只读数组，赋值会失败。
  SELECTED_GROUPS=()
  FILTERS=()
  for g in 1 2 3; do
    local flag=RUN_GROUP_$g
    [[ ${!flag} == 0 || ${!flag} == 1 ]] || die "$flag must be 0 or 1"
    [[ ${!flag} -eq 1 ]] && SELECTED_GROUPS+=("$g")
  done
  local f
  for f in EAKF QCF_RHF; do
    local flag=RUN_$f
    [[ ${!flag} == 0 || ${!flag} == 1 ]] || die "$flag must be 0 or 1"
    [[ ${!flag} -eq 1 ]] && FILTERS+=("$f")
  done
  ((${#SELECTED_GROUPS[@]})) && ((${#FILTERS[@]})) || die "select at least one coupling group and one filter"
  [[ $HX_STATE_DOMAIN == d01 || $HX_STATE_DOMAIN == d02 ]] || die "HX_STATE_DOMAIN must be d01/d02"
  ((ASSIM_CHANNEL >= 1 && ASSIM_CHANNEL <= 6)) || die "channel must be 1..6"
  case $MODE in
    DRY_RUN)   plan ;;
    PREFLIGHT) preflight ;;
    RUN)       run ;;
    *)         die "MODE must be RUN, PREFLIGHT or DRY_RUN" ;;
  esac
}

main "$@"
