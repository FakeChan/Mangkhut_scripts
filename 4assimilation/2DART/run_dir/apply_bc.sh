#!/bin/bash
# ==============================================================================
# apply_bc.sh — 对集合成员 firstguess_d0X.mem* 的 OM_TMP 施加偏差订正
#
#   对每个 domain:
#     delta = bc_fg_d0X.ensmean - firstguess_d0X.ensmean   (整场逐网格点)
#     member.OM_TMP = member.OM_TMP + delta                (整场相加, 广播机制)
#
# 订正管线:
#   1. ncdiff 计算 delta 场 (仅 OM_TMP)
#   2. python 就地订正每个成员: 读 delta 与成员 OM_TMP, 相加后写回
#      (netCDF4 r+ 只读写 OM_TMP 一个变量, 不重写整个文件;
#       delta 的缺失值在内存中清零, 陆地 _FillValue 不污染成员)
#
# 依赖 NCO (ncdiff / ncks / ncdump) 与 python3(netCDF4)。
# 可选环境变量 NPROC: 并行订正成员数 (默认 1), 内存需能容纳 NPROC 个 OM_TMP 全场。
#
# 用法:
#   ./apply_bc.sh
#   可用环境变量覆盖:
#     cycle_dir   源目录 (默认 .../4assimilation/0mem_all_time/10_00_00)
#     ensmem_dir  工作/输出目录 (默认 .../0mem_all_time/cyclingDA/10_00_00)
#
# 注意: 重跑会重复叠加 delta，重跑前请先清空 WORK_DIR。
# ==============================================================================

set -u

# ==============================================================================
# 1. 用户配置区域
# ==============================================================================

# 源目录: 存放 firstguess_d0X.mem* 与 ensmean 文件
SRC_DIR="${cycle_dir:-/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/10_00_00}"

# 工作目录: 拷贝成员并施加订正
WORK_DIR="${ensmem_dir:-/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_00_00}"

# 参与订正的 domain 列表
DOMAINS=(d01 d02)

# 订正变量
VAR=OM_TMP

# 预期垂直层数 (仅作检查提示，不强制终止)
NLEV_EXPECT=30

# 并行订正成员数 (可选, 默认串行; 内存需能容纳 NPROC 个 OM_TMP 全场)
NPROC="${NPROC:-1}"

# ==============================================================================
# 2. 功能函数
# ==============================================================================

die() {
    echo "[Error] $*" >&2
    exit 1
}

# 检查 NCO 工具与 python3 是否可用
check_tools() {
    for t in ncdiff ncks ncdump; do
        command -v "$t" >/dev/null 2>&1 || die "缺少 NCO 工具: $t (请先加载 nco 环境)"
    done
    command -v python3 >/dev/null 2>&1 || die "缺少 python3 (delta 缺失值清零需要, 含 netCDF4 模块)"
    python3 -c 'import netCDF4' >/dev/null 2>&1 || die "python3 缺少 netCDF4 模块"
}

# ==============================================================================
# 3. 主程序
# ==============================================================================

echo "================================================"
echo "WRF OM_TMP Bias Correction Tool (in-place, NPROC=${NPROC})"
echo "================================================"
echo "源目录:    $SRC_DIR"
echo "工作目录:  $WORK_DIR"
echo "Domain:    ${DOMAINS[*]}"
echo "变量:      $VAR"
echo ""

check_tools

# --- Step 1: 前置检查 (源目录 + ensmean 文件) ---
echo "Step 1: 前置检查"
[ -d "$SRC_DIR" ] || die "源目录不存在: $SRC_DIR"

missing=0
for dom in "${DOMAINS[@]}"; do
    for f in "firstguess_${dom}.ensmean" "bc_fg_${dom}.ensmean"; do
        if [ ! -f "$SRC_DIR/$f" ]; then
            echo "[Error] 缺少必需文件: $SRC_DIR/$f"
            missing=1
        fi
    done
done
[ "$missing" -eq 0 ] || die "前置文件缺失，终止。"

# --- Step 2: 拷贝成员到工作目录 ---
echo ""
echo "Step 2: 拷贝成员文件"
mkdir -p "$WORK_DIR/deltas" || die "无法创建 $WORK_DIR/deltas"

n_total=0
for dom in "${DOMAINS[@]}"; do
    n=0
    for src in "$SRC_DIR"/firstguess_${dom}.mem*; do
        [ -e "$src" ] || continue
        dst="$WORK_DIR/$(basename "$src")"
        # 仅当目标不存在或源文件更新时才拷贝 (POSIX mtime 判断，兼容 GNU/macOS)
        if [ ! -e "$dst" ] || [ "$src" -nt "$dst" ]; then
            cp -p "$src" "$dst" || echo "[Warn] 拷贝失败: $src"
        fi
        n=$((n + 1))
    done
    echo "  [$dom] 拷贝成员: $n"
    n_total=$((n_total + n))
done
[ "$n_total" -gt 0 ] || die "未找到任何成员文件 (firstguess_d0X.mem*)"

# --- Step 3: 逐 domain 计算 delta 并订正 ---
echo ""
echo "Step 3: 计算 delta 并订正成员"
# 就地订正小脚本: 只读写 OM_TMP 一个变量 (netCDF4 r+, 不重写整个文件);
# delta 的缺失值在内存中清零, 陆地 _FillValue 不污染成员
cat > "$WORK_DIR/deltas/bc_add.py" <<'PYEOF'
import sys
import netCDF4
import numpy as np

member, delta_file, var = sys.argv[1], sys.argv[2], sys.argv[3]
with netCDF4.Dataset(delta_file) as nd:
    delta = np.ma.filled(nd.variables[var][:], 0.0)
with netCDF4.Dataset(member, 'r+') as nm:
    v = nm.variables[var]
    v[:] = v[:] + delta
PYEOF

for dom in "${DOMAINS[@]}"; do
    delta_file="$WORK_DIR/deltas/delta_${dom}.nc"
    echo "  [$dom] 计算 delta = bc_fg_${dom}.ensmean - firstguess_${dom}.ensmean ..."
    ncdiff -v "$VAR" "$SRC_DIR/bc_fg_${dom}.ensmean" \
           "$SRC_DIR/firstguess_${dom}.ensmean" "$delta_file" \
        || die "[$dom] ncdiff 失败"

    # 层数检查 (仅提示)
    nlev=$(ncdump -h "$delta_file" | grep -o 'bottom_top = [0-9]*' | grep -o '[0-9]*$')
    if [ -n "$nlev" ] && [ "$nlev" -ne "$NLEV_EXPECT" ]; then
        echo "  [Warn] [$dom] OM_TMP 层数 $nlev 与预期 $NLEV_EXPECT 不一致!"
    else
        echo "  [$dom] OM_TMP 层数: ${nlev:-?} (预期 $NLEV_EXPECT)"
    fi

    ok=0; fail=0
    if [ "$NPROC" -gt 1 ]; then
        # 并行模式: xargs -P, 不打印逐成员 [Check]
        members=()
        for f in "$WORK_DIR"/firstguess_${dom}.mem*; do
            [ -e "$f" ] || continue
            members+=("$f")
        done
        if printf '%s\n' "${members[@]}" | xargs -P "$NPROC" -I{} \
            python3 "$WORK_DIR/deltas/bc_add.py" "{}" "$delta_file" "$VAR"; then
            ok=${#members[@]}
        else
            fail=${#members[@]}   # xargs 任一分进程失败会返回非零, 此处保守记为全失败
        fi
    else
        for f in "$WORK_DIR"/firstguess_${dom}.mem*; do
            [ -e "$f" ] || continue

            # 第一个成员: 抽查订正前值 (便于人工核对)
            if [ "$ok" -eq 0 ] && [ "$fail" -eq 0 ]; then
                echo "    [Check] 订正前 $f :"
                ncks -H -C -s '%10.4f ' -v "$VAR" -d bottom_top,0 -d south_north,1 -d west_east,1 "$f" 2>/dev/null | head -3
            fi

            if python3 "$WORK_DIR/deltas/bc_add.py" "$f" "$delta_file" "$VAR"; then
                ok=$((ok + 1))
                # 第一个成员: 抽查订正后值
                if [ "$ok" -eq 1 ] && [ "$fail" -eq 0 ]; then
                    echo "    [Check] 订正后 $f :"
                    ncks -H -C -s '%10.4f ' -v "$VAR" -d bottom_top,0 -d south_north,1 -d west_east,1 "$f" 2>/dev/null | head -3
                fi
            else
                fail=$((fail + 1))
                echo "    [Warn] 订正失败: $f"
            fi
        done
    fi
    echo "  [$dom] 订正完成: 成功 $ok / 失败 $fail"
done

# --- Step 4: 汇总 ---
echo ""
echo "================================================"
echo "所有任务结束。结果保存在: $WORK_DIR"
echo "delta 场保存在: $WORK_DIR/deltas/ (可留作检查)"
echo "提示: 重跑会重复叠加 delta，如需重跑请先清空 $WORK_DIR"
echo "================================================"
