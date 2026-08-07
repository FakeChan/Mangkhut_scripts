#!/bin/bash
# ==============================================================================
# apply_bc.sh — 对集合成员 firstguess_d0X.mem* 的 OM_TMP 施加偏差订正
#
#   对每个 domain:
#     delta = bc_fg_d0X.ensmean - firstguess_d0X.ensmean   (整场逐网格点)
#     member.OM_TMP = member.OM_TMP + delta                (整场相加, 广播机制)
#
# 订正管线 (每个成员):
#   1. ncdiff  计算 delta 场 (仅 OM_TMP)
#   2. ncrename delta 变量改名 OM_TMP_delta (避免与成员变量重名)
#   3. python  将 delta 的缺失值清零 (陆地 _FillValue 不污染成员)
#   4. ncks -A 把 OM_TMP_delta 追加进成员文件 (netCDF3 仅追加, 便宜)
#   5. ncap2   OM_TMP = OM_TMP + OM_TMP_delta (其余变量原样保留)
#   6. ncks -x 剔除临时变量 OM_TMP_delta, 覆盖回成员文件名
#
# 依赖 NCO (ncdiff / ncrename / ncks / ncap2 / ncdump) 与 python3(netCDF4)。
# 注意: ncbo 输出只含公共变量, 会丢弃成员的其他变量, 故不用 ncbo 做加法。
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

# ==============================================================================
# 2. 功能函数
# ==============================================================================

die() {
    echo "[Error] $*" >&2
    exit 1
}

# 检查 NCO 工具与 python3 是否可用
check_tools() {
    for t in ncdiff ncrename ncks ncap2 ncdump; do
        command -v "$t" >/dev/null 2>&1 || die "缺少 NCO 工具: $t (请先加载 nco 环境)"
    done
    command -v python3 >/dev/null 2>&1 || die "缺少 python3 (delta 缺失值清零需要, 含 netCDF4 模块)"
    python3 -c 'import netCDF4' >/dev/null 2>&1 || die "python3 缺少 netCDF4 模块"
}

# ==============================================================================
# 3. 主程序
# ==============================================================================

echo "================================================"
echo "WRF OM_TMP Bias Correction Tool (ncap2 pipeline)"
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
for dom in "${DOMAINS[@]}"; do
    delta_file="$WORK_DIR/deltas/delta_${dom}.nc"
    echo "  [$dom] 计算 delta = bc_fg_${dom}.ensmean - firstguess_${dom}.ensmean ..."
    ncdiff -v "$VAR" "$SRC_DIR/bc_fg_${dom}.ensmean" \
           "$SRC_DIR/firstguess_${dom}.ensmean" "$delta_file" \
        || die "[$dom] ncdiff 失败"
    ncrename -v "$VAR,${VAR}_delta" "$delta_file" \
        || die "[$dom] ncrename 失败"

    # delta 缺失值清零: 若 ensmean 在陆地格点为 _FillValue, delta 在陆地也为缺失,
    # 直接相加会把成员的陆地值污染成缺失。NCO 的 mask_miss/delete_miss 行为不可靠,
    # 用 python 一步完成 (netCDF4 集群已有)。
    python3 - "$delta_file" "$dom" "$VAR" <<'EOF' || die "[$dom] delta 缺失值清零失败"
import sys
import netCDF4
import numpy as np

f, dom, var = sys.argv[1], sys.argv[2], sys.argv[3]
vname = var + "_delta"
with netCDF4.Dataset(f, 'r+') as nc:
    v = nc.variables[vname]
    d = v[:]
    n = int(np.ma.getmaskarray(d).sum())
    if n > 0:
        v[:] = np.ma.filled(d, 0.0)
        if '_FillValue' in v.ncattrs():
            v.delncattr('_FillValue')
    print(f"  [{dom}] delta 缺失值清零: {n} 个点")
EOF

    # 层数检查 (仅提示)
    nlev=$(ncdump -h "$delta_file" | grep -o 'bottom_top = [0-9]*' | grep -o '[0-9]*$')
    if [ -n "$nlev" ] && [ "$nlev" -ne "$NLEV_EXPECT" ]; then
        echo "  [Warn] [$dom] OM_TMP 层数 $nlev 与预期 $NLEV_EXPECT 不一致!"
    else
        echo "  [$dom] OM_TMP 层数: ${nlev:-?} (预期 $NLEV_EXPECT)"
    fi

    ok=0; fail=0
    for f in "$WORK_DIR"/firstguess_${dom}.mem*; do
        [ -e "$f" ] || continue

        # 第一个成员: 抽查订正前值 (便于人工核对)
        if [ "$ok" -eq 0 ] && [ "$fail" -eq 0 ]; then
            echo "    [Check] 订正前 $f :"
            ncks -H -C -s '%10.4f ' -v "$VAR" -d bottom_top,0 -d south_north,1 -d west_east,1 "$f" 2>/dev/null | head -3
        fi

        # 1) ncks -A: 把 OM_TMP_delta 追加进成员 (仅 header+EOF, 不重写全文件)
        # 2) ncap2:  整场相加, 成员其余变量原样保留
        # 3) ncks -x: 剔除临时变量 OM_TMP_delta, 输出直接覆盖回成员文件名
        if ncks -A -v "${VAR}_delta" "$delta_file" "$f" \
            && ncap2 -O -s "$VAR=${VAR}+${VAR}_delta" "$f" "$f.step.nc" \
            && ncks -O -x -v "${VAR}_delta" "$f.step.nc" "$f" \
            && rm -f "$f.step.nc"; then
            ok=$((ok + 1))
            # 第一个成员: 抽查订正后值
            if [ "$ok" -eq 1 ] && [ "$fail" -eq 0 ]; then
                echo "    [Check] 订正后 $f :"
                ncks -H -C -s '%10.4f ' -v "$VAR" -d bottom_top,0 -d south_north,1 -d west_east,1 "$f" 2>/dev/null | head -3
            fi
        else
            fail=$((fail + 1))
            echo "    [Warn] 订正失败: $f"
            rm -f "$f.step.nc"
        fi
    done
    echo "  [$dom] 订正完成: 成功 $ok / 失败 $fail"
done

# --- Step 4: 汇总 ---
echo ""
echo "================================================"
echo "所有任务结束。结果保存在: $WORK_DIR"
echo "delta 场保存在: $WORK_DIR/deltas/ (可留作检查)"
echo "提示: 重跑会重复叠加 delta，如需重跑请先清空 $WORK_DIR"
echo "================================================"
