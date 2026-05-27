#!/bin/bash

# =======================================================
# WRF 集合成员 OM_TMP 变量协方差膨胀脚本 (NCO 安全版)
# =======================================================

# 1. 参数设置
WORK_DIR="."
LAMBDA=5              # 膨胀系数 (例如 1.20 代表放大 20% 离散度)
NUM_MEMBERS=50           # 集合成员数量
VAR_NAME="OM_TMP"        # 需要膨胀的变量名
domain=d01

# 计算第二个权重 (1 - lambda)
# 使用 awk 确保高精度的浮点数计算
WEIGHT_2=$(awk -v l="$LAMBDA" 'BEGIN {printf "%.4f", 1.0 - l}')

cd "$WORK_DIR" || exit 1

echo "==================================================="
echo "开始执行 NCO 集合膨胀"
echo "目标变量: ${VAR_NAME}"
echo "膨胀系数 (Lambda): ${LAMBDA}"
echo "平均场权重 (1-Lambda): ${WEIGHT_2}"
echo "==================================================="

# 2. 检查 NCO 环境是否正常
# 确保模块已加载且命令可执行，避免 ncks 出现异常损坏原始数据
if ! command -v ncks &> /dev/null || ! command -v ncflint &> /dev/null || ! command -v nces &> /dev/null; then
    echo "❌ 致命错误: 无法调用 NCO 命令。"
    echo "请检查你的模块环境 (例如执行 module load nco) 或环境变量 PATH。"
    exit 1
fi

# 3. 构建成员文件列表
FILE_LIST=""
for i in $(seq -w 1 "$NUM_MEMBERS"); do
    FILE_LIST="$FILE_LIST firstguess_${domain}.mem0${i}"
done

# 4. 计算指定变量的集合平均 (Ensemble Mean)
# -O 覆盖输出, -v 仅提取目标变量，大幅节省计算和 I/O 时间
echo ">> 正在计算集合平均场..."
nces -O -v ${VAR_NAME} ${FILE_LIST} ens_mean_${VAR_NAME}.nc

if [ $? -ne 0 ]; then
    echo "❌ 集合平均计算失败，请检查 firstguess.mem* 文件是否存在。"
    exit 1
fi

# 5. 循环对每个成员执行膨胀操作
echo ">> 正在对 ${NUM_MEMBERS} 个成员执行膨胀操作..."
for i in $(seq -w 1 "$NUM_MEMBERS"); do
    MEM_FILE="firstguess_${domain}.mem0${i}"
    TMP_FILE="tmp_inflated_${i}.nc"

    # Step A: 核心推导实现 x_new = lambda * x + (1-lambda) * mean
    # ncflint 的 -w 参数指定前后两个文件的权重
    ncflint -O -v ${VAR_NAME} -w ${LAMBDA},${WEIGHT_2} ${MEM_FILE} ens_mean_${VAR_NAME}.nc ${TMP_FILE}

    # Step B: 将膨胀后更新的变量追加/覆写回原成员文件
    # -A (Append) 模式只会覆盖对应的变量数据，原文件的其他变量和全局属性完全不受影响
    ncks -A -v ${VAR_NAME} ${TMP_FILE} ${MEM_FILE}

    echo "   已更新成员: ${MEM_FILE}"

    # 清理当前成员的临时文件
    rm -f ${TMP_FILE}
done

# 6. 最终清理
rm -f ens_mean_${VAR_NAME}.nc

echo "==================================================="
echo "✅ 所有成员的 ${VAR_NAME} 变量膨胀处理完成，文件属性已安全保留！"
