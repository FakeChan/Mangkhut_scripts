# 1. 获取 output 文件中的所有变量名，并格式化为逗号分隔的列表
#VARS_IN_OUTPUT=$(ncks -m output_mean.nc | grep 'type = ' | cut -d':' -f1 | tr '\n' ',' | sed 's/,$//')

# 2. 从 firstguess 追加到 output，但排除 (-x) 上一步中找到的所有变量
#ncks -A -x -v $VARS_IN_OUTPUT firstguess.mem001 output_mean.nc

# 1. 获取变量列表，同时抑制 ncks -m 可能的警告 (stderr)
VARS_IN_OUTPUT=$(ncks -m output_mean.nc 2>/dev/null | grep 'type = ' | cut -d':' -f1 | tr '\n' ',' | sed 's/,$//')

# 2. 执行追加操作，抑制标准输出 (stdout)
ncks -A -x -v $VARS_IN_OUTPUT firstguess.mem001 output_mean.nc > /dev/null
