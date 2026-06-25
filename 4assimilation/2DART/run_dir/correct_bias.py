import os
import shutil
import netCDF4
import numpy as np
import sys
import glob
import re

# ==============================================================================
# 1. 用户配置区域
# ==============================================================================

# 列表文件匹配模式
# 脚本会自动寻找当前目录下所有符合此模式的文件
# 例如: wrfout_list_d01.txt, wrfout_list_d02.txt, wrfout_list_d03.txt ...
LIST_FILE_PATTERN = './wrfout_list_d*.txt'

# 唯一的真值文件
# 脚本将计算这个文件中 VAR_NAME 的平均值作为所有 Domain 的目标参考
TRUTH_FILE = '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d03_2018-09-10_00:00:00'

# 工作目录
# 优先从环境变量读取，如果没有设置，则默认使用当前目录下的 'corrected_output'
WORK_DIR = os.environ.get('ensmem_dir')
if not WORK_DIR:
    WORK_DIR = '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_00_00'
    print(f"[Warning] 环境变量 'ensmem_dir' 未设置，将使用默认目录: {WORK_DIR}")

# 变量名
VAR_NAME = 'OM_TMP'

# ==============================================================================
# 2. 功能函数
# ==============================================================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"创建工作目录: {directory}")
        except OSError as e:
            print(f"[Error] 无法创建目录 {directory}: {e}")
            sys.exit(1)

def read_file_list(list_file):
    if not os.path.exists(list_file):
        print(f"错误: 列表文件不存在 -> {list_file}")
        return []
    with open(list_file, 'r') as f:
        # 过滤空行和注释
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return lines

def copy_files(file_paths, target_dir):
    new_paths = []
    print(f"  -> 正在拷贝 {len(file_paths)} 个文件到工作目录...")
    for src in file_paths:
        if not os.path.exists(src):
            print(f"     [Skip] 源文件不存在: {src}")
            continue
        filename = os.path.basename(src)
        dst = os.path.join(target_dir, filename)
        
        # 仅当目标不存在或源文件更新时才拷贝（提高重复运行效率）
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            shutil.copy2(src, dst)
        
        new_paths.append(dst)
    return new_paths

def get_vertical_profile_mean(file_path, var_name):
    """
    计算垂直廓线的空间平均值。
    """
    try:
        with netCDF4.Dataset(file_path, 'r') as nc:
            if var_name not in nc.variables:
                print(f"  [Warning] 变量 {var_name} 不在文件中: {os.path.basename(file_path)}")
                return None
            
            data = nc.variables[var_name][:]
            
            # 数据清洗：Mask 掉陆地或无效值
            data = np.ma.masked_less(data, 270.0)

            # --- 维度判断与平均计算 ---
            if data.ndim == 4: # (Time, Z, Y, X)
                return np.ma.mean(data, axis=(0, 2, 3))
            elif data.ndim == 3: # (Time, Y, X) 单层
                scalar_mean = np.ma.mean(data)
                return np.array([scalar_mean])
            else:
                print(f"  [Warning] 变量维度异常 ({data.ndim})，跳过: {file_path}")
                return None

    except Exception as e:
        print(f"  [Error] 读取文件出错 {file_path}: {e}")
        return None

def apply_profile_correction(member_files, bias_profile, var_name):
    """
    将计算好的 bias_profile 加到成员文件的每一层上
    """
    print(f"  -> 正在应用垂直廓线订正 (共 {len(member_files)} 个文件)...")
    
    count = 0
    for f_path in member_files:
        try:
            with netCDF4.Dataset(f_path, 'r+') as nc:
                if var_name not in nc.variables:
                    continue
                
                var = nc.variables[var_name]
                data = var[:] 
                
                if data.ndim == 4:
                    n_levels = data.shape[1]
                    if len(bias_profile) != n_levels:
                        print(f"     [Error] {os.path.basename(f_path)} 层数不匹配! 文件:{n_levels}, Bias:{len(bias_profile)}")
                        continue
                    
                    # Broadcasting: (Z,) -> (1, Z, 1, 1)
                    bias_broadcast = bias_profile[None, :, None, None]
                    var[:] = data + bias_broadcast
                    
                elif data.ndim == 3:
                    var[:] = data + bias_profile[0]
                
                count += 1
        except Exception as e:
            print(f"     [Error] 写入文件失败 {f_path}: {e}")

    print(f"  -> 订正完成 (成功处理 {count}/{len(member_files)} 个)。")

def process_domain_group(list_file, truth_profile, var_name):
    """
    处理单个 Domain 列表的核心逻辑
    """
    # 从文件名提取 Domain ID (例如 'd01') 用于显示
    # 假设文件名格式包含 'd数字'
    match = re.search(r'(d\d+)', os.path.basename(list_file))
    domain_id = match.group(1).upper() if match else "UNKNOWN_DOMAIN"
    
    print(f"\n[{domain_id}] 正在处理列表: {list_file}")
    
    # 1. 读取列表并拷贝文件
    member_paths_raw = read_file_list(list_file)
    if not member_paths_raw:
        print(f"  [Skip] 列表为空或无法读取，跳过。")
        return

    local_members = copy_files(member_paths_raw, WORK_DIR)
    if not local_members:
        print(f"  [Skip] 没有成功拷贝任何文件，跳过。")
        return

    # 2. 计算集合的垂直廓线平均 (Ensemble Profile)
    # 读取第一个文件确定层数
    first_profile = get_vertical_profile_mean(local_members[0], var_name)
    if first_profile is None:
        print("  [Error] 无法读取第一个成员的变量信息，该组处理终止。")
        return
        
    num_levels = len(first_profile)
    print(f"  -> 检测到垂直层数: {num_levels}")
    
    # 验证与真值层数的一致性
    if len(truth_profile) != num_levels:
        print(f"  [Fatal Error] 真值层数 ({len(truth_profile)}) 与当前Domain层数 ({num_levels}) 不一致！")
        print("  -> 无法订正，跳过此 Domain。")
        return

    # 累加计算平均
    sum_profile = np.zeros(num_levels)
    valid_count = 0
    
    # 为了进度条不过于冗长，只打印关键信息
    print("  -> 正在计算集合平均廓线...")
    for f in local_members:
        p = get_vertical_profile_mean(f, var_name)
        if p is not None and len(p) == num_levels:
            sum_profile += p
            valid_count += 1
            
    if valid_count == 0:
        print("  [Error] 有效成员数为0，无法计算平均。")
        return
        
    ens_profile = sum_profile / valid_count
    
    # 3. 计算偏差并订正
    # Bias = Truth - Ensemble
    bias_profile = truth_profile - ens_profile
    
    print(f"  -> 偏差计算完成。Layer 0 Bias: {bias_profile[0]:.4f}")
    
    apply_profile_correction(local_members, bias_profile, var_name)


# ==============================================================================
# 3. 主程序
# ==============================================================================

if __name__ == "__main__":
    print("================================================")
    print("WRF Bias Correction Tool (Multi-Domain Robust)")
    print("================================================")
    
    ensure_dir(WORK_DIR)
    
    # --- Step 1: 读取真值廓线 ---
    print("\nStep 1: 读取真值 (Reference) 廓线")
    if not os.path.exists(TRUTH_FILE):
        print(f"[Fatal] 真值文件不存在: {TRUTH_FILE}")
        sys.exit(1)
        
    truth_profile = get_vertical_profile_mean(TRUTH_FILE, VAR_NAME)
    if truth_profile is None:
        print("[Fatal] 读取真值变量失败")
        sys.exit(1)
    
    print(f"真值读取成功 | 层数: {len(truth_profile)} | 来源: {os.path.basename(TRUTH_FILE)}")

    # --- Step 2: 动态搜索并处理所有 Domain ---
    print("\nStep 2: 搜索文件列表")
    list_files = sorted(glob.glob(LIST_FILE_PATTERN))
    
    if not list_files:
        print(f"[Warning] 未找到任何匹配 '{LIST_FILE_PATTERN}' 的列表文件！")
        print("请确认 txt 文件已放置在脚本运行目录下。")
    else:
        print(f"发现 {len(list_files)} 个任务列表: {[os.path.basename(f) for f in list_files]}")
        
        for list_file in list_files:
            process_domain_group(list_file, truth_profile, VAR_NAME)
    
    print("\n================================================")
    print(f"所有任务结束。结果保存在: {WORK_DIR}")
