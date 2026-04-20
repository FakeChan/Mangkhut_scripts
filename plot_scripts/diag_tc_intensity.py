import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ================= 配置区域 =================
# 服务器上的基础路径 (请修改为你实际的绝对路径)
base_path = "/share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA/run_wrf/10_00_00" 
folders = ['006', '043', '044', '015', '029', '037']

# 假设年份和月份，用于构建datetime对象以便在x轴上完美显示
# 请将 2023-08 替换为你实际的年份和月份
year, month = 2018, 9 
day = 10
time_strs = ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00']
domain = 'd02' # 假设分析的是 d01，如有需要请修改为 d02 或 d03

# 初始化存储字典
# 存储时间序列 (datetime objects)
times = [datetime(year, month, day, int(t.split(':')[0]), int(t.split(':')[1])) for t in time_strs]

# 要诊断的变量
diagnostics = {
    'MU_min': 'Minimum MU (Pa)',
    'PSFC_min': 'Minimum Surface Pressure (Pa)',
    'W_UP_MAX': 'Max Updraft Velocity (m/s)',
    'WSPD10MAX': 'Max 10m Wind Speed (m/s)'
}

results = {folder: {var: [] for var in diagnostics.keys()} for folder in folders}

# ================= 数据提取 =================
print("开始提取数据...")
for folder in folders:
    for t_str in time_strs:
        # 构建文件名，例如: wrfout_d01_2023-08-10_00:00:00
        filename = f"wrfout_{domain}_{year:04d}-{month:02d}-{day:02d}_{t_str}:00"
        filepath = os.path.join(base_path, folder, filename)
        
        if not os.path.exists(filepath):
            print(f"警告: 找不到文件 {filepath}，用 NaN 填充。")
            for var in diagnostics.keys():
                results[folder][var].append(np.nan)
            continue
            
        try:
            with nc.Dataset(filepath, 'r') as ds:
                # 1. 提取 MU 的极小值
                # MU 是二维/三维变量，极小值通常在台风中心
                mu = ds.variables['MU'][0, :, :]
                results[folder]['MU_min'].append(np.min(mu))
                
                # 2. 提取 PSFC 的极小值 (表面气压极小值)
                psfc = ds.variables['PSFC'][0, :, :]
                results[folder]['PSFC_min'].append(np.min(psfc))
                
                # 3. 提取最大上升气流 (来自 nwp_diagnostics)
                if 'W_UP_MAX' in ds.variables:
                    w_up_max = ds.variables['W_UP_MAX'][0, :, :]
                    results[folder]['W_UP_MAX'].append(np.max(w_up_max))
                else:
                    results[folder]['W_UP_MAX'].append(np.nan)
                    
                # 4. 提取最大10米风速 (来自 nwp_diagnostics)
                if 'WSPD10MAX' in ds.variables:
                    wspd10max = ds.variables['WSPD10MAX'][0, :, :]
                    results[folder]['WSPD10MAX'].append(np.max(wspd10max))
                else:
                    results[folder]['WSPD10MAX'].append(np.nan)
                    
        except Exception as e:
            print(f"读取 {filepath} 时出错: {e}")
            for var in diagnostics.keys():
                results[folder][var].append(np.nan)

print("数据提取完成，正在绘制趋势图...")

# ================= 绘制趋势图 =================
fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
axs = axs.flatten()

# 线条样式设置 (颜色和标记)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'v', '*']

for i, (var_key, var_name) in enumerate(diagnostics.items()):
    ax = axs[i]
    for j, folder in enumerate(folders):
        ax.plot(times, results[folder][var_key], 
                label=f'Exp {folder}', 
                color=colors[j], 
                marker=markers[j], 
                linewidth=2, markersize=6)
    
    ax.set_title(var_name, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (UTC)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 格式化 x 轴时间显示
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    
    # 对于气压和 MU，极小值越小说明台风越强，我们反转 y 轴以便于直观理解 (线往上走代表增强)
    if 'min' in var_key:
        # ax.invert_yaxis()
        ax.set_ylabel(f'{var_name} (Inverted)', fontsize=12)
    else:
        ax.set_ylabel(var_name, fontsize=12)

# 添加统一图例
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
           ncol=6, fontsize=12, frameon=False)

plt.tight_layout()
plt.savefig("./figs/tc_diagnostics_spindown.png", bbox_inches='tight')