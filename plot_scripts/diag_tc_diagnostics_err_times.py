import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from wrf import getvar, latlon_coords
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.lines as mlines
# ==========================================
# 1. 基础诊断函数 (保持原样)
# ==========================================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def calc_typhoon_mslp(ncfile, timeidx=0):
    slp = getvar(ncfile, "slp", timeidx=timeidx)
    mslp = np.min(slp.values)
    center_idx = np.unravel_index(np.argmin(slp.values, axis=None), slp.shape)
    return mslp, center_idx

def calc_typhoon_mws(ncfile, timeidx=0):
    u10 = getvar(ncfile, "U10", timeidx=timeidx)
    v10 = getvar(ncfile, "V10", timeidx=timeidx)
    wspd10 = np.sqrt(u10**2 + v10**2)
    mws = np.max(wspd10.values)
    return mws, wspd10

def calc_typhoon_rmw(ncfile, wspd10, center_idx, timeidx=0):
    lats, lons = latlon_coords(wspd10)
    center_lat = lats.values[center_idx]
    center_lon = lons.values[center_idx]
    max_wind_idx = np.unravel_index(np.argmax(wspd10.values, axis=None), wspd10.shape)
    max_wind_lat = lats.values[max_wind_idx]
    max_wind_lon = lons.values[max_wind_idx]
    return haversine_distance(center_lat, center_lon, max_wind_lat, max_wind_lon)

def calc_typhoon_r17(ncfile, wspd10, center_idx, timeidx=0, search_radius_km=600):
    lats, lons = latlon_coords(wspd10)
    center_lat = lats.values[center_idx]
    center_lon = lons.values[center_idx]
    dist_matrix = haversine_distance(center_lat, center_lon, lats.values, lons.values)
    valid_mask = dist_matrix <= search_radius_km
    r17_points_dist = dist_matrix[(wspd10.values >= 17.0) & valid_mask]
    if len(r17_points_dist) == 0: return 0.0, 0.0 
    r17_max = np.max(r17_points_dist)
    contour_points = dist_matrix[(wspd10.values >= 16.5) & (wspd10.values <= 17.5) & valid_mask]
    r17_mean = np.mean(contour_points) if len(contour_points) > 0 else r17_max
    return r17_max, r17_mean

def get_all_diagnostics(ncfile, timeidx=0):
    mslp, center_idx = calc_typhoon_mslp(ncfile, timeidx)
    mws, wspd10 = calc_typhoon_mws(ncfile, timeidx)
    rmw = calc_typhoon_rmw(ncfile, wspd10, center_idx, timeidx)
    _, r17_mean = calc_typhoon_r17(ncfile, wspd10, center_idx, timeidx)
    return mslp, mws, rmw, r17_mean

# ==========================================
# 2. 时序可视化绘图函数
# ==========================================
def plot_timeseries_diagnostics(df, var_name, ylabel, save_path, exp_colors, exp_linestyles, member_markers):
    """
    绘制指定变量在时间序列上的 RMSE 箱须图和成员轨迹线 (区分成员形状)
    """
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    sns.set_theme(style="whitegrid")
    
    times = sorted(df['Time_Str'].unique())
    time_mapping = {t: i for i, t in enumerate(times)}
    experiments = list(exp_colors.keys())
    n_exps = len(experiments)
    
    # 计算 X 轴偏移量，对齐箱须图
    box_width = 0.8
    offsets = {}
    for i, exp in enumerate(experiments):
        offsets[exp] = (i - (n_exps - 1) / 2) * (box_width / n_exps)

    # 1. 绘制箱须图
    sns.boxplot(
        data=df, x='Time_Str', y=var_name, hue='Experiment', 
        palette=exp_colors, ax=ax, width=box_width, 
        boxprops=dict(alpha=0.3), showfliers=False # 稍微调低箱子的透明度，突出散点
    )

    # 2. 绘制散点和每个成员的连接线
    for exp in experiments:
        exp_data = df[df['Experiment'] == exp]
        members = exp_data['Member'].unique()
        
        for mem in members:
            mem_data = exp_data[exp_data['Member'] == mem].sort_values('Time_Obj')
            if mem_data.empty: continue
            
            x_coords = [time_mapping[t] + offsets[exp] for t in mem_data['Time_Str']]
            y_coords = mem_data[var_name].values
            
            # 根据成员编号获取对应的形状，如果没有定义则默认用圆圈 'o'
            marker_style = member_markers.get(mem, 'o')
            
            ax.plot(x_coords, y_coords, 
                    color=exp_colors[exp], 
                    linestyle=exp_linestyles[exp], 
                    alpha=0.7, linewidth=1.2, 
                    marker=marker_style, markersize=7) # 稍微调大了 marker 以便看清形状

    # 图表修饰
    ax.set_title(f"Time Series of {var_name.upper()} Absolute Error (vs NR)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Time (MM-DD HH:MM)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    plt.xticks(rotation=45)
    
    # ---------------- 3. 构建双图例系统 ----------------
    
    # (A) 实验图例 (颜色和线型)
    # 取消 Seaborn 自动生成的图例，我们要自己画
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        
    exp_handles = []
    for exp in experiments:
        h = mlines.Line2D([], [], color=exp_colors[exp], linestyle=exp_linestyles[exp], 
                          linewidth=2, label=exp)
        exp_handles.append(h)
    
    leg_exp = ax.legend(handles=exp_handles, title="Experiments (Color/Line)", 
                        loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(leg_exp) # 把第一个图例“固定”在画布上，防止被第二个覆盖
    
    # (B) 成员图例 (形状)
    mem_handles = []
    # 只为当前数据集中实际存在的成员生成图例
    present_members = sorted(df['Member'].unique())
    for mem in present_members:
        mk = member_markers.get(mem, 'o')
        # 用灰色显示形状图例，避免与实验颜色混淆
        h = mlines.Line2D([], [], color='gray', marker=mk, linestyle='None', 
                          markersize=8, label=f'Mem {mem:03d}')
        mem_handles.append(h)
        
    # 添加第二个图例 (稍微往下放一点，y=0.7)
    ax.legend(handles=mem_handles, title="Ensemble Members (Shape)", 
              loc='upper left', bbox_to_anchor=(1.02, 0.7))
    
    # ---------------------------------------------------
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"[{var_name}] 时间序列对比图已保存至: {save_path}")
    plt.close()


# ==========================================
# 3. 主程序：批量读取与处理
# ==========================================
if __name__ == '__main__':
    # -------- 实验参数配置 --------
    # 定义分析的时间窗口和间隔
    start_time = datetime(2018, 9, 10, 0, 0) # 起始时间 (示例: 00:00)
    end_time   = datetime(2018, 9, 10, 6, 0) # 结束时间 (示例: 06:00)
    interval   = timedelta(minutes=30)       # 时间间隔
    
    nr_dir = "/scratch/lililei1/kcfu/tc_mangkhut/NR" # NR 文件夹
    filter_kind = "EAKF"
    # 实验字典: {实验名称: 实验的主路径}
    exp_dirs = {
        "Exp_oceanAssim0Run0": "/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim0Run0",
        "Exp_oceanAssim0Run1": "/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim0Run1",
        "Exp_oceanAssim1Run1": "/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim1Run1",
    }
    
    member_indices = [6,15,29,37,43,44]
    output_dir = './figs'
    os.makedirs(output_dir, exist_ok=True)
    
    # 为不同实验配置颜色和线型
    exp_colors = {
    "Exp_oceanAssim0Run0": "goldenrod",       
    "Exp_oceanAssim0Run1": "dodgerblue",    
    "Exp_oceanAssim1Run1": "crimson",       
    }
    
    
    exp_linestyles = {
    "Exp_oceanAssim0Run0": "-",                  # 实线 (Solid) - 通常给 Control 实验
    "Exp_oceanAssim0Run1": "--",                 # 虚线 (Dashed)
    "Exp_oceanAssim1Run1": "-.",                 # 点划线 (Dash-dot)
    }
    
    member_markers = {
        6:  'o', 
        15: 's', 
        29: '^', 
        37: 'D', 
        43: 'v', 
        44: 'p'
    }
    # 初始化数据存储列表
    records = []

    # -------- 开始时间循环提取数据 --------
    curr_time = start_time
    while curr_time <= end_time:
        time_suffix = curr_time.strftime("%Y-%m-%d_%H:%M:%S")
        display_time = curr_time.strftime("%m-%d %H:%M") # 用于图表横坐标显示
        print(f"\n>>> 正在处理时间步: {time_suffix}")
        
        # 1. 读取该时刻的 NR 真值
        nr_filepath = f"{nr_dir}/wrfout_d03_{time_suffix}"
        try:
            with Dataset(nr_filepath) as nc_nr:
                nr_mslp, nr_mws, nr_rmw, nr_r17 = get_all_diagnostics(nc_nr, timeidx=0)
        except Exception as e:
            print(f"  [跳过] 无法读取 NR 文件 {nr_filepath}: {e}")
            curr_time += interval
            continue # 如果没有 NR，就算不出 Error，跳过此时次

        # 2. 读取各个实验的集合成员
        for exp_name, exp_base_path in exp_dirs.items():
            for mem_idx in member_indices:
                # 构建成员文件路径 (根据你之前的格式调整)
                mem_str = f"{mem_idx:03d}"
                # 假设文件存放在形如 /path/to/EAKF/mem001/wrfout_d02_... 的目录中
                ens_filepath = f"{exp_base_path}/{filter_kind}/{mem_str}/wrfout_d02_{time_suffix}" 
                
                
                try:
                    with Dataset(ens_filepath) as nc_ens:
                        ens_mslp, ens_mws, ens_rmw, ens_r17 = get_all_diagnostics(nc_ens, timeidx=0)
                        
                        # 计算绝对误差 (|Ens - NR|)，评估单点 RMSE/Magnitude
                        # 存入列表
                        records.append({
                            'Time_Obj': curr_time,
                            'Time_Str': display_time,
                            'Experiment': exp_name,
                            'Member': mem_idx,
                            'mslp': ens_mslp - nr_mslp,
                            'mws': ens_mws - nr_mws,
                            'rmw': ens_rmw - nr_rmw,
                            'r17': ens_r17 - nr_r17
                        })
                except Exception as e:
                    print(f"file do not exist")

        # 前进到下一个 30 分钟
        curr_time += interval

    # -------- 数据转化与绘图 --------
    print("\n数据提取完毕，正在生成可视化图像...")
    df = pd.DataFrame(records)
    
    if df.empty:
        print("未提取到任何有效数据，请检查路径。")
    else:
        plot_configs = [
            ('mslp', 'Absolute Error (hPa)'),
            ('mws', 'Absolute Error (m/s)'),
            ('rmw', 'Absolute Error (km)'),
            ('r17', 'Absolute Error (km)')
        ]
        
        for var_key, ylabel in plot_configs:
            save_path = os.path.join(output_dir, f'{filter_kind}_ts_error_{var_key}.png')
            plot_timeseries_diagnostics(
                df=df, 
                var_name=var_key, 
                ylabel=ylabel, 
                save_path=save_path,
                exp_colors=exp_colors,
                exp_linestyles=exp_linestyles,
                member_markers=member_markers 
            )
    