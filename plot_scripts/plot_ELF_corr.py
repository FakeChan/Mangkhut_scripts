import numpy as np
import netCDF4
import xarray as xr
import matplotlib
from scipy.spatial import cKDTree
import warnings
# 设置无头模式，防止在没有GUI的服务器上画图报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """计算地球表面两点（或数组）之间的 Haversine 距离"""
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def nc_read1(filename, var):
    """提取完整数组"""
    with netCDF4.Dataset(filename, 'r') as ncfile:
        return ncfile.variables[var][:]

def extract_obs_seq(filepath):
    """提取 obs_seq 的经纬度和前向模拟值"""
    print(f"正在解析文件: {filepath} ...")
    obs_lats, obs_lons, y_prior_list = [], [], []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith('loc3d'):
            idx += 1 
            parts = lines[idx].strip().split()
            if len(parts) >= 2:
                obs_lons.append(np.degrees(float(parts[0])))
                obs_lats.append(np.degrees(float(parts[1])))
                
        elif line.startswith('external_FO'):
            member_values = []
            for _ in range(10):
                idx += 1
                member_values.extend([float(x) for x in lines[idx].strip().split()])
            if len(member_values) == 50:
                y_prior_list.append(member_values)
            else:
                warnings.warn(f"第 {idx} 行附近提取成员数量异常！")
        idx += 1

    return np.array(obs_lats), np.array(obs_lons), np.array(y_prior_list)

# ==========================================
# 主程序：计算相关系数随距离的变化
# ==========================================
if __name__ == '__main__':
    N_members = 50
    var = 'QVAPOR'
    target_level = 15  
    
    # --- 路径配置 (请确保与你的环境一致) ---
    prior_dir = '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_00_00'
    NR_file = '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d01_2018-09-10_00:00:00'
    obs_seq_file = '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir/obs_seq.out_kctest1_d01_10_00_00_quantile_allsky'
    sample_wrf_file = f'{prior_dir}/firstguess_d01.mem001'
    
    # 提取观测元数据
    hx_lats, hx_lons, hx_ens_lists = extract_obs_seq(obs_seq_file)  
    Nobs = len(hx_lats)
    
    # ---------------------------------------------------------
    # 读取网格信息
    # ---------------------------------------------------------
    print("正在读取网格信息...")
    with xr.open_dataset(sample_wrf_file) as ds:
        grid_lats = ds['XLAT'].isel(Time=0).values
        grid_lons = ds['XLONG'].isel(Time=0).values

    # ---------------------------------------------------------
    # 一次性读取 50 个成员的先验背景场全图
    # ---------------------------------------------------------
    print(f"正在读取 50 个集合成员的先验场全局数据 ({var})...")
    x_prior_full_list = []
    for imem in range(1, N_members + 1):
        mem = f'{imem:03d}'
        ens_file = f'{prior_dir}/firstguess_d01.mem{mem}'
        data = nc_read1(ens_file, var)
        if data.ndim == 4:
            x_prior_full_list.append(data[0, target_level, :, :])
        elif data.ndim == 3:
            x_prior_full_list.append(data[0, :, :])
        else:
            x_prior_full_list.append(data[:, :])
            
    x_prior_full_arr = np.array(x_prior_full_list)  # 形状: (50, NY, NX)
    
    # ---------------------------------------------------------
    # 核心计算：双层循环按距离分组计算相关系数
    # ---------------------------------------------------------
    bin_edges = np.arange(0, 1001, 50)  
    bin_centers = bin_edges[:-1] + 25   
    
    mean_corr_results = []
    var_corr_results = []
    
    print(f"\n开始双层聚合计算相关系数 (共 {len(bin_centers)} 个距离 Bin, {Nobs} 个观测):")
    print("-" * 60)
    
    for i in range(len(bin_centers)):
        d_min = bin_edges[i]
        d_max = bin_edges[i+1]
        
        # 用于存储当前 Bin 内所有观测对的皮尔逊相关系数
        correlations_in_bin = []
        total_K_in_bin = 0 
        
        # 遍历所有的观测
        for iobs in range(Nobs): 
            target_lat = hx_lats[iobs]
            target_lon = hx_lons[iobs]
            
            # 当前观测的50个成员模拟值
            y_ens = hx_ens_lists[iobs] # 形状 (50,)
            y_mean = np.mean(y_ens)
            y_anom = y_ens - y_mean
            y_var_sum = np.sum(y_anom**2)
            
            if y_var_sum == 0:
                continue # 如果观测自身没有集合离散度，跳过
                
            # 计算距离并抠出网格
            distance_matrix = calculate_haversine_distance(grid_lats, grid_lons, target_lat, target_lon)
            mask = (distance_matrix >= d_min) & (distance_matrix < d_max)
            K_obs = np.sum(mask) 
            
            if K_obs > 0:
                # 提取对应网格状态变量的50个成员，并转置为 (K_obs, 50)
                x_ens_bin = x_prior_full_arr[:, mask].T 
                
                # --- 高速向量化计算皮尔逊相关系数 R ---
                x_mean_bin = np.mean(x_ens_bin, axis=1, keepdims=True) # (K_obs, 1)
                x_anom = x_ens_bin - x_mean_bin                        # (K_obs, 50)
                x_var_sum = np.sum(x_anom**2, axis=1)                  # (K_obs,)
                
                # 过滤掉方差为 0 的网格（如纯陆地无海温扰动点），防止除以 0
                valid_mask = x_var_sum > 0 
                
                if np.any(valid_mask):
                    # 协方差分子 (这里因为上下都有 N-1，所以可以约掉，直接用平方和计算)
                    numerator = np.sum(x_anom[valid_mask] * y_anom, axis=1)
                    # 离差平方和的几何平均分母
                    denominator = np.sqrt(x_var_sum[valid_mask] * y_var_sum)
                    
                    # 得到这批有效网格的相关系数，并转换为 Python 列表
                    R_values = numerator / denominator
                    correlations_in_bin.extend(R_values.tolist())
                    
                total_K_in_bin += K_obs

        # 计算并保存当前距离 Bin 的统计特征
        if len(correlations_in_bin) == 0:
            print(f"半径 [{d_min:4d} - {d_max:4d} km]: 无有效数据, 记录为 NaN")
            mean_corr_results.append(np.nan)
            var_corr_results.append(np.nan)
        else:
            mean_R = np.mean(correlations_in_bin)
            var_R = np.var(correlations_in_bin)
            mean_corr_results.append(mean_R)
            var_corr_results.append(var_R)
            print(f"半径 [{d_min:4d} - {d_max:4d} km]: 累积了 {total_K_in_bin:7d} 对, 平均 R = {mean_R:+.4f}, 方差 = {var_R:.4f}")
            
    print("-" * 60)

    # ---------------------------------------------------------
    # 绘制带方差阴影的相关系数衰减曲线图
    # ---------------------------------------------------------
    mean_corr_arr = np.array(mean_corr_results)
    var_corr_arr = np.array(var_corr_results)
    var_corr_arr = np.sqrt(var_corr_arr)
    print("\n正在生成趋势图...")
    plt.figure(figsize=(12, 7))
    
    # 绘制中心平均线
    plt.plot(bin_centers, mean_corr_arr, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, markersize=7, label='Mean Correlation')
    
    # 绘制方差阴影
    # (注：根据要求此处绘制了平均值 ± 方差。在统计学绘图中，有时也会使用 ± 标准差 np.sqrt(var) 来表示 1σ 区间)
    plt.fill_between(bin_centers, 
                     mean_corr_arr - var_corr_arr, 
                     mean_corr_arr + var_corr_arr, 
                     color='#1f77b4', alpha=0.25, label='$\pm \sigma$ ')
    
    # 绘制 y=0 的基准线
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    
    plt.title(f'Spatial Correlation between $H(x)$ and {var}', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Distance from Observation (km)', fontsize=14)
    plt.ylabel('Pearson Correlation Coefficient ($R$)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.8)
    plt.xticks(bin_edges[::2]) # 稍微稀疏一点刻度防止拥挤
    plt.ylim(-1.0, 1.0) # 相关系数的理论极值
    plt.legend(fontsize=12, loc='upper right')
    
    out_img = f'./figs/Correlation_distance_curve_{var}.png' 
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"画图完成！图片已保存为: {out_img}")