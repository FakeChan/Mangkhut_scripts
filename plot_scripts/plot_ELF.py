import numpy as np
import netCDF4
import xarray as xr
import matplotlib
from scipy.spatial import cKDTree
from scipy import stats
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算地球表面两点（或数组）之间的 Haversine 距离。
    """
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def calculate_elf_alpha(x_prior_ens, y_prior_ens, x_true, y_true, obs_err_var):
    """
    鲁棒版 ELF 核心算法：增加了对极小方差和零方差的安全保护
    """
    N = x_prior_ens.shape[1]
    
    x_mean = np.mean(x_prior_ens, axis=1)
    y_mean = np.mean(y_prior_ens, axis=1)
    
    y_var = np.var(y_prior_ens, axis=1, ddof=1) 
    
    # 安全计算后验方差
    u_var = (y_var * obs_err_var) / (y_var + obs_err_var)
    
    # 安全除法：防止 y_var 为 0 时出现 NaN
    var_ratio = np.divide(u_var, y_var, out=np.zeros_like(u_var), where=(y_var!=0))
    
    A = (var_ratio - 1.0) * y_mean
    B = np.divide(u_var, obs_err_var, out=np.zeros_like(u_var), where=(obs_err_var!=0))
    
    b_hat = np.zeros_like(x_mean)
    for k in range(len(x_mean)):
        if y_var[k] == 0:
            b_hat[k] = 0.0
        else:
            cov_xy = np.cov(x_prior_ens[k, :], y_prior_ens[k, :], ddof=1)[0, 1]
            b_hat[k] = cov_xy / y_var[k]
        
    numerator_terms = b_hat * (x_true - x_mean) * (A + B * y_true)
    numerator_sum = np.sum(numerator_terms)
    
    denominator_terms = (b_hat ** 2) * (
        A**2 + 
        2 * A * B * y_true + 
        (B**2) * (obs_err_var + y_true**2)
    )
    denominator_sum = np.sum(denominator_terms)
    
    if denominator_sum == 0:
        return 0.0 
    
    alpha = numerator_sum / denominator_sum
    alpha = max(0.0, alpha)
    
    return alpha

def nc_read1(filename, var):
    """提取完整数组，避免 squeeze 带来的维度丢失问题"""
    with netCDF4.Dataset(filename, 'r') as ncfile:
        data = ncfile.variables[var][:]
        return data

def extract_obs_seq(filepath):
    """
    根据特定规则从 DART obs_seq 文件中提取经纬度和集合成员前向模拟值。
    
    规则:
    1. 遇到 'loc3d'，提取下一行的前两个浮点数作为经纬度（弧度转角度）。
    2. 遇到 'external_FO'，提取接下来 10 行的数据，每行 5 个，展平为 50 个成员的数组。
    
    返回:
    obs_lats         : ndarray, shape (K,) 观测纬度 (度)
    obs_lons         : ndarray, shape (K,) 观测经度 (度)
    y_prior_ensemble : ndarray, shape (K, 50) 集合成员先验值
    """
    print(f"正在按 [loc3d / external_FO] 规则解析文件: {filepath} ...")
    
    obs_lats = []
    obs_lons = []
    obs = []
    y_prior_list = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        
        # 规则 1：提取经纬度
        if line.startswith('loc3d'):
            idx += 1  # 移动到下一行
            parts = lines[idx].strip().split()
            if len(parts) >= 2:
                lon_rad = float(parts[0])
                lat_rad = float(parts[1])
                # 将弧度转换为角度并保存
                obs_lons.append(np.degrees(lon_rad))
                obs_lats.append(np.degrees(lat_rad))
                
        # 规则 2：提取集合成员的前向模拟值
        elif line.startswith('external_FO'):
            member_values = []
            # 往下读取 10 行
            for _ in range(10):
                idx += 1
                parts = lines[idx].strip().split()
                member_values.extend([float(x) for x in parts])
            
            # 校验是否成功提取了 50 个值
            if len(member_values) == 50:
                y_prior_list.append(member_values)
            else:
                warnings.warn(f"在第 {idx} 行附近，提取到的成员数量为 {len(member_values)}，不等于 50！")
                
        idx += 1

    K = len(obs_lats)
    print(f"解析完成！共提取 {K} 个观测位置，以及 {len(y_prior_list)} 组集合成员数据。")
    
    if K != len(y_prior_list):
        raise ValueError(f"严重错误：提取到的经纬度数量 ({K}) 与 external_FO 矩阵组数 ({len(y_prior_list)}) 不一致！请检查文件结构。")

    return np.array(obs_lats), np.array(obs_lons), np.array(y_prior_list)
# ==========================================
# 主程序：按距离分组计算 ELF 并画图
# ==========================================
# ==========================================
# 主程序：按距离分组计算 ELF 并画图
# ==========================================
if __name__ == '__main__':
    N_members = 50
    var = 'OM_S'  # 或者 'QVAPOR'
    target_level = 0
    
    prior_dir = '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_00_00'
    # prior_dir = '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/10_00_00_noinflatedOcean'
    NR_file = '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain/wrfout_d01_2018-09-10_00:00:00'
    obs_seq_file = '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir/obs_seq.out_kctest1_d01_10_00_00_LACC_ch4'
    sample_wrf_file = f'{prior_dir}/firstguess_d01.mem001'
    truth_txt_file = '/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/3obs_BT_LACC/AMSUA/BT_LACC_10_00_00/obs_d01_ch4_totalline.txt'
    
    # 提取观测元数据
    hx_lats, hx_lons, hx_ens_lists = extract_obs_seq(obs_seq_file)  
    obs_array = np.loadtxt(truth_txt_file)
    Nobs = len(hx_lats)
    
    # ---------------------------------------------------------
    # 步骤 1 & 2：读取网格、KDTree 匹配并读取真实场
    # ---------------------------------------------------------
    print("正在计算全局距离矩阵准备及读取真实场...")
    with xr.open_dataset(sample_wrf_file) as ds:
        grid_lats = ds['XLAT'].isel(Time=0).values
        grid_lons = ds['XLONG'].isel(Time=0).values
    
    NR_data = nc_read1(NR_file, var)
    if NR_data.ndim == 4:
        x_true_d03 = NR_data[0, target_level, :, :]
    elif NR_data.ndim == 3:
        x_true_d03 = NR_data[0, :, :]
    else:
        x_true_d03 = NR_data[:, :]
        
    with xr.open_dataset(NR_file) as ds_nr:
        nr_lats = ds_nr['XLAT'].isel(Time=0).values
        nr_lons = ds_nr['XLONG'].isel(Time=0).values
        
    nr_points = np.column_stack((nr_lats.ravel(), nr_lons.ravel()))
    d01_points = np.column_stack((grid_lats.ravel(), grid_lons.ravel()))
    nr_values_flat = x_true_d03.ravel()

    # 使用 'linear' (双线性插值) 替代最近邻
    x_true_flat = griddata(nr_points, nr_values_flat, d01_points, method='linear')

    # 处理插值可能产生的 NaN（比如 d01 边界稍微超出了 NR 边界）
    # 对于这些越界点，退化为最近邻插值来填补
    if np.isnan(x_true_flat).any():
        print("检测到边界 NaN，正在使用最近邻填补边界...")
        x_true_nearest = griddata(nr_points, nr_values_flat, d01_points, method='nearest')
        nan_mask = np.isnan(x_true_flat)
        x_true_flat[nan_mask] = x_true_nearest[nan_mask]

    x_true_full = x_true_flat.reshape(grid_lats.shape)

    # ---------------------------------------------------------
    # 步骤 3：一次性读取 50 个成员的先验背景场全图
    # ---------------------------------------------------------
    print("正在读取 50 个集合成员的先验场全局数据...")
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
            
    x_prior_full_arr = np.array(x_prior_full_list)  # (50, NY, NX)
    
    # ---------------------------------------------------------
    # 步骤 4：按照半径 Bin 循环 -> 观测循环 -> 汇总计算 ELF
    # ---------------------------------------------------------
    bin_edges = np.arange(0, 501, 50)  
    bin_centers = bin_edges[:-1] + 25   # 【修复6】320km的bin，中心点应该是加 160
    alpha_results = []                   # 【修复4】必须在所有循环外初始化
    
    print(f"\n开始双层聚合计算 ELF (共 {len(bin_centers)} 个距离 Bin, {Nobs} 个观测):")
    print("-" * 60)
    
    inc_x_list=[]
    err_x_list=[]
    for i in range(len(bin_centers)):
        d_min = bin_edges[i]
        d_max = bin_edges[i+1]
        
        # 直接累加公式 (12) 的分子和分母
        bin_numerator = 0.0
        bin_denominator = 0.0
        total_K_in_bin = 0 
        
        # 遍历所有的观测 (每个观测单独计算 Aj 和 Bj)
        for iobs in range(Nobs): 
            target_lat = hx_lats[iobs]
            target_lon = hx_lons[iobs]
            y_true_scalar = obs_array[iobs]
            obs_err_scalar = 0.0625
            
            # --- 【严格遵循论文】：单独计算当前观测 j 的物理属性 ---
            y_prior_single = hx_ens_lists[iobs] # 形状 (50,)
            y_mean = np.mean(y_prior_single)
            y_var = np.var(y_prior_single, ddof=0)
            
            if y_var == 0:
                continue # 观测方差为0，无法同化，跳过
                
            u_var = (y_var * obs_err_scalar) / (y_var + obs_err_scalar)
            
            # 计算观测 j 的 Aj 和 Bj (标量)
            var_ratio = u_var / y_var
            Aj = (var_ratio - 1.0) * y_mean
            Bj = u_var / obs_err_scalar
            
            # 提前算好观测 j 的期望增量部分，节省内部循环算力
            # Expected Increment = Aj + Bj * y_t
            exp_inc = Aj + Bj * y_true_scalar
            # Expected Variance term = Aj^2 + 2*Aj*Bj*y_t + Bj^2*(var_o + y_t^2)
            exp_var_term = (Aj**2) + (2 * Aj * Bj * y_true_scalar) + (Bj**2) * (obs_err_scalar + y_true_scalar**2)
            
            # -------------------------------------------------------------
            # 计算该观测到所有网格的距离，找到当前 Bin 内的配对网格
            distance_matrix = calculate_haversine_distance(grid_lats, grid_lons, target_lat, target_lon)
            mask = (distance_matrix >= d_min) & (distance_matrix < d_max)
            K_obs = np.sum(mask) 
            
            if K_obs > 0:
                # 提取配对网格的状态变量数据
                x_true_bin = x_true_full[mask]           # (K_obs,)
                x_prior_bin = x_prior_full_arr[:, mask].T # (K_obs, 50)
                
                # 高速向量化计算 x 和 y 的协方差 (替代原来低效的 for 循环 np.cov)
                x_mean_bin = np.mean(x_prior_bin, axis=1) # (K_obs,)
                
                error_x_bin=x_true_bin-x_mean_bin
                
                    
                x_anom = x_prior_bin - x_mean_bin[:, None] # (K_obs, 50)
                y_anom = y_prior_single - y_mean           # (50,)
                
                # 向量化计算 K_obs 个协方差，并除以 y_var 得到回归系数 b_hat
                cov_xy = np.sum(x_anom * y_anom, axis=1) / (N_members )
                b_hat = cov_xy / y_var # (K_obs,)
                inc_x_bin=b_hat*exp_inc
                if i==1:
                    for k in range(K_obs):
                        inc_x_list.append(inc_x_bin[k])
                        err_x_list.append(error_x_bin[k])
                
                # --- 将当前观测 j 对周围 K_obs 个网格的投影，累加到分子分母中 ---
                # 分子累加: \hat{b}_k * (x_t - \bar{x}_k) * Expected_Increment
                prior_error = x_true_bin - x_mean_bin
                prior_error_centered = prior_error
                numerator_terms = b_hat * prior_error_centered * exp_inc
                
                bin_numerator += np.sum(numerator_terms)
                # 分母累加: \hat{b}_k^2 * Expected_Variance_term
                denominator_terms = (b_hat**2) * exp_var_term
                bin_denominator += np.sum(denominator_terms)
                
                total_K_in_bin += K_obs

        # 当前距离 Bin 所有的观测都累加完毕，计算最终的 Alpha
        if total_K_in_bin == 0 or bin_denominator == 0:
            print(f"半径 [{d_min:4d} - {d_max:4d} km]: 无有效数据, 记录为 NaN")
            alpha_results.append(np.nan)
        else:
            alpha = bin_numerator / bin_denominator
            # ELF 安全截断：限制在 0 到 1.5 之间观察趋势
            # alpha = max(0.0, min(10000, alpha)) 
            alpha_results.append(alpha)
            print(f"半径 [{d_min:4d} - {d_max:4d} km]: 累积了 {total_K_in_bin:7d} 对 (y,x), Alpha = {alpha:.4f}")
            
    
    print("-" * 60)

    # ---------------------------------------------------------
    # 步骤 5：绘制并保存 ELF 随半径变化的曲线图
    # ---------------------------------------------------------
    print("\n正在生成趋势图...")
    fig,ax=plt.subplots(figsize=(10, 6))
    ax.plot(bin_centers, alpha_results, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    ax.set_title('Empirical Localization Function (ELF)', fontsize=15)
    ax.set_xlabel('Distance (km)', fontsize=13)
    ax.set_ylabel('Localization Coefficient (Alpha)', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(bin_edges)  
    ax.set_ylim(-1.5, 1.5)       
    
    out_img = f'./figs/ELF_distance_curve_{var}.png' 
    fig.savefig(out_img, dpi=300, bbox_inches='tight')
    
    fig2,ax2 = plt.subplots(figsize=(10,6))
    ax2.scatter(inc_x_list,err_x_list)
    ax2.set_xlabel('$increment of x (x_a-x_p )$')
    ax2.set_ylabel('$error of x (x_t-x_p)$')
    
    linear_reg=stats.linregress(inc_x_list,err_x_list)
    
    print(f'linear regress coeff : {linear_reg.slope}')
    out_img = f'./figs/ELF_scatter_deltas_{var}.png'
    fig2.savefig(out_img, dpi=300, bbox_inches='tight')
