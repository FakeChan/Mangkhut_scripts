import numpy as np
import netCDF4
import xarray as xr
import matplotlib
from scipy.spatial import cKDTree
import warnings
# 设置无头模式
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def nc_read1(filename, var):
    with netCDF4.Dataset(filename, 'r') as ncfile:
        return ncfile.variables[var][:]

def extract_obs_seq(filepath):
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
# 主程序：计算垂直 ELF (Alpha vs Model Level)
# ==========================================
if __name__ == '__main__':
    N_members = 50
    var = 'QVAPOR'  # 替换为你要测算的变量，如 'QVAPOR' (大气) 或 'OM_TMP' (海洋)
    
    # 【核心参数】水平局地化截断半径。只计算落在这个半径内的网格点的垂直廓线。
    # 建议设置在 30-60km 左右，代表观测点附近的网格柱
    horizontal_cutoff_km = 40.0 
    
    prior_dir = '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_00_00'
    NR_file = '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d01_2018-09-10_00:00:00'
    obs_seq_file = '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir/obs_seq.out_kctest1_d01_10_00_00_quantile'
    sample_wrf_file = f'{prior_dir}/firstguess_d01.mem001'
    truth_txt_file = '/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/3obs_BT/AMSUA/BT_10_00_00/obs_d01_ch4_totalline.txt'
    
    # 提取观测元数据
    hx_lats, hx_lons, hx_ens_lists = extract_obs_seq(obs_seq_file)  
    obs_array = np.loadtxt(truth_txt_file)
    Nobs = len(hx_lats)
    
    # ---------------------------------------------------------
    # 步骤 1：读取水平网格并准备 KDTree
    # ---------------------------------------------------------
    print("正在计算全局距离矩阵准备及读取真实场...")
    with xr.open_dataset(sample_wrf_file) as ds:
        grid_lats = ds['XLAT'].isel(Time=0).values
        grid_lons = ds['XLONG'].isel(Time=0).values
        NY, NX = grid_lats.shape
        
    with xr.open_dataset(NR_file) as ds_nr:
        nr_lats = ds_nr['XLAT'].isel(Time=0).values
        nr_lons = ds_nr['XLONG'].isel(Time=0).values
        
    print("正在构建 KDTree 进行网格匹配计算...")
    nr_points = np.column_stack((nr_lats.ravel(), nr_lons.ravel()))
    tree = cKDTree(nr_points)
    d01_points = np.column_stack((grid_lats.ravel(), grid_lons.ravel()))
    _, indices = tree.query(d01_points)

    # ---------------------------------------------------------
    # 步骤 2：读取 Nature Run 三维全场并逐层插值
    # ---------------------------------------------------------
    NR_data = nc_read1(NR_file, var)
    # 取出 3D 数据 (NZ, NY_nr, NX_nr)
    if NR_data.ndim == 4:
        x_true_d03_3d = NR_data[0, :, :, :]
    elif NR_data.ndim == 3:
        # 如果变量只有 2D (比如 SST)，为了代码兼容性增加一个假垂直层
        x_true_d03_3d = NR_data[0, :, :][np.newaxis, :, :] 
    
    NZ = x_true_d03_3d.shape[0]
    print(f"目标变量 {var} 的垂直层数 (NZ) = {NZ}")
    
    # 创建一个空的 3D 数组存放映射到 d01 的真实场
    x_true_full_3d = np.zeros((NZ, NY, NX))
    for k in range(NZ):
        x_true_full_3d[k, :, :] = x_true_d03_3d[k, :, :].ravel()[indices].reshape((NY, NX))

    # ---------------------------------------------------------
    # 步骤 3：读取 50 个集合成员的三维全场
    # ---------------------------------------------------------
    print("正在读取 50 个集合成员的先验场三维全局数据...")
    # x_prior_full_arr 最终形状将是 (50, NZ, NY, NX)
    x_prior_full_list = []
    for imem in range(1, N_members + 1):
        mem = f'{imem:03d}'
        ens_file = f'{prior_dir}/firstguess_d01.mem{mem}'
        data = nc_read1(ens_file, var)
        if data.ndim == 4:
            x_prior_full_list.append(data[0, :, :, :])
        elif data.ndim == 3:
            x_prior_full_list.append(data[0, :, :][np.newaxis, :, :])
            
    x_prior_full_arr = np.array(x_prior_full_list)

    # ---------------------------------------------------------
    # 预计算：每个观测的水平局地化掩码 (极大节省内层循环时间)
    # ---------------------------------------------------------
    obs_masks = []
    print(f"正在计算每个观测点水平 {horizontal_cutoff_km}km 内的网格掩码...")
    for iobs in range(Nobs):
        dist = calculate_haversine_distance(grid_lats, grid_lons, hx_lats[iobs], hx_lons[iobs])
        obs_masks.append(dist <= horizontal_cutoff_km)

    # ---------------------------------------------------------
    # 步骤 4：外层按模式层循环，内层按观测循环累加 ELF
    # ---------------------------------------------------------
    alpha_results = []
    
    print(f"\n开始计算垂直 ELF 廓线 (共 {NZ} 层):")
    print("-" * 60)
    
    for k in range(NZ):
        bin_numerator = 0.0
        bin_denominator = 0.0
        total_K_in_level = 0 
        
        for iobs in range(Nobs): 
            y_prior_single = hx_ens_lists[iobs]
            y_mean = np.mean(y_prior_single)
            y_var = np.var(y_prior_single, ddof=1)
            y_true_scalar = obs_array[iobs]
            obs_err_scalar = 0.25
            
            if y_var == 0: continue
                
            u_var = (y_var * obs_err_scalar) / (y_var + obs_err_scalar)
            var_ratio = u_var / y_var
            Aj = (var_ratio - 1.0) * y_mean
            Bj = u_var / obs_err_scalar
            
            exp_inc = Aj + Bj * y_true_scalar
            exp_var_term = (Aj**2) + (2 * Aj * Bj * y_true_scalar) + (Bj**2) * (obs_err_scalar + y_true_scalar**2)
            
            # 获取该观测水平搜索半径内的网格
            mask = obs_masks[iobs]
            K_obs = np.sum(mask) 
            
            if K_obs > 0:
                # 提取【第 k 层】对应网格的数据
                x_true_bin = x_true_full_3d[k, mask]              # (K_obs,)
                x_prior_bin = x_prior_full_arr[:, k, mask].T      # (K_obs, 50)
                
                # 向量化计算协方差
                x_mean_bin = np.mean(x_prior_bin, axis=1)         # (K_obs,)
                x_anom = x_prior_bin - x_mean_bin[:, None]        # (K_obs, 50)
                y_anom = y_prior_single - y_mean                  # (50,)
                
                cov_xy = np.sum(x_anom * y_anom, axis=1) / (N_members - 1)
                b_hat = cov_xy / y_var                            # (K_obs,)
                
                # 累加分子分母
                numerator_terms = b_hat * (x_true_bin - x_mean_bin) * exp_inc
                
                bin_numerator += np.sum(numerator_terms)
                
                denominator_terms = (b_hat**2) * exp_var_term
                bin_denominator += np.sum(denominator_terms)
                
                total_K_in_level += K_obs

        # 当前层结算
        if total_K_in_level == 0 or bin_denominator == 0:
            alpha_results.append(np.nan)
            print(f"模式层 {k:2d}: 无有效配对, 记录为 NaN")
        else:
            alpha = bin_numerator / bin_denominator
            # alpha = max(0.0, min(1.5, alpha)) # 限制范围便于画图
            alpha_results.append(alpha)
            print(f"模式层 {k:2d}: 累积 {total_K_in_level:6d} 对, Alpha = {alpha:.4f}")
            
    print("-" * 60)

    # ---------------------------------------------------------
    # 步骤 5：绘制垂直 ELF 廓线图
    # ---------------------------------------------------------
    levels = np.arange(NZ)
    
    print("\n正在生成垂直趋势图...")
    plt.figure(figsize=(7, 9))
    
    # 注意：气象学中通常将垂直层画在 Y 轴。如果 k=0 是底层，我们将图的 Y 轴原点放在底部
    plt.plot(alpha_results, levels, marker='o', linestyle='-', color='red', linewidth=2.5, markersize=8)
    
    plt.title(f'Vertical ELF for {var}', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Localization Coefficient (Alpha)', fontsize=14)
    plt.ylabel('Model Level Index', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设定 X 轴范围
    plt.xlim(-0.05, 2.0)
    # 根据 WRF 习惯设定 Y 轴（0 在最下方代表地面/海面）
    plt.ylim(0, NZ-1)
    
    out_img = f'./figs/Vertical_ELF_Profile_{var}.png' 
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"画图完成！图片已保存为: {out_img}")