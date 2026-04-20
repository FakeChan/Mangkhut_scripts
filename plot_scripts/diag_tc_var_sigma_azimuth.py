import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

def destagger(var, stagger_dim):
    """将错位网格(Staggered grid)上的变量插值到质量中心网格上"""
    if stagger_dim == 0:   # Z方向 (如 W, PH)
        return (var[:-1, :, :] + var[1:, :, :]) / 2.0
    elif stagger_dim == 1: # Y方向 (如 V)
        return (var[:, :-1, :] + var[:, 1:, :]) / 2.0
    elif stagger_dim == 2: # X方向 (如 U)
        return (var[:, :, :-1] + var[:, :, 1:]) / 2.0
    return var

def get_wrf_variable(ds, var_name, theta_3d=None):
    """
    智能提取并处理指定的 WRF 变量
    如果 var_name 是内置的复合变量（如 Vt, Vr），则进行相应计算
    """
    if var_name == 'Vt' or var_name == 'Vr':
        U = destagger(ds.variables['U'][0, :, :, :], 2)
        V = destagger(ds.variables['V'][0, :, :, :], 1)
        if var_name == 'Vt':
            return -U * np.sin(theta_3d) + V * np.cos(theta_3d) # 切向风
        else:
            return U * np.cos(theta_3d) + V * np.sin(theta_3d)  # 径向风
            
    elif var_name == 'T':
        P = ds.variables['P'][0, :, :, :] + ds.variables['PB'][0, :, :, :]
        return (ds.variables['THM'][0, :, :, :] + 300.0) * (P / 100000.0)**0.2854
        
    else:
        # 通用变量提取
        var = ds.variables[var_name][0, ...]
        # 自动探测是否需要去错位 (检查是否有 staggered 属性)
        if hasattr(ds.variables[var_name], 'stagger'):
            stagger_str = ds.variables[var_name].stagger
            if 'Z' in stagger_str: var = destagger(var, 0)
            elif 'Y' in stagger_str: var = destagger(var, 1)
            elif 'X' in stagger_str: var = destagger(var, 2)
        return var

def calc_single_azimuthal_mean(wrf_file, var_name, r_max_km=300, dr_km=5):
    """计算单个成员的方位角平均"""
    ds = nc.Dataset(wrf_file)
    
    # 获取网格信息和高度
    dx = getattr(ds, 'DX', 7500.0)
    PSFC = ds.variables['PSFC'][0, :, :]
    Z = ds.variables['PH'][0, :, :, :] + ds.variables['PHB'][0, :, :, :]
    Z = destagger(Z / 9.81, 0) 
    
    # 寻找台风中心
    cy, cx = np.unravel_index(np.argmin(PSFC, axis=None), PSFC.shape)
    
    # 构建极坐标系
    ny, nx = PSFC.shape
    y, x = np.mgrid[0:ny, 0:nx]
    dist_x = (x - cx) * dx
    dist_y = (y - cy) * dx
    r = np.sqrt(dist_x**2 + dist_y**2) / 1000.0  # (km)
    theta = np.arctan2(dist_y, dist_x)
    
    # 提取目标变量
    nz = Z.shape[0]
    theta_3d = np.broadcast_to(theta, (nz, ny, nx))
    target_var = get_wrf_variable(ds, var_name, theta_3d)
    
    # 径向分箱
    bins = np.arange(0, r_max_km + dr_km, dr_km)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    var_mean = np.zeros((nz, len(bin_centers)))
    Z_mean = np.zeros((nz, len(bin_centers)))
    
    for k in range(nz):
        var_mean[k, :], _, _ = binned_statistic(r.flatten(), target_var[k, :, :].flatten(), statistic='mean', bins=bins)
        Z_mean[k, :], _, _ = binned_statistic(r.flatten(), Z[k, :, :].flatten(), statistic='mean', bins=bins)
    
    env_mask = (bin_centers >= 200) & (bin_centers <= 300)
    var_env = np.nanmean(var_mean[:, env_mask], axis=1)
    var_anom = var_mean - var_env[:, np.newaxis]
    Z_1d = np.nanmean(Z_mean, axis=1) / 1000.0 # 转换为 km
    ds.close()
    
    return bin_centers, Z_1d, var_mean,var_anom

def calc_ensemble_statistics(base_dir, wrf_filename, member_indices, var_name, r_max_km=300, dr_km=5):
    """批量读取集合成员并计算 Mean 和 Sigma (Std Dev)"""
    all_var_means = []
    all_var_anoms = []
    ref_r, ref_z = None, None
    valid_members = 0
    
    print(f"开始处理变量: {var_name}")
    for idx in member_indices:
        mem_str = f"mem{idx:03d}"
        file_path = f"{base_dir}/{wrf_filename}.{mem_str}"
        
        if not os.path.exists(file_path):
            print(f"  警告: 文件不存在, 跳过 {mem_str}")
            continue
            
        try:
            r, z, var_mean, var_anom = calc_single_azimuthal_mean(file_path, var_name, r_max_km, dr_km)
            all_var_means.append(var_mean)
            all_var_anoms.append(var_anom)
            if ref_r is None:
                ref_r, ref_z = r, z # 以第一个成功读取的成员网格为准
            valid_members += 1
            print(f"  成功处理: {mem_str}")
        except Exception as e:
            print(f"  处理 {mem_str} 时报错: {e}")
            
    if valid_members == 0:
        raise ValueError("没有找到任何有效的成员文件！")
        
    print(f"总计处理完成 {valid_members} 个集合成员。")
    
    # 堆叠数据 (Members, Z, R) 并计算统计量
    all_var_means = np.array(all_var_means)
    all_var_anoms = np.array(all_var_anoms)
    all_var_envs=all_var_means-all_var_anoms
    
    var_env_mean=np.nanmean(all_var_envs,axis=0) # ensemble mean env
    ens_mean = np.nanmean(all_var_means, axis=0) # 集合均值
    ens_anom = ens_mean- var_env_mean            # ensemble mean anomaly
    ens_std  = np.nanstd(all_var_means, axis=0)  # 集合标准差 (Sigma)
    
    return ref_r, ref_z, ens_anom, ens_std

def plot_ensemble_structure(r, z, ens_mean, ens_std, var_name, title_prefix=""):
    """绘制结果：填色为均值，等值线为标准差"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 1. 绘制集合均值 (填色图)
    # 这里自动根据数据的极值生成 colorbar levels，你也可以根据具体变量手动写死
    mean_min, mean_max = np.nanmin(ens_mean), np.nanmax(ens_mean)
    levels_mean = np.linspace(mean_min, mean_max, 21)
    cf = ax.contourf(r, z, ens_mean, levels=levels_mean, cmap='RdBu_r', extend='both')
    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(f'{var_name} Ensemble Mean', fontsize=12)
    
    # 2. 绘制集合标准差 Sigma (等值线)
    # 取 std 的最大值决定画几根线，避免线太密或太疏
    std_max = np.nanmax(ens_std)
    if std_max > 0:
        levels_std = np.linspace(0.1 * std_max, std_max, 6) # 取 6 根等值线
        cs = ax.contour(r, z, ens_std, levels=levels_std, colors='black', linewidths=1.5, alpha=0.8)
        ax.clabel(cs, fmt='%.2f', inline=True, fontsize=10, colors='black')
    
    # 图表设置
    ax.set_ylim(0, 16) 
    ax.set_xlim(0, 200)
    ax.set_xlabel('Radius from Center (km)', fontsize=12)
    ax.set_ylabel('Height (km)', fontsize=12)
    ax.set_title(f'Ensmean anomaly (Shaded) & mean Spread (Contours) of {var_name}', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"./figs/{title_prefix.replace(' ', '_')}_{var_name}_ens_structure.png")
    plt.show()

# ================= 调用示例 =================
if __name__ == "__main__":
    # 配置路径
    base_dir = "/scratch/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_06_00/EAKF"
    wrf_filename = "firstguess_d02" # 统一的文件名
    
    # 指定集合成员编号列表，例如 mem001 到 mem050
    # 这里使用 range(1, 51) 会生成 1 到 50，对应 mem001 - mem050
    member_indices =[6,15,29,37,43,44] 
    
    # 你想诊断的变量（支持 'Vt'切向风, 'Vr'径向风, 'T'温度, 'W'垂直速度, 'QVAPOR'水汽等）
    target_variable = 'T' 
    
    # 执行计算
    r, z, ens_mean, ens_std = calc_ensemble_statistics(
        base_dir=base_dir, 
        wrf_filename=wrf_filename, 
        member_indices=member_indices, 
        var_name=target_variable
    )
    
    # 绘制图像
    plot_ensemble_structure(r, z, ens_mean, ens_std, var_name=target_variable, title_prefix="diag_azimuth")