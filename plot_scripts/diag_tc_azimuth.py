import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

def destagger(var, stagger_dim):
    """将错位网格(Staggered grid)上的变量插值到质量中心网格上"""
    if stagger_dim == 0:
        return (var[:-1, :, :] + var[1:, :, :]) / 2.0
    elif stagger_dim == 1:
        return (var[:, :-1, :] + var[:, 1:, :]) / 2.0
    elif stagger_dim == 2:
        return (var[:, :, :-1] + var[:, :, 1:]) / 2.0
    return var

def calc_azimuthal_mean(wrf_file, r_max_km=300, dr_km=5):
    """计算方位角平均的切向风和温度距平"""
    ds = nc.Dataset(wrf_file)
    
    # 1. 提取基础变量
    # 注意：这里取第一个时间步长
    P = ds.variables['P'][0, :, :, :] + ds.variables['PB'][0, :, :, :] # 全气压 (Pa)
    T = (ds.variables['THM'][0, :, :, :] + 300.0) * (P / 100000.0)**0.2854 # 扰动位温转实际温度 (K)
    U = ds.variables['U'][0, :, :, :]
    V = ds.variables['V'][0, :, :, :]
    PSFC = ds.variables['PSFC'][0, :, :]
    Z = ds.variables['PH'][0, :, :, :] + ds.variables['PHB'][0, :, :, :]
    Z = Z / 9.81 # 位势高度 (m)
    Z = destagger(Z, 0) # 垂直方向去错位
    
    # 获取网格间距 (假设 dx = dy)
    dx = getattr(ds, 'DX', 7500.0) # 如果属性中没有，默认为 12km (请根据你的D01实际情况修改)
    
    # 2. 去错位 (Destagger U and V to mass points)
    U = destagger(U, 2)
    V = destagger(V, 1)
    
    # 3. 寻找台风中心 (使用海平面气压极小值)
    # 你也可以用平滑后的气压或涡度中心，这里用最简单的最小值
    cy, cx = np.unravel_index(np.argmin(PSFC, axis=None), PSFC.shape)
    
    # 4. 构建相对中心的极坐标系
    ny, nx = PSFC.shape
    y, x = np.mgrid[0:ny, 0:nx]
    
    # 计算每个格点距离中心的物理距离 (米)
    dist_x = (x - cx) * dx
    dist_y = (y - cy) * dx
    r = np.sqrt(dist_x**2 + dist_y**2) / 1000.0  # 转换为公里 (km)
    
    # 计算方位角 (Angle)
    theta = np.arctan2(dist_y, dist_x)
    
    # 5. 计算切向风 (Tangential Wind)
    # 公式: Vt = -U*sin(theta) + V*cos(theta)
    # 因为U和V在不同高度不同，我们需要利用广播机制
    theta_3d = np.broadcast_to(theta, U.shape)
    Vt = -U * np.sin(theta_3d) + V * np.cos(theta_3d)
    
    # 6. 径向分箱 (Radial Binning)
    bins = np.arange(0, r_max_km + dr_km, dr_km)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    nz = U.shape[0]
    
    Vt_mean = np.zeros((nz, len(bin_centers)))
    T_mean = np.zeros((nz, len(bin_centers)))
    Z_mean = np.zeros((nz, len(bin_centers)))
    
    # 逐层进行方位角平均
    for k in range(nz):
        # 使用 binned_statistic 计算每个径向环内的平均值
        Vt_mean[k, :], _, _ = binned_statistic(r.flatten(), Vt[k, :, :].flatten(), statistic='mean', bins=bins)
        T_mean[k, :], _, _ = binned_statistic(r.flatten(), T[k, :, :].flatten(), statistic='mean', bins=bins)
        Z_mean[k, :], _, _ = binned_statistic(r.flatten(), Z[k, :, :].flatten(), statistic='mean', bins=bins)
        
    # 7. 计算温度距平 (Temperature Anomaly)
    # 物理定义：内核温度 减去 环境温度。环境温度定义为半径 200km 到 300km 的平均
    env_mask = (bin_centers >= 200) & (bin_centers <= 300)
    T_env = np.nanmean(T_mean[:, env_mask], axis=1) # 获取每层的环境平均温度
    T_anom = T_mean - T_env[:, np.newaxis] # 利用广播机制相减
    
    # 获取平均高度的 1D 廓线 (用于 Y 轴)
    Z_1d = np.nanmean(Z_mean, axis=1) / 1000.0 # 转换为 km
    
    ds.close()
    return bin_centers, Z_1d, Vt_mean, T_anom

def plot_typhoon_structure(wrf_file, title_prefix=""):
    r, z, Vt, Tanom = calc_azimuthal_mean(wrf_file)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # 1. 绘制温度距平 (暖心) - 填色图
    # 限制色标范围，暖心一般在 2~10 度
    levels_t = np.arange(-2, 12, 1)
    cf = ax.contourf(r, z, Tanom, levels=levels_t, cmap='RdBu_r', extend='both')
    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Temperature Anomaly (K)', fontsize=12)
    
    # 2. 绘制切向风 (旋转) - 等值线
    # 切向风一般 10m/s 一档
    levels_v = np.arange(10, 80, 5)
    cs = ax.contour(r, z, Vt, levels=levels_v, colors='black', linewidths=1.5)
    ax.clabel(cs, fmt='%d', inline=True, fontsize=10)
    
    # 图表设置
    ax.set_ylim(0, 16) # 通常关注对流层 (0-16km)
    ax.set_xlim(0, 200) # 关注内核区 (0-200km)
    ax.set_xlabel('Radius from Center (km)', fontsize=12)
    ax.set_ylabel('Height (km)', fontsize=12)
    ax.set_title(f'{title_prefix} - Azimuthal Mean Structure', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"./figs/{title_prefix.replace(' ', '_')}_structure.png")

# ================= 调用示例 =================
# 你可以分别传入你的控制实验(未同化)和同化后实验(如a, b, c)的 10_00:00 (同化初始场) 和 10_01:00 (积分1小时后) 的 wrfout 文件
# wrf_file_path = "/share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA/run_wrf/10_00_00/043/wrfout_d02_2018-09-10_00:00:00" 
# plot_typhoon_structure(wrf_file_path, title_prefix="Exp_043_00h")

wrf_file_path = "/scratch/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_06_00/EAKF/firstguess_d02.ensmean"
plot_typhoon_structure(wrf_file_path, title_prefix="Exp_postmean_06h")