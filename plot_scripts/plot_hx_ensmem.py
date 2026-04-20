import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.spatial import cKDTree

# =============================================================================
# 1. 路径和参数配置 (可在此处自由修改具体位置和图片清晰度)
# =============================================================================
wrfrun_dir = "/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst"          # WRF输出所在主目录
obs_dir = "/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/4ens_BT"                # 观测Hx文件所在主目录
profile_dir = "/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/profile/profile_d01"        # profile经纬度文件所在主目录

wrf_filename = "wrfout_d01_2018-09-10_00:00:00"
obs_filename = "obs_d01_ch4_totalline.txt"
prof_filename = "prof10_00:00.dat" # 如果系统中包含冒号，请将其改为 "prof10_00:00.dat"

output_image_path = "./figs/correlation_profiles.png" # 输出图片的具体位置
image_dpi = 300                  # 输出图片DPI

num_members = 50                 # 集合成员数量
total_points = 676               # 总格点数
select_points = 300              # 随机挑选的格点数
plot_ocean    = True
# =============================================================================
# 2. 读取 676 个格点的经纬度信息
# =============================================================================
def get_lat_lon_from_prof(filepath):
    lats, lons = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # 寻找经纬度记录的标识行
            if '! Elevation (km), latitude and longitude (degrees)' in line:
                if i + 1 < len(lines):
                    parts = lines[i+1].strip().split()
                    if len(parts) >= 3:
                        lats.append(float(parts[1]))
                        lons.append(float(parts[2]))
    return np.array(lats), np.array(lons)

# 默认 50 个成员的格点位置完全相同，因此只读取第一个成员的 profile 获取坐标
prof_file_mem001 = os.path.join(profile_dir, "mem001", prof_filename)
lats_obs, lons_obs = get_lat_lon_from_prof(prof_file_mem001)

if len(lats_obs) != total_points:
    print(f"警告：读取到的格点数为 {len(lats_obs)}，而非预期的 {total_points} 个。")

# =============================================================================
# 3. 随机挑选 300 个格点
# =============================================================================
np.random.seed(42) # 设置随机种子以保证结果可复现，可按需移除
sel_idx = np.random.choice(len(lats_obs), select_points, replace=False)
lats_sel = lats_obs[sel_idx]
lons_sel = lons_obs[sel_idx]

# =============================================================================
# 4. 利用第一个成员的 WRF 坐标为提取做准备 (使用 KDTree 寻找最近网格点)
# =============================================================================
wrf_file_mem001 = os.path.join(wrfrun_dir, "mem001", wrf_filename)
nc_ref = Dataset(wrf_file_mem001)
xlat = nc_ref.variables['XLAT'][0, :, :]
xlong = nc_ref.variables['XLONG'][0, :, :]
nz = nc_ref.variables['T'].shape[1] # 获取垂直层数

# 构建 KDTree 并匹配最接近的 WRF (j, i) 索引
tree = cKDTree(np.column_stack((xlat.ravel(), xlong.ravel())))
_, idx_1d = tree.query(np.column_stack((lats_sel, lons_sel)))
j_indices, i_indices = np.unravel_index(idx_1d, xlat.shape)
nc_ref.close()

# =============================================================================
# 5. 遍历 50 个集合成员，提取数据
# =============================================================================
# 初始化三维数组存放提取结果 [格点数, 成员数, 垂直层数] / [格点数, 成员数]
T_all = np.zeros((select_points, num_members, nz))
QV_all = np.zeros((select_points, num_members, nz))
P_all = np.zeros((select_points, num_members, nz))
OM_TMP_all = np.zeros((select_points, num_members, 30))
Hx_all = np.zeros((select_points, num_members))

for m in range(1, num_members + 1):
    mem_str = f"mem{m:03d}"
    
    # 5.1 读取当前成员的 Hx
    obs_file = os.path.join(obs_dir, mem_str, 'AMSUA','BT_10_00_00',obs_filename)
    hx_data = np.loadtxt(obs_file)
    Hx_all[:, m-1] = hx_data[sel_idx]
    
    # 5.2 读取当前成员的 WRF 数据
    wrf_file = os.path.join(wrfrun_dir, mem_str, wrf_filename)
    nc = Dataset(wrf_file)
    
    # 获取三维变量 (取第0个时间步)
    # T 默认是扰动位温，加上 300K 为总位温
    theta = nc.variables['T'][0, :, :, :] + 300.0
    # 气压 P 默认是扰动气压加上基础气压 (Pa)
    P = nc.variables['P'][0, :, :, :] + nc.variables['PB'][0, :, :, :]
    # QVAPOR 直接读取
    QV = nc.variables['QVAPOR'][0, :, :, :]
    
    OM_TMP=nc.variables['OM_TMP'][0, :, :, :]
    
    # 计算实际气温 (K)
    T_K = theta * (P / 100000.0)**(287.0 / 1004.0)
    
    # 将选定格点的垂直廓线存入数组
    for p_idx, (jj, ii) in enumerate(zip(j_indices, i_indices)):
        T_all[p_idx, m-1, :] = T_K[:, jj, ii]
        QV_all[p_idx, m-1, :] = QV[:, jj, ii]
        P_all[p_idx, m-1, :] = P[:, jj, ii]
        OM_TMP_all[p_idx,m-1,:]=OM_TMP[:, jj, ii]
        
    nc.close()
    print(f"{mem_str} 提取完毕。")

# =============================================================================
# 6. 沿集合成员维度计算相关系数，并计算 300 个点的空间平均剖面
# =============================================================================
corr_T_Hx = np.zeros(nz)
corr_QV_Hx = np.zeros(nz)
corr_T_T900 = np.zeros(nz)
cov_T_Hx = np.zeros(nz)
corr_OM_TMP = np.zeros(30)

for z_ocean in range(30):
    sum_c_OM_TMP = 0.0
    valid_points=0
    for p in range(select_points):
        OM_TMP_layer = OM_TMP_all[p, :, z_ocean]
        Hx_val = Hx_all[p, :]
        c_OM_TMP = np.corrcoef(OM_TMP_layer, Hx_val)[0,1] 
        if not np.isnan(c_OM_TMP) :
                sum_c_OM_TMP += c_OM_TMP
                valid_points += 1
    if valid_points > 0:
        corr_OM_TMP[z_ocean] = sum_c_OM_TMP / valid_points
for z in range(nz):
    sum_c_T = 0.0
    sum_c_QV = 0.0
    sum_c_T900 = 0.0
    sum_cov_T = 0.0
    valid_points = 0
    
    for p in range(select_points):
        # 沿集合成员展开的 50 个样本
        T_layer = T_all[p, :, z]
        QV_layer = QV_all[p, :, z]
        Hx_val = Hx_all[p, :]
        
        # 计算该格点平均气压最靠近 900hPa (90000 Pa) 的所在层
        P_mean_prof = np.mean(P_all[p, :, :], axis=0)
        z_900 = np.argmin(np.abs(P_mean_prof - 90000.0))
        T_900_layer = T_all[p, :, z_900]
        
        # 计算该点气温与 Hx、QVAPOR与 Hx，以及气温与最靠近900hPa的气温相关系数
        c_T = np.corrcoef(T_layer, Hx_val)[0, 1]
        c_QV = np.corrcoef(QV_layer, Hx_val)[0, 1]
        c_T900 = np.corrcoef(T_layer, T_900_layer)[0, 1]
        
        cov_val = np.cov(T_layer, Hx_val)[0, 1]
        
        # 剔除可能存在的异常值 (如集合方差为0时)
        if not np.isnan(c_T) and not np.isnan(c_QV) and not np.isnan(c_T900):
            sum_c_T += c_T
            sum_c_QV += c_QV
            sum_c_T900 += c_T900
            sum_cov_T += cov_val
            valid_points += 1
            
    if valid_points > 0:
        corr_T_Hx[z] = sum_c_T / valid_points
        corr_QV_Hx[z] = sum_c_QV / valid_points
        corr_T_T900[z] = sum_c_T900 / valid_points
        cov_T_Hx[z] = sum_cov_T / valid_points

# >>>>> 新增：计算温度的集合离散度 (Spread) 开始 <<<<<
# T_all 的维度是 (格点数 300, 成员数 50, 垂直层数 nz)
# 1. 沿成员维度(axis=1)求标准差，得到每个格点、每层的离散度
T_spread = np.std(T_all, axis=1) 
QV_spread = np.std(QV_all, axis=1)
# 2. 沿格点维度(axis=0)求平均，得到整层平均的离散度廓线
mean_T_spread = np.mean(T_spread, axis=0)
mean_Qv_spread = np.mean(QV_spread, axis=0)
# >>>>> 新增：计算温度的集合离散度 (Spread) 结束 <<<<<
# =============================================================================
# 7. 绘制三张子图
# =============================================================================
fig, axes = plt.subplots(1, 6, figsize=(30, 6), sharey=True)
layers = np.arange(nz) # 使用垂直层数（索引）做Y轴
ocean_layers = np.arange(30)
# 子图1：气温与Hx的相关系数
axes[0].plot(corr_T_Hx, layers, color='firebrick', lw=2)
axes[0].set_title('Corr(T, Hx)', fontsize=14)
axes[0].set_xlabel('Correlation Coefficient', fontsize=12)
axes[0].set_ylabel('Vertical Layer Index', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.6)

# 子图2：水汽混合比与Hx的相关系数
axes[1].plot(corr_QV_Hx, layers, color='steelblue', lw=2)
axes[1].set_title('Corr(QVAPOR, Hx)', fontsize=14)
axes[1].set_xlabel('Correlation Coefficient', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.6)

# 子图3：气温与 900hPa 气温的相关系数
axes[2].plot(corr_T_T900, layers, color='forestgreen', lw=2)
axes[2].set_title('Corr(T, T at ~900hPa)', fontsize=14)
axes[2].set_xlabel('Correlation Coefficient', fontsize=12)
axes[2].grid(True, linestyle='--', alpha=0.6)

# >>>>> 新增：子图4的绘制代码 开始 <<<<<
# 子图4：气温的集合离散度 (Spread)
# axes[3].plot(mean_T_spread, layers, color='purple', lw=2)
# axes[3].set_title('T Ensemble Spread ($\sigma_T$)', fontsize=14)
axes[3].plot(mean_Qv_spread, layers, color='purple', lw=2)
axes[3].set_title('Qv Ensemble Spread ($\sigma_Q$)', fontsize=14)
axes[3].set_xlabel('Standard Deviation (K)', fontsize=12)
axes[3].grid(True, linestyle='--', alpha=0.6)
# 在图上画一根900hPa大致位置的参考线（假定在第10层左右，你可以根据实际输出微调高度）
axes[3].axhline(y=10, color='gray', linestyle=':', label='~900 hPa')
axes[3].legend(loc='upper right')
# >>>>> 新增：子图4的绘制代码 结束 <<<<<
axes[4].plot(cov_T_Hx, layers, color='darkorange', lw=2)
axes[4].set_title('Cov(T, Hx)', fontsize=14)
axes[4].set_xlabel('Covariance ($K^2$)', fontsize=12)
axes[4].grid(True, linestyle='--', alpha=0.6)
axes[4].axhline(y=10, color='gray', linestyle=':', label='~900 hPa')
axes[4].legend(loc='upper right')


axes[5].plot(corr_OM_TMP, ocean_layers, color='goldenrod', lw=2)
axes[5].set_title('Corr(OM_TMP,Hx)', fontsize=14)
axes[5].set_xlabel('Correlation Coefficient', fontsize=12)
axes[5].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(output_image_path, dpi=image_dpi, bbox_inches='tight')
print(f"相关系数剖面图已保存至: {output_image_path}")
