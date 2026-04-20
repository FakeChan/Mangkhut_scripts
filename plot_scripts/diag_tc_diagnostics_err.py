import numpy as np
from netCDF4 import Dataset
from wrf import getvar, latlon_coords
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_ensemble_errors(save_path,errors_dict):
    """
    绘制集合预报 4 个诊断量的误差分布图
    """
    # 1. 将误差字典转换为 Pandas DataFrame，方便 Seaborn 处理
    df_errors = pd.DataFrame(errors_dict)
    
    # 2. 设置绘图风格与中文字体支持
    sns.set_theme(style="whitegrid")
    # 注意：如果你在 Mac 或 Linux 环境下，可能需要把 'SimHei' 换成你系统里有的中文字体，如 'Arial Unicode MS' 或 'WenQuanYi Micro Hei'
    # plt.rcParams['font.sans-serif'] = ['SimHei'] 
    # plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
    
    # 3. 创建 2x2 的子图画布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('dirtribution of ensemble errors', fontsize=16, fontweight='bold')
    
    # 4. 定义四个变量的绘图配置：(字典键名, 所在子图, 标题, 单位, 主题色)
    plot_configs = [
        ('mslp', axes[0, 0], 'MSlP ERROR', 'hPa', 'skyblue'),
        ('mws',  axes[0, 1], 'MWS ERROR', 'm/s', 'salmon'),
        ('rmw',  axes[1, 0], 'RMW ERROR', 'km',  'lightgreen'),
        ('r17',  axes[1, 1], 'R17 ERROR', 'km', 'plum')
    ]
    
    # 5. 循环绘制每个子图
    for key, ax, title, unit, color in plot_configs:
        # 画箱线图：展示 25%, 50%, 75% 分位数及分布范围
        sns.boxplot(y=df_errors[key], ax=ax, color=color, width=0.4, 
                    boxprops=dict(alpha=0.7), showfliers=False)
        
        # 画带抖动的散点图：把 50 个成员的具体误差打在图上
        sns.stripplot(y=df_errors[key], ax=ax, color='black', alpha=0.5, jitter=0.15)
        
        # 画一条红色的 Y=0 虚线（代表完美预报、零误差）
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Perfect (Zero Error)')
        
        # 设置标题和标签
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(f"error ({unit})", fontsize=12)
        ax.legend(loc='upper right')
    
    # 6. 调整布局并显示/保存
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为总标题留出空间
    
    # 推荐保存为高分辨率图片以便放到论文或报告中
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"The figure has been saved to {save_path}")
    

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算地球上两点（或一点与一个网格矩阵）之间的距离（单位：公里）
    """
    R = 6371.0 # 地球平均半径，单位公里
    
    # 转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def calc_typhoon_mslp(ncfile, timeidx=0):
    """
    1. 计算台风最低海平面气压 (MSLP) 及台风中心位置
    """
    # 提取海平面气压 (hPa)
    slp = getvar(ncfile, "slp", timeidx=timeidx)
    
    # 获取最低气压值
    mslp = np.min(slp.values)
    
    # 获取最低气压所在的网格索引 (y, x)，以此作为台风中心
    center_idx = np.unravel_index(np.argmin(slp.values, axis=None), slp.shape)
    
    return mslp, center_idx

def calc_typhoon_mws(ncfile, timeidx=0):
    """
    2. 计算 10m 最大风速 (MWS)
    """
    # 提取 10米 U风 和 V风
    u10 = getvar(ncfile, "U10", timeidx=timeidx)
    v10 = getvar(ncfile, "V10", timeidx=timeidx)
    
    # 计算风速大小 (m/s)
    wspd10 = np.sqrt(u10**2 + v10**2)
    
    # 获取最大风速
    mws = np.max(wspd10.values)
    
    return mws, wspd10

def calc_typhoon_rmw(ncfile, wspd10, center_idx, timeidx=0):
    """
    3. 计算最大风速半径 (RMW)
    """
    # 获取经纬度网格
    lats, lons = latlon_coords(wspd10)
    
    # 台风中心经纬度
    center_lat = lats.values[center_idx]
    center_lon = lons.values[center_idx]
    
    # 获取最大风速所在的网格索引
    max_wind_idx = np.unravel_index(np.argmax(wspd10.values, axis=None), wspd10.shape)
    max_wind_lat = lats.values[max_wind_idx]
    max_wind_lon = lons.values[max_wind_idx]
    
    # 计算中心到最大风速点的距离 (km)
    rmw = haversine_distance(center_lat, center_lon, max_wind_lat, max_wind_lon)
    
    return rmw

def calc_typhoon_r17(ncfile, wspd10, center_idx, timeidx=0, search_radius_km=600):
    """
    4. 计算 17m/s 风速半径 (R17) (即7级风圈半径)
    """
    lats, lons = latlon_coords(wspd10)
    center_lat = lats.values[center_idx]
    center_lon = lons.values[center_idx]
    
    # 计算所有网格点到台风中心的距离矩阵
    dist_matrix = haversine_distance(center_lat, center_lon, lats.values, lons.values)
    
    # 设定一个搜索范围（比如台风中心 600 公里内），避免把背景风场（如冷空气、季风槽）的高风速算进去
    valid_mask = dist_matrix <= search_radius_km
    
    # 提取在搜索范围内，且风速 >= 17 m/s 的所有点的距离
    r17_points_dist = dist_matrix[(wspd10.values >= 17.0) & valid_mask]
    
    if len(r17_points_dist) == 0:
         return 0.0, 0.0 # 若没有大于17m/s的风速
         
    # 计算 R17。气象上 R17 通常有两种表达：
    # 1. 17m/s 风速的最远影响距离 (最大值)
    r17_max = np.max(r17_points_dist)
    
    # 2. 17m/s 等风速线的平均半径 (提取16.5~17.5m/s附近的点求平均)
    contour_points = dist_matrix[(wspd10.values >= 16.5) & (wspd10.values <= 17.5) & valid_mask]
    r17_mean = np.mean(contour_points) if len(contour_points) > 0 else r17_max
    
    return r17_max, r17_mean

if __name__ == '__main__':
    def get_all_diagnostics(ncfile, timeidx=0):
        # 1. MSLP & Center
        mslp, center_idx = calc_typhoon_mslp(ncfile, timeidx)
        
        # 2. MWS & Wind Field
        mws, wspd10 = calc_typhoon_mws(ncfile, timeidx)
        
        # 3. RMW
        rmw = calc_typhoon_rmw(ncfile, wspd10, center_idx, timeidx)
        
        # 4. R17 (提取平均半径)
        _, r17_mean = calc_typhoon_r17(ncfile, wspd10, center_idx, timeidx)
        
        return mslp, mws, rmw, r17_mean

    # ==========================================
    # 1. 参数设置与文件路径准备
    # ==========================================
    time_index = 0  # 假设评估第一个时次
    nr_filepath = "/scratch/lililei1/kcfu/tc_mangkhut/NR/wrfout_d03_2018-09-10_06:00:00"  # 真值文件路径

    # 假设 50 个成员的文件名规律为 wrfout_d01_ens_01 到 wrfout_d01_ens_50
    # 请根据你的实际文件路径修改这里
    # ens_filepaths = [f"/scratch/lililei1/kcfu/tc_mangkhut/5cyclingDA/postAnal_EAKF/d01_10_00_00/analysis_d02.mem{i:03d}" for i in range(1, 51)]
    ens_filepaths = [f"/scratch/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_06_00/EAKF/firstguess_d02.mem{i:03d}" for i in range(1, 51)]
    save_fig_path='./figs/ensemble_error_distributions_1006_prior.png'
    # 初始化用于存储误差的列表
    errors = {
        "mslp": [],
        "mws": [],
        "rmw": [],
        "r17": []
    }

    # ==========================================
    # 2. 计算真值 (Nature Run)
    # ==========================================
    print("正在读取真值 (NR) 文件...")
    with Dataset(nr_filepath) as nc_nr:
        nr_mslp, nr_mws, nr_rmw, nr_r17 = get_all_diagnostics(nc_nr, timeidx=time_index)

    print(f"真值诊断结果 -> MSLP: {nr_mslp:.1f} hPa, MWS: {nr_mws:.1f} m/s, RMW: {nr_rmw:.1f} km, R17: {nr_r17:.1f} km")
    print("-" * 50)

    # ==========================================
    # 3. 循环计算 50 个集合成员及其误差
    # ==========================================
    print("开始计算 50 个集合成员的误差...")

    for idx, filepath in enumerate(ens_filepaths):
        try:
            with Dataset(filepath) as nc_ens:
                ens_mslp, ens_mws, ens_rmw, ens_r17 = get_all_diagnostics(nc_ens, timeidx=time_index)
                
                # 计算误差 (Error = Forecast - Truth)
                errors["mslp"].append(ens_mslp - nr_mslp)
                errors["mws"].append(ens_mws - nr_mws)
                errors["rmw"].append(ens_rmw - nr_rmw)
                errors["r17"].append(ens_r17 - nr_r17)
                
            # 每处理 10 个打印一次进度，避免运行时像卡死了一样
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/50 个成员...")
                
        except Exception as e:
            print(f"处理文件 {filepath} 时出错: {e}")

    print("-" * 50)

    # ==========================================
    # 4. 统计分析：计算误差均值和标准差
    # ==========================================
    # 转换为 numpy 数组以便计算
    for key in errors:
        errors[key] = np.array(errors[key])

    # 打印最终统计结果
    print("【集合预报误差统计结果 (Error = Ens - NR)】")
    print(f"1. MSLP 误差: 均值 (Bias) = {np.mean(errors['mslp']):.2f} hPa,  标准差 (Std) = {np.std(errors['mslp']):.2f} hPa")
    print(f"2. MWS  误差: 均值 (Bias) = {np.mean(errors['mws']):.2f} m/s, 标准差 (Std) = {np.std(errors['mws']):.2f} m/s")
    print(f"3. RMW  误差: 均值 (Bias) = {np.mean(errors['rmw']):.2f} km,  标准差 (Std) = {np.std(errors['rmw']):.2f} km")
    print(f"4. R17  误差: 均值 (Bias) = {np.mean(errors['r17']):.2f} km,  标准差 (Std) = {np.std(errors['r17']):.2f} km")
    plot_ensemble_errors(save_fig_path,errors)