import numpy as np
import netCDF4
import xarray as xr

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算地球表面两点（或数组）之间的 Haversine 距离。
    
    参数:
    -----------
    lat1, lon1 : float 或 ndarray
        第一点的纬度和经度（单位：度）。例如：WRF的 XLAT 和 XLONG 数组。
    lat2, lon2 : float 或 ndarray
        第二点的纬度和经度（单位：度）。例如：目标观测点的经纬度。
        
    返回:
    -----------
    distance : float 或 ndarray
        两点之间的球面大圆距离，单位为公里 (km)。
    """
    # 地球平均半径 (单位: 公里)
    R = 6371.0
    
    # 将十进制度数转化为弧度
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # 计算经纬度差值
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine 公式核心计算
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # 计算最终距离
    distance = R * c
    
    return distance
def calculate_elf_alpha(x_prior_ens, y_prior_ens, x_true, y_true, obs_err_var):
    """
    计算基于 OSSE 的经验局域化系数 (ELF) - 对应论文公式 (8)-(12)
    
    参数 (Parameters):
    -----------------
    x_prior_ens : ndarray, shape (K, N)
        状态变量(State variable)的先验集合。K 是配对数量(子集大小)，N 是集合成员数。
    y_prior_ens : ndarray, shape (K, N)
        观测变量(Observation)的先验集合。
    x_true      : ndarray, shape (K,)
        状态变量的真实值 (Truth)。
    y_true      : ndarray, shape (K,)
        观测变量的真实无偏值 (Truth of observation)。
    obs_err_var : ndarray, shape (K,) 或 float
        指定的观测误差方差 (\sigma_o^2)。
        
    返回 (Returns):
    --------------
    alpha : float
        计算得到的局域化系数
    """
    # 获取集合成员数 N
    N = x_prior_ens.shape[1]
    
    # 1. 计算先验集合均值 (\overline{x}_k 和 \overline{y}_p)
    x_mean = np.mean(x_prior_ens, axis=1)
    y_mean = np.mean(y_prior_ens, axis=1)
    
    # 2. 计算先验集合方差 (\sigma_p^2)
    y_var = np.var(y_prior_ens, axis=1, ddof=1) # 样本方差
    
    # 3. 计算后验方差 (\sigma_u^2) - 对应公式 (10)
    # \sigma_u^2 = [(\sigma_p^2)^-1 + (\sigma_o^2)^-1]^-1
    u_var = 1.0 / (1.0 / y_var + 1.0 / obs_err_var)
    
    # 4. 计算卡尔曼滤波权重系数 A_k 和 B_k - 对应公式 (8) 和 (9)
    # A_k = (\sigma_u^2 / \sigma_p^2 - 1) * \overline{y}_p
    # B_k = \sigma_u^2 / \sigma_p^2
    var_ratio = u_var / y_var
    A = (var_ratio - 1.0) * y_mean
    B = var_ratio
    
    # 5. 计算样本回归系数 (\hat{b}_k)
    # 即 x 和 y 的协方差 除以 y 的方差
    b_hat = np.zeros_like(x_mean)
    for k in range(len(x_mean)):
        cov_xy = np.cov(x_prior_ens[k, :], y_prior_ens[k, :], ddof=1)[0, 1]
        b_hat[k] = cov_xy / y_var[k]
        
    # 6. 计算公式(12)的分子部分 (Numerator)
    # \hat{b}_k * (x_t - \overline{x}_k) * (A_k + B_k * y_t)
    numerator_terms = b_hat * (x_true - x_mean) * (A + B * y_true)
    numerator_sum = np.sum(numerator_terms)
    
    # 7. 计算公式(12)的分母部分 (Denominator)
    # \hat{b}_k^2 * [A_k^2 + 2*A_k*B_k*y_t + B_k^2 * (\sigma_o^2 + y_t^2)]
    denominator_terms = (b_hat ** 2) * (
        A**2 + 
        2 * A * B * y_true + 
        (B**2) * (obs_err_var + y_true**2)
    )
    denominator_sum = np.sum(denominator_terms)
    
    # 8. 得到最终的 \alpha
    if denominator_sum == 0:
        return 0.0 # 避免除以 0
    alpha = numerator_sum / denominator_sum
    
    # 论文中提到：如果计算出的 alpha 小于 0，通常会被当作噪声设为 0
    alpha = max(0.0, alpha)
    
    return alpha

def nc_read1(filename,var):
    with netCDF4.Dataset(filename,'r') as ncfile:
        data = ncfile.variables[var][:].squeeze()
        return data

def get_ens_val_1loc(ens_dir,var,Nens,jidx,iidx):
    val_ens=np.zeros((1,Nens))
    for imem in range(1,Nens+1):
        mem = f'{imem:03d}'
        ens_file = f'{ens_dir}/firstguess_d01.mem{mem}'
        ens_data = nc_read1(ens_file,var)
        val_ens[0,imem-1] = ens_data[0, jidx, iidx]
    return val_ens

def get_ij_from_latlon(wrf_file, target_lat, target_lon):
    """
    根据给定的经纬度，在 wrfout 文件中查找最近的网格点索引 (j, i)。
    
    参数:
    wrf_file (str): wrfout 文件的绝对或相对路径
    target_lat (float): 目标纬度
    target_lon (float): 目标经度
    
    返回:
    tuple: (j_idx, i_idx) 对应的南北(j)和东西(i)索引
    """
    # 1. 打开 wrfout 文件
    # 使用 xarray 读取非常方便，且只在需要时加载数据
    with xr.open_dataset(wrf_file) as ds:
        # 2. 提取经纬度二维数组
        # wrfout 的 XLAT/XLONG 通常包含 Time 维度 (Time, south_north, west_east)
        # 我们只需要取第一个时次 [0, :, :] 即可
        lat_array = ds['XLAT'].isel(Time=0).values
        lon_array = ds['XLONG'].isel(Time=0).values
        
    # 3. 计算目标点与网格点之间的欧式距离平方
    # 对于区域模式的高分辨率网格，直接使用欧式距离的平方近似寻找最近点即可
    dist_sq = (lat_array - target_lat)**2 + (lon_array - target_lon)**2
    
    # 4. 找到最小距离对应的一维索引，并将其转换为二维的 (j, i) 索引
    j_idx, i_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
    
    # 确保返回的是原生 int 类型，方便后续切片使用
    return int(j_idx), int(i_idx)
# ==========================================
# 使用示例（针对你的“单个观测同化”场景）
# ==========================================
if __name__ == '__main__':
    # num of ensmembers
    N_members = 50
    var='OM_TMP'
    prior_dir='/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_00_00'
    NR_file='/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d03_2018-09-10_00:00:00'
    hx_raw="""
            259.0000       257.9400       256.7100       257.0200       255.7500
            261.9300       257.4000       256.6300       256.6900       256.8900
            256.4800       256.2100       256.0700       256.6300       257.4300
            256.7700       257.6800       257.5500       256.4600       256.5700
            257.1000       257.6900       257.2500       257.1100       257.3200
            257.6400       256.2600       256.4300       256.2900       256.2600
            257.3900       258.2400       256.5600       257.1600       256.9200
            256.5500       256.6400       256.3000       256.2900       257.5800
            256.5900       257.9700       256.6700       257.4200       256.2800
            256.5200       257.5600       256.9100       258.2300       256.5900
    """
    hx_ens = np.fromstring(hx_raw, sep=' ')
    # K=1，代表只有一个卫星亮温观测
    K = 1 
    target_lat = 13.2933
    target_lon = 147.1829
    
    jidx, iidx = get_ij_from_latlon(f'{prior_dir}/firstguess_d01.mem001',target_lat , target_lon)
    # 模拟输入数据（请替换为你自己的同化输出数据）
    # 注意：这里的数据形状是 (K, N_members)
    x_prior_ensemble = get_ens_val_1loc(prior_dir, var ,N_members, jidx =jidx, iidx= iidx)
    y_prior_ensemble = np.reshape(hx_ens,(K, N_members), order = 'F')

    j_NR, i_NR = get_ij_from_latlon(NR_file,target_lat , target_lon)
    
    x_true_value = nc_read1(NR_file,var)[0, j_NR, i_NR]
    y_true_value = np.array([256.93]) # 真实亮温值 (无观测误差的理论值)
    obs_error_variance = np.array([0.25]) # 卫星亮温的观测误差方差

    # 计算局域化系数
    alpha_estimated = calculate_elf_alpha(
        x_prior_ensemble, 
        y_prior_ensemble, 
        x_true_value, 
        y_true_value, 
        obs_error_variance
    )

    print(f"计算得到的经验局域化系数 (Alpha): {alpha_estimated:.4f}")