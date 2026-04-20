import numpy as np
import netCDF4
import xarray as xr
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

def get_ens_val_Klocs(ens_dir, var, Nens, j_indices, i_indices):
    K = len(j_indices)
    val_ens = np.zeros((K, Nens))
    
    for imem in range(1, Nens + 1):
        mem = f'{imem:03d}'
        ens_file = f'{ens_dir}/firstguess_d01.mem{mem}'
        ens_data = nc_read1(ens_file, var)
        
        # 兼容不同维度的 WRF 输出，动态适配切片
        if ens_data.ndim == 4:
            val_ens[:, imem-1] = ens_data[0, 0, j_indices, i_indices]
        elif ens_data.ndim == 3:
            val_ens[:, imem-1] = ens_data[0, j_indices, i_indices]
        else:
            val_ens[:, imem-1] = ens_data[j_indices, i_indices]
            
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
        lat_array = ds['XLAT'].isel(Time=0).values
        lon_array = ds['XLONG'].isel(Time=0).values
        
    K = len(target_lat)
    j_indices = np.zeros(K, dtype=int)
    i_indices = np.zeros(K, dtype=int)
    
    for k in range(K):
        dist_sq = (lat_array - target_lat[k])**2 + (lon_array - target_lon[k])**2
        j, i = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        j_indices[k] = j
        i_indices[k] = i
        
    return j_indices, i_indices


def extract_elf_data_from_obs_seq(filepath):
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
# 使用示例（针对你的“单个观测同化”场景）
# ==========================================
import numpy as np

# ... (保留之前定义的所有函数) ...

if __name__ == '__main__':
    N_members = 50
    var = 'OM_TMP'
    
    prior_dir = '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_00_00'
    NR_file = '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d03_2018-09-10_00:00:00'
    sample_wrf_file = f'{prior_dir}/firstguess_d01.mem001'
    
    obs_seq_file = '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir/obs_seq.out_kctest1_d01_10_00_00_quantile'
    # 新增：你的真实无噪声观测文件路径
    truth_txt_file = '/share/home/lililei1/kcfu/tc_mangkhut/3create_obs/hx_rttov/3obs_BT/AMSUA/BT_10_00_00/obs_d01_ch4_totalline.txt'

    # ---------------------------------------------------------
    # 第一步：提取观测元数据 (注意：第三个返回值我们命名为 y_noisy_obs，不再作为 truth 使用)
    # ---------------------------------------------------------
    obs_lats, obs_lons, y_prior_ensemble = extract_elf_data_from_obs_seq(obs_seq_file)
    K = len(obs_lats)
    obs_err_variance_array = np.full(K, 0.25)
    
    # ---------------------------------------------------------
    # 【新增核心步骤】：直接读取 TXT 文件获取真正的 y_true_array
    # ---------------------------------------------------------
    print(f"正在从TXT文件读取真实的无噪声观测值: {truth_txt_file}")
    # np.loadtxt 会自动将 K 行 1 列的文本转化为 shape 为 (K,) 的一维数组
    y_true_array = np.loadtxt(truth_txt_file)
    
    # 加入一个安全校验，防止 txt 的行数和 obs_seq 里的观测数对不上
    if len(y_true_array) != K:
        raise ValueError(f"维度不匹配！obs_seq中有 {K} 个观测，但TXT文件中有 {len(y_true_array)} 个真实值。")

    # ---------------------------------------------------------
    # 第二步：批量匹配索引并提取背景场数据
    # ---------------------------------------------------------
    print("正在批量计算经纬度索引...")
    j_idx_arr, i_idx_arr = get_ij_from_latlon(sample_wrf_file, obs_lats, obs_lons)
    
    print("正在批量提取先验状态场...")
    x_prior_ensemble = get_ens_val_Klocs(prior_dir, var, N_members, j_idx_arr, i_idx_arr)

    # ---------------------------------------------------------
    # 第三步：批量提取真实状态场 (Nature Run)
    # ---------------------------------------------------------
    print("正在提取 Nature Run 真实场...")
    NR_data = nc_read1(NR_file, var)
    x_true_array = np.zeros(K)
    
    for k in range(K):
        j_nr = j_idx_arr[k]
        i_nr = i_idx_arr[k]
        if NR_data.ndim == 4:
            x_true_array[k] = NR_data[0, 0, j_nr, i_nr]
        elif NR_data.ndim == 3:
            x_true_array[k] = NR_data[0, j_nr, i_nr]
        else:
            x_true_array[k] = NR_data[j_nr, i_nr]

    # ---------------------------------------------------------
    # 第四步：执行 ELF 局地化系数计算
    # ---------------------------------------------------------
    print(f"正在基于 K={K} 的样本量计算 ELF Alpha...")
    alpha_estimated = calculate_elf_alpha(
        x_prior_ensemble, 
        y_prior_ensemble, 
        x_true_array, 
        y_true_array,  # <--- 这里传入的就是从 TXT 读出来的纯净真值！
        obs_err_variance_array
    )

    print("-" * 40)
    print(f"最终计算得到的经验局域化系数 (Alpha): {alpha_estimated:.6f}")