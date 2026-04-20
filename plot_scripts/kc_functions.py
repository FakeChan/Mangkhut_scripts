#some functions used 
#make sure you have excuted this module before you import it
import netCDF4 
from wrf import to_np,getvar
import numpy as np
import os
def nc_read1(filename,var):
    with netCDF4.Dataset(filename,'r') as ncfile:
        data = ncfile.variables[var][:].squeeze()
        return data

def getTClocation(file):
    """
    Get the location of the TC center from the slp field
    :param file: path to the netCDF file
    :return: tuple of (iTC, jTC) indices of the TC center
    """
    # Use netCDF4 to read the file
    data = netCDF4.Dataset(file)
    
    # Get the sea level pressure (slp) variable
    slp = to_np(getvar(data, 'slp', timeidx=0))
    
    # Find the minimum slp value and its indices
    slpMin = np.min(slp[:, :])
    indexMin = np.argwhere(slp[:, :] == slpMin)
    
    # Extract the i and j indices of the TC center
    jTC = indexMin[0][0]
    iTC = indexMin[0][1]
    
    return iTC, jTC

def match_typhoon_grid(file_A_path, file_B_path):
    """
    在文件B中找到距离文件A台风中心最近的格点 (Python版)
    
    输入:
        file_A_path : 参考文件路径 (用于确定台风中心经纬度)
        file_B_path : 目标文件路径 (需要在该文件中找对应格点)
    
    输出:
        b_lat      : 文件 B 中匹配点的纬度
        b_lon      : 文件 B 中匹配点的经度
        b_i_index  : 文件 B 中匹配点的 West-East 索引 (Grid X, 0-based)
        b_j_index  : 文件 B 中匹配点的 South-North 索引 (Grid Y, 0-based)
    """

    # --- 步骤 1: 读取文件 A 并定位台风中心 ---
    print(f'正在处理文件 A: {file_A_path} ...')
    
    if not os.path.exists(file_A_path) or not os.path.exists(file_B_path):
        raise FileNotFoundError('错误: 输入的文件路径不存在。')

    # 使用 context manager 自动关闭文件
    with netCDF4.Dataset(file_A_path, 'r') as nc_A:
        # 1.1 获取气压变量 (优先找 SLP, 否则用 PSFC)
        # netCDF4 的 variables 类似字典
        var_names_A = nc_A.variables.keys()
        
        # 简单的大小写和变量名匹配逻辑
        p_var_name = None
        if 'SLP' in var_names_A:
            p_var_name = 'SLP'
        elif 'slp' in var_names_A:
            p_var_name = 'slp'
        elif 'PSFC' in var_names_A:
            p_var_name = 'PSFC'
        else:
            raise ValueError(f'错误: 文件 A ({file_A_path}) 中未找到气压变量 (SLP, slp 或 PSFC)。')
            
        print(f'  > 使用变量: {p_var_name}')
        
        # 1.2 读取数据
        # [:] 读取所有数据为 numpy 数组
        p_data = nc_A.variables[p_var_name][:]
        lat_data = nc_A.variables['XLAT'][:]
        lon_data = nc_A.variables['XLONG'][:]
        
        # 处理维度: 如果是 3维 (Time, Lat, Lon), 取第一个时间层
        # 注意: Python NetCDF 通常是 (Time, South_North, West_East)
        if p_data.ndim == 3:
            p_data = p_data[0, :, :]
        if lat_data.ndim == 3:
            lat_data = lat_data[0, :, :]
        if lon_data.ndim == 3:
            lon_data = lon_data[0, :, :]
            
        # 1.3 寻找气压最小值的索引
        # argmin 返回的是扁平化后的线性索引
        min_p_idx_linear = np.argmin(p_data)
        # unravel_index 将线性索引转换为 (row, col) 坐标
        # row -> South-North (j), col -> West-East (i)
        min_idx_tuple = np.unravel_index(min_p_idx_linear, p_data.shape)
        
        # 获取中心经纬度和气压值
        min_p_val = p_data[min_idx_tuple]
        center_lat = lat_data[min_idx_tuple]
        center_lon = lon_data[min_idx_tuple]
        
        print(f'  > 文件 A 台风中心定位成功: {min_p_val:.2f} Pa')
        print(f'  > 目标坐标: Lat = {center_lat:.4f}, Lon = {center_lon:.4f}')


    # --- 步骤 2: 在文件 B 中寻找最近格点 ---
    print(f'正在处理文件 B: {file_B_path} ...')
    
    with netCDF4.Dataset(file_B_path, 'r') as nc_B:
        # 2.1 读取文件 B 的经纬度网格
        lat_data_B = nc_B.variables['XLAT'][:]
        lon_data_B = nc_B.variables['XLONG'][:]
        
        if lat_data_B.ndim == 3:
            lat_field_B = lat_data_B[0, :, :]
            lon_field_B = lon_data_B[0, :, :]
        else:
            lat_field_B = lat_data_B
            lon_field_B = lon_data_B
            
        # 2.2 计算距离 (欧几里得距离平方)
        dist_sq = (lat_field_B - center_lat)**2 + (lon_field_B - center_lon)**2
        
        # 2.3 找到最小距离的索引
        min_dist_idx_linear = np.argmin(dist_sq)
        min_idx_B_tuple = np.unravel_index(min_dist_idx_linear, dist_sq.shape)
        
        # min_idx_B_tuple 是 (row, col) 即 (South-North, West-East)
        b_j_index_0based = min_idx_B_tuple[0] # Row
        b_i_index_0based = min_idx_B_tuple[1] # Col
        
        # 提取结果
        b_lat = lat_field_B[min_idx_B_tuple]
        b_lon = lon_field_B[min_idx_B_tuple]
        
        # 输出
        # 注意: 这里返回的是 0-based 索引 (Python 标准)。
        # 如果需要和 MATLAB (1-based) 完全一致的数字用于打印，请在输出时 +1
        b_i_index = b_i_index_0based
        b_j_index = b_j_index_0based
        
        print(f'  > 文件 B 匹配成功。')
        print(f'  > 匹配点坐标: Lat = {b_lat:.4f}, Lon = {b_lon:.4f}')
        print(f'  > 网格索引 (0-based): X(W-E)={b_i_index}, Y(S-N)={b_j_index}')
        
        return b_lat, b_lon, b_i_index, b_j_index
    

def calculate_rmse(y_true, y_pred):
    # 确保输入是 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算误差平方的平均值，再开平方根
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return float(f"{rmse:.2g}")

def getNC_realT(nc_ds,level_index):
    data_values=nc_ds['THM'][0][level_index, :, :].values
    P_values = nc_ds['P'][0][level_index, :, :].values
    PB_values = nc_ds['PB'][0][level_index, :, :].values
    tot_p = P_values + PB_values
    theta = data_values+300
    R_Cp = 287.0 / 1004.0
    data_values = theta * ((tot_p / 100000.0) ** R_Cp)
    
    return data_values

