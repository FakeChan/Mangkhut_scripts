import numpy as np
import xarray as xr
import netCDF4
from wrf import getvar, to_np
from matplotlib import pyplot as plt
import os
from scipy.interpolate import griddata
import warnings

# 忽略一些警告（如wrf的meta数据警告）
warnings.filterwarnings("ignore")

#==========================================================
# 1. 移植并适配的工具函数 (来自 plot_EAKFvsQCF_anal.py)
#==========================================================

def getTClocation(nc_path):
    """
    使用 SLP 最小值定位台风中心 (适配版)
    """
    try:
        # 尝试使用 wrf-python 读取 (处理诊断变量更佳)
        with netCDF4.Dataset(nc_path) as ds:
            slp = getvar(ds, 'slp', timeidx=0)
            slp_np = to_np(slp)
            min_idx = np.unravel_index(np.argmin(slp_np), slp_np.shape)
            # 返回 (j, i) -> (lat_idx, lon_idx)
            return min_idx[0], min_idx[1]
    except Exception as e:
        print(f"Warning: Failed to use wrf-python for SLP, trying raw reading. Error: {e}")
        # 备用方案：直接读取 P 和 PB (如果需要) 或其他变量，这里简化处理
        raise e

def getTCnest_grid(NR_path, domain='d02'):
    """
    修改版的 getTCnest:
    只负责生成目标网格的经纬度坐标 (extract_lats, extract_lons) 和在 NR 中的索引 (grid_x, grid_y)。
    不再直接读取数据，以便在外部灵活处理 3D 变量 (TK, WSPD 等)。
    """
    # check the resolution of NR 
    ncdata = xr.open_dataset(NR_path)
    try:
        dx = ncdata.attrs['DX']
        # 放宽检查，仅打印警告
        if not np.isclose(dx, 300, atol=10.0): 
            print(f"Warning: Resolution of NR is {dx}, not strictly 300m as expected.")
    except KeyError:
        print("Warning: 'DX' attribute not found.")

    print(f'  - NR Resolution check: DX={dx}')
    
    # Get center index of TC in NR
    [jTC, iTC] = getTClocation(NR_path) # 注意 getTClocation 返回 (j, i)
    
    # Set size of grid
    # 逻辑保留原脚本设定：基于300m NR进行降采样
    if domain == 'd01':
        grid_step = 25
        half_grid_size = 90 / 7.5 
    elif domain == 'd02':
        grid_step = 5
        half_grid_size = 90 / 1.5
    else:
        raise ValueError(f"Error! domain should be d01/d02, instead of {domain}")
    
    # 计算偏移量 (Indices)
    offsets = (np.arange(-half_grid_size, half_grid_size + 1) * grid_step).astype(int)
    
    target_j = jTC + offsets
    target_i = iTC + offsets
    
    # 创建网格索引
    # 注意 meshgrid 的顺序: grid_x 对应 i (lon), grid_y 对应 j (lat)
    grid_x, grid_y = np.meshgrid(target_i, target_j)
    
    # 边界检查
    max_j, max_i = ncdata.dims['south_north'], ncdata.dims['west_east']
    grid_x = np.clip(grid_x, 0, max_i - 1)
    grid_y = np.clip(grid_y, 0, max_j - 1)

    # 提取目标网格的经纬度
    lats = ncdata['XLAT'][0].values
    lons = ncdata['XLONG'][0].values
    extract_lats = lats[grid_y, grid_x]
    extract_lons = lons[grid_y, grid_x]
    
    ncdata.close()
    
    return extract_lats, extract_lons, grid_x, grid_y

def interp_grid(grid_lats, grid_lons, input_data, input_lats, input_lons, method='linear'):
    """
    移植的插值函数: 将 input_data 插值到 grid_lats/grid_lons
    """
    # Flatten arrays
    points = np.column_stack((input_lons.flatten(), input_lats.flatten()))
    values = input_data.flatten()
    
    # Interpolate
    # 注意：griddata 对于大数组可能较慢。
    # grid_lons, grid_lats 是目标网格 (2D)
    interp_values = griddata(points, values, (grid_lons, grid_lats), method=method)
    return interp_values

#==========================================================
# 2. 核心数据处理与绘图逻辑
#==========================================================

def get_wrf_variable_3d(file_path, variable):
    """
    使用 wrf-python 读取 3D 变量 (自动处理 destagger 和诊断变量如 wspd, tk)
    """
    with netCDF4.Dataset(file_path) as ds:
        if variable == 'U': 
            data = getvar(ds, 'ua', timeidx=0)
        elif variable == 'V': 
            data = getvar(ds, 'va', timeidx=0)
        elif variable in ['T', 'tk']: 
            data = getvar(ds, 'tk', timeidx=0) # Temperature in Kelvin
        elif variable in ['Qv', 'QVAPOR']: 
            data = getvar(ds, 'QVAPOR', timeidx=0)
        elif variable == 'wspd':
            u = getvar(ds, 'ua', timeidx=0)
            v = getvar(ds, 'va', timeidx=0)
            data = np.sqrt(u**2 + v**2)
        else:
            # 尝试直接读取
            try:
                data = getvar(ds, variable, timeidx=0)
            except:
                raise ValueError(f"Variable {variable} not supported.")
                
        return to_np(data)

def get_lat_lon(file_path, variable):
    """
    获取文件的经纬度网格 (2D)
    针对 U/V 变量可能需要处理交错网格，但 getvar('ua') 已经去交错，
    所以通常直接用 XLAT, XLONG 即可。
    """
    with netCDF4.Dataset(file_path) as ds:
        # 使用 wrf.getvar 获取 lat/lon，确保与数据形状一致
        # 简单起见，读取 XLAT, XLONG
        lat = to_np(getvar(ds, 'XLAT', timeidx=0))
        lon = to_np(getvar(ds, 'XLONG', timeidx=0))
    return lat, lon

def plot_rmse_profiles_v2(files, analysis_names, variable, output_path, domain='d02'):
    print(f"\n--- Processing Variable: '{variable}' ---")

    # 1. 准备 NR (Truth) 数据和目标网格
    # ------------------------------------------------
    print(f"  1. Locating TC in NR and generating target grid ({files['nr']})...")
    # 获取目标网格坐标和 NR 中的索引
    target_lats, target_lons, nr_idx_x, nr_idx_y = getTCnest_grid(files['nr'], domain=domain)
    
    # 读取 NR 3D 数据
    print(f"  2. Reading NR 3D data...")
    nr_data_full = get_wrf_variable_3d(files['nr'], variable)
    
    # 切片 NR 数据 (利用 getTCnest 计算的索引)
    # 假设数据维度是 [level, lat, lon]
    data_nr_sliced = nr_data_full[:, nr_idx_y, nr_idx_x]
    
    num_levels = data_nr_sliced.shape[0]
    vertical_levels = np.arange(num_levels)
    print(f"     NR sliced shape: {data_nr_sliced.shape}")

    # 2. 处理 Analysis 文件 (插值到 NR 网格)
    # ------------------------------------------------
    anal_keys = ['fg', 'an1', 'an2']
    rmse_results = {}

    for key in anal_keys:
        filepath = files[key]
        name = analysis_names.get(key, 'First Guess' if key == 'fg' else key)
        print(f"  3. Processing {name}...")
        
        # 读取 Analysis 全场数据
        anal_data_full = get_wrf_variable_3d(filepath, variable)
        anal_lats, anal_lons = get_lat_lon(filepath, variable)
        
        # 准备插值结果容器
        anal_data_interp = np.zeros_like(data_nr_sliced)
        
        # 逐层插值 (因为 griddata 是 2D 插值)
        # 优化：如果网格完全一致且无需插值（例如同分辨率同区域），可以直接切片，
        # 但既然要求使用 interp_grid 逻辑，我们强制执行插值以处理位置偏差。
        print(f"     Interpolating {num_levels} levels (this may take a while)...")
        for k in range(num_levels):
            # 简单的进度显示
            if k % 10 == 0: print(f"       Level {k}/{num_levels}", end='\r')
            
            level_data = anal_data_full[k, :, :]
            
            # 调用 interp_grid
            interped_slice = interp_grid(
                grid_lats=target_lats,
                grid_lons=target_lons,
                input_data=level_data,
                input_lats=anal_lats,
                input_lons=anal_lons,
                method='linear' # 线性插值
            )
            anal_data_interp[k, :, :] = interped_slice
            
        print(f"       Done.")
        
        # 计算 RMSE
        error_sq = (anal_data_interp - data_nr_sliced)**2
        rmse = np.sqrt(np.nanmean(error_sq, axis=(1, 2))) # 忽略 NaN (插值边界可能产生 NaN)
        rmse_results[key] = rmse

    # 3. 绘图
    # ------------------------------------------------
    print("  4. Generating plot...")
    fig, ax = plt.subplots(figsize=(6, 8))
    
    ax.plot(rmse_results['fg'], vertical_levels, 'k--', label='First Guess', linewidth=2)
    ax.plot(rmse_results['an1'], vertical_levels, 'r-', label=analysis_names['an1'], linewidth=2)
    ax.plot(rmse_results['an2'], vertical_levels, 'b-', label=analysis_names['an2'], linewidth=2)
    
    ax.set_xlabel(f'RMSE ({variable})', fontsize=12)
    ax.set_ylabel('Vertical Level', fontsize=12)
    ax.set_title(f'RMSE Profile: {variable}', fontsize=14, fontweight='bold')
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 保存
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"--- Saved to: {output_path} ---")
    plt.close(fig)

#==========================================================
# 3. 执行入口
#==========================================================
if __name__ == "__main__":
    # 文件路径
    FILES = {
        'nr':  '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d03_2018-09-10_00:00:00',
        'fg':  '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/run_dir/EAKF/preassim_mean_d02.nc',
        'an1': '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/run_dir/EAKF/postassim_mean_d02.nc',
        'an2': '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/run_dir/QCF_RHF/postassim_mean_d02.nc'
    }
    ANALYSIS_NAMES = {'an1': 'EAKF', 'an2': 'QCF_RHF'}
    
    # 任务列表
    TASKS = [
        {'variable': 'T'},
        {'variable': 'Qv'},
        {'variable': 'wspd'}
    ]

    # 输出目录
    FIGS_BASE_DIR = '/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/figs/'

    # 运行
    for task in TASKS:
        var_name = task['variable']
        out_name = f"RMSE_profile_{var_name}_interp.png"
        out_path = os.path.join(FIGS_BASE_DIR, out_name)
        
        try:
            plot_rmse_profiles_v2(FILES, ANALYSIS_NAMES, var_name, out_path, domain='d02')
        except Exception as e:
            print(f"Error processing {var_name}: {e}")
            import traceback
            traceback.print_exc()