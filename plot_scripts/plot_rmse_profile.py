from param import *
import xarray as xr
import netCDF4
from wrf import getvar, to_np
import numpy as np
from matplotlib import pyplot as plt
import os

#==========================================================
# 1. 核心功能函数
#==========================================================

def get_tc_location(nc_file):
    """
    通过海平面气压（SLP）最小值确定台风中心的位置。
    """
    with netCDF4.Dataset(nc_file) as ds:
        slp = getvar(ds, 'slp', timeidx=0)
        slp_min_idx = np.unravel_index(np.argmin(to_np(slp)), slp.shape)
        return slp_min_idx[0], slp_min_idx[1] # 返回 (j, i) 索引

def extract_3d_data_around_tc(file_path, variable, radius):
    """
    智能3D数据提取函数:
    1. 在当前文件中定位台风中心。
    2. 围绕该中心提取一个三维数据立方体。
    """
    print(f"  - Reading {variable} and finding TC center in {os.path.basename(file_path)}...")
    center_j, center_i = get_tc_location(file_path)
    
    j_start, j_end = center_j - radius, center_j + radius
    i_start, i_end = center_i - radius, center_i + radius
    
    # 根据变量名判断是大气场还是海洋场
    if variable.startswith('OM_'):
        # --- 海洋场提取逻辑 (使用 xarray) ---
        with xr.open_dataset(file_path) as ds:
            if variable not in ds.variables:
                raise ValueError(f"变量 '{variable}' 在文件 {file_path} 中未找到。")
            
            da = ds[variable]
            # 假设维度顺序是 [time, level, j, i]
            data_subset = da[0, :, j_start:j_end, i_start:i_end]
            return data_subset.values
            # return da.values
    else:
        # --- 大气场提取逻辑 (使用 wrf-python) ---
        with netCDF4.Dataset(file_path) as ds:
            if variable in ['U', 'ua']: field_3d = to_np(getvar(ds, 'ua', timeidx=0))
            elif variable in ['V', 'va']: field_3d = to_np(getvar(ds, 'va', timeidx=0))
            elif variable in ['T', 'tk']: field_3d = to_np(getvar(ds, 'tk', timeidx=0))
            elif variable in ['Qv', 'QVAPOR']: field_3d = to_np(getvar(ds, 'QVAPOR', timeidx=0))
            elif variable in ['wspd']: field_3d=np.sqrt(to_np(getvar(ds,'ua',timeidx=0))**2+to_np(getvar(ds,'va',timeidx=0))**2)
            else: raise ValueError(f"不支持的大气变量: {variable}")
            
            return field_3d[:, j_start:j_end, i_start:i_end]
            # return field_3d

def plot_rmse_profiles(files, analysis_names, variable, output_path):
    """
    计算并绘制RMSE垂直剖面图。
    """
    print(f"\n--- 开始为变量 '{variable}' 计算并绘制RMSE剖面图 ---")

    # 1. 提取各个场的三维数据 (每个场围绕自己的TC中心)
    data_nr = extract_3d_data_around_tc(files['nr'], variable, RADIUS_GRID_POINTS)[:,truth_mask_dic['d02->d01'],truth_mask_dic['d02->d01']]
    data_fg = extract_3d_data_around_tc(files['fg'], variable, RADIUS_GRID_POINTS)[:,anal_mask_dic['d01_j'],anal_mask_dic['d01_i']]
    data_an1 = extract_3d_data_around_tc(files['an1'], variable, RADIUS_GRID_POINTS)[:,anal_mask_dic['d01_j'],anal_mask_dic['d01_i']]
    data_an2 = extract_3d_data_around_tc(files['an2'], variable, RADIUS_GRID_POINTS)[:,anal_mask_dic['d01_j'],anal_mask_dic['d01_i']]
    
    num_levels = data_nr.shape[0]
    vertical_levels = np.arange(num_levels)
    
    # 2. 逐层计算RMSE
    print("  - Calculating RMSE at each vertical level...")
    rmse_fg, rmse_an1, rmse_an2 = [], [], []

    for k in range(num_levels):
        # 真值
        truth_slice = data_nr[k, :, :]
        # print(f'size of truth_slice: {np.size(truth_slice)}')
        # print(f'size of data_fg: {np.size(data_fg[k, :, :])}')
        # 计算 FG vs NR 的RMSE
        error_sq_fg = (data_fg[k, :, :] - truth_slice)**2
        
        rmse_fg.append(np.sqrt(error_sq_fg.mean()))
        
        # 计算 AN1 vs NR 的RMSE
        error_sq_an1 = (data_an1[k, :, :] - truth_slice)**2
        rmse_an1.append(np.sqrt(error_sq_an1.mean()))

        # 计算 AN2 vs NR 的RMSE
        error_sq_an2 = (data_an2[k, :, :] - truth_slice)**2
        rmse_an2.append(np.sqrt(error_sq_an2.mean()))

    # 3. 绘图
    print("  - Generating plot...")
    fig, ax = plt.subplots(figsize=(8, 10))
    
    ax.plot(rmse_fg, vertical_levels, 'k--', label='First Guess', linewidth=2)
    ax.plot(rmse_an1, vertical_levels, 'r-', label=analysis_names['an1'], linewidth=2)
    ax.plot(rmse_an2, vertical_levels, 'b-', label=analysis_names['an2'], linewidth=2)
    
    ax.set_xlabel('Root Mean Square Error (RMSE)', fontsize=12)
    ax.set_ylabel('Vertical Level Index', fontsize=12)
    ax.set_title(f'RMSE Profile for {variable}', fontsize=16, fontweight='bold')
    
    # 通常大气廓线图将0层（高层）放在顶部，海洋廓线图将0层（表层）放在顶部
    # ax.invert_yaxis()
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 4. 保存图像
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"--- 剖面图已保存至: {output_path} ---")
    plt.close(fig)

#==========================================================
# 2. 脚本执行入口 (任务控制面板)
#==========================================================
if __name__ == "__main__":
    # --- a. 文件路径设置 (统一) ---
    FILES = {
        'nr':  '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d02_2018-09-10_06:00:00',
        'fg':  '/share/home/lililei1/kcfu/tc_mangkhut/6forecast_only_stable/output/10_00_00/firstguess.ensmean.100600',
        'an1': '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_06_00/EAKF/firstguess.ensmean',
        'an2': '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_06_00/QCF_RHF/firstguess.ensmean'
    }
    ANALYSIS_NAMES = {'an1': 'EAKF', 'an2': 'QCF_RHF'}
    # --- b. RMSE剖面图绘制任务列表 ---
    # 在此列表中指定您想为哪些变量绘制RMSE剖面图
    TASKS = [
        {'variable': 'T'},
        {'variable': 'Qv'},
        {'variable':'wspd'}
    ]

    # --- c. 全局参数 ---
    RADIUS_GRID_POINTS = 30
    FIGS_BASE_DIR = '/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/figs/'

    # --- d. 执行所有任务 ---
    for task in TASKS:
        variable_name = task['variable']
        
        # 自动生成输出路径
        output_dir = os.path.join(FIGS_BASE_DIR)
        output_filename = f"RMSE_profile_{variable_name}_r{RADIUS_GRID_POINTS}.png"
        output_path = os.path.join(output_dir, output_filename)

        # 调用统一的绘图函数
        plot_rmse_profiles(
            files=FILES,
            analysis_names=ANALYSIS_NAMES,
            variable=variable_name,
            output_path=output_path,
        )
