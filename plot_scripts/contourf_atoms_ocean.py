import xarray as xr
import netCDF4
from wrf import getvar, to_np
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import subprocess
import cmocean
import os
from param import *
#==========================================================
# 1. 统一的数据提取函数 (已更新)
#==========================================================

def extract_data(file_path, variable, level_type, level_value, center_j, center_i, radius):
    """
    智能数据提取函数：现在所有类型都基于整数索引。
    - level_type='atm_index': 按大气模式层索引提取 (使用 wrf-python)。
    - level_type='ocean_index': 按海洋模式层索引提取 (使用 xarray)。
    """
    j_start, j_end = center_j - radius, center_j + radius
    i_start, i_end = center_i - radius, center_i + radius

    if level_type == 'atm_index':
        # --- 大气场提取逻辑 ---
        with netCDF4.Dataset(file_path) as ds:
            if variable in ['U', 'ua']: field_3d = to_np(getvar(ds, 'ua', timeidx=0))
            elif variable in ['V', 'va']: field_3d = to_np(getvar(ds, 'va', timeidx=0))
            elif variable in ['T', 'tk']: field_3d = to_np(getvar(ds, 'tk', timeidx=0))
            elif variable in ['Qv', 'QVAPOR']: field_3d = to_np(getvar(ds, 'QVAPOR', timeidx=0))
            else: raise ValueError(f"不支持的大气变量: {variable}")
            
            field_level = field_3d[level_value, :, :]
            # temporarily commit to compute domain averaged RMSE
            # return field_level[j_start:j_end, i_start:i_end]
              
            return field_level

    elif level_type == 'ocean_index':
        # --- 海洋场提取逻辑 (遵照您的代码) ---
        with xr.open_dataset(file_path) as ds:
            if variable not in ds.variables:
                raise ValueError(f"变量 '{variable}' 在文件 {file_path} 中未找到。")
            
            da = ds[variable]
            # 假设维度顺序是 [time, level, j, i]
            # 使用 level_value (即 ocean_level) 作为索引直接切片
            # data_subset = da[0, level_value, j_start:j_end, i_start:i_end]
            data_subset = da[0, level_value, :, :]
            return data_subset.values
    else:
        raise ValueError(f"未知的 level_type: '{level_type}'。请选择 'atm_index' 或 'ocean_index'。")

#==========================================================
# 2. 统一的绘图函数 (无变动)
#==========================================================

def plot_comparison(files, analysis_names, variable, level_type, level_value, radius, output_path, domain_name):
    """
    通用的对比绘图函数，适用于大气和海洋。
    """
    print(f"\n--- 开始绘制 {domain_name} 场: {variable} ---")
    
    print(f"从 {os.path.basename(files['nr'])} 中确定系统中心...")
    jTC, iTC = get_tc_location(files['nr'])
    print(f"系统中心索引: (j={jTC}, i={iTC})")

    print(f"正在提取 {variable} 在 {level_type.split('_')[-1]} = {level_value} 的数据...")
    data = {key: extract_data(path, variable, level_type, level_value, jTC, iTC, radius) for key, path in files.items()}
    for key,value in data.items():
        if key=='nr':
            mask = truth_mask_dic['d02->d01']
            #all values should be 2d
            data[key] = value[mask,mask]
        else:
            mask_j=anal_mask_dic['d01_j']
            mask_i=anal_mask_dic['d01_i']
            data[key]=value[mask_j,mask_i]
    increment1 = data['an1'] - data['nr']
    increment2 = data['an2'] - data['nr']
    
    vmin_abs = min(data['nr'].min(), data['fg'].min(), data['an1'].min(), data['an2'].min())
    vmax_abs = max(data['nr'].max(), data['fg'].max(), data['an1'].max(), data['an2'].max())
    levels_abs = np.linspace(vmin_abs, vmax_abs, 16)
    cmap_abs = cmocean.cm.thermal
    
    vmax_inc = max(np.abs(increment1).max(), np.abs(increment2).max()) if increment1.size > 0 else 0.1
    vmin_inc = -vmax_inc
    levels_inc = np.linspace(vmin_inc, vmax_inc, 17)
    cmap_inc = cmocean.cm.balance

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 14))
    title_suffix = f"at {level_type.replace('_', ' ').title()} {level_value}"
    fig.suptitle(f'{domain_name}: {variable} {title_suffix} (Radius: {radius})', fontsize=22, y=0.97)
    ax_list = axes.flat
    
    cf_nr = ax_list[0].contourf(data['nr'], cmap=cmap_abs, levels=levels_abs, vmin=vmin_abs, vmax=vmax_abs)
    ax_list[0].set_title('Nature Run (Truth)', fontsize=16)
    
    cf_an1 = ax_list[1].contourf(data['an1'], cmap=cmap_abs, levels=levels_abs, vmin=vmin_abs, vmax=vmax_abs)
    ax_list[1].set_title(f"Analysis ({analysis_names['an1']})", fontsize=16)
    
    cf_an2 = ax_list[2].contourf(data['an2'], cmap=cmap_abs, levels=levels_abs, vmin=vmin_abs, vmax=vmax_abs)
    ax_list[2].set_title(f"Analysis ({analysis_names['an2']})", fontsize=16)
    add_cbar(fig, ax_list[2], cf_an2, label=f'{variable} Value')

    ax_list[3].contourf(data['fg'], cmap=cmap_abs, levels=levels_abs, vmin=vmin_abs, vmax=vmax_abs)
    ax_list[3].set_title('First Guess (Prior)', fontsize=16)
    
    ax_list[4].contourf(increment1, cmap=cmap_inc, levels=levels_inc, vmin=vmin_inc, vmax=vmax_inc)
    ax_list[4].set_title(f"Error ({analysis_names['an1']} - NR)", fontsize=16)
    
    cf_inc2=ax_list[5].contourf(increment2, cmap=cmap_inc, levels=levels_inc, vmin=vmin_inc, vmax=vmax_inc)
    ax_list[5].set_title(f"Error ({analysis_names['an2']} - NR)", fontsize=16)
    add_cbar(fig, ax_list[5], cf_inc2, label='Increment')

    for ax in ax_list:
        ax.set_xlabel('X Grid Index'); ax.set_ylabel('Y Grid Index')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 0.94, 0.94])
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"--- 图像已保存至: {output_path} ---")
    plt.close(fig)

#==========================================================
# 3. 通用辅助函数 (无变动)
#==========================================================
def add_cbar(fig, ax, pcm, label=''):
    pad, size = 0.02, 0.05
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + pad, pos.y0, size * pos.width, pos.height])
    cbar = fig.colorbar(pcm, cax=cax, label=label)
    cbar.locator = ticker.MaxNLocator(nbins=7)
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()

def get_tc_location(nc_file):
    with netCDF4.Dataset(nc_file) as ds:
        slp = getvar(ds, 'slp', timeidx=0)
        slp_min_idx = np.unravel_index(np.argmin(to_np(slp)), slp.shape)
        return slp_min_idx[0], slp_min_idx[1]

#==========================================================
# 4. 脚本执行入口 (任务控制面板 - 已更新)
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
    atm_pres='500hpa'
    # --- b. 绘图任务列表 (已更新) ---
    # 在此列表中添加或修改您想绘制的任何图形
    TASKS = [
        {'domain_name': 'Atmosphere', 'variable': 'T',      'level_type': 'atm_index',   'level_value': plev_dict[atm_pres]},
        {'domain_name': 'Atmosphere', 'variable': 'U',      'level_type': 'atm_index',   'level_value': plev_dict[atm_pres]},
        {'domain_name': 'Atmosphere', 'variable': 'QVAPOR',      'level_type': 'atm_index',   'level_value': plev_dict[atm_pres]},
        # !! 海洋场现在使用 'ocean_index' 和整数索引 !!
        {'domain_name': 'Ocean',      'variable': 'OM_TMP', 'level_type': 'ocean_index', 'level_value': 0}, # 0 = 表层
        {'domain_name': 'Ocean',      'variable': 'OM_S',   'level_type': 'ocean_index', 'level_value': 0},
        {'domain_name': 'Ocean',      'variable': 'OM_U',   'level_type': 'ocean_index', 'level_value': 0},
        {'domain_name': 'Ocean',      'variable': 'OM_V',   'level_type': 'ocean_index', 'level_value': 0},
        # {'domain_name': 'Ocean',      'variable': 'OM_TMP', 'level_type': 'ocean_index', 'level_value': 5}, # 示例: 第6层
    ]

    # --- c. 全局参数 ---
    RADIUS_GRID_POINTS = 30
    FIGS_BASE_DIR = '/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/figs/'

    # --- d. 执行所有任务 ---
    for task in TASKS:
        level_str = str(abs(task['level_value']))
        output_dir = os.path.join(FIGS_BASE_DIR, task['domain_name'].lower())
        if task['domain_name']=='Atmosphere':
            output_filename = f"{task['variable']}_{task['level_type'].split('_')[-1]}{level_str}_{atm_pres}_r{RADIUS_GRID_POINTS}.png"
        elif task['domain_name']=='Ocean':
            output_filename = f"{task['variable']}_{task['level_type'].split('_')[-1]}{level_str}_r{RADIUS_GRID_POINTS}.png"
        output_path = os.path.join(output_dir, output_filename)

        plot_comparison(
            files=FILES,
            analysis_names=ANALYSIS_NAMES,
            variable=task['variable'],
            level_type=task['level_type'],
            level_value=task['level_value'],
            radius=RADIUS_GRID_POINTS,
            output_path=output_path,
            domain_name=task['domain_name']
        )