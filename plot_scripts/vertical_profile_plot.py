# plot_vertical_profile.py

import xarray as xr
import netCDF4
from wrf import getvar, to_np, get_cartopy, latlon_coords
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import cmocean
import os
from param import *

# ==========================================================
# 1. 新的数据提取函数 (用于垂直剖面)
# ==========================================================

def extract_vertical_profile(key,file_path, variable, center_j, center_i, radius, cross_section_axis='i'):
    """
    提取穿过TC中心的垂直剖面数据。
    - cross_section_axis='i': 提取东西向 (W-E) 的垂直剖面。
    - cross_section_axis='j': 提取南北向 (N-S) 的垂直剖面。
    返回一个2D数组 [vertical_levels, horizontal_distance]。
    """
    with netCDF4.Dataset(file_path) as ds:
        # 使用 wrf-python 获取3D变量场
        if variable in ['U', 'ua']: field_3d = to_np(getvar(ds, 'ua', timeidx=0))
        elif variable in ['V', 'va']: field_3d = to_np(getvar(ds, 'va', timeidx=0))
        elif variable in ['T', 'tk']: field_3d = to_np(getvar(ds, 'tk', timeidx=0))
        elif variable in ['Qv', 'QVAPOR']: field_3d = to_np(getvar(ds, 'QVAPOR', timeidx=0))
        else: raise ValueError(f"不支持的大气变量: {variable}")
        if key=='nr':
            slice_mask_j=truth_mask_dic['d02->d01']
            slice_mask_i=truth_mask_dic['d02->d01']
        else:
            slice_mask_j=anal_mask_dic['d01_j']
            slice_mask_i=anal_mask_dic['d01_i']
            center_j,center_i=mask_j_i_to_d01(center_j,center_i,'d02')
        # 根据指定的轴向提取剖面
        if cross_section_axis == 'i': # 东西向剖面
            i_start, i_end = center_i - radius, center_i + radius
            profile_2d = field_3d[:, center_j, slice_mask_i]
        elif cross_section_axis == 'j': # 南北向剖面
            j_start, j_end = center_j - radius, center_j + radius
            profile_2d = field_3d[:, slice_mask_j, center_i]
        else:
            raise ValueError("cross_section_axis 必须是 'i' 或 'j'")
            
        return profile_2d

# ==========================================================
# 2. 新的绘图函数 (用于垂直剖面)
# ==========================================================

def plot_vertical_comparison(files, analysis_names, variable, radius, output_path, cross_section_axis='i'):
    """
    绘制并比较不同分析场的垂直剖面图。
    """
    print(f"\n--- 开始绘制垂直剖面图: {variable} ({'W-E' if cross_section_axis=='i' else 'N-S'} cross-section) ---")
    
    print(f"从 {os.path.basename(files['nr'])} 中确定系统中心...")
    jTC, iTC = get_tc_location(files['nr'])
    print(f"系统中心索引: (j={jTC}, i={iTC})")

    print(f"正在提取 {variable} 的垂直剖面数据...")
    
    # 提取数据
    data = {key: extract_vertical_profile(key,path, variable, jTC, iTC, radius, cross_section_axis) for key, path in files.items()}
    
    # 注意: 这里的masking逻辑可能需要根据垂直剖面的维度进行调整，暂时简化处理
    # 如果需要对垂直剖面进行domain的裁剪，需要定义新的masking逻辑
    increment1 = data['an1'] - data['fg']
    increment2 = data['an2'] - data['fg']
    
    # 动态设定颜色范围
    vmin_abs = min(data['nr'].min(), data['fg'].min(), data['an1'].min(), data['an2'].min())
    vmax_abs = max(data['nr'].max(), data['fg'].max(), data['an1'].max(), data['an2'].max())
    levels_abs = np.linspace(vmin_abs, vmax_abs, 16)
    cmap_abs = cmocean.cm.thermal
    
    vmax_inc = max(np.abs(increment1).max(), np.abs(increment2).max())
    vmin_inc = -vmax_inc
    levels_inc = np.linspace(vmin_inc, vmax_inc, 17)
    cmap_inc = cmocean.cm.balance

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 14))
    title_suffix = f"({ 'W-E' if cross_section_axis == 'i' else 'N-S'} cross-section through TC center)"
    fig.suptitle(f'Vertical Profile: {variable} {title_suffix} (Radius: {radius} grid points)', fontsize=22, y=0.97)
    ax_list = axes.flat
    
    # 准备坐标轴
    if cross_section_axis=='i':
        slice_mask=anal_mask_dic['d01_i']
        x_coords = np.arange(slice_mask.start,slice_mask.stop,slice_mask.step)
    elif cross_section_axis=='j':
        slice_mask=anal_mask_dic['d01_j']
        x_coords = np.arange(slice_mask.start,slice_mask.stop,slice_mask.step)
    num_vertical_levels = data['nr'].shape[0]
    y_coords = np.arange(num_vertical_levels) # 暂时使用模式层索引作为Y轴
    
    # 绘图
    cf_nr = ax_list[0].contourf(x_coords, y_coords, data['nr'], cmap=cmap_abs, levels=levels_abs)
    ax_list[0].set_title('Nature Run (Truth)', fontsize=16)
    
    cf_an1 = ax_list[1].contourf(x_coords, y_coords, data['an1'], cmap=cmap_abs, levels=levels_abs)
    ax_list[1].set_title(f"Analysis ({analysis_names['an1']})", fontsize=16)
    
    cf_an2 = ax_list[2].contourf(x_coords, y_coords, data['an2'], cmap=cmap_abs, levels=levels_abs)
    ax_list[2].set_title(f"Analysis ({analysis_names['an2']})", fontsize=16)
    add_cbar(fig, ax_list[2], cf_an2, label=f'{variable} Value')

    ax_list[3].contourf(x_coords, y_coords, data['fg'], cmap=cmap_abs, levels=levels_abs)
    ax_list[3].set_title('First Guess (Prior)', fontsize=16)
    
    ax_list[4].contourf(x_coords, y_coords, increment1, cmap=cmap_inc, levels=levels_inc)
    ax_list[4].set_title(f"Increment ({analysis_names['an1']} - FG)", fontsize=16)
    
    cf_inc2 = ax_list[5].contourf(x_coords, y_coords, increment2, cmap=cmap_inc, levels=levels_inc)
    ax_list[5].set_title(f"Increment ({analysis_names['an2']} - FG)", fontsize=16)
    add_cbar(fig, ax_list[5], cf_inc2, label='Increment')

    for ax in ax_list:
        ax.set_xlabel('Horizontal Distance from TC center (grid points)')
        ax.set_ylabel('Vertical Model Level Index')
        ax.grid(True, linestyle='--', alpha=0.5)
        # 可以考虑将Y轴从模式层索引转换为气压或高度
        # ax.invert_yaxis() # 气象上通常将高层放在上面

    plt.tight_layout(rect=[0, 0, 0.94, 0.94])
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"--- 图像已保存至: {output_path} ---")
    plt.close(fig)

# ==========================================================
# 3. 通用辅助函数 (无变动)
# ==========================================================
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

# ==========================================================
# 4. 脚本执行入口 (任务控制面板)
# ==========================================================
if __name__ == "__main__":
    # --- a. 文件路径设置 (与原脚本保持一致) ---
    FILES = {
        'nr':  '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d02_2018-09-10_06:00:00',
        'fg':  '/share/home/lililei1/kcfu/tc_mangkhut/6forecast_only_stable/output/10_00_00/firstguess.ensmean.100600',
        'an1': '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_06_00/EAKF/firstguess.ensmean',
        'an2': '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_06_00/QCF_RHF/firstguess.ensmean'
    }
    ANALYSIS_NAMES = {'an1': 'EAKF', 'an2': 'QCF_RHF'}

    # --- b. 绘图任务列表 (修改为垂直剖面任务) ---
    #axis: j -> north-south
    #      i -> east-west
    TASKS = [
        {'variable': 'T', 'cross_section_axis': 'j'},  # 
        {'variable': 'U', 'cross_section_axis': 'j'},  # 
        {'variable': 'QVAPOR', 'cross_section_axis': 'j'} # 
    ]

    # --- c. 全局参数 ---
    RADIUS_GRID_POINTS = 50  # 垂直剖面可以取更大半径以显示更广范围
    FIGS_BASE_DIR = '/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/figs/vertical_profiles/'

    # --- d. 执行所有任务 ---
    for task in TASKS:
        output_dir = os.path.join(FIGS_BASE_DIR, task['variable'])
        output_filename = f"vertical_profile_{task['variable']}_{task['cross_section_axis']}_axis_r{RADIUS_GRID_POINTS}.png"
        output_path = os.path.join(output_dir, output_filename)

        plot_vertical_comparison(
            files=FILES,
            analysis_names=ANALYSIS_NAMES,
            variable=task['variable'],
            radius=RADIUS_GRID_POINTS,
            output_path=output_path,
            cross_section_axis=task['cross_section_axis']
        )