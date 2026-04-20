import xarray as xr
import numpy as np
import netCDF4
import os
# --- 用户设置 ---
# 输入文件路径

work_dir=os.getenv('current_work_dir')
time_str=os.getenv('this_time')
mem_str=os.getenv('member')
file_background = f'fg'  # 未扰动的分析场
file_perturbation = f'diff' # 来自random_cv的扰动文件

# 输出文件路径 (保留您原有的逻辑)
file_output = f'wrfinput_2018091200_mem001_tapperd' # 最终的、边界扰动被削弱的初始场 (netCDF4格式)

# 定义边界缓冲区的宽度（单位：格点数）
# 例如，20表示从最外层边界向内20个格点为过渡区
buffer_grid_points = 20

# --- 定义需要应用掩码的扰动变量列表 ---
# 根据变量所在的网格点进行分类
VARS_PERT_MASS = ['T','THM', 'P','PH', 'QVAPOR', 'MU', 'PSFC'] # 位于格点中心的变量
VARS_PERT_U = ['U'] # U-staggered 变量
VARS_PERT_V = ['V'] # V-staggered 变量

# --- 脚本开始 ---

print("正在打开文件...")
ds_bg = xr.open_dataset(file_background)
ds_pert = xr.open_dataset(file_perturbation)

# --- 获取所有需要的维度信息 ---
# 非交错网格 (mass points)
nx = ds_bg.dims['west_east']
ny = ds_bg.dims['south_north']
# U-交错网格
nx_stag = ds_bg.dims['west_east_stag']
# V-交错网格
ny_stag = ds_bg.dims['south_north_stag']

print(f"Domain 尺寸 (Mass): nx={nx}, ny={ny}")
print(f"Domain 尺寸 (U-stag): nx_stag={nx_stag}")
print(f"Domain 尺寸 (V-stag): ny_stag={ny_stag}")
print(f"边界缓冲区宽度: {buffer_grid_points} 格点")


# --- 创建权重向量和三种不同的2D掩码 ---
def create_taper_1d(total_len, buffer_len):
    """创建一维的余弦 taper 权重"""
    if buffer_len * 2 > total_len:
        raise ValueError("缓冲区宽度过大，导致中心区域消失！请减小 buffer_grid_points。")
    weights = np.ones(total_len)
    taper = 0.5 * (1 + np.cos(np.linspace(np.pi, 0, buffer_len))) # 从 0 到 1 的余弦过渡
    weights[:buffer_len] = taper
    weights[-buffer_len:] = taper[::-1]
    return weights

# 1. 为质量点创建掩码
weights_x = create_taper_1d(nx, buffer_grid_points)
weights_y = create_taper_1d(ny, buffer_grid_points)
mask_mass = xr.DataArray(np.outer(weights_y, weights_x), dims=['south_north', 'west_east'])

# 2. 为U点创建掩码
weights_x_stag = create_taper_1d(nx_stag, buffer_grid_points)
mask_u = xr.DataArray(np.outer(weights_y, weights_x_stag), dims=['south_north', 'west_east_stag'])

# 3. 为V点创建掩码
weights_y_stag = create_taper_1d(ny_stag, buffer_grid_points)
mask_v = xr.DataArray(np.outer(weights_y_stag, weights_x), dims=['south_north_stag', 'west_east'])

print("已为 mass, u, v 三种网格点创建了专用的权重掩码。")


# --- 应用掩码并生成输出文件 ---
print("开始应用掩码到扰动变量...")
# 从背景场开始，创建一个用于输出的数据集副本
ds_out = ds_bg.copy(deep=True)

# 将所有要处理的变量合并到一个列表
all_vars_to_taper = VARS_PERT_MASS + VARS_PERT_U + VARS_PERT_V

for pert_var in all_vars_to_taper:
    # 检查扰动文件中是否存在该变量
    if pert_var not in ds_pert:
        print(f"提示: 扰动变量 '{pert_var}' 不在扰动文件中，跳过。")
        continue

    # 确定对应的背景场变量名 (例如 U_PERT -> U)
    bg_var = pert_var.replace('_PERT', '')
    if bg_var not in ds_out:
        print(f"警告: 背景变量 '{bg_var}' 不在背景文件中，无法添加扰动 '{pert_var}'。")
        continue

    print(f"正在处理: {pert_var} -> {bg_var}")
    
    pert_field = ds_pert[pert_var]
    
    # 根据变量类型选择正确的掩码
    if pert_var in VARS_PERT_U:
        selected_mask = mask_u
    elif pert_var in VARS_PERT_V:
        selected_mask = mask_v
    else: # 默认为质量点变量
        selected_mask = mask_mass

    # 应用掩码 (xarray 会自动将2D掩码广播到3D/4D变量)
    tapered_pert = pert_field * selected_mask
    
    # 将加权后的扰动加到背景场上
    ds_out[bg_var] =ds_bg[bg_var] + tapered_pert


print("所有变量处理完毕。正在保存输出文件...")
# 保存为 NetCDF4 格式
ds_out.to_netcdf(file_output, mode='a')
ds_out.close()
print(f"成功创建文件: {file_output}，变量维度已根据交错网格正确处理。")