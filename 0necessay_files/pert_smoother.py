import xarray as xr
import numpy as np
import os


work_dir=os.getenv('current_work_dir')
time_str=os.getenv('this_time')
mem_str=os.getenv('member')
file_background = f'{work_dir}/fg'  # 未扰动的分析场
file_perturbation = f'{work_dir}/diff' # 来自random_cv的扰动文件

# 输出文件路径
file_output = f'{work_dir}/wrfinput_{time_str}_{mem_str}_tapperd' # 最终的、边界扰动被削弱的初始场 (netCDF4格式)
# --- 高斯函数参数设置 ---
# 定义高斯函数的标准差(sigma)，单位为格点数。
# 它控制扰动衰减的速度。值越大，扰动区域越小。
# 一个好的初始值可以是 domain 宽度(或高度)的 1/6。
# 如果设置为 None, 脚本会自动计算为 domain 宽度的 1/6。
SIGMA_FACTOR_X = 6.0 # sigma_x = nx / SIGMA_FACTOR_X
SIGMA_FACTOR_Y = 6.0 # sigma_y = ny / SIGMA_FACTOR_Y

# --- 定义需要应用掩码的扰动变量列表 ---
VARS_PERT_MASS = ['T','THM', 'P','PH', 'QVAPOR', 'MU', 'PSFC'] # 位于格点中心的变量
VARS_PERT_U = ['U'] # U-staggered 变量
VARS_PERT_V = ['V'] # V-staggered 变量

# --- 脚本开始 ---

print("正在打开文件...")
ds_bg = xr.open_dataset(file_background)
ds_pert = xr.open_dataset(file_perturbation)

# --- 获取所有需要的维度信息 ---
dims = ds_bg.dims
nx, ny = dims['west_east'], dims['south_north']
nx_stag, ny_stag = dims['west_east_stag'], dims['south_north_stag']

print(f"Domain 尺寸 (Mass): nx={nx}, ny={ny}")

# --- 根据用户设置计算 sigma 值 ---
sigma_x = nx / SIGMA_FACTOR_X
sigma_y = ny / SIGMA_FACTOR_Y
sigma_x_stag = nx_stag / SIGMA_FACTOR_X
sigma_y_stag = ny_stag / SIGMA_FACTOR_Y

print(f"高斯掩码参数: sigma_x={sigma_x:.2f}, sigma_y={sigma_y:.2f} (格点)")

def create_gaussian_mask_2d(shape_y, shape_x, sigma_y, sigma_x):
    """为给定维度和sigma创建二维高斯掩码"""
    center_y = (shape_y - 1) / 2.0
    center_x = (shape_x - 1) / 2.0
    
    y = np.arange(shape_y)
    x = np.arange(shape_x)
    
    # 计算每个点到中心的距离的平方 (已归一化 by sigma)
    dist_y2 = ((y - center_y) / sigma_y)**2
    dist_x2 = ((x - center_x) / sigma_x)**2
    
    # 使用numpy的outer方法高效计算二维指数
    exponent = np.add.outer(dist_y2, dist_x2)
    
    mask = np.exp(-0.5 * exponent)
    return mask

# --- 创建三种不同的2D高斯掩码 ---
# 1. 为质量点创建掩码
mask_mass_np = create_gaussian_mask_2d(ny, nx, sigma_y, sigma_x)
mask_mass = xr.DataArray(mask_mass_np, dims=['south_north', 'west_east'])

# 2. 为U点创建掩码
mask_u_np = create_gaussian_mask_2d(ny, nx_stag, sigma_y, sigma_x_stag)
mask_u = xr.DataArray(mask_u_np, dims=['south_north', 'west_east_stag'])

# 3. 为V点创建掩码
mask_v_np = create_gaussian_mask_2d(ny_stag, nx, sigma_y_stag, sigma_x)
mask_v = xr.DataArray(mask_v_np, dims=['south_north_stag', 'west_east'])

print("已为 mass, u, v 三种网格点创建了专用的高斯掩码。")


# --- 应用掩码并生成输出文件 ---
print("开始应用掩码到扰动变量...")
ds_out = ds_bg.copy(deep=True)
all_vars_to_taper = VARS_PERT_MASS + VARS_PERT_U + VARS_PERT_V

for pert_var in all_vars_to_taper:
    if pert_var not in ds_pert:
        continue
    bg_var = pert_var.replace('_PERT', '')
    if bg_var not in ds_out:
        continue

    print(f"正在处理: {pert_var} -> {bg_var}")
    pert_field = ds_pert[pert_var]
    
    if pert_var in VARS_PERT_U:
        selected_mask = mask_u
    elif pert_var in VARS_PERT_V:
        selected_mask = mask_v
    else:
        selected_mask = mask_mass
        
    tapered_pert = pert_field * selected_mask
    ds_out[bg_var] = ds_bg[bg_var] + tapered_pert

print("所有变量处理完毕。正在保存输出文件...")
ds_out.to_netcdf(file_output, mode='a')
ds_out.close()
print(f"成功输出至文件: {file_output}")
