import xarray as xr
import numpy as np
import os

# --- 用户设置 ---
# 输入文件路径 (保留您原有的逻辑)
work_dir=os.getenv('current_work_dir')
time_str=os.getenv('this_time')
mem_str=os.getenv('member')
file_background = f'fg'  # 未扰动的分析场
file_perturbation = f'diff' # 来自random_cv的扰动文件

# 输出文件路径 (保留您原有的逻辑)
file_output = f'wrfinput_2018091200_mem001_tapperd' # 最终的、边界扰动被削弱的初始场 (netCDF4格式)

# --- 水平高斯函数参数设置 ---
SIGMA_FACTOR_X = 5.0 # sigma_x = nx / SIGMA_FACTOR_X
SIGMA_FACTOR_Y = 5.0 # sigma_y = ny / SIGMA_FACTOR_Y

# --- (新功能) 垂直高斯函数 Tapering 设置 ---
# True: 启用垂直tapering, False: 只进行水平tapering
ENABLE_VERTICAL_TAPER = False
# 定义垂直高斯函数的“特征高度”或标准差(sigma_z)。
# 这是一个模式层数。在这个层数，扰动权重会衰减到约60%。
# 值越大，代表扰动能延伸到越高的高度。一个合理的初始值可以是总层数的1/2或1/3。
GAUSSIAN_TAPER_LEVEL_Z = 20.0

# --- 定义需要应用掩码的变量列表 (保留您原有的逻辑) ---
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
# 获取垂直维度
nz = dims.get('bottom_top', 1)

print(f"Domain 尺寸 (Mass): nx={nx}, ny={ny}, nz={nz}")

# --- 根据用户设置计算 sigma 值 ---
sigma_x = nx / SIGMA_FACTOR_X
sigma_y = ny / SIGMA_FACTOR_Y
sigma_x_stag = nx_stag / SIGMA_FACTOR_X
sigma_y_stag = ny_stag / SIGMA_FACTOR_Y

print(f"水平高斯掩码参数: sigma_x={sigma_x:.2f}, sigma_y={sigma_y:.2f} (格点)")

def create_gaussian_mask_2d(shape_y, shape_x, sigma_y, sigma_x):
    """为给定维度和sigma创建二维高斯掩码"""
    center_y = (shape_y - 1) / 2.0
    center_x = (shape_x - 1) / 2.0
    y = np.arange(shape_y)
    x = np.arange(shape_x)
    dist_y2 = ((y - center_y) / sigma_y)**2
    dist_x2 = ((x - center_x) / sigma_x)**2
    exponent = np.add.outer(dist_y2, dist_x2)
    mask = np.exp(-0.5 * exponent)
    return mask

# --- 创建三种不同的2D水平掩码 ---
mask_mass_2d = xr.DataArray(create_gaussian_mask_2d(ny, nx, sigma_y, sigma_x), dims=['south_north', 'west_east'])
mask_u_2d = xr.DataArray(create_gaussian_mask_2d(ny, nx_stag, sigma_y, sigma_x_stag), dims=['south_north', 'west_east_stag'])
mask_v_2d = xr.DataArray(create_gaussian_mask_2d(ny_stag, nx, sigma_y_stag, sigma_x), dims=['south_north_stag', 'west_east'])
print("已创建二维水平高斯掩码。")


# --- (新功能) 创建基于高斯函数的一维垂直掩码 ---
vertical_mask_1d = xr.DataArray(np.ones(nz), dims=['bottom_top']) # 默认权重为1
if ENABLE_VERTICAL_TAPER and nz > 1:
    sigma_z = float(GAUSSIAN_TAPER_LEVEL_Z)
    if sigma_z <= 0:
        raise ValueError("GAUSSIAN_TAPER_LEVEL_Z 必须是一个正数。")
    
    # 创建从0到nz-1的垂直层级索引
    z_levels = np.arange(nz)
    
    # 计算高斯函数权重，中心(峰值)在z=0处
    exponent = (z_levels / sigma_z)**2
    v_weights = np.exp(-0.5 * exponent)
    
    vertical_mask_1d = xr.DataArray(v_weights, dims=['bottom_top'])
    print(f"已创建一维垂直高斯掩码: sigma_z = {sigma_z} 层。")
else:
    print("未启用垂直Tapering。")


# --- 应用掩码并生成输出文件 (保留您原有的逻辑) ---
print("开始应用掩码到扰动变量...")
ds_out = xr.open_dataset(file_output)
all_vars_to_taper = VARS_PERT_MASS + VARS_PERT_U + VARS_PERT_V

for pert_var in all_vars_to_taper:
    if pert_var not in ds_pert:
        continue
    bg_var = pert_var
    if bg_var not in ds_out:
        continue

    print(f"正在处理: {pert_var} -> {bg_var}")
    pert_field = ds_pert[pert_var]
    print(pert_field.max())
    
    # 根据变量维度选择正确的掩码
    if 'bottom_top' in pert_field.dims:
        # 3D变量: 应用水平和垂直组合的掩码
        if pert_var in VARS_PERT_U:
            selected_mask = mask_u_2d * vertical_mask_1d
        elif pert_var in VARS_PERT_V:
            selected_mask = mask_v_2d * vertical_mask_1d
        else: # Mass point
            selected_mask = mask_mass_2d * vertical_mask_1d
    else:
        # 2D变量: 只应用水平掩码
        if pert_var in VARS_PERT_U:
            selected_mask = mask_u_2d
        elif pert_var in VARS_PERT_V:
            selected_mask = mask_v_2d
        else: # Mass point (e.g., PSFC, MU)
            selected_mask = mask_mass_2d
            
    tapered_pert = pert_field * selected_mask
    print(tapered_pert.max())
    ds_out[bg_var] = ds_bg[bg_var] + tapered_pert

print("所有变量处理完毕。正在更新内存中的数据集...")
# 您的原始代码中没有保存文件的步骤，这里同样不添加。
# ds_out.to_netcdf(file_output, mode='w', format='NETCDF4')

print(f"成功在内存中更新了 {file_output} 的数据。")