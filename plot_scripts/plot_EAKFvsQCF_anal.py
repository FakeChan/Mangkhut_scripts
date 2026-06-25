from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4
from scipy.interpolate import griddata
from kc_functions import getTClocation,calculate_rmse,getNC_realT
from param import *
def getTCnest(NR_path,domain,var,plev,ocean_lev=0):
    # create a nest from 300m NR
    
    #-------------------------------
    # check the resolution of NR 
    try:
        ncdata = xr.open_dataset(NR_path)
        
        # 修改 1: attrs 是字典，使用 ['DX'] 访问
        dx = ncdata.attrs['DX']
        
        # 修改 2: 使用 np.isclose 处理浮点数精度 (例如 299.9999 vs 300)
        # 如果 dx 不接近 300 (允许 0.1m 的误差)，则报错
        if not np.isclose(dx, 7500, atol=0.1):
            # 修改 3: 使用 raise 抛出异常来终止程序
            raise ValueError(f"Error! Resolution of NR is not 300m. Current DX: {dx}")
            
    except KeyError:
        # 防止文件中根本没有 DX 属性的情况
        raise ValueError("Error! 'DX' attribute not found in the file.")
    #-------------------------------
    
    #check pass, get center index of TC
    print('resolution check pass : 300m')
    [iTC,jTC]=getTClocation(NR_path)
    
    #-------------------------------
    #set size of grid for comparison, radius=120km
    
    
    if domain=='d01':
        grid_step=1
        half_grid_size=240/7.5
    elif domain == 'd02':
        grid_step=5
        half_grid_size=210/1.5
    else:
        raise ValueError(f"Error! domain should be d01\d02, instead of{domain}")
    
    offsets = (np.arange(-half_grid_size, half_grid_size + 1) * grid_step).astype(int)
    
    target_i=iTC+offsets
    target_j=jTC+offsets
    
    grid_x, grid_y = np.meshgrid(target_i, target_j)
    lats = ncdata['XLAT'][0].values
    lons = ncdata['XLONG'][0].values
    extract_lats = lats[grid_y, grid_x]
    extract_lons = lons[grid_y, grid_x]
    
    var_data=ncdata[var][0]
    if 'bottom_top' in var_data.dims or 'bottom_top_stag' in var_data.dims:
        if var =='THM':
            data_values=getNC_realT(ncdata,plev_dict[plev])
        else:
            data_values = var_data[plev_dict[plev], :, :].values
        NR_grid=data_values[grid_y, grid_x]
    elif 'ocean_layer_stag' in var_data.dims:
        #ocean vars
        data_values = var_data[ocean_lev,:,:].values
        NR_grid=data_values[grid_y, grid_x]
    else:
        # 2D 变量 (如 T2, PSFC)
        data_values = var_data.values
        NR_grid=data_values[grid_y, grid_x]
        
        
    return extract_lats,extract_lons,NR_grid

def interp_grid(grid_lats,grid_lons,input_ncdata,input_lats,input_lons,method='linear'):
    # interp input_ncdata to grid lats
    # note: the input should be 2d-data instead of a nc file 
    
    #---------------------------------
    # read vars and lat\lon from input file
    points = np.column_stack((input_lons.flatten(), input_lats.flatten()))
    values = input_ncdata.flatten()
    interp_valus=griddata(points, values, (grid_lons, grid_lats), method=method)
    return interp_valus

if __name__ =='__main__':
    var='OM_TMP'
    domain='d01'
    plev='850hpa'
    ocean_lev=0
    #-----------------------------
    # anal_file_list=['/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/10_00_00/firstguess_d01.ensmean',
    #                 '/scratch/lililei1/kcfu/tc_mangkhut/5cyclingDA/postAnal_EAKF/d01_10_00_00/analysis_d01.ensmean',
    #                 '/scratch/lililei1/kcfu/tc_mangkhut/5cyclingDA/postAnal_QCF_RHF/d01_10_00_00/analysis_d01.ensmean']
    
    anal_file_list=['/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/cyclingDA/10_00_00/firstguess_d01.ensmean',
                    '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/run_dir/EAKF/postassim_mean_d01.nc',
                    '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/run_dir/QCF_RHF/postassim_mean_d01.nc']
    
    title_list=['firstguess','EAKF','QCF_RHF']
    NR_path='/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain/wrfout_d01_2018-09-10_00:00:00'
    #-----------------------------
    #first extract NR data
    extract_lats,extract_lons,NR_values=getTCnest(NR_path,domain,var=var,plev=plev,ocean_lev=ocean_lev)
    print('----------------NR extracted---------------------')
    
    # List to hold all data arrays for min/max calculation
    all_data_for_scale = [NR_values]
    interp_values_list = []
    
    # --- Data Processing Loop for finding global min/max ---
    for ifile,file in enumerate(anal_file_list):
        print(f'now proceeding {file[-26:-1]}')
        nc_ds=xr.open_dataset(file,engine='netcdf4')
        
        if var=='U':
            input_lats=nc_ds['XLAT_U'].values
            input_lons=nc_ds['XLONG_U'].values
        elif var =='V':
            input_lats=nc_ds['XLAT_V'].values
            input_lons=nc_ds['XLONG_V'].values
        else:
            input_lats=nc_ds['XLAT'].values
            input_lons=nc_ds['XLONG'].values
            
        var_data=nc_ds[var][0]
        # 注意: 这里假设 plev_dict 在其他地方 (如 param.py 或 kc_functions.py) 已定义
        if 'bottom_top' in var_data.dims or 'bottom_top_stag' in var_data.dims:
            if var =='THM':
                data_values=getNC_realT(nc_ds,plev_dict[plev])
            else:
                data_values = var_data[plev_dict[plev], :, :].values
            string=plev
        elif 'ocean_layer_stag' in var_data.dims:
            #ocean vars
            data_values = var_data[ocean_lev,:,:].values
            string=f'ocean{ocean_lev}'
        else:
            # 2D 变量 (如 T2, PSFC)
            data_values = var_data.values
            string=''
            
        #interpolate
        print(f'shape of lon:{input_lons.size}')
        print(f'shape of {var}: {data_values.size}')
        interp_values=interp_grid(
            grid_lats=extract_lats,
            grid_lons=extract_lons,
            input_ncdata=data_values,
            input_lats=input_lats,
            input_lons=input_lons,
            method='linear'
        )
        interp_values_list.append(interp_values)
        all_data_for_scale.append(interp_values) # Collect all data for scale calculation
    
    # 计算全局最小值和最大值，用于统一色条
    # Flatten所有数组并合并，然后计算非NaN的min/max
    all_values_flat = np.concatenate([a.flatten() for a in all_data_for_scale])
    vmin = np.nanmin(all_values_flat)
    vmax = np.nanmax(all_values_flat)
    print(f'Calculated unified scale: vmin={vmin}, vmax={vmax}')
    
# --- Plotting with unified colorbar and jet colormap ---
    # ... (前面的数据读取和插值代码保持不变) ...

    # ==============================================================================
    # 1. 准备第一行（绝对量）的 Levels
    # ==============================================================================
    # Flatten所有绝对量数组并计算 min/max
    all_abs_values = np.concatenate([a.flatten() for a in all_data_for_scale])
    vmin_abs = np.nanmin(all_abs_values)
    vmax_abs = np.nanmax(all_abs_values)
    # 第一行 Levels (绝对值)
    levels_abs = np.linspace(vmin_abs, vmax_abs, 60)
    print(f'Row 1 Scale (Absolute): vmin={vmin_abs:.5f}, vmax={vmax_abs:.5f}')

    # ==============================================================================
    # 2. 准备第二行（增量/Diff）的 Levels
    # ==============================================================================
    # 先计算所有差值，找出最大的绝对误差，做成对称色标
    all_diff_values = []
    diff_data_list = [] # 存储计算好的差值，避免绘图时重复计算
    RMSE_values=[]
    for ival,val in enumerate(interp_values_list):
        if ival ==0:
            diff=NR_values-val
            fg=val
            diff_data_list.append(diff)
            all_diff_values.append(diff.flatten())
            print("NR_values 包含 NaN 吗?", np.isnan(NR_values).any())
            print("val 包含 NaN 吗?", np.isnan(val).any())
            print("数组为空吗?", len(val.flatten()) == 0)
            rmse=calculate_rmse(NR_values.flatten(),val.flatten())
            RMSE_values.append(rmse)
            continue
        diff = val - fg
        diff_data_list.append(diff)
        all_diff_values.append(diff.flatten())
        print("NR_values 包含 NaN 吗?", np.isnan(NR_values).any())
        print("val 包含 NaN 吗?", np.isnan(val).any())
        print("数组为空吗?", len(val.flatten()) == 0)
        rmse=calculate_rmse(NR_values.flatten(),val.flatten())
        RMSE_values.append(rmse)
    
    all_diff_concat = np.concatenate(all_diff_values)
    # 找出差值中绝对值最大的数，例如 max_diff 为 0.005
    max_diff = np.nanmax(np.abs(all_diff_concat))
    # 设置对称范围 [-0.005, 0.005]
    levels_diff = np.linspace(-max_diff, max_diff, 60)
    print(f'Row 2 Scale (Diff): range= +/- {max_diff:.5f}')

    # ==============================================================================
    # 3. 开始绘图
    # ==============================================================================
    # 注意：这里 ax 形状是 (2, 4) -> [2行, (1个NR + 3个Anal)列]
    fig, ax = plt.subplots(2, len(anal_file_list) + 1, figsize=(40, 20)) # 稍微调小一点figsize，太大会导致字体显得极小

    # --- 绘制第一行：绝对量 (NR + Analysis) ---
    
    # 1.1 画 NR (Reference)
    ax_NR = ax[0, 0]
    cf_abs = ax_NR.pcolor(NR_values, cmap='jet',vmin=vmin_abs,vmax=vmax_abs)
    ax_NR.set_title('NR (Truth)', fontsize=24)
    
    # 1.2 画 Analysis
    for ifile, interp_values in enumerate(interp_values_list):
        axs = ax[0, ifile + 1]
        axs.pcolor(interp_values, cmap='jet',vmin=vmin_abs,vmax=vmax_abs)
        
        # 标题处理
        title_parts = anal_file_list[ifile].split('/')
        # 取出 EAKF/preassim 这种区分度高的字段
        title_str = f"{title_parts[-2]}\n{title_list[ifile]}" 
        # title_str=title_list[ifile]
        axs.set_title(title_str, fontsize=24)

    # --- 绘制第二行：增量 (Analysis - NR) ---
    
    # 2.1 NR 对应的 Diff 位置通常留白或画全0，这里我们将其留白
    ax[1, 0].axis('off') # 关闭 NR 下方的子图
    
    # 2.2 画 Diffs
    cf_diff = None
    for ifile, diff_val in enumerate(diff_data_list):
        axs = ax[1, ifile + 1]
        # 使用 seismic 或 bwr (Blue-White-Red) colormap
        cf_diff = axs.pcolor(diff_val, cmap='seismic',vmin=-max_diff,vmax=max_diff)
        axs.set_title(f'RMSE: {RMSE_values[ifile]}', fontsize=24)

    # ==============================================================================
    # 4. 添加 Colorbars (调整大小和位置)
    # ==============================================================================
    
    # 调整子图间距，给右侧留出放 colorbar 的空间
    plt.subplots_adjust(right=0.9, wspace=0.2, hspace=0.3)
    
    # 第一行的 Colorbar (绝对值)
    # 位置参数: [left, bottom, width, height] -> 放在整个画布右侧对应第一行的位置
    cbar_ax1 = fig.add_axes([0.91, 0.55, 0.015, 0.3]) 
    cb1 = fig.colorbar(cf_abs, cax=cbar_ax1)
    cb1.ax.tick_params(labelsize=16)
    cb1.set_label(f'{var} (Absolute)', fontsize=24)

    # 第二行的 Colorbar (差值)
    # 位置参数: [left, bottom, width, height] -> 放在整个画布右侧对应第二行的位置
    cbar_ax2 = fig.add_axes([0.91, 0.12, 0.015, 0.3])
    cb2 = fig.colorbar(cf_diff, cax=cbar_ax2)
    cb2.ax.tick_params(labelsize=16)
    cb2.set_label(f'{var} Increment (Anal - NR)', fontsize=24)

    # 删除无数据的坐标轴 (主要是 ax[1,0] 如果没关掉的话)
    for axs in ax.flat:
        if not axs.has_data() and axs != ax[1,0]: # ax[1,0] 我们手动关了
             fig.delaxes(axs)

    # 保存图片
    save_path = f'/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/figs/EAKFvsQCF_{var}_{string}_{domain}_v2.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight') # bbox_inches='tight' 防止裁掉colorbar
    print(f'Figure saved to {save_path}')
        