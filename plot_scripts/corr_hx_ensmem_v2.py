from param import om_z, eta
from matplotlib.scale import FuncScale, register_scale
import os
import sys
import numpy as np
import netCDF4 as nc
import wrf
from kc_functions import nc_read1, match_typhoon_grid
import pandas as pd
import subprocess
import matplotlib.pyplot as plt

# ==========================================
# 自定义坐标轴比例尺 (保持不变)
# ==========================================
def piecewise_scale(y):
    y = np.asarray(y)
    return np.where(y <= 100, y*4, (y + 1300)*2/7)

def piecewise_inverse(y):
    y = np.asarray(y)
    return np.where(y <= 200, y/4, y*3.5 - 1300)

class PiecewiseScale(FuncScale):
    name = 'piecewise'
    def __init__(self, axis):
        FuncScale.__init__(self, axis, functions=(piecewise_scale, piecewise_inverse))

register_scale(PiecewiseScale)

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 设置工作路径
    pwd = "/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts"
    if os.getcwd() != pwd:
        os.chdir(pwd)

    # basic settings
    work_dir = "/share/home/lililei1/kcfu/tc_mangkhut"
    obs_dir = f"{work_dir}/3create_obs/hx_rttov/3obs_BT"
    ensmem_dir = f"/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst"
    ens_hx_dir = f'{work_dir}/3create_obs/hx_rttov/4ens_BT'
    NR_path = '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d03_2018-09-10_00:00:00'
    domain = 'd01'
    day_list = ['10']
    hour_list = ['00']
    minute = '00'
    itime = 0
    wrf_time = day_list[itime] + '_' + hour_list[itime] + ':' + minute  # 10_00:00
    time = day_list[itime] + '_' + hour_list[itime] + '_' + minute      # 10_00_00
    mem_list = list(np.arange(1, 51))
    varname_list = ['T', 'OM_TMP', 'U', 'OM_U', 'QVAPOR']
    channel_list = '1,2,3,4'.split(',')
    nobs = 676
    instrument = 'AMSUA'
    read_cache = False  # 如果重新计算，建议设为False
    plot_all_channel = False

    # configure of wrfout
    nAtmos_level = 56
    nOcean_level = 30
    p_bot = 1000.
    p_top = 50.
    p = eta * (p_bot - p_top) + p_top

    # Grid settings
    i_parent_start = 27
    j_parent_start = 88
    d02_start_dd = '09'
    d02_start_hh = '06'
    
    # Calculate TC center
    lat, lon, iCenter_index, jCenter_index = match_typhoon_grid(NR_path, f'{ensmem_dir}/mem001/wrfout_d01_2018-09-10_00:00:00')

    # 确保缓存目录存在
    subprocess.run(['mkdir', '-p', f'{pwd}/corr_cache'])

    for ich, chnum in enumerate(channel_list):
        print(f'channel {chnum} of {instrument} is calc')
        
        # --------------------------------------------------
        # 读取观测数据 (用于占位或后续校验，计算相关主要用集合Hx)
        obs_file = f'{obs_dir}/{instrument}/BT_{time}/obs_{domain}_ch{chnum}_totalline_withpert.txt'
        obs = np.loadtxt(obs_file) # shape: (nobs,)

        dfs = [] # 用于存储不同变量的相关系数廓线

        # --------------------------------------------------
        # 循环计算每个变量的相关系数
        for ivar, varname in enumerate(varname_list):
            cache_file = f'{pwd}/corr_cache/mean_ch{chnum}_corr_{varname}.csv'

            if read_cache and os.path.exists(cache_file):
                print(f"  Reading cache for {varname}")
                corr_series = pd.read_csv(cache_file, index_col=0).iloc[:, 0]
                dfs.append(corr_series)
            else:
                print(f"  Calculating {varname}...")
                length_of_obs = int(np.sqrt(nobs))
                radius = int(length_of_obs / 2)
                ens_Jslice = slice(jCenter_index - radius + 1, jCenter_index + radius + 1)
                ens_Islice = slice(iCenter_index - radius + 1, iCenter_index + radius + 1)

                wrfout_ens = []
                ens_hx = []

                # 1. 读取所有集合成员的数据
                for imem, member in enumerate(mem_list):
                    member_str = "mem{:03d}".format(member)
                    member_dir = f"{ensmem_dir}/{member_str}"
                    
                    # 读取 Hx (模拟亮温) [Shape: (nobs,)]
                    hx = np.loadtxt(f'{ens_hx_dir}/{member_str}/{instrument}/BT_{time}/obs_{domain}_ch{chnum}_totalline.txt')
                    ens_hx.append(hx)
                    
                    # 读取 State Variable (状态变量) [Shape: (nz, ny, nx)]
                    wrf_name = f'{member_dir}/wrfout_{domain}_2018-09-{wrf_time}:00'
                    wrf_var = wrf.to_np(nc_read1(wrf_name, varname)[:, ens_Jslice, ens_Islice])
                    
                    # 处理位温转气温
                    if varname == 'T':
                        p_var = wrf.to_np(nc_read1(wrf_name, 'P')[:, ens_Jslice, ens_Islice])
                        pb = wrf.to_np(nc_read1(wrf_name, 'PB')[:, ens_Jslice, ens_Islice])
                        p_full = p_var + pb
                        wrf_var = (wrf_var + 300) * ((p_full / 100000.0) ** 0.286)
                    
                    wrfout_ens.append(wrf_var)

                # 转换为 numpy 数组以便进行向量化计算
                # ens_hx_np shape: (N_ens, N_obs) -> (50, 676)
                ens_hx_np = np.array(ens_hx) 
                # wrfout_np shape: (N_ens, N_lev, Ny, Nx)
                wrfout_np = np.array(wrfout_ens)

                # 确定垂直层数
                if varname[:2] == 'OM':
                    zlevel = nOcean_level
                else:
                    zlevel = nAtmos_level
                
                level_list = np.arange(zlevel)
                corr_list = []

                # 2. 逐层计算相关系数 (核心修改部分)
                for nlevel in range(zlevel):
                    # 准备 X: 状态变量
                    # 取出该层数据: (N_ens, Ny, Nx) -> Flatten -> (N_ens, N_obs)
                    # 注意：保持 order='F' 以匹配 Fortran/Matlab 的读取顺序，确保空间点与 obs 一一对应
                    X = wrfout_np[:, nlevel, :, :].reshape(len(mem_list), -1, order='F')
                    
                    # 准备 Y: 观测算子对应的模拟值 (所有层共用同一套BT，因为BT是积分量)
                    Y = ens_hx_np 

                    # --- 向量化相关系数计算 ---
                    # 公式: Corr = Mean((X-X_bar)(Y-Y_bar)) / (Std_X * Std_Y)
                    
                    X_mean = np.mean(X, axis=0) # 沿集合维度的平均 (N_obs,)
                    Y_mean = np.mean(Y, axis=0)
                    
                    X_anom = X - X_mean         # 距平 (N_ens, N_obs)
                    Y_anom = Y - Y_mean

                    covariance = np.mean(X_anom * Y_anom, axis=0) # (N_obs,)
                    std_product = np.std(X, axis=0) * np.std(Y, axis=0)

                    # 避免分母为0的警告
                    with np.errstate(divide='ignore', invalid='ignore'):
                        corr_pointwise = covariance / std_product
                    
                    # 关键步骤：先取绝对值，再空间平均 (Spatially Average of Absolute Correlation)
                    # 这样可以体现该层“是否有相关性”，而不受正负抵消影响
                    mean_abs_corr = np.nanmean(np.abs(corr_pointwise))
                    
                    corr_list.append(mean_abs_corr)

                # 保存结果
                corr_series = pd.Series(corr_list, index=level_list, name='correlation')
                corr_series.to_csv(cache_file)
                dfs.append(corr_series)

        # ===================================================
        # Plotting (修改以适配新的数据格式)
        # ===================================================
        
        # V1: Single channel plot
        if not plot_all_channel:
            fig, axs = plt.subplots(1, len(varname_list), figsize=(4 * len(varname_list), 2 * len(varname_list)))
            kc_blue = "#091508"
            
            for ivar, varname in enumerate(varname_list):
                ax = axs[ivar]
                corr = dfs[ivar] # Series
                
                if varname[:2] != 'OM':
                    # 大气变量绘制
                    # 注意：如果 p 的长度比 corr 多1 (界面 vs 层心)，需要切片
                    plot_p = p[:-1] if len(p) > len(corr) else p
                    # 确保维度匹配
                    min_len = min(len(corr), len(plot_p))
                    ax.plot(corr.values[:min_len], plot_p[:min_len], linestyle='-', color=kc_blue)
                    
                    ax.set_title(varname)
                    ax.set_ylabel('pressure(hPa)')
                    ax.set_xlabel('correlation (abs)') # 标注改为 abs
                    ax.invert_yaxis()
                else:
                    # 海洋变量绘制
                    ax.plot(corr.values, om_z, color=kc_blue)
                    ax.set_title(varname)
                    ax.set_ylabel('depth(m)')
                    ax.set_xlabel('correlation (abs)')
                    ax.set_yscale('piecewise')
                    ax.set_yticks([5, 25, 50, 100, 200, 400, 750])
                    ax.set_yticklabels(['5', '25', '50', '100', '200', '400', '750'])
                    ax.invert_yaxis()
                
                # 由于取了绝对值，相关系数都是正的，x轴从0开始
                ax.set_xlim(left=0) 
                ax.grid(True, linestyle='--', alpha=0.6)
                
            plt.savefig(f"{pwd}/figs/corr_profile/corr_ch{chnum}.png", format='png', dpi=300, bbox_inches='tight')
            plt.close()

    # V2: All channels plot (Optional)
    if plot_all_channel:
        fig, axs2 = plt.subplots(1, len(varname_list), figsize=(4 * len(varname_list), 2 * len(varname_list)))
        kc_blue = "#0072BD"
        kc_red = "#D0002E"
        kc_green = "#009E73"
        eva2_black = "#091508"
        my_palette = [kc_blue, kc_green, kc_red, eva2_black]

        for ich, chnum in enumerate(channel_list):
            # 重新读取缓存 (因为上面的循环可能已经计算过了)
            dfs_plot = []
            for ivar, varname in enumerate(varname_list):
                cache_file = f'{pwd}/corr_cache/mean_ch{chnum}_corr_{varname}.csv'
                if os.path.exists(cache_file):
                    dfs_plot.append(pd.read_csv(cache_file, index_col=0).iloc[:, 0])
                else:
                    print(f"Warning: Cache missing for {varname} ch{chnum}")
                    continue
            
            if not dfs_plot: continue

            for ivar, varname in enumerate(varname_list):
                ax = axs2[ivar]
                corr = dfs_plot[ivar]
                
                if varname[:2] != 'OM':
                    plot_p = p[:-1] if len(p) > len(corr) else p
                    min_len = min(len(corr), len(plot_p))
                    ax.plot(corr.values[:min_len], plot_p[:min_len], linestyle='-', color=my_palette[ich % 4], label=f'ch{chnum}')
                    ax.set_title(f'BT and {varname}')
                    ax.set_ylabel('pressure(hPa)')
                    ax.set_xlabel('correlation (abs)')
                else:
                    ax.plot(corr.values, om_z, color=my_palette[ich % 4], label=f'ch{chnum}')
                    ax.set_title(f'BT and {varname}')
                    ax.set_ylabel('depth(m)')
                    ax.set_xlabel('correlation (abs)')
                    ax.set_yscale('piecewise')
                    ax.set_yticks([5, 25, 50, 100, 200, 400, 750])
                    ax.set_yticklabels(['5', '25', '50', '100', '200', '400', '750'])
                
                if ich == len(channel_list) - 1:
                    ax.invert_yaxis()
                    ax.set_xlim(left=0)
                    ax.grid(True, linestyle='--', alpha=0.6)

        plt.legend(loc='upper right', bbox_to_anchor=(0.8, -0.10), ncol=len(channel_list), markerscale=2, fontsize=15)
        plt.savefig(f"{pwd}/figs/corr_profile/corr_all_ch.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()