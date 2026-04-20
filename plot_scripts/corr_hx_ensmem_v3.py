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
    read_cache = False  # 如果想重新计算，请设为 False
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
        # 读取观测数据 (用于占位或后续校验)
        obs_file = f'{obs_dir}/{instrument}/BT_{time}/obs_{domain}_ch{chnum}_totalline_withpert.txt'
        obs = np.loadtxt(obs_file)

        dfs = [] # 用于存储不同变量的相关系数廓线

        # --------------------------------------------------
        # 循环计算每个变量的相关系数
        for ivar, varname in enumerate(varname_list):
            cache_file = f'{pwd}/corr_cache/mean_ch{chnum}_corr_{varname}.csv'

            # 只有当 read_cache 为 True 且文件存在时才读取缓存
            if read_cache and os.path.exists(cache_file):
                print(f"  Reading cache for {varname}")
                # 读取时保留 index (level)，第一列是数据
                corr_series = pd.read_csv(cache_file, index_col=0).iloc[:, 0]
                dfs.append(corr_series)
            else:
                print(f"  Calculating {varname} (All members, Full Grid)...")
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

                # 确定垂直层数
                if varname[:2] == 'OM':
                    zlevel = nOcean_level
                else:
                    zlevel = nAtmos_level
                
                level_list = np.arange(zlevel)
                corr_list = []

                # 2. 逐层计算相关系数 (使用全部数据)
                for nlevel in range(zlevel):
                    X_total = []
                    Y_total = []

                    # 收集该层所有成员、所有格点的数据
                    for im in range(len(mem_list)):
                        # 取出第 im 个成员，第 nlevel 层的数据
                        # wrfout_ens[im] shape: (nz, ny, nx)
                        # data_slice shape: (ny, nx)
                        data_slice = wrfout_ens[im][nlevel]
                        
                        # 拉直 (Flatten)。务必使用 order='F' 以匹配 Fortran 顺序 (如果原 obs 生成也是按列优先)
                        # 这样 (ny, nx) 会变成 (ny*nx,) 即 (676,)
                        X_flat = data_slice.flatten(order='F')
                        X_total.append(X_flat)
                        
                        # Hx 是一维的 (676,)，直接添加
                        Y_total.append(ens_hx[im])
                    
                    # 拼接所有成员的数据
                    # X_all shape: (50 * 676,) = (33800,)
                    X_all = np.concatenate(X_total)
                    Y_all = np.concatenate(Y_total)

                    # 计算相关系数
                    # np.corrcoef 返回矩阵 [[1, r], [r, 1]]，取 [0, 1]
                    # 这里**不取绝对值**，直接计算原始相关系数
                    r = np.corrcoef(X_all, Y_all)[0, 1]
                    
                    corr_list.append(r)

                # 保存结果
                corr_series = pd.Series(corr_list, index=level_list, name='correlation')
                corr_series.to_csv(cache_file)
                dfs.append(corr_series)

        # ===================================================
        # Plotting
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
                    plot_p = p[:-1] if len(p) > len(corr) else p
                    min_len = min(len(corr), len(plot_p))
                    ax.plot(corr.values[:min_len], plot_p[:min_len], linestyle='-', color=kc_blue)
                    
                    ax.set_title(varname)
                    ax.set_ylabel('pressure(hPa)')
                    ax.set_xlabel('correlation') # 移除 (abs) 标签
                    ax.invert_yaxis()
                else:
                    # 海洋变量绘制
                    ax.plot(corr.values, om_z, color=kc_blue)
                    ax.set_title(varname)
                    ax.set_ylabel('depth(m)')
                    ax.set_xlabel('correlation')
                    ax.set_yscale('piecewise')
                    ax.set_yticks([5, 25, 50, 100, 200, 400, 750])
                    ax.set_yticklabels(['5', '25', '50', '100', '200', '400', '750'])
                    ax.invert_yaxis()
                
                # 添加 0 线参考
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
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
            dfs_plot = []
            for ivar, varname in enumerate(varname_list):
                cache_file = f'{pwd}/corr_cache/mean_ch{chnum}_corr_{varname}.csv'
                if os.path.exists(cache_file):
                    dfs_plot.append(pd.read_csv(cache_file, index_col=0).iloc[:, 0])
                else:
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
                    ax.set_xlabel('correlation')
                else:
                    ax.plot(corr.values, om_z, color=my_palette[ich % 4], label=f'ch{chnum}')
                    ax.set_title(f'BT and {varname}')
                    ax.set_ylabel('depth(m)')
                    ax.set_xlabel('correlation')
                    ax.set_yscale('piecewise')
                    ax.set_yticks([5, 25, 50, 100, 200, 400, 750])
                    ax.set_yticklabels(['5', '25', '50', '100', '200', '400', '750'])
                
                if ich == len(channel_list) - 1:
                    ax.invert_yaxis()
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
                    ax.grid(True, linestyle='--', alpha=0.6)

        plt.legend(loc='upper right', bbox_to_anchor=(0.8, -0.10), ncol=len(channel_list), markerscale=2, fontsize=15)
        plt.savefig(f"{pwd}/figs/corr_profile/corr_all_ch.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()