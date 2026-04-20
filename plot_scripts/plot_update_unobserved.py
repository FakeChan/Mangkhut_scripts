import numpy as np 
import math
import subprocess
import xarray as xr
from param import *
from kc_functions import getTClocation
import re
from wrf import getvar
import netCDF4
import cmocean
from matplotlib import pyplot as plt
def extract_xxhpa_data(wrf_file, iTC, jTC,radius,press):
    """
    提取以给定经纬度为中心、周围30个网格范围内的850hPa层数据。
    
    参数：
        wrf_file (str): WRF输出文件路径
        center_lat (float): 中心点纬度
        center_lon (float): 中心点经度
        buffer (int): 周围网格范围(默认30)
        
    返回：
        dict: 包含 u, v, t, q 的切片数据(xarray.DataArray)
    """
    # 打开WRF文件
    ds = netCDF4.Dataset(wrf_file)
    
    j_start =  jTC-radius
    j_end   =  jTC+radius
    i_start =  iTC-radius
    i_end   =  iTC+radius
    # 获取850hPa气压层数据
    results = {}
    
    # 获取三维气压场（质量点）
    p = getvar(ds, 'pressure')
    traget_press=plev_dict[f'{str(press)}hpa']
    
    # 处理U风场（错格X方向）
    ua = getvar(ds, 'ua')
    # u_850 = interplevel(ua, p, press)
    u_850=ua[traget_press,:,:]
    results['u']=u_850[j_start:j_end,i_start:i_end]
    
    
    # 处理V风场（错格Y方向）
    va = getvar(ds, 'va')
    # v_850 = interplevel(va, p, press)
    v_850=va[traget_press,:,:]
    results['v']=v_850[j_start:j_end,i_start:i_end]
    
    # 处理温度T（质量点）
    # t = getvar(ds, 'tk')
    t=ds.variables['THM'][0,:,:,:]
    # t_850 = interplevel(t, p, press)
    t_850=t[traget_press,:,:]
    results['T'] = t_850[j_start:j_end,i_start:i_end]
    
    # 处理比湿Q（质量点）
    q = getvar(ds, 'QVAPOR')
    # q_850 = interplevel(q, p, press)
    q_850=q[traget_press,:,:]
    results['Qv'] = q_850[j_start:j_end,i_start:i_end]
    
    return results
def read_variable_from_nc(filename, var_name="T"):
    """
    使用 xarray 从 NetCDF 文件中读取指定的变量。

    参数:
        filename (str): .nc 文件的路径。
        var_name (str): 要读取的变量的名称。
    """
    
    try:
        # 使用 with 语句打开数据集，可以确保文件在使用后被自动关闭
        with xr.open_dataset(filename) as ds:
            
            # 检查变量是否存在于数据集中
            if var_name in ds:
                # --- 核心步骤: 读取变量 ---
                # 通过类似字典键的方式访问变量
                variable = ds[var_name]
                
                print(f"\nread variable'{var_name}'")
                
            else:
                print(f"\nerror. variable'{var_name}' not found")

    except FileNotFoundError:
        print(f"error. file '{filename}'not found")
    except Exception as e:
        print(f"unknown error: {e}")
    return variable

def extract_obs_prior_from_file(filepath,pattern):
    """
    从指定的日志文件中读取内容，并提取所有 'obs_prior' 的数值。

    参数:
    filepath (str): 日志文件的路径。

    返回:
    list[float]: 包含所有提取到的 'obs_prior' 数值的浮点数列表。
                 如果文件未找到或没有匹配项，则返回空列表。
    """
    # 定义用于匹配 'obs_prior' 数据的正则表达式
    # \s+      匹配一个或多个空白字符
    # ([-]?\d+\.\d+) 捕获一个可能带负号的浮点数
    
    
    extracted_data = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 遍历文件中的每一行
            for line in f:
                # 在当前行中搜索匹配的模式
                match = pattern.search(line)
                if match:
                    # 如果找到匹配项，提取第一个捕获组（即数值）
                    # 并将其转换为浮点数后添加到列表中
                    value_str = match.group(1)
                    extracted_data.append(float(value_str))
    except FileNotFoundError:
        print(f"错误: 文件未找到 '{filepath}'")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

    return extracted_data    
if __name__ == "__main__":
    pwd='/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts'
    cache_dir=f'{pwd}/postAnal_cache'
    subprocess.run(['mkdir','-p',f'{cache_dir}'])
    #==========================
    #parameters
    radius=30
    cutoff_list=[0.1,0.05,0.01]
    vert_norm_list=[40000.0,80000.0,400000.0,800000.0]
    var_list=['u','v','T','Qv']
    var='U'
    traget_press='850hpa'
    plevel=plev_dict[traget_press]
    ocean_var=True
    if ocean_var:
        plevel=0
    onlyTCarea=True
    read_cache=True
    plot_increment=True
    save_fig=read_cache
    single_obs=True
    plot_update=plot_increment
    #==================================================================
    #read data
    if single_obs:
        suffix=''
        suffix_out='singleobs'
    else:
        suffix='allobs'
        suffix_out=suffix
    anal_name_list=[]
    for cutoff in cutoff_list:
        for vert_norm in vert_norm_list:
            horizLoc=str(round(2*cutoff*6400))
            vertLoc=str(math.floor(2*cutoff*vert_norm/1000))
            anal_name=f'horizLoc{horizLoc}.vertLoc{vertLoc}'
            anal_name_list.append(anal_name)
    file_list=['/scratch/lililei1/kcfu/tc_mangkhut/NR/d01/wrfout_d01_2018-09-10_00:00:00',
            '/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/10_00_00/firstguess.ensmean']
    
    anal_dir=f'/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/postAssimData/10_00_00/{suffix_out}'
    fg_dir=f'/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/0mem_all_time/10_00_00'
#----------------------------------------------------------
    #fisrt get the location of TC from NR
    NR=file_list[0]
    iTC, jTC = getTClocation(NR)
    j_start =  jTC-radius
    j_end   =  jTC+radius
    i_start =  iTC-radius
    i_end   =  iTC+radius
    
    #cycle in different localization parameters
    if not read_cache:
        for ianal,anal in enumerate(anal_name_list):
            file_mean=f'firstguess.ensmean'
            nc_mean=f'{fg_dir}/{file_mean}'
            print(nc_mean)
            var_press_mean=xr.open_dataset(nc_mean,engine='netcdf4')[var].values[0,plevel,j_start:j_end,i_start:i_end]
            
            #empty cov matrix
            cov_xy=np.zeros_like(var_press_mean)
            for imem in range(1,51):
                mem=f'{imem:03d}'
                file_mem=f'firstguess.mem{mem}'
                nc_mem=f'{fg_dir}/{file_mem}'
                # print(nc_mem)
                var_press_mem=xr.open_dataset(nc_mem,engine='netcdf4')[var].values[0,plevel,j_start:j_end,i_start:i_end]
                diff_mem=var_press_mem-var_press_mean
            #plot update pattern of vars
                if plot_update:
                    # print('plot regression of unobserved update')
                    log=f'{anal_dir}/dart_log.out.{anal}'
                    pattern = re.compile(r'obs_prior:\s+([-]?\d+\.\d+)')
                    obs_prior=extract_obs_prior_from_file(log,pattern=pattern)
                    pattern= re.compile(r'obs_increment:\s+([-]?\d+\.\d+)')
                    obs_inc=extract_obs_prior_from_file(log,pattern=pattern)
                    # print(obs_prior)
                    obs_prior_mem=obs_prior[imem-1]
                    obs_inc_mem=obs_inc[imem-1]
                cov_xy+=diff_mem*(obs_prior_mem-np.mean(obs_prior))
            cov_xy=cov_xy/49
            cov_yy=np.var(obs_prior,ddof=1) #Bessel's correction
            delta_y=np.mean(obs_inc)
            delta_x=cov_xy/cov_yy*delta_y
            np.savetxt(f'{anal_dir}/delta_x_{anal}',delta_x)
    #================================================================================================
    #only plot when read cache
    elif read_cache:
        cmap=cmocean.cm.thermal
        nanal=len(anal_name_list)
        fig,axes=plt.subplots(ncols=len(vert_norm_list),nrows=math.ceil(nanal/len(vert_norm_list)),figsize=(30, 30))
        for ianal,anal in enumerate(anal_name_list):
            data=np.loadtxt(f'{anal_dir}/delta_x_{anal}')
            ax=axes.flat[ianal]
            ax.set_title(anal,fontsize=18)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            contour=ax.contourf(data, cmap=cmap, levels=10)
        fig.savefig('/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/figs/update_x.png',dpi=300)