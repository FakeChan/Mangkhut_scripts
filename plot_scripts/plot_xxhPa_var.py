import xarray as xr
import netCDF4
from wrf import getvar, ll_to_xy, destagger, interplevel,to_np
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import subprocess
import cmocean
import math
from param import *
import re
#==========================================================
#functions
from scipy.interpolate import interp1d
def add_cbar(ax,pcm):
    # from new bing, set a new ax for cbar,
    # to prevent cbar taking up space for contourf

    pad=1.1; size=0.05
    pos=ax.get_position() #[x0,y0,width,height]
    pos_cbar=[pos.x0+pos.width*pad,pos.y0,pos.width*size,pos.height]
    cax=fig.add_axes(pos_cbar)

    cbar=fig.colorbar(pcm,
                      cax=cax,)
    cbar.locator=ticker.MaxNLocator(nbins=7)
    cbar.formatter.set_powerlimits((0, 0)) # about using scientific notation: https://stackoverflow.com/questions/25983218/scientific-notation-colorbar-in-matplotlib
    cbar.update_ticks()
    return
def find_d26(temp_profile, depths):
    """计算单个点的26°C等温线深度"""
    # 检查数据有效性
    if np.isnan(temp_profile).any():
        return np.nan
    
    # 寻找温度等于26°C的精确匹配
    exact_match = np.where(temp_profile == 26)[0]
    if len(exact_match) > 0:
        return depths[exact_match[0]]  # 返回最浅的精确匹配
    
    # 检测温度跨越26°C的区间
    sign = np.sign(temp_profile - 26)
    sign_changes = np.diff(sign)
    cross_indices = np.where(sign_changes != 0)[0]
    
    if len(cross_indices) == 0:
        return np.nan  # 无跨越
    
    # 遍历所有跨越点，计算插值深度
    d26_candidates = []
    for idx in cross_indices:
        T1, T2 = temp_profile[idx], temp_profile[idx+1]
        z1, z2 = depths[idx], depths[idx+1]
        if T1 == T2:
            d26 = (z1 + z2) / 2
        else:
            d26 = z1 + (26 - T1) * (z2 - z1) / (T2 - T1)
        d26_candidates.append(d26)
    
    return min(d26_candidates)  # 返回最浅的等温线深度

def getTClocation(file):
    """
    Get the location of the TC center from the slp field
    :param file: path to the netCDF file
    :return: tuple of (iTC, jTC) indices of the TC center
    """
    # Use netCDF4 to read the file
    data = netCDF4.Dataset(file)
    
    # Get the sea level pressure (slp) variable
    slp = to_np(getvar(data, 'slp', timeidx=0))
    
    # Find the minimum slp value and its indices
    slpMin = np.min(slp[:, :])
    indexMin = np.argwhere(slp[:, :] == slpMin)
    
    # Extract the i and j indices of the TC center
    jTC = indexMin[0][0]
    iTC = indexMin[0][1]
    
    return iTC, jTC

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

def extract_obs_prior_data(file_path,pattern):
    """
    从指定的日志文件中读取内容，提取所有 'fkc msg: obs_prior' 后面的数值。

    这个函数使用正则表达式来精确匹配和提取数据，能够处理数字前后有不同数量空格的情况。

    参数:
        file_path (str): 日志文件的完整路径。

    返回:
        list: 一个包含所有提取到的整数值的列表。如果文件未找到或发生其他错误，
              将返回一个空列表。
    """
    # 这个正则表达式会查找以 'fkc msg: obs_prior' 开头，
    # 后面跟着一个冒号（前后可能有空格），然后捕获一个或多个数字。
    # \s* 表示零个或多个空白字符
    # (\d+) 是一个捕获组，用于匹配一个或多个数字
    
    
    extracted_numbers = []
    
    try:
        # 使用 'with open' 语句来安全地打开和关闭文件
        with open(file_path, 'r', encoding='utf-8') as f:
            # 逐行读取文件
            for line in f:
                # 对每一行应用正则表达式进行匹配
                match = pattern.match(line.strip())
                if match:
                    # 如果匹配成功，提取第一个捕获组的内容（即数字）
                    # 并将其转换为整数后添加到列表中
                    number_str = match.group(1)
                    extracted_numbers.append(int(number_str))
                        
    except FileNotFoundError:
        print(f"错误：无法找到文件 '{file_path}'。请检查文件名和路径是否正确。")
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
        
    return extracted_numbers

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
    traget_press=850
    onlyTCarea=True
    read_cache=True
    plot_increment=True
    save_fig=read_cache
    single_obs=False
    plot_update=plot_increment
    
    cmap = cmocean.cm.thermal  # 温度专用
    
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
    
    anal_dir=f'/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/postAssimData/10_00_00/{suffix}'
    title_list=['NR','prior']
    nfile=len(file_list)
    nanal=len(anal_name_list)
    for ivar,var in enumerate(var_list):
        fig,axes=plt.subplots(ncols=len(vert_norm_list),nrows=math.ceil(nanal/len(vert_norm_list))+1,figsize=(30, 30))
        if read_cache:
            for i in range(nfile):
                print(f'processing {title_list[i]}')
                var_press=np.loadtxt(f'{cache_dir}/{var}_{title_list[i]}_{traget_press}.txt')
                if title_list[i]=='NR':
                    vmin=var_press.min()
                    vmax=var_press.max()
                    var_truth=var_press
                ax=axes.flat[i]
                ax.set_title(title_list[i],fontsize=18)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                
                contour=ax.contourf(var_press, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
                if i==1:
                    add_cbar(axes[0,len(vert_norm_list)-3],contour)
                    var_fg=var_press
            #=====================================
            #plot NR-FG
            #only plot when plot_increment is true
            if plot_increment:
                ax=axes.flat[3]
                print(var_truth[20,20])
                print(var_fg[20,20])
                ini_error=var_truth-var_fg
                print(ini_error[20,20])
                vmin=ini_error.min()
                vmax=ini_error.max()
                contour=ax.contourf(ini_error, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
                add_cbar(ax,contour)
            #=====================================
            for ianal,anal in enumerate(anal_name_list):
                print(f'processing {anal}')
                var_press=np.loadtxt(f'{cache_dir}/{var}_{anal}_{traget_press}.txt')
                if plot_increment:
                    var_press_save=var_press
                    var_press=var_press-var_fg
                if ianal==0:
                    vmin=var_press.min()
                    vmax=var_press.max()
                ax=axes.flat[ianal+len(vert_norm_list)]
                ax.set_title(anal,fontsize=18)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                contour=ax.contourf(var_press, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
                if (ianal+1)%len(vert_norm_list)==0:
                    add_cbar(ax,contour)  
                
        #================================================================================================
        else:
            #first plot NR and fg
            for i in range(nfile):
                print(f'processing {title_list[i]}')
                file=file_list[i]
                ax=axes.flat[i]
                data=xr.open_dataset(file,engine='netcdf4')
                
                # Get the location of the TC center
                if i==0: # i=0 is NR
                    iTC_NR, jTC_NR = getTClocation(file)
                iTC,jTC=iTC_NR,jTC_NR
                print(f'calculating {var} at {traget_press} of {title_list[i]}')
                
                var_press=extract_xxhpa_data(file,iTC=iTC,jTC=jTC,press=traget_press,radius=radius)[var]
                if title_list[i]=='NR':
                    vmin=var_press.min()
                    vmax=var_press.max()
                    var_truth=var_press
                if i==1: #i=1 is prior
                    var_fg=var_press
                np.savetxt(f'{cache_dir}/{var}_{title_list[i]}_{traget_press}.txt',var_press)
                ax.set_title(title_list[i],fontsize=12)
                # ax.set_xlabel('Longitude')
                # ax.set_ylabel('Latitude')
                
                # contour=ax.contourf(var_press, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
                # if i==1:
                #     add_cbar(axes[0,len(vert_norm_list)-1],contour)
            #=====================================
            #plot NR-FG
            ax=axes.flat[2]
            ini_error=var_truth-var_fg
            vmin=ini_error.min()
            vmax=ini_error.max()
            # contour=ax.contourf(ini_error, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
            #=====================================
            for ianal,anal in enumerate(anal_name_list):
                print(f'processing {anal}')
                file=f'{anal_dir}/analysis.ensmean.{anal}'
                ax=axes.flat[ianal+len(vert_norm_list)]
                data=xr.open_dataset(file,engine='netcdf4')
                
                # Get the location of the TC center
                iTC,jTC=iTC_NR,jTC_NR
                print(f'calculating {var} at {traget_press} of {anal}')
                var_press=extract_xxhpa_data(file,iTC=iTC,jTC=jTC,press=traget_press,radius=radius)[var]
                
                # if plot_increment:
                #     var_press=var_press-var_fg
                if ianal==0:
                    vmin=var_press.min()
                    vmax=var_press.max()
                np.savetxt(f'{cache_dir}/{var}_{anal}_{traget_press}.txt',var_press)
                ax.set_title(anal,fontsize=12)
                # ax.set_xlabel('Longitude')
                # ax.set_ylabel('Latitude')
                # contour=ax.contourf(var_press, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
                # if (ianal+1)%len(vert_norm_list)==0:
                #     add_cbar(ax,contour)
        #=============================================================================================================
        if save_fig:
            #make sure save fig dir exists
            subprocess.run(['mkdir','-p',f'{pwd}/figs/postAssimAnal'])
            plt.subplots_adjust(right=0.85)
            # cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            # fig.colorbar(contour,cax=cbar_ax,label='Depth (m)')
            #delete blank subfigs
            for ax in axes.flat:
                if not ax.has_data():
                    fig.delaxes(ax)
            plt.savefig(f"{pwd}/figs/postAssimAnal/{var}_{traget_press}_onlyTCarea_{str(onlyTCarea)}_r{radius}_{suffix_out}.png",format='png',dpi=300,bbox_inches='tight')
