import xarray as xr
import netCDF4
from wrf import getvar,to_np
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import subprocess
import cmocean
import math
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

def get_d26(om_var, om_z, iTC, jTC, radius, onlyTCarea):
    """
    Calculate the depth of the 26°C isotherm
    :param om_var: temperature variable
    :param om_z: depth variable
    :param iTC: i index of the TC center
    :param jTC: j index of the TC center
    :param radius: radius for averaging
    :param onlyTCarea: boolean to indicate if only TC area should be considered
    :return: d26 array
    """
    if onlyTCarea:
        jTCslice=slice(jTC-radius,jTC+radius)
        iTCslice=slice(iTC-radius,iTC+radius)
        var=om_var[0,:,jTCslice,iTCslice]
        size=var.shape
        d26=np.zeros((size[1],size[2]))*np.nan
        
        for j in range(size[1]):
            for i in range(size[2]):
                d26[j,i]=find_d26(var[:,j,i]-273.15,om_z)
    else:
        var=om_var[0,:,:,:]
        size=var.shape
        d26=np.zeros((size[1],size[2]))*np.nan
        
        for j in range(size[1]):
            for i in range(size[2]):
                d26[j,i]=find_d26(var[:,j,i]-273.15,om_z)
    return d26

if __name__ == "__main__":
    pwd='/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts'
    cache_dir=f'{pwd}/postAnal_cache'
    subprocess.run(['mkdir','-p',f'{cache_dir}'])
    #==========================
    #parameters
    radius=30
    cutoff_list=[0.1,0.05,0.01]
    vert_norm_list=[40000.0,80000.0,400000.0,800000.0]
    var_list=['OM_U','OM_V','OM_TMP']
    onlyTCarea=True
    read_cache=True
    plot_increment=True
    save_fig=read_cache
    single_obs=True
    ocean_level=0    # 0-29 of ocean levels
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
                data=np.loadtxt(f'{cache_dir}/SeaSurf_{var}_{title_list[i]}.txt')
                if title_list[i]=='NR':
                    truth=data
                    vmin=data.min()
                    vmax=data.max()
                elif title_list[i]=='prior':
                    prior=data
                ax=axes.flat[i]
                ax.set_title(title_list[i])
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                
                contour=ax.contourf(data, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
                if i==0:
                    add_cbar(axes[0,len(vert_norm_list)-3],contour)
            #=======================
            #plot NR - FG
            if plot_increment:
                ax=axes.flat[3]
                ini_error=truth-prior
                vmin=ini_error.min()
                vmax=ini_error.max()
                contour=ax.contourf(truth-prior, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
                add_cbar(ax,contour)
            #=======================  
            for ianal,anal in enumerate(anal_name_list):
                data=np.loadtxt(f'{cache_dir}/SeaSurf_{var}_{anal}.txt')
                if plot_increment:
                    data=data-prior
                    if ianal == 0:
                        vmin=data.min()
                        vmax=data.max()
                ax=axes.flat[ianal+len(vert_norm_list)]
                ax.set_title(anal)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                contour=ax.contourf(data, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
                if (ianal+1)%len(vert_norm_list)==0:
                    add_cbar(ax,contour)
        #=============================================================================================
        else:
            #first plot NR and fg
            for i in range(nfile):
                print(f'processing {title_list[i]}')
                file=file_list[i]
                ax=axes.flat[i]
                data=xr.open_dataset(file,engine='netcdf4')
                # Get the location of the TC center
                if i==0:
                    iTC_NR, jTC_NR = getTClocation(file)
                iTC,jTC=iTC_NR,jTC_NR
                if onlyTCarea:
                    jTCslice=slice(jTC-radius,jTC+radius)
                    iTCslice=slice(iTC-radius,iTC+radius)
                    om_var=data[var][0,ocean_level,jTCslice,iTCslice]
                else:
                    om_var=data[var][0,ocean_level,:,:]
                if title_list[i]=='NR':
                    truth=om_var
                    vmin=om_var.min().item() #om_var is xaaray
                    vmax=om_var.max().item()
                elif title_list[i]=='prior':
                    prior=om_var
                om_z=data['OM_DEPTH'][0,:,0,0]
                
                
                print(f'saving SEA SRUF {var} of {title_list[i]}')
                # d26=get_d26(om_var, om_z, iTC, jTC, radius, onlyTCarea)
                np.savetxt(f'{cache_dir}/SeaSurf_{var}_{title_list[i]}.txt',om_var)
                # ax.set_title(title_list[i])
                # ax.set_xlabel('Longitude')
                # ax.set_ylabel('Latitude')
                
                # contour=ax.contourf(om_var, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
                # if i==0:
                #     add_cbar(axes[0,len(vert_norm_list)-1],contour)
            #=======================
            #plot NR - FG
            ax=axes.flat[2]
            ini_error=truth-prior
            vmin=ini_error.min()
            vmax=ini_error.max()
            # contour=ax.contourf(truth-prior, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
            #=======================  
            for ianal,anal in enumerate(anal_name_list):
                print(f'processing {anal}')
                file=f'{anal_dir}/analysis.ensmean.{anal}'
                ax=axes.flat[ianal+len(vert_norm_list)]
                data=xr.open_dataset(file,engine='netcdf4')
                
                # Get the location of the TC center
                iTC,jTC=iTC_NR,jTC_NR
                if onlyTCarea:
                    jTCslice=slice(jTC-radius,jTC+radius)
                    iTCslice=slice(iTC-radius,iTC+radius)
                    om_var=data[var][0,ocean_level,jTCslice,iTCslice]
                else:
                    om_var=data[var][0,ocean_level,:,:]
                    
                if ianal==0:
                    vmin=om_var.min().item() #om_var is xaaray
                    vmax=om_var.max().item()
                    
                print(f'saving SeaSurf {var} of {anal}')
                # d26=get_d26(om_var, om_z, iTC, jTC, radius, onlyTCarea)
                np.savetxt(f'{cache_dir}/SeaSurf_{var}_{anal}.txt',om_var)
                # ax.set_title(anal)
                # ax.set_xlabel('Longitude')
                # ax.set_ylabel('Latitude')
                # contour=ax.contourf(om_var, cmap=cmap, levels=10,vmin=vmin,vmax=vmax)
                # if (ianal+1)%len(vert_norm_list)==0:
                #     add_cbar(ax,contour)
        #make sure save fig dir exists
        #======================================================================================
        if save_fig:
            subprocess.run(['mkdir','-p',f'{pwd}/figs/postAssimAnal'])
            plt.subplots_adjust(right=0.85)
            # cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            # fig.colorbar(contour,cax=cbar_ax,label='Depth (m)')
            #delete blank subfigs
            for ax in axes.flat:
                if not ax.has_data():
                    fig.delaxes(ax)
            plt.savefig(f"{pwd}/figs/postAssimAnal/ocean_SeaSurf_{var}_onlyTCarea_{str(onlyTCarea)}_r{radius}_{suffix_out}.png",format='png',dpi=300,bbox_inches='tight')
