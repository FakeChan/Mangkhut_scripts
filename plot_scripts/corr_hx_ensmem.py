from param import om_z,eta
from matplotlib.scale import FuncScale, register_scale
# 定义分段变换函数
def piecewise_scale(y):
    y = np.asarray(y)
    return np.where(y <= 100, y*4, (y + 1300)*2/7)  # 0-100映射到0-400，100-800映射到400-600

# 定义逆变换函数
def piecewise_inverse(y):
    y = np.asarray(y)
    return np.where(y <= 200, y/4, y*3.5 - 1300)

# 注册自定义比例尺
class PiecewiseScale(FuncScale):
    name = 'piecewise'
    def __init__(self, axis):
        FuncScale.__init__(self, axis, functions=(piecewise_scale, piecewise_inverse))

register_scale(PiecewiseScale)
import os
pwd="/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts"
os.chdir(pwd)
import sys
import numpy as np
import netCDF4 as nc
import wrf
from kc_functions import nc_read1,match_typhoon_grid
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import random

if __name__ == "__main__":
    #==================================================
    #basic settings
    work_dir="/share/home/lililei1/kcfu/tc_mangkhut"
    obs_dir=f"{work_dir}/3create_obs/hx_rttov/3obs_BT"
    ensmem_dir=f"/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst"
    ens_hx_dir=f'{work_dir}/3create_obs/hx_rttov/4ens_BT'
    NR_path='/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/wrfout_d03_2018-09-10_00:00:00'
    domain='d01'
    day_list=['10']
    hour_list=['00']
    minute='00'
    itime=0
    wrf_time=day_list[itime]+'_'+hour_list[itime]+':'+minute#10_00:00
    time=day_list[itime]+'_'+hour_list[itime]+'_'+minute#10_00_00
    mem_list=list(np.arange(1,51))
    varname_list=['T','OM_TMP','U','OM_U','QVAPOR']
    channel_list='1,2,3,4'.split(',')
    nobs=676
    nloop=50
    loop_size=30 #randomly get $loopsize obs each time from $nobs*nmem total obs 
    instrument='AMSUA'
    read_cache=False
    plot_all_channel=False
    #==================================================
    #configure of wrfout
    nAtmos_level=56
    nOcean_level=30
    p_bot=1000.
    p_top=50.

    # p=eta*(p_bot-p_top)+p_top
    p=np.arange(0,57)
    #PLEASE just fill the j and i_start with the RAW number in namelist.input
    i_parent_start=27
    j_parent_start=88

    d02_start_dd='09'
    d02_start_hh='06'
    time_index=int(((int(day_list[itime])-int(d02_start_dd))*24
                +int(hour_list[itime])-int(d02_start_hh))/3)
    
    NR_jlist=np.loadtxt(f'{work_dir}/0necessay_files/NR_d02_jlist.txt')
    NR_ilist=np.loadtxt(f'{work_dir}/0necessay_files/NR_d02_ilist.txt')
    #calc the TC center index in d01(domain)
    lat,lon,iCenter_index,jCenter_index=match_typhoon_grid(NR_path,f'{ensmem_dir}/mem001/wrfout_d01_2018-09-10_00:00:00')
    for ich,chnum in enumerate(channel_list):
        print(f'chnnel{chnum} of {instrument} is calc')
        #read ensmem vars and calc corr
    
        if read_cache:
            dfs=[]
            for ivar,varname in enumerate(varname_list):
                corr=pd.read_csv(f'{pwd}/corr_cache/mean_ch{chnum}_corr_{varname}.csv')
                dfs.append(corr)
        else:
            length_of_obs=np.sqrt(nobs)#the length of the area where obs locates
            radius=int(length_of_obs/2)
            ens_Jslice=slice(jCenter_index-radius+1,jCenter_index+radius+1)
            ens_Islice=slice(iCenter_index-radius+1,iCenter_index+radius+1)
            obs_list=np.arange(nobs).tolist()
            loop_list=np.arange(nloop).tolist()
            dfs=[]
            for ivar,varname in enumerate(varname_list):
                wrfout_ens=[]
                ens_hx=[]
                #-------------------------
                #read hx and x in case of large calculation
                for imem,member in enumerate(mem_list):
                    member="mem{:03d}".format(member)
                    member_dir=f"{ensmem_dir}/{member}"
                    hx=np.loadtxt(f'{ens_hx_dir}/{member}/{instrument}/BT_{time}/obs_{domain}_ch{chnum}_totalline.txt')
                    wrf_name=f'{member_dir}/wrfout_{domain}_2018-09-{wrf_time}:00'
                    wrf_var=wrf.to_np(nc_read1(wrf_name,varname)[:,ens_Jslice,ens_Islice])
                    if varname=='T':
                        p_var=wrf.to_np(nc_read1(wrf_name,'P')[:,ens_Jslice,ens_Islice])
                        pb=wrf.to_np(nc_read1(wrf_name,'PB')[:,ens_Jslice,ens_Islice])
                        p_var=p_var+pb
                        wrf_var=(wrf_var+300)*((p_var/100000.0)**0.286) #potential temp to true temp
                    wrfout_ens.append(wrf_var)
                    ens_hx.append(hx) 
                #-----------------------
                #judge if the var is Oceanic(OM) or not
                if varname[:2] == 'OM':
                    zlevel=nOcean_level
                else:
                    zlevel=nAtmos_level
                level_list=np.arange(zlevel).tolist()
                corr_record=pd.DataFrame(np.nan,index=level_list,columns=loop_list)
                #---------------------
                #each obs point,calc corr
                for nlevel in range(zlevel):
                    
                    icount=0
                    while icount<nloop:
                        var_mem_list=[]
                        ens_hx_list=[]
                        for i in range(loop_size):
                            imem=random.randint(0,len(mem_list)-1)
                            iobs=random.randint(0,nobs-1)

                            wrf_var2=wrfout_ens[imem]
                            wrf_var_level=np.reshape(wrf_var2[nlevel],nobs*1,order='F')
                            #append list
                            var_mem_list.append(wrf_var_level[iobs])
                            ens_hx_list.append(ens_hx[imem][iobs])  
                        corr=np.corrcoef(var_mem_list,ens_hx_list)[0,1]
                        corr_record.loc[nlevel,icount]=corr
                        icount+=1
                subprocess.run(['mkdir','-p',f'{pwd}/corr_cache'])
                #calc mean npoints corr at each level
                corr_mean=corr_record.mean(axis=1)
                corr_mean.to_csv(f'{pwd}/corr_cache/mean_ch{chnum}_corr_{varname}.csv')
                dfs.append(corr_mean)
            # df_mean=sum(dfs)/len(dfs)
            # df_mean.to_csv(f'{pwd}/corr_cache/corr_mean.csv')
        #===================================================
        #plot
        #v1 temporate version
        if not plot_all_channel:
            fig,axs=plt.subplots(1,len(varname_list), figsize=(4*len(varname_list),2*len(varname_list)))
            kc_blue="#091508"
            for ivar,varname in enumerate(varname_list):
                ax=axs[ivar]
                if read_cache:
                    corr=dfs[ivar].iloc[:,1]
                else:
                    corr=dfs[ivar]
                if varname[:2] != 'OM':
                    ax.plot(corr,p[:-1],linestyle='-',color=kc_blue)
                    ax.set_title(varname)
                    ax.set_ylabel('pressure(hPa)')
                    ax.set_xlabel('correlation')
                    ax.invert_yaxis()
                else:  
                    ax.plot(corr,om_z,color=kc_blue)
                    ax.set_title(varname)
                    ax.set_ylabel('depth(m)')
                    ax.set_xlabel('correlation')
                    ax.set_yscale('piecewise')
                    ax.set_yticks([5, 25, 50, 100, 200, 400, 750])
                    ax.set_yticklabels(['5', '25', '50','100', '200', '400', '750'])
                    # ax.invert_yaxis()
                ax.axvline(x=0,             # x轴位置
                                        color='black',         # 颜色
                                        linestyle='--',    # 线型（实线、虚线等）
                                        linewidth=2)
            plt.savefig(f"{pwd}/figs/corr_profile/corr_ch{chnum}.png",format='png',dpi=300,bbox_inches='tight')
    
    if plot_all_channel:
        fig,axs2=plt.subplots(1,len(varname_list), figsize=(4*len(varname_list),2*len(varname_list)))
        kc_blue="#0072BD"   #"6CACE4"
        kc_red="#D0002E"            #"#D55E00"
        kc_green="#009E73"
        eva2_black= "#091508"
        my_palette = [kc_blue, kc_green, kc_red, eva2_black]
        for ich,chnum in enumerate(channel_list):
            dfs=[]
            for ivar,varname in enumerate(varname_list):
                corrs=pd.read_csv(f'{pwd}/corr_cache/mean_ch{chnum}_corr_{varname}.csv')
                dfs.append(corrs)

                ax=axs2[ivar]
                corr=dfs[ivar].iloc[:,1]

                if varname[:2] != 'OM':
                    ax.plot(corr,p[:-1],linestyle='-',color=my_palette[ich],label=f'ch{ich+1}')
                    ax.set_title(f'BT and {varname}')
                    ax.set_ylabel('pressure(hPa)')
                    ax.set_xlabel('correlation')
                    
                else:  
                    ax.plot(corr,om_z,color=my_palette[ich],label=f'ch{ich+1}')
                    ax.set_title(f'BT and {varname}')
                    ax.set_ylabel('depth(m)')
                    ax.set_xlabel('correlation')
                    ax.set_yscale('piecewise')
                    ax.set_yticks([5, 25, 50, 100, 200, 400, 750])
                    ax.set_yticklabels(['5', '25', '50','100', '200', '400', '750'])
                    
                if ich==len(channel_list)-1:
                    ax.invert_yaxis()
                ax.axvline(x=0,             # x轴位置
                                        color='black',         # 颜色
                                        linestyle='--',    # 线型（实线、虚线等）
                                        linewidth=2)
        plt.legend( 
                    loc='upper right', 
                    bbox_to_anchor=(0.8, -0.10),
                    ncol=len(channel_list), 
                    markerscale=2,
                    fontsize=15)
        
        plt.savefig(f"{pwd}/figs/corr_profile/corr_all_ch.png",format='png',dpi=300,bbox_inches='tight')
        
    
