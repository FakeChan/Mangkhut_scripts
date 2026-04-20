def interp_var(our_lons,our_lats,x,y,data_12,nx,ny):
    data=data_12
    # our_lons=data_75.XLONG_U.data.flatten()
    # our_lats=data_75.XLAT_U.data.flatten()
    # x = data.XLONG_U.data.flatten()
    # y = data.XLAT_U.data.flatten()
    z = data.data.flatten()
    
    # 使用 pyproj.transform() 将这些网格坐标点从 WRF 模型的投影坐标系转换回经纬度坐标系（PlateCarree投影），结果存储在 our_lons 和 our_lats
    #our_lons, our_lats = pyproj.transform(proj_75, proj_12, xx, yy)
    z_target_grid = griddata((x, y), z, (our_lons, our_lats), method='cubic')
    z_target_grid = z_target_grid.reshape((ny,nx))
    return z_target_grid

import time
import numpy as np
import xarray as xr
import netCDF4 as nc
from scipy.interpolate import griddata
import os
domain='d01'
start=time.perf_counter()
ds_12=xr.open_dataset('./pert_d01_080606_001')
os.system('cp wrfinput_'+domain+' wrfinput_'+domain+'_copy')
ds_75=xr.open_dataset('./wrfinput_'+domain)
ds_rewrite=xr.open_dataset('./wrfinput_'+domain+'_copy')#覆写的文件
vars=list(ds_75.data_vars.keys())
nvar=len(vars)

for i in range(nvar):
    ivar=vars[i]
    print(ivar+' will be interped.')
    dims=ds_75.variables[ivar].dims
    #维度   
    ndim=len(dims)
    if ndim <= 2 :
        continue
    elif ndim ==3:
        nz = 1
    else:
        nz=ds_75.variables[ivar].shape[-3]
    sn=dims[-2]
    we=dims[-1]
    nx, ny = ds_75.dims[we], ds_75.dims[sn]
    #nz=ds_75.variables[ivar].shape[-3]
    var_interp=np.zeros([nz,ny,nx])
    #======================================
    #判断stagger类型
    stagger_type=ds_75.variables[ivar].attrs['stagger']
    if stagger_type=='X':
        our_lons=ds_75.XLONG_U.data.flatten()
        our_lats=ds_75.XLAT_U.data.flatten()
        x=ds_12.XLONG_U.data.flatten()
        y=ds_12.XLAT_U.data.flatten()
    elif stagger_type=='Y':
        our_lons=ds_75.XLONG_V.data.flatten()
        our_lats=ds_75.XLAT_V.data.flatten()
        x=ds_12.XLONG_V.data.flatten()
        y=ds_12.XLAT_V.data.flatten()
    else:
        our_lons=ds_75.XLONG.data.flatten()
        our_lats=ds_75.XLAT.data.flatten()
        x=ds_12.XLONG.data.flatten()
        y=ds_12.XLAT.data.flatten()
    #======================================
    #判断是否有单位
    try:
        iunit=ds_75.variables[ivar].attrs['units']
    except:
        print(ivar+' has trouble when getting unit. Continue')
        continue
    unit_exist= iunit != ''
    if unit_exist:
        try:
            if ndim==4:
                for k in range(nz):
                    data_12=ds_12.variables[ivar][0,k]
                    data_75=ds_75.variables[ivar][0,k]
                    var_interp[k,:,:]=interp_var(our_lons,our_lats,x,y,data_12,nx,ny)
                
                ds_rewrite.variables[ivar][0]=var_interp
            
            elif ndim==3:
                for k in range(nz):
                    data_12=ds_12.variables[ivar][0]
                    data_75=ds_75.variables[ivar][0]
                    var_interp[k,:,:]=interp_var(our_lons,our_lats,x,y,data_12,nx,ny)
                
                ds_rewrite.variables[ivar][0]=var_interp[0]
        except:
            print(ivar+' went something wrong. Continue')
            continue
    else:
        print(ivar+' has no unit. Continue')
        continue
        

ds_rewrite.to_netcdf('./wrfout_interp_'+domain+'.nc')
end=time.perf_counter()

print('This script takes %s s to finish.'%(end-start))
