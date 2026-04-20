from wrf import getvar
import netCDF4
import xarray as xr
file='/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/postAssimData/10_00_00/singleobs/analysis.mem001.horizLoc1280.vertLoc8'
ds=netCDF4.Dataset(file)
data=xr.open_dataset(file,engine='netcdf4')
var=data['THM'].values[0,10,:,:]
print(var)