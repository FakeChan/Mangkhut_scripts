import numpy as np
f_obs='/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/obs_d01/obsinput_10_00_00.txt'
f_ensBT='/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/mergeFOd01_10_00/merged_FO_data.txt'

obs=np.loadtxt(f_obs)
ensBT=np.loadtxt(f_ensBT)
print(np.shape(obs))

ens_mean=np.mean(ensBT,axis=1)
obs=obs[:,10]
omb=obs-ens_mean
index=np.argmax(omb)
print(index)
np.savetxt('merged_maxinv_FO.txt',ensBT[index,:].reshape(1,-1),fmt='%.4f')