import netCDF4
import numpy as np

# 打开 DART 生成的 inflation restart 文件
domain_list=['d01','d02']
inf=2.0
for idomain,domain in enumerate(domain_list):
    nc = netCDF4.Dataset(f'input_priorinf_mean_{domain}.nc', 'r+')

    
    atmos_vars = ['OM_TMP','OM_U','OM_V','OM_S'] 

    for var_name in atmos_vars:
        if var_name in nc.variables:
            print(f"Setting inflation to {inf} for {var_name} (Atmosphere)")
            # 获取变量数据
            var_data = nc.variables[var_name][:]
            var_data[:] = inf 
            # 写回文件
            nc.variables[var_name][:] = var_data

    # 其他变量 (如 SALT, TEMP, U_CUR 等) 保持原值 (例如 1.05) 不变
    nc.close()
    print(f"{domain} Done. Atmosphere inflation locked.")
