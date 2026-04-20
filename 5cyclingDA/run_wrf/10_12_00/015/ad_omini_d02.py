import netCDF4 as nc
import numpy as np
import os 
# ================= 设置文件路径 =================
file_a_path = 'wrfinput_d02'  # 目标文件 (接收数据)
file_b_path = 'wrfinput_d01_gfs'  # 源文件 (提供垂直廓线)
vars_to_copy = ['OM_TINI', 'OM_SINI']
# ===============================================

# 1. 打开文件
ds_a = nc.Dataset(file_a_path, 'r+')  # 读写
ds_b = nc.Dataset(file_b_path, 'r')   # 只读

# 2. 获取 A 的水平网格大小
# 通常维度是 (Time, bottom_top, south_north, west_east)
# 注意：你需要确认 A 中用来代表垂直层的维度名称，通常可能是 'bottom_top' 或 'zp_max' 等
# 这里我们动态获取 A 的维度大小
dims_a = ds_a.dimensions
ny = dims_a['south_north'].size
nx = dims_a['west_east'].size

print(f"目标网格 A 大小: south_north={ny}, west_east={nx}")

for var_name in vars_to_copy:
    if var_name in ds_b.variables:
        print(f"正在处理: {var_name} ...")
        
        # --- 步骤 1: 从 B 提取垂直廓线 ---
        src_var = ds_b.variables[var_name]
        # 假设 B 的数据形状是 [Time, Vertical, Lat, Lon]
        # 我们取第一个时间层，中间的经纬度点 (避免边界效应，虽然理论上它是均一的)
        # 如果 B 是 1D (Time, Vertical)，代码会自动适配
        
        if src_var.ndim == 4: # [Time, Z, Y, X]
            # 取中间的一个点作为代表
            mid_y = src_var.shape[2] // 2
            mid_x = src_var.shape[3] // 2
            profile = src_var[0, :, mid_y, mid_x] # 结果是一维数组 [Z]
        elif src_var.ndim == 2: # [Time, Z] 只有垂直层
            profile = src_var[0, :]
        else:
            print(f"警告：变量 {var_name} 的维度结构 {src_var.shape} 不常见，跳过。")
            continue
            
        print(f"  > 提取垂直层数: {len(profile)}")

        # --- 步骤 2: 写入 A (广播数据) ---
        # 检查变量是否在 A 中，不在则创建
        if var_name not in ds_a.variables:
            print(f"  > 在 A 中创建变量 {var_name}...")
            # 注意：这里我们必须使用 A 文件的维度名称
            # 假设维度顺序为 (Time, bottom_top, south_north, west_east)
            # 这里的 'bottom_top' 需要你确认一下是否对应海洋层，有时候叫 'layers_ocean'
            
            # 自动寻找对应的垂直维度名 (根据大小匹配)
            z_dim_name = None
            for dname in ds_a.dimensions:
                if ds_a.dimensions[dname].size == len(profile):
                    z_dim_name = dname
                    break
            
            if z_dim_name is None:
                # 如果自动匹配失败，默认尝试使用源变量的维度名，或者你需要手动指定
                # 常见海洋维度: 'zp_max', 'k_m', 'bottom_top'
                z_dim_name = src_var.dimensions[1] 
                print(f"  > 警告: 未能自动匹配垂直维度，尝试使用名称: {z_dim_name}")

            dst_dims = ('Time', z_dim_name, 'south_north', 'west_east')
            
            dst_var = ds_a.createVariable(var_name, src_var.dtype, dst_dims)
            dst_var.setncatts(src_var.__dict__) # 复制属性
        else:
            dst_var = ds_a.variables[var_name]

        # --- 步骤 3: 逐层填充 (极速且省内存) ---
        # 不需要构建巨大的 3D 数组，我们一层一层填，内存占用极小
        for k in range(len(profile)):
            val = profile[k]
            # 将该层的所有网格点赋值为同一个值
            dst_var[0, k, :, :] = val
        
        print(f"  > {var_name} 写入完成。")

    else:
        print(f"错误: B 中没有 {var_name}")

ds_a.close()
ds_b.close()
print("所有操作完成。")
