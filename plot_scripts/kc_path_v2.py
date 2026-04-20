import os
import glob
import netCDF4 as nc
import numpy as np
import pandas as pd
from wrf import getvar, latlon_coords

def analyze_ensemble_members(ensemble_base_dirs, output_file):
    """
    分析WRF集合预报成员，使用 wrf-python 提取台风路径和强度。

    Args:
        ensemble_base_dirs (list): 包含集合成员目录的列表。
        output_file (str): 输出CSV文件的路径。
    """
    all_tracks_data = []

    for member_dir in ensemble_base_dirs:
        if not os.path.isdir(member_dir):
            print(f"警告: 目录 '{member_dir}' 不存在，已跳过。")
            continue

        member_name = os.path.basename(member_dir)
        # wrfout文件通常很大，我们可以通过组合文件来处理
        # 注意：这里假设文件是按时间顺序命名的
        wrfout_files = sorted(glob.glob(os.path.join(member_dir, 'wrfout_d01_2018*')))

        if not wrfout_files:
            print(f"警告: 在 '{member_dir}' 中未找到 wrfout 文件，已跳过。")
            continue

        print(f"正在处理成员: {member_name}")

        try:
            # 将多个wrfout文件集合成一个数据集进行处理
            wrf_dataset = [nc.Dataset(f) for f in wrfout_files]
            
            # 获取总的时间步数
            num_times = getvar(wrf_dataset, "times", timeidx=None).shape[0]

            for i in range(num_times):
                # 使用 getvar 诊断 SLP
                slp = getvar(wrf_dataset, "slp", timeidx=i)
                
                # 获取经纬度坐标
                lats, lons = latlon_coords(slp)
                
                # 找到最低SLP的值和位置
                min_slp_val = float(np.min(slp).values)
                min_slp_idx = np.unravel_index(np.argmin(slp.values), slp.shape)
                
                # 获取最低SLP点的经纬度
                track_lat = float(lats[min_slp_idx].values)
                track_lon = float(lons[min_slp_idx].values)
                
                # 获取时间信息
                time_str = pd.to_datetime(slp.Time.values).strftime('%Y-%m-%d_%H:%M:%S')

                all_tracks_data.append({
                    'member': member_name,
                    'time': time_str,
                    'min_slp': min_slp_val,
                    'lat': track_lat,
                    'lon': track_lon
                })
                print(f"  - 时间: {time_str}, 最低SLP: {min_slp_val:.2f} hPa")

        except Exception as e:
            print(f"处理成员 '{member_name}' 时出错: {e}")
        finally:
            # 确保关闭所有打开的文件
            if 'wrf_dataset' in locals():
                for ds in wrf_dataset:
                    ds.close()


    # 创建DataFrame并保存到CSV
    if all_tracks_data:
        df = pd.DataFrame(all_tracks_data)
        df.sort_values(by=['member', 'time'], inplace=True)
        df.to_csv(output_file, index=False)
        print(f"\n成功将所有成员的路径数据写入 '{output_file}'")
    else:
        print("\n未提取到任何数据。")
# --- 脚本执行入口 ---
if __name__ == "__main__":
    # --- 用户需要修改的部分 ---
    
    # 1. 集合成员的目录路径
    #    例如: ['/share/home/mem001', '/share/home/mem002', ...]
    memlist=list(np.arange(1,30))
    memlist.extend([31,32,36,38,39,40,41,42,43,45,47,48,49,50,51,54,62,69,70,71,73])
    ensemble_dirs = [f'/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst/mem{str(i).zfill(3)}/' for i in memlist] # 示例：mem001 到 mem071

    # 2. 输出文件的路径
    output_csv_path = '/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/tracks.csv'

    # --- 执行分析 ---
    analyze_ensemble_members(ensemble_dirs, output_csv_path)
