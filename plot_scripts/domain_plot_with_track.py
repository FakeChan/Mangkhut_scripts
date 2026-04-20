import f90nml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import pandas as pd

def get_wrf_domain_info(namelist_path):
    """
    解析namelist.wps文件,计算所有域的几何信息和总绘图范围。
    返回地图投影、各域的位置尺寸列表，以及总的绘图范围。
    """
    if not os.path.exists(namelist_path):
        print(f"错误: 未找到文件 '{namelist_path}'")
        return None, None, None

    try:
        nml = f90nml.read(namelist_path)
        geogrid_nml = nml['geogrid']
    except Exception as e:
        print(f"解析 namelist 文件时出错: {e}")
        return None, None, None

    # 创建地图投影
    map_proj_nml = geogrid_nml['map_proj'].strip().lower()
    ref_lat = geogrid_nml['ref_lat']
    ref_lon = geogrid_nml['ref_lon']
    truelat1 = geogrid_nml['truelat1']
    truelat2 = geogrid_nml.get('truelat2', truelat1)
    stand_lon = geogrid_nml.get('stand_lon', ref_lon)
    
    if map_proj_nml == 'lambert':
        projection = ccrs.LambertConformal(
            central_longitude=stand_lon, central_latitude=ref_lat,
            standard_parallels=(truelat1, truelat2))
    else:
        raise ValueError(f"不支持的地图投影: {map_proj_nml}。")

    # 提取并计算每个域的边界
    e_we_list = np.atleast_1d(geogrid_nml['e_we'])
    e_sn_list = np.atleast_1d(geogrid_nml['e_sn'])
    max_dom = len(e_we_list)
    
    ################
    #fkc:暂时用1
    max_dom=1
    ################
    ratio_list = np.atleast_1d(geogrid_nml.get('parent_grid_ratio', [1]*max_dom))
    i_start_list = np.atleast_1d(geogrid_nml.get('i_parent_start', [1]*max_dom))
    j_start_list = np.atleast_1d(geogrid_nml.get('j_parent_start', [1]*max_dom))

    domain_corners = []
    domain_rects = []
    
    parent_dx = geogrid_nml['dx']
    parent_dy = geogrid_nml['dy']
    width_d01 = (e_we_list[0] - 1) * parent_dx
    height_d01 = (e_sn_list[0] - 1) * parent_dy
    parent_ll_x = -width_d01 / 2.0
    parent_ll_y = -height_d01 / 2.0

    for i in range(max_dom):
        e_we = e_we_list[i]
        e_sn = e_sn_list[i]

        if i == 0:
            dx, dy = parent_dx, parent_dy
            ll_x, ll_y = parent_ll_x, parent_ll_y
        else:
            ratio = ratio_list[i]
            i_start = i_start_list[i]
            j_start = j_start_list[i]
            dx, dy = parent_dx / ratio, parent_dy / ratio
            offset_x = (i_start - 1) * parent_dx
            offset_y = (j_start - 1) * parent_dy
            ll_x, ll_y = parent_ll_x + offset_x, parent_ll_y + offset_y
            
        width = (e_we - 1) * dx
        height = (e_sn - 1) * dy
        
        domain_rects.append({'ll_x': ll_x, 'll_y': ll_y, 'width': width, 'height': height})
        domain_corners.extend([[ll_x, ll_y], [ll_x + width, ll_y + height]])
        
        parent_dx, parent_dy = dx, dy
        parent_ll_x, parent_ll_y = ll_x, ll_y

    corners = np.array(domain_corners)
    min_x, min_y = corners.min(axis=0)
    max_x, max_y = corners.max(axis=0)
    buffer_x = (max_x - min_x) * 0.1
    buffer_y = (max_y - min_y) * 0.1
    extent = [min_x - buffer_x, max_x + buffer_x, min_y - buffer_y, max_y + buffer_y]
    
    return projection, domain_rects, extent

def read_NR_track(lon_file, lat_file):
    """
    从两个单独的文本文件中读取 NR 路径的经纬度。
    假定每个文件只包含一列数字。
    """
    if not os.path.exists(lon_file):
        print(f"警告: 未找到 NR 路径经度文件 '{lon_file}', 将不绘制 NR 路径。")
        return None, None
    if not os.path.exists(lat_file):
        print(f"警告: 未找到 NR 路径纬度文件 '{lat_file}', 将不绘制 NR 路径。")
        return None, None
        
    try:
        # 使用 numpy.loadtxt 读取简单的文本列数据
        nr_lons = np.loadtxt(lon_file)
        nr_lats = np.loadtxt(lat_file)
        
        if nr_lons.size == 0 or nr_lats.size == 0:
            print(f"警告: NR 路径文件 '{lon_file}' 或 '{lat_file}' 为空。")
            return None, None
            
        if nr_lons.shape != nr_lats.shape:
             print(f"警告: NR 路径经纬度文件点数不匹配。 Lons: {nr_lons.size}, Lats: {nr_lats.size}")
             # 尝试截取为最短长度
             min_len = min(nr_lons.size, nr_lats.size)
             nr_lons = nr_lons[:min_len]
             nr_lats = nr_lats[:min_len]

        return nr_lons, nr_lats
    except Exception as e:
        print(f"读取 NR 路径文件时出错: {e}")
        return None, None

def plot_NR_track(ax, lons, lats, map_projection):
    """
    在给定的ax上绘制“NR 路径”。
    """
    # 定义 track 数据的原始坐标系（经纬度）
    geodetic_proj = ccrs.PlateCarree()
    
    try:
        # --- 手动将经纬度转换为地图的投影坐标 ---
        transformed_points = map_projection.transform_points(geodetic_proj, lons, lats)
        
        # 绘制 NR 路径
        ax.plot(transformed_points[:, 0], transformed_points[:, 1],
                linewidth=3.0, color='yellow', 
                marker='+', markersize=5,
                label='NR') # <-- 标签为 NR Typhoon
    except Exception as e:
        print(f"绘制 NR 路径时出错: {e}")

def plot_ensemble_tracks(ax, tracks_csv_path, map_projection):
    """
    在给定的ax上绘制集合成员的台风路径。
    """
    if not os.path.exists(tracks_csv_path):
        print(f"警告: 未找到路径文件 '{tracks_csv_path}', 将不绘制路径。")
        return
    try:
        df = pd.read_csv(tracks_csv_path)
        if df.empty:
            print(f"警告: 路径文件 '{tracks_csv_path}' 为空。")
            return
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return
    
    member_col = 'member'
    if 'member' not in df.columns or df['member'].isnull().all():
        first_time = df['time'].iloc[0]
        num_members = (df['time'] == first_time).sum()
        if num_members > 0:
            df['synthetic_member'] = df.index % num_members
            member_col = 'synthetic_member'
        else:
            return

    geodetic_proj = ccrs.PlateCarree()
    
    # 循环处理每个成员
    for member_id in df[member_col].unique():
        member_data = df[df[member_col] == member_id]
        lons = member_data['lon'].values
        lats = member_data['lat'].values
        transformed_points = map_projection.transform_points(geodetic_proj, lons, lats)
        ax.plot(transformed_points[:, 0], transformed_points[:, 1],
                linewidth=0.5, alpha=0.7, color='gray')
    
    # 对集合平均路径也执行相同的操作
    mean_track = df.groupby('time')[['lon', 'lat']].mean().reset_index()
    mean_lons = mean_track['lon'].values
    mean_lats = mean_track['lat'].values
    mean_transformed = map_projection.transform_points(geodetic_proj, mean_lons, mean_lats)
    ax.plot(mean_transformed[:, 0], mean_transformed[:, 1],
            linewidth=1.5, alpha=1.0, color='black', label='Ensemble Mean Track')

def read_bdeck_track(best_track_file_path):
    """
    从 b-deck (如 bwp262018.dat) 文件中读取最佳路径的经纬度。
    """
    if not os.path.exists(best_track_file_path):
        print(f"警告: 未找到 b-deck 路径文件 '{best_track_file_path}', 将不绘制 b-deck 路径。")
        return None, None
        
    try:
        # 使用 pandas 读取 csv, 第3列(索引2)是时间, 第7列(索引6)是纬度, 第8列(索引7)是经度
        df = pd.read_csv(best_track_file_path, header=None, sep=',', 
                         on_bad_lines='skip', engine='python')

        if df.empty:
             print(f"警告: b-deck 路径文件 '{best_track_file_path}' 为空或无法解析。")
             return None, None
        
        # b-deck文件经常在同一时间点有多个条目（例如不同风圈半径）
        # 我们只保留基于时间、纬度、经度的唯一组合
        # 索引 2: 时间 (例如 2018090618)
        # 索引 6: 纬度 (例如 120N)
        # 索引 7: 经度 (例如 1693E)
        df_unique = df.drop_duplicates(subset=[2, 6, 7])
        
        # .str.strip() 对于CSV中的空格至关重要
        lat_strs = df_unique[6].str.strip()
        lon_strs = df_unique[7].str.strip()
        
        # 转换经纬度格式 (例如 '120N' -> 12.0, '1693E' -> 169.3)
        best_lats = lat_strs.apply(lambda x: float(x[:-1]) / 10.0 if x.endswith('N') else -float(x[:-1]) / 10.0)
        best_lons = lon_strs.apply(lambda x: float(x[:-1]) / 10.0 if x.endswith('E') else -float(x[:-1]) / 10.0)

        return best_lons.values, best_lats.values
        
    except Exception as e:
        print(f"读取或解析 b-deck 路径文件 '{best_track_file_path}' 时出错: {e}")
        return None, None

def plot_bdeck_track(ax, lons, lats, map_projection):
    """
    在给定的ax上绘制“Best Track”（来自 b-deck）。
    """
    geodetic_proj = ccrs.PlateCarree()
    
    try:
        transformed_points = map_projection.transform_points(geodetic_proj, lons, lats)
        
        # 绘制 Best Track，使用不同样式（例如虚线、不同颜色）
        ax.plot(transformed_points[:, 0], transformed_points[:, 1],
                linewidth=2.0, alpha=0.9, color='magenta', # 使用洋红色
                linestyle='--', # 使用虚线
                label='Best Track (b-deck)')
    except Exception as e:
        print(f"绘制 b-deck 路径时出错: {e}")


def plot_domains_and_tracks(namelist_path, tracks_csv_path, output_path, 
                            nr_track_lon_file, nr_track_lat_file,
                            bdeck_track_file):
    """主绘图函数"""
    projection, domain_rects, extent = get_wrf_domain_info(namelist_path)

    if projection is None:
        print("获取域信息失败，无法继续绘图。")
        return

    extent_width = extent[1] - extent[0]
    extent_height = extent[3] - extent[2]
    aspect_ratio = extent_width / extent_height
    
    fig_height = 10
    fig_width = fig_height * aspect_ratio
    max_fig_size = 20
    if fig_width > max_fig_size:
        fig_width = max_fig_size
        fig_height = fig_width / aspect_ratio
    if fig_height > max_fig_size:
        fig_height = max_fig_size
        fig_width = fig_height * aspect_ratio

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_title("WRF Domain Configuration and Ensemble Tracks", fontsize=16)
    ax.set_extent(extent, crs=projection)

    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='black')
    ax.coastlines(resolution='50m')

    for i, rect in enumerate(domain_rects):
        domain_color = 'blue' if i == 0 else 'red'
        ax.add_patch(mpatches.Rectangle((rect['ll_x'], rect['ll_y']), rect['width'], rect['height'],
            fill=None, edgecolor=domain_color, linewidth=2))
        ax.text(rect['ll_x'] + rect['width'] * 0.02, rect['ll_y'] + rect['height'] * 0.02, f'd0{i+1}',
                color=domain_color, fontsize=12, weight='bold')

    # 1. 读取并绘制 NR 路径 (红色实线)
    nr_lons, nr_lats = read_NR_track(nr_track_lon_file, nr_track_lat_file)
    if nr_lons is not None and nr_lats is not None:
        plot_NR_track(ax, nr_lons, nr_lats, projection)
        
    # 2. 读取并绘制 b-deck 路径 (洋红色虚线)
    bdeck_lons, bdeck_lats = read_bdeck_track(bdeck_track_file)
    if bdeck_lons is not None and bdeck_lats is not None:
        plot_bdeck_track(ax, bdeck_lons, bdeck_lats, projection)
    
    # 3. 绘制集合路径 (灰色和黑色)
    plot_ensemble_tracks(ax, tracks_csv_path, projection)
    
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels, gl.right_labels = False, False
    
    d01_patch = mpatches.Patch(color='blue', label='d01')
    # d02_patch = mpatches.Patch(color='red', label='d02+')
    
    # 自动获取所有已添加的标签
    handles, labels = ax.get_legend_handles_labels()
    # 添加域的标签
    handles.extend([d01_patch])
    ax.legend(handles=handles)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"成功保存图像到 '{output_path}'")

# --- 脚本执行入口 ---
if __name__ == "__main__":
    # --- 用户需要修改的部分 ---

    # 1. namelist.wps 文件路径
    namelist_file = '/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/WPS-4.1/namelist.wps.mangkhut.fixed_d02'
    
    # 2. 从第一个脚本生成的 ensemble_tracks.csv 文件路径
    tracks_file = '/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst/ensemble_tracks.csv'

    # 3. 输出图像的路径
    output_figure_path = '/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/figs/domain_with_all_tracks.png'

    # 4. NR 台风路径的经度文件 (请替换为您的真实路径)
    nr_lon_path = '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/NR_d01_lon.txt' 

    # 5. NR 台风路径的纬度文件 (请替换为您的真实路径)
    nr_lat_path = '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/NR_d01_lat.txt'
    
    # 6. b-deck 最佳路径文件 (例如 bwp262018.dat, 请替换为您的真实路径)
    bdeck_file = '/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/JTWCmangkhutBST.dat'
    
    # --- 执行绘图 ---
    plot_domains_and_tracks(
        namelist_file, 
        tracks_file, 
        output_figure_path,
        nr_lon_path,
        nr_lat_path,
        bdeck_file
    )