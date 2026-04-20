import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

def parse_namelist(namelist_text):
    """提取 namelist.wps 中的参数，支持多重嵌套提取"""
    
    # 定义一个辅助函数，用来读取类似 "1, 5, 25" 的列表形式参数
    def get_list(pattern, text, dtype=int):
        match = re.search(pattern, text)
        if match:
            return [dtype(x.strip()) for x in match.group(1).split(',')]
        return []

    # 提取数组参数
    e_we = get_list(r'e_we\s*=\s*([\d,\s]+)', namelist_text, int)
    e_sn = get_list(r'e_sn\s*=\s*([\d,\s]+)', namelist_text, int)
    parent_grid_ratio = get_list(r'parent_grid_ratio\s*=\s*([\d,\s]+)', namelist_text, int)
    i_parent_start = get_list(r'i_parent_start\s*=\s*([\d,\s]+)', namelist_text, int)
    j_parent_start = get_list(r'j_parent_start\s*=\s*([\d,\s]+)', namelist_text, int)
    
    # 提取单一参数
    dx = float(re.search(r'dx\s*=\s*([\d\.]+)', namelist_text).group(1))
    dy = float(re.search(r'dy\s*=\s*([\d\.]+)', namelist_text).group(1))
    ref_lat = float(re.search(r'ref_lat\s*=\s*([\d\.\-]+)', namelist_text).group(1))
    ref_lon = float(re.search(r'ref_lon\s*=\s*([\d\.\-]+)', namelist_text).group(1))
    truelat1 = float(re.search(r'truelat1\s*=\s*([\d\.\-]+)', namelist_text).group(1))
    truelat2 = float(re.search(r'truelat2\s*=\s*([\d\.\-]+)', namelist_text).group(1))
    
    return {
        'e_we': e_we, 'e_sn': e_sn, 'parent_grid_ratio': parent_grid_ratio,
        'i_parent_start': i_parent_start, 'j_parent_start': j_parent_start,
        'dx': dx, 'dy': dy, 'ref_lat': ref_lat, 'ref_lon': ref_lon,
        'truelat1': truelat1, 'truelat2': truelat2
    }

def plot_wrf_domains(namelist_text):
    # 1. 解析参数
    nl = parse_namelist(namelist_text)
    num_domains = len(nl['e_we'])
    
    # 2. 定义投影 (Lambert)
    wrf_proj = ccrs.LambertConformal(
        central_longitude=nl['ref_lon'],
        central_latitude=nl['ref_lat'],
        standard_parallels=(nl['truelat1'], nl['truelat2'])
    )
    
    # 3. 初始化画布
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=wrf_proj)
    
    # --- 核心计算：推导所有 Domain 的物理边界 ---
    
    # 对于 D01，投影中心即为地理中心 (0, 0)
    width_d01 = (nl['e_we'][0] - 1) * nl['dx']
    height_d01 = (nl['e_sn'][0] - 1) * nl['dy']
    
    # D01 的左下角坐标 (在 Lambert 投影米制坐标系下)
    x_ll_d01 = -width_d01 / 2.0
    y_ll_d01 = -height_d01 / 2.0
    
    # 设置显示范围稍微比 D01 大一点，留白好看
    margin = 150000 # 150 km
    ax.set_extent([x_ll_d01 - margin, -x_ll_d01 + margin, 
                   y_ll_d01 - margin, -y_ll_d01 + margin], crs=wrf_proj)
    
    # 添加地理要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=2)
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue', zorder=1)
    
    # 绘制经纬度网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, 
                      linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
    gl.x_inline = False; gl.y_inline = False
    gl.top_labels = True; gl.bottom_labels = False
    gl.left_labels = True; gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.MultipleLocator(10)
    gl.ylocator = mticker.MultipleLocator(10)
    gl.xlabel_style = {'rotation': 0, 'size': 10}

    # ==========================================
    # 循环绘制 D01, D02 等所有方框
    # ==========================================
    # 储存各个嵌套网格相对于父网格的实际 dx, dy
    current_dx = nl['dx']
    current_dy = nl['dy']
    
    # 分配不同 domain 的框线颜色，d01黑，d02红，d03蓝等
    colors = ['black', 'red', 'blue', 'green']
    
    for i in range(num_domains):
        domain_id = i + 1
        
        if i == 0:
            # D01 直接画
            rect_x = x_ll_d01
            rect_y = y_ll_d01
            rect_w = width_d01
            rect_h = height_d01
        else:
            # D02+ 的分辨率要除以 parent_grid_ratio
            current_dx = current_dx / nl['parent_grid_ratio'][i]
            current_dy = current_dy / nl['parent_grid_ratio'][i]
            
            # 计算左下角偏移量: (i_parent_start - 1) * 上一层网格的 dx
            parent_dx = current_dx * nl['parent_grid_ratio'][i]
            parent_dy = current_dy * nl['parent_grid_ratio'][i]
            
            rect_x = rect_x + (nl['i_parent_start'][i] - 1) * parent_dx
            rect_y = rect_y + (nl['j_parent_start'][i] - 1) * parent_dy
            
            # 计算嵌套网格的长宽
            rect_w = (nl['e_we'][i] - 1) * current_dx
            rect_h = (nl['e_sn'][i] - 1) * current_dy
        
        # 绘制方形补丁 (Patch)
        rect = mpatches.Rectangle(
            (rect_x, rect_y), rect_w, rect_h,
            linewidth=2.5, edgecolor=colors[i % len(colors)], 
            facecolor='none', transform=wrf_proj, zorder=3
        )
        ax.add_patch(rect)
        
        # 添加文本标签 (D01, D02...) 放在网格左下角偏内侧
        label_offset_x = rect_w * 0.02
        label_offset_y = rect_h * 0.02
        ax.text(rect_x + label_offset_x, rect_y + label_offset_y, f'D0{domain_id}',
                color=colors[i % len(colors)], fontsize=14, fontweight='bold', 
                transform=wrf_proj, zorder=4,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 添加中心点标记
    ax.plot(nl['ref_lon'], nl['ref_lat'], marker='*', color='red', markersize=12, 
            transform=ccrs.PlateCarree(), label='Domain Center', zorder=5)
    
    plt.title(f"ENS D01 & D02 domain", fontsize=14, pad=25)
    plt.legend(loc='lower left')
    
    plt.savefig('./figs/ENS_nested_domains.png', dpi=300, bbox_inches='tight')
    plt.show()

namelist_content = """
&geogrid
parent_id            = 1, 1
 parent_grid_ratio    = 1, 5
 i_parent_start       = 1, 23 
 j_parent_start       = 1, 123
 e_we                 = 401, 1001
 e_sn                 = 401, 801
 geog_data_res        = 'default', 'default'
 dx                   = 7500
 dy                   = 7500
 map_proj             = 'lambert'
 ref_lat              = 14.227
 ref_lon              = 147.471
 truelat1             = 15.254
 truelat2             = 15.254
 pole_lat             = 90
 pole_lon             = 0
 stand_lon            = 144.472
 geog_data_path       = '/share/home/lililei1/kcfu/data_repository/V37/'
 opt_geogrid_tbl_path = './geogrid/'
/
"""

# 执行绘制
plot_wrf_domains(namelist_content)