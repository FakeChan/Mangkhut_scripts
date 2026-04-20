import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# --- 用户设置 ---
# 请将这里的文件名替换为您要分析的文件
WRF_INPUT_FILE = 'wrfinput_this'
WRF_BDY_FILE = 'wrfbdy_this'
# --- 结束设置 ---


def compare_wrfinput_vs_wrfbdy(input_file, bdy_file):
    """
    1. 计算 wrfinput 中 (PH+PHB)*(MU+MUB) 的北边界廓面。
    2. 直接读取 wrfbdy 中 PH_BYE 的廓面。
    3. 绘制并排对比图。
    """
    print(f"--- 开始验证对比 ---")

    # =========================================================================
    #  部分 1: 处理 wrfinput 文件 (与之前类似)
    # =========================================================================
    print(f"\n[部分 1] 正在处理 wrfinput 文件: {input_file}")
    try:
        ds_input = xr.open_dataset(input_file)
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到。")
        return

    try:
        ph_pert = ds_input['PH'].isel(Time=0)
        ph_base = ds_input['PHB'].isel(Time=0)
        mu_pert = ds_input['MU'].isel(Time=0)
        mu_base = ds_input['MUB'].isel(Time=0)
    except KeyError as e:
        print(f"错误: wrfinput 文件中缺少变量: {e}。")
        return

    total_ph_north = (ph_pert).isel(south_north=-1)
    total_mu_north = (mu_pert + mu_base).isel(south_north=-1)
    calculated_profile = total_ph_north * total_mu_north
    print("成功从 wrfinput 计算出廓面。")

    # =========================================================================
    #  部分 2: 处理 wrfbdy 文件
    # =========================================================================
    print(f"\n[部分 2] 正在处理 wrfbdy 文件: {bdy_file}")
    try:
        ds_bdy = xr.open_dataset(bdy_file)
    except FileNotFoundError:
        print(f"错误: 文件 '{bdy_file}' 未找到。")
        return

    try:
        # 提取北边界变量，并选择第一个时间步
        ph_bye = ds_bdy['PH_BYE'].isel(Time=0)
        
        # 动态找到边界宽度维度的名称 (通常是 'bdy_width')
        # PH_BYE 维度应为 ('bottom_top', 'bdy_width', 'west_east')
        bdy_dim_name = [d for d in ph_bye.dims if d not in ['bottom_top', 'west_east']][0]
        print(f"检测到边界宽度维度为: '{bdy_dim_name}'")

        # 选择最内层的边界数据 (索引为0)
        direct_profile = ph_bye.isel({bdy_dim_name: 0})
        print(f"成功从 wrfbdy 提取 PH_BYE 廓面 (在 '{bdy_dim_name}'=0 处)。")

    except (KeyError, IndexError) as e:
        print(f"错误: 处理 wrfbdy 文件时出错: {e}。请确认文件包含 'PH_BYE' 变量且维度正确。")
        return

    # =========================================================================
    #  部分 3: 绘制对比图
    # =========================================================================
    print("\n[部分 3] 正在生成对比图...")
    # 创建一个包含两个子图的画布
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 14), sharex=True, sharey=True)
    fig.suptitle('Hypothesis Verification: wrfinput vs wrfbdy', fontsize=20)

    # 为了直接比较，统一两个图的颜色范围
    vmin = min(calculated_profile.min(), direct_profile.min())
    vmax = max(calculated_profile.max(), direct_profile.max())
    print(f"统一颜色范围: vmin={vmin:.2f}, vmax={vmax:.2f}")

    # --- 绘制第一个子图: 来自 wrfinput 的计算结果 ---
    ax1 = axes[0]
    west_east_coords = calculated_profile.coords['west_east'].values
    num_vertical_levels = calculated_profile.shape[0]
    vertical_axis = np.arange(num_vertical_levels)
    
    contour1 = ax1.pcolormesh(west_east_coords, vertical_axis, calculated_profile.values,
                              cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax1.set_title('Calculated from wrfinput: (PH+PHB)*(MU+MUB) at Northern Boundary', fontsize=14)
    ax1.set_ylabel('Model Vertical Level Index', fontsize=12)
    fig.colorbar(contour1, ax=ax1, orientation='vertical', label='Value')

    # --- 绘制第二个子图: 直接从 wrfbdy 读取的结果 ---
    ax2 = axes[1]
    west_east_coords_bdy = direct_profile.coords['west_east'].values
    num_vertical_levels_bdy = direct_profile.shape[0]
    vertical_axis_bdy = np.arange(num_vertical_levels_bdy)

    contour2 = ax2.pcolormesh(west_east_coords_bdy, vertical_axis_bdy, direct_profile.values,
                              cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax2.set_title('Directly Read from wrfbdy: PH_BYE (Innermost Boundary Layer)', fontsize=14)
    ax2.set_xlabel('West-East Grid Point Index', fontsize=12)
    ax2.set_ylabel('Model Vertical Level Index', fontsize=12)
    fig.colorbar(contour2, ax=ax2, orientation='vertical', label='Value')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应主标题

    # 保存图像
    output_image_file = 'comparison_profile.png'
    plt.savefig(output_image_file)
    print(f"\n成功保存对比图为: {output_image_file}")


if __name__ == '__main__':
    compare_wrfinput_vs_wrfbdy(WRF_INPUT_FILE, WRF_BDY_FILE)