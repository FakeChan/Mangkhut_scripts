import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from wrf import getvar
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap

def get_atmos_spread_cmap():
    """
    大气科学中常见的非负变量色标：
    white-blue-green-yellow-red
    适合 spread、RMSE、降水、误差幅度等非负变量。
    """
    colors = [
        "#ffffff",  # white
        "#d7f0ff",  # very light blue
        "#74add1",  # blue
        "#4575b4",  # deep blue
        "#1a9850",  # green
        "#91cf60",  # light green
        "#ffffbf",  # yellow
        "#fdae61",  # orange
        "#f46d43",  # red-orange
        "#d73027",  # red
        "#7f0000",  # dark red
    ]

    return LinearSegmentedColormap.from_list(
        "atmos_spread_cmap",
        colors,
        N=256
    )
# ==========================================
# 空间计算基础函数
# ==========================================
def calc_spatial_rmse(member_field, truth_field):
    """
    计算单个成员相对于真值的空间均方根误差 RMSE
    """
    diff = member_field - truth_field
    return np.sqrt(np.nanmean(diff ** 2))


def calc_ensemble_spread(members_fields):
    """
    计算逐格点集合离散度 Spread

    members_fields: shape = (M, Ny, Nx)
    返回: shape = (Ny, Nx)

    注意：
    这里返回的是集合标准差，而不是集合方差。
    """
    variance_field = np.nanvar(members_fields, axis=0, ddof=1)
    spread_field = np.sqrt(variance_field)
    return spread_field


# ==========================================
# 绘图函数：直接按照数组网格绘制方正 contourf
# ==========================================
def plot_spread_contourf_grid(
    df_spread,
    exp_names,
    target_var,
    output_dir,
    cmap="viridis",
    nlevels=21,
    use_percentile=False,
    percentile=99,
    filter_kind=''
):
    """
    绘制半小时一次的集合离散度 contourf 图。

    特点：
    1. 不使用 XLAT / XLONG
    2. 不使用 Lambert 投影
    3. 直接按照二维数组的 i-j 网格绘图
    4. 每个格点显示为方正网格
    5. 所有子图共用统一 colorbar

    行：实验
    列：时间
    """

    if df_spread.empty:
        raise RuntimeError("df_spread 为空，请检查成员文件路径、变量名、时间范围或 filter_kind。")

    # 所有实验、所有时间统一色标范围
    all_values = np.concatenate([
        np.asarray(item).ravel()
        for item in df_spread["Spread"].values
    ])

    all_values = all_values[np.isfinite(all_values)]

    if all_values.size == 0:
        raise RuntimeError("所有 Spread 值均为 NaN 或无效值，无法确定 colorbar 范围。")

    vmin = 0.0

    if use_percentile:
        vmax = np.nanpercentile(all_values, percentile)
    else:
        vmax = np.nanmax(all_values)

    if vmax <= vmin:
        raise RuntimeError(f"colorbar 范围异常：vmin={vmin}, vmax={vmax}")

    if target_var == 'TSK':
        vmax=1.5
    levels = np.linspace(vmin, vmax, nlevels)

    time_list = sorted(df_spread["Time_Obj"].drop_duplicates().tolist())

    nrows = len(exp_names)
    ncols = len(time_list)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.2 * ncols, 3.2 * nrows),
        constrained_layout=True
    )

    # 保证 axes 始终是二维数组
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    cf = None

    for i, exp_name in enumerate(exp_names):
        for j, time_obj in enumerate(time_list):

            ax = axes[i, j]

            sub = df_spread[
                (df_spread["Experiment"] == exp_name) &
                (df_spread["Time_Obj"] == time_obj)
            ]

            if sub.empty:
                ax.set_visible(False)
                continue

            spread_field = np.asarray(sub.iloc[0]["Spread"])

            ny, nx = spread_field.shape

            # 直接构造数组下标网格
            # x 对应 WRF west_east 方向
            # y 对应 WRF south_north 方向
            x = np.arange(nx)
            y = np.arange(ny)
            xx, yy = np.meshgrid(x, y)
            atm_cmap = get_atmos_spread_cmap()
            cf = ax.contourf(
                xx,
                yy,
                spread_field,
                levels=levels,
                cmap=atm_cmap,
                extend="max"
            )

            # 第一行写时间
            if i == 0:
                ax.set_title(time_obj.strftime("%m-%d %H:%M"), fontsize=10)

            # 第一列写实验名
            if j == 0:
                ax.set_ylabel(exp_name, fontsize=11)

            ax.set_xlabel("west_east grid index", fontsize=9)

            # 方正网格关键设置
            ax.set_aspect("equal", adjustable="box")

            # 保证数组第 0 行在图的下方
            ax.set_xlim(0, nx - 1)
            ax.set_ylim(0, ny - 1)

            ax.tick_params(labelsize=8)

    if cf is None:
        raise RuntimeError("没有任何有效子图被绘制，请检查 df_spread 内容。")

    cbar = fig.colorbar(
        cf,
        ax=axes,
        orientation="vertical",
        shrink=0.85,
        pad=0.01
    )

    cbar.set_label(f"{target_var} ensemble spread", fontsize=11)

    fig.suptitle(
        f"{target_var} Ensemble Spread on Model Grid",
        fontsize=16
    )

    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        f"{filter_kind}_{target_var}_ensemble_spread_contourf_model_grid.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">>> 已保存图像: {save_path}")


# ==========================================
# 主程序：批量读取与处理
# ==========================================
if __name__ == "__main__":

    # -------- 实验参数配置 --------
    start_time = datetime(2018, 9, 10, 0, 0)
    end_time   = datetime(2018, 9, 10, 6, 0)
    interval   = timedelta(minutes=30)

    nr_dir = "/scratch/lililei1/kcfu/tc_mangkhut/NR"

    filter_kind = "QCF_RHF"

    exp_dirs = {
        "Exp_NoDA": "/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim0Run0",
        "Exp_DA": "/scratch/lililei1/kcfu/tc_mangkhut/cycle_test/6mem_oceanAssim1Run1",
    }

    member_indices = [6, 15, 29, 37, 43, 44]

    # WRF 中可按需要改成 SST、TSK 等
    target_var = "TSK"

    output_dir = "./figs"
    os.makedirs(output_dir, exist_ok=True)

    # -------- 初始化数据存储 --------
    rmse_records = []
    spread_records = []

    # ==========================================
    # 开始时间循环
    # ==========================================
    curr_time = start_time

    while curr_time <= end_time:

        time_suffix = curr_time.strftime("%Y-%m-%d_%H:%M:%S")
        display_time = curr_time.strftime("%m-%d %H:%M")

        print(f">>> 正在处理时间步: {time_suffix}")

        # --------------------------------------------------
        # 如果后面还想计算 RMSE，可以打开这一段读取 NR 真值
        # --------------------------------------------------
        # nr_filepath = f"{nr_dir}/wrfout_d03_{time_suffix}"
        # try:
        #     with Dataset(nr_filepath) as nc_nr:
        #         nr_field = getvar(nc_nr, target_var, timeidx=0).values
        # except Exception as e:
        #     print(f"  [跳过] 无法读取 NR 文件 {nr_filepath}: {e}")
        #     curr_time += interval
        #     continue

        # --------------------------------------------------
        # 循环处理不同实验
        # --------------------------------------------------
        for exp_name, exp_base_path in exp_dirs.items():

            exp_members_data = []

            for mem_idx in member_indices:

                mem_str = f"{mem_idx:03d}"

                ens_filepath = (
                    f"{exp_base_path}/{filter_kind}/{mem_str}/"
                    f"wrfout_d01_{time_suffix}"
                )

                try:
                    with Dataset(ens_filepath) as nc_ens:

                        ens_field = getvar(
                            nc_ens,
                            target_var,
                            timeidx=0
                        ).values

                        exp_members_data.append(ens_field)

                        # --------------------------------------------------
                        # 如果要计算每个成员相对于 NR 的 RMSE，打开下面代码
                        # --------------------------------------------------
                        # mem_rmse = calc_spatial_rmse(ens_field, nr_field)
                        # rmse_records.append({
                        #     "Time_Obj": curr_time,
                        #     "Time_Str": display_time,
                        #     "Experiment": exp_name,
                        #     "Member": mem_idx,
                        #     "RMSE": mem_rmse
                        # })

                except Exception as e:
                    print(f"  [跳过] 无法读取 {ens_filepath}: {e}")
                    continue

            # --------------------------------------------------
            # 计算该实验、该时间的逐格点集合离散度
            # --------------------------------------------------
            if len(exp_members_data) > 1:

                exp_members_array = np.stack(exp_members_data, axis=0)

                exp_spread = calc_ensemble_spread(exp_members_array)

                spread_records.append({
                    "Time_Obj": curr_time,
                    "Time_Str": display_time,
                    "Experiment": exp_name,
                    "Spread": exp_spread,
                    "N_members": len(exp_members_data)
                })

                print(
                    f"  {exp_name}: 成功读取 {len(exp_members_data)} 个成员，"
                    f"Spread min={np.nanmin(exp_spread):.4f}, "
                    f"max={np.nanmax(exp_spread):.4f}"
                )

            else:
                print(
                    f"  [跳过] {exp_name} 在 {time_suffix} "
                    f"有效成员数不足，无法计算 spread。"
                )

        curr_time += interval

    # ==========================================
    # 转为 DataFrame
    # ==========================================
    df_rmse = pd.DataFrame(rmse_records)
    df_spread = pd.DataFrame(spread_records)

    # 可选：保存 spread 记录信息，不保存二维场本身
    if not df_spread.empty:
        df_info = df_spread[[
            "Time_Obj",
            "Time_Str",
            "Experiment",
            "N_members"
        ]].copy()

        csv_path = os.path.join(output_dir, f"{target_var}_spread_record_info.csv")
        df_info.to_csv(csv_path, index=False)
        print(f">>> 已保存记录信息: {csv_path}")

    # ==========================================
    # 绘制统一 colorbar 的方正网格 contourf 图
    # ==========================================
    plot_spread_contourf_grid(
        df_spread=df_spread,
        exp_names=list(exp_dirs.keys()),
        target_var=target_var,
        output_dir=output_dir,
        cmap="viridis",
        nlevels=21,
        use_percentile=False,
        filter_kind=filter_kind
    )