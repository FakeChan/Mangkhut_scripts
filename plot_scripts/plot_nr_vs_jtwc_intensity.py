#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the NR (nature run) intensity of Typhoon Mangkhut (2018) against the
JTWC best track.

Intensity is described by two quantities:
    - minimum surface pressure (hPa)
    - maximum 10-m wind speed (kt)

NR data
    /share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain/wrfout_d01_*
    The d01 series covers 2018-09-09 00Z .. 2018-09-10 09Z (33 h): 3-hourly
    up to 2018-09-10 00Z, then 30-minutely.

    NOTE ON PRESSURE: the NR wrfout files contain PSFC but no SLP variable.
    Over the open ocean the TC centre is at sea level, so the minimum PSFC is
    used here as the centre pressure and differs from true MSLP by well under
    1 hPa. The panel is therefore labelled "surface pressure".

JTWC data
    /share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/JTWCmangkhutBST.dat
    ATCF b-deck records, 6-hourly, 2018-09-06 18Z .. 2018-09-17 12Z. Several
    rows share a timestamp (one per wind-radii entry); only the first row of
    each timestamp is kept.

Figure layout: two panels sharing the x axis, restricted to the NR window.
    top    : minimum surface pressure (hPa), y axis inverted so that a rising
             curve means intensification
    bottom : maximum 10-m wind speed (kt), with a secondary m/s axis

All parameters are configured explicitly in this file. No command-line
arguments are used.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

# 计算节点上 matplotlib 缓存目录（避免 HOME 配额/权限问题）
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd


# =====================
# User configuration
# =====================
# --- 路径（以 lililei1 服务器为准） ---
NR_DIR = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain")
NR_DOMAIN = "d01"                       # d01 = 7.5 km, 全程 33 h
JTWC_FILE = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/JTWCmangkhutBST.dat")
OUT_DIR = Path("/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/figs")
OUT_STEM = "tc_intensity_nr_vs_jtwc"

# --- 时间窗（None = 自动取 NR 文件的最早/最晚时刻，即只画 NR 覆盖的 33 h） ---
PLOT_START = None
PLOT_END = None

# --- 极值搜索方式 ---
#   "domain"   : 在全网格上取最小气压 / 最大风速（默认；d01 内山竹是绝对主导系统）
#   "jtwc_box" : 在 JTWC 插值中心 +-SEARCH_HALF_WIDTH_DEG 的方框内取极值
SEARCH_MODE = "domain"
SEARCH_HALF_WIDTH_DEG = 8.0

# --- 缓存（NR 抽取一次约 1-3 分钟，重绘时直接复用 CSV） ---
USE_CACHE = True

# --- 单位换算 ---
PA_TO_HPA = 0.01
MS_TO_KT = 1.9438444924406

# --- 绘图样式 ---
FIG_SIZE = (9.0, 7.0)
FIG_DPI = 300
FONT_SIZE = 12
NR_COLOR = "#D0002E"        # 与 param.py 的 kc_red 一致
JTWC_COLOR = "#111111"
NR_LINEWIDTH = 1.8
JTWC_LINEWIDTH = 1.4
JTWC_MARKERSIZE = 5.0


# =====================
# Functions
# =====================
# ATCF b-deck 的纬度是"十分之一度 + N/S"，例如 120N = 12.0N；
# 经度同理，例如 1693E = 169.3E。
LAT_RE = re.compile(r"^\s*(\d{2,3})([NS])\s*$")
LON_RE = re.compile(r"^\s*(\d{3,5})([EW])\s*$")
BDECK_TIME_RE = re.compile(r"^\d{10}$")
WRFOUT_TIME_RE = re.compile(
    r"wrfout_(d\d\d)_(\d{4})-(\d{2})-(\d{2})_(\d{2}):(\d{2}):(\d{2})"
)


def parse_latlon(token):
    """把 b-deck 的 120N / 1693E 之类的字段解析成十进制度，失败返回 None。"""
    token = token.strip()
    match = LAT_RE.match(token)
    if match:
        value = float(match.group(1)) / 10.0
        return value if match.group(2) == "N" else -value
    match = LON_RE.match(token)
    if match:
        value = float(match.group(1)) / 10.0
        return value if match.group(2) == "E" else -value
    return None


def read_bdeck(path):
    """
    读取 ATCF b-deck，返回按时间排序的 DataFrame。

    列：time, lat, lon, vmax_kt, mslp_hpa

    同一个时次在 b-deck 里可能有多行（每行一个风圈半径条目），这些行的
    中心位置/最大风/中心气压完全一致，因此每个时次只保留第一条。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JTWC b-deck 文件不存在: {path}")

    records = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 10:
                continue
            if not BDECK_TIME_RE.match(fields[2]):
                continue

            # 用正则定位纬度字段，避免不同 b-deck 变体多/少一列时整体错位。
            # 纬度之后依次是 经度、VMAX、MSLP。
            lat_idx = next(
                (i for i, item in enumerate(fields) if LAT_RE.match(item)), None
            )
            if lat_idx is None or lat_idx + 3 >= len(fields):
                continue

            lat = parse_latlon(fields[lat_idx])
            lon = parse_latlon(fields[lat_idx + 1])
            if lat is None or lon is None:
                continue

            try:
                vmax_kt = float(fields[lat_idx + 2])
            except ValueError:
                vmax_kt = np.nan
            try:
                mslp_hpa = float(fields[lat_idx + 3])
            except ValueError:
                mslp_hpa = np.nan

            try:
                stamp = datetime.strptime(fields[2], "%Y%m%d%H")
            except ValueError:
                continue

            if stamp in records:
                continue
            records[stamp] = {
                "time": stamp,
                "lat": lat,
                "lon": lon,
                "vmax_kt": vmax_kt,
                "mslp_hpa": mslp_hpa,
            }

    if not records:
        raise ValueError(f"没有从 {path} 解析出任何 b-deck 记录")

    frame = pd.DataFrame(list(records.values()))
    return frame.sort_values("time").reset_index(drop=True)


def list_nr_files(nr_dir, domain):
    """列出指定 domain 的 NR wrfout 文件，按文件名里的时间戳排序。"""
    nr_dir = Path(nr_dir)
    if not nr_dir.is_dir():
        raise FileNotFoundError(f"NR 目录不存在: {nr_dir}")

    files = []
    for path in sorted(nr_dir.glob(f"wrfout_{domain}_*")):
        match = WRFOUT_TIME_RE.search(path.name)
        if not match:
            continue
        stamp = datetime(
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
            int(match.group(6)),
            int(match.group(7)),
        )
        files.append((stamp, path))

    if not files:
        raise FileNotFoundError(f"{nr_dir} 下没有找到 wrfout_{domain}_* 文件")

    files.sort(key=lambda item: item[0])
    return files


def _read_2d(dataset, name):
    """读取 (Time, south_north, west_east) 变量的第一个时次，填成 float64。"""
    if name not in dataset.variables:
        raise KeyError(f"wrfout 中缺少变量 {name}")
    raw = dataset.variables[name][0, :, :]
    return np.ma.filled(np.ma.asarray(raw, dtype=np.float64), np.nan)


def nr_intensity(path, search_mode="domain", jtwc_center=None, half_width_deg=8.0):
    """
    从一个 NR wrfout 中提取台风强度。

    返回 dict：time 由调用方补充，这里给出
        mslp_hpa   最小地面气压（hPa）
        vmax_kt    最大 10 m 风速（kt）
        center_lat / center_lon  最小气压所在位置
    """
    with nc.Dataset(str(path)) as dataset:
        psfc = _read_2d(dataset, "PSFC")
        u10 = _read_2d(dataset, "U10")
        v10 = _read_2d(dataset, "V10")
        lats = _read_2d(dataset, "XLAT")
        lons = _read_2d(dataset, "XLONG")

    if psfc.shape != lats.shape or psfc.shape != lons.shape:
        raise ValueError(f"{path}: PSFC 与 XLAT/XLONG 形状不一致")

    # 10 m 风速：网格相对风与地球相对风只差一个旋转，模长不变，
    # 所以直接对 U10/V10 求模即可，无需 uvmet10 旋转。
    wind_speed = np.hypot(u10, v10)

    mask = np.ones(psfc.shape, dtype=bool)
    if search_mode == "jtwc_box" and jtwc_center is not None:
        center_lat, center_lon = jtwc_center
        mask &= np.abs(lats - center_lat) <= half_width_deg
        mask &= np.abs(lons - center_lon) <= half_width_deg
        if not np.any(mask & np.isfinite(psfc)):
            mask = np.ones(psfc.shape, dtype=bool)   # 框内无有效值，退回全域

    pressure = np.where(mask, psfc, np.nan)
    speed = np.where(mask, wind_speed, np.nan)

    if not np.any(np.isfinite(pressure)):
        raise ValueError(f"{path}: 搜索范围内没有有效的 PSFC 值")

    center_idx = np.unravel_index(np.nanargmin(pressure), pressure.shape)

    return {
        "mslp_hpa": float(pressure[center_idx]) * PA_TO_HPA,
        "vmax_kt": float(np.nanmax(speed)) * MS_TO_KT,
        "center_lat": float(lats[center_idx]),
        "center_lon": float(lons[center_idx]),
    }


def extract_nr_series(files, search_mode="domain", jtwc_frame=None,
                      half_width_deg=8.0):
    """遍历 NR 文件，抽取强度时间序列。"""
    rows = []
    total = len(files)
    for index, (stamp, path) in enumerate(files, start=1):
        jtwc_center = None
        if search_mode == "jtwc_box" and jtwc_frame is not None:
            jtwc_center = _jtwc_center_at(jtwc_frame, stamp)

        print(f"[{index}/{total}] {path.name}", flush=True)
        try:
            values = nr_intensity(
                path,
                search_mode=search_mode,
                jtwc_center=jtwc_center,
                half_width_deg=half_width_deg,
            )
        except Exception as error:                      # noqa: BLE001
            print(f"    跳过（读取失败）: {error}", flush=True)
            continue

        values["time"] = stamp
        rows.append(values)

    if not rows:
        raise ValueError("所有 NR 文件都读取失败，没有可用数据")

    frame = pd.DataFrame(rows)
    return frame.sort_values("time").reset_index(drop=True)


def _jtwc_center_at(jtwc_frame, stamp):
    """把 JTWC 位置线性插值到给定时刻，用于 jtwc_box 搜索模式。"""
    if jtwc_frame is None or jtwc_frame.empty:
        return None
    times = jtwc_frame["time"].to_numpy()
    target = np.datetime64(stamp)
    if target < times[0] or target > times[-1]:
        return None
    lat = np.interp(target.astype("datetime64[s]").astype(float),
                    times.astype("datetime64[s]").astype(float),
                    jtwc_frame["lat"].to_numpy())
    lon = np.interp(target.astype("datetime64[s]").astype(float),
                    times.astype("datetime64[s]").astype(float),
                    jtwc_frame["lon"].to_numpy())
    return float(lat), float(lon)


def load_or_build_nr_series(files, cache_path, search_mode, jtwc_frame,
                            half_width_deg, use_cache):
    """带 CSV 缓存的 NR 序列抽取。"""
    cache_path = Path(cache_path)
    if use_cache and cache_path.is_file():
        frame = pd.read_csv(cache_path, parse_dates=["time"])
        if len(frame) == len(files):
            print(f"命中缓存: {cache_path}（{len(frame)} 个时次）", flush=True)
            return frame
        print(
            f"缓存时次数 {len(frame)} 与文件数 {len(files)} 不一致，重新计算",
            flush=True,
        )

    frame = extract_nr_series(
        files,
        search_mode=search_mode,
        jtwc_frame=jtwc_frame,
        half_width_deg=half_width_deg,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    print(f"已写入缓存: {cache_path}", flush=True)
    return frame


def plot_intensity(nr_frame, jtwc_frame, out_png, plot_start, plot_end):
    """画上下两栏的强度对比图。"""
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE + 1,
            "axes.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE - 1,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "axes.linewidth": 0.8,
        }
    )

    figure, (ax_pressure, ax_wind) = plt.subplots(
        2,
        1,
        figsize=FIG_SIZE,
        sharex=True,
        dpi=FIG_DPI,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.10},
    )

    jtwc_label = "JTWC best track (6-hourly)"
    nr_label = f"NR {NR_DOMAIN.upper()} (nature run)"

    # ---------- 上：中心气压 ----------
    ax_pressure.plot(
        jtwc_frame["time"],
        jtwc_frame["mslp_hpa"],
        color=JTWC_COLOR,
        linestyle="--",
        linewidth=JTWC_LINEWIDTH,
        marker="o",
        markersize=JTWC_MARKERSIZE,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label=jtwc_label,
        zorder=3,
    )
    ax_pressure.plot(
        nr_frame["time"],
        nr_frame["mslp_hpa"],
        color=NR_COLOR,
        linewidth=NR_LINEWIDTH,
        label=nr_label,
        zorder=4,
    )
    ax_pressure.set_ylabel("Minimum surface pressure (hPa)")
    ax_pressure.invert_yaxis()          # 气压下降 = 增强 = 曲线向上
    ax_pressure.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    # ---------- 下：最大 10 m 风速 ----------
    ax_wind.plot(
        jtwc_frame["time"],
        jtwc_frame["vmax_kt"],
        color=JTWC_COLOR,
        linestyle="--",
        linewidth=JTWC_LINEWIDTH,
        marker="o",
        markersize=JTWC_MARKERSIZE,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label=jtwc_label,
        zorder=3,
    )
    ax_wind.plot(
        nr_frame["time"],
        nr_frame["vmax_kt"],
        color=NR_COLOR,
        linewidth=NR_LINEWIDTH,
        label=nr_label,
        zorder=4,
    )
    ax_wind.set_ylabel("Maximum 10-m wind speed (kt)")
    ax_wind.set_xlabel("Time (UTC)")
    ax_wind.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    # 右侧 m/s 副轴
    secondary = ax_wind.secondary_yaxis(
        "right",
        functions=(lambda kt: kt / MS_TO_KT, lambda ms: ms * MS_TO_KT),
    )
    secondary.set_ylabel("Maximum 10-m wind speed (m/s)")

    # ---------- 峰值标注 ----------
    stats = []
    if not nr_frame.empty:
        nr_pressure = nr_frame.loc[nr_frame["mslp_hpa"].idxmin()]
        nr_wind = nr_frame.loc[nr_frame["vmax_kt"].idxmax()]
        stats.append(
            f"NR  min {nr_pressure['mslp_hpa']:.1f} hPa "
            f"@ {nr_pressure['time']:%m-%d %HZ}"
        )
        stats.append(
            f"NR  max {nr_wind['vmax_kt']:.0f} kt "
            f"@ {nr_wind['time']:%m-%d %HZ}"
        )
    if not jtwc_frame.empty:
        jtwc_pressure = jtwc_frame.loc[jtwc_frame["mslp_hpa"].idxmin()]
        jtwc_wind = jtwc_frame.loc[jtwc_frame["vmax_kt"].idxmax()]
        stats.append(
            f"JTWC min {jtwc_pressure['mslp_hpa']:.0f} hPa "
            f"@ {jtwc_pressure['time']:%m-%d %HZ}"
        )
        stats.append(
            f"JTWC max {jtwc_wind['vmax_kt']:.0f} kt "
            f"@ {jtwc_wind['time']:%m-%d %HZ}"
        )

    if stats:
        ax_pressure.text(
            0.015,
            0.04,
            "\n".join(stats[:2]),
            transform=ax_pressure.transAxes,
            va="bottom",
            ha="left",
            fontsize=FONT_SIZE - 2,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "0.7", "alpha": 0.85},
        )
        ax_wind.text(
            0.015,
            0.96,
            "\n".join(stats[2:]),
            transform=ax_wind.transAxes,
            va="top",
            ha="left",
            fontsize=FONT_SIZE - 2,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "0.7", "alpha": 0.85},
        )

    # ---------- 坐标轴 ----------
    ax_wind.set_xlim(plot_start, plot_end)
    ax_wind.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax_wind.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%HZ"))
    ax_wind.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 3)))

    handles, labels = ax_pressure.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.suptitle(
        "Typhoon Mangkhut (2018) intensity: NR vs JTWC best track",
        fontsize=FONT_SIZE + 2,
        y=0.955,
    )

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"已保存图像: {out_png}", flush=True)


def main():
    print("=" * 68)
    print("NR vs JTWC intensity for Typhoon Mangkhut (2018)")
    print(f"  NR dir   : {NR_DIR}")
    print(f"  NR domain: {NR_DOMAIN}")
    print(f"  JTWC file: {JTWC_FILE}")
    print(f"  out dir  : {OUT_DIR}")
    print("=" * 68)

    jtwc_frame = read_bdeck(JTWC_FILE)
    print(
        f"JTWC: {len(jtwc_frame)} 个时次，"
        f"{jtwc_frame['time'].min():%Y-%m-%d %HZ} .. "
        f"{jtwc_frame['time'].max():%Y-%m-%d %HZ}",
        flush=True,
    )

    nr_files = list_nr_files(NR_DIR, NR_DOMAIN)
    print(
        f"NR: {len(nr_files)} 个文件，"
        f"{nr_files[0][0]:%Y-%m-%d %HZ} .. {nr_files[-1][0]:%Y-%m-%d %HZ}",
        flush=True,
    )

    cache_path = OUT_DIR / f"{OUT_STEM}_nr_{NR_DOMAIN}_cache.csv"
    nr_frame = load_or_build_nr_series(
        nr_files,
        cache_path,
        search_mode=SEARCH_MODE,
        jtwc_frame=jtwc_frame,
        half_width_deg=SEARCH_HALF_WIDTH_DEG,
        use_cache=USE_CACHE,
    )

    # x 轴只画 NR 覆盖的时间窗，JTWC 曲线相应裁剪
    plot_start = pd.Timestamp(PLOT_START) if PLOT_START else nr_frame["time"].min()
    plot_end = pd.Timestamp(PLOT_END) if PLOT_END else nr_frame["time"].max()
    jtwc_window = jtwc_frame[
        (jtwc_frame["time"] >= plot_start) & (jtwc_frame["time"] <= plot_end)
    ].reset_index(drop=True)

    print(
        f"绘图窗口: {plot_start:%Y-%m-%d %HZ} .. {plot_end:%Y-%m-%d %HZ}"
        f"（JTWC 落在窗口内 {len(jtwc_window)} 个时次）",
        flush=True,
    )

    out_png = OUT_DIR / f"{OUT_STEM}.png"
    plot_intensity(nr_frame, jtwc_window, out_png, plot_start, plot_end)

    # 长表 CSV：便于后续复用
    nr_table = nr_frame.copy()
    nr_table.insert(0, "source", f"NR_{NR_DOMAIN}")
    jtwc_table = jtwc_window.copy()
    jtwc_table.insert(0, "source", "JTWC")
    combined = pd.concat([nr_table, jtwc_table], ignore_index=True)
    out_csv = OUT_DIR / f"{OUT_STEM}.csv"
    combined.to_csv(out_csv, index=False)
    print(f"已保存数据表: {out_csv}", flush=True)

    # 控制台摘要
    print("-" * 68)
    print(
        f"NR  : min PSFC {nr_frame['mslp_hpa'].min():.1f} hPa, "
        f"max wind {nr_frame['vmax_kt'].max():.1f} kt"
    )
    if not jtwc_window.empty:
        print(
            f"JTWC: min MSLP {jtwc_window['mslp_hpa'].min():.0f} hPa, "
            f"max wind {jtwc_window['vmax_kt'].max():.0f} kt"
        )
    print("-" * 68)


if __name__ == "__main__":
    main()
