"""TSK-HFX 误差传递诊断绘图(仅消费缓存与表格,不读 wrfout)。

依据 docs/superpowers/plans/2026-10-31-tsk-hfx-error-pathway.md Task 2-4:
  - Task 2 主图(δTSK、ΔSE_D、δH、C;4 行主时次 × 4 列,两方法一份)、
    有符号误差补图、成员差异图与正文案例成员图;
  - Task 3 重建残差图与替换分量图;
  - Task 4 居中/固定双映射误差图、径向廓线与评分柱状图。
  - 每张图输出 PNG(检查版)+ PDF(矢量文字)+ 图注(.txt)+ 图源数据(.npz)。

平均顺序:所有空间平均场均为"先成员内逐点计算,再 6 成员等权平均(仅在
6 成员共同覆盖点上)";成员指标平均见 member_metrics.csv(逐成员 RMSE 后平均)。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import tsk_hfx_pathway_diagnostics as tp

# =====================================================================
# User configuration
# =====================================================================

PLOT_CONFIG: dict = {
    "run_environment": "server",  # 与诊断库一致;local_smoke 只输出表格图(无缓存时跳过)
    "methods": ("EAKF", "QCF_RHF"),
    "main_hours": (0.5, 1.0, 1.5, 2.0),
    # 色标:有符号场(δ、ΔSE、C)以 0 为中心,幅度取各面板共同 |值| 的百分位;
    # 裁剪只影响显示,原始数值不裁剪,裁剪比例写入图注。
    "signed_percentile": 98.0,
    "signed_percentile_floor": 1.0e-12,
    "cmap_signed": "RdBu_r",
    "figure_formats": ("png", "pdf"),
    "figure_dpi": 200,
    "map_fontsize": 8.0,
    "nr_center_marker": "k*",
    "nr_circle_color": "0.25",
}

VARIABLE_LABELS = {
    "delta_tsk": ("δTSK (S−W)", "K"),
    "dse_d": ("ΔSE$_D$ (squared-err. change of D=TSK−T$_{air}$)", "K$^2$"),
    "delta_hfx": ("δHFX (S−W)", "W m$^{-2}$"),
    "cross_h": ("C = 2 e$_{H,W}$·δHFX", "(W m$^{-2}$)$^2$"),
    "e_tsk_w": ("e$_{TSK,W}$ = TSK$_W$−TSK$_{NR}$", "K"),
    "e_tair_w": ("e$_{Tair,W}$", "K"),
    "e_hfx_w": ("e$_{H,W}$ = HFX$_W$−HFX$_{NR}$", "W m$^{-2}$"),
    "e_d_w": ("e$_{D,W}$ (D = TSK−T$_{air}$)", "K"),
    "e_d_s": ("e$_{D,S}$", "K"),
    "e_dtheta_w": ("e$_{Dθ,W}$ (θ-gap error)", "K"),
    "residual_w": ("r$_W$ = F$_{WW}$−HFX$_W$ (recon − model)", "W m$^{-2}$"),
    "residual_s": ("r$_S$ = F$_{SS}$−HFX$_S$ (recon − model)", "W m$^{-2}$"),
    "dF_T": ("dF$_T$ = F$_{SW}$−F$_{WW}$ (SST-only)", "W m$^{-2}$"),
    "dF_rest": ("dF$_{rest}$ = F$_{SS}$−F$_{SW}$", "W m$^{-2}$"),
    "delta_actual": ("actual δHFX (model output)", "W m$^{-2}$"),
    "e_hfx_centered_w": ("e$_{H,W}$ storm-centered", "W m$^{-2}$"),
    "e_hfx_fixed_w": ("e$_{H,W}$ fixed-geolocation", "W m$^{-2}$"),
}


# =====================================================================
# 基础工具
# =====================================================================

def output_root() -> Path:
    tp.CONFIG["run_environment"] = PLOT_CONFIG["run_environment"]
    return Path(tp.paths(tp.CONFIG)["output_root"])


def cache_dir(kind: str) -> Path:
    return output_root() / "cache" / kind


def tables_dir() -> Path:
    return output_root() / "tables"


def figures_dir() -> Path:
    path = output_root() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_dir() -> Path:
    path = output_root() / "source_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _circle_points(center_lat: float, center_lon: float, radius_km: float, n: int = 181):
    """大圆轨迹点(经纬度),用于叠加 75/150 km 圈。"""
    az = np.linspace(0.0, 2.0 * np.pi, n)
    lat1 = np.deg2rad(center_lat)
    lon1 = np.deg2rad(center_lon)
    angular = radius_km / tp.CONFIG["earth_radius_km"]
    lat2 = np.arcsin(np.sin(lat1) * np.cos(angular)
                     + np.cos(lat1) * np.sin(angular) * np.cos(az))
    lon2 = lon1 + np.arctan2(np.sin(az) * np.sin(angular) * np.cos(lat1),
                             np.cos(angular) - np.sin(lat1) * np.sin(lat2))
    return np.rad2deg(lat2), np.rad2deg(lon2)


def _nr_center_of(case: tp.xr.Dataset) -> tuple[float, float]:
    meta = json.loads(case.attrs.get("meta_json", "{}"))
    return float(meta["nr_center"]["lat"]), float(meta["nr_center"]["lon"])


def _load_case(method: str, member: str, hour: float) -> tp.xr.Dataset | None:
    path = cache_dir("case") / f"case_{tp.case_id(method, member, hour)}.nc"
    if not path.exists():
        return None
    return tp.load_case_cache(path)


def _member_mean_common(fields: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """仅在 6 成员(或给定成员)共同覆盖点上等权平均;返回 (mean, coverage)。"""
    stack = np.stack([np.asarray(f, dtype=float) for f in fields], axis=0)
    coverage = np.all(np.isfinite(stack), axis=0)
    mean = np.full(stack.shape[1:], np.nan)
    if np.any(coverage):
        mean[coverage] = stack[:, coverage].mean(axis=0)
    return mean, coverage


def _symmetric_scale(panels: list[np.ndarray], percentile: float) -> tuple[float, float]:
    chunks = [np.abs(np.asarray(p, dtype=float)[np.isfinite(p)]).ravel()
              for p in panels if np.any(np.isfinite(p))]
    if not chunks:
        return -1.0, 1.0
    values = np.concatenate(chunks)
    if values.size == 0:
        return -1.0, 1.0
    amp = float(np.nanpercentile(values, percentile))
    if amp <= PLOT_CONFIG["signed_percentile_floor"]:
        amp = float(np.nanmax(values))
    if amp <= PLOT_CONFIG["signed_percentile_floor"]:
        amp = 1.0
    return -amp, amp


def _clip_stats(values: np.ndarray, vmin: float, vmax: float) -> dict:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        return {"n_finite": 0, "clipped_fraction": np.nan,
                "min": np.nan, "max": np.nan}
    return {
        "n_finite": int(finite.size),
        "clipped_fraction": float(np.mean((finite < vmin) | (finite > vmax))),
        "min": float(finite.min()), "max": float(finite.max()),
    }


def _save_figure(fig, stem: str) -> list[str]:
    saved = []
    for fmt in PLOT_CONFIG["figure_formats"]:
        path = figures_dir() / f"{stem}.{fmt}"
        fig.savefig(path, dpi=PLOT_CONFIG["figure_dpi"], bbox_inches="tight")
        saved.append(str(path))
    plt.close(fig)
    return saved


def _write_caption(stem: str, lines: list[str]) -> str:
    path = figures_dir() / f"{stem}.caption.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _save_source(stem: str, **arrays) -> str:
    path = source_dir() / f"{stem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return str(path)


def _add_center_overlays(ax, case: tp.xr.Dataset, fontsize: float) -> None:
    clat, clon = _nr_center_of(case)
    ax.plot(clon, clat, PLOT_CONFIG["nr_center_marker"], markersize=7,
            transform=ax.transData, zorder=5)
    for radius, style in ((tp.CONFIG["inner_radius_km"], ":"), (tp.CONFIG["radius_km"], "--")):
        lat_c, lon_c = _circle_points(clat, clon, radius)
        ax.plot(lon_c, lat_c, style, color=PLOT_CONFIG["nr_circle_color"],
                linewidth=0.9, zorder=4)


def _panel(ax, lon: np.ndarray, lat: np.ndarray, values: np.ndarray,
           vmin: float, vmax: float, title: str, unit: str,
           case: tp.xr.Dataset | None = None, cmap: str | None = None):
    valid = np.isfinite(values)
    if valid.any():
        ax.pcolormesh(lon, lat, np.where(valid, values, np.nan),
                      cmap=cmap or PLOT_CONFIG["cmap_signed"],
                      vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.nanmean(lat)))))
    else:
        ax.text(0.5, 0.5, "no valid points", transform=ax.transAxes,
                ha="center", va="center", fontsize=PLOT_CONFIG["map_fontsize"])
    if case is not None:
        _add_center_overlays(ax, case, PLOT_CONFIG["map_fontsize"])
    ax.set_title(f"{title}\n[{unit}]", fontsize=PLOT_CONFIG["map_fontsize"], pad=3)
    ax.tick_params(labelsize=PLOT_CONFIG["map_fontsize"] - 1)


# =====================================================================
# Task 2 主图:δTSK / ΔSE_D / δH / C
# =====================================================================

MAIN_COLUMNS = ("delta_tsk", "dse_d", "delta_hfx", "cross_h")
SIGNED_COLUMNS = ("e_tsk_w", "e_tair_w", "e_hfx_w", "e_d_w", "e_d_s", "e_dtheta_w")


def _collect_mean_fields(columns, hours, methods) -> dict[tuple[str, str, float], dict]:
    """(column, method, hour) -> {mean, coverage, case, clip_needed members}。

    平均顺序:先成员内逐点(derive_error_fields),再 6 成员共同覆盖点等权平均。
    """
    result: dict[tuple[str, str, float], dict] = {}
    for method in methods:
        for hour in hours:
            cases = []
            for member in tp.CONFIG["members"]:
                case = _load_case(method, member, hour)
                if case is not None:
                    cases.append(case)
            if not cases:
                continue
            err_fields = {}
            for case in cases:
                err = tp.derive_error_fields(case)
                for column in columns:
                    err_fields.setdefault(column, []).append(
                        np.asarray(err[column].values, dtype=float))
            ref_case = cases[0]
            for column in columns:
                mean, coverage = _member_mean_common(err_fields[column])
                result[(column, method, hour)] = {
                    "mean": mean, "coverage": coverage, "case": ref_case,
                    "lat": np.asarray(ref_case["lat"].values, dtype=float),
                    "lon": np.asarray(ref_case["lon"].values, dtype=float),
                }
    return result


def plot_main_spatial() -> list[str]:
    """主图:每方法一份,4 行主时次 × 4 列;同列跨方法/时次共用色标。"""
    produced = []
    hours = PLOT_CONFIG["main_hours"]
    fields = _collect_mean_fields(MAIN_COLUMNS, hours, PLOT_CONFIG["methods"])
    scales = {}
    for column in MAIN_COLUMNS:
        panels = [v["mean"] for (c, _, _), v in fields.items() if c == column]
        scales[column] = _symmetric_scale(panels, PLOT_CONFIG["signed_percentile"])
    for method in PLOT_CONFIG["methods"]:
        stem = f"task2_main_spatial_{method}"
        fig, axes = plt.subplots(len(hours), len(MAIN_COLUMNS),
                                 figsize=(3.4 * len(MAIN_COLUMNS), 3.0 * len(hours)),
                                 facecolor="white", layout="constrained")
        source_arrays = {"lat": None, "lon": None}
        clip_lines = [f"色标为 ±|值| P{PLOT_CONFIG['signed_percentile']:g}(全方法/主时次共同),"
                      "以 0 为中心;裁剪只影响显示。"]
        coverage_lines = []
        for r, hour in enumerate(hours):
            for c, column in enumerate(MAIN_COLUMNS):
                ax = axes[r, c]
                item = fields.get((column, method, hour))
                title, unit = VARIABLE_LABELS[column]
                if item is None:
                    ax.text(0.5, 0.5, "cache missing", transform=ax.transAxes, ha="center")
                    ax.set_title(f"{title} t={hour:g}h", fontsize=PLOT_CONFIG["map_fontsize"])
                    continue
                vmin, vmax = scales[column]
                stats = _clip_stats(item["mean"], vmin, vmax)
                _panel(ax, item["lon"], item["lat"], item["mean"], vmin, vmax,
                       f"t={hour:g} h  {title}", unit, case=item["case"])
                clip_lines.append(
                    f"t={hour:g}h {column}: 有效点 {stats['n_finite']}, "
                    f"裁剪比例 {stats['clipped_fraction'] * 100:.2f}%, "
                    f"原始范围 [{stats['min']:.4g}, {stats['max']:.4g}] {unit}")
                coverage_lines.append(
                    f"t={hour:g}h {column}: 6 成员共同覆盖 "
                    f"{100.0 * float(np.mean(item['coverage'])):.1f}% 子窗点")
                if source_arrays["lat"] is None:
                    source_arrays["lat"] = item["lat"]
                    source_arrays["lon"] = item["lon"]
                source_arrays[f"{column}_t{hour:g}"] = item["mean"]
                source_arrays[f"coverage_t{hour:g}"] = item["coverage"]
        for c, column in enumerate(MAIN_COLUMNS):
            vmin, vmax = scales[column]
            sm = plt.cm.ScalarMappable(cmap=PLOT_CONFIG["cmap_signed"],
                                       norm=plt.Normalize(vmin, vmax))
            fig.colorbar(sm, ax=axes[:, c].tolist(), shrink=0.85, pad=0.02,
                         label=VARIABLE_LABELS[column][1])
        fig.suptitle(
            f"[{method}] Task2 main: where TSK improves, do D and HFX errors degrade?\n"
            "each panel = 6-member equal-weight mean of member-wise fields (common coverage only);"
            " NR center and 75/150 km circles overlaid",
            fontsize=11)
        produced += _save_figure(fig, stem)
        caption = [
            f"[{method}] Task2 主图:4 行 = 主时次 0.5/1/1.5/2 h,4 列 = δTSK、ΔSE_D、δH、C。",
            "平均顺序:mean_i(2 e_{H,W,i}·δH_i) 等成员内逐点计算后的 6 成员等权平均,"
            "严禁用 2·mean(e)·mean(δ) 替代;仅 6 成员共同覆盖点参与,覆盖率见下。",
            "第 2 列 ΔSE_D 为温差(TSK−T_air)平方误差变化 e_D,S²−e_D,W²(带符号),不是温差误差本身。",
            "叠加:NR 最低 PSFC 中心(星号)、75 km(点线)与 150 km(虚线)圆。",
            *coverage_lines,
            *clip_lines,
            "数据源:cache/case/case_*.nc(member 内逐点场),由 plot_tsk_hfx_pathway.py 生成。",
        ]
        _write_caption(stem, caption)
        _save_source(stem, scales=json.dumps({k: list(v) for k, v in scales.items()}),
                     **source_arrays)
        produced.append(stem)
    return produced


def plot_signed_errors() -> list[str]:
    """有符号误差补图:e_TSK,W、e_Tair,W、e_H,W、e_D,W/S、e_Dθ,W。"""
    produced = []
    hours = PLOT_CONFIG["main_hours"]
    fields = _collect_mean_fields(SIGNED_COLUMNS, hours, PLOT_CONFIG["methods"])
    for method in PLOT_CONFIG["methods"]:
        stem = f"task2_signed_errors_{method}"
        fig, axes = plt.subplots(len(hours), len(SIGNED_COLUMNS),
                                 figsize=(2.9 * len(SIGNED_COLUMNS), 3.0 * len(hours)),
                                 facecolor="white", layout="constrained")
        source_arrays = {}
        clip_lines = []
        for r, hour in enumerate(hours):
            for c, column in enumerate(SIGNED_COLUMNS):
                ax = axes[r, c]
                item = fields.get((column, method, hour))
                title, unit = VARIABLE_LABELS[column]
                if item is None:
                    ax.text(0.5, 0.5, "cache missing", transform=ax.transAxes, ha="center")
                    continue
                vmin, vmax = _symmetric_scale([item["mean"]], PLOT_CONFIG["signed_percentile"])
                stats = _clip_stats(item["mean"], vmin, vmax)
                _panel(ax, item["lon"], item["lat"], item["mean"], vmin, vmax,
                       f"t={hour:g} h  {title}", unit, case=item["case"])
                clip_lines.append(f"t={hour:g}h {column}: 范围 [{stats['min']:.4g}, {stats['max']:.4g}] {unit}")
                source_arrays[f"{column}_t{hour:g}"] = item["mean"]
                source_arrays[f"coverage_t{hour:g}"] = item["coverage"]
                if "lat" not in source_arrays:
                    source_arrays["lat"] = item["lat"]
                    source_arrays["lon"] = item["lon"]
        for c, column in enumerate(SIGNED_COLUMNS):
            panels = [v["mean"] for (col, m, _), v in fields.items()
                      if col == column and m == method]
            vmin, vmax = _symmetric_scale(panels, PLOT_CONFIG["signed_percentile"])
            sm = plt.cm.ScalarMappable(cmap=PLOT_CONFIG["cmap_signed"],
                                       norm=plt.Normalize(vmin, vmax))
            fig.colorbar(sm, ax=axes[:, c].tolist(), shrink=0.85, pad=0.02,
                         label=VARIABLE_LABELS[column][1])
        fig.suptitle(f"[{method}] Task2 signed errors (weak experiment; what sign is the compensation?)",
                     fontsize=11)
        produced += _save_figure(fig, stem)
        _write_caption(stem, [
            f"[{method}] 有符号误差图:e_TSK,W、e_Tair,W、e_H,W、e_D,W、e_D,S、e_Dθ,W(6 成员先算后平均)。",
            "用于判断所谓补偿对应何种实际误差符号;不要只画差值就推断原状态偏暖/偏冷。",
            "每列色标为该方法 4 个主时次共同 ±P98;单位见各面板标题。",
            *clip_lines,
        ])
        _save_source(stem, **source_arrays)
        produced.append(stem)
    return produced


def plot_member_distribution() -> list[str]:
    """成员差异图:逐成员 ΔMSE_H / ΔMSE_TSK 柱状(0.5–2 h 平均)并标注正文案例成员。"""
    produced = []
    metrics_path = tables_dir() / "member_metrics.csv"
    if not metrics_path.exists():
        return produced
    metrics = pd.read_csv(metrics_path)
    sub = metrics[(metrics["mask_kind"] == "ocean_common") & (metrics["region"] == "all")
                  & (metrics["lead_hour"] >= 0.5) & (metrics["lead_hour"] <= 2.0)]
    if sub.empty:
        return produced
    ranking = tp.member_dse_h_ranking(metrics)
    stem = "task2_member_distribution"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor="white")
    width = 0.35
    for ax, var, unit in ((axes[0], "hfx_delta_mse", "(W m$^{-2}$)$^2$"),
                          (axes[1], "tsk_delta_mse", "K$^2$")):
        positions = np.arange(len(PLOT_CONFIG["methods"]) * len(tp.CONFIG["members"]), dtype=float)
        pos = 0
        for mi, method in enumerate(PLOT_CONFIG["methods"]):
            for oi, offset in enumerate((-width / 2, width / 2)):
                values = []
                for member in tp.CONFIG["members"]:
                    hit = sub[(sub["method"] == method) & (sub["member"] == member)]
                    values.append(float(hit[var].mean()) if len(hit) else np.nan)
                ax.bar(positions[pos] + offset, values, width=width,
                       color="#0072B2" if oi == 0 else "#D55E00",
                       label=f"{method}" if pos < 2 else None)
                pos += 1
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{m}\n{mem}" for m in PLOT_CONFIG["methods"]
                            for mem in tp.CONFIG["members"]], fontsize=8)
        ax.set_ylabel(f"mean ΔMSE ({unit}; 0.5–2 h, member-internal then equal-weight)")
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    highlight = [m for m in (ranking.get("strongest_member"), ranking.get("weakest_member")) if m]
    if highlight:
        axes[0].set_title(f"ΔMSE_H; case-study members (max/min of member-mean ΔMSE_H): "
                          f"{', '.join(highlight)}", fontsize=9)
    fig.suptitle("Task2 member spread: all members retained, no cherry-picking", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    produced += _save_figure(fig, stem)
    _write_caption(stem, [
        "成员差异图:逐成员 0.5–2 h 平均 ΔMSE_H(左)与 ΔMSE_TSK(右);",
        "区域 = all(r≤150 km),掩膜 = ocean_common(三方海洋公共)。",
        "选择规则:" + str(ranking.get("rule", "")),
        f"选定成员:最强 ΔMSE_H = {ranking.get('strongest_member')},"
        f"最弱 = {ranking.get('weakest_member')};其余成员以本图与 member_metrics.csv 保留。",
        "蓝色 = EAKF,橙红 = QCF_RHF;同方法内成员等权,不合并两方法为独立样本。",
    ])
    table = sub.groupby(["method", "member"])[["hfx_delta_mse", "tsk_delta_mse",
                                               "hfx_cross", "hfx_square"]].mean()
    table.to_csv(tables_dir() / "member_distribution_0p5_2h.csv")
    _save_source(stem, table=table.reset_index().to_numpy(dtype=str))
    produced.append(stem)
    return produced


def plot_case_members() -> list[str]:
    """正文案例成员图:两名成员的 δH 与 C(4 主时次 × 2 成员)。"""
    produced = []
    metrics_path = tables_dir() / "member_metrics.csv"
    if not metrics_path.exists():
        return produced
    ranking = tp.member_dse_h_ranking(pd.read_csv(metrics_path))
    members = [m for m in (ranking.get("strongest_member"), ranking.get("weakest_member")) if m]
    if not members:
        return produced
    hours = PLOT_CONFIG["main_hours"]
    for method in PLOT_CONFIG["methods"]:
        rows = len(hours) * len(members)
        fig, axes = plt.subplots(rows, 2, figsize=(6.6, 2.6 * rows), facecolor="white",
                                 layout="constrained")
        axes = np.atleast_2d(axes)
        source_arrays = {}
        row = 0
        for member in members:
            for hour in hours:
                case = _load_case(method, member, hour)
                if case is None:
                    row += 1
                    continue
                err = tp.derive_error_fields(case)
                for c, column in enumerate(("delta_hfx", "cross_h")):
                    values = np.asarray(err[column].values, dtype=float)
                    finite = values[np.isfinite(values)]
                    amp = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
                    amp = amp if amp > 0 else 1.0
                    title, unit = VARIABLE_LABELS[column]
                    _panel(axes[row, c],
                           np.asarray(case["lon"].values, dtype=float),
                           np.asarray(case["lat"].values, dtype=float),
                           values, -amp, amp, f"{member} t={hour:g} h  {title}", unit,
                           case=case)
                    source_arrays[f"{member}_t{hour:g}_{column}"] = values
                row += 1
        fig.suptitle(f"[{method}] case-study members: δHFX and C (odd rows: strongest ΔMSE_H {members[0]};"
                     f" even rows: weakest {members[-1] if len(members) > 1 else members[0]})", fontsize=11)
        stem = f"task2_case_members_{method}"
        produced += _save_figure(fig, stem)
        _write_caption(stem, [
            f"[{method}] 正文案例成员图:按平均 ΔMSE_H 最大({members[0]})与最小"
            f"({members[-1] if len(members) > 1 else members[0]})选取,规则见 member_dse_h_ranking;",
            "δH 与 C 为该成员逐点场(未做成员间平均);NR 中心与 75/150 km 圈叠加;",
            "每面板色标为该面板自身 ±P98,以 0 为中心。",
        ])
        if source_arrays:
            _save_source(stem, **source_arrays)
        produced.append(stem)
    return produced


# =====================================================================
# Task 3:重建残差与替换分量
# =====================================================================

RECON_COLUMNS = ("residual_w", "residual_s", "dF_T", "dF_rest", "delta_actual")


def plot_reconstruction() -> list[str]:
    produced = []
    hours = PLOT_CONFIG["main_hours"]
    recon_dir = cache_dir("recon")
    if not recon_dir.exists():
        return produced
    fields: dict[tuple[str, str, float], dict] = {}
    for method in PLOT_CONFIG["methods"]:
        for hour in hours:
            per_member = {column: [] for column in RECON_COLUMNS}
            ref_case = None
            for member in tp.CONFIG["members"]:
                path = recon_dir / f"recon_{tp.case_id(method, member, hour)}.nc"
                case_path = cache_dir("case") / f"case_{tp.case_id(method, member, hour)}.nc"
                if not (path.exists() and case_path.exists()):
                    continue
                recon = tp.load_recon_cache(path)
                mask = np.asarray(recon["mask_recon"].values, dtype=bool)
                if ref_case is None:
                    ref_case = tp.load_case_cache(case_path)
                for column in RECON_COLUMNS:
                    values = np.asarray(recon[column].values, dtype=float)
                    per_member[column].append(np.where(mask, values, np.nan))
            if ref_case is None:
                continue
            for column in RECON_COLUMNS:
                mean, coverage = _member_mean_common(per_member[column])
                fields[(column, method, hour)] = {
                    "mean": mean, "coverage": coverage, "case": ref_case,
                    "lat": np.asarray(ref_case["lat"].values, dtype=float),
                    "lon": np.asarray(ref_case["lon"].values, dtype=float),
                }
    scales = {}
    for column in RECON_COLUMNS:
        panels = [v["mean"] for (c, _, _), v in fields.items() if c == column]
        scales[column] = _symmetric_scale(panels, PLOT_CONFIG["signed_percentile"])
    for method in PLOT_CONFIG["methods"]:
        stem = f"task3_reconstruction_{method}"
        fig, axes = plt.subplots(len(hours), len(RECON_COLUMNS),
                                 figsize=(3.3 * len(RECON_COLUMNS), 3.0 * len(hours)),
                                 facecolor="white", layout="constrained")
        source_arrays = {}
        clip_lines = []
        for r, hour in enumerate(hours):
            for c, column in enumerate(RECON_COLUMNS):
                ax = axes[r, c]
                item = fields.get((column, method, hour))
                title, unit = VARIABLE_LABELS[column]
                if item is None:
                    ax.text(0.5, 0.5, "cache missing", transform=ax.transAxes, ha="center")
                    continue
                vmin, vmax = scales[column]
                stats = _clip_stats(item["mean"], vmin, vmax)
                _panel(ax, item["lon"], item["lat"], item["mean"], vmin, vmax,
                       f"t={hour:g} h  {title}", unit, case=item["case"])
                clip_lines.append(f"t={hour:g}h {column}: clipped {stats['clipped_fraction'] * 100:.2f}%,"
                                  f" 范围 [{stats['min']:.4g}, {stats['max']:.4g}] {unit}")
                source_arrays[f"{column}_t{hour:g}"] = item["mean"]
                source_arrays[f"coverage_t{hour:g}"] = item["coverage"]
                if "lat" not in source_arrays:
                    source_arrays["lat"] = item["lat"]
                    source_arrays["lon"] = item["lon"]
        for c, column in enumerate(RECON_COLUMNS):
            vmin, vmax = scales[column]
            sm = plt.cm.ScalarMappable(cmap=PLOT_CONFIG["cmap_signed"],
                                       norm=plt.Normalize(vmin, vmax))
            fig.colorbar(sm, ax=axes[:, c].tolist(), shrink=0.85, pad=0.02,
                         label=VARIABLE_LABELS[column][1])
        fig.suptitle(f"[{method}] Task3 reconstruction residuals and replacement components "
                     "(6-member common-coverage mean; quality in reconstruction_quality.csv)",
                     fontsize=11)
        produced += _save_figure(fig, stem)
        _write_caption(stem, [
            f"[{method}] Task3:列 = r_W、r_S(重建−模式残差)、dF_T(只替换海温)、"
            "dF_rest(替换其余输入)、实际 δH。",
            "dF_T/dF_rest 是两种条件路径;反向路径与等权平均见 replacement_metrics.csv;",
            "残差质量未达 strict 的案例不得用于物理归因,阈值与敏感性见 reconstruction_quality.csv。",
            *clip_lines,
        ])
        _save_source(stem, scales=json.dumps({k: list(v) for k, v in scales.items()}),
                     **source_arrays)
        produced.append(stem)
    return produced


# =====================================================================
# Task 4:居中/固定双映射
# =====================================================================

def plot_storm_maps() -> list[str]:
    produced = []
    hours = PLOT_CONFIG["main_hours"]
    storm_dir = cache_dir("storm")
    if not storm_dir.exists():
        return produced
    for method in PLOT_CONFIG["methods"]:
        stem = f"task4_storm_maps_{method}"
        fig, axes = plt.subplots(len(hours), 2, figsize=(7.8, 3.0 * len(hours)),
                                 facecolor="white", layout="constrained")
        source_arrays = {}
        for r, hour in enumerate(hours):
            means = {("centered"): [], ("fixed"): []}
            ref = None
            for member in tp.CONFIG["members"]:
                path = storm_dir / f"storm_{tp.case_id(method, member, hour)}.nc"
                if not path.exists():
                    continue
                storm = tp.load_storm_cache(path)
                if ref is None:
                    ref = storm
                gx, gy = np.meshgrid(np.asarray(storm["gx"].values, dtype=float),
                                     np.asarray(storm["gy"].values, dtype=float))
                nr_centered = np.asarray(storm["hfx_nr_centered"].values, dtype=float)
                nr_fixed = np.asarray(storm["hfx_nr_fixed"].values, dtype=float)
                w_centered = np.asarray(storm["hfx_w_centered"].values, dtype=float)
                w_fixed = np.asarray(storm["hfx_w_fixed"].values, dtype=float)
                matched = np.asarray(storm["matched_hfx"].values, dtype=bool)
                means["centered"].append(np.where(matched, w_centered - nr_centered, np.nan))
                means["fixed"].append(np.where(matched, w_fixed - nr_fixed, np.nan))
            if ref is None:
                for c in range(2):
                    axes[r, c].text(0.5, 0.5, "cache missing", transform=axes[r, c].transAxes, ha="center")
                continue
            for c, mapping in enumerate(("centered", "fixed")):
                mean, coverage = _member_mean_common(means[mapping])
                finite = mean[np.isfinite(mean)]
                amp = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
                amp = amp if amp > 0 else 1.0
                title = (f"t={hour:g} h  e$_{{H,W}}$ "
                         + ("storm-centered" if mapping == "centered" else "fixed geolocation (NR center)"))
                _panel(axes[r, c], gx, gy, mean, -amp, amp, title, "W m$^{-2}$")
                source_arrays[f"{mapping}_t{hour:g}"] = mean
                source_arrays[f"coverage_{mapping}_t{hour:g}"] = coverage
                if "gx" not in source_arrays:
                    source_arrays["gx"] = gx
                    source_arrays["gy"] = gy
        fig.suptitle(f"[{method}] Task4: weak-exp HFX error, storm-centered vs fixed mapping "
                     "(mean over members on matched_hfx common mask)", fontsize=11)
        produced += _save_figure(fig, stem)
        _write_caption(stem, [
            f"[{method}] 居中映射:各方以自己的最低 PSFC 中心;固定映射:三方均按 NR 中心;",
            "两版评分均在三方×两映射均有效的匹配公共掩膜上计算,不能解释为严格的位置误差因果分解;",
            "插值与覆盖敏感性见 storm_relative_metrics.csv 与 center_diagnostics.csv。",
        ])
        _save_source(stem, **source_arrays)
        produced.append(stem)
    return produced


def plot_radial_profiles() -> list[str]:
    radial_path = tables_dir() / "radial_profiles.csv"
    if not radial_path.exists():
        return []
    radial = pd.read_csv(radial_path)
    radial = radial[radial["lead_hour"] <= 2.0]
    if radial.empty:
        return []
    produced = []
    for variable, unit in (("tsk", "K"), ("hfx", "W m$^{-2}$"), ("d", "K")):
        stem = f"task4_radial_{variable}"
        fig, ax = plt.subplots(figsize=(6.8, 4.4), facecolor="white")
        for method, linestyle in zip(PLOT_CONFIG["methods"], ("-", (0, (5, 2)))):
            for party, color in (("w", "#D55E00"), ("s", "#009E73")):
                sub = radial[(radial["variable"] == variable)
                             & (radial["method"] == method) & (radial["party"] == party)]
                if sub.empty:
                    continue
                grouped = (sub.groupby("bin_center_km")
                           .agg(mean=("mean", "mean"), n=("n", "sum")))
                ax.plot(grouped.index, grouped["mean"], color=color,
                        linestyle=linestyle, marker="o", markersize=3,
                        label=f"{method} {'weak' if party == 'w' else 'strong'}")
        ax.set_xlabel("r from NR center (km, storm-centered mapping)")
        ax.set_ylabel(f"{variable.upper()} regional mean ({unit}; member-internal then equal-weight)")
        ax.set_title(f"Task4 radial profile (0.5–2 h, matched common mask): {variable.upper()}")
        ax.grid(linestyle=":", alpha=0.6)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        produced += _save_figure(fig, stem)
        _write_caption(stem, [
            f"径向廓线:居中映射、10 km 环带内匹配公共掩膜点等权平均;",
            "先成员内、再成员等权;0.5–2 h 全部主时次合并平均;单位 {0}。".format(unit),
            "评分差不是严格的位置误差因果分解; NR 场本身也经同一居中插值。",
        ])
        table = (radial[radial["variable"] == variable]
                 .groupby(["method", "party", "bin_center_km"])["mean"].mean())
        table.to_csv(tables_dir() / f"radial_{variable}_member_mean.csv")
        produced.append(stem)
    return produced


def plot_storm_bars() -> list[str]:
    metrics_path = tables_dir() / "storm_relative_metrics.csv"
    if not metrics_path.exists():
        return []
    metrics = pd.read_csv(metrics_path)
    sub = metrics[metrics["lead_hour"] <= 2.0]
    if sub.empty:
        return []
    stem = "task4_storm_bars"
    variables = ("tsk", "hfx", "d")
    fig, axes = plt.subplots(1, len(variables), figsize=(4.2 * len(variables), 4.0),
                             facecolor="white", layout="constrained")
    width = 0.35
    for ax, variable in zip(np.atleast_1d(axes), variables):
        key_c = f"{variable}_paired_rmse_change_centered"
        key_f = f"{variable}_paired_rmse_change_fixed"
        positions = np.arange(len(PLOT_CONFIG["methods"]), dtype=float)
        for offset, key, color, label in ((-width / 2, key_c, "#0072B2", "storm-centered"),
                                          (width / 2, key_f, "#D55E00", "fixed geolocation")):
            values = [float(sub[sub["method"] == m][key].mean()) for m in PLOT_CONFIG["methods"]]
            ax.bar(positions + offset, values, width=width, color=color, label=label)
        unit = {"tsk": "K", "hfx": "W m$^{-2}$", "d": "K"}[variable]
        ax.set_xticks(positions)
        ax.set_xticklabels(PLOT_CONFIG["methods"])
        ax.set_ylabel(f"RMSE(S)−RMSE(W) ({unit}; matched common mask)")
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        ax.set_title(f"{variable.upper()}: paired RMSE change (0.5–2 h mean)")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Task4: paired S−W RMSE change (member-internal then equal-weight; "
                 "not a causal decomposition of position error)", fontsize=11)
    produced = _save_figure(fig, stem)
    _write_caption(stem, [
        "S−W 配对 RMSE 变化:同一匹配公共掩膜上,居中映射与固定地理映射并列;",
        "正值 = 强试验 RMSE 更大;评分同时含位移、插值与掩膜影响,不能称为位置误差的唯一物理贡献;",
        "原表原掩膜评分另行保留于 member_metrics.csv(mask_kind=original)。",
    ])
    _save_source(stem,
                 values=sub.groupby("method")[
                     [f"{v}_paired_rmse_change_{m}" for v in variables for m in ("centered", "fixed")]
                 ].mean().reset_index(drop=False).select_dtypes("number").to_numpy(dtype=float))
    return [stem] + produced


# =====================================================================
# 入口
# =====================================================================

def main() -> None:
    steps = (
        ("task2_main_spatial", plot_main_spatial),
        ("task2_signed_errors", plot_signed_errors),
        ("task2_member_distribution", plot_member_distribution),
        ("task2_case_members", plot_case_members),
        ("task3_reconstruction", plot_reconstruction),
        ("task4_storm_maps", plot_storm_maps),
        ("task4_radial", plot_radial_profiles),
        ("task4_storm_bars", plot_storm_bars),
    )
    for name, function in steps:
        try:
            produced = function()
            print(f"[plot] {name}: {len(produced)} outputs", flush=True)
        except Exception as error:  # 单个图族失败不影响其他图族
            print(f"[plot] {name} FAILED: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
