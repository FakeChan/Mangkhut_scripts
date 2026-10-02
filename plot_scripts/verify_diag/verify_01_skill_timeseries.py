"""诊断一:完整误差时间序列与海温-通量空间对应关系(SYNTHETIC 默认)。

对 OM_TMP 表层(om)、TSK(tsk)、HFX(hfx)、LH(lh),逐方法、逐成员、逐时刻、
逐区域计算弱/强耦合相对 NR 的平均偏差、MSE、RMSE,强弱误差差值,以及相对
RMSE 改善率:

    rmse_improvement_pct = 100*(RMSE_weak - RMSE_strong)/RMSE_weak   正值=改善
    *_change_strong_minus_weak = 强 - 弱误差差值                      正值=恶化

并按用户口径在共同有效格点上,对 (tsk, hfx) 与 (tsk, lh) 两组变量计算逐点

    dSE_X = (X_strong - X_NR)^2 - (X_weak - X_NR)^2

将逐点海温误差变化与通量误差变化交叉分类为四类(改善/恶化组合),
"未变化"情况按单侧/双侧单独处理。注意:区域平均的海温改善不能直接解释为
同一格点上的通量恶化;本诊断给出的正是逐点对应关系。

运行: python verify_01_skill_timeseries.py (默认合成模式)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import verify_common as vc
from verify_common import VerifyConfig

SCRIPT_NAME = "verify01_skill_timeseries"

# =====================
# 可编辑配置区
# =====================
CONFIG = VerifyConfig(
    mode="synthetic",  # "synthetic"(默认) | "real"(需同时改 real 配置并确认,见 README)
    output_dirname=SCRIPT_NAME,
    plot_regions=("r000_300",),  # 时间序列图绘制的区域
)
#: 空间分类图与逐点场保存的案例(方法, 成员, 预报时效);可增删
MAP_CASES = (
    ("EAKF", "006", 0.5),
    ("EAKF", "006", 3.0),
    ("QCF_RHF", "015", 3.0),
)
#: 分类配对:(海温侧, 通量)。按研究口径用 TSK;如需用 om(OM_TMP 表层)可改此键。
CLASSIFICATION_PAIRS = (("tsk", "hfx"), ("tsk", "lh"))

METRIC_COLUMNS = [
    "mode", "method", "member", "time_hour", "valid_time", "region", "variable", "unit",
    "weight_method", "n_valid", "status",
    "weak_bias", "strong_bias",
    "mse_weak", "mse_strong",
    "rmse_weak", "rmse_strong",
    "mse_change_strong_minus_weak", "rmse_change_strong_minus_weak",
    "rmse_improvement_pct",
]
CLASSIFICATION_COLUMNS = [
    "mode", "method", "member", "time_hour", "valid_time", "pair", "region",
    "sst_variable", "flux_variable", "unit", "weight_method", "n_valid", "status",
    *(f"frac_{label}" for label in vc.CATEGORY_LABELS.values()),
]
STATUS_COLUMNS = ["mode", "method", "member", "time_hour", "valid_time", "status"]


def _expected_times(config: VerifyConfig) -> dict:
    """各时间窗口的期望时次数(用于汇总覆盖率列)。"""
    hours = [hour for hour, _ in config.times]
    return {
        name: sum(1 for hour in hours if lo <= hour <= hi)
        for name, lo, hi in vc.WINDOWS
    }


def run(config: VerifyConfig = CONFIG) -> Path:
    """执行诊断一并写出全部结果;返回输出目录。合成模式仅使用内存合成数据。"""
    vc.check_mode(config)
    if config.mode == "synthetic":
        from verify_synthetic import build_synthetic_provider

        provider = build_synthetic_provider(config)
    else:
        provider = vc.RealWrfProvider(config)

    mode_tag = config.mode.upper()
    out_dir = vc.ensure_output_dir(config, SCRIPT_NAME)
    metric_rows: list[dict] = []
    class_rows: list[dict] = []
    status_rows: list[dict] = []

    for time_hour, time_name in config.times:
        try:
            context = vc.build_time_context(provider, config, time_hour)
        except Exception as error:  # NR 读取/中心识别/配准失败:显式记录,不中断其余时次
            for method in config.methods:
                for member in config.members:
                    status_rows.append({
                        "mode": mode_tag, "method": method, "member": member,
                        "time_hour": time_hour, "valid_time": time_name,
                        "status": f"nr_context_failed: {type(error).__name__}: {error}",
                    })
            continue
        center_lat, center_lon = context["center"]
        print(f"[{mode_tag}] t={time_hour:g}h nr_center=({center_lat:.3f},{center_lon:.3f})", flush=True)
        pending_fields: list[dict] = []
        for method in config.methods:
            for member in config.members:
                if not provider.has_case(method, member, time_hour):
                    status_rows.append({
                        "mode": mode_tag, "method": method, "member": member,
                        "time_hour": time_hour, "valid_time": time_name,
                        "status": "missing_member_or_time",
                    })
                    continue
                try:
                    strong = provider.exp_fields(config.strong_experiment, method, member, time_hour)
                    weak = provider.exp_fields(config.weak_experiment, method, member, time_hour)
                except Exception as error:
                    status_rows.append({
                        "mode": mode_tag, "method": method, "member": member,
                        "time_hour": time_hour, "valid_time": time_name,
                        "status": f"read_error: {type(error).__name__}: {error}",
                    })
                    continue
                base = {
                    "mode": mode_tag,
                    "method": method,
                    "member": member,
                    "time_hour": time_hour,
                    "valid_time": time_name,
                    "weight_method": vc.WEIGHT_LABEL,
                }
                # ---- 主指标:逐变量逐区域(强弱与 NR 共同有效格点) ----
                for variable in ("om", "tsk", "hfx", "lh"):
                    for region_name, region_mask in context["regions"].items():
                        mask = region_mask & context["ocean"]
                        stats = vc.paired_error_metrics(
                            strong[variable], weak[variable], context["nr"][variable], mask
                        )
                        metric_rows.append({
                            **base,
                            "region": region_name,
                            "variable": variable,
                            "unit": vc.VARIABLE_UNITS[variable],
                            **stats,
                        })
                # ---- 空间分类:海温-通量逐点对应 ----
                for sst_var, flux_var in CLASSIFICATION_PAIRS:
                    for region_name, region_mask in context["regions"].items():
                        mask = region_mask & context["ocean"] & vc.finite_mask(
                            strong[sst_var], weak[sst_var], context["nr"][sst_var],
                            strong[flux_var], weak[flux_var], context["nr"][flux_var],
                        )
                        dse_sst = vc.se_change(strong[sst_var], weak[sst_var], context["nr"][sst_var])
                        dse_flux = vc.se_change(strong[flux_var], weak[flux_var], context["nr"][flux_var])
                        category = vc.classify_change_categories(
                            dse_sst, dse_flux, config.sst_se_tol, config.flux_se_tol
                        )
                        fractions = vc.category_fractions(category, mask)
                        class_rows.append({
                            **base,
                            "pair": f"{sst_var}-{flux_var}",
                            "region": region_name,
                            "sst_variable": sst_var,
                            "flux_variable": flux_var,
                            "unit": f"({vc.VARIABLE_UNITS[sst_var]},{vc.VARIABLE_UNITS[flux_var]})^2",
                            **fractions,
                        })
                        if (
                            (method, member, time_hour) in MAP_CASES
                            and region_name == config.union_region
                        ):
                            pending_fields.append({
                                "method": method,
                                "member": member,
                                "time_hour": time_hour,
                                "pair": f"{sst_var}-{flux_var}",
                                "dse_sst": np.where(mask, dse_sst, np.nan),
                                "dse_flux": np.where(mask, dse_flux, np.nan),
                                "category": np.where(mask, category, -99),
                            })
        # ---- 本时次配置案例的逐点场与分类图 ----
        for case in pending_fields:
            tag = (
                f"{case['method']}_{case['member']}_t{case['time_hour']:g}_"
                f"{case['pair'].replace('-', '_')}"
            )
            vc.save_point_fields(
                out_dir / "fields" / f"verify01_dSE_{tag}.npz",
                dse_sst=case["dse_sst"],
                dse_flux=case["dse_flux"],
                category=case["category"],
                distance_km=context["distance"],
                lat=context["static"]["lat"],
                lon=context["static"]["lon"],
                nr_center_lat=context["center"][0],
                nr_center_lon=context["center"][1],
                mode=np.asarray(mode_tag),
                method=np.asarray(case["method"]),
                member=np.asarray(case["member"]),
                time_hour=np.asarray(case["time_hour"]),
            )
            vc.plot_category_map(
                case["category"],
                context["distance"],
                f"[{mode_tag}] SST-flux error-change category, {tag}\n"
                f"(tol: {config.sst_se_tol:g} K^2, {config.flux_se_tol:g} (W m-2)^2)",
                out_dir / "figs" / f"verify01_category_map_{tag}.png",
            )

    # =====================
    # 汇总与输出(run_status 最先落盘:即使后续汇总/绘图异常,缺失与错误也已持久化)
    # =====================
    status = pd.DataFrame(status_rows, columns=STATUS_COLUMNS)
    vc.write_csv(status, out_dir / "verify01_run_status.csv")

    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    classifications = pd.DataFrame(class_rows, columns=CLASSIFICATION_COLUMNS)

    summary_region_window = vc.summarize_member_metrics(
        metrics, keys=("mode", "method", "region", "variable", "unit"),
        expected_times=_expected_times(config),
        expected_members=len(config.members),
    )
    # 方法级主汇总:仅用完整 0-300 km 区域,不把并集与子环带混合平均;
    # n_members_improved 在选定区域/窗口内按唯一成员计数(<= n_members)。
    summary_all = summary_region_window[
        summary_region_window.region == config.union_region
    ].drop(columns=["region"]).reset_index(drop=True)

    vc.write_csv(metrics, out_dir / "verify01_member_metrics.csv")
    vc.write_csv(classifications, out_dir / "verify01_member_classification_fractions.csv")
    vc.write_csv(summary_region_window, out_dir / "verify01_summary_by_region_window.csv")
    vc.write_csv(summary_all, out_dir / "verify01_summary_by_method.csv")

    _plot_timeseries(metrics, config, out_dir, mode_tag)
    vc.write_columns_readme(out_dir / "columns_readme.txt", _columns_readme_lines(config))
    _print_summary(metrics, classifications, status, mode_tag)
    ok_rows = int((metrics.status == "ok").sum()) if len(metrics) else 0
    vc.write_overall_status(out_dir, ok_rows, len(status), mode_tag)
    return out_dir


def _plot_timeseries(metrics: pd.DataFrame, config: VerifyConfig, out_dir: Path, mode_tag: str) -> None:
    """时序图;缺失时刻在序列中补 NaN 断线,不跨接前后时次。"""
    hours = [hour for hour, _ in config.times]
    for variable in ("om", "tsk", "hfx", "lh"):
        for region_name in config.plot_regions:
            selected = metrics[(metrics.variable == variable) & (metrics.region == region_name)]
            if selected.empty:
                continue
            series_rmse: dict[str, tuple] = {}
            series_impr: dict[str, tuple] = {}
            for method in config.methods:
                for member in config.members:
                    member_frame = selected[
                        (selected.method == method) & (selected.member == member)
                    ]
                    if member_frame.empty:
                        continue
                    aligned = vc.align_times_to_hours(member_frame, hours)
                    label = f"{method}-{member}"
                    series_rmse[f"{label} weak"] = (
                        aligned.index, aligned.rmse_weak.to_numpy(dtype=float)
                    )
                    series_rmse[f"{label} strong"] = (
                        aligned.index, aligned.rmse_strong.to_numpy(dtype=float)
                    )
                    series_impr[label] = (
                        aligned.index, aligned.rmse_improvement_pct.to_numpy(dtype=float)
                    )
            suffix = f"{variable}_{region_name}"
            vc.plot_metric_timeseries(
                series_rmse,
                f"[{mode_tag}] RMSE vs NR, {variable} ({region_name})",
                f"RMSE ({vc.VARIABLE_UNITS[variable]})",
                out_dir / "figs" / f"verify01_rmse_timeseries_{suffix}.png",
            )
            vc.plot_metric_timeseries(
                series_impr,
                f"[{mode_tag}] RMSE improvement, {variable} ({region_name})",
                "rmse improvement (%)",
                out_dir / "figs" / f"verify01_improvement_timeseries_{suffix}.png",
            )


def _columns_readme_lines(config: VerifyConfig) -> list[str]:
    return [
        "verify01 输出列说明(SYNTHETIC/REAL 由 mode 列标记)",
        "",
        f"- mode: {config.mode.upper()}(合成/真实标记)",
        "- weight_method: 所有主结果的权重方式 = 'equal_weight_valid_points'(等权有效格点平均)",
        "- rmse_improvement_pct = 100*(RMSE_weak - RMSE_strong)/RMSE_weak,正值表示强耦合改善;",
        "  RMSE_weak 为零时置 NaN(status=zero_rmse_weak),其余绝对误差指标仍然有效。",
        "- mse/rmse_change_strong_minus_weak = 强 - 弱,正值表示强耦合误差更大(恶化)。",
        "- weak/strong_bias = mean(X_exp - X_NR)。",
        "- 汇总表两种口径(不可混用):",
        "    mean_member_rmse_improvement_pct:各成员相对改善率的平均值(成员先做窗口内时间平均);",
        "    pooled_rmse_improvement_pct:平均 RMSE 的相对变化。",
        "- verify01_summary_by_region_window.csv:按 (区域×窗口) 的逐区域成员汇总,保留全部区域;",
        "  verify01_summary_by_method.csv:主汇总,仅用完整 0-300 km 区域(r000_300),",
        "  不把并集与子环带混合平均;n_members_improved 为该区域/窗口内唯一成员数(<= n_members)。",
        "- 覆盖率列(分母显式化):mean/min/max_member_times_read 为成员窗口内已读取时次数;",
        "  mean/min_member_valid_abs_times 为有效绝对误差时次数(status=ok 或 zero_rmse_weak,",
        "  零基准 RMSE 的绝对误差视为有效);mean_member_valid_improvement_times 为改善率有限",
        "  的时次数;expected_times/member_abs_time_coverage_ratio 以配置时刻数为分母;",
        "  expected_members/member_coverage_ratio 以配置成员数为分母(完全缺失的成员计入分母)。",
        "- 分类占比列 frac_*:分母为该 (时刻, 配对, 区域) 的共同有效格点数 n_valid;",
        f"  判定容差:sst {config.sst_se_tol:g} K^2、flux {config.flux_se_tol:g} (W m-2)^2",
        "  (可调,非物理常数;容差直接作用于 dSE=2e·dF+dF^2,不能开方解释为普适的增量阈值)。",
        "- 分类含义:1=海温改善&通量改善, 2=海温改善&通量恶化, 3=海温恶化&通量改善,",
        "  4=海温恶化&通量恶化, 0=双侧未变化, -1=单侧未变化(单独处理)。",
        "- 区域平均海温改善不能直接解释为同一格点上的通量恶化;逐点分类即为此设计。",
        "- fields/*.npz 保存逐点 dSE 与 category 场(含 mode/method/member/time 元数据),",
        "  供复查图件和与诊断二的 dSE 逐点场对应。",
        "- 时序图对缺失时刻补 NaN 断线,不跨接前后时次。",
    ]


def _print_summary(metrics, classifications, status, mode_tag: str) -> None:
    ok = int((metrics.status == "ok").sum()) if len(metrics) else 0
    empty = int((metrics.status == "empty_mask").sum()) if len(metrics) else 0
    zero = int((metrics.status == "zero_rmse_weak").sum()) if len(metrics) else 0
    print(
        f"[{mode_tag}] verify01: {len(metrics)} metric rows ({ok} ok, {empty} empty_mask, "
        f"{zero} zero_rmse_weak); {len(classifications)} classification rows; "
        f"{len(status)} skipped cases",
        flush=True,
    )


if __name__ == "__main__":
    output_dir = run(CONFIG)
    print(f"[{CONFIG.mode.upper()}] verify01 done -> {output_dir}", flush=True)
