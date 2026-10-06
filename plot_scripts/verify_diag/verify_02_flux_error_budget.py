"""诊断二:通量误差变化的精确分解(SYNTHETIC 默认)。

对 HFX、LH 分别定义(逐格点):

    e  = F_weak - F_NR
    dF = F_strong - F_weak
    dSE = 2 e dF + (dF)^2

在统一掩膜(区域 & 海洋 & 全部数组共同有效)上等权平均:

    delta_mse_direct = MSE_strong - MSE_weak
                     = cross_term (2<e dF>) + increment_square_term (<dF^2>)
    closure_residual = delta_mse_direct - (cross_term + increment_square_term)

注意:
- 使用未去均值的误差乘积,不得替换为协方差;
- 逐点乘积先平均,不得用 <e><dF> 代替 <e dF>;
- 解释保持克制:净误差减小=通量误差改善;交叉项非负=区域整体变化方向不利;
  交叉项为负但净误差增大=存在纠错倾向但净效果不利;净变化在容差内=unchanged。
  这只是误差预算描述,不自动等同于物理因果结论。

运行: python verify_02_flux_error_budget.py (默认合成模式)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import verify_common as vc
from verify_common import VerifyConfig

SCRIPT_NAME = "verify02_flux_error_budget"

# =====================
# 可编辑配置区
# =====================
CONFIG = VerifyConfig(
    mode="real",  # "synthetic"(默认) | "real"(需同时改 real 配置并确认,见 README)
    output_dirname=SCRIPT_NAME,
    plot_regions=("r000_300", "r000_075", "r075_150", "r150_300"),
    real=vc.RealPathConfig(
        acknowledge_real_mode=True,   # 确认开关,缺省会拒绝运行
    )
)
#: 保存逐点 dSE 场(.npz)的案例(方法, 成员, 预报时效)
POINT_FIELD_CASES = (
    ("EAKF", "006", 0.5),
    ("EAKF", "006", 3.0),
    ("QCF_RHF", "015", 3.0),
)
#: 通量变量
FLUX_VARIABLES = ("hfx", "lh")
#: 条件子集的海温侧变量(默认 TSK;dSE_T 单位 K^2,与 flux_se_tol 共用阈值)
SST_VARIABLE = "tsk"
#: 交叉项方向状态的独立容差(单位 = (变量单位)^2,与净变化容差分开;
#: 容差只影响分类,不修改数值;接近零标记为中性,不归入"方向不利")
CROSS_TERM_TOL = 1.0  # (W m-2)^2
#: interpretation 的 unchanged 判定容差:|delta_mse_direct| <= 该值视为净无变化
#: (单位 (变量单位)^2;可调,非物理常数)
BUDGET_UNCHANGED_TOL = 1.0e-9

BUDGET_COLUMNS = [
    "mode", "method", "member", "time_hour", "valid_time", "region", "variable", "unit",
    "weight_method", "n_valid", "status", "improvement_status",
    "mse_weak", "mse_strong", "rmse_weak", "rmse_strong", "rmse_improvement_pct",
    "delta_mse_direct", "cross_term", "increment_square_term", "closure_residual",
    "interpretation",
]
STATUS_COLUMNS = ["mode", "method", "member", "time_hour", "valid_time", "status"]

#: 条件子集预算表(独立表,不替换原有全区域预算;分母与状态显式)
CONDITIONAL_COLUMNS = [
    "mode", "method", "member", "time_hour", "valid_time", "region",
    "variable", "unit", "sst_variable", "sst_threshold_k2", "cross_term_tol_w2",
    "subset", "weight_method",
    "n_all_common", "n_subset", "subset_fraction_of_all_common",
    "n_valid", "status", "improvement_status", "cross_term_direction",
    "mse_weak", "mse_strong", "rmse_weak", "rmse_strong",
    "delta_mse_direct", "cross_term", "increment_square_term", "closure_residual",
]


def _expected_times(config: VerifyConfig) -> dict:
    hours = [hour for hour, _ in config.times]
    return {
        name: sum(1 for hour in hours if lo <= hour <= hi)
        for name, lo, hi in vc.WINDOWS
    }


def interpret_budget(budget: dict, unchanged_tol: float = BUDGET_UNCHANGED_TOL) -> str:
    """克制的误差预算描述标签(非物理因果结论)。"""
    if budget["status"] != "ok":
        return budget["status"]
    direct = budget["delta_mse_direct"]
    cross = budget["cross_term"]
    if abs(direct) <= unchanged_tol:
        return "unchanged"
    if direct < 0.0:
        return "flux_error_improved"
    if cross >= 0.0:
        return "change_direction_unfavorable"
    return "corrective_tendency_but_net_unfavorable"


def compute(config: VerifyConfig = CONFIG, provider=None, out_dir: Path | None = None) -> dict:
    """诊断二计算部分(内存复用入口):不做汇总与落盘,返回逐案例原始表。

    返回键 status / budgets / conditional,与 run() 落盘的
    verify02_run_status / verify02_member_budget /
    verify02_conditional_budget 三张主表逐列一致。
    out_dir 给定时另存 POINT_FIELD_CASES 的逐点场;为 None 时无任何文件写入。
    """
    vc.check_mode(config)
    if provider is None:
        provider = vc.make_provider(config)

    mode_tag = config.mode.upper()
    budget_rows: list[dict] = []
    conditional_rows: list[dict] = []
    status_rows: list[dict] = []

    for time_hour, time_name in config.times:
        try:
            context = vc.build_time_context(provider, config, time_hour)
        except Exception as error:
            for method in config.methods:
                for member in config.members:
                    status_rows.append({
                        "mode": mode_tag, "method": method, "member": member,
                        "time_hour": time_hour, "valid_time": time_name,
                        "status": f"nr_context_failed: {type(error).__name__}: {error}",
                    })
            continue
        print(f"[{mode_tag}] t={time_hour:g}h nr_center={context['center']}", flush=True)
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
                dse_sst = vc.se_change(
                    strong[SST_VARIABLE], weak[SST_VARIABLE], context["nr"][SST_VARIABLE]
                )
                for variable in FLUX_VARIABLES:
                    dse = vc.se_change(strong[variable], weak[variable], context["nr"][variable])
                    for region_name, region_mask in context["regions"].items():
                        mask = region_mask & context["ocean"] & vc.finite_mask(
                            strong[variable], weak[variable], context["nr"][variable]
                        )
                        budget = vc.flux_error_budget(
                            strong[variable], weak[variable], context["nr"][variable], mask
                        )
                        if budget["status"] == "ok":
                            budget["rmse_weak"] = float(np.sqrt(budget["mse_weak"]))
                            budget["rmse_strong"] = float(np.sqrt(budget["mse_strong"]))
                            if budget["rmse_weak"] > 0.0:
                                budget["rmse_improvement_pct"] = (
                                    100.0
                                    * (budget["rmse_weak"] - budget["rmse_strong"])
                                    / budget["rmse_weak"]
                                )
                                budget["improvement_status"] = "ok"
                            else:
                                # 零基准:改善率无定义,但其余绝对误差指标仍有效
                                budget["rmse_improvement_pct"] = np.nan
                                budget["improvement_status"] = "zero_rmse_weak"
                        else:
                            budget["rmse_weak"] = np.nan
                            budget["rmse_strong"] = np.nan
                            budget["rmse_improvement_pct"] = np.nan
                            budget["improvement_status"] = budget["status"]
                        budget_rows.append({
                            **base,
                            "region": region_name,
                            "variable": variable,
                            "unit": f"({vc.VARIABLE_UNITS[variable]})^2",
                            **budget,
                            "interpretation": interpret_budget(budget),
                        })
                        # ---- 条件子集预算(新增;独立表,分母 = 同案例/区域/变量的
                        # all_common 点数;子集只按海温误差变化筛选)----
                        # 实际弱误差(F_weak_model - F_NR)与实际强弱增量(强-弱)
                        error_model = np.asarray(weak[variable], dtype=float) - context["nr"][variable]
                        increment_actual = vc.strong_minus_weak(
                            strong[variable], weak[variable]
                        )
                        base_mask = (
                            np.asarray(region_mask, dtype=bool) & context["ocean"]
                            & vc.finite_mask(
                                strong[variable], weak[variable],
                                context["nr"][variable],
                                strong[SST_VARIABLE], weak[SST_VARIABLE],
                                context["nr"][SST_VARIABLE],
                            )
                        )
                        subset_masks = vc.sst_subset_masks(
                            dse_sst, base_mask, config.sst_se_tol
                        )
                        n_all_common = int(np.count_nonzero(subset_masks["all_common"]))
                        for subset_name, subset_mask in subset_masks.items():
                            subset_budget = vc.cross_term_stats(
                                error_model, increment_actual, subset_mask
                            )
                            if subset_budget["status"] == "ok":
                                subset_budget["rmse_weak"] = float(np.sqrt(subset_budget["mse_weak"]))
                                subset_budget["rmse_strong"] = float(np.sqrt(subset_budget["mse_strong"]))
                                if subset_budget["rmse_weak"] > 0.0:
                                    subset_budget["improvement_status"] = "ok"
                                else:
                                    subset_budget["improvement_status"] = "zero_rmse_weak"
                            else:
                                subset_budget["rmse_weak"] = np.nan
                                subset_budget["rmse_strong"] = np.nan
                                subset_budget["improvement_status"] = subset_budget["status"]
                            conditional_rows.append({
                                **base,
                                "region": region_name,
                                "variable": variable,
                                "unit": f"({vc.VARIABLE_UNITS[variable]})^2",
                                "sst_variable": SST_VARIABLE,
                                "sst_threshold_k2": config.sst_se_tol,
                                "cross_term_tol_w2": CROSS_TERM_TOL,
                                "subset": subset_name,
                                "n_all_common": n_all_common,
                                "n_subset": int(subset_budget["n_valid"]),
                                "subset_fraction_of_all_common": (
                                    subset_budget["n_valid"] / n_all_common
                                    if n_all_common > 0 else np.nan
                                ),
                                **subset_budget,
                                "cross_term_direction": vc.cross_term_direction(
                                    subset_budget["cross_term"], CROSS_TERM_TOL
                                ),
                            })
                        if (
                            (method, member, time_hour) in POINT_FIELD_CASES
                            and region_name == config.union_region
                        ):
                            pending_fields.append({
                                "method": method,
                                "member": member,
                                "time_hour": time_hour,
                                "variable": variable,
                                "dse": np.where(mask, dse, np.nan),
                                "e": np.where(
                                    mask, weak[variable] - context["nr"][variable], np.nan
                                ),
                                "df": np.where(
                                    mask, strong[variable] - weak[variable], np.nan
                                ),
                            })
        # 逐点场仅落盘模式保存(内存复用跳过)
        if out_dir is not None:
            for case in pending_fields:
                tag = (
                    f"{case['method']}_{case['member']}_t{case['time_hour']:g}_{case['variable']}"
                )
                vc.save_point_fields(
                    out_dir / "fields" / f"verify02_dSE_points_{tag}.npz",
                    dse=case["dse"],
                    error_weak=case["e"],
                    increment=case["df"],
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

    return {
        "status": pd.DataFrame(status_rows, columns=STATUS_COLUMNS),
        "budgets": pd.DataFrame(budget_rows, columns=BUDGET_COLUMNS),
        "conditional": pd.DataFrame(conditional_rows, columns=CONDITIONAL_COLUMNS),
    }


def run(config: VerifyConfig = CONFIG) -> Path:
    """执行诊断二并写出全部结果;返回输出目录。合成模式仅使用内存合成数据。"""
    vc.check_mode(config)
    provider = vc.make_provider(config)
    mode_tag = config.mode.upper()
    out_dir = vc.ensure_output_dir(config, SCRIPT_NAME)
    frames = compute(config, provider=provider, out_dir=out_dir)
    status = frames["status"]
    budgets = frames["budgets"]
    conditional = frames["conditional"]

    # =====================
    # 汇总与输出(run_status 最先落盘)
    # =====================
    vc.write_csv(status, out_dir / "verify02_run_status.csv")

    summary_region_window = vc.summarize_member_metrics(
        budgets, keys=("mode", "method", "region", "variable", "unit"),
        expected_times=_expected_times(config),
        expected_members=len(config.members),
    )
    # 主汇总仅用完整 0-300 km 区域(含交叉项/平方项/闭合残差的窗口汇总,
    # 可直接复核闭合),不把并集与子环带混合平均。
    summary_all = summary_region_window[
        summary_region_window.region == config.union_region
    ].drop(columns=["region"]).reset_index(drop=True)

    vc.write_csv(conditional, out_dir / "verify02_conditional_budget.csv")
    vc.write_csv(budgets, out_dir / "verify02_member_budget.csv")
    vc.write_csv(summary_region_window, out_dir / "verify02_summary_by_region_window.csv")
    vc.write_csv(summary_all, out_dir / "verify02_summary_by_method.csv")
    # 条件预算的窗口汇总(先成员内时间平均,后跨成员等权;不混合格点/时次)。
    # 逐成员表保留 mean_n_all_common(分母的成员均值);方法级表为成员等权平均,
    # 比率列(如 subset_fraction)为"成员平均率",不是总量之比。
    # 键与期望分母在分支前定义(P2 #5):全缺失时 cond_keys 不再 UnboundLocalError,
    # 空表保留完整列结构。期望分母(P2 #4)与原有主表口径一致:
    # expected_times 按窗口、expected_members 按配置成员数,
    # 覆盖率 = 实际/期望,用于判断新增汇总的证据覆盖程度。
    cond_keys = ["mode", "method", "region", "variable", "unit", "subset", "window"]
    expected_members = len(config.members)
    expected_times_by_window = _expected_times(config)
    if not conditional.empty:
        conditional = conditional.copy()
        conditional["window"] = [
            vc.window_of(hour) for hour in conditional["time_hour"]
        ]
        conditional = conditional.explode("window", ignore_index=True)
        per_member = (
            conditional.groupby(cond_keys + ["member"], sort=True)
            .agg(
                mean_delta_mse_direct=("delta_mse_direct", "mean"),
                mean_cross_term=("cross_term", "mean"),
                mean_increment_square_term=("increment_square_term", "mean"),
                mean_closure_residual=("closure_residual", "mean"),
                mean_mse_weak=("mse_weak", "mean"),
                mean_mse_strong=("mse_strong", "mean"),
                mean_n_subset=("n_valid", "mean"),
                mean_n_all_common=("n_all_common", "mean"),
                mean_subset_fraction=("subset_fraction_of_all_common", "mean"),
                n_times_read=("time_hour", "count"),
                n_times_valid=("status", lambda s: int((s == "ok").sum())),
            )
            .reset_index()
        )
        per_member["expected_times"] = (
            per_member.window.map(expected_times_by_window).astype(int)
        )
        per_member["expected_members"] = expected_members
        per_member["member_time_coverage_ratio"] = (
            per_member.n_times_valid / per_member.expected_times.where(
                per_member.expected_times > 0
            )
        )
        vc.write_csv(per_member, out_dir / "verify02_conditional_summary_by_member.csv")
        agg = per_member.drop(columns=["member"]).groupby(cond_keys, sort=True).agg(
            n_members=("mean_delta_mse_direct", "count"),
            mean_delta_mse_direct=("mean_delta_mse_direct", "mean"),
            mean_cross_term=("mean_cross_term", "mean"),
            mean_increment_square_term=("mean_increment_square_term", "mean"),
            mean_closure_residual=("mean_closure_residual", "mean"),
            mean_mse_weak=("mean_mse_weak", "mean"),
            mean_mse_strong=("mean_mse_strong", "mean"),
            mean_n_subset=("mean_n_subset", "mean"),
            mean_n_all_common=("mean_n_all_common", "mean"),
            mean_subset_fraction=("mean_subset_fraction", "mean"),
            mean_member_times_read=("n_times_read", "mean"),
            mean_member_valid_times=("n_times_valid", "mean"),
            expected_times=("expected_times", "first"),
            expected_members=("expected_members", "first"),
            min_member_valid_times=("n_times_valid", "min"),
            max_member_valid_times=("n_times_valid", "max"),
        ).reset_index()
        agg["member_time_coverage_ratio"] = (
            agg.mean_member_valid_times / agg.expected_times.where(agg.expected_times > 0)
        )
        agg["member_coverage_ratio"] = (
            agg.n_members / agg.expected_members.where(agg.expected_members > 0)
        )
        vc.write_csv(agg, out_dir / "verify02_conditional_summary_by_method.csv")
    else:
        member_columns = cond_keys + [
            "member", "mean_delta_mse_direct", "mean_cross_term",
            "mean_increment_square_term", "mean_closure_residual",
            "mean_mse_weak", "mean_mse_strong", "mean_n_subset",
            "mean_n_all_common", "mean_subset_fraction",
            "n_times_read", "n_times_valid", "expected_times", "expected_members",
            "member_time_coverage_ratio",
        ]
        method_columns = cond_keys + [
            "n_members", "mean_delta_mse_direct", "mean_cross_term",
            "mean_increment_square_term", "mean_closure_residual",
            "mean_mse_weak", "mean_mse_strong", "mean_n_subset",
            "mean_n_all_common", "mean_subset_fraction",
            "mean_member_times_read", "mean_member_valid_times",
            "expected_times", "expected_members",
            "min_member_valid_times", "max_member_valid_times",
            "member_time_coverage_ratio", "member_coverage_ratio",
        ]
        vc.write_csv(pd.DataFrame(columns=member_columns),
                     out_dir / "verify02_conditional_summary_by_member.csv")
        vc.write_csv(pd.DataFrame(columns=method_columns),
                     out_dir / "verify02_conditional_summary_by_method.csv")

    _plot_budgets(budgets, config, out_dir, mode_tag)
    vc.write_columns_readme(out_dir / "columns_readme.txt", _columns_readme_lines(config))
    _print_summary(budgets, status, mode_tag)
    ok_rows = int((budgets.status == "ok").sum()) if len(budgets) else 0
    vc.write_overall_status(out_dir, ok_rows, len(status), mode_tag)
    vc.save_config_snapshot(out_dir, config)
    return out_dir


def _member_then_mean(values: pd.DataFrame, column: str, members: tuple) -> float:
    """先成员内对时间等权平均,再跨成员等权平均(缺失时次不改变成员权重)。"""
    return vc.member_then_mean(values, column, members)


def _plot_budgets(budgets: pd.DataFrame, config: VerifyConfig, out_dir: Path, mode_tag: str) -> None:
    """交叉项、平方项、直接误差变化与闭合残差的时序图;以及分量柱状图。

    时序图为成员平均(逐时刻对可用成员等权,缺失时次补 NaN 断线);
    柱状图先做成员内时间平均、再跨成员等权,且按区域分别绘制,
    不把并集与子环带混合。
    """
    hours = [hour for hour, _ in config.times]
    for variable in FLUX_VARIABLES:
        for region_name in config.plot_regions:
            selected = budgets[(budgets.variable == variable) & (budgets.region == region_name)]
            if selected.empty:
                continue
            series: dict[str, tuple] = {}
            residual_series: dict[str, tuple] = {}
            for method in config.methods:
                method_frame = selected[selected.method == method]
                if method_frame.empty:
                    continue
                grouped = method_frame.groupby("time_hour", sort=True).agg(
                    cross_term=("cross_term", "mean"),
                    increment_square_term=("increment_square_term", "mean"),
                    delta_mse_direct=("delta_mse_direct", "mean"),
                    closure_residual=("closure_residual", "mean"),
                ).reindex(hours)  # 缺失时次补 NaN 断线,不跨接
                label = f"{method} (member mean)"
                series[f"{label}: direct dMSE"] = (grouped.index, grouped.delta_mse_direct)
                series[f"{label}: cross term"] = (grouped.index, grouped.cross_term)
                series[f"{label}: square term"] = (grouped.index, grouped.increment_square_term)
                residual_series[label] = (grouped.index, grouped.closure_residual)
            suffix = f"{variable}_{region_name}"
            vc.plot_metric_timeseries(
                series,
                f"[{mode_tag}] flux error budget, {variable} ({region_name})\n"
                "dMSE = cross + square (positive = strong worse)",
                f"dMSE ({vc.VARIABLE_UNITS[variable]})^2",
                out_dir / "figs" / f"verify02_budget_timeseries_{suffix}.png",
            )
            vc.plot_metric_timeseries(
                residual_series,
                f"[{mode_tag}] budget closure residual, {variable} ({region_name})",
                "closure residual",
                out_dir / "figs" / f"verify02_closure_residual_{suffix}.png",
            )
            # 分量柱状图:成员先均、后跨成员,逐方法;仅对该区域
            bar_values: dict[str, list] = {}
            for method in config.methods:
                method_frame = selected[selected.method == method]
                if method_frame.empty:
                    continue
                bar_values[f"{method}: cross"] = [
                    _member_then_mean(method_frame, "cross_term", config.members)
                ]
                bar_values[f"{method}: square"] = [
                    _member_then_mean(method_frame, "increment_square_term", config.members)
                ]
                bar_values[f"{method}: direct"] = [
                    _member_then_mean(method_frame, "delta_mse_direct", config.members)
                ]
            if bar_values:
                vc.plot_budget_bars(
                    [f"all cases ({variable}, {region_name})"],
                    bar_values,
                    f"[{mode_tag}] mean budget components, {variable} ({region_name})",
                    f"dMSE ({vc.VARIABLE_UNITS[variable]})^2",
                    out_dir / "figs" / f"verify02_budget_bars_{variable}_{region_name}.png",
                )


def _columns_readme_lines(config: VerifyConfig) -> list[str]:
    return [
        "verify02 输出列说明(SYNTHETIC/REAL 由 mode 列标记)",
        "",
        "公式(逐格点): e = F_weak - F_NR; dF = F_strong - F_weak;",
        "  dSE = 2 e dF + dF^2。",
        "区域平均(等权有效格点):",
        "  delta_mse_direct = MSE_strong - MSE_weak",
        "  cross_term = 2<e dF>(逐点乘积的平均,不是 <e><dF>,也不是协方差)",
        "  increment_square_term = <dF^2>",
        "  closure_residual = delta_mse_direct - (cross_term + increment_square_term)",
        "- 单位列 unit 为 (变量单位)^2。",
        "- weight_method = 'equal_weight_valid_points'(等权有效格点平均)。",
        "- improvement_status:rmse_improvement_pct 的可用性状态(ok / zero_rmse_weak);",
        "  零基准时改善率置 NaN,其余绝对误差指标仍然有效。",
        "- interpretation(克制的描述,非物理因果;unchanged 容差 "
        f"{BUDGET_UNCHANGED_TOL:g} (W m-2)^2 可调):",
        "    flux_error_improved:净误差减小(通量误差改善);",
        "    change_direction_unfavorable:交叉项非负,区域整体变化方向不利;",
        "    corrective_tendency_but_net_unfavorable:交叉项为负但净误差增大,",
        "      存在纠错倾向,但幅度或空间分布使净效果不利;",
        "    unchanged:净变化在容差内。",
        "- 汇总表:verify02_summary_by_region_window.csv 逐区域;主汇总",
        "  verify02_summary_by_method.csv 仅用 r000_300,不把并集与子环带混合;",
        "  其中含 mean_cross_term / mean_increment_square_term / mean_closure_residual,",
        "  可直接复核 mean_closure_residual ≈ 0(闭合)。",
        "- 覆盖率列同 verify01:已读取/有效绝对误差/有效改善率时次分开统计,",
        "  并给出 expected_times/expected_members 与对应覆盖率(分母显式)。",
        "- 时序图对缺失时刻补 NaN 断线,不跨接前后时次。",
        "- fields/*.npz 保存逐点 e、dF、dSE 场(含 mode/method/member/time 元数据),",
        "  可与诊断一的逐点分类直接对应。",
        "- 柱状图先做成员内时间平均再跨成员等权,按区域分别绘制。",
    ]


def _print_summary(budgets, status, mode_tag: str) -> None:
    ok = int((budgets.status == "ok").sum()) if len(budgets) else 0
    residuals = budgets.closure_residual.to_numpy(dtype=float) if len(budgets) else np.array([])
    finite_residuals = residuals[np.isfinite(residuals)]
    max_abs = float(np.max(np.abs(finite_residuals))) if finite_residuals.size else float("nan")
    print(
        f"[{mode_tag}] verify02: {len(budgets)} budget rows ({ok} ok); "
        f"max |closure_residual| = {max_abs:.3e}; {len(status)} skipped cases",
        flush=True,
    )


if __name__ == "__main__":
    output_dir = run(CONFIG)
    print(f"[{CONFIG.mode.upper()}] verify02 done -> {output_dir}", flush=True)
