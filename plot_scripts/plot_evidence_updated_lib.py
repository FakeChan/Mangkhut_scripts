"""plot_evidence_updated:三张科研汇报图(图 1/2/3)与可选图 4 的数据聚合与
质量筛选纯函数库。

- 无 matplotlib、无文件 I/O 副作用:所有函数只做 pandas 变换,可独立测试;
- 统计顺序(任务书要求):先在成员内对时次等权平均,再跨成员等权平均;
  不用平均 RMSE 之比替代平均配对改善率;
- 质量筛选定义复用 diagnostic_reports/20261006_verify_diag_rerun_review/
  analysis_d03/analyze_d03.py 的字段与阈值,不放宽、不修改:
    联合筛选(joint)= 目标子集内弱端/强端重构残差、重构变化与实际变化之差
      的 RMS 均 <= 实际变化 RMS × 0.5(joint_status ==
      residual_small_vs_explained_diff);
    保守交集(conservative)= 联合筛选 且 同区域 Phase A(强、弱)均合格
      (quality_flag==Q 已包含最大绝对残差/实际变化 RMS<=5 的条件);
  本模块不重新计算残差,只消费 verify_03 已输出的 joint_status /
  attribution_flag / a_both_pass 与 analyze_d03 的 11/12 汇总表。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# =====================
# 窗口与常量
# =====================
WINDOWS = (("early_0p5_2h", 0.5, 2.0), ("late_3_6h", 3.0, 6.0))

#: 诊断三质量标记(与 analyze_d03.py 的 Q/FAIL 常量一致)
QUALITY_PASS = "residual_small_vs_explained_diff"
QUALITY_FAIL = "residual_not_negligible_attribution_requires_verification"

#: 联合表 gate 名称(analyze_d03.py 的 12_report_focus_metrics.csv gate 列)
GATE_JOINT = "joint_pass"
GATE_CONSERVATIVE = "joint_pass_and_regionA_both_pass"

#: 主区域(NR 中心 0–300 km 海洋并集)
REGION_MAIN = "r000_300"

#: RMS 比阈值(联合筛选;来自 verify_03 的 PHASE_A_RMSE_RATIO_MAX)
JOINT_RATIO_MAX = 0.5

METHOD_COLORS = {"EAKF": "#42949E", "QCF_RHF": "#0F4D92"}
METHOD_LABELS = {"EAKF": "EAKF", "QCF_RHF": "QCF-RHF"}


# =====================
# 通用聚合
# =====================
def add_window_column(df: pd.DataFrame, time_col: str = "time_hour",
                      windows=WINDOWS) -> pd.DataFrame:
    """追加 window 列:窗口定义外的时次(如 0 h、2.5 h)window 为 NaN,
    保留在表中供全时次使用,但不参与早晚窗口汇总。"""
    out = df.copy()
    assigned = pd.Series(pd.NA, index=out.index)
    for name, lo, hi in windows:
        assigned.loc[out[time_col].between(lo, hi)] = name
    out["window"] = assigned
    return out


def member_time_then_cross_mean(
    df: pd.DataFrame,
    value_cols,
    group_cols,
    member_col: str = "member",
    time_col: str = "time_hour",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """两段等权聚合(任务书统计顺序)。

    第一段:按 (group_cols, member) 分组对 value_cols 求均值(成员内对
    可用时次等权),并记录每成员的时次数 n_times_used;
    第二段:跨成员等权平均,并记录 n_members、min/max 时次数。

    group_cols 不应包含 time_col 与 member_col。
    返回 (result, per_member) 两张表。
    """
    group_cols = tuple(group_cols)
    value_cols = tuple(value_cols)
    key = list(group_cols) + [member_col]
    g = df.groupby(key, dropna=False)
    per_member = g[list(value_cols)].mean().reset_index()
    times = g.size().reset_index(name="n_times_used")
    per_member = per_member.merge(times, on=key)
    agg = per_member.groupby(list(group_cols), dropna=False).agg(
        **{**{f"mean_member_{c}": (c, "mean") for c in value_cols},
           "n_members": (member_col, "nunique"),
           "min_member_times_used": ("n_times_used", "min"),
           "max_member_times_used": ("n_times_used", "max")}
    ).reset_index()
    return agg, per_member


# =====================
# 图 1:相对 RMSE 改善率(诊断一)
# =====================
def skill_member_timeseries(
    member_metrics: pd.DataFrame,
    region: str = REGION_MAIN,
    value_col: str = "rmse_improvement_pct",
    status_ok: str = "ok",
) -> pd.DataFrame:
    """图 1 细线数据:逐成员逐时次的配对相对 RMSE 改善率(长表)。

    误差参照为 NR;status 非 ok 的行剔除。
    """
    df = member_metrics.copy()
    df = df[df.region.eq(region) & df.status.eq(status_ok)]
    keep = ["method", "member", "time_hour", "variable", value_col,
            "unit", "status"]
    return df[[c for c in keep if c in df.columns]].sort_values(
        ["variable", "method", "member", "time_hour"]
    )


def skill_summary_by_window(
    member_metrics: pd.DataFrame,
    region: str = REGION_MAIN,
    value_col: str = "rmse_improvement_pct",
    windows=WINDOWS,
    status_ok: str = "ok",
) -> pd.DataFrame:
    """图 1 窗口汇总:成员内时间平均后跨成员等权的改善率。

    返回列:window, method, variable, mean_member_rmse_improvement_pct,
    n_members。窗口外时次(如 0 h)不参与。
    """
    df = member_metrics.copy()
    df = df[df.region.eq(region) & df.status.eq(status_ok) & df[value_col].notna()]
    df = add_window_column(df, windows=windows)
    df = df[df.window.notna()]
    if df.empty:
        return pd.DataFrame(columns=["window", "method", "variable",
                                     f"mean_member_{value_col}", "n_members"])
    per_member = (
        df.groupby(["window", "method", "variable", "member"], sort=True)
        [value_col].mean().reset_index()
    )
    result = (
        per_member.groupby(["window", "method", "variable"], sort=True)
        [value_col].mean().reset_index(name=f"mean_member_{value_col}")
    )
    counts = (
        per_member.groupby(["window", "method", "variable"], sort=True)
        ["member"].nunique().reset_index(name="n_members")
    )
    return result.merge(counts, on=["window", "method", "variable"])


# =====================
# 图 2:实际通量误差预算(诊断二条件预算)
# =====================
def budget_summary_by_window(
    conditional_budget: pd.DataFrame,
    region: str = REGION_MAIN,
    subsets=("all_common", "sst_improved"),
    status_ok: str = "ok",
    windows=WINDOWS,
) -> pd.DataFrame:
    """图 2 汇总:ΔMSE=C+S 的早晚对比,成员内时间平均后跨成员等权。

    输入为 verify02_conditional_budget.csv(逐成员逐时次逐子集)。
    返回列:window, method, variable, subset,
      mean_member_delta_mse_direct / mean_member_cross_term /
      mean_member_increment_square_term(逐行满足 ΔMSE=C+S)、
      n_members、min/max_member_times_used。
    非 ok 行与窗口外时次不参与;空子集自然缺失,不填零。
    """
    df = conditional_budget.copy()
    df = df[df.region.eq(region) & df.status.eq(status_ok) & df.subset.isin(subsets)]
    value_cols = ("delta_mse_direct", "cross_term", "increment_square_term")
    key = ["window", "method", "variable", "subset", "member"]
    g = win_assign(df, windows).groupby(key, dropna=False)
    per_member = g[list(value_cols)].mean().reset_index()
    times = g.size().reset_index(name="n_times_used")
    per_member = per_member.merge(times, on=key)
    agg = per_member.groupby(
        ["window", "method", "variable", "subset"], sort=True
    ).agg(
        mean_member_delta_mse_direct=("delta_mse_direct", "mean"),
        mean_member_cross_term=("cross_term", "mean"),
        mean_member_increment_square_term=("increment_square_term", "mean"),
        n_members=("member", "nunique"),
        min_member_times_used=("n_times_used", "min"),
        max_member_times_used=("n_times_used", "max"),
    ).reset_index()
    return agg


def win_assign(df: pd.DataFrame, windows=WINDOWS, time_col: str = "time_hour"):
    """按窗口过滤并打 window 标签(用于窗口汇总输入)。"""
    parts = []
    for name, lo, hi in windows:
        d = df[df[time_col].between(lo, hi)].copy()
        d["window"] = name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# =====================
# 图 3:离线 SST 响应与质量覆盖(诊断三)
# =====================
#: 图三的显式「方法 × 窗口」分组顺序:覆盖表、联合指标表、保守指标表
#: 全部按此键重排后再生成横轴标签,禁止各表分别排序后直接转数组。
FIG3_GROUP_KEYS = (
    ("EAKF", "early_0p5_2h"),
    ("QCF_RHF", "early_0p5_2h"),
    ("EAKF", "late_3_6h"),
    ("QCF_RHF", "late_3_6h"),
)
FIG3_METHOD_LABELS = {"EAKF": "EAKF", "QCF_RHF": "QCF-RHF"}
FIG3_WINDOW_LABELS = {"early_0p5_2h": "early", "late_3_6h": "late"}


def align_to_group_keys(frame: pd.DataFrame, keys, key_cols) -> tuple:
    """按显式分组键重排 frame;缺失组合保留为全 NaN 行并单独报告,重复键报错。

    返回 (aligned, missing):aligned 与 keys 行序一致(key 列在前,已
    reset_index);missing 是 frame 中不存在的键列表。缺失组合不填零,
    由调用方以「缺失」而非 0 呈现。
    """
    cols = list(key_cols)
    duplicated = frame.duplicated(cols, keep=False)
    if duplicated.any():
        seen = frame.loc[duplicated, cols].drop_duplicates()
        raise ValueError(f"duplicate group keys: {seen.to_dict('records')}")
    present = set(map(tuple, frame[cols].itertuples(index=False, name=None)))
    missing = [tuple(key) for key in keys if tuple(key) not in present]
    index = pd.MultiIndex.from_tuples([tuple(key) for key in keys], names=cols)
    aligned = frame.set_index(cols).reindex(index)
    return aligned.reset_index(), missing


def fig3_panel_data(coverage: pd.DataFrame, metrics_joint: pd.DataFrame,
                    metrics_conservative: pd.DataFrame,
                    keys=FIG3_GROUP_KEYS) -> dict:
    """图三三面板的数值准备(纯函数,不绘图):三表按同一显式键重排。

    coverage 为 lib.focus_coverage 输出(含 n_rows/n_pass/
    n_joint_and_regionA_pass);metrics_* 为 12 表按 gate 过滤后的行。
    返回等长 numpy 数组与标签;任何表中缺失的组合在数值上为 NaN 并列入
    missing,不填零。
    """
    cov, cov_missing = align_to_group_keys(coverage, keys, ["method", "window"])
    met_j, mj_missing = align_to_group_keys(
        metrics_joint, keys, ["method", "window"])
    met_c, mc_missing = align_to_group_keys(
        metrics_conservative, keys, ["method", "window"])
    missing = sorted(set(cov_missing) | set(mj_missing) | set(mc_missing))
    labels = [
        f"{FIG3_METHOD_LABELS[method]}\n{FIG3_WINDOW_LABELS[window]}"
        for method, window in keys
    ]
    return {
        "keys": tuple(keys),
        "labels": labels,
        "missing": missing,
        "joint_pass": cov["n_pass"].to_numpy(dtype=float),
        "joint_den": cov["n_rows"].to_numpy(dtype=float),
        "cons_pass": cov["n_joint_and_regionA_pass"].to_numpy(dtype=float),
        "cons_den": cov["n_rows"].to_numpy(dtype=float),
        "cons_members": met_c["n_members"].to_numpy(dtype=float),
        "ratio_joint": met_j["mean_member_sst_to_actual_rms_ratio"].to_numpy(dtype=float),
        "ratio_conservative": met_c["mean_member_sst_to_actual_rms_ratio"].to_numpy(dtype=float),
        "csst_joint": met_j["mean_member_cross_model_error_sst_response"].to_numpy(dtype=float),
        "csst_conservative": met_c["mean_member_cross_model_error_sst_response"].to_numpy(dtype=float),
    }


# =====================
# 图 1/2 展示辅助(纯函数)
# =====================
def auto_ylim(values, pad_frac: float = 0.08, include_zero: bool = True) -> tuple:
    """覆盖全部有限值的纵轴范围(含零参考线),两端留 pad_frac 相对边距。

    空输入返回占位 (-1.0, 1.0)。用于替代固定 ylim,避免截断成员曲线;
    手动配置的轴不经过本函数。
    """
    vals = np.asarray(
        [v for v in np.ravel(values) if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(vals.min()), float(vals.max())
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = hi - lo
    if span <= 0.0:
        span = max(abs(hi), 1.0e-6)
    return (lo - pad_frac * span, hi + pad_frac * span)


def fig2_panel_map(axes):
    """2x2 轴数组 -> 图二独立面板导出名(左上/右上/左下/右下按真实内容)。

    布局约定:axes[0][0]=HFX early,axes[0][1]=HFX late,
    axes[1][0]=LH early,axes[1][1]=LH late。导出名与内容一一对应,
    不再依赖 fig.axes 数字索引。
    """
    return {
        "u2_hfx_early": axes[0][0],
        "u2_hfx_late": axes[0][1],
        "u2_lh_early": axes[1][0],
        "u2_lh_late": axes[1][1],
    }


def fig2_uniform_members(budget: pd.DataFrame, expected: int = 6) -> bool:
    """图二全部「窗口×方法×变量×子集」组是否都为 expected 个成员。

    全部为 6 时可在整图图注统一说明成员数;否则需在组标签中逐组标注。
    """
    sub = budget[budget.variable.isin(("hfx", "lh"))]
    if sub.empty:
        return False
    counts = sub.groupby(["window", "method", "variable", "subset"])[
        "n_members"].first()
    return bool((counts == expected).all())


def focus_coverage(focus_quality: pd.DataFrame,
                   windows=("early_0p5_2h", "late_3_6h"),
                   variables=("hfx", "lh"),
                   methods=("EAKF", "QCF_RHF"),
                   subset: str = "sst_improved") -> pd.DataFrame:
    """图 3A 数据:联合与保守交集的通过案例数(来自 11 表)。

    11 表(analysis_d03/11_report_focus_quality.csv)逐行含 n_rows(期望
    案例)、n_pass(联合筛选通过)、n_joint_and_regionA_pass(保守交集通过,
    含 Phase A 双合格与最大绝对残差条件)。
    """
    q = focus_quality[focus_quality.window.isin(windows)
                      & focus_quality.variable.isin(variables)
                      & focus_quality.method.isin(methods)
                      & focus_quality.subset.eq(subset)]
    return q[["window", "method", "variable", "subset", "n_rows", "n_pass",
              "n_fail", "n_joint_and_regionA_pass"]].sort_values(
        ["window", "method", "variable"]
    )


def focus_metrics_by_gate(focus_metrics: pd.DataFrame,
                          gates=(GATE_JOINT, GATE_CONSERVATIVE),
                          windows=("early_0p5_2h", "late_3_6h"),
                          variables=("hfx", "lh"),
                          methods=("EAKF", "QCF_RHF")) -> pd.DataFrame:
    """图 3B/C 数据:12 表(已限定 r000_300 + sst_improved + 早晚窗口)
    按两档 gate 过滤,含 RMS 比、C_SST、同号率与参与比例。"""
    m = focus_metrics[focus_metrics.window.isin(windows)
                      & focus_metrics.variable.isin(variables)
                      & focus_metrics.method.isin(methods)
                      & focus_metrics.gate.isin(gates)]
    return m.sort_values(["window", "method", "variable", "gate"])
