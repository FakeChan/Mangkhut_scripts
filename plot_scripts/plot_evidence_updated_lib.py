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
