"""Summarize the local strong/weak forecast-skill cache.

The independent resampling unit is one method/member paired case. Time is
averaged within case before bootstrap resampling, preserving temporal dependence.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# =====================
# User configuration
# =====================
CACHE_DIR = Path(
    os.environ.get(
        "OMTMP_CACHE_DIR",
        Path(__file__).resolve().parent / "omtmp_skill_cache",
    )
)
BOOTSTRAP_SAMPLES = 20_000
RANDOM_SEED = 20260804
WINDOWS = {
    "early_0p5_2h": (0.5, 2.0),
    "late_3_6h": (3.0, 6.0),
    "all_0p5_6h": (0.5, 6.0),
}


def bootstrap_mean_ci(values, samples=BOOTSTRAP_SAMPLES, seed=RANDOM_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, (0.025, 0.975)))


def summarize_case_values(case_values, group_columns, value_columns):
    rows = []
    grouped = case_values.groupby(group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_cases"] = group[["method", "member"]].drop_duplicates().shape[0]
        for value in value_columns:
            values = group[value].to_numpy(float)
            low, high = bootstrap_mean_ci(values)
            row[f"mean_{value}"] = float(np.nanmean(values))
            row[f"median_{value}"] = float(np.nanmedian(values))
            row[f"win_fraction_{value}"] = float(np.nanmean(values > 0.0))
            row[f"ci025_{value}"] = low
            row[f"ci975_{value}"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def add_windows(frame):
    parts = []
    for name, (start, stop) in WINDOWS.items():
        selected = frame[(frame.time_hour >= start) & (frame.time_hour <= stop)].copy()
        selected["window"] = name
        parts.append(selected)
    return pd.concat(parts, ignore_index=True)


def metric_summary(metrics):
    windowed = add_windows(metrics)
    case_columns = [
        "window", "method", "member", "region", "mask_type", "variable", "unit"
    ]
    averaged = windowed.groupby(case_columns, as_index=False).agg(
        strong_rmse=("strong_rmse", "mean"),
        weak_rmse=("weak_rmse", "mean"),
        rmse_improvement=("rmse_improvement", "mean"),
        mae_improvement=("mae_improvement", "mean"),
        bias_abs_improvement=("bias_abs_improvement", "mean"),
    )
    averaged["rmse_improvement_pct"] = (
        100.0 * averaged.rmse_improvement / averaged.weak_rmse
    )
    all_methods = averaged.copy()
    all_methods["method_group"] = "ALL"
    by_method = averaged.copy()
    by_method["method_group"] = by_method.method
    combined = pd.concat((all_methods, by_method), ignore_index=True)
    group_columns = [
        "window", "method_group", "region", "mask_type", "variable", "unit"
    ]
    return summarize_case_values(
        combined,
        group_columns,
        [
            "rmse_improvement", "rmse_improvement_pct", "mae_improvement",
            "bias_abs_improvement", "strong_rmse", "weak_rmse",
        ],
    )


def vertical_summary(vertical):
    windowed = add_windows(vertical)
    case_columns = [
        "window", "method", "member", "region", "mask_type", "variable", "unit", "level"
    ]
    averaged = windowed.groupby(case_columns, as_index=False).agg(
        strong_rmse=("strong_rmse", "mean"),
        weak_rmse=("weak_rmse", "mean"),
        rmse_improvement=("rmse_improvement", "mean"),
    )
    averaged["rmse_improvement_pct"] = 100.0 * averaged.rmse_improvement / averaged.weak_rmse
    all_methods = averaged.copy()
    all_methods["method_group"] = "ALL"
    by_method = averaged.copy()
    by_method["method_group"] = by_method.method
    combined = pd.concat((all_methods, by_method), ignore_index=True)
    return summarize_case_values(
        combined,
        ["window", "method_group", "region", "mask_type", "variable", "unit", "level"],
        ["rmse_improvement", "rmse_improvement_pct", "strong_rmse", "weak_rmse"],
    )


def intensity_summary(intensity):
    windowed = add_windows(intensity)
    values = [
        "track_improvement_km", "min_psfc_improvement_hpa", "max_ws10_improvement"
    ]
    averaged = windowed.groupby(["window", "method", "member"], as_index=False)[values].mean()
    all_methods = averaged.copy()
    all_methods["method_group"] = "ALL"
    by_method = averaged.copy()
    by_method["method_group"] = by_method.method
    return summarize_case_values(
        pd.concat((all_methods, by_method), ignore_index=True),
        ["window", "method_group"],
        values,
    )


def fmt(value, digits=4):
    return "nan" if not np.isfinite(value) else f"{value:.{digits}g}"


def build_findings(summary, vertical, intensity, blocks):
    primary = summary[
        (summary.window == "early_0p5_2h")
        & (summary.method_group == "ALL")
        & (summary.region == "r000_300")
        & (summary.mask_type == "ocean")
    ].set_index("variable")
    lines = [
        "# Strong-versus-weak skill findings",
        "",
        "Positive improvement is `RMSE_weak - RMSE_strong`; positive values favor strong coupling.",
        "The 95% intervals bootstrap 12 paired method/member cases after averaging time within each case.",
        "",
        "## Early pathway window (0.5–2 h, ocean, 0–300 km)",
        "",
        "| Variable | Mean improvement | Relative | Case win fraction | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for variable in ("om", "tsk", "hfx", "lh", "t2", "q2", "theta_l1", "qv_l1"):
        row = primary.loc[variable]
        lines.append(
            f"| {variable} | {fmt(row.mean_rmse_improvement)} {row.unit} | "
            f"{fmt(row.mean_rmse_improvement_pct, 3)}% | "
            f"{row.win_fraction_rmse_improvement:.0%} | "
            f"[{fmt(row.ci025_rmse_improvement)}, {fmt(row.ci975_rmse_improvement)}] |"
        )

    method_rows = summary[
        (summary.window == "early_0p5_2h")
        & (summary.region == "r000_300")
        & (summary.mask_type == "ocean")
        & summary.variable.isin(["om", "tsk", "q2", "qv_l1"])
        & summary.method_group.isin(["QCF_RHF", "EAKF"])
    ]
    lines += ["", "## Method split", "", "| Method | Variable | Relative RMSE improvement | Win fraction |", "|---|---|---:|---:|"]
    for row in method_rows.sort_values(["method_group", "variable"]).itertuples():
        lines.append(
            f"| {row.method_group} | {row.variable} | {fmt(row.mean_rmse_improvement_pct, 3)}% | "
            f"{row.win_fraction_rmse_improvement:.0%} |"
        )

    vertical_low = vertical[
        (vertical.window == "early_0p5_2h")
        & (vertical.method_group == "ALL")
        & (vertical.region == "r000_300")
        & (vertical.mask_type == "ocean")
        & (vertical.level <= 5)
    ]
    lines += ["", "## Lowest five model levels", "", "| Variable | Level | Relative RMSE improvement | Win fraction |", "|---|---:|---:|---:|"]
    for row in vertical_low.sort_values(["variable", "level"]).itertuples():
        lines.append(
            f"| {row.variable} | {int(row.level)} | {fmt(row.mean_rmse_improvement_pct, 3)}% | "
            f"{row.win_fraction_rmse_improvement:.0%} |"
        )

    intensity_early = intensity[
        (intensity.window == "early_0p5_2h") & (intensity.method_group == "ALL")
    ].iloc[0]
    lines += [
        "",
        "## TC intensity/track (0.5–2 h)",
        "",
        f"- Track-error improvement: {fmt(intensity_early.mean_track_improvement_km)} km; "
        f"win fraction {intensity_early.win_fraction_track_improvement_km:.0%}.",
        f"- Minimum-PSFC absolute-error improvement: {fmt(intensity_early.mean_min_psfc_improvement_hpa)} hPa; "
        f"win fraction {intensity_early.win_fraction_min_psfc_improvement_hpa:.0%}.",
        f"- Maximum-10m-wind absolute-error improvement: {fmt(intensity_early.mean_max_ws10_improvement)} m s-1; "
        f"win fraction {intensity_early.win_fraction_max_ws10_improvement:.0%}.",
    ]

    forecast_blocks = blocks[blocks.time_hour > 0.0]
    for branch in ("s", "w"):
        forecast_blocks[f"om_tsk_{branch}"] = (
            forecast_blocks[f"om_{branch}"] - forecast_blocks[f"tsk_{branch}"]
        )
    lines += [
        "",
        "## OM_TMP to TSK handoff",
        "",
        f"- Strong blockwise RMS(OM_TMP0-TSK), t>0: "
        f"{np.sqrt(np.nanmean(forecast_blocks.om_tsk_s**2)):.6g} K.",
        f"- Weak blockwise RMS(OM_TMP0-TSK), t>0: "
        f"{np.sqrt(np.nanmean(forecast_blocks.om_tsk_w**2)):.6g} K.",
        "",
        "## Interpretation",
        "",
        "The strong experiment has a robust ocean/skin-temperature skill advantage. "
        "A small and mostly sign-consistent moisture advantage appears in Q2/QVAPOR, but heat/moisture flux RMSE does not improve. "
        "Temperature, circulation, intensity, and track advantages are weak or inconsistent. "
        "Thus the OM_TMP→TSK handoff is active, while the beneficial part of the signal is strongly attenuated at the flux and atmospheric-response steps.",
    ]
    return "\n".join(lines) + "\n"


def main():
    metrics = pd.read_csv(CACHE_DIR / "omtmp_skill_metrics.csv", dtype={"member": str})
    vertical = pd.read_csv(CACHE_DIR / "omtmp_skill_vertical.csv", dtype={"member": str})
    intensity = pd.read_csv(CACHE_DIR / "omtmp_skill_intensity.csv", dtype={"member": str})
    blocks = pd.read_csv(CACHE_DIR / "omtmp_skill_blocks.csv", dtype={"member": str})

    summary = metric_summary(metrics)
    vertical_out = vertical_summary(vertical)
    intensity_out = intensity_summary(intensity)
    summary.to_csv(CACHE_DIR / "omtmp_skill_summary.csv", index=False)
    vertical_out.to_csv(CACHE_DIR / "omtmp_skill_vertical_summary.csv", index=False)
    intensity_out.to_csv(CACHE_DIR / "omtmp_skill_intensity_summary.csv", index=False)
    (CACHE_DIR / "omtmp_skill_findings.md").write_text(
        build_findings(summary, vertical_out, intensity_out, blocks), encoding="utf-8"
    )
    print(f"wrote {len(summary)} metric, {len(vertical_out)} vertical, {len(intensity_out)} intensity summaries")


if __name__ == "__main__":
    main()
