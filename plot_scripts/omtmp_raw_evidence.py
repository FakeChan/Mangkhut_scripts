"""从原始 wrfout 直接计算 OM_TMP 证据图(图 1/2/3 + 可选图 4)的全部输入表。

链路(全程内存 DataFrame,不依赖任何诊断 CSV):

    wrfout(强/弱试验 + NR)
      -> verify_diag 三个诊断的 compute() 内存入口
           verify_01:逐成员逐时次 RMSE 改善率与 TSK-通量逐点分类
           verify_02:通量误差预算 ΔMSE = C + S(全域与条件子集)
           verify_03:固定大气 SST 替换(阶段 A 重建 + 阶段 B 条件替换 + 联合表)
      -> analyze_d03 逻辑移植(joint 表质量分层与 11/12 焦点汇总;
         原脚本 diagnostic_reports/20261006_verify_diag_rerun_review/
         analysis_d03/analyze_d03.py 的忠实移植,仅去掉文件读写)
      -> 图 1/2/3 的 member_metrics / conditional / focus_quality / focus_metrics

运行成本:真实模式完整计算约 2 小时(诊断三离线通量重建占绝大部分)。
结果以 pickle 缓存(非 CSV)存放于 cache_dir,CACHE_POLICY 控制:
    "auto"    缓存存在且指纹一致则复用,否则重算(默认)
    "refresh" 强制重算并覆盖缓存
    "reuse"   只读缓存,缺失或指纹不符即报错
指纹由运行模式、三个诊断的配置快照与相关源码文件的尺寸/修改时间构成,
任何配置或代码改动都会自动失效旧缓存。
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
VERIFY_DIAG_DIR = SCRIPT_DIR / "verify_diag"
if str(VERIFY_DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFY_DIAG_DIR))

import verify_common as vc  # noqa: E402
import verify_01_skill_timeseries as v1  # noqa: E402
import verify_02_flux_error_budget as v2  # noqa: E402
import verify_03_fixed_atmosphere_flux as v3  # noqa: E402


# =====================================================================
# analyze_d03 移植(与原脚本逐行对应;Q/FAIL/窗口/键定义保持不变)
# =====================================================================
#: 诊断三质量标记(与 plot_evidence_updated_lib.QUALITY_PASS/FAIL 一致)
QUALITY_PASS = "residual_small_vs_explained_diff"
QUALITY_FAIL = "residual_not_negligible_attribution_requires_verification"
#: 主连接键(joint 与 verify02 条件预算按此对应)
D3_KEY = ["method", "member", "time_hour", "region", "variable"]
#: analyze_d03 的窗口定义(比绘图的早晚窗口多两个诊断窗口)
D3_WINDOWS = [
    ("post_initial_0p5_6h", 0.5, 6),
    ("early_0p5_2h", 0.5, 2),
    ("transition_2p5h", 2.5, 2.5),
    ("late_3_6h", 3, 6),
]


def _d3_win(frame: pd.DataFrame) -> pd.DataFrame:
    """按 D3_WINDOWS 打窗口标签(analyze_d03.win 的移植)。"""
    parts = []
    for name, lo, hi in D3_WINDOWS:
        d = frame[frame.time_hour.between(lo, hi)].copy()
        d["window"] = name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _d3_member_mean(group: pd.DataFrame, col: str) -> float:
    """先成员内等权、再跨成员等权(analyze_d03.member_mean 的移植)。"""
    return group.groupby("member")[col].mean().mean()


def _d3_quality(frame: pd.DataFrame, keys, flag: str) -> pd.DataFrame:
    """质量覆盖率汇总(analyze_d03.quality 的移植)。"""
    rows = []
    for key, g in frame.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        passed = g[flag].eq(QUALITY_PASS)
        failed = g[flag].eq(QUALITY_FAIL)
        den = int((passed | failed).sum())
        r = dict(zip(keys, key))
        r.update(
            n_rows=len(g), n_pass=int(passed.sum()), n_fail=int(failed.sum()),
            n_unevaluable_or_empty=int((~(passed | failed)).sum()),
            n_evaluable=den,
            pass_fraction_evaluable=float(passed.sum() / den) if den else np.nan,
            n_members=g.member.nunique(),
        )
        if "n_subset" in g:
            r.update(
                n_numeric_valid=int(g.status.eq("ok").sum()),
                mean_n_subset=_d3_member_mean(g, "n_subset"),
                mean_subset_fraction=_d3_member_mean(g, "subset_fraction_of_all_common"),
                n_phaseA_region_both_pass=int(g.a_both_pass.sum()),
                n_joint_and_regionA_pass=int((passed & g.a_both_pass).sum()),
            )
        rows.append(r)
    return pd.DataFrame(rows)


#: 12 表的指标列(analyze_d03.metrics;保持原顺序)
D3_METRICS = [
    "cross_model_error_sst_response", "rms_sst_response", "rms_actual_response",
    "rms_sst_minus_actual", "sst_to_actual_rms_ratio", "sign_agreement",
    "sign_eligible_fraction", "actual_mse_weak", "actual_mse_strong",
    "actual_delta_mse_direct", "actual_cross_term",
    "actual_increment_square_term", "actual_closure_residual",
    "attr_weak_rmse_ratio", "attr_strong_rmse_ratio", "attr_diff_agree_ratio",
    "subset_fraction_of_all_common",
]


def analyze_d03_tables(
    phase_a: pd.DataFrame,
    phase_b: pd.DataFrame,
    joint: pd.DataFrame,
    conditional: pd.DataFrame,
) -> dict:
    """analyze_d03.py 的忠实移植:输入四张原始表,输出全部汇总表(无文件写入)。

    phase_a / phase_b / joint 来自 verify_03,conditional 来自 verify_02
    (与原脚本读取的四张 CSV 逐列一致)。返回 dict,关键键:
      focus_quality / focus_metrics(11/12 焦点表,图 3 直接消费)、
      integrity(完整性核验,原 integrity_and_mask_check.json)。
    """
    a, b, j, c = phase_a, phase_b, joint, conditional

    ar = (
        a.assign(a_pass=a.quality_flag.eq(QUALITY_PASS))
        .groupby(D3_KEY)
        .agg(
            a_both_pass=("a_pass", "all"),
            phaseA_max_abs_ratio=("residual_maxabs_to_rms_ratio", "max"),
        )
        .reset_index()
    )
    j = j.merge(ar, on=D3_KEY, validate="many_to_one")
    b = b.merge(ar, on=D3_KEY, validate="one_to_one")

    qa = _d3_quality(_d3_win(a), ["window", "method", "experiment", "variable", "region"], "quality_flag")
    qb = _d3_quality(_d3_win(b), ["window", "method", "variable", "region"], "attribution_flag")
    qj = _d3_quality(_d3_win(j), ["window", "method", "variable", "region", "subset"], "joint_status")
    qt = _d3_quality(j, ["time_hour", "method", "variable", "region", "subset"], "joint_status")

    rows = []
    for key, g0 in _d3_win(j).groupby(["window", "method", "variable", "region", "subset"]):
        selectors = {
            "all_numeric": g0.status.eq("ok"),
            "joint_pass": g0.status.eq("ok") & g0.joint_status.eq(QUALITY_PASS),
            "joint_fail": g0.status.eq("ok") & g0.joint_status.eq(QUALITY_FAIL),
            "joint_pass_and_regionA_both_pass": (
                g0.status.eq("ok") & g0.joint_status.eq(QUALITY_PASS) & g0.a_both_pass
            ),
        }
        for gate, selection in selectors.items():
            g = g0[selection]
            r = dict(zip(["window", "method", "variable", "region", "subset"], key))
            r.update(gate=gate, n_rows=len(g), n_members=g.member.nunique())
            per = g.groupby("member").size()
            r.update(
                min_cases_per_present_member=int(per.min()) if len(per) else 0,
                max_cases_per_present_member=int(per.max()) if len(per) else 0,
            )
            for col in D3_METRICS:
                r["mean_member_" + col] = _d3_member_mean(g, col)
                r["mean_case_" + col] = g[col].mean()
            for prefix, col, tol in (
                ("sst_cross", "cross_model_error_sst_response", 1.0),
                ("actual_cross", "actual_cross_term", 1.0),
                ("actual_delta", "actual_delta_mse_direct", 1.0),
            ):
                r[prefix + "_n_positive"] = int((g[col] > tol).sum())
                r[prefix + "_n_negative"] = int((g[col] < -tol).sum())
                r[prefix + "_n_neutral"] = int((g[col].abs() <= tol).sum())
            r["n_finite_ratio"] = int(g.sst_to_actual_rms_ratio.notna().sum())
            r["n_ratio_below_floor"] = int(
                g.actual_response_status.eq("actual_response_below_floor").sum()
            )
            r["n_finite_sign_agreement"] = int(g.sign_agreement.notna().sum())
            r["sum_sign_eligible"] = g.n_sign_eligible.sum()
            r["pooled_sign_agreement_description_only"] = (
                g.n_sign_agree.sum() / g.n_sign_eligible.sum()
                if g.n_sign_eligible.sum() else np.nan
            )
            rows.append(r)
    m = pd.DataFrame(rows)

    qm = (
        _d3_win(j)
        .groupby(["window", "method", "member", "variable", "region", "subset"])
        .agg(
            n_rows=("time_hour", "size"),
            n_quality_pass=("joint_status", lambda x: int(x.eq(QUALITY_PASS).sum())),
            n_quality_fail=("joint_status", lambda x: int(x.eq(QUALITY_FAIL).sum())),
        )
        .reset_index()
    )

    comparison = j.merge(c, on=D3_KEY + ["subset"], suffixes=("_03", "_02"),
                         validate="one_to_one", indicator=True)
    pairs = [
        ("n_subset_03", "n_subset_02"), ("n_common", "n_all_common"),
        ("actual_mse_weak", "mse_weak"), ("actual_mse_strong", "mse_strong"),
        ("actual_delta_mse_direct", "delta_mse_direct"),
        ("actual_cross_term", "cross_term"),
        ("actual_increment_square_term", "increment_square_term"),
        ("actual_closure_residual", "closure_residual"),
    ]
    comp = comparison[D3_KEY + ["subset", "_merge"]].copy()
    check = {}
    for x, y in pairs:
        diff = comparison[x] - comparison[y]
        comp[x + "_minus_02"] = diff
        check[x] = {
            "max_absolute_difference": float(diff.abs().max()),
            "n_equal_within_1e-8": int(
                np.isclose(comparison[x], comparison[y], rtol=1e-12, atol=1e-8,
                           equal_nan=True).sum()
            ),
            "n_rows": len(comparison),
        }

    rows = []
    for key, g in a.groupby(["time_hour", "method", "variable", "region"]):
        r = dict(zip(["time_hour", "method", "variable", "region"], key))
        r.update(
            n_rows=len(g),
            mean_reconstruction_rmse=g.recon_minus_model_rmse.mean(),
            mean_reconstruction_bias=g.recon_minus_model_bias.mean(),
            max_reconstruction_maxabs=g.recon_minus_model_maxabs.max(),
            mean_actual_response_rms=g.model_strong_minus_weak_rms.mean(),
            mean_total_response_mismatch_rms=g.residual_between_diffs_rms.mean(),
            max_reconstruction_rmse=g.recon_minus_model_rmse.max(),
        )
        rows.append(r)
    absolute = pd.DataFrame(rows)

    rows = []
    for key, g0 in _d3_win(b).groupby(["window", "method", "variable", "region"]):
        for gate, sel in (
            ("all_numeric", g0.status.eq("ok")),
            ("phaseB_pass", g0.attribution_flag.eq(QUALITY_PASS)),
            ("phaseA_B_both_pass", g0.attribution_flag.eq(QUALITY_PASS) & g0.a_both_pass),
        ):
            g = g0[sel]
            r = dict(zip(["window", "method", "variable", "region"], key))
            r.update(gate=gate, n_rows=len(g), n_members=g.member.nunique())
            for col in [
                "dF_SST", "dF_rest", "dF_reconstructed", "dF_SST_reverse",
                "path_split_difference", "actual_model_diff_mean",
                "dF_total_minus_actual_rms", "Fww_minus_nr_rmse",
                "Fsw_minus_nr_rmse", "sst_only_rmse_improvement_pct",
            ]:
                r["mean_member_" + col] = _d3_member_mean(g, col)
            r["max_abs_split_closure"] = g.split_closure_residual.abs().max()
            r["mean_abs_path_split_difference"] = g.path_split_difference.abs().mean()
            rows.append(r)
    components = pd.DataFrame(rows)

    check.update(
        dict(
            n_phaseA=len(a), n_phaseB=len(b), n_joint=len(j),
            phaseA_status=a.status.value_counts().to_dict(),
            phaseB_status=b.status.value_counts().to_dict(),
            joint_status=j.status.value_counts().to_dict(),
            n_duplicate_joint_keys=int(j.duplicated(D3_KEY + ["subset"]).sum()),
            max_n_nonfinite_output=int(j.n_nonfinite_output.max()),
            max_actual_budget_closure=float(j.actual_closure_residual.abs().max()),
            max_phaseB_split_closure=float(b.split_closure_residual.abs().max()),
            n_phaseB_pass_but_phaseA_not_both_pass=int(
                ((b.time_hour > 0) & b.attribution_flag.eq(QUALITY_PASS) & ~b.a_both_pass).sum()
            ),
        )
    )

    initialization = j[j.time_hour.eq(0)][
        D3_KEY + ["subset", "n_subset", "status", "actual_response_status",
                  "joint_status", "rms_sst_response", "rms_actual_response",
                  "cross_model_error_sst_response", "actual_delta_mse_direct"]
    ]

    # 报告焦点表(11/12):r000_300、sst_improved、早晚窗口
    focus_quality = qj[
        qj.region.eq("r000_300")
        & qj.subset.isin(["all_common", "sst_improved"])
        & qj.window.isin(["early_0p5_2h", "late_3_6h"])
    ][["window", "method", "variable", "subset", "n_rows", "n_pass", "n_fail",
       "n_joint_and_regionA_pass"]]
    focus_metrics = m[
        m.region.eq("r000_300")
        & m.subset.eq("sst_improved")
        & m.window.isin(["early_0p5_2h", "late_3_6h"])
        & m.gate.isin(["joint_pass", "joint_pass_and_regionA_both_pass"])
    ][["window", "method", "variable", "gate", "n_rows", "n_members",
       "mean_member_cross_model_error_sst_response", "sst_cross_n_positive",
       "sst_cross_n_negative", "mean_member_sst_to_actual_rms_ratio",
       "mean_member_sign_agreement", "mean_member_sign_eligible_fraction",
       "mean_member_actual_cross_term", "mean_member_actual_delta_mse_direct"]]

    return {
        "phaseA_quality_by_window": qa,
        "phaseB_quality_by_window": qb,
        "joint_quality_by_window": qj,
        "joint_quality_by_time": qt,
        "metrics_stratified": m,
        "joint_coverage_by_member": qm,
        "joint_vs_diagnostic02_budget_comparison": comp,
        "phaseA_absolute_residuals_by_time": absolute,
        "phaseB_components_by_window": components,
        "initialization_rows": initialization,
        "focus_quality": focus_quality.reset_index(drop=True),
        "focus_metrics": focus_metrics.reset_index(drop=True),
        "integrity": check,
    }


# =====================================================================
# 从原始 wrfout 计算(verify_diag compute() 串联)
# =====================================================================
def _evidence_configs(mode: str) -> tuple:
    """按目标模式覆盖三个诊断的 CONFIG(mode 之外全部沿用脚本默认)。"""
    return (
        replace(v1.CONFIG, mode=mode),
        replace(v2.CONFIG, mode=mode),
        replace(v3.CONFIG, mode=mode),
    )


def compute_raw_evidence(mode: str = "real", verbose: bool = True) -> dict:
    """从原始 wrfout(或合成数据)一路算到图 1/2/3 所需全部表。

    mode="real" 读服务器/本地真实 wrfout(路径与试验名沿用 verify_diag 默认
    配置);mode="synthetic" 用固定种子合成数据(冒烟测试用,无真实 I/O)。
    """
    cfg1, cfg2, cfg3 = _evidence_configs(mode)
    tag = mode.upper()

    if verbose:
        print(f"[{tag}] verify_01: member metrics + classification ...", flush=True)
    f1 = v1.compute(cfg1)
    if verbose:
        print(
            f"[{tag}] verify_01 done: {len(f1['metrics'])} metric rows, "
            f"{len(f1['classifications'])} classification rows", flush=True,
        )

    if verbose:
        print(f"[{tag}] verify_02: flux error budget ...", flush=True)
    f2 = v2.compute(cfg2)
    if verbose:
        print(
            f"[{tag}] verify_02 done: {len(f2['budgets'])} budget rows, "
            f"{len(f2['conditional'])} conditional rows", flush=True,
        )

    if verbose:
        print(
            f"[{tag}] verify_03: fixed-atmosphere SST replacement "
            "(real mode may take ~2 h) ...", flush=True,
        )
    f3 = v3.compute(cfg3)
    if verbose:
        print(
            f"[{tag}] verify_03 done: phaseA={len(f3['phase_a'])}, "
            f"phaseB={len(f3['phase_b'])}, joint={len(f3['joint'])} rows", flush=True,
        )

    if verbose:
        print(f"[{tag}] analyze_d03: quality stratification + focus tables ...", flush=True)
    d3 = analyze_d03_tables(f3["phase_a"], f3["phase_b"], f3["joint"], f2["conditional"])
    if verbose:
        print(
            f"[{tag}] analyze_d03 done: integrity "
            f"{json.dumps(d3['integrity'], ensure_ascii=False)[:400]}", flush=True,
        )

    return {
        "mode": mode,
        "member_metrics": f1["metrics"],
        "classifications": f1["classifications"],
        "run_status": {
            "verify01": f1["status"], "verify02": f2["status"], "verify03": f3["status"],
        },
        "budgets": f2["budgets"],
        "conditional": f2["conditional"],
        "phase_a": f3["phase_a"],
        "phase_b": f3["phase_b"],
        "joint": f3["joint"],
        "focus_quality": d3["focus_quality"],
        "focus_metrics": d3["focus_metrics"],
        "d03_tables": d3,
    }


# =====================================================================
# pickle 缓存(指纹 = 模式 + 三个诊断配置快照 + 相关源码文件戳)
# =====================================================================
def _fingerprint(mode: str) -> str:
    payload = {
        "mode": mode,
        "configs": [vc.config_snapshot(cfg) for cfg in _evidence_configs(mode)],
        "code": [],
    }
    for module in (vc, v1, v2, v3):
        path = Path(module.__file__)
        stat = path.stat()
        payload["code"].append([path.name, stat.st_size, int(stat.st_mtime)])
    here = Path(__file__)
    stat = here.stat()
    payload["code"].append([here.name, stat.st_size, int(stat.st_mtime)])
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_raw_evidence(
    cache_dir: Path | str,
    cache_policy: str = "auto",
    mode: str = "real",
) -> dict:
    """图 1/2/3 输入帧的统一入口:优先读缓存,否则从原始数据计算并写缓存。

    返回 compute_raw_evidence 的结果 dict(至少含 member_metrics /
    classifications / conditional / focus_quality / focus_metrics)。
    """
    if cache_policy not in ("auto", "refresh", "reuse"):
        raise ValueError(f"unsupported cache_policy: {cache_policy!r}")
    cache_dir = Path(cache_dir)
    fingerprint = _fingerprint(mode)
    cache_path = cache_dir / f"raw_evidence_{mode}_{fingerprint[:16]}.pkl"

    if cache_policy in ("auto", "reuse") and cache_path.exists():
        with open(cache_path, "rb") as handle:
            cached = pickle.load(handle)
        if cached.get("fingerprint") == fingerprint:
            print(f"[cache] reuse {cache_path}", flush=True)
            return cached["frames"]
        if cache_policy == "reuse":
            raise RuntimeError(f"cache fingerprint mismatch: {cache_path}")
        print(f"[cache] fingerprint changed, recompute (was {cache_path.name})", flush=True)

    if cache_policy == "reuse":
        raise FileNotFoundError(
            f"cache_policy='reuse' but no valid cache under {cache_dir} "
            f"for mode={mode!r}; run with 'auto' or 'refresh' first"
        )

    frames = compute_raw_evidence(mode=mode)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump({"fingerprint": fingerprint, "frames": frames}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cache] saved {cache_path}", flush=True)
    return frames


if __name__ == "__main__":
    # 独立运行入口:直接计算/刷新缓存(python omtmp_raw_evidence.py [mode])
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="real", choices=("real", "synthetic"))
    parser.add_argument("--cache-dir", default=str(SCRIPT_DIR / "omtmp_raw_evidence_cache"))
    parser.add_argument("--cache-policy", default="auto",
                        choices=("auto", "refresh", "reuse"))
    args = parser.parse_args()
    result = ensure_raw_evidence(args.cache_dir, args.cache_policy, args.mode)
    print({key: (len(value) if hasattr(value, "__len__") else value)
           for key, value in result.items() if key != "d03_tables"})
