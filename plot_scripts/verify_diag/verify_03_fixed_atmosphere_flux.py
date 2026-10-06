"""诊断三:固定大气的海温替换诊断(SYNTHETIC 默认)。

复用现有物理实现 reconstruct_ocean_fluxes()(omtmp_flux_decomposition.py,
内部调用 wrf41_sfclayrev,本次不修改其物理实现)。

阶段 A(重建一致性诊断):用某一试验自身的输入重建 HFX/LH,与其原始模式通量
比较(强、弱耦合分别检查),保留偏差、RMSE、最大绝对差;并保留"原始模式强减弱
通量差"、"重建通量强减弱通量差"以及两者残差(含 RMS 与最大绝对差)。
质量判断条件可配置,但仅输出状态标记,不宣称"物理验证通过";真实运行时若
残差相对待解释的强弱通量差不可忽略,归因结果标记为待核实,并把该标记传递到
阶段 B 的表格与图件。

阶段 B(固定输入后的条件替换):

    F_ww = f(T_weak,  A_weak)
    F_sw = f(T_strong, A_weak)
    F_ss = f(T_strong, A_strong)
    F_ws = f(T_weak,  A_strong)   (可配置的第四组合,检查替换顺序敏感性)

其中 A 表示除被替换表面温度之外的全部重建输入——不仅是温湿风,还包括摩擦
速度初值(UST)、动力粗糙度初值(由 UST 诊断)、离地高度、气压等辅助输入。
固定其余输入只替换表面温度时,物理函数按自身算法重新计算稳定度和交换系数,
不人为固定函数内部计算量。

输出:
    dF_SST           = F_sw - F_ww   (只替换海温的响应)
    dF_rest          = F_ss - F_sw   (替换其余输入后的响应;不直接宣称纯大气反馈)
    dF_reconstructed = F_ss - F_ww
并比较其与实际模式通量差(强 - 弱,统一经 vc.strong_minus_weak 计算);
同时计算 F_ww/F_sw/F_ss 相对 NR 通量的误差,判断只替换海温时误差增大还是
减小,并保留重建残差以免把重建误差误当作物理机制。

运行: python verify_03_fixed_atmosphere_flux.py (默认合成模式)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import verify_common as vc
from verify_common import VerifyConfig

SCRIPT_NAME = "verify03_fixed_atmosphere_flux"
VERIFY_DIAG_DIR = Path(__file__).resolve().parent
#: 本目录位于 plot_scripts 下;保留旧布局(项目根/verify_diag)作为回退
PLOT_SCRIPTS_DIR = VERIFY_DIAG_DIR.parent
_FALLBACK_PLOT_SCRIPTS_DIR = VERIFY_DIAG_DIR.parents[1] / "Mangkhut_scripts" / "plot_scripts"

# =====================
# 可编辑配置区
# =====================
CONFIG = VerifyConfig(
    mode="real",  # "synthetic"(默认) | "real"(需同时改 real 配置并确认,见 README)
    output_dirname=SCRIPT_NAME,
    plot_regions=("r000_300",),
    real=vc.RealPathConfig(
        acknowledge_real_mode=True,   # 确认开关,缺省会拒绝运行
    )
)
#: 重建所用表面温度来源:
#:   "TSK"           试验自身 TSK(通量方案实际使用的表面温度;默认,推荐)
#:   "OM_TMP_LAYER0" 用 OM_TMP 第 0 层代替 TSK。
#: TODO(待核实):除非已核实模式内 TSK<-"OM_TMP 第一层"的传递关系,否则不要改此选项。
SURFACE_TEMP_SOURCE = "TSK"
#: 是否计算第四组合 F_ws(替换顺序敏感性检查)
INCLUDE_REVERSE_PATH = True
#: 保存逐点场的案例(方法, 成员, 预报时效)
POINT_FIELD_CASES = (
    ("EAKF", "006", 0.5),
    ("EAKF", "006", 3.0),
)
#: 阶段 A 重建质量判断条件(可配置;仅用于标记归因可信度,不是"物理验证通过"的判据):
#: 主判据:recon-model 残差 RMS 与 |模式强弱通量差| RMS 之比的上限
PHASE_A_RMSE_RATIO_MAX = 0.5
#: 辅助判据:recon-model 最大绝对差与模式强弱差 RMS 之比的上限
#: (用 RMS 而非 |均值| 作尺度,避免正负抵消)
PHASE_A_MAXABS_TO_RMS_RATIO_MAX = 5.0
#: 网格距与 sfctcflx 选项。TODO(待核实):DX_M 取自 analyze_omtmp_pathway.py(1500 m),
#: ISFTCFLX 同上(0);真实运行前需对照试验 namelist 确认。
DX_M = 1500.0
ISFTCFLX = 0
#: 初始摩擦速度/粗糙度下限(复用 analyze_omtmp_pathway.py 的钳制口径)
USTAR_FLOOR, Z0M_FLOOR = 1.0e-4, 1.27e-7

FLUX_VARIABLES = ("hfx", "lh")

# ---- 联合诊断(离线 SST 响应 vs 实际通量变化;新增,独立表) ----
#: 条件子集的海温侧变量(dSE_T 单位 K^2)
JOINT_SST_VARIABLE = "tsk"
#: 符号一致率的通量增量阈值(单位 W m-2;与 dSE 的 (W m-2)^2 阈值不同)
FLUX_INCREMENT_TOL = 1.0  # W m-2
#: RMS 比率的分母下限(W m-2;与符号一致率阈值分开配置):
#: 实际响应 RMS 低于该值时比率置 NaN 并标记 actual_response_below_floor,
#: 避免近零分母的极大比率主导平均;下限只影响比率,不改原始 RMS。
SST_RATIO_DENOMINATOR_FLOOR = 0.5  # W m-2
#: 归因三判据的比值上限(联合表的质量评估沿用阶段 A/B 的 PHASE_A_RMSE_RATIO_MAX)
JOINT_COLUMNS = [
    "mode", "method", "member", "time_hour", "valid_time", "region",
    "variable", "unit", "subset", "sst_variable", "sst_threshold_k2",
    "flux_increment_tol_wm2", "weight_method",
    "n_input_valid", "n_common", "n_tsk_common", "n_nonfinite_output", "n_subset",
    "subset_fraction_of_all_common", "status", "actual_response_status",
    "joint_status", "attribution_flag_region_ref",
    "attr_weak_rmse_ratio", "attr_strong_rmse_ratio", "attr_diff_agree_ratio",
    "cross_model_error_sst_response", "rms_sst_response", "rms_actual_response",
    "rms_sst_minus_actual", "sst_to_actual_rms_ratio",
    "n_sign_eligible", "n_sign_agree", "sign_agreement", "sign_eligible_fraction",
    "sign_status",
    "actual_mse_weak", "actual_mse_strong", "actual_delta_mse_direct",
    "actual_cross_term", "actual_increment_square_term", "actual_closure_residual",
]

PHASE_A_COLUMNS = [
    "mode", "experiment", "method", "member", "time_hour", "valid_time", "region",
    "variable", "unit", "weight_method", "n_valid", "status",
    "recon_minus_model_bias", "recon_minus_model_rmse", "recon_minus_model_maxabs",
    "model_strong_minus_weak_mean", "model_strong_minus_weak_rms",
    "recon_strong_minus_weak_mean",
    "residual_between_diffs_mean", "residual_between_diffs_rms",
    "residual_between_diffs_maxabs",
    "residual_rmse_ratio", "residual_maxabs_to_rms_ratio", "quality_flag",
]
PHASE_B_COLUMNS = [
    "mode", "method", "member", "time_hour", "valid_time", "region",
    "variable", "unit", "weight_method", "n_valid", "status",
    "n_nonfinite_output", "improvement_status", "attribution_flag",
    "attr_weak_rmse_ratio", "attr_strong_rmse_ratio", "attr_diff_agree_ratio",
    "dF_SST", "dF_rest", "dF_reconstructed", "split_closure_residual",
    "dF_SST_reverse", "dF_rest_reverse", "path_split_difference",
    "actual_model_diff_mean", "dF_SST_minus_actual",
    "dF_total_minus_actual_rms", "dF_total_minus_actual_maxabs",
    "Fww_minus_nr_bias", "Fww_minus_nr_rmse",
    "Fsw_minus_nr_bias", "Fsw_minus_nr_rmse",
    "Fss_minus_nr_bias", "Fss_minus_nr_rmse",
    "sst_only_rmse_improvement_pct", "sst_only_improvement_status",
]
STATUS_COLUMNS = ["mode", "method", "member", "time_hour", "valid_time", "status"]


def _ratio_on_mask(numerator_field, scale_field, mask: np.ndarray) -> float:
    """numerator RMS / scale RMS(都在同一子集掩膜上;分母为零或空集给 NaN)。"""
    numerator_rms = vc.rms(numerator_field, mask)
    scale_rms = vc.rms(scale_field, mask)
    if not (np.isfinite(scale_rms) and scale_rms > 0.0 and np.isfinite(numerator_rms)):
        return np.nan
    return numerator_rms / scale_rms


def _joint_quality_flag(residual_weak, residual_strong, total_minus_actual, actual, mask) -> str:
    """联合表按子集重评的重建质量状态(不继承全区域标记)。"""
    weak_ratio = _ratio_on_mask(residual_weak, actual, mask)
    strong_ratio = _ratio_on_mask(residual_strong, actual, mask)
    agree_ratio = _ratio_on_mask(total_minus_actual, actual, mask)
    return attribution_quality_flag(weak_ratio, strong_ratio, agree_ratio)


def attribution_quality_flag(
    weak_rmse_ratio, strong_rmse_ratio, diff_agree_ratio
) -> str:
    """归因可信度标记(必要条件筛选,非物理验证)。

    三个必要条件(阈值均为 PHASE_A_RMSE_RATIO_MAX,可配置):
      1. 弱试验重建残差 RMS / 模式强弱差 RMS;
      2. 强试验重建残差 RMS / 模式强弱差 RMS;
      3. 重建差与实际模式差的逐点差 RMS / 模式强弱差 RMS。
    全部满足才给 residual_small_vs_explained_diff;任一不满足即为
    residual_not_negligible_attribution_requires_verification。
    只检查弱耦合会漏掉"弱准强错"的情形(审查反例:模式 100/201,
    重建 100/101,弱残差为 0 而强残差与重建差都不可忽略)。
    """
    threshold = PHASE_A_RMSE_RATIO_MAX
    ratios = (weak_rmse_ratio, strong_rmse_ratio, diff_agree_ratio)
    if any(not np.isfinite(value) for value in ratios):
        return "attribution_criteria_not_evaluable"
    if all(value <= threshold for value in ratios):
        return "residual_small_vs_explained_diff"
    return "residual_not_negligible_attribution_requires_verification"


def import_reconstructor():
    """按需导入现有物理实现(无导入副作用,已静态审查)。"""
    global PLOT_SCRIPTS_DIR
    if not (PLOT_SCRIPTS_DIR / "omtmp_flux_decomposition.py").exists():
        PLOT_SCRIPTS_DIR = _FALLBACK_PLOT_SCRIPTS_DIR
    if str(PLOT_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(PLOT_SCRIPTS_DIR))
    try:
        from omtmp_flux_decomposition import reconstruct_ocean_fluxes
    except ImportError as error:  # pragma: no cover - 环境问题
        raise RuntimeError(
            f"无法导入 Mangkhut_scripts/plot_scripts/omtmp_flux_decomposition.py: {error}"
        ) from error
    return reconstruct_ocean_fluxes


def _validate_reconstruction_inputs(index: dict, surface: np.ndarray) -> None:
    """重建输入的物理有效范围预检(清晰失败状态,而非仅在函数内部报错)。"""
    checks = (
        ("surface temperature", surface, lambda x: np.all(np.isfinite(x))),
        ("air pressure", index["air_pressure_pa"], lambda x: np.all(x > 0.0)),
        ("surface pressure", index["surface_pressure_pa"], lambda x: np.all(x > 0.0)),
        ("height AGL", index["height_agl_m"], lambda x: np.all(x > 0.0)),
        ("initial friction velocity", index["initial_friction_velocity_ms"], lambda x: np.all(x > 0.0)),
        ("initial momentum roughness", index["initial_momentum_roughness_m"], lambda x: np.all(x > 0.0)),
    )
    for name, values, predicate in checks:
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"reconstruction input {name} contains non-finite values")
        if not predicate(values):
            raise ValueError(f"reconstruction input {name} outside physical valid range")
    for key in vc.RECONSTRUCTION_INPUT_KEYS:
        values = np.asarray(index[key], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"reconstruction input {key} contains non-finite values")


def call_reconstruction(reconstructor, inputs: dict, ts: np.ndarray, mask: np.ndarray):
    """在统一掩膜上调用现有物理函数;掩膜外输出 NaN。

    输入数组压平为 (1, n) 二维(物理函数要求 2-D);初始摩擦速度/粗糙度按
    analyze_omtmp_pathway.py 的口径做下限钳制(钳制前先做物理有效范围预检)。
    物理函数内部自行重算稳定度与交换系数,本封装不固定任何内部量。
    """
    index = {
        key: np.asarray(inputs[key], dtype=float)[mask]
        for key in vc.RECONSTRUCTION_INPUT_KEYS
    }
    surface = np.asarray(ts, dtype=float)[mask]
    _validate_reconstruction_inputs(index, surface)
    result = reconstructor(
        air_temperature_k=index["air_temperature_k"][None, :],
        surface_temperature_k=surface[None, :],
        vapor_mixing_ratio=index["vapor_mixing_ratio"][None, :],
        air_pressure_pa=index["air_pressure_pa"][None, :],
        surface_pressure_pa=index["surface_pressure_pa"][None, :],
        height_agl_m=index["height_agl_m"][None, :],
        u_ms=index["u_ms"][None, :],
        v_ms=index["v_ms"][None, :],
        initial_friction_velocity_ms=np.maximum(
            index["initial_friction_velocity_ms"], USTAR_FLOOR
        )[None, :],
        initial_momentum_roughness_m=np.maximum(
            index["initial_momentum_roughness_m"], Z0M_FLOOR
        )[None, :],
        dx_m=DX_M,
        isftcflx=ISFTCFLX,
    )
    full = {}
    for name in ("qfx", "lh", "hfx"):
        values = np.full(mask.shape, np.nan)
        values[mask] = np.asarray(result[name], dtype=float).ravel()
        full[name] = values
    return full


#: 重建输出参与比较的通量名(诊断三只用 hfx/lh)
RECON_FLUX_NAMES = ("hfx", "lh")


def region_output_status(region_mask, input_mask, compare_mask) -> tuple[int, int, str]:
    """单个区域的有效点数、非有限输出点数与状态。

    n          = compare_mask & region_mask 的点数(实际参与比较);
    nonfinite  = 该区域内输入有效但被输出掩膜剔除的点数
                 (按区域计,域外无效点不影响本区域状态);
    status     = "ok" / "nonfinite_reconstruction_output"(输入有效、输出无效)
               / "empty_input"(该区域本来就没有有效输入,非输出失效)。
    前两种状态不得混写为普通 empty_mask:失败原因与点数必须保留。
    """
    region_mask = np.asarray(region_mask, dtype=bool)
    input_mask = np.asarray(input_mask, dtype=bool)
    compare_mask = np.asarray(compare_mask, dtype=bool)
    n = int(np.count_nonzero(compare_mask & region_mask))
    nonfinite = int(np.count_nonzero(region_mask & input_mask & ~compare_mask))
    if n > 0:
        status = "ok"
    elif nonfinite > 0:
        status = "nonfinite_reconstruction_output"
    elif int(np.count_nonzero(region_mask & input_mask)) == 0:
        status = "empty_input"
    else:  # pragma: no cover - 防御分支(有输入、有比较点却两者皆零,矛盾)
        status = "empty_mask"
    return n, nonfinite, status


def evaluate_reconstruction_outputs(reconstructor, inputs: dict, ts: np.ndarray, mask: np.ndarray):
    """调用重建并对输出建立共同有限掩膜。

    返回 (flux dict, combined_mask, status, n_invalid_inside):
    status 为 "nonfinite_reconstruction_output" 表示在有效输入掩膜内出现了
    非有限输出(调用方必须据此报告,不得各列独立删点)。
    """
    flux = call_reconstruction(reconstructor, inputs, ts, mask)
    combined = np.array(mask, dtype=bool)
    invalid = 0
    for name in RECON_FLUX_NAMES:
        finite = np.isfinite(flux[name])
        invalid += int(np.count_nonzero(mask & ~finite))
        combined &= finite
    status = "ok" if invalid == 0 else "nonfinite_reconstruction_output"
    return flux, combined, status, invalid


def _surface_temperature(raw: dict) -> np.ndarray:
    if SURFACE_TEMP_SOURCE == "TSK":
        return np.asarray(raw["TSK"], dtype=float)
    if SURFACE_TEMP_SOURCE == "OM_TMP_LAYER0":
        # TODO(待核实):未经核实不得默认使用;仅保留接口。
        raise NotImplementedError(
            "OM_TMP 第一层代替 TSK 的做法待核实,默认禁用;请先完成数据约定审查。"
        )
    raise ValueError(f"unknown SURFACE_TEMP_SOURCE: {SURFACE_TEMP_SOURCE}")


def compute(config: VerifyConfig = CONFIG, provider=None, reconstructor=None,
            out_dir: Path | None = None) -> dict:
    """诊断三计算部分(内存复用入口):不做汇总与落盘,返回逐案例原始表。

    返回键 status / phase_a / phase_b / joint,与 run() 落盘的
    verify03_run_status / verify03_phaseA_reconstruction /
    verify03_phaseB_replacement / verify03_joint_response_actual
    四张主表逐列一致。out_dir 给定时另存 POINT_FIELD_CASES 的逐点场;
    为 None 时无任何文件写入。
    """
    vc.check_mode(config)
    if provider is None:
        provider = vc.make_provider(config)
    if reconstructor is None:
        reconstructor = import_reconstructor()

    mode_tag = config.mode.upper()
    phase_a_rows: list[dict] = []
    phase_b_rows: list[dict] = []
    joint_rows: list[dict] = []
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
                    _process_case(
                        provider, reconstructor, config, context,
                        method, member, time_hour, time_name, mode_tag,
                        phase_a_rows, phase_b_rows, status_rows, joint_rows, out_dir,
                    )
                except Exception as error:
                    status_rows.append({
                        "mode": mode_tag, "method": method, "member": member,
                        "time_hour": time_hour, "valid_time": time_name,
                        "status": f"reconstruction_failed: {type(error).__name__}: {error}",
                    })

    return {
        "status": pd.DataFrame(status_rows, columns=STATUS_COLUMNS),
        "phase_a": pd.DataFrame(phase_a_rows, columns=PHASE_A_COLUMNS),
        "phase_b": pd.DataFrame(phase_b_rows, columns=PHASE_B_COLUMNS),
        "joint": pd.DataFrame(joint_rows, columns=JOINT_COLUMNS),
    }


def run(config: VerifyConfig = CONFIG) -> Path:
    """执行诊断三(阶段 A + 阶段 B);返回输出目录。合成模式仅用内存合成数据。"""
    vc.check_mode(config)
    provider = vc.make_provider(config)
    reconstructor = import_reconstructor()
    mode_tag = config.mode.upper()
    out_dir = vc.ensure_output_dir(config, SCRIPT_NAME)
    frames = compute(
        config, provider=provider, reconstructor=reconstructor, out_dir=out_dir
    )
    status = frames["status"]
    phase_a = frames["phase_a"]
    phase_b = frames["phase_b"]
    joint = frames["joint"]

    # =====================
    # 汇总与输出(run_status 最先落盘)
    # =====================
    vc.write_csv(status, out_dir / "verify03_run_status.csv")
    vc.write_csv(phase_a, out_dir / "verify03_phaseA_reconstruction.csv")
    vc.write_csv(phase_b, out_dir / "verify03_phaseB_replacement.csv")
    vc.write_csv(joint, out_dir / "verify03_joint_response_actual.csv")
    _plot(phase_a, phase_b, config, out_dir, mode_tag)
    # 联合表的窗口汇总(先成员内时间平均,后跨成员等权;逐子集)
    joint_sum_cols = [
        "cross_model_error_sst_response", "rms_sst_response", "rms_actual_response",
        "rms_sst_minus_actual", "sst_to_actual_rms_ratio", "sign_agreement",
        "actual_delta_mse_direct", "actual_cross_term",
        "actual_increment_square_term", "actual_closure_residual",
    ]
    # 汇总键在分支前定义,保证空表分支也能使用(P2 #5 同类问题)
    joint_keys = ["mode", "method", "region", "variable", "unit", "subset", "window"]
    if not joint.empty:
        joint = joint.copy()
        joint["window"] = [vc.window_of(hour) for hour in joint["time_hour"]]
        joint = joint.explode("window", ignore_index=True)
        # 质量分层(P2 #1):全样本数值保留供描述,同时附质量合格/失败计数
        # 与质量覆盖率,不静默丢失质量不合格时次的信息。
        #   n_times_valid            = status==ok 的时次数(数值有效);
        #   n_times_quality_ok       = 数值有效且 joint_status 为合格标记的时次数;
        #   n_times_quality_failed   = 数值有效但质量不合格/不可评估的时次数;
        #   quality_coverage_ratio   = ok/(ok+failed);无有效时次时为 NaN。
        QUALITY_OK_FLAG = "residual_small_vs_explained_diff"

        def _quality_counts(flags: pd.Series, valid: pd.Series) -> pd.Series:
            ok = int(((flags == QUALITY_OK_FLAG) & valid).sum())
            failed = int((valid & (flags != QUALITY_OK_FLAG)).sum())
            coverage = ok / (ok + failed) if (ok + failed) > 0 else np.nan
            return pd.Series({
                "n_times_quality_ok": ok,
                "n_times_quality_failed": failed,
                "quality_coverage_ratio": coverage,
            })

        joint_member = (
            joint.groupby(joint_keys + ["member"], sort=True)
            .agg(
                **{f"mean_{col}": (col, "mean") for col in joint_sum_cols},
                mean_n_subset=("n_subset", "mean"),
                mean_n_common=("n_common", "mean"),
                mean_n_input_valid=("n_input_valid", "mean"),
                mean_n_tsk_common=("n_tsk_common", "mean"),
                mean_subset_fraction=("subset_fraction_of_all_common", "mean"),
                mean_sign_eligible_fraction=("sign_eligible_fraction", "mean"),
                n_times_read=("time_hour", "count"),
                n_times_valid=("status", lambda s: int((s == "ok").sum())),
                n_times_quality_ok=("joint_status", lambda s: int(
                    (s == QUALITY_OK_FLAG).sum()
                )),
                n_times_quality_failed=("joint_status", lambda s: int(
                    (s != QUALITY_OK_FLAG).sum()
                )),
            )
            .reset_index()
        )
        joint_member["quality_coverage_ratio"] = (
            joint_member.n_times_quality_ok
            / (joint_member.n_times_quality_ok + joint_member.n_times_quality_failed)
            .replace(0, np.nan)
        )
        vc.write_csv(joint_member, out_dir / "verify03_joint_summary_by_member.csv")
        joint_method = (
            joint_member.drop(columns=["member"])
            .groupby(joint_keys, sort=True)
            .agg(
                n_members=("mean_cross_model_error_sst_response", "count"),
                **{f"mean_{col}": (f"mean_{col}", "mean") for col in joint_sum_cols},
                mean_n_subset=("mean_n_subset", "mean"),
                mean_n_common=("mean_n_common", "mean"),
                mean_n_input_valid=("mean_n_input_valid", "mean"),
                mean_n_tsk_common=("mean_n_tsk_common", "mean"),
                mean_subset_fraction=("mean_subset_fraction", "mean"),
                mean_member_times_read=("n_times_read", "mean"),
                mean_member_valid_times=("n_times_valid", "mean"),
                n_times_quality_ok=("n_times_quality_ok", "sum"),
                n_times_quality_failed=("n_times_quality_failed", "sum"),
            )
            .reset_index()
        )
        joint_method["quality_coverage_ratio"] = (
            joint_method.n_times_quality_ok
            / (joint_method.n_times_quality_ok + joint_method.n_times_quality_failed)
            .replace(0, np.nan)
        )
        vc.write_csv(joint_method, out_dir / "verify03_joint_summary_by_method.csv")
    else:
        vc.write_csv(pd.DataFrame(columns=JOINT_COLUMNS),
                     out_dir / "verify03_joint_response_actual.csv")
        vc.write_csv(pd.DataFrame(columns=joint_keys + ["member"]),
                     out_dir / "verify03_joint_summary_by_member.csv")
        vc.write_csv(pd.DataFrame(columns=joint_keys),
                     out_dir / "verify03_joint_summary_by_method.csv")
    vc.write_columns_readme(out_dir / "columns_readme.txt", _columns_readme_lines(config))
    _print_summary(phase_a, phase_b, status, mode_tag)
    ok_rows = int((phase_b.status == "ok").sum()) if len(phase_b) else 0
    vc.write_overall_status(out_dir, ok_rows, len(status), mode_tag)
    vc.save_config_snapshot(out_dir, config)
    return out_dir


def _phase_a_quality(residual_stats: dict, model_diff_rms: float) -> tuple[float, float, str]:
    """由残差统计与模式强弱差 RMS 计算质量标记(RMS 尺度,避免正负抵消)。"""
    if residual_stats["status"] != "ok":
        return np.nan, np.nan, residual_stats["status"]
    if not (model_diff_rms > 0.0):
        return np.nan, np.nan, "model_diff_zero_rms"
    ratio = residual_stats["rms"] / model_diff_rms
    maxabs_ratio = residual_stats["maxabs"] / model_diff_rms
    if (
        ratio <= PHASE_A_RMSE_RATIO_MAX
        and maxabs_ratio <= PHASE_A_MAXABS_TO_RMS_RATIO_MAX
    ):
        flag = "residual_small_vs_explained_diff"
    else:
        flag = "residual_not_negligible_attribution_requires_verification"
    return ratio, maxabs_ratio, flag


def _process_case(
    provider, reconstructor, config, context,
    method, member, time_hour, time_name, mode_tag,
    phase_a_rows, phase_b_rows, status_rows, joint_rows, out_dir,
) -> None:
    """单个 (方法, 成员, 时刻) 的阶段 A 与阶段 B。"""
    raw_strong = provider.exp_raw_inputs(config.strong_experiment, method, member, time_hour)
    raw_weak = provider.exp_raw_inputs(config.weak_experiment, method, member, time_hour)
    inputs_strong = vc.derive_lowest_level_inputs(raw_strong)
    inputs_weak = vc.derive_lowest_level_inputs(raw_weak)
    ts_strong = _surface_temperature(raw_strong)
    ts_weak = _surface_temperature(raw_weak)
    fields_strong = provider.exp_fields(config.strong_experiment, method, member, time_hour)
    fields_weak = provider.exp_fields(config.weak_experiment, method, member, time_hour)
    # (strong, weak) 元组顺序;所有 strong_minus_weak 差值统一走
    # vc.strong_minus_weak([0], [1]),防止下标误用导致符号反转
    model_flux = {
        "hfx": (fields_strong["hfx"], fields_weak["hfx"]),
        "lh": (fields_strong["lh"], fields_weak["lh"]),
    }
    nr_hfx = context["nr"]["hfx"]
    nr_lh = context["nr"]["lh"]
    nr_flux = {"hfx": nr_hfx, "lh": nr_lh}

    # ---- 统一掩膜:海洋 & 全部参与数组共同有效(阶段 A/B 分开设掩膜) ----
    mask_phase_a = context["ocean"] & vc.unified_reconstruction_mask(
        inputs_weak, inputs_strong, ts_weak, ts_strong,
        extra=(
            model_flux["hfx"][0], model_flux["hfx"][1],
            model_flux["lh"][0], model_flux["lh"][1],
        ),
    )
    mask_phase_b = context["ocean"] & vc.unified_reconstruction_mask(
        inputs_weak, inputs_strong, ts_weak, ts_strong,
        extra=(
            model_flux["hfx"][0], model_flux["hfx"][1],
            model_flux["lh"][0], model_flux["lh"][1],
            nr_hfx, nr_lh,
        ),
    )
    if not np.any(mask_phase_a):
        status_rows.append({
            "mode": mode_tag, "method": method, "member": member,
            "time_hour": time_hour, "valid_time": time_name,
            "status": "empty_phaseA_mask",
        })
        return

    recon_weak, _, _, _ = evaluate_reconstruction_outputs(
        reconstructor, inputs_weak, ts_weak, mask_phase_a
    )
    recon_strong, _, _, _ = evaluate_reconstruction_outputs(
        reconstructor, inputs_strong, ts_strong, mask_phase_a
    )
    recon_flux = {
        "hfx": (recon_strong["hfx"], recon_weak["hfx"]),
        "lh": (recon_strong["lh"], recon_weak["lh"]),
    }

    base = {
        "mode": mode_tag,
        "method": method,
        "member": member,
        "time_hour": time_hour,
        "valid_time": time_name,
        "weight_method": vc.WEIGHT_LABEL,
    }

    # =====================
    # 阶段 A:重建一致性(每行单一统计掩膜:重建输出与模式场共同有限)
    # =====================
    for experiment_name, exp_index in (
        ("weak", 1),
        ("strong", 0),
    ):
        recon_output = recon_flux
        model_output = model_flux
        for variable in FLUX_VARIABLES:
            residual = recon_output[variable][exp_index] - model_output[variable][exp_index]
            # 强减弱差值:统一 strong - weak 方向
            diff_model = vc.strong_minus_weak(
                model_output[variable][0], model_output[variable][1]
            )
            diff_recon = vc.strong_minus_weak(
                recon_output[variable][0], recon_output[variable][1]
            )
            diff_residual = diff_recon - diff_model
            for region_name, region_mask in context["regions"].items():
                stat_mask = (
                    np.asarray(mask_phase_a, dtype=bool)
                    & np.asarray(region_mask, dtype=bool)
                    & np.isfinite(residual)
                    & np.isfinite(diff_recon)
                    & np.isfinite(diff_model)
                )
                inside = np.asarray(mask_phase_a, dtype=bool) & np.asarray(region_mask, dtype=bool)
                n_inside = int(np.count_nonzero(inside))
                residual_bad = inside & ~np.isfinite(residual)
                # diff_recon 含弱/强两组重建;一组无效会让另一组的差值也被
                # 共同掩膜删点,此处显式记录该共同失效原因
                diff_bad = (
                    inside & np.isfinite(residual)
                    & (~np.isfinite(diff_recon) | ~np.isfinite(diff_model))
                )
                # 先确定失效原因再判空:全无效输出必须报告 nonfinite 状态,
                # 不得写成普通 empty_mask 让失败原因丢失;区域无有效输入时
                # 报告 empty_input(与输出失效区分)
                if n_inside == 0:
                    status = "empty_input"
                elif np.any(residual_bad):
                    status = "nonfinite_reconstruction_output"
                elif np.any(diff_bad):
                    status = "nonfinite_reconstruction_diff"
                else:
                    status = "ok"
                n = int(np.count_nonzero(stat_mask))
                if n == 0:
                    phase_a_rows.append({
                        **base, "experiment": experiment_name, "region": region_name,
                        "variable": variable, "unit": vc.VARIABLE_UNITS[variable],
                        "n_valid": 0, "status": status,
                        **{key: np.nan for key in PHASE_A_COLUMNS if key not in (
                            "mode", "experiment", "method", "member", "time_hour",
                            "valid_time", "region", "variable", "unit", "weight_method",
                            "n_valid", "status",
                        )},
                    })
                    continue
                residual_data = residual[stat_mask]
                diff_model_data = diff_model[stat_mask]
                diff_residual_data = diff_residual[stat_mask]
                residual_stats = {
                    "status": "ok",
                    "mean": float(np.mean(residual_data)),
                    "rms": float(np.sqrt(np.mean(residual_data**2))),
                    "maxabs": float(np.max(np.abs(residual_data))),
                }
                model_diff_rms = float(np.sqrt(np.mean(diff_model_data**2)))
                ratio, maxabs_ratio, quality_flag = _phase_a_quality(
                    residual_stats, model_diff_rms
                )
                phase_a_rows.append({
                    **base,
                    "experiment": experiment_name,
                    "region": region_name,
                    "variable": variable,
                    "unit": vc.VARIABLE_UNITS[variable],
                    "n_valid": n,
                    "status": status,
                    "recon_minus_model_bias": residual_stats["mean"],
                    "recon_minus_model_rmse": residual_stats["rms"],
                    "recon_minus_model_maxabs": residual_stats["maxabs"],
                    "model_strong_minus_weak_mean": float(np.mean(diff_model_data)),
                    "model_strong_minus_weak_rms": model_diff_rms,
                    "recon_strong_minus_weak_mean": float(np.mean(diff_recon[stat_mask])),
                    "residual_between_diffs_mean": float(np.mean(diff_residual_data)),
                    "residual_between_diffs_rms": float(
                        np.sqrt(np.mean(diff_residual_data**2))
                    ),
                    "residual_between_diffs_maxabs": float(
                        np.max(np.abs(diff_residual_data))
                    ),
                    "residual_rmse_ratio": ratio,
                    "residual_maxabs_to_rms_ratio": maxabs_ratio,
                    "quality_flag": quality_flag,
                })

    # =====================
    # 阶段 B:固定输入后的条件替换(输出建立共同有限掩膜)
    # =====================
    if not np.any(mask_phase_b):
        status_rows.append({
            "mode": mode_tag, "method": method, "member": member,
            "time_hour": time_hour, "valid_time": time_name,
            "status": "empty_phaseB_mask",
        })
        return
    flux_ww, mask_ww, _, _ = evaluate_reconstruction_outputs(
        reconstructor, inputs_weak, ts_weak, mask_phase_b
    )
    flux_sw, mask_sw, _, _ = evaluate_reconstruction_outputs(
        reconstructor, inputs_weak, ts_strong, mask_phase_b
    )
    flux_ss, mask_ss, _, _ = evaluate_reconstruction_outputs(
        reconstructor, inputs_strong, ts_strong, mask_phase_b
    )
    flux_ws = None
    mask_ws = None
    if INCLUDE_REVERSE_PATH:
        flux_ws, mask_ws, _, _ = evaluate_reconstruction_outputs(
            reconstructor, inputs_strong, ts_weak, mask_phase_b
        )
    # 归因质量标记改为在每个区域的最终比较掩膜上、用弱+强+差一致三判据
    # 逐一评估(见下方区域循环),不再跨区域复制单一标记。

    pending_fields: list[dict] = []
    for variable in FLUX_VARIABLES:
        # 比较用共同有限掩膜:四种重建(启用时)、模式通量与 NR 通量全部有限;
        # 先在全网格求一次 compare 掩膜,非有限点数再按区域统计,
        # 避免 300 km 外的无效点把所有区域都标成非有限输出。
        valid_parts = [
            mask_ww, mask_sw, mask_ss,
            np.isfinite(model_flux[variable][0]), np.isfinite(model_flux[variable][1]),
            np.isfinite(nr_flux[variable]),
        ]
        if flux_ws is not None:
            valid_parts.append(mask_ws)
        mask_compare = np.array(mask_phase_b, dtype=bool)
        for part in valid_parts:
            mask_compare &= np.asarray(part, dtype=bool)

        d_sst = flux_sw[variable] - flux_ww[variable]
        d_rest = flux_ss[variable] - flux_sw[variable]
        d_total = flux_ss[variable] - flux_ww[variable]
        d_sst_rev = flux_ss[variable] - flux_ws[variable] if flux_ws is not None else None
        d_rest_rev = flux_ws[variable] - flux_ww[variable] if flux_ws is not None else None
        # 实际模式通量差:strong - weak(统一符号入口)
        actual = vc.strong_minus_weak(model_flux[variable][0], model_flux[variable][1])
        total_minus_actual = np.asarray(d_total, dtype=float) - actual
        # 归因三判据的分子(逐点残差场),分母为各区域内的模式强弱差 RMS。
        # 残差用阶段 B 自身的替换输出(F_ww/F_ss)构建,与最终比较掩膜严格一致
        # (而非阶段 A 在 A 掩膜上的重建输出)
        residual_weak = flux_ww[variable] - model_flux[variable][1]
        residual_strong = flux_ss[variable] - model_flux[variable][0]

        for region_name, region_mask in context["regions"].items():
            region_mask = np.asarray(region_mask, dtype=bool)
            combined = mask_compare & region_mask
            n, n_nonfinite_region, status = region_output_status(
                region_mask, mask_phase_b, mask_compare
            )
            nan_row = {key: np.nan for key in PHASE_B_COLUMNS if key not in (
                "mode", "method", "member", "time_hour", "valid_time",
                "region", "variable", "unit", "weight_method", "n_valid",
                "status", "improvement_status", "attribution_flag",
                "sst_only_improvement_status",
            )}
            if n == 0:
                # 使用 region_output_status 的真实状态与计数:
                # empty_input(原本无有效输入)与 nonfinite_reconstruction_output
                # (输入有效但输出全无效)分开报告,失败原因与点数不丢失
                phase_b_rows.append({
                    **base, "region": region_name, "variable": variable,
                    "unit": vc.VARIABLE_UNITS[variable], "n_valid": 0,
                    **nan_row,
                    "status": status,
                    "n_nonfinite_output": n_nonfinite_region,
                    "improvement_status": status,
                    "sst_only_improvement_status": status,
                    "attribution_flag": "empty_mask",
                })
                continue

            def masked_mean(field):
                return float(np.mean(np.asarray(field, dtype=float)[combined]))

            def masked_rms(field):
                return float(np.sqrt(np.mean(np.asarray(field, dtype=float)[combined] ** 2)))

            def bias_rmse(field):
                diff = np.asarray(field, dtype=float)[combined] - nr_flux[variable][combined]
                return float(np.mean(diff)), float(np.sqrt(np.mean(diff**2)))

            ww_bias, ww_rmse = bias_rmse(flux_ww[variable])
            sw_bias, sw_rmse = bias_rmse(flux_sw[variable])
            ss_bias, ss_rmse = bias_rmse(flux_ss[variable])
            d_sst_mean = masked_mean(d_sst)
            d_rest_mean = masked_mean(d_rest)
            d_total_mean = masked_mean(d_total)
            actual_mean = masked_mean(actual)
            if ww_rmse > 0.0:
                sst_improvement = 100.0 * (ww_rmse - sw_rmse) / ww_rmse
                sst_improvement_status = "ok"
            else:
                sst_improvement = np.nan
                sst_improvement_status = "zero_rmse_baseline"
            # 归因三判据(在该区域最终比较掩膜上逐一评估,数值一并保存):
            # 弱/强重建残差 RMS 比 + 重建差-实际差 RMS 比,分母为模式强弱差 RMS
            model_diff_rms = masked_rms(actual)
            if model_diff_rms > 0.0:
                weak_ratio = masked_rms(residual_weak) / model_diff_rms
                strong_ratio = masked_rms(residual_strong) / model_diff_rms
                diff_agree_ratio = masked_rms(total_minus_actual) / model_diff_rms
            else:
                weak_ratio = strong_ratio = diff_agree_ratio = np.nan
            attribution_flag = attribution_quality_flag(
                weak_ratio, strong_ratio, diff_agree_ratio
            )
            row = {
                **base,
                "region": region_name,
                "variable": variable,
                "unit": vc.VARIABLE_UNITS[variable],
                "n_valid": n,
                **nan_row,
                "status": status,
                "n_nonfinite_output": n_nonfinite_region,
                "improvement_status": sst_improvement_status,
                "attribution_flag": attribution_flag,
                "attr_weak_rmse_ratio": weak_ratio,
                "attr_strong_rmse_ratio": strong_ratio,
                "attr_diff_agree_ratio": diff_agree_ratio,
                "dF_SST": d_sst_mean,
                "dF_rest": d_rest_mean,
                "dF_reconstructed": d_total_mean,
                "split_closure_residual": d_total_mean - (d_sst_mean + d_rest_mean),
                "dF_SST_reverse": masked_mean(d_sst_rev) if d_sst_rev is not None else np.nan,
                "dF_rest_reverse": masked_mean(d_rest_rev) if d_rest_rev is not None else np.nan,
                "path_split_difference": (
                    d_sst_mean - masked_mean(d_sst_rev) if d_sst_rev is not None else np.nan
                ),
                "actual_model_diff_mean": actual_mean,
                "dF_SST_minus_actual": d_sst_mean - actual_mean,
                "dF_total_minus_actual_rms": masked_rms(total_minus_actual),
                "dF_total_minus_actual_maxabs": float(
                    np.max(np.abs(total_minus_actual[combined]))
                ),
                "Fww_minus_nr_bias": ww_bias,
                "Fww_minus_nr_rmse": ww_rmse,
                "Fsw_minus_nr_bias": sw_bias,
                "Fsw_minus_nr_rmse": sw_rmse,
                "Fss_minus_nr_bias": ss_bias,
                "Fss_minus_nr_rmse": ss_rmse,
                "sst_only_rmse_improvement_pct": sst_improvement,
                "sst_only_improvement_status": sst_improvement_status,
            }
            phase_b_rows.append(row)
            # ---- 联合诊断(新增;全部配置案例都计算,不受 POINT_FIELD_CASES 限制)。
            # 共同掩膜 = 区域 & 海洋 & 强/弱/NR 的 TSK 与通量 & NR 通量 &
            # 全部参与比较的重建输出(含 Fws);TSK 也须有限,
            # 不同指标不各自删点。子集只按海温误差变化筛选。----
            dse_sst_joint = vc.se_change(
                fields_strong[JOINT_SST_VARIABLE],
                fields_weak[JOINT_SST_VARIABLE],
                context["nr"][JOINT_SST_VARIABLE],
            )
            # 三级点数口径(P2 #3):
            #   n_input_valid = 输入掩膜(区域&海洋&mask_phase_b,不含重建输出);
            #   n_common      = 重建输出后的共同有效点(mask_compare);
            #   n_tsk_common  = 再加 TSK 共同有限(联合分析实际可用点)。
            # 由此可分别恢复"重建造成的样本损失"与"TSK 造成的样本损失"。
            region_ocean = np.asarray(region_mask, dtype=bool) & context["ocean"]
            n_input_valid = int(np.count_nonzero(region_ocean & mask_phase_b))
            n_common = int(np.count_nonzero(region_ocean & mask_compare))
            joint_base_mask = (
                region_ocean & mask_compare
                & np.isfinite(fields_strong[JOINT_SST_VARIABLE])
                & np.isfinite(fields_weak[JOINT_SST_VARIABLE])
                & np.isfinite(context["nr"][JOINT_SST_VARIABLE])
            )
            n_tsk_common = int(np.count_nonzero(joint_base_mask))
            joint_subsets = vc.sst_subset_masks(dse_sst_joint, joint_base_mask, config.sst_se_tol)
            e_model_joint = (
                np.asarray(model_flux[variable][1], dtype=float) - nr_flux[variable]
            )
            joint_attribution = attribution_quality_flag(
                weak_ratio, strong_ratio, diff_agree_ratio
            ) if model_diff_rms > 0.0 else "attribution_criteria_not_evaluable"
            for subset_name, subset_mask in joint_subsets.items():
                stats = vc.response_vs_actual_stats(
                    e_model_joint, d_sst, actual, subset_mask, FLUX_INCREMENT_TOL,
                    ratio_denominator_floor=SST_RATIO_DENOMINATOR_FLOOR,
                )
                if stats["status"] == "ok":
                    # 联合表唯一的子集质量标记:在每个子集掩膜上重算三判据;
                    # 空子集/指标不可用的行不继承全区域标记
                    subset_quality = _joint_quality_flag(
                        residual_weak, residual_strong, total_minus_actual,
                        actual, subset_mask,
                    )
                else:
                    subset_quality = stats["status"]
                joint_rows.append({
                    "mode": mode_tag,
                    "method": method,
                    "member": member,
                    "time_hour": time_hour,
                    "valid_time": time_name,
                    "region": region_name,
                    "variable": variable,
                    "unit": vc.VARIABLE_UNITS[variable],
                    "subset": subset_name,
                    "sst_variable": JOINT_SST_VARIABLE,
                    "sst_threshold_k2": config.sst_se_tol,
                    "flux_increment_tol_wm2": FLUX_INCREMENT_TOL,
                    "weight_method": vc.WEIGHT_LABEL,
                    "n_input_valid": n_input_valid,
                    "n_common": n_common,
                    "n_tsk_common": n_tsk_common,
                    "n_nonfinite_output": n_nonfinite_region,
                    "n_subset": stats["n_valid"],
                    "subset_fraction_of_all_common": (
                        stats["n_valid"] / n_tsk_common if n_tsk_common > 0 else np.nan
                    ),
                    "status": stats["status"],
                    "actual_response_status": stats["actual_response_status"],
                    # 全域参考标记(仅供对照,不是子集质量):
                    # 子集质量的唯一依据是下面的 joint_status
                    "attribution_flag_region_ref": joint_attribution,
                    # 联合表的质量状态按子集单独评估,不继承全区域标记:
                    # 残差比在该子集掩膜上重算;空子集/不可用行记 stats 状态
                    "joint_status": subset_quality,
                    "attr_weak_rmse_ratio": _ratio_on_mask(residual_weak, actual, subset_mask),
                    "attr_strong_rmse_ratio": _ratio_on_mask(residual_strong, actual, subset_mask),
                    "attr_diff_agree_ratio": _ratio_on_mask(total_minus_actual, actual, subset_mask),
                    **{
                        **{key: stats[key] for key in (
                            "cross_model_error_sst_response", "rms_sst_response",
                            "rms_actual_response", "rms_sst_minus_actual",
                            "sst_to_actual_rms_ratio",
                            "n_sign_eligible", "n_sign_agree", "sign_agreement",
                            "sign_eligible_fraction", "sign_status",
                        )},
                        "actual_mse_weak": stats["actual_mse_weak"],
                        "actual_mse_strong": stats["actual_mse_strong"],
                        "actual_delta_mse_direct": stats["actual_delta_mse_direct"],
                        "actual_cross_term": stats["actual_cross_term"],
                        "actual_increment_square_term": stats["actual_increment_square_term"],
                        "actual_closure_residual": stats["actual_closure_residual"],
                    },
                })
            # 实际 C/S/ΔMSE 列名前缀处理:统一为 actual_*
            if (method, member, time_hour) in POINT_FIELD_CASES and region_name == config.union_region:
                pending_fields.append((variable, d_sst, d_rest, d_total, actual, flux_ww, flux_sw, flux_ss, flux_ws, mask_compare))
    # 逐点场仅落盘模式保存(内存复用跳过)
    if out_dir is None:
        return
    for variable, d_sst, d_rest, d_total, actual, fww, fsw, fss, fws, mask_c in pending_fields:
        tag = f"{method}_{member}_t{time_hour:g}_{variable}"
        arrays = {
            "dF_SST": np.where(mask_c, d_sst, np.nan),
            "dF_rest": np.where(mask_c, d_rest, np.nan),
            "dF_reconstructed": np.where(mask_c, d_total, np.nan),
            "actual_model_diff": np.where(mask_c, actual, np.nan),
            "Fww": np.where(mask_c, fww[variable], np.nan),
            "Fsw": np.where(mask_c, fsw[variable], np.nan),
            "Fss": np.where(mask_c, fss[variable], np.nan),
            "nr_flux": np.where(mask_c, nr_flux[variable], np.nan),
            "compare_mask": mask_c,
            "distance_km": context["distance"],
            "lat": context["static"]["lat"],
            "lon": context["static"]["lon"],
            "nr_center_lat": context["center"][0],
            "nr_center_lon": context["center"][1],
            "mode": np.asarray(mode_tag),
            "method": np.asarray(method),
            "member": np.asarray(member),
            "time_hour": np.asarray(time_hour),
        }
        if fws is not None:
            arrays["Fws"] = np.where(mask_c, fws[variable], np.nan)
        vc.save_point_fields(out_dir / "fields" / f"verify03_replacement_{tag}.npz", **arrays)


def _plot(phase_a: pd.DataFrame, phase_b: pd.DataFrame, config: VerifyConfig, out_dir: Path, mode_tag: str) -> None:
    """阶段 A 残差时序图(逐方法、逐试验,缺失时次补 NaN 断线)
    与阶段 B 分量柱状图(含归因标记提示)。"""
    hours = [hour for hour, _ in config.times]
    for variable in FLUX_VARIABLES:
        for region_name in config.plot_regions:
            selected = phase_a[
                (phase_a.variable == variable) & (phase_a.region == region_name)
            ]
            if selected.empty:
                continue
            series: dict[str, tuple] = {}
            for method in config.methods:
                method_frame = selected[selected.method == method]
                if method_frame.empty:
                    continue
                grouped = method_frame.groupby(["experiment", "time_hour"], sort=True).agg(
                    bias=("recon_minus_model_bias", "mean"),
                    rmse=("recon_minus_model_rmse", "mean"),
                )
                for experiment in ("weak", "strong"):
                    if experiment not in grouped.index.get_level_values(0):
                        continue
                    experiment_frame = vc.align_times_to_hours(
                        grouped.loc[experiment].reset_index(), hours
                    )
                    label = f"{method}-{experiment} exp"
                    series[f"{label}: bias"] = (
                        experiment_frame.index, experiment_frame.bias.to_numpy(dtype=float)
                    )
                    series[f"{label}: rmse"] = (
                        experiment_frame.index, experiment_frame.rmse.to_numpy(dtype=float)
                    )
            vc.plot_metric_timeseries(
                series,
                f"[{mode_tag}] Phase A reconstruction residual, {variable} ({region_name})\n"
                "per method and experiment; see quality_flag in phase-A CSV",
                f"{vc.VARIABLE_UNITS[variable]}",
                out_dir / "figs" / f"verify03_phaseA_residual_{variable}_{region_name}.png",
            )
        # 阶段 B 柱状图:主区域,成员先均;标题附归因标记分布
        union_rows = phase_b[
            (phase_b.variable == variable)
            & (phase_b.region == config.plot_regions[0])
        ]
        if union_rows.empty:
            continue
        bar_values: dict[str, list] = {}
        for method in config.methods:
            selected = union_rows[union_rows.method == method]
            if selected.empty:
                continue
            bar_values[f"{method}: dF_SST"] = [
                vc.member_then_mean(selected, "dF_SST", config.members)
            ]
            bar_values[f"{method}: dF_rest"] = [
                vc.member_then_mean(selected, "dF_rest", config.members)
            ]
            bar_values[f"{method}: dF_total"] = [
                vc.member_then_mean(selected, "dF_reconstructed", config.members)
            ]
            bar_values[f"{method}: actual"] = [
                vc.member_then_mean(selected, "actual_model_diff_mean", config.members)
            ]
        if not bar_values:
            continue
        flags = union_rows.attribution_flag.value_counts(dropna=False)
        flag_text = "; ".join(f"{name}={count}" for name, count in flags.items())
        vc.plot_budget_bars(
            [f"all cases ({variable}, {config.plot_regions[0]})"],
            bar_values,
            f"[{mode_tag}] Phase B replacement components, {variable}\n"
            f"attribution flags on B mask: {flag_text}",
            f"dF ({vc.VARIABLE_UNITS[variable]})",
            out_dir / "figs" / f"verify03_phaseB_components_{variable}.png",
        )


def _columns_readme_lines(config: VerifyConfig) -> list[str]:
    reverse_note = (
        f"已启用第四组合 F_ws(替换顺序敏感性);path_split_difference = "
        f"dF_SST - dF_SST_reverse,非零即路径依赖。"
        if INCLUDE_REVERSE_PATH
        else "第四组合 F_ws 未启用。"
    )
    return [
        "verify03 输出列说明(SYNTHETIC/REAL 由 mode 列标记)",
        "",
        f"- 表面温度来源 SURFACE_TEMP_SOURCE = {SURFACE_TEMP_SOURCE}"
        "(TODO(待核实):如改用 OM_TMP 第一层需先完成数据约定审查)。",
        f"- dx_m = {DX_M} m,isftcflx = {ISFTCFLX}(TODO(待核实):取自",
        "  analyze_omtmp_pathway.py,需对照试验 namelist)。",
        f"- {reverse_note}",
        "- A 的含义:除被替换表面温度之外的全部重建输入,不仅包括温湿风,还包括摩擦速度",
        "  初值(UST)、动力粗糙度初值(由 UST 诊断)、离地高度、气压等辅助输入。",
        "- 固定其余输入只替换表面温度时,物理函数按自身算法重新计算稳定度和交换系数。",
        "- 实际模式通量差与重建通量差均为 strong - weak 方向(正值=强试验更大),",
        "  统一经 vc.strong_minus_weak 计算。",
        "- 阶段 A 列:recon_minus_model_* = 重建 - 该试验原始模式通量;",
        "  model/recon_strong_minus_weak_mean = 强减弱的通量差(模式/重建);",
        "  residual_between_diffs_* = 两种差值之残差的均值/RMS/最大绝对差;",
        "  residual_rmse_ratio = 残差 RMS / 模式强弱差 RMS(主判据);",
        "  residual_maxabs_to_rms_ratio 用 RMS 尺度(避免正负抵消)。",
        "  quality_flag 仅为可信度标记(阈值 "
        f"{PHASE_A_RMSE_RATIO_MAX:g}/{PHASE_A_MAXABS_TO_RMS_RATIO_MAX:g},可配置),",
        "  不构成「物理验证通过」;真实运行若残差不可忽略,归因结果须标记待核实。",
        "- 阶段 B 列:status/n_valid 按区域反映重建输出与模式、NR 通量的共同有限掩膜;",
        "  n_nonfinite_output = 该区域内输入有效但输出被剔除的点数(status=nonfinite_",
        "  reconstruction_output 时必须核查);attribution_flag 在每个区域的最终比较掩膜",
        "  上用三个必要条件逐一评估(弱重建残差比、强重建残差比、重建差-实际差比,",
        "  分母均为该区域模式强弱差 RMS),判据数值保存在 attr_*_ratio 列,",
        "  不跨区域复制标记;阈值 "
        f"{PHASE_A_RMSE_RATIO_MAX:g}(可配置),不构成「物理验证通过」。",
        "- dF_SST = F_sw-F_ww(只替换海温);dF_rest = F_ss-F_sw = 替换其余输入后的",
        "  响应(不直接宣称纯大气反馈);dF_reconstructed = F_ss-F_ww;",
        "  split_closure_residual 应接近 0(恒等式)。",
        "- dF_total_minus_actual_rms/maxabs:逐点 (F_ss-F_ww)-(模式强弱差) 的 RMS 与",
        "  最大绝对差,供评估替换分解对实际差的解释程度。",
        "- 误差列 F*_minus_nr_* 为 F_ww/F_sw/F_ss 相对 NR 通量的偏差/RMSE;",
        "  sst_only_rmse_improvement_pct = 100*(RMSE(F_ww)-RMSE(F_sw))/RMSE(F_ww),",
        "  正值表示只替换海温使(重建)通量误差减小;分母为零时置 NaN 并由",
        "  sst_only_improvement_status=zero_rmse_baseline 标记。",
        "- 重建残差(phase A / attribution_flag)必须与阶段 B 一起阅读,",
        "  避免把重建误差误当作物理机制。",
        "- 联合表 verify03_joint_response_actual.csv(逐子集):",
        "  点数三级口径 n_input_valid(输入掩膜)/n_common(重建输出后)/",
        "  n_tsk_common(再加 TSK 有限)分别恢复重建与 TSK 造成的样本损失;",
        "  n_nonfinite_output 为该区域内重建输出无效点数(部分失效时行仍为",
        "  status=ok,损失由该列承载);joint_status 是子集质量的唯一标记",
        "  (空子集/不可评估行不继承全区域标记),attribution_flag_region_ref",
        "  仅为全域对照;flux_increment_tol_wm2 单位为 W m-2(注意不是平方);",
        "  sst_threshold_k2 单位为 K^2;actual_* 预算列单位为 (W m-2)^2;",
        "  actual_response_status=actual_response_below_floor 表示实际响应 RMS",
        f"  低于分母下限 {SST_RATIO_DENOMINATOR_FLOOR:g} W m-2,比率置 NaN。",
        "- 联合汇总(verify03_joint_summary_by_member/_method.csv):全样本数值",
        "  保留供描述,并附 n_times_quality_ok/failed 与 quality_coverage_ratio",
        "  (合格时次占数值有效时次的比例);质量不合格的时次不参与物理解释,",
        "  但不从统计中静默删除。",
        "- fields/*.npz 保存 Fww/Fsw/Fss(/Fws)、NR 通量、dF 各分量、共同掩膜与元数据,",
        "  可离线重算各组 RMSE。",
        "- weight_method = 'equal_weight_valid_points'(等权有效格点平均)。",
    ]


def _print_summary(phase_a, phase_b, status, mode_tag: str) -> None:
    flags = phase_a.quality_flag.value_counts(dropna=False).to_dict() if len(phase_a) else {}
    print(
        f"[{mode_tag}] verify03: {len(phase_a)} phase-A rows, {len(phase_b)} phase-B rows, "
        f"{len(status)} skipped cases; quality flags: {flags}",
        flush=True,
    )


if __name__ == "__main__":
    output_dir = run(CONFIG)
    print(f"[{CONFIG.mode.upper()}] verify03 done -> {output_dir}", flush=True)
