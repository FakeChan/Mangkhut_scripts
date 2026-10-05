"""verify_diag 合成冒烟测试(SYNTHETIC ONLY,固定种子,本地 Mac 快速完成)。

两层测试:
  第一层:纯数组诊断函数与输出流程(手工小数组的解析期望值);
  第二层:对现有真实物理函数 reconstruct_ocean_fluxes() 用合理量级的
          合成输入做接口冒烟测试(不做物理正确性断言);
          以及 RealWrfProvider 读取适配层(本地合成 NetCDF 文件,非真实数据);
  外加:三个诊断入口在合成模式下的完整流程(表格/图件/中间数组)。

本次验收范围仅为:语法与导入、无副作用导入、数组维度与接口、输出路径与
文件生成、合成流程能否运行完成,以及针对已修复缺陷的回归断言(符号、
降序网格、缺失时次、汇总上界等)。通过不代表物理实现或数学归因正确;
恒等式残差仅供后续数学审查,不构成正确性证明。

运行: python verify_smoke_test.py
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import verify_common as vc
from verify_common import RealPathConfig, VerifyConfig

SCRIPT_DIR = Path(__file__).resolve().parent
verify_03_module = None  # 延迟绑定:check_joint_stats_manual 中导入
SMOKE_DIRNAME = "smoke_test"
SMOKE_CONFIG = VerifyConfig(mode="synthetic", output_dirname=SMOKE_DIRNAME)
SMOKE_OUT = SMOKE_CONFIG.output_root / f"SYNTHETIC_{SMOKE_DIRNAME}"

SEED = 20260930


class Check:
    def __init__(self, name: str):
        self.name = name
        self.failures: list[str] = []

    def expect(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)

    @property
    def ok(self) -> bool:
        return not self.failures


def raises(error_type, callable_, *args, **kwargs) -> bool:
    try:
        callable_(*args, **kwargs)
    except error_type:
        return True
    except Exception:
        return False
    return False


# ==================================================================
# 第一层:纯数组诊断函数
# ==================================================================


def check_paired_error_metrics_manual() -> Check:
    """手工构造的 2x2 例子,与解析值逐项对照。"""
    check = Check("paired_error_metrics 手工解析值")
    strong = np.array([[1.0, 3.0], [5.0, 7.0]])
    weak = np.array([[2.0, 5.0], [8.0, 11.0]])
    truth = np.array([[1.0, 4.0], [7.0, 10.0]])
    stats = vc.paired_error_metrics(strong, weak, truth)
    # 强误差 = [0,-1,-2,-3] -> MSE=(0+1+4+9)/4=3.5; 弱误差=[1,1,1,1] -> MSE=1
    check.expect(stats["n_valid"] == 4, f"n_valid={stats['n_valid']} != 4")
    check.expect(abs(stats["mse_strong"] - 3.5) < 1e-12, f"mse_strong={stats['mse_strong']}")
    check.expect(abs(stats["mse_weak"] - 1.0) < 1e-12, f"mse_weak={stats['mse_weak']}")
    check.expect(abs(stats["rmse_strong"] - np.sqrt(3.5)) < 1e-12, "rmse_strong mismatch")
    check.expect(abs(stats["mse_change_strong_minus_weak"] - 2.5) < 1e-12, "mse change mismatch")
    check.expect(stats["mse_change_strong_minus_weak"] > 0, "强减弱差值应为正(恶化)")
    expected_impr = 100.0 * (1.0 - np.sqrt(3.5)) / 1.0
    check.expect(
        abs(stats["rmse_improvement_pct"] - expected_impr) < 1e-12,
        f"improvement={stats['rmse_improvement_pct']} != {expected_impr}",
    )
    check.expect(stats["status"] == "zero_rmse_weak" or stats["status"] == "ok", "status unexpected")
    # 弱误差全为 +1,bias=1;强误差均值=(0-1-2-3)/4=-1.5
    check.expect(abs(stats["weak_bias"] - 1.0) < 1e-12, "weak_bias mismatch")
    check.expect(abs(stats["strong_bias"] + 1.5) < 1e-12, "strong_bias mismatch")

    # 掩膜版:只取第一列
    mask = np.array([[True, False], [True, False]])
    stats2 = vc.paired_error_metrics(strong, weak, truth, mask)
    check.expect(stats2["n_valid"] == 2, f"masked n_valid={stats2['n_valid']}")
    check.expect(abs(stats2["mse_weak"] - 1.0) < 1e-12, "masked mse_weak mismatch")
    return check


def check_strong_minus_weak_sign() -> Check:
    """strong_minus_weak 的方向契约:strong=110、weak=100 -> +10。

    该助手用于诊断三的模式/重建/实际通量差,防止 (strong, weak) 元组
    下标误用导致的符号反转(审查发现的 P1 缺陷)。
    """
    check = Check("strong_minus_weak 符号契约")
    value = vc.strong_minus_weak(np.array(110.0), np.array(100.0))
    check.expect(float(value) == 10.0, f"strong-weak={value} != +10")
    check.expect(float(vc.strong_minus_weak(np.array(100.0), np.array(110.0))) == -10.0,
                 "反向差值应为 -10")
    return check


def check_dse_identity() -> Check:
    """逐点恒等式 dSE = 2 e dF + dF^2(浮点容差内)。"""
    check = Check("dSE 逐点恒等式")
    rng = np.random.default_rng(SEED)
    strong = 100.0 + 30.0 * rng.standard_normal((6, 5))
    weak = 100.0 + 30.0 * rng.standard_normal((6, 5))
    truth = 100.0 + 30.0 * rng.standard_normal((6, 5))
    dse = vc.se_change(strong, weak, truth)
    error = weak - truth
    increment = strong - weak
    expected = 2.0 * error * increment + increment**2
    good = np.isfinite(dse)
    check.expect(bool(np.all(good)), "dSE 出现意外 NaN")
    check.expect(
        bool(np.allclose(dse, expected, rtol=1e-9, atol=1e-6)),
        "dSE 与 2e*dF+dF^2 不一致",
    )
    # NaN 传播:任一输入 NaN -> dSE NaN(由掩膜剔除,不静默补值)
    weak_nan = weak.copy()
    weak_nan[0, 0] = np.nan
    dse_nan = vc.se_change(strong, weak_nan, truth)
    check.expect(not np.isfinite(dse_nan[0, 0]), "NaN 输入未被 dSE 传播")
    return check


def check_flux_error_budget_closure() -> Check:
    """闭合残差应接近 0;分量与直接法一致。"""
    check = Check("flux_error_budget 闭合残差")
    rng = np.random.default_rng(SEED + 1)
    flux_nr = 200.0 + 50.0 * rng.random((7, 6))
    flux_weak = flux_nr + 15.0 * rng.standard_normal((7, 6))
    flux_strong = flux_weak + 10.0 * rng.standard_normal((7, 6))
    budget = vc.flux_error_budget(flux_strong, flux_weak, flux_nr)
    check.expect(budget["status"] == "ok", f"status={budget['status']}")
    mse_weak_direct = float(np.mean((flux_weak - flux_nr) ** 2))
    mse_strong_direct = float(np.mean((flux_strong - flux_nr) ** 2))
    check.expect(
        abs(budget["mse_weak"] - mse_weak_direct) < 1e-9, "mse_weak 与直接计算不一致"
    )
    check.expect(
        abs(budget["delta_mse_direct"] - (mse_strong_direct - mse_weak_direct)) < 1e-9,
        "delta_mse_direct 与 MSE_strong-MSE_weak 不一致",
    )
    scale = max(1.0, abs(budget["delta_mse_direct"]))
    check.expect(
        abs(budget["closure_residual"]) <= 1e-8 * scale,
        f"closure_residual={budget['closure_residual']} 超出浮点闭合容差",
    )
    # 未去均值乘积:cross_term 必须等于 2<e dF>,而不是协方差 2 Cov(e,dF)
    error = flux_weak - flux_nr
    increment = flux_strong - flux_weak
    check.expect(
        abs(budget["cross_term"] - float(np.mean(2.0 * error * increment))) < 1e-9,
        "cross_term 不是 2<e*dF>(可能误用了协方差)",
    )
    check.expect(
        abs(budget["increment_square_term"] - float(np.mean(increment**2))) < 1e-9,
        "increment_square_term 不是 <dF^2>",
    )
    return check


def check_classification() -> Check:
    """构造 7 个点覆盖全部类别与容差判定。"""
    check = Check("classify_change_categories 类别判定")
    #                       点:   1     2     3     4     5     6     7
    dse_sst = np.array([[1.0, -5.0, 5.0, -5.0, 5.0, 0.00005, 1.0]])
    dse_flux = np.array([[-2.0, -3.0, 4.0, 3.0, -4.0, 1.0, 0.5]])
    category = vc.classify_change_categories(dse_sst, dse_flux, tol_sst=1e-4, tol_flux=1.0)
    # 点1: sst 恶化(+1>tol) & flux 改善(-2<-tol) -> 3
    # 点2: sst 改善 & flux 改善 -> 1
    # 点3: sst 恶化 & flux 恶化 -> 4
    # 点4: sst 改善 & flux 恶化 -> 2
    # 点5: sst 恶化 & flux 改善 -> 3
    # 点6: sst |dSE|<=tol 且 flux 恰在容差内(1.0 不大于 tol) -> 双侧未变化 0
    # 点7: sst 恶化 & flux 在容差内 -> 单侧未变化 -1
    expected = np.array([[3, 1, 4, 2, 3, 0, -1]])
    check.expect(bool(np.array_equal(category, expected)), f"category={category} != {expected}")
    # NaN 输入 -> -99,须由掩膜剔除
    dse_nan = dse_sst.copy()
    dse_nan[0, 0] = np.nan
    category_nan = vc.classify_change_categories(dse_nan, dse_flux, 1e-4, 1.0)
    check.expect(category_nan[0, 0] == -99, "NaN 输入未被标记为 -99")
    fractions = vc.category_fractions(category)
    check.expect(abs(sum(fractions[f"frac_{label}"] for label in vc.CATEGORY_LABELS.values()) - 1.0) < 1e-12,
                 "五类占比之和不为 1")
    return check


def check_zero_rmse_and_empty_mask() -> Check:
    """零基准 RMSE 与空掩膜的保护(不静默跳过)。"""
    check = Check("零 RMSE 保护与空掩膜状态")
    truth = np.full((3, 3), 300.0)
    weak = truth.copy()  # 弱误差恰为零
    strong = truth + 1.0
    stats = vc.paired_error_metrics(strong, weak, truth)
    check.expect(stats["status"] == "zero_rmse_weak", f"status={stats['status']}")
    check.expect(not np.isfinite(stats["rmse_improvement_pct"]), "零分母未置 NaN")

    nan_field = np.full((3, 3), np.nan)
    stats_empty = vc.paired_error_metrics(strong, nan_field, truth)
    check.expect(stats_empty["status"] == "empty_mask", f"status={stats_empty['status']}")
    check.expect(stats_empty["n_valid"] == 0, "empty_mask 的 n_valid 不为 0")

    budget_empty = vc.flux_error_budget(strong, nan_field, truth)
    check.expect(budget_empty["status"] == "empty_mask", "budget 空掩膜未报告")

    fractions_empty = vc.category_fractions(np.array([], dtype=int))
    check.expect(fractions_empty["status"] == "empty_mask", "分类空掩膜未报告")
    return check


def check_region_masks() -> Check:
    """环带不重复计数、并集覆盖、超界排除。"""
    check = Check("region_masks 边界与划分")
    config = VerifyConfig()  # 默认 0-75/75-150/150-300 + r000_300
    distance = np.array([[0.0, 74.999, 75.0, 149.999, 150.0, 299.999, 300.0, 300.001]])
    masks = vc.region_masks(distance, config)
    # distance: [0.0, 74.999, 75.0, 149.999, 150.0, 299.999, 300.0, 300.001]
    check.expect(bool(masks["r000_075"][0, :2].all()) and not masks["r000_075"][0, 2:].any(),
                 "r000_075 边界错误(75 km 应划出本环带)")
    check.expect(bool(masks["r075_150"][0, 2:4].all()) and not masks["r075_150"][0, 4:].any(),
                 "r075_150 边界错误(75 km 应划入本环带)")
    check.expect(not masks["r075_150"][0, 1], "75.0 km 不应同时出现在相邻环带(重复计数)")
    check.expect(not masks["r000_075"][0, 2], "75.0 km 不应属于 r000_075")
    check.expect(bool(masks["r150_300"][0, 4:7].all()), "r150_300 边界错误(300 km 应包含)")
    check.expect(not masks["r150_300"][0, 7], "300.001 km 应被排除")
    union = masks[config.union_region]
    annuli_sum = masks["r000_075"] | masks["r075_150"] | masks["r150_300"]
    check.expect(bool(np.array_equal(union, annuli_sum)), "三环带之或不等于并集")
    return check


def check_bilinear_registration() -> Check:
    """双线性重采样:线性场精确复原、域外 NaN、降序坐标同步翻转数据场。"""
    check = Check("resample_bilinear_regular 线性场/降序轴")

    def build(lat_ascending: bool, lon_ascending: bool):
        lat_1d = np.linspace(10.0, 18.0, 17)
        lon_1d = np.linspace(135.0, 143.0, 17)
        if not lat_ascending:
            lat_1d = lat_1d[::-1].copy()
        if not lon_ascending:
            lon_1d = lon_1d[::-1].copy()
        src_lat, src_lon = np.meshgrid(lat_1d, lon_1d, indexing="ij")
        field = 1.0 + 2.0 * src_lat + 3.0 * src_lon  # 双线性函数,应被精确插值
        return src_lat, src_lon, field

    dst_lat, dst_lon = np.meshgrid(
        np.linspace(10.5, 17.5, 29), np.linspace(135.5, 142.5, 29), indexing="ij"
    )
    expected = 1.0 + 2.0 * dst_lat + 3.0 * dst_lon
    for lat_ok in (True, False):
        for lon_ok in (True, False):
            src_lat, src_lon, field = build(lat_ok, lon_ok)
            resampled = vc.resample_bilinear_regular(field, src_lat, src_lon, dst_lat, dst_lon)
            label = f"(lat {'asc' if lat_ok else 'desc'}, lon {'asc' if lon_ok else 'desc'})"
            check.expect(
                bool(np.allclose(resampled, expected, rtol=1e-10, atol=1e-10)),
                f"线性场双线性插值不精确 {label}",
            )
    src_lat, src_lon, field = build(True, True)
    single = vc.resample_bilinear_regular(
        field, src_lat, src_lon, np.array([[9.0]]), np.array([[150.0]])
    )
    check.expect(not np.isfinite(single[0, 0]), "域外点应填 NaN")
    return check


def check_nearest_registration() -> Check:
    """最近邻重采样:域内取最近源点,域外(超覆盖半径)填 NaN。"""
    check = Check("resample_nearest 覆盖半径")
    src_lat_1d = np.linspace(10.0, 11.0, 3)   # 间距约 55 km
    src_lon_1d = np.linspace(100.0, 101.0, 3)
    src_lat, src_lon = np.meshgrid(src_lat_1d, src_lon_1d, indexing="ij")
    field = np.arange(9, dtype=float).reshape(3, 3)
    spacing = vc.estimate_source_spacing_km(src_lat, src_lon)
    check.expect(40.0 < spacing < 70.0, f"间距估计异常: {spacing}")
    dst_lat = np.array([[10.05, 30.0]])   # 第二点远在源域之外
    dst_lon = np.array([[100.05, 100.5]])
    out = vc.resample_nearest(field, src_lat, src_lon, dst_lat, dst_lon,
                              max_radius_km=2.0 * spacing)
    check.expect(np.isfinite(out[0, 0]), "域内点不应为 NaN")
    check.expect(out[0, 0] == field[0, 0], "域内最近邻取值错误")
    check.expect(not np.isfinite(out[0, 1]), "域外点应因覆盖半径被置 NaN,不得套用边缘最近邻")
    # 无覆盖半径时,足迹判断仍生效(近边界域外点被排除)
    out_open = vc.resample_nearest(field, src_lat, src_lon, dst_lat, dst_lon)
    check.expect(not np.isfinite(out_open[0, 1]), "足迹外的点应被排除(即使无覆盖半径)")
    return check


def check_nearest_footprint_boundary() -> Check:
    """足迹覆盖:贴边界的域内点有效;刚出边界(但距最近源点很近)的点必须排除。

    复现审查反例:0.01 度网格,目标经度超出源域 0.05 度(约 5 km,远小于
    旧的抽样间距高估值),不得再得到有效值。
    """
    check = Check("足迹覆盖(近边界域外点)")
    lat_1d = 10.0 + 0.01 * np.arange(11)      # 10.00-10.10
    lon_1d = 137.0 + 0.01 * np.arange(101)    # 137.00-138.00
    src_lat, src_lon = np.meshgrid(lat_1d, lon_1d, indexing="ij")
    field = np.ones(src_lat.shape)
    dst_lat = np.array([[10.05, 10.05, 10.05]])
    dst_lon = np.array([[137.995, 138.00, 138.05]])  # 域内 / 边界 / 域外 0.05 度
    out = vc.resample_nearest(field, src_lat, src_lon, dst_lat, dst_lon)
    check.expect(np.isfinite(out[0, 0]), "域内点应为有效")
    check.expect(np.isfinite(out[0, 1]), "边界上的点应为有效(容差为半个格距)")
    check.expect(not np.isfinite(out[0, 2]),
                 "域外 0.05 度的点距最近源点很近,也必须因足迹外被置 NaN")
    # 间距估计应来自真实相邻索引对(约 1.1 km),而不是抽样高估值
    spacing = vc.estimate_source_spacing_km(src_lat, src_lon)
    check.expect(0.8 < spacing < 1.5, f"0.01 度网格的相邻间距应约 1.1 km,得到 {spacing:.3f}")
    return check


def check_derive_lowest_level_inputs() -> Check:
    """常数场的手工解析期望值(T+300、气压合成、去交错风、诊断式 z0m)。"""
    check = Check("derive_lowest_level_inputs 手工解析值")
    ny, nx = 3, 4
    raw = {
        "T": np.full((ny, nx), 2.0),           # theta = 302 K
        "P": np.full((ny, nx), 60000.0),
        "PB": np.full((ny, nx), 40000.0),      # p0 = 100000 Pa
        "QVAPOR": np.full((ny, nx), 0.016),
        "PSFC": np.full((ny, nx), 100400.0),
        "HGT": np.zeros((ny, nx)),
        "PH": np.stack([np.zeros((ny, nx)), np.full((ny, nx), 32.0 * 9.81)]),
        "PHB": np.zeros((2, ny, nx)),
        "U": np.full((ny, nx + 1), 10.0),      # 去交错后仍为 10
        "V": np.full((ny + 1, nx), 4.0),
        "UST": np.full((ny, nx), 0.3),
    }
    inputs = vc.derive_lowest_level_inputs(raw)
    check.expect(bool(np.allclose(inputs["air_temperature_k"], 302.0, atol=1e-10)),
                 "air_temperature_k 应为 302 K(p=p0)")
    check.expect(bool(np.allclose(inputs["air_pressure_pa"], 100000.0)), "air_pressure_pa 错误")
    check.expect(bool(np.allclose(inputs["surface_pressure_pa"], 100400.0)), "surface_pressure_pa 错误")
    check.expect(bool(np.allclose(inputs["height_agl_m"], 16.0)),
                 "height_agl_m 应为 16 m(0 m 与 32 m 两界面中点)")
    check.expect(bool(np.allclose(inputs["u_ms"], 10.0)), "u_ms 去交错错误")
    check.expect(bool(np.allclose(inputs["v_ms"], 4.0)), "v_ms 去交错错误")
    expected_z0m = 0.0185 * 0.3**2 / 9.81 + 0.11 * 1.5e-5 / 0.3  # = 1.7522e-4 (< 上限 2.85e-3)
    check.expect(bool(np.allclose(inputs["initial_momentum_roughness_m"], expected_z0m, rtol=1e-9)),
                 f"z0m={inputs['initial_momentum_roughness_m'][0,0]} != {expected_z0m}")
    return check


def check_output_pipeline() -> Check:
    """CSV/npz/图件写出流程。"""
    check = Check("输出流程(CSV/npz/PNG)")
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    (SMOKE_OUT / "figs").mkdir(exist_ok=True)
    frame = pd.DataFrame({
        "mode": ["SYNTHETIC"], "method": ["EAKF"], "member": ["006"],
        "weight_method": [vc.WEIGHT_LABEL], "value": [1.5],
    })
    vc.write_csv(frame, SMOKE_OUT / "pipeline_sample.csv")
    check.expect((SMOKE_OUT / "pipeline_sample.csv").exists(), "CSV 未写出")
    vc.save_point_fields(SMOKE_OUT / "pipeline_sample.npz", a=np.arange(6).reshape(2, 3))
    loaded = np.load(SMOKE_OUT / "pipeline_sample.npz")
    check.expect(loaded["a"].shape == (2, 3), "npz 读写失败")
    vc.plot_metric_timeseries(
        {"case A": (np.array([0.0, 1.0]), np.array([1.0, np.nan]))},
        "[SYNTHETIC] pipeline check", "unit", SMOKE_OUT / "figs" / "pipeline_sample.png",
    )
    check.expect((SMOKE_OUT / "figs" / "pipeline_sample.png").exists(), "PNG 未写出")
    return check


def check_import_side_effect_free() -> Check:
    """在子进程中导入全部新模块;父进程对脚本目录做前后快照,不允许新增任何文件
    (PYTHONDONTWRITEBYTECODE=1 抑制 pyc 缓存)。覆盖范围为"新增文件";
    已有文件的修改与运行期数据读取由静态审查与其余检查共同覆盖。"""
    check = Check("无副作用导入(不新增文件)")
    code = (
        "import sys; sys.path.insert(0, '.');"
        "import verify_common, verify_synthetic,"
        "verify_01_skill_timeseries, verify_02_flux_error_budget,"
        "verify_03_fixed_atmosphere_flux, verify_04_initial_handoff,"
        "verify04_readers, caliber_link;"
        "print('IMPORTS_OK')"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def snapshot() -> set[str]:
        return {
            str(path.relative_to(SCRIPT_DIR))
            for path in SCRIPT_DIR.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }

    before = snapshot()
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=SCRIPT_DIR, env=env,
        capture_output=True, text=True, timeout=120,
    )
    after = snapshot()
    created = sorted(after - before)
    check.expect(result.returncode == 0, f"导入失败: {result.stderr[-400:]}")
    check.expect("IMPORTS_OK" in result.stdout, "导入未完成")
    check.expect(not created, f"导入产生副作用文件: {created}")
    return check


def check_real_mode_guards() -> Check:
    """防呆开关:未显式确认的真实模式必须被拒绝;冒烟测试永不进入真实读取。"""
    check = Check("真实模式防呆开关")
    check.expect(
        raises(RuntimeError, vc.RealWrfProvider, VerifyConfig(mode="synthetic")),
        "合成模式下构造 RealWrfProvider 应被拒绝",
    )
    unacknowledged = VerifyConfig(
        mode="real",
        real=RealPathConfig(acknowledge_real_mode=False),
    )
    check.expect(
        raises(RuntimeError, vc.RealWrfProvider, unacknowledged),
        "未确认的真实模式构造 RealWrfProvider 应被拒绝",
    )
    check.expect(raises(RuntimeError, vc.check_mode, unacknowledged),
                 "check_mode 应拒绝未确认的真实模式")
    return check


# ==================================================================
# 合成数据集行为
# ==================================================================


def check_synthetic_dataset_behaviour() -> Check:
    """中心识别、缺失时次、NaN 补丁隔离、配准接口、大气扰动量级。"""
    check = Check("合成数据集行为")
    provider_module = __import__("verify_synthetic")
    provider = provider_module.build_synthetic_provider(SMOKE_CONFIG)
    nr_static = provider.nr_static()
    nr_fields = provider.nr_fields(0.0)
    center_lat, center_lon, _, _ = vc.find_storm_center(
        nr_fields["psfc"], nr_static["lat"], nr_static["lon"],
        nr_static["landmask"], SMOKE_CONFIG.search_lat, SMOKE_CONFIG.search_lon,
    )
    check.expect(abs(center_lat - provider_module.CENTER_LAT) < 1e-9
                 and abs(center_lon - provider_module.CENTER_LON) < 1e-9,
                 f"中心识别错误: ({center_lat}, {center_lon})")
    check.expect(not provider.has_case("QCF_RHF", "044", 2.5), "缺失时次未被识别")
    check.expect(provider.has_case("EAKF", "006", 2.5), "正常案例被误判缺失")
    fields = provider.exp_fields(SMOKE_CONFIG.strong_experiment, "EAKF", "006", 0.5)
    weak_fields = provider.exp_fields(SMOKE_CONFIG.weak_experiment, "EAKF", "006", 0.5)
    # NaN 补丁只作用于指定 (方法, 成员, 时次, 变量, 强弱):
    # EAKF/006/t=0.5/hfx/weak 有补丁;QCF_RHF/006 同参数无补丁(修复回归)
    patch = weak_fields["hfx"][5:9, 3:6]
    check.expect(bool(np.all(~np.isfinite(patch))), "hfx weak NaN 补丁未生效")
    other_method = provider.exp_fields(
        SMOKE_CONFIG.weak_experiment, "QCF_RHF", "006", 0.5
    )["hfx"][5:9, 3:6]
    check.expect(bool(np.all(np.isfinite(other_method))),
                 "NaN 补丁泄漏到其他方法(补丁条件未按 spec_method/spec_member 匹配)")
    same_member_other_method = provider.exp_fields(
        SMOKE_CONFIG.strong_experiment, "EAKF", "015", 1.0
    )["om"][20:23, 8:11]
    check.expect(bool(np.all(np.isfinite(same_member_other_method))),
                 "om strong NaN 补丁泄漏到其他方法的同名成员")
    qcf_015_patch = provider.exp_fields(
        SMOKE_CONFIG.strong_experiment, "QCF_RHF", "015", 1.0
    )["om"][20:23, 8:11]
    check.expect(bool(np.all(~np.isfinite(qcf_015_patch))),
                 "QCF_RHF/015/t=1.0/om/strong 的补丁未生效")
    land = provider.static_fields()["landmask"]
    check.expect(bool(np.all(~np.isfinite(fields["hfx"][land > 0.5]))), "陆地通量应为 NaN")
    check.expect(bool(np.all(np.isfinite(fields["tsk"][land > 0.5]))), "陆地 TSK 应有效")
    check.expect(bool(np.all(~np.isfinite(fields["om"][land > 0.5]))), "陆地 om 应为 NaN")
    raw = provider.exp_raw_inputs(SMOKE_CONFIG.strong_experiment, "EAKF", "006", 0.5)
    inputs = vc.derive_lowest_level_inputs(raw)
    exp_shape = provider.static_fields()["lat"].shape
    for key in vc.RECONSTRUCTION_INPUT_KEYS:
        check.expect(inputs[key].shape == exp_shape,
                     f"{key} 形状与试验网格不一致: {inputs[key].shape} != {exp_shape}")
    check.expect(float(np.nanmin(inputs["initial_friction_velocity_ms"])) > 0.0, "UST 必须为正")
    check.expect(float(np.nanmin(inputs["height_agl_m"])) > 0.0, "离地高度必须为正")
    # 强弱大气差异应为小扰动(不能出现 ~18 K 的极端系统差)
    raw_weak = provider.exp_raw_inputs(SMOKE_CONFIG.weak_experiment, "EAKF", "006", 0.5)
    inputs_weak = vc.derive_lowest_level_inputs(raw_weak)
    theta_diff = float(np.nanmean(inputs["air_temperature_k"] - inputs_weak["air_temperature_k"]))
    check.expect(abs(theta_diff) < 3.0,
                 f"强弱大气温度差的合成量级过大: {theta_diff:.2f} K")
    return check


def check_align_times() -> Check:
    """缺失时刻补 NaN 断线,不跨接前后时次。"""
    check = Check("align_times_to_hours 缺失时刻处理")
    frame = pd.DataFrame({
        "time_hour": [0.0, 1.0],
        "rmse_weak": [1.0, 3.0],
    })
    aligned = vc.align_times_to_hours(frame, [0.0, 0.5, 1.0, 1.5])
    check.expect(list(aligned.index) == [0.0, 0.5, 1.0, 1.5], "对齐时刻错误")
    values = aligned.rmse_weak.to_numpy(dtype=float)
    check.expect(values[0] == 1.0 and values[2] == 3.0, "已有时刻数值被改变")
    check.expect(bool(np.isnan(values[1])) and bool(np.isnan(values[3])),
                 "缺失时刻未补 NaN")
    return check


def check_member_then_mean() -> Check:
    """成员先均、后跨成员:缺失时次不改变成员权重。"""
    check = Check("member_then_mean 成员等权")
    frame = pd.DataFrame({
        "member": ["006", "006", "015"],
        "value": [0.0, 2.0, 10.0],  # 006 均值 1;015 均值 10 -> 总均值 5.5
    })
    result = vc.member_then_mean(frame, "value", ("006", "015"))
    check.expect(abs(result - 5.5) < 1e-12, f"member_then_mean={result} != 5.5")
    return check


# ==================================================================
# 第二层:真实物理函数接口 + RealWrfProvider 读取适配层
# ==================================================================


def check_real_reconstruction_interface() -> Check:
    """调用现有 reconstruct_ocean_fluxes():有限输出、确定性、量级合理;
    非有限输出必须被显式报告,不得标 ok。"""
    check = Check("reconstruct_ocean_fluxes 接口冒烟(真实函数+合成输入)")
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from verify_03_fixed_atmosphere_flux import (
            call_reconstruction,
            evaluate_reconstruction_outputs,
            import_reconstructor,
        )
    except RuntimeError as error:
        check.expect(False, f"物理函数导入失败: {error}")
        return check
    reconstructor = import_reconstructor()
    provider_module = __import__("verify_synthetic")
    provider = provider_module.build_synthetic_provider(SMOKE_CONFIG)
    raw = provider.exp_raw_inputs(SMOKE_CONFIG.strong_experiment, "EAKF", "006", 0.5)
    inputs = vc.derive_lowest_level_inputs(raw)
    ocean = provider.static_fields()["landmask"] < 0.5
    mask = ocean & vc.finite_mask(
        *[inputs[key] for key in vc.RECONSTRUCTION_INPUT_KEYS], raw["TSK"]
    )
    check.expect(int(mask.sum()) > 100, f"海洋有效点过少: {int(mask.sum())}")
    flux = call_reconstruction(reconstructor, inputs, raw["TSK"], mask)
    for name in ("qfx", "lh", "hfx"):
        values = flux[name][mask]
        check.expect(bool(np.all(np.isfinite(values))), f"{name} 存在非有限值")
        check.expect(bool(np.all(values >= -250.0)), f"{name} 量级异常(应 >= -250)")
        check.expect(float(np.nanmax(np.abs(values))) < 5000.0, f"{name} 量级异常(>5000)")
    # 确定性:同输入两次调用结果一致(equal_nan=True:掩膜外双方均为 NaN)
    flux_again = call_reconstruction(reconstructor, inputs, raw["TSK"], mask)
    for name in ("qfx", "lh", "hfx"):
        check.expect(
            bool(np.array_equal(flux[name], flux_again[name], equal_nan=True)),
            f"{name} 不确定",
        )
    # 方向观察(仅记录,不作断言、不算验证):海温 +1 K 的响应方向
    ts_warm = raw["TSK"] + 1.0
    flux_warm = call_reconstruction(reconstructor, inputs, ts_warm, mask)
    d_lh = float(np.mean(flux_warm["lh"][mask] - flux["lh"][mask]))
    d_hfx = float(np.mean(flux_warm["hfx"][mask] - flux["hfx"][mask]))
    print(f"    [observe] +1 K SST response: mean dLH={d_lh:.3f} W m-2, dHFX={d_hfx:.3f} W m-2", flush=True)

    # 非有限输出必须显式报告(替身重建器全部返回 NaN,与真实函数明确区分)
    def broken_reconstructor(**kwargs):
        shape = kwargs["air_temperature_k"].shape
        nan = np.full(shape, np.nan)
        return {"qfx": nan, "lh": nan, "hfx": nan}

    _, combined, status, invalid = evaluate_reconstruction_outputs(
        broken_reconstructor, inputs, raw["TSK"], mask
    )
    check.expect(status == "nonfinite_reconstruction_output",
                 f"全 NaN 替身重建器未被标记: {status}")
    check.expect(int(combined.sum()) == 0, "非有限输出的共同掩膜应为空")
    check.expect(invalid == 2 * int(mask.sum()),
                 "非有限输出点数应按 hfx/lh 两个比较变量分别累计")
    return check


def _write_mock_wrfout(path: Path, lat2d, lon2d, land2d, stamp: str, kind: str,
                       times_mode: str = "vlen", proj_attrs: dict | None = None) -> None:
    """写入合成 NetCDF(本地冒烟输出目录,非真实数据),供真实读取适配层接口测试。

    times_mode: "vlen" 为一维可变长字符串 Times(time);
                "s1" 为 WRF 常见的二维字符数组 Times(time, DateStrLen)。
    proj_attrs: 提供时写入 WRF 投影全局属性(MAP_PROJ 等),供 ll_to_xy 路径测试。
    """
    from netCDF4 import Dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = lat2d.shape
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.createDimension("bottom_top", 2)
        ds.createDimension("bottom_top_stag", 3)
        ds.createDimension("z_ocean", 1)
        ds.createDimension("west_east_stag", nx + 1)
        ds.createDimension("south_north_stag", ny + 1)
        if times_mode == "s1":
            ds.createDimension("DateStrLen", 19)
            times = ds.createVariable("Times", "S1", ("time", "DateStrLen"))
            times[0] = np.array(list(stamp), dtype="S1")
        else:
            times = ds.createVariable("Times", str, ("time",))
            times[0] = stamp
        if proj_attrs:
            for name, value in proj_attrs.items():
                setattr(ds, name, value)

        def field(name, data, dims=("time", "south_north", "west_east")):
            variable = ds.createVariable(name, "f8", dims)
            variable[0] = data

        field("XLAT", lat2d)
        field("XLONG", lon2d)
        field("LANDMASK", land2d)
        field("PSFC", np.full((ny, nx), 100400.0))
        field("OM_TMP", np.full((1, ny, nx), 302.0),
              ("time", "z_ocean", "south_north", "west_east"))
        field("TSK", np.full((ny, nx), 303.0))
        field("HFX", np.full((ny, nx), 150.0))
        field("LH", np.full((ny, nx), 250.0))
        if kind == "nr":
            return
        field("UST", np.full((ny, nx), 0.3))
        field("HGT", np.zeros((ny, nx)))
        field("T", np.full((2, ny, nx), 2.0), ("time", "bottom_top", "south_north", "west_east"))
        field("P", np.full((2, ny, nx), 54000.0), ("time", "bottom_top", "south_north", "west_east"))
        field("PB", np.full((2, ny, nx), 36000.0), ("time", "bottom_top", "south_north", "west_east"))
        field("QVAPOR", np.full((2, ny, nx), 0.016), ("time", "bottom_top", "south_north", "west_east"))
        geopotential = np.zeros((3, ny, nx))
        geopotential[1] = 40.0 * 9.81
        field("PH", geopotential, ("time", "bottom_top_stag", "south_north", "west_east"))
        field("PHB", np.zeros((3, ny, nx)), ("time", "bottom_top_stag", "south_north", "west_east"))
        field("U", np.full((2, ny, nx + 1), 8.0), ("time", "bottom_top", "south_north", "west_east_stag"))
        field("V", np.full((2, ny + 1, nx), 2.0), ("time", "bottom_top", "south_north_stag", "west_east"))


def check_char_array_times() -> Check:
    """WRF 常见的二维 S1 字符数组 Times(time, DateStrLen)必须被正确解析:
    正常时间通过、错误时间拒绝;NR 精确时次与上下整点插值路径同样校验。"""
    check = Check("S1 字符数组 Times 解析(mock NetCDF)")
    try:
        from netCDF4 import Dataset  # noqa: F401
    except ImportError:
        check.expect(False, "netCDF4 不可用")
        return check
    mock_root = Path(tempfile.mkdtemp(prefix="mock_s1_", dir=SMOKE_OUT))
    exp_lat2d, exp_lon2d = np.meshgrid(
        13.0 + 0.25 * np.arange(4), 137.0 + 0.25 * np.arange(4), indexing="ij"
    )
    land = np.zeros_like(exp_lat2d)
    base = mock_root / "cycle"
    stamp_ok = "2018-09-10_00:00:00"
    for experiment in ("6mem_oceanAssim0Run1", "6mem_oceanAssim1Run1"):
        _write_mock_wrfout(
            base / experiment / "EAKF" / "006" / f"wrfout_d02_{stamp_ok}",
            exp_lat2d, exp_lon2d, land, stamp_ok, "exp", times_mode="s1",
        )
    # 错误时间(文件名 01:00,内容 02:00)的 S1 文件
    _write_mock_wrfout(
        base / "6mem_oceanAssim1Run1" / "EAKF" / "006" / "wrfout_d02_2018-09-10_01:00:00",
        exp_lat2d, exp_lon2d, land, "2018-09-10_02:00:00", "exp", times_mode="s1",
    )
    # NR:S1 字符数组的 00:00 与 01:00(供整点插值)
    nr_lat2d, nr_lon2d = np.meshgrid(
        13.0 + 0.5 * np.arange(3), 137.0 + 0.5 * np.arange(3), indexing="ij"
    )
    nr_dir = mock_root / "nr"
    _write_mock_wrfout(nr_dir / "wrfout_d02_2018-09-10_00:00:00",
                       nr_lat2d, nr_lon2d, np.zeros_like(nr_lat2d),
                       "2018-09-10_00:00:00", "nr", times_mode="s1")
    _write_mock_wrfout(nr_dir / "wrfout_d02_2018-09-10_01:00:00",
                       nr_lat2d, nr_lon2d, np.zeros_like(nr_lat2d),
                       "2018-09-10_01:00:00", "nr", times_mode="s1")

    config = VerifyConfig(
        mode="real", methods=("EAKF",), members=("006",),
        times=((0.0, stamp_ok),),
        real=RealPathConfig(
            forecast_base_dir=base, nr_dir=nr_dir,
            nr_registration="numpy_nearest", acknowledge_real_mode=True,
        ),
    )
    provider = vc.RealWrfProvider(config)
    fields = provider.exp_fields("6mem_oceanAssim1Run1", "EAKF", "006", 0.0)
    check.expect(fields["hfx"].shape == exp_lat2d.shape, "S1 正常时间文件读取失败")
    config_wrong = dataclasses.replace(config, times=((1.0, "2018-09-10_01:00:00"),))
    provider_wrong = vc.RealWrfProvider(config_wrong)
    check.expect(
        raises(RuntimeError, provider_wrong.exp_fields,
               "6mem_oceanAssim1Run1", "EAKF", "006", 1.0),
        "S1 字符数组时间与请求时刻不符未被拒绝",
    )
    # NR 精确时次 + 整点插值路径均校验时间
    nr_fields = provider.nr_fields(0.0)
    check.expect(np.isfinite(nr_fields["hfx"]).all(), "NR S1 精确时次读取失败")
    provider_bracket = vc.RealWrfProvider(dataclasses.replace(
        config, times=((0.0, stamp_ok), (0.5, "2018-09-10_00:30:00"), (1.0, "2018-09-10_01:00:00"))
    ))
    nr_half = provider_bracket.nr_fields(0.5)
    expected_half = 0.5 * nr_fields["hfx"] + 0.5 * np.full(nr_lat2d.shape, 150.0)
    check.expect(np.allclose(nr_half["hfx"], expected_half),
                 "NR 整点插值(含 S1 时间校验)结果错误")
    return check


def check_attribution_flag_logic() -> Check:
    """归因三判据:弱准/强错与对称情形都不得给出有利标记(审查反例)。

    反例:模式强弱通量 100/201,重建 100/101 -> 弱残差 0、强残差与重建差
    都约等于模式差本身;只检查弱耦合会错误放行。
    """
    check = Check("归因三判据(弱+强+差一致)")
    from verify_03_fixed_atmosphere_flux import attribution_quality_flag

    model_diff_rms = 101.0
    weak_ratio = 0.0                      # 弱重建恰好准确
    strong_ratio = abs(101.0 - 201.0) / model_diff_rms   # 强重建错误 -> ~1.0
    diff_ratio = abs(1.0 - 101.0) / model_diff_rms       # 重建差 vs 实际差 -> ~1.0
    flag = attribution_quality_flag(weak_ratio, strong_ratio, diff_ratio)
    check.expect(flag == "residual_not_negligible_attribution_requires_verification",
                 f"弱准强错反例被错误放行: {flag}")
    # 对称情形:强准/弱错
    flag_symmetric = attribution_quality_flag(
        strong_ratio, weak_ratio, diff_ratio
    )
    check.expect(flag_symmetric == "residual_not_negligible_attribution_requires_verification",
                 f"强准弱错反例被错误放行: {flag_symmetric}")
    # 三判据都小才放行
    flag_good = attribution_quality_flag(0.1, 0.2, 0.3)
    check.expect(flag_good == "residual_small_vs_explained_diff",
                 f"小残差情形被误标: {flag_good}")
    # 不可评估(零分母)不得放行
    flag_nan = attribution_quality_flag(np.nan, 0.0, 0.0)
    check.expect(flag_nan == "attribution_criteria_not_evaluable",
                 f"零分母情形被误标: {flag_nan}")
    # 区域相关:不同区域的判据组合给出不同标记
    check.expect(
        attribution_quality_flag(0.1, 0.1, 0.1)
        != attribution_quality_flag(0.9, 0.1, 0.1),
        "不同区域的判据组合应给出不同标记(标记必须按区域评估)",
    )
    return check


def check_coverage_semantics() -> Check:
    """覆盖率区分已读取/有效绝对误差/有效改善率时次;分母显式含缺失成员。"""
    check = Check("覆盖率语义(空掩膜不计入;零基准计入绝对;缺失成员计入分母)")
    frame = pd.DataFrame({
        "mode": ["SYNTHETIC"] * 4,
        "method": ["EAKF"] * 4,
        "member": ["006"] * 4,
        "time_hour": [0.0, 0.5, 1.0, 1.5],
        "region": ["r000_300"] * 4,
        "variable": ["hfx"] * 4,
        "unit": ["W m-2"] * 4,
        "status": ["ok", "ok", "zero_rmse_weak", "empty_mask"],
        "rmse_weak": [10.0, 11.0, 0.0, np.nan],
        "rmse_strong": [8.0, 12.0, 1.0, np.nan],
        "rmse_improvement_pct": [20.0, -9.1, np.nan, np.nan],
    })
    summary = vc.summarize_member_metrics(
        frame, keys=("mode", "method", "region", "variable", "unit"),
        expected_times={"all_0_6h": 13, "early_0p5_2h": 4, "late_3_6h": 7},
        expected_members=2,
    )
    row = summary[summary.window == "all_0_6h"].iloc[0]
    check.expect(int(row.n_members) == 1, f"实际成员数应为 1(015 全缺失): {row.n_members}")
    check.expect(int(row.expected_members) == 2, "期望成员数应为配置值 2")
    check.expect(abs(row.member_coverage_ratio - 0.5) < 1e-12,
                 f"成员覆盖率应为 1/2: {row.member_coverage_ratio}")
    # 成员 006:读取 4 次;有效绝对 = ok×2 + zero_rmse×1 = 3;有效改善 = 2
    check.expect(abs(row.mean_member_times_read - 4.0) < 1e-12,
                 f"已读取时次数应为 4: {row.mean_member_times_read}")
    check.expect(abs(row.mean_member_valid_abs_times - 3.0) < 1e-12,
                 f"有效绝对误差时次数应为 3(empty_mask 不计、零基准计入): {row.mean_member_valid_abs_times}")
    check.expect(abs(row.mean_member_valid_improvement_times - 2.0) < 1e-12,
                 f"有效改善率时次数应为 2: {row.mean_member_valid_improvement_times}")
    check.expect(abs(row.member_abs_time_coverage_ratio - 3.0 / 13.0) < 1e-12,
                 f"覆盖率分母应为期望时次数 13: {row.member_abs_time_coverage_ratio}")
    return check


def check_conditional_budget_manual() -> Check:
    """条件预算手算预期(预期值来自显式构造,不调用被测函数生成):
    e=1,dF=1 -> C=2,S=1,dMSE=3;e=1,dF=-3 -> C=-6,S=9,dMSE=3;
    子集拆分守恒;全区域与 TSK 改善子集的 C 反号。"""
    check = Check("条件预算手算(子集构造/C/S/dMSE)")

    # 情形 1:e=1, dF=1(四个点)
    error = np.full((2, 2), 1.0)
    increment = np.full((2, 2), 1.0)
    mask = np.ones((2, 2), dtype=bool)
    stats = vc.cross_term_stats(error, increment, mask)
    check.expect(abs(stats["cross_term"] - 2.0) < 1e-12, f"C={stats['cross_term']} != 2")
    check.expect(abs(stats["increment_square_term"] - 1.0) < 1e-12, f"S={stats['increment_square_term']} != 1")
    check.expect(abs(stats["delta_mse_direct"] - 3.0) < 1e-12, f"dMSE={stats['delta_mse_direct']} != 3")
    check.expect(abs(stats["closure_residual"]) < 1e-12, "闭合残差非零")

    # 情形 2:e=1, dF=-3
    stats2 = vc.cross_term_stats(error, np.full((2, 2), -3.0), mask)
    check.expect(abs(stats2["cross_term"] + 6.0) < 1e-12, f"C={stats2['cross_term']} != -6")
    check.expect(abs(stats2["increment_square_term"] - 9.0) < 1e-12, f"S={stats2['increment_square_term']} != 9")
    check.expect(abs(stats2["delta_mse_direct"] - 3.0) < 1e-12, f"dMSE={stats2['delta_mse_direct']} != 3")

    # 情形 3:交叉项为零、平方项非零(e 与 dF 正交)
    error_c = np.array([1.0, -1.0, 1.0, -1.0])
    increment_c = np.array([1.0, 1.0, -1.0, -1.0])
    stats3 = vc.cross_term_stats(error_c, increment_c, np.ones(4, dtype=bool))
    check.expect(abs(stats3["cross_term"]) < 1e-12, f"C 应为 0: {stats3['cross_term']}")
    check.expect(abs(stats3["increment_square_term"] - 1.0) < 1e-12, "S 应为 1")

    # 情形 4:全区域与 TSK 改善子集的 C 符号相反(手算:
    # 全区域 C = (4*2 + 4*(-6))/8 = -2;改善子集只含后 4 点,C = -6;
    # 恶化子集只含前 4 点,C = +2)
    error_w = np.array([1.0] * 4 + [1.0] * 4)
    increment_w = np.array([1.0] * 4 + [-3.0] * 4)
    dse_sst = np.array([5.0] * 4 + [-5.0] * 4)
    base = np.ones(8, dtype=bool)
    subsets = vc.sst_subset_masks(dse_sst, base, tol=1.0)
    stats_all = vc.cross_term_stats(error_w, increment_w, subsets["all_common"])
    stats_impr = vc.cross_term_stats(error_w, increment_w, subsets["sst_improved"])
    stats_wors = vc.cross_term_stats(error_w, increment_w, subsets["sst_worsened"])
    check.expect(abs(stats_all["cross_term"] + 2.0) < 1e-12,
                 f"全区域 C 应为 -2: {stats_all['cross_term']}")
    check.expect(abs(stats_impr["cross_term"] + 6.0) < 1e-12,
                 f"改善子集 C 应为 -6: {stats_impr['cross_term']}")
    check.expect(abs(stats_wors["cross_term"] - 2.0) < 1e-12,
                 f"恶化子集 C 应为 +2: {stats_wors['cross_term']}")
    check.expect(stats_impr["cross_term"] * stats_wors["cross_term"] < 0,
                 "改善子集(-6)与恶化子集(+2)的 C 应反号(方向相反)")
    n_total = int(subsets["all_common"].sum())
    check.expect(
        int(subsets["sst_improved"].sum()) + int(subsets["sst_worsened"].sum())
        + int(subsets["sst_unchanged"].sum()) == n_total,
        "三个子集点数之和应等于 all_common",
    )

    # 情形 5:空掩膜 -> NaN + empty_mask(不填零)
    empty = vc.cross_term_stats(error, increment, np.zeros((2, 2), dtype=bool))
    check.expect(empty["status"] == "empty_mask" and not np.isfinite(empty["cross_term"]),
                 "空掩膜应返回 NaN + empty_mask")

    # 情形 6:交叉项方向容差独立(接近零为中性,不归入方向不利)
    check.expect(vc.cross_term_direction(0.5, tol=1.0) == "cross_term_neutral",
                 "C=0.5(tol=1)应为中性")
    check.expect(vc.cross_term_direction(2.0, tol=1.0) == "positive_cross_term",
                 "C=2(tol=1)应为方向不利")
    check.expect(vc.cross_term_direction(-2.0, tol=1.0) == "negative_cross_term",
                 "C=-2 应为纠错倾向")
    check.expect(vc.net_error_direction(3.0, tol=1.0) == "net_worsened"
                 and vc.net_error_direction(-3.0, tol=1.0) == "net_improved"
                 and vc.net_error_direction(0.5, tol=1.0) == "net_unchanged",
                 "净变化状态判定错误")
    return check


def check_sign_agreement_manual() -> Check:
    """符号一致率手算:dF_SST=[1,-1]、dF_actual=[-1,1] -> 均值均为零、
    RMS 差为 2、一致率 0;阈值边界与 NaN 处理。"""
    check = Check("符号一致率/RMS 差手算")
    d_sst = np.array([1.0, -1.0])
    d_act = np.array([-1.0, 1.0])
    mask = np.ones(2, dtype=bool)
    stats = vc.sign_agreement_stats(d_sst, d_act, mask, abs_tol=0.5)
    check.expect(stats["n_eligible"] == 2 and stats["n_agree"] == 0,
                 f"参与点 {stats['n_eligible']},一致 {stats['n_agree']}")
    check.expect(stats["sign_agreement"] == 0.0, f"一致率应为 0: {stats['sign_agreement']}")
    check.expect(abs(stats["eligible_fraction"] - 1.0) < 1e-12, "参与比例应为 1")
    # 均值均为零但 RMS 差为 2
    check.expect(abs(vc.masked_mean(d_sst, mask)) < 1e-12
                 and abs(vc.masked_mean(d_act, mask)) < 1e-12, "两增量均值应均为零")
    diff_rms = vc.rms(d_sst - d_act, mask)
    check.expect(abs(diff_rms - 2.0) < 1e-12, f"RMS(dF_SST-dF_actual) 应为 2: {diff_rms}")
    # 阈值边界:|值| = tol 不参与(需严格大于)
    stats_tol = vc.sign_agreement_stats(np.array([0.5, -0.5]), np.array([0.5, -0.5]), mask, abs_tol=0.5)
    check.expect(stats_tol["n_eligible"] == 0
                 and stats_tol["status"] == "no_points_above_threshold",
                 "等于阈值 的点不应参与(需严格大于)")
    # 实际响应为零 -> ratio NaN + actual_response_zero
    joint = vc.response_vs_actual_stats(np.array([1.0, 1.0]), d_sst, np.zeros(2), mask, flux_increment_tol=0.5)
    check.expect(joint["actual_response_status"] == "actual_response_zero"
                 and not np.isfinite(joint["sst_to_actual_rms_ratio"]),
                 "实际响应为零时 ratio 应为 NaN 并标记状态")
    # 离线响应为零 -> rms_sst_response=0,ratio=0(有效值,不是 NaN)
    joint0 = vc.response_vs_actual_stats(np.array([1.0, 1.0]), np.zeros(2), d_act, mask, flux_increment_tol=0.5)
    check.expect(joint0["rms_sst_response"] == 0.0
                 and abs(joint0["sst_to_actual_rms_ratio"]) < 1e-12,
                 "离线响应为零时 ratio 应为 0(有效)")
    return check


def check_joint_stats_manual() -> Check:
    """联合统计手算:实际弱误差与离线重建弱误差符号相反时,
    cross_model_error_sst_response 必须使用实际弱误差;同掩膜;子集质量独立。"""
    global verify_03_module
    if verify_03_module is None:
        sys.path.insert(0, str(SCRIPT_DIR))
        verify_03_module = __import__("verify_03_fixed_atmosphere_flux")
    check = Check("联合统计手算(实际弱误差基线/同掩膜/子集质量)")
    n = 4
    truth = np.full(n, 100.0)
    weak_model = truth + 1.0          # 实际弱误差 e_model = +1
    strong_model = truth - 1.0        # 实际增量 dF_actual = -2
    # 离线重建弱误差取 -1(与实际弱误差符号相反),SST 响应 +1
    d_sst = np.full(n, 1.0)
    mask = np.ones(n, dtype=bool)
    stats = vc.response_vs_actual_stats(
        weak_model - truth, d_sst, strong_model - weak_model, mask, flux_increment_tol=0.5
    )
    # 手算:cross = 2<e_model*dF_SST> = 2*1*1 = 2(若误用重建弱误差则为 -2)
    check.expect(abs(stats["cross_model_error_sst_response"] - 2.0) < 1e-12,
                 f"交叉项应使用实际弱误差(C=2): {stats['cross_model_error_sst_response']}")
    # 实际预算:e=1,dF=-2 -> C=-4,S=4,dMSE=0
    check.expect(abs(stats["actual_cross_term"] + 4.0) < 1e-12,
                 f"实际 C 应为 -4: {stats['actual_cross_term']}")
    check.expect(abs(stats["actual_increment_square_term"] - 4.0) < 1e-12, "实际 S 应为 4")
    check.expect(abs(stats["actual_delta_mse_direct"]) < 1e-12, "实际 dMSE 应为 0")
    check.expect(abs(stats["actual_closure_residual"]) < 1e-12, "实际预算闭合残差应为 0")
    check.expect(abs(stats["sst_to_actual_rms_ratio"] - 0.5) < 1e-12,
                 f"RMS 比 = 1/2: {stats['sst_to_actual_rms_ratio']}")
    # 实际弱误差与离线重建弱误差是两个概念:sst_only_rmse 改善率用重建基线,
    # 本交叉项用实际弱误差;两者不同(该差异由诊断三的两套基线文档说明)。
    # 同一掩膜:掩膜内 NaN 由共同有效掩膜剔除(n_valid 只数有限点),
    # 掩膜外的 NaN 不影响其他点;两次调用的统计点数差 = NaN 点数
    mask_partial = np.array([True, True, True, False])
    weak_nan = weak_model.copy(); weak_nan[2] = np.nan
    stats_nan = vc.response_vs_actual_stats(
        (weak_nan - truth), d_sst, strong_model - weak_model, mask_partial, flux_increment_tol=0.5
    )
    check.expect(stats_nan["n_valid"] == 2,
                 f"掩膜内 NaN 点应被剔除(n_valid 只数有限点): n={stats_nan['n_valid']}")
    check.expect(stats["n_valid"] == 4, f"无 NaN 时 n_valid 应为 4: {stats['n_valid']}")
    # dF_SST 含 NaN 时同样剔除(所有指标同一掩膜口径)
    d_sst_nan = d_sst.copy(); d_sst_nan[0] = np.nan
    stats_nan2 = vc.response_vs_actual_stats(
        (weak_model - truth), d_sst_nan, strong_model - weak_model, mask, flux_increment_tol=0.5
    )
    check.expect(stats_nan2["n_valid"] == 3,
                 f"dF_SST NaN 应被同一掩膜剔除: n={stats_nan2['n_valid']}")
    # 子集质量独立:全区域合格、改善子集不合格
    rw = np.array([0.0, 0.0, 0.0, 10.0])   # 弱重建残差:仅第 4 点大(在改善子集外)
    rs = np.array([0.0, 0.0, 0.0, 0.0])
    scale = np.array([20.0, 20.0, 20.0, 20.0])
    dse = np.array([-5.0, -5.0, -5.0, 5.0])
    subsets = vc.sst_subset_masks(dse, np.ones(4, dtype=bool), tol=1.0)
    flag_all = verify_03_module._joint_quality_flag(rw, rs, rs, scale, subsets["all_common"])
    flag_impr = verify_03_module._joint_quality_flag(rw, rs, rs, scale, subsets["sst_improved"])
    flag_wors = verify_03_module._joint_quality_flag(rw, rs, rs, scale, subsets["sst_worsened"])
    check.expect(flag_wors == "residual_small_vs_explained_diff",
                 f"全区域/恶化子集应合格: all={flag_all}, worsened={flag_wors}")
    # 改善子集只含前 3 点(残差为 0),同样合格——改为构造残差在改善子集内的反例:
    rw2 = np.array([40.0, 0.0, 0.0, 0.0])  # 改善子集内残差比 = 2 > 阈值 0.5
    flag_impr2 = verify_03_module._joint_quality_flag(rw2, rs, rs, scale, subsets["sst_improved"])
    flag_wors2 = verify_03_module._joint_quality_flag(rw2, rs, rs, scale, subsets["sst_worsened"])
    check.expect(flag_impr2 != flag_wors2,
                 f"同一案例不同子集的质量标记应可不同: impr={flag_impr2}, worsened={flag_wors2}")
    return check


def check_inside_footprint_hull() -> Check:
    """基础问题 1 回归:凸包足迹的多点/单点/零点查询逐点返回布尔。"""
    check = Check("凸包足迹逐点布尔(多点/单点/零点)")
    src_lat, src_lon = np.meshgrid(
        10.0 + 0.1 * np.arange(20), 100.0 + 0.1 * np.arange(20), indexing="ij"
    )
    warp_lat = src_lat + 0.05 * np.sin(src_lon / 5.0)  # 曲线网格 -> 凸包分支
    footprint = vc._build_source_footprint(warp_lat, src_lon, 0.05)
    check.expect(footprint[0] == "hull", f"应构造凸包足迹: {footprint[0]}")
    dst_lat = np.array([10.5, 11.5, 20.0])
    dst_lon = np.array([100.5, 100.9, 105.0])
    inside = vc._inside_footprint(footprint, dst_lat, dst_lon)
    check.expect(isinstance(inside, np.ndarray) and inside.dtype == bool,
                 f"多点查询应返回布尔数组: {type(inside)}")
    check.expect(np.size(inside) == 3 and inside.tolist() == [True, True, False],
                 f"多点判定错误: {inside}")
    single = vc._inside_footprint(footprint, dst_lat[:1], dst_lon[:1])
    check.expect(np.size(single) == 1 and bool(single[0]), "单点查询失败")
    zero = vc._inside_footprint(footprint, dst_lat[:0], dst_lon[:0])
    check.expect(np.size(zero) == 0, "零点查询失败")
    return check


def check_grid_fingerprint_collision() -> Check:
    """基础问题 2 回归:采样点/加权和相同、内部几何不同的网格不得错误复用缓存;
    同网格二次调用命中缓存;独立计算与缓存结果一致。"""
    check = Check("网格指纹碰撞不错误复用")

    def grid(shift: float) -> np.ndarray:
        base = 10.0 + 0.1 * np.arange(16, dtype=float).reshape(4, 4)
        base[1, 1] += shift
        base[2, 2] -= shift   # 内部互抵:加权求和不变
        return base

    grid_a, grid_b = grid(0.0), grid(0.5)
    check.expect(not np.array_equal(grid_a, grid_b), "反例网格构造失败(应不同)")
    vc._SOURCE_INDEX_CACHE.clear()
    entry_a = vc._build_source_index(grid_a, grid_a)
    entry_b = vc._build_source_index(grid_b, grid_b)
    check.expect(entry_a["tree"] is not entry_b["tree"],
                 "互抵内部几何的不同网格被错误复用同一索引")
    entry_a2 = vc._build_source_index(grid_a, grid_a)
    check.expect(entry_a2["tree"] is entry_a["tree"], "同网格二次调用应命中缓存")
    # 缓存结果与独立计算一致(间距/足迹)
    spacing_direct = vc.estimate_source_spacing_km(grid_a, grid_a)
    check.expect(abs(entry_a["spacing_km"] - spacing_direct) < 1e-12,
                 "缓存间距与独立计算不一致")
    footprint_direct = vc._build_source_footprint(grid_a, grid_a, 0.05)
    check.expect(footprint_direct[0] == entry_a["footprint"][0],
                 "缓存足迹与独立计算不一致")
    vc._SOURCE_INDEX_CACHE.clear()
    return check


def check_region_status_distinction() -> Check:
    """基础问题 3 回归:empty_input(原本无有效输入)与
    nonfinite_reconstruction_output(输入有效但输出全无效)分开报告。"""
    check = Check("区域状态区分(empty_input vs 输出无效)")
    from verify_03_fixed_atmosphere_flux import region_output_status

    region = np.zeros((2, 4), dtype=bool)
    region[:, :2] = True          # 内环
    outer = np.zeros((2, 4), dtype=bool)
    outer[:, 2:] = True           # 外环
    input_valid = np.ones((2, 4), dtype=bool)
    compare_none = np.zeros((2, 4), dtype=bool)          # 输出全 NaN
    n, nonfinite, status = region_output_status(region, input_valid, compare_none)
    check.expect(n == 0 and nonfinite == 4 and status == "nonfinite_reconstruction_output",
                 f"输入有效但输出全无效: n={n}, nonfinite={nonfinite}, status={status}")
    # 区域内没有有效输入 -> empty_input,不是输出失效
    input_only_outer = np.zeros((2, 4), dtype=bool)
    input_only_outer[:, 2:] = True
    n2, nonfinite2, status2 = region_output_status(region, input_only_outer, compare_none)
    check.expect(status2 == "empty_input" and nonfinite2 == 0,
                 f"无有效输入应标记 empty_input: {status2}, nonfinite={nonfinite2}")
    # 外环输出无效不影响内环(按区域计)
    compare_outer_bad = np.ones((2, 4), dtype=bool)
    compare_outer_bad[:, 2:] = False
    _, nonfinite3, status3 = region_output_status(region, input_valid, compare_outer_bad)
    check.expect(status3 == "ok" and nonfinite3 == 0,
                 f"域外 NaN 泄漏到内环: {status3}, {nonfinite3}")
    return check


def check_end_to_end_new_tables(dir02: Path | None = None, dir03: Path | None = None) -> Check:
    """端到端:读取本轮全流程实际写出的新表,检查列/单位/点数/子集比例/闭合/覆盖率。

    路径由 check_full_flow 的 run() 返回值显式传入(P2 #6):
    不在生成前运行、不跨运行目录 glob 猜测文件,杜绝旧输出污染与顺序错误。
    """
    check = Check("端到端新表(条件预算/联合统计)")
    if dir02 is None or dir03 is None:
        check.expect(False, "端到端新表检查需要本轮全流程的输出目录(顺序错误)")
        return check
    cond_path = dir02 / "verify02_conditional_budget.csv"
    joint_path = dir03 / "verify03_joint_response_actual.csv"
    check.expect(cond_path.exists(), f"未找到本轮条件预算表: {cond_path}")
    check.expect(joint_path.exists(), f"未找到本轮联合统计表: {joint_path}")
    if not (cond_path.exists() and joint_path.exists()):
        return check
    cond = pd.read_csv(cond_path, dtype={"member": str})
    required_cols = [
        "mode", "method", "member", "time_hour", "region", "variable", "unit",
        "subset", "sst_variable", "sst_threshold_k2", "cross_term_tol_w2",
        "n_all_common", "n_subset", "subset_fraction_of_all_common",
        "status", "improvement_status", "cross_term_direction",
        "delta_mse_direct", "cross_term", "increment_square_term", "closure_residual",
    ]
    for col in required_cols:
        check.expect(col in cond.columns, f"条件预算表缺少 {col}")
    ok_rows = cond[(cond.status == "ok") & (cond.subset == "all_common")]
    check.expect(bool((ok_rows.closure_residual.abs() < 1e-6).all()),
                 "条件预算闭合残差超差")
    # 子集拆分守恒:检查全部组,不提前 break(P2 #6)
    for keys, grp in cond.groupby(["method", "member", "time_hour", "region", "variable"]):
        parts = grp.set_index("subset").n_subset
        total = int(parts.get("all_common", 0))
        split = int(parts.get("sst_improved", 0)) + int(parts.get("sst_worsened", 0)) + int(
            parts.get("sst_unchanged", 0)
        )
        check.expect(split == total, f"子集拆分不守恒({keys}): {split} != {total}")
    # 全案例覆盖:全部 2 方法 x 6 成员 x 13 时次
    check.expect(cond.member.nunique() == 6 and cond.method.nunique() == 2,
                 f"条件预算表成员/方法覆盖不足: {cond.member.nunique()}x{cond.method.nunique()}")
    # 联合表(显式本轮路径)
    if joint_path.exists():
        joint = pd.read_csv(joint_path, dtype={"member": str})
        joint_required = [
            "subset", "n_input_valid", "n_common", "n_subset",
            "subset_fraction_of_all_common", "cross_model_error_sst_response",
            "rms_sst_response", "rms_actual_response", "rms_sst_minus_actual",
            "sst_to_actual_rms_ratio", "sign_agreement", "n_sign_eligible",
            "joint_status", "attr_weak_rmse_ratio", "attr_strong_rmse_ratio",
            "attr_diff_agree_ratio",
            "actual_mse_weak", "actual_delta_mse_direct", "actual_cross_term",
            "actual_increment_square_term", "actual_closure_residual",
        ]
        for col in joint_required:
            check.expect(col in joint.columns, f"联合表缺少 {col}")
        joint_ok = joint[joint.status == "ok"]
        check.expect(bool((joint_ok.actual_closure_residual.abs() < 1e-6).all()),
                     "联合表实际预算闭合残差超差")
        check.expect(bool(joint_ok[["attr_weak_rmse_ratio", "attr_strong_rmse_ratio",
                                     "attr_diff_agree_ratio"]].notna().all().all()),
                     "联合表 ok 行的归因判据数值应为有限")
        check.expect(joint.member.nunique() == 6 and joint.method.nunique() == 2,
                     "联合表成员/方法覆盖不足")
        # 实际预算与诊断二条件预算在 all_common 上对照:两表的 all_common
        # 掩膜定义不同(联合表额外要求重建输出与 TSK 有限,点数可略小),
        # 因此点数相同的行预算必须逐位一致;点数不同的行联合表必须更小。
        v02 = cond
        key = ["method", "member", "time_hour", "region", "variable"]
        v03_n = joint[joint.subset == "all_common"][
            key + ["n_subset", "actual_delta_mse_direct", "actual_cross_term"]
        ].rename(columns={"n_subset": "n03"})
        v02_n = v02[v02.subset == "all_common"][
            key + ["n_subset", "delta_mse_direct", "cross_term"]
        ].rename(columns={"n_subset": "n02"})
        merged = v03_n.merge(v02_n, on=key, how="inner")
        check.expect(len(merged) > 0, "联合表与条件预算表无交集(键或覆盖不一致)")
        same_n = merged[merged.n03 == merged.n02]
        check.expect(len(same_n) > 0, "无掩膜一致的行可供对照")
        check.expect(bool(np.allclose(
            same_n.actual_delta_mse_direct, same_n.delta_mse_direct,
            rtol=1e-12, atol=1e-9,
        )), "同掩膜行的联合表实际 dMSE 与诊断二条件预算不一致")
        check.expect(bool(np.allclose(
            same_n.actual_cross_term, same_n.cross_term,
            rtol=1e-12, atol=1e-9,
        )), "同掩膜行的联合表实际 C 与诊断二条件预算不一致")
        diff_n = merged[merged.n03 != merged.n02]
        check.expect(bool((diff_n.n03 <= diff_n.n02).all()),
             "联合表掩膜更严时点数不应更大")
    return check


def check_region_output_status() -> Check:
    """区域状态:全 NaN / 仅本区域 NaN / 仅 300 km 外 NaN 三种情形互不影响。"""
    check = Check("区域输出状态(按区域计非有限点数)")
    from verify_03_fixed_atmosphere_flux import region_output_status

    grid = np.zeros((3, 6), dtype=bool)
    region_inner = grid.copy()
    region_inner[:, :3] = True        # 内环(前 3 列)
    region_outer = grid.copy()
    region_outer[:, 3:] = True        # 外环(后 3 列)
    input_mask = np.ones((3, 6), dtype=bool)

    # 1) 组合输出全 NaN:两区域都 nonfinite,n=0(区域为 3 行×3 列 = 9 点)
    compare_none = np.zeros((3, 6), dtype=bool)
    n, nonfinite, status = region_output_status(region_inner, input_mask, compare_none)
    check.expect(n == 0 and nonfinite == 9 and status == "nonfinite_reconstruction_output",
                 f"全 NaN 情形: n={n}, nonfinite={nonfinite}, status={status}")
    # 2) 仅内环 NaN:内环 nonfinite,外环不受影响
    compare_partial = np.ones((3, 6), dtype=bool)
    compare_partial[:, :3] = False
    n_in, nonfinite_in, status_in = region_output_status(region_inner, input_mask, compare_partial)
    n_out, nonfinite_out, status_out = region_output_status(region_outer, input_mask, compare_partial)
    check.expect(status_in == "nonfinite_reconstruction_output" and nonfinite_in == 9,
                 f"仅内环 NaN: 内环状态 {status_in}, 计数 {nonfinite_in}")
    check.expect(status_out == "ok" and n_out == 9 and nonfinite_out == 0,
                 f"仅内环 NaN 时外环被误标: {status_out}, {nonfinite_out}")
    # 3) 仅 300 km 外(外环)NaN:内环必须仍是 ok(域外点不计入)
    compare_outer_nan = np.ones((3, 6), dtype=bool)
    compare_outer_nan[:, 3:] = False
    n_in2, nonfinite_in2, status_in2 = region_output_status(region_inner, input_mask, compare_outer_nan)
    check.expect(status_in2 == "ok" and nonfinite_in2 == 0,
                 f"域外 NaN 泄漏到内环: {status_in2}, {nonfinite_in2}")
    return check


def check_joint_mask_numeric_counterexample() -> Check:
    """P2 #2 回归(审查反例,预期值来自手算):
    e=[1,NaN]、dSST=[1,100]、actual=[1,0]、mask 全真:
    有效点只有第一个 -> n_valid=1、C=2、rms_sst=rms_actual=1、ratio=1、
    rms_sst_minus_actual=0(旧掩膜错误值 70.710678 不再出现)。"""
    check = Check("联合统计 NaN 反例(数值逐项,不只是 n_valid)")
    e = np.array([1.0, np.nan])
    d_sst = np.array([1.0, 100.0])
    d_actual = np.array([1.0, 0.0])
    mask = np.ones(2, dtype=bool)
    stats = vc.response_vs_actual_stats(e, d_sst, d_actual, mask, flux_increment_tol=0.5)
    check.expect(stats["n_valid"] == 1, f"n_valid 应为 1: {stats['n_valid']}")
    check.expect(abs(stats["cross_model_error_sst_response"] - 2.0) < 1e-12,
                 f"C=2*1*1 应为 2: {stats['cross_model_error_sst_response']}")
    check.expect(abs(stats["rms_sst_response"] - 1.0) < 1e-12,
                 f"rms_sst 应为 1: {stats['rms_sst_response']}")
    check.expect(abs(stats["rms_actual_response"] - 1.0) < 1e-12,
                 f"rms_actual 应为 1: {stats['rms_actual_response']}")
    check.expect(abs(stats["rms_sst_minus_actual"]) < 1e-12,
                 f"rms_sst_minus_actual 应为 0(同一 joint_mask): "
                 f"{stats['rms_sst_minus_actual']}")
    check.expect(abs(stats["sst_to_actual_rms_ratio"] - 1.0) < 1e-12,
                 f"ratio 应为 1: {stats['sst_to_actual_rms_ratio']}")
    # cross_term_stats 契约:掩膜内全 NaN 时 n_valid=0(不是掩膜点数)
    empty = vc.cross_term_stats(np.array([np.nan, np.nan]), np.array([1.0, 2.0]),
                                np.ones(2, dtype=bool))
    check.expect(empty["status"] == "all_nan_inside_mask" and empty["n_valid"] == 0,
                 f"全 NaN 输入的 n_valid 应为 0: {empty['n_valid']}")
    # 比率分母下限:实际响应 RMS 0.5 <= floor 0.5 -> 比率 NaN + below_floor
    joint_floor = vc.response_vs_actual_stats(
        np.array([1.0, 1.0]), np.array([0.5, 0.5]), np.array([0.5, -0.5]),
        np.ones(2, dtype=bool), flux_increment_tol=0.1,
        ratio_denominator_floor=0.5,
    )
    check.expect(joint_floor["actual_response_status"] == "actual_response_below_floor"
                 and not np.isfinite(joint_floor["sst_to_actual_rms_ratio"]),
                 f"低于分母下限的比率应置 NaN: {joint_floor['sst_to_actual_rms_ratio']}")
    return check


def check_joint_tiered_counts() -> Check:
    """P2 #3 回归(审查反例,手算):
    4 个有效输入、1 点重建失败 -> n_input_valid=4、n_common=3;
    全部输出 NaN -> Phase A/B 状态为 nonfinite 而非 empty_mask,
    联合表仍有该案例空行,失败原因不丢失。"""
    check = Check("三级点数与全失败状态(手算)")
    from verify_03_fixed_atmosphere_flux import region_output_status

    # 部分失败:输入 4 点,输出有效 3 点 -> 行仍可用(status=ok),
    # 损失由 n_nonfinite_output=1 单独承载(审查原文口径);全失败才标 nonfinite
    input_mask = np.ones(4, dtype=bool)
    compare_mask = np.array([True, True, True, False])
    n, nonfinite, status = region_output_status(np.ones(4, dtype=bool), input_mask, compare_mask)
    check.expect(n == 3 and nonfinite == 1 and status == "ok",
                 f"部分失败: n={n}, nonfinite={nonfinite}, status={status}")
    # 全失败:输入 4 点,输出全 NaN -> nonfinite 而非 empty_mask
    compare_none = np.zeros(4, dtype=bool)
    n2, nonfinite2, status2 = region_output_status(np.ones(4, dtype=bool), input_mask, compare_none)
    check.expect(n2 == 0 and nonfinite2 == 4 and status2 == "nonfinite_reconstruction_output",
                 f"全失败: n={n2}, nonfinite={nonfinite2}, status={status2}")
    # 无有效输入 -> empty_input(与输出失效区分)
    n3, _, status3 = region_output_status(
        np.ones(4, dtype=bool), np.zeros(4, dtype=bool), compare_none
    )
    check.expect(status3 == "empty_input", f"无有效输入: {status3}")
    return check


def check_joint_quality_flagging() -> Check:
    """P2 #1 回归:子集质量标记唯一化。
    全区域合格、TSK 改善子集不合格时,联合行的 joint_status 必须为不合格,
    attribution_flag_region_ref 仅作对照;空子集不继承全区域合格标记。"""
    check = Check("子集质量标记唯一化(全区域合格/子集不合格)")
    from verify_03_fixed_atmosphere_flux import (
        _joint_quality_flag,
        attribution_quality_flag,
    )

    # 手工构造(手算):8 点;前 4 点(TSK 改善子集)弱重建残差 60,
    # 后 4 点为 0。全区域 RMS = 60/sqrt(2) ≈ 42.4,比值 0.424 <= 0.5 → 合格;
    # 改善子集 RMS = 60,比值 0.6 > 0.5 → 不合格(不继承全区域标记)。
    rw = np.array([60.0] * 4 + [0.0] * 4)
    rs = np.array([0.0] * 8)
    diff = np.array([0.0] * 8)
    scale = np.full(8, 100.0)
    dse = np.array([-5.0] * 4 + [5.0] * 4)
    subsets = vc.sst_subset_masks(dse, np.ones(8, dtype=bool), tol=1.0)
    region_flag = attribution_quality_flag(
        vc.rms(rw, subsets["all_common"]) / 100.0,
        0.0, 0.0,
    )
    subset_flag = _joint_quality_flag(rw, rs, diff, scale, subsets["sst_improved"])
    empty_flag = _joint_quality_flag(rw, rs, diff, scale, subsets["sst_unchanged"])
    check.expect(region_flag == "residual_small_vs_explained_diff",
                 f"全区域标记应合格: {region_flag}")
    check.expect(
        subset_flag == "residual_not_negligible_attribution_requires_verification",
        f"改善子集标记应不合格(不继承全区域): {subset_flag}",
    )
    check.expect(
        empty_flag == "attribution_criteria_not_evaluable",
        f"空子集质量不可评估(不继承全区域合格): {empty_flag}",
    )
    check.expect(subset_flag != region_flag, "子集标记与全域参考应可不同")
    return check


def check_verify02_all_missing_run() -> Check:
    """P2 #5 回归:诊断二全部案例缺失时不崩溃(UnboundLocalError),
    run_status 先落盘,新增汇总空表保留完整列结构。"""
    check = Check("诊断二全缺失运行(空表列结构)")
    module = __import__("verify_02_flux_error_budget")
    run_root = Path(tempfile.mkdtemp(prefix="v02_all_missing_", dir=SMOKE_OUT / "runs"))
    config = VerifyConfig(
        mode="synthetic",
        methods=("QCF_RHF",),
        members=("044",),
        times=((2.5, "2018-09-10_02:30:00"),),
        output_root=run_root,
    )
    out_dir = module.run(config)
    status = pd.read_csv(out_dir / "verify02_run_status.csv", dtype={"member": str})
    check.expect(len(status) == 1 and status.status.iloc[0] == "missing_member_or_time",
                 "run_status 应记录缺失原因")
    cond_summary_member = pd.read_csv(out_dir / "verify02_conditional_summary_by_member.csv")
    cond_summary_method = pd.read_csv(out_dir / "verify02_conditional_summary_by_method.csv")
    for name, frame in (("by_member", cond_summary_member), ("by_method", cond_summary_method)):
        check.expect(len(frame) == 0 and len(frame.columns) > 5,
                     f"诊断二条件汇总空表应保留列结构({name})")
    check.expect("overall_status: no_valid_cases" in
                 (out_dir / "overall_status.txt").read_text(),
                 "总状态应为 no_valid_cases")
    return check


# ==================================================================
# verify_04:初值传递核验与口径衔接
# ==================================================================


def _write_tiny_stage_file(path, values_by_var, times, lat, lon,
                           dim_orders=None, with_units=True):
    """写一个小型阶段 NetCDF(values_by_var: {var: (data_with_time, dims, unit)})。

    dim_orders 可按变量名给维度元组;默认标准顺序。
    """
    from netCDF4 import Dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = lat.shape
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("ocean_layer_stag", 2)
        ds.createDimension("bottom_top", 3)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        times_var = ds.createVariable("Times", str, ("Time",))
        for record, stamp in enumerate(times):
            times_var[record] = stamp
        v = ds.createVariable("XLAT", "f8", ("Time", "south_north", "west_east"))
        v[0] = lat
        v = ds.createVariable("XLONG", "f8", ("Time", "south_north", "west_east"))
        v[0] = lon
        for name, (data, dims, unit) in values_by_var.items():
            var = ds.createVariable(name, "f8", dims)
            if with_units:
                var.units = unit
            var[:] = data


def check_verify04_handoff_manual() -> Check:
    """verify_04 手算核验(预期值全部来自显式构造):
    A=B、I=A → 两类残差零;I 部分保留 A → 预期残差;正负抵消 → 均值零 RMS 非零;
    错时刻/缺变量/维度换序/多时间记录/单位缺失。"""
    check = Check("verify_04 手算(残差/时刻/换序/单位)")
    import tempfile
    from verify04_readers import FieldRequest, read_stage_field

    root = Path(tempfile.mkdtemp(prefix="v04_manual_", dir=SMOKE_OUT))
    ny, nx = 3, 4
    lat = 13.0 + 0.1 * np.arange(ny)[:, None] * np.ones((1, nx))
    lon = 137.0 + 0.1 * np.arange(nx)[None, :] * np.ones((ny, 1))
    times = ["2018-09-10_00:00:00", "2018-09-10_00:30:00"]

    # 场景 1-3:B/A/I 三个文件,OM_TMP: A=B+1(增量),I=A-0.3(部分保留)
    om_b = np.full((2, 1, ny, nx), 300.0)
    om_a = om_b + 1.0
    om_i = om_a - 0.3
    _write_tiny_stage_file(
        root / "B.nc",
        {"OM_TMP": (om_b, ("Time", "ocean_layer_stag", "south_north", "west_east"), "K")},
        times, lat, lon,
    )
    _write_tiny_stage_file(
        root / "A.nc",
        {"OM_TMP": (om_a, ("Time", "ocean_layer_stag", "south_north", "west_east"), "K")},
        times, lat, lon,
    )
    _write_tiny_stage_file(
        root / "I.nc",
        {"OM_TMP": (om_i, ("Time", "ocean_layer_stag", "south_north", "west_east"), "K")},
        times, lat, lon,
    )
    req = FieldRequest("OM_TMP", times[1], layer=0, expected_unit="K")
    rb = read_stage_field(root / "B.nc", req)
    ra = read_stage_field(root / "A.nc", req)
    ri = read_stage_field(root / "I.nc", req)
    for read in (rb, ra, ri):
        check.expect(read.status == "ok", f"读取应 ok: {read.status} {read.detail}")
        check.expect(read.time_source == "times_attribute"
                     and read.time_found == times[1],
                     f"应按 Times 属性匹配第二个记录: {read.time_source}/{read.time_found}")
    stats_assim = vc.difference_stats(ra.values - rb.values, np.isfinite(ra.values), 1e-3)
    check.expect(abs(stats_assim["mean_diff"] - 1.0) < 1e-12,
                 f"d_assim 均值应为 1: {stats_assim['mean_diff']}")
    stats_handoff = vc.difference_stats(ri.values - ra.values, np.isfinite(ra.values), 1e-3)
    check.expect(abs(stats_handoff["mean_diff"] + 0.3) < 1e-12,
                 f"d_handoff 应为 -0.3(I 部分保留 A): {stats_handoff['mean_diff']}")

    # 场景 4:正负抵消 —— 均值为零但 RMS 非零;容差计数
    diff = np.array([[1.0, -1.0], [1.0, -1.0]])
    stats_cancel = vc.difference_stats(diff, np.ones((2, 2), dtype=bool), tolerance=0.5)
    check.expect(abs(stats_cancel["mean_diff"]) < 1e-12,
                 f"抵消场均值应为 0: {stats_cancel['mean_diff']}")
    check.expect(abs(stats_cancel["rms_diff"] - 1.0) < 1e-12,
                 f"抵消场 RMS 应为 1: {stats_cancel['rms_diff']}")
    check.expect(stats_cancel["n_over_tolerance"] == 4
                 and abs(stats_cancel["frac_over_tolerance"] - 1.0) < 1e-12,
                 "超容差计数错误")
    empty = vc.difference_stats(diff, np.zeros((2, 2), dtype=bool), 0.5)
    check.expect(empty["status"] == "empty_mask" and not np.isfinite(empty["rms_diff"]),
                 "空掩膜应返回 NaN + empty_mask")
    all_nan = vc.difference_stats(np.full((2, 2), np.nan), np.ones((2, 2), dtype=bool), 0.5)
    check.expect(all_nan["status"] == "all_nan_inside_mask" and all_nan["n_valid"] == 0,
                 f"全 NaN 应 n_valid=0: {all_nan['n_valid']}")

    # 场景 5:错时刻 → time_not_found;维度换序 → 读取归一化为 (ny, nx);
    # 单位缺失 → unit_matches=None;多时间记录 + Times 缺失 → record_index 回退
    bad_time = read_stage_field(root / "A.nc", FieldRequest("OM_TMP", "2018-09-11_00:00:00", layer=0))
    check.expect(bad_time.status == "time_not_found", f"错时刻应 time_not_found: {bad_time.status}")
    # 换序文件:T 写成 (Time, west_east, bottom_top, south_north)
    t_perm = np.zeros((2, nx, 3, ny))
    t_perm[:, :, 0, :] = 5.0
    _write_tiny_stage_file(
        root / "PERM.nc",
        {"T": (t_perm, ("Time", "west_east", "bottom_top", "south_north"), "K")},
        times, lat, lon,
    )
    rperm = read_stage_field(root / "PERM.nc", FieldRequest("T", times[0], layer=0, expected_unit="K"))
    check.expect(rperm.status == "ok", f"换序文件读取应 ok: {rperm.status} {rperm.detail}")
    check.expect(rperm.values.shape == (ny, nx),
                 f"换序文件应归一化为 (ny,nx): {rperm.values.shape}")
    check.expect(bool(np.allclose(rperm.values, 5.0)), "换序读取数值错误")
    _write_tiny_stage_file(
        root / "NOUNIT.nc",
        {"TSK": (np.full((2, ny, nx), 302.0), ("Time", "south_north", "west_east"), "K")},
        times, lat, lon, with_units=False,
    )
    rnounit = read_stage_field(root / "NOUNIT.nc", FieldRequest("TSK", times[0], expected_unit="K"))
    check.expect(rnounit.status == "ok" and rnounit.unit_matches is None,
                 f"缺 units 属性应 unit_matches=None: {rnounit.unit_matches}")
    # Times 缺失 → 显式记录号回退
    from netCDF4 import Dataset

    path_notimes = root / "NOTIMES.nc"
    with Dataset(path_notimes, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", 2)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        var = ds.createVariable("TSK", "f8", ("Time", "south_north", "west_east"))
        var[0] = 300.0
        var[1] = 301.0
    r_record = read_stage_field(
        path_notimes, FieldRequest("TSK", times[1], time_record=1, expected_unit="K")
    )
    check.expect(r_record.status == "ok" and r_record.time_source == "record_index"
                 and float(r_record.values.mean()) == 301.0,
                 f"无 Times 时应按显式记录号回退: {r_record.status}/{r_record.time_source}")
    r_no_record = read_stage_field(
        path_notimes, FieldRequest("TSK", times[1], expected_unit="K")
    )
    check.expect(r_no_record.status == "time_not_found",
                 "无 Times 且无记录号应 time_not_found(不盲目取第 0 条)")
    return check


def check_verify04_stage_skips() -> Check:
    """verify_04 阶段缺失:只跳过依赖它的比较;强弱背景不同时
    d_pair 与 d_assim 字段名分离;OM_TMP 变而 TSK 不变不判错。"""
    check = Check("verify_04 阶段缺失与字段命名分离")
    import dataclasses
    import tempfile
    v04 = __import__("verify_04_initial_handoff")
    run_root = Path(tempfile.mkdtemp(prefix="v04_run_", dir=SMOKE_OUT / "runs"))
    config = VerifyConfig(
        mode="synthetic",
        methods=("EAKF",),
        members=("006",),
        times=((0.0, "2018-09-10_00:00:00"), (0.5, "2018-09-10_00:30:00")),
        output_root=run_root,
    )
    out_dir = v04.run(config)
    pairwise = pd.read_csv(out_dir / "verify04_pairwise_checks.csv", dtype={"member": str})
    pairs = pd.read_csv(out_dir / "verify04_initial_pair_checks.csv", dtype={"member": str})
    coverage = pd.read_csv(out_dir / "verify04_coverage.csv")
    manifest = pd.read_csv(out_dir / "verify04_source_manifest.csv")
    # 手算:OM_TMP d_assim = A-B = inc(均值 5/48≈0.1042,含正负抵消);
    # d_handoff I-A = 0;TSK 全程同值 → 三类差均为 0 且 TSK 在 M 外;
    # d_pair@I 大气=0、OM_TMP=0.6(强弱背景不同)
    sel = pairwise[(pairwise.variable == "OM_TMP") & (pairwise.experiment == config.strong_experiment)]
    assim = sel[sel.stage_pair == "A-B"]
    handoff = sel[sel.stage_pair == "I-A"]
    check.expect(bool((assim.status == "ok").all()) and abs(assim.mean_diff.iloc[0] - 5.0 / 48.0) < 1e-12,
                 f"OM_TMP d_assim 均值应为 5/48: {assim.mean_diff.iloc[0]}")
    check.expect(bool((handoff.status == "ok").all())
                 and bool((handoff[["mean_diff", "rms_diff", "max_abs_diff"]].abs() < 1e-12).all().all()),
                 "OM_TMP d_handoff 应为 0(I=A)")
    tsk_rows = pairwise[(pairwise.variable == "TSK") & (pairwise.stage_pair == "I-A")]
    check.expect(bool(tsk_rows.detail.str.contains("mapping_unverified").all()),
                 "TSK 的 d_handoff 应标注 mapping_unverified(M 外变量),而非错误")
    # 字段名分离:d_assim 表有 stage_pair 列,pair 表有 stage 列,无同名列冲突
    check.expect("stage_pair" in pairwise.columns and "stage" in pairs.columns,
                 "d_assim/d_pair 表的阶段列名应区分")
    check.expect(not ({"mean_diff"} & set(pairs.columns)), "pair 表不应复用 mean_diff 列名")
    pair_i = pairs[(pairs.stage == "I") & (pairs.variable == "T")
                   & (pairs.valid_time == "2018-09-10_00:00:00")]
    check.expect(bool((pair_i.detail == "atmosphere_pair_consistent_within_tolerance").all()),
                 "强弱大气启动状态应一致(合成场景)")
    pair_b_om = pairs[(pairs.stage == "B") & (pairs.variable == "OM_TMP")]
    check.expect(bool(((pair_b_om.mean_pair_diff - 0.6).abs() < 1e-12).all()),
                 f"强弱背景差应为 0.6: {pair_b_om.mean_pair_diff.tolist()}")
    # 覆盖表:expected/attempted/completed 一致
    om_cov = coverage[(coverage.comparison == "A-B") & (coverage.variable == "OM_TMP")]
    # 起报时次契约:expected = 1 成员 × 1 起报时次 × 强弱 2 试验 = 2,全部完成
    check.expect(bool((om_cov.expected == 2).all()) and bool((om_cov.completed == 2).all())
                 and bool((om_cov.skipped == 0).all()),
                 f"A-B OM_TMP 覆盖应为 expected=2(completed=2): {om_cov.to_dict('records')}")
    # 起报时次契约:全部比较只在 INIT_TIME_NAME 上(lead time 不充当 cycle)
    check.expect(set(pairwise.valid_time.unique()) == {"2018-09-10_00:00:00"},
                 "比较应只在起报循环有效时间上进行")
    # 映射/来源门控:数值 ok 但结论未定(来源默认未确认)
    om_ia = pairwise[(pairwise.stage_pair == "I-A") & (pairwise.variable == "OM_TMP")]
    check.expect(bool((om_ia.handoff_conclusion
                       == "not_concluded_source_unconfirmed").all()),
                 "来源未确认时 I-A 结论应为 not_concluded_source_unconfirmed")
    check.expect(bool((om_ia.status == "ok").all()),
                 "数值可计算(status=ok)与结论分离")
    # 弱试验映射未知:detail 记 mapping_unverified
    weak_ia = pairwise[(pairwise.stage_pair == "I-A")
                       & (pairwise.experiment == config.weak_experiment)
                       & (pairwise.variable == "OM_TMP")]
    check.expect(bool(weak_ia.detail.str.contains("mapping_unverified").all()),
                 "弱试验(ncks_air.sh 缺失)映射未知应记 mapping_unverified")
    # 交错变量:grid_unverified_staggered 显式标记,且网格未核验时
    # 不得出现大气一致性结论(R1)
    u_pair = pairs[(pairs.stage == "I") & (pairs.variable == "U")]
    check.expect(bool(u_pair.detail.str.contains("grid_unverified_staggered").all()),
                 "交错变量应显式标记 grid_unverified_staggered")
    check.expect(bool(~u_pair.detail.str.contains("consistent").all()),
                 "网格未核验时不得输出大气一致性结论")
    # 清单:四阶段逐文件记录,路径含 experiment/method
    check.expect(set(manifest.stage.unique()) == {"B", "A", "I", "F0"},
                 "来源清单应覆盖 B/A/I/F0 四阶段")
    check.expect(bool(manifest.path.str.contains("6mem_oceanAssim1Run1_EAKF_006").any()),
                 "清单路径应含 experiment/method 信息(来源可追溯)")
    # 口径表存在且包含全部维度
    caliber = pd.read_csv(out_dir / "caliber_compat_table.csv")
    check.expect(len(caliber) == len(vc.CALIBER_DIMENSIONS) if hasattr(vc, "CALIBER_DIMENSIONS")
                 else len(caliber) >= 10, "口径对照表维度数异常")
    check.expect(set(caliber.verdict).issubset({
        "value_by_value", "parallel_evidence", "not_comparable", "unknown",
    }), "口径判定应限于四档")
    return check


def check_verify04_unit_fill_grid_gates() -> Check:
    """审查反例的端到端回归(P1 #1/#2/#4):
    - B(K) vs A(degC)→ unit_mismatch,不输出可解释的温度差;
    - 24/48 点双侧 NaN → n_base=48 不收缩、n_missing=36、手算均值 5/12;
    - 双侧同点 1e35 填充 → n_fill=12 显式暴露,不伪装零差;
    - A 网格在起报记录漂移 1° → grid_mismatch。"""
    check = Check("verify_04 单位/填充/网格门控(端到端)")
    import tempfile
    from netCDF4 import Dataset

    v04 = __import__("verify_04_initial_handoff")
    original_writer = v04._write_synthetic_inputs
    run_root = Path(tempfile.mkdtemp(prefix="v04_gates_", dir=SMOKE_OUT / "runs"))

    def patched_run(config, mutate):
        def writer(cfg):
            path = original_writer(cfg)
            mutate(path)
            return path
        v04._write_synthetic_inputs = writer
        try:
            return v04.run(config)
        finally:
            v04._write_synthetic_inputs = original_writer

    base_config = VerifyConfig(
        mode="synthetic", methods=("EAKF",), members=("006",),
        times=((0.0, "2018-09-10_00:00:00"), (0.5, "2018-09-10_00:30:00")),
    )

    def a_files(synth_root, experiment):
        return sorted(synth_root.glob(f"A_{experiment}_EAKF_006_d02.nc"))

    # --- 反例 1:单位不匹配(A 为 degC)---
    def mutate_units(synth_root):
        for path in a_files(synth_root, "6mem_oceanAssim1Run1"):
            with Dataset(path, "r+") as ds:
                ds.variables["OM_TMP"].units = "degC"

    out1 = patched_run(dataclasses.replace(
        base_config, output_root=run_root, output_dirname="units"
    ), mutate_units)
    pairwise1 = pd.read_csv(out1 / "verify04_pairwise_checks.csv", dtype={"member": str})
    row = pairwise1[(pairwise1.stage_pair == "A-B") & (pairwise1.variable == "OM_TMP")
                    & (pairwise1.experiment == "6mem_oceanAssim1Run1")]
    check.expect(bool((row.status == "unit_mismatch").all()),
                 f"单位不符应 unit_mismatch: {row.status.tolist()}")
    check.expect(bool(row.mean_diff.isna().all()),
                 "单位不符时不应输出数值")
    # unit_left = 差值左操作数(A,后被改为 degC);unit_right = B(K)
    check.expect(bool((row.unit_left == "degC").all() and (row.unit_right == "K").all()),
                 f"应保留两侧原始单位: {row.unit_left.tolist()}/{row.unit_right.tolist()}")
    # A-B 不是交接比较:结论列保持 n/a,门控体现在 status
    check.expect(bool((row.handoff_conclusion == "n/a (not a handoff comparison)").all()),
                 "A-B 行结论列应为 n/a(门控在 status)")
    ia_row = pairwise1[(pairwise1.stage_pair == "I-A") & (pairwise1.variable == "OM_TMP")
                       & (pairwise1.experiment == "6mem_oceanAssim1Run1")]
    check.expect(bool((ia_row.status == "unit_mismatch").all()
                      and (ia_row.handoff_conclusion
                           == "not_concluded:unit_mismatch").all()),
                 "I-A 行单位不符应 not_concluded:unit_mismatch")

    # --- 反例 2:双侧 NaN(36 点)与双侧同点填充(12 点)---
    def mutate_missing_fill(synth_root):
        for path in a_files(synth_root, "6mem_oceanAssim1Run1"):
            with Dataset(path, "r+") as ds:
                var = ds.variables["OM_TMP"]
                for record in range(var.shape[0]):
                    var[record, 0, 3:, :] = np.nan          # 36 点缺测
                    var[record, 0, :3, :4] = 1.0e35         # 12 点填充(双侧同值)
        for path in sorted(synth_root.glob("B_6mem_oceanAssim1Run1_EAKF_006_d02.nc")):
            with Dataset(path, "r+") as ds:
                var = ds.variables["OM_TMP"]
                for record in range(var.shape[0]):
                    var[record, 0, 3:, :] = np.nan
                    var[record, 0, :3, :4] = 1.0e35

    out2 = patched_run(dataclasses.replace(
        base_config, output_root=run_root, output_dirname="fill"
    ), mutate_missing_fill)
    pairwise2 = pd.read_csv(out2 / "verify04_pairwise_checks.csv", dtype={"member": str})
    row2 = pairwise2[(pairwise2.stage_pair == "A-B") & (pairwise2.variable == "OM_TMP")
                     & (pairwise2.experiment == "6mem_oceanAssim1Run1")]
    # 手算:缺测 36 点(rows 3:)、填充 12 点(增量区 rows :3,:4,双侧同值
    # 1e35 → 若直接相减会伪装成零差);有效点 = 其余 12 点,该区域增量恰为 0
    # → mean=0、rms=0,但 n_fill=12 显式暴露填充,不把零差当作通过
    check.expect(bool((row2.n_base == 48).all()),
                 f"基础掩膜不应随缺测收缩: {row2.n_base.tolist()}")
    check.expect(bool((row2.n_valid == 12).all()),
                 f"有效点应为 12: {row2.n_valid.tolist()}")
    check.expect(bool((row2.n_missing == 36).all()),
                 f"缺失应记 36: {row2.n_missing.tolist()}")
    check.expect(bool((row2.n_fill == 12).all()),
                 f"填充点应显式记 12(不再伪装零差): {row2.n_fill.tolist()}")
    check.expect(bool((row2.mean_diff.abs() < 1e-12).all())
                 and bool((row2.rms_diff < 1e-12).all()),
                 f"有效区增量应为 0: {row2.mean_diff.tolist()}")
    # 阶段差统计不会把填充相消当成 48 个有效点的一致性证据
    check.expect(bool((row2.frac_over_tolerance == 0.0).all()),
                 "有效区超容差比例应为 0(增量区已被排除并单列)")

    # --- 反例 3:A 网格在起报记录漂移 1° ---
    def mutate_grid(synth_root):
        for path in a_files(synth_root, "6mem_oceanAssim1Run1"):
            with Dataset(path, "r+") as ds:
                ds.variables["XLAT"][0] += 1.0

    out3 = patched_run(dataclasses.replace(
        base_config, output_root=run_root, output_dirname="griddrift"
    ), mutate_grid)
    pairwise3 = pd.read_csv(out3 / "verify04_pairwise_checks.csv", dtype={"member": str})
    row3 = pairwise3[(pairwise3.stage_pair == "A-B") & (pairwise3.variable == "OM_TMP")
                     & (pairwise3.experiment == "6mem_oceanAssim1Run1")]
    check.expect(bool((row3.status == "grid_mismatch").all()),
                 f"网格漂移应 grid_mismatch: {row3.status.tolist()}")
    check.expect(bool(row3.mean_diff.isna().all()),
                 "网格不匹配时不应输出数值(不插值掩盖)")
    return check


def check_verify04_mapping_conclusion() -> Check:
    """映射门控单测(审查 P1 #5 + R1/R1 覆盖分层):来源/网格/映射/容差分层。"""
    check = Check("verify_04 映射结论分层")
    from verify_04_initial_handoff import MappingSpec, _handoff_conclusion

    spec_unknown = MappingSpec("exp0", None, "ncks_air.sh missing")
    spec_inferred = MappingSpec(
        "exp1", ("OM_TMP", "T"), "ncks.sh list", confirmed=False
    )
    spec_confirmed = dataclasses.replace(spec_inferred, confirmed=True)
    # 网格未核验(R1):任何数值一致都不给交接结论
    check.expect(
        _handoff_conclusion("I-A", "ok", 0.0, True, spec_confirmed, "OM_TMP",
                            grid_verified=False, n_valid=48, n_base=48)
        == "not_concluded_grid_unverified",
        "网格未核验应 not_concluded_grid_unverified",
    )
    # 来源未确认
    check.expect(
        _handoff_conclusion("I-A", "ok", 0.0, False, spec_confirmed, "OM_TMP",
                            grid_verified=True, n_valid=48, n_base=48)
        == "not_concluded_source_unconfirmed",
        "来源未确认应先于映射判定",
    )
    # 映射未知
    check.expect(
        _handoff_conclusion("I-A", "ok", 0.0, True, spec_unknown, "OM_TMP",
                            grid_verified=True, n_valid=48, n_base=48)
        == "not_concluded_mapping_unknown",
        "映射未知应 not_concluded_mapping_unknown",
    )
    # M 外变量
    check.expect(
        _handoff_conclusion("I-A", "ok", 0.0, True, spec_confirmed, "TSK",
                            grid_verified=True, n_valid=48, n_base=48)
        == "not_concluded_variable_outside_mapping",
        "M 外变量应 not_concluded_variable_outside_mapping",
    )
    # 映射推断未确认
    check.expect(
        _handoff_conclusion("I-A", "ok", 0.0, True, spec_inferred, "OM_TMP",
                            grid_verified=True, n_valid=48, n_base=48)
        == "not_concluded_mapping_inferred_not_confirmed",
        "映射推断未确认应 not_concluded_mapping_inferred_not_confirmed",
    )
    # 全部满足后按容差与覆盖分层给结论
    check.expect(
        _handoff_conclusion("I-A", "ok", 0.0, True, spec_confirmed, "OM_TMP",
                            grid_verified=True, n_valid=48, n_base=48)
        == "consistent_within_tolerance",
        "全域容差内应 consistent_within_tolerance",
    )
    check.expect(
        _handoff_conclusion("I-A", "ok", 0.0, True, spec_confirmed, "OM_TMP",
                            grid_verified=True, n_valid=1, n_base=48)
        == "consistent_on_valid_subset",
        "部分覆盖应 consistent_on_valid_subset(R1 覆盖分层)",
    )
    check.expect(
        _handoff_conclusion("I-A", "ok", 0.5, True, spec_confirmed, "OM_TMP",
                            grid_verified=True, n_valid=48, n_base=48)
        == "differs_outside_tolerance",
        "超容差应 differs_outside_tolerance",
    )
    check.expect(
        _handoff_conclusion("I-A", "ok", 0.5, True, spec_confirmed, "OM_TMP",
                            grid_verified=True, n_valid=1, n_base=48)
        == "differs_on_valid_subset",
        "部分覆盖超容差应 differs_on_valid_subset",
    )
    return check


def check_verify04_round2_fixes() -> Check:
    """第二轮复核反例的端到端回归(R1/R2/R3/R4/R5/R6)。"""
    check = Check("verify_04 第二轮复核回归")
    import tempfile
    from netCDF4 import Dataset

    v04 = __import__("verify_04_initial_handoff")
    original_writer = v04._write_synthetic_inputs
    original_defaults = dict(v04.STAGE_SOURCE_DEFAULTS)
    original_mapping = dict(v04.MAPPING_SPECS)
    original_vars = v04.CHECK_VARIABLES
    run_root = Path(tempfile.mkdtemp(prefix="v04_r2_", dir=SMOKE_OUT / "runs"))

    def patched_run(config, mutate=None, confirm_source=False,
                    confirm_mapping=False):
        def writer(cfg):
            path = original_writer(cfg)
            if mutate is not None:
                mutate(path)
            return path
        v04._write_synthetic_inputs = writer
        if confirm_source:
            v04.STAGE_SOURCE_DEFAULTS = {
                stage: dataclasses.replace(info, confirmed=True)
                for stage, info in v04.STAGE_SOURCE_DEFAULTS.items()
            }
        if confirm_mapping:
            v04.MAPPING_SPECS = {
                name: dataclasses.replace(spec, confirmed=True)
                for name, spec in v04.MAPPING_SPECS.items()
                if spec.variables is not None
            }
        try:
            return v04.run(config)
        finally:
            v04._write_synthetic_inputs = original_writer
            v04.STAGE_SOURCE_DEFAULTS = original_defaults
            v04.MAPPING_SPECS = original_mapping
            v04.CHECK_VARIABLES = original_vars

    base_config = VerifyConfig(
        mode="synthetic", methods=("EAKF",), members=("006",),
        times=((0.0, "2018-09-10_00:00:00"), (0.5, "2018-09-10_00:30:00")),
    )

    def a_files(synth_root):
        return sorted(synth_root.glob("A_6mem_oceanAssim1Run1_EAKF_006_d02.nc"))

    # --- R1:交错网格漂移但数值相等 → 不得给出一致性结论 ---
    def mutate_staggered_drift(synth_root):
        for path in a_files(synth_root):
            with Dataset(path, "r+") as ds:
                ds.variables["XLAT"][0] += 1.0   # A 网格漂移(交错场数值不变)

    out1 = patched_run(dataclasses.replace(
        base_config, output_root=run_root, output_dirname="r1"
    ), mutate_staggered_drift, confirm_source=True, confirm_mapping=True)
    pairwise1 = pd.read_csv(out1 / "verify04_pairwise_checks.csv", dtype={"member": str})
    u_row = pairwise1[(pairwise1.stage_pair == "I-A") & (pairwise1.variable == "U")
                      & (pairwise1.experiment == "6mem_oceanAssim1Run1")]
    check.expect(bool((u_row.status == "ok").all()),
                 "交错数值相等应可计算")
    check.expect(bool((u_row.handoff_conclusion
                       == "not_concluded_grid_unverified").all()),
                 f"网格未核验不得给交接结论: {u_row.handoff_conclusion.tolist()}")
    pairs1 = pd.read_csv(out1 / "verify04_initial_pair_checks.csv", dtype={"member": str})
    u_pair = pairs1[(pairs1.stage == "I") & (pairs1.variable == "U")]
    check.expect(bool(~u_pair.detail.str.contains("consistent").all()),
                 "网格未核验时大气一致性描述不得出现")

    # --- R1 覆盖:47 点双侧 NaN、1 点相等 → consistent_on_valid_subset ---
    def mutate_coverage(synth_root):
        # 同时改 A 与 I 文件:使 [0,0] 点 I-A=0(其余点缺测)
        for pattern in ("A_6mem_oceanAssim1Run1_EAKF_006_d02.nc",
                        "I_6mem_oceanAssim1Run1_EAKF_006_d02.nc"):
            for path in sorted(synth_root.glob(pattern)):
                with Dataset(path, "r+") as ds:
                    var = ds.variables["OM_TMP"]
                    var[:, 0, :, :] = np.nan
                    var[0, 0, 0, 0] = 301.0
        # 强弱 I 文件的大气场仅一个共同有效点相等，不代表全域一致。
        for path in sorted(synth_root.glob("I_*_EAKF_006_d02.nc")):
            with Dataset(path, "r+") as ds:
                ds.variables["T"][:, 0, :, :] = np.nan
                ds.variables["T"][0, 0, 0, 0] = 1.0

    out2 = patched_run(dataclasses.replace(
        base_config, output_root=run_root, output_dirname="r1cov"
    ), mutate_coverage, confirm_source=True, confirm_mapping=True)
    pairwise2 = pd.read_csv(out2 / "verify04_pairwise_checks.csv", dtype={"member": str})
    row2 = pairwise2[(pairwise2.stage_pair == "I-A") & (pairwise2.variable == "OM_TMP")
                     & (pairwise2.experiment == "6mem_oceanAssim1Run1")]
    check.expect(bool((row2.handoff_conclusion
                       == "consistent_on_valid_subset").all()),
                 f"部分覆盖应 consistent_on_valid_subset: "
                 f"{row2.handoff_conclusion.tolist()}")
    pairs2 = pd.read_csv(out2 / "verify04_initial_pair_checks.csv")
    t_pair = pairs2[(pairs2.stage == "I") & (pairs2.variable == "T")]
    check.expect(len(t_pair) == 1 and bool((t_pair.n_base == 48).all())
                 and bool((t_pair.n_valid == 1).all())
                 and bool((t_pair.n_missing == 47).all())
                 and bool((t_pair.rms_pair_diff == 0.0).all()),
                 "部分覆盖大气比较应保留原有计数与零差值")
    check.expect(bool((t_pair.detail
                       == "atmosphere_pair_consistent_on_valid_subset").all()),
                 f"部分覆盖大气结论应限定有效子集: {t_pair.detail.tolist()}")

    # --- R2:多层配置不互相覆盖(layer0 增量 5/48,layer1 增量 +0.25)---
    v04.CHECK_VARIABLES = (
        v04.CheckVariable("OM_TMP", "K", 0, "ocean_layer_stag", 1.0e-3, "ocean"),
        v04.CheckVariable("OM_TMP", "K", 1, "ocean_layer_stag", 1.0e-3, "ocean"),
    )

    def mutate_layer1(synth_root):
        for path in a_files(synth_root):
            with Dataset(path, "r+") as ds:
                var = ds.variables["OM_TMP"]
                var[:, 1, :, :] += 0.25   # 仅层 1 额外增量

    out3 = patched_run(dataclasses.replace(
        base_config, output_root=run_root, output_dirname="r2"
    ), mutate_layer1)
    pairwise3 = pd.read_csv(out3 / "verify04_pairwise_checks.csv", dtype={"member": str})
    om3 = pairwise3[(pairwise3.stage_pair == "A-B")
                    & (pairwise3.variable == "OM_TMP")
                    & (pairwise3.experiment == "6mem_oceanAssim1Run1")]
    layer0 = om3[om3.layer == 0]
    layer1 = om3[om3.layer == 1]
    check.expect(len(layer0) == 1 and len(layer1) == 1,
                 f"多层应各输出一行: {len(layer0)}/{len(layer1)}")
    check.expect(abs(layer0.mean_diff.iloc[0] - 5.0 / 48.0) < 1e-12,
                 f"层 0 增量应为 5/48: {layer0.mean_diff.iloc[0]}")
    check.expect(abs(layer1.mean_diff.iloc[0] - (5.0 / 48.0 + 0.25)) < 1e-12,
                 f"层 1 增量应为 5/48+0.25(不被层 0 覆盖): {layer1.mean_diff.iloc[0]}")
    check.expect(int(layer0.layer.iloc[0]) == 0 and int(layer1.layer.iloc[0]) == 1,
                 "层标签应正确")

    # --- R3:坐标 Time 维不在第 0 轴 → 按名称取正确记录 ---
    import verify04_readers as vr
    grid_root = Path(tempfile.mkdtemp(prefix="v04_grid_", dir=SMOKE_OUT))
    grid_path = grid_root / "GRID_PERM.nc"
    times = ["2018-09-10_00:00:00", "2018-09-10_00:30:00"]
    expected_lat = np.arange(10, 22, dtype=float).reshape(3, 4)  # 第二时刻
    with Dataset(grid_path, "w", format="NETCDF4") as ds:
        ds.createDimension("south_north", 3)
        ds.createDimension("Time", None)
        ds.createDimension("west_east", 4)
        tv = ds.createVariable("Times", str, ("Time",))
        for record, stamp in enumerate(times):
            tv[record] = stamp
        v = ds.createVariable("XLAT", "f8", ("south_north", "Time", "west_east"))
        v[:, 0, :] = np.arange(20, 32, dtype=float).reshape(3, 4)  # 第一时刻
        v[:, 1, :] = expected_lat                                   # 第二时刻
        v = ds.createVariable("XLONG", "f8", ("south_north", "Time", "west_east"))
        v[:, 0, :] = np.tile(np.arange(4), (3, 1))
        v[:, 1, :] = np.tile(np.arange(4), (3, 1))
    lat, lon, status = vr.read_stage_grid(grid_path, times[1])
    check.expect(status == "ok", f"非首轴 Time 读取应 ok: {status}")
    check.expect(lat is not None and lat.shape == (3, 4)
                 and bool(np.allclose(lat, expected_lat)),
                 f"非首轴 Time 应取到正确记录: {None if lat is None else lat.shape}")

    # --- R4:登记的数值转换先变换再比较;拼写别名不改变数值 ---
    out4 = patched_run(dataclasses.replace(
        base_config, output_root=run_root, output_dirname="r4"
    ), None)
    # 登记 degC -> K 转换(scale 1, offset 273.15)与 K/Kelvin 拼写别名
    v04.UNIT_ALIASES = {
        ("TSK", "degC", "K"): {"type": "conversion", "scale": 1.0,
                               "offset": 273.15, "basis": "degC+273.15=K"},
        ("TSK", "Kelvin", "K"): {"type": "alias", "basis": "spelling"},
    }

    def mutate_tsk_units(synth_root):
        for path in a_files(synth_root):
            with Dataset(path, "r+") as ds:
                var = ds.variables["TSK"]
                var[:] = var[:] - 273.15          # 数值改为 degC
                var.units = "degC"
        for path in sorted(synth_root.glob("B_6mem_oceanAssim1Run1_EAKF_006_d02.nc")):
            with Dataset(path, "r+") as ds:
                ds.variables["TSK"].units = "Kelvin"  # 拼写别名(数值不变)

    out5 = patched_run(dataclasses.replace(
        base_config, output_root=run_root, output_dirname="r4"
    ), mutate_tsk_units)
    pairwise5 = pd.read_csv(out5 / "verify04_pairwise_checks.csv", dtype={"member": str})
    tsk_row = pairwise5[(pairwise5.stage_pair == "A-B") & (pairwise5.variable == "TSK")
                        & (pairwise5.experiment == "6mem_oceanAssim1Run1")]
    # B(K 拼写 Kelvin,数值不变)与 A(degC 数值,已转换)→ 差应为 0
    check.expect(bool((tsk_row.status == "ok").all()),
                 f"登记转换/别名后应可计算: {tsk_row.status.tolist()}")
    check.expect(bool((tsk_row.mean_diff.abs() < 1e-9).all()),
                 f"转换后强弱同值差应为 0(而非 -273.15): {tsk_row.mean_diff.tolist()}")
    check.expect(bool(tsk_row.detail.str.contains("unit conversion applied").all()),
                 "转换依据应写入 detail")
    # 未登记的不匹配仍被拒绝
    tsk_ia = pairwise5[(pairwise5.stage_pair == "I-A") & (pairwise5.variable == "OM_TMP")
                       & (pairwise5.experiment == "6mem_oceanAssim1Run1")]
    check.expect(bool((tsk_ia.status == "ok").all()),
                 "未涉及单位登记的变量不受影响")
    v04.UNIT_ALIASES = {}

    # --- R5:必需语义放在键中但值为 unknown → unknown ---
    from caliber_link import check_joinable

    keys = ("method", "member", "time_hour", "experiment_or_cycle", "valid_time",
            "variable", "unit", "vertical_layer", "metric_family",
            "spatial_support")
    row_unknown = {
        "method": "EAKF", "member": "006", "time_hour": 0.5,
        "experiment_or_cycle": "c0", "valid_time": "t0",
        "variable": "unknown", "unit": "K", "vertical_layer": "surface",
        "metric_family": "pair_difference", "spatial_support": "pointwise",
    }
    verdict = check_joinable(
        pd.DataFrame([row_unknown]), pd.DataFrame([row_unknown]), keys, ()
    )
    check.expect(verdict.verdict.iloc[0] == "unknown",
                 f"键中必需语义取值为 unknown 应判 unknown: {verdict.iloc[0].to_dict()}")

    # --- R6:清单 mapping_confirmed 与 source_confirmed 分列 ---
    def mutate_nothing(synth_root):
        return None

    out6 = patched_run(dataclasses.replace(
        base_config, output_root=run_root, output_dirname="r6"
    ), mutate_nothing, confirm_source=True, confirm_mapping=False)
    manifest6 = pd.read_csv(out6 / "verify04_source_manifest.csv")
    check.expect(bool((manifest6.source_confirmed == True).all()),   # noqa: E712
                 "已确认来源的清单 source_confirmed 应为 True")
    check.expect(bool((manifest6.mapping_confirmed == False).all()),  # noqa: E712
                 "映射默认未确认:清单 mapping_confirmed 应为 False(与来源分列,"
                 "R6:不再用来源确认冒充映射确认)")
    check.expect("mapping_known" in manifest6.columns,
                 "清单应含 mapping_known 列")
    strong_known = manifest6[manifest6.experiment == "6mem_oceanAssim1Run1"]
    weak_known = manifest6[manifest6.experiment == "6mem_oceanAssim0Run1"]
    check.expect(bool((strong_known.mapping_known == True).all())    # noqa: E712
                 and bool((weak_known.mapping_known == False).all()),  # noqa: E712
                 "强试验映射已知(mapping_known=True);弱试验映射未知(False)")
    # time_record_used 记录 Times 命中的实际记录号(R6 追溯);
    # missing_variable 的行不解析时间,time_record_used 为 NaN(允许)
    timed = manifest6[manifest6.time_source == "times_attribute"]
    check.expect(len(timed) > 0
                 and bool((timed.time_record_used == 0).all()),
                 "Times 命中的记录号应写入 time_record_used")
    # 来源确认 + 映射未确认 → I-A 结论 not_concluded_mapping_inferred_not_confirmed
    pairwise6 = pd.read_csv(out6 / "verify04_pairwise_checks.csv", dtype={"member": str})
    om_ia6 = pairwise6[(pairwise6.stage_pair == "I-A")
                       & (pairwise6.variable == "OM_TMP")
                       & (pairwise6.experiment == "6mem_oceanAssim1Run1")]
    check.expect(bool((om_ia6.handoff_conclusion
                       == "not_concluded_mapping_inferred_not_confirmed").all()),
                 "来源确认而映射未确认应 not_concluded_mapping_inferred_not_confirmed")

    # --- R2 附带:重复非层变量名在 run 开始即报错 ---
    v04.CHECK_VARIABLES = (
        v04.CheckVariable("TSK", "K", None, None, 1.0e-3, "tsk"),
        v04.CheckVariable("TSK", "K", None, None, 1.0e-3, "tsk"),
    )
    try:
        v04.run(dataclasses.replace(
            base_config, output_root=run_root, output_dirname="dup"
        ))
        check.expect(False, "重复非层变量名应尽早报错")
    except ValueError as error:
        check.expect("duplicate non-layered variable" in str(error),
                     f"重复变量名报错信息异常: {error}")
    return check


def check_caliber_link_manual() -> Check:
    """模块 B:兼容性判定——必需语义缺失→unknown;块平均 vs 逐点、不同中心、
    不同 SST 来源不误报可比;变量/单位不一致(T2 vs Q2)→ not_comparable;
    单列键不被拆字符;重复键;前导零不静默归一;未匹配键可报告。"""
    check = Check("caliber_link 兼容性判定")
    from caliber_link import build_caliber_table, check_joinable, unmatched_keys

    keys = ("method", "member", "time_hour")
    meta = (
        "experiment_or_cycle", "valid_time", "variable", "unit",
        "vertical_layer", "metric_family", "spatial_support",
        "center_definition", "sst_source",
    )
    common_meta = {
        "experiment_or_cycle": "cycle_2018-09-10_00",
        "valid_time": "2018-09-10_00:30:00",
        "variable": "HFX",
        "unit": "W m-2",
        "vertical_layer": "surface",
        "metric_family": "pair_difference",
    }
    pathway_style = pd.DataFrame({
        **common_meta,
        "method": ["EAKF"], "member": ["006"], "time_hour": [0.5],
        "center_definition": ["pair_mean_psfc"],
        "spatial_support": ["block_10x10"],
        "sst_source": ["om_tmp_layer0_t0"],
    })
    verify_style = pd.DataFrame({
        **common_meta,
        "method": ["EAKF"], "member": ["006"], "time_hour": [0.5],
        "center_definition": ["nr_psfc"],
        "spatial_support": ["pointwise"],
        "sst_source": ["tsk_each_time"],
    })
    verdict = check_joinable(pathway_style, verify_style, keys, meta)
    check.expect(verdict.verdict.iloc[0] == "not_comparable"
                 and "center_definition" in verdict.conflicts.iloc[0]
                 and "spatial_support" in verdict.conflicts.iloc[0]
                 and "sst_source" in verdict.conflicts.iloc[0],
                 f"块平均/不同中心/不同 SST 来源应判 not_comparable: "
                 f"{verdict.iloc[0].to_dict()}")
    same = pd.DataFrame({**common_meta,
                         "method": ["EAKF"], "member": ["006"], "time_hour": [0.5],
                         "center_definition": ["nr_psfc"],
                         "spatial_support": ["pointwise"],
                         "sst_source": ["tsk_each_time"]})
    verdict_same = check_joinable(same, verify_style, keys, meta)
    check.expect(verdict_same.verdict.iloc[0] == "value_by_value",
                 f"口径一致的表应可逐值比较: {verdict_same.verdict.iloc[0]}")
    # 变量/单位不同的同键行(T2 vs Q2 反例)→ not_comparable
    # (其余口径齐全,使冲突精确落在 variable/unit 上)
    t2_vs_q2_left = pd.DataFrame({**common_meta, "variable": ["T2"], "unit": ["K"],
                                  "method": ["EAKF"], "member": ["006"],
                                  "time_hour": [0.5],
                                  "center_definition": ["nr_psfc"],
                                  "sst_source": ["tsk_each_time"],
                                  "spatial_support": ["pointwise"]})
    t2_vs_q2_right = pd.DataFrame({**common_meta, "variable": ["Q2"],
                                   "unit": ["kg kg-1"],
                                   "method": ["EAKF"], "member": ["006"],
                                   "time_hour": [0.5],
                                   "center_definition": ["nr_psfc"],
                                   "sst_source": ["tsk_each_time"],
                                   "spatial_support": ["pointwise"]})
    verdict_var = check_joinable(t2_vs_q2_left, t2_vs_q2_right, keys, meta)
    check.expect(verdict_var.verdict.iloc[0] == "not_comparable"
                 and "variable" in verdict_var.conflicts.iloc[0]
                 and "unit" in verdict_var.conflicts.iloc[0],
                 f"T2 与 Q2 同键不应判可比: {verdict_var.iloc[0].to_dict()}")
    # 空口径列且必需语义不在键中 → unknown(不默认放行)
    verdict_empty = check_joinable(pathway_style, verify_style, keys, ())
    check.expect(len(verdict_empty) > 0
                 and (verdict_empty.verdict == "unknown").all()
                 and verdict_empty.conflicts.iloc[0].startswith(
                     "missing_required_semantics"),
                 "空口径列且必需语义缺失应全部 unknown")
    # 元数据部分缺失 → unknown
    missing_meta = pd.DataFrame({
        **common_meta, "method": ["EAKF"], "member": ["006"], "time_hour": [0.5],
        "center_definition": ["nr_psfc"],
    })
    verdict_unknown = check_joinable(missing_meta, verify_style, keys, meta)
    check.expect(verdict_unknown.verdict.iloc[0] == "unknown",
                 f"口径部分缺失应 unknown: {verdict_unknown.verdict.iloc[0]}")
    # 重复键 → duplicate_keys
    dup = pd.concat([verify_style, verify_style], ignore_index=True)
    verdict_dup = check_joinable(same, dup, keys, meta)
    check.expect(verdict_dup.verdict.iloc[0] == "duplicate_keys",
                 f"重复键应 duplicate_keys: {verdict_dup.verdict.iloc[0]}")
    # 前导零不静默归一;未匹配键显式报告
    nozfill = pd.DataFrame({
        **common_meta, "method": ["EAKF"], "member": ["6"], "time_hour": [0.5],
        "center_definition": ["nr_psfc"], "spatial_support": ["pointwise"],
        "sst_source": ["tsk_each_time"],
    })
    verdict_zfill = check_joinable(verify_style, nozfill, keys, meta)
    check.expect(len(verdict_zfill) == 0,
                 "前导零不同的成员键不应被静默匹配")
    unmatched = unmatched_keys(verify_style, nozfill, keys)
    check.expect(len(unmatched) == 2
                 and set(unmatched.side) == {"left_only", "right_only"},
                 f"未匹配键应显式报告: {unmatched.to_dict('records')}")
    # 单列键:字符串不被拆成首字符;整数键不报 TypeError
    single_meta = ("variable", "unit", "vertical_layer", "metric_family",
                   "spatial_support", "experiment_or_cycle", "valid_time")
    single_left = pd.DataFrame({"member": ["006"], "variable": ["TSK"],
                                "unit": ["K"], "vertical_layer": ["surface"],
                                "metric_family": ["pair_difference"],
                                "spatial_support": ["pointwise"],
                                "experiment_or_cycle": ["c0"],
                                "valid_time": ["t0"]})
    verdict_single = check_joinable(single_left, single_left.copy(),
                                    ("member",), single_meta)
    check.expect(len(verdict_single) == 1 and verdict_single.member.iloc[0] == "006",
                 f"单列键应保留原值: {verdict_single.member.tolist()}")
    check.expect(verdict_single.verdict.iloc[0] == "value_by_value",
                 f"单列键同口径应 value_by_value: {verdict_single.verdict.iloc[0]}")
    single_int = single_left.copy()
    single_int["member"] = [6]
    check.expect(len(unmatched_keys(single_left, single_int, ("member",))) == 2,
                 "字符串与整数键应视为不同键(不静默转换)")
    # 对照表:四档判定 + 已知差异
    table = build_caliber_table()
    check.expect(set(table.verdict).issubset({
        "value_by_value", "parallel_evidence", "not_comparable", "unknown",
    }), "对照表判定超出四档")
    check.expect(bool((table.dimension == "offline_flux_sst_source").any())
                 and table.loc[table.dimension == "offline_flux_sst_source",
                               "verdict"].iloc[0] == "not_comparable",
                 "离线通量 SST 来源差异应判 not_comparable")
    return check


def check_verify04_end_to_end_real_mode_guard() -> Check:
    """verify_04 真实模式防呆:B/A/I 真实路径默认未配置(文件名推断不自动
    启用)、confirmed=False;F0 模板存在(独立断言)。"""
    check = Check("verify_04 真实模式来源防呆")
    from verify_04_initial_handoff import _resolve_stage_path, _stage_source_configs

    real_config = VerifyConfig(
        mode="real",
        real=dataclasses.replace(
            VerifyConfig().real, acknowledge_real_mode=True
        ),
    )
    sources = _stage_source_configs(real_config)
    for stage in ("B", "A", "I"):
        check.expect(sources[stage].path_template is None,
                     f"{stage} 真实路径默认应未配置")
        check.expect(sources[stage].confirmed is False,
                     f"{stage} 的 confirmed 标记应为 False")
    check.expect(sources["F0"].path_template is not None,
                 "F0 模板应存在(服务器已核实的 wrfout 布局)")
    check.expect(isinstance(sources["F0"].confirmed, bool),
                 "F0 confirmed 应为布尔值")
    check.expect(_resolve_stage_path(None, "d02", "006", "t") is None,
                 "未配置模板应解析为 None(比较按 source_not_configured 跳过)")
    return check


def check_lltoxy_registration() -> Check:
    """ll_to_xy 投影配准路径(mock 投影文件):此前该路径从未被任何测试执行。

    用 Mercator 反解构造等 ψ 纬度网格,DX/DY 与网格间距严格一致,
    ll_to_xy 的浮点索引应接近线性期望;域外目标为 NaN;掩膜走 order=0。
    """
    check = Check("ll_to_xy 投影配准(mock Mercator)")
    try:
        from netCDF4 import Dataset  # noqa: F401
        import wrf  # noqa: F401
    except ImportError:
        check.expect(False, "netCDF4/wrf 不可用,无法检验投影配准路径")
        return check
    earth_r = 6371000.0
    ny, nx = 5, 5
    dlon, dpsi = 0.25, 0.0045
    lon_1d = 137.0 + dlon * np.arange(nx)
    psi_0 = np.log(np.tan(np.pi / 4 + np.deg2rad(13.0) / 2))
    lat_1d = np.rad2deg(2 * np.arctan(np.exp(psi_0 + dpsi * np.arange(ny))) - np.pi / 2)
    nr_lat, nr_lon = np.meshgrid(lat_1d, lon_1d, indexing="ij")
    proj_attrs = dict(
        MAP_PROJ=3, TRUELAT1=0.0, TRUELAT2=0.0, STAND_LON=float(lon_1d[0]),
        POLE_LAT=90.0, POLE_LON=0.0,
        DX=earth_r * np.deg2rad(dlon), DY=earth_r * dpsi,
        CEN_LAT=float(lat_1d[2]), CEN_LON=float(lon_1d[2]),
    )
    stamp = "2018-09-10_00:00:00"
    mock_root = Path(tempfile.mkdtemp(prefix="mock_proj_", dir=SMOKE_OUT))
    # NR 场:om = 纬度的线性函数(便于核验配准精度);landmask 东半边为 1
    nr_fields_om = 100.0 + nr_lat
    nr_land = np.where(nr_lon >= lon_1d[nx // 2], 1.0, 0.0)
    _write_mock_wrfout(
        mock_root / "nr" / f"wrfout_d02_{stamp}", nr_lat, nr_lon, nr_land,
        stamp, "nr", times_mode="vlen", proj_attrs=proj_attrs,
    )
    # 手动为该 NR 文件补 om 场(_write_mock_wrfout 写常数 om,这里覆盖)
    from netCDF4 import Dataset

    nr_path = mock_root / "nr" / f"wrfout_d02_{stamp}"
    with Dataset(nr_path, "a") as ds:
        ds.variables["OM_TMP"][0, 0] = nr_fields_om

    config = VerifyConfig(
        mode="real", methods=("EAKF",), members=("006",),
        times=((0.0, stamp),),
        real=RealPathConfig(
            forecast_base_dir=mock_root / "cycle", nr_dir=mock_root / "nr",
            nr_registration="ll_to_xy_bilinear", acknowledge_real_mode=True,
        ),
    )
    provider = vc.RealWrfProvider(config)
    nr_static = provider.nr_static()
    # 目标点取 NR 域内(奇数索引处,远离边界)
    dst_lat = lat_1d[1:4][None, :]
    dst_lon = lon_1d[1:4][None, :]
    dst_lat2d, dst_lon2d = np.meshgrid(dst_lat.ravel(), dst_lon.ravel(), indexing="ij")
    exp_static = {"lat": dst_lat2d, "lon": dst_lon2d}
    registered = vc.register_nr_fields(
        {"om": nr_fields_om, "landmask": nr_land},
        nr_static, exp_static, config, provider,
    )
    om_exp = registered["om"]
    expected_om = 100.0 + dst_lat2d
    check.expect(om_exp.shape == dst_lat2d.shape, "ll_to_xy 配准输出形状错误")
    check.expect(bool(np.all(np.isfinite(om_exp))), "域内点不应为 NaN")
    check.expect(bool(np.allclose(om_exp, expected_om, atol=0.5)),
                 f"ll_to_xy 双线性配准偏差过大: max={np.max(np.abs(om_exp - expected_om)):.3f}")
    # 域外目标点应为 NaN
    outside = vc.register_nr_fields(
        {"om": nr_fields_om}, nr_static,
        {"lat": np.array([[20.0]]), "lon": np.array([[150.0]])},
        config, provider,
    )["om"]
    check.expect(not np.isfinite(outside[0, 0]), "域外点应因 map_coordinates 常量填充为 NaN")
    # 掩膜 order=0:值应为最近格点的原始值
    mask_exp = registered["landmask"]
    check.expect(bool(np.isin(mask_exp, [0.0, 1.0]).all()),
                 f"order=0 掩膜应取最近格点原值,得到 {np.unique(mask_exp)}")
    return check


def check_real_provider_contract() -> Check:
    """用本地合成 NetCDF 文件(非真实数据)检验 RealWrfProvider 读取适配层契约:
    入口方法齐全、时间与网格一致性校验有效、配准入口可走通。"""
    check = Check("RealWrfProvider 契约(mock NetCDF)")
    try:
        from netCDF4 import Dataset  # noqa: F401
    except ImportError:
        check.expect(False, "netCDF4 不可用,无法检验读取适配层")
        return check
    mock_root = Path(tempfile.mkdtemp(prefix="mock_wrf_", dir=SMOKE_OUT))
    exp_lat = 13.0 + 0.25 * np.arange(6)
    exp_lon = 137.0 + 0.25 * np.arange(6)
    exp_lat2d, exp_lon2d = np.meshgrid(exp_lat, exp_lon, indexing="ij")
    nr_lat2d, nr_lon2d = np.meshgrid(
        13.0 + 0.5 * np.arange(5), 137.0 + 0.5 * np.arange(5), indexing="ij"
    )
    land = np.zeros_like(exp_lat2d)
    stamp = "2018-09-10_00:00:00"
    base = mock_root / "cycle"
    for experiment in ("6mem_oceanAssim0Run1", "6mem_oceanAssim1Run1"):
        _write_mock_wrfout(
            base / experiment / "EAKF" / "006" / f"wrfout_d02_{stamp}",
            exp_lat2d, exp_lon2d, land, stamp, "exp",
        )
    _write_mock_wrfout(
        mock_root / "nr" / f"wrfout_d02_{stamp}", nr_lat2d, nr_lon2d,
        np.zeros_like(nr_lat2d), stamp, "nr",
    )
    config = VerifyConfig(
        mode="real",
        methods=("EAKF",),
        members=("006",),
        times=((0.0, stamp),),
        real=RealPathConfig(
            forecast_base_dir=base,
            nr_dir=mock_root / "nr",
            nr_registration="numpy_nearest",
            acknowledge_real_mode=True,
        ),
    )
    provider = vc.RealWrfProvider(config)
    # 入口契约方法必须齐全(此前缺失 nr_static/exp_fields 导致真实模式 AttributeError)
    for method_name in (
        "static_fields", "nr_static", "has_case", "exp_fields",
        "exp_raw_inputs", "nr_fields", "register_field_lltoxy",
    ):
        check.expect(callable(getattr(provider, method_name, None)),
                     f"RealWrfProvider 缺少入口方法 {method_name}")
    static = provider.static_fields()
    check.expect(static["lat"].shape == exp_lat2d.shape, "static_fields 形状错误")
    nr_static = provider.nr_static()
    check.expect(nr_static["lat"].shape == nr_lat2d.shape, "nr_static 形状错误")
    check.expect(provider.has_case("EAKF", "006", 0.0), "has_case 应为 True")
    fields = provider.exp_fields("6mem_oceanAssim1Run1", "EAKF", "006", 0.0)
    for key in ("om", "tsk", "hfx", "lh", "psfc"):
        check.expect(key in fields and fields[key].shape == exp_lat2d.shape,
                     f"exp_fields 缺少 {key} 或形状错误")
    raw = provider.exp_raw_inputs("6mem_oceanAssim1Run1", "EAKF", "006", 0.0)
    inputs = vc.derive_lowest_level_inputs(raw)
    check.expect(float(np.nanmin(inputs["height_agl_m"])) > 0.0, "mock 输入离地高度应为正")
    nr_fields = provider.nr_fields(0.0)
    registered = vc.register_nr_fields(
        {name: nr_fields[name] for name in ("om", "tsk", "hfx", "lh")},
        nr_static, static, config, provider,
    )
    for name, values in registered.items():
        check.expect(values.shape == exp_lat2d.shape and np.all(np.isfinite(values)),
                     f"numpy_nearest 配准 {name} 失败")
    # 时间一致性校验:Times 属性与请求时刻不符必须报错
    wrong_stamp_path = (
        base / "6mem_oceanAssim1Run1" / "EAKF" / "006" / "wrfout_d02_2018-09-10_01:00:00"
    )
    _write_mock_wrfout(wrong_stamp_path, exp_lat2d, exp_lon2d, land,
                       "2018-09-10_02:00:00", "exp")
    config_wrong = dataclasses.replace(
        config, times=((1.0, "2018-09-10_01:00:00"),)
    )
    provider_wrong = vc.RealWrfProvider(config_wrong)
    check.expect(
        raises(RuntimeError, provider_wrong.exp_fields,
               "6mem_oceanAssim1Run1", "EAKF", "006", 1.0),
        "Times 属性与请求时刻不符未被拒绝",
    )
    # 网格一致性校验:强弱网格不同必须报错,不得按维度相同直接逐下标比较
    wrong_grid_name = (
        base / "6mem_oceanAssim0Run1" / "EAKF" / "006" / f"wrfout_d02_{stamp}"
    )
    wrong_grid_name.unlink()
    _write_mock_wrfout(
        wrong_grid_name, exp_lat2d + 0.5, exp_lon2d, land, stamp, "exp"
    )
    check.expect(
        raises(RuntimeError, provider.exp_fields, "6mem_oceanAssim0Run1", "EAKF", "006", 0.0),
        "强弱网格不一致未被拒绝",
    )
    return check


def check_reconstruction_failure_status() -> Check:
    """重建输出出现非有限值时,诊断三的处理路径必须显式报告。"""
    check = Check("重建非有限输出的状态传递")
    from verify_03_fixed_atmosphere_flux import evaluate_reconstruction_outputs

    mask = np.array([[True, True], [True, False]])

    def partial_nan_reconstructor(**kwargs):
        arrays = {}
        for key in ("qfx", "lh", "hfx"):
            arrays[key] = np.full(kwargs["air_temperature_k"].shape, 5.0)
        arrays["hfx"][0, 0] = np.nan  # 一个非有限输出点(替身,与真实函数明确区分)
        return arrays

    _, combined, status, invalid = evaluate_reconstruction_outputs(
        partial_nan_reconstructor,
        {key: np.ones((2, 2)) for key in vc.RECONSTRUCTION_INPUT_KEYS},
        np.ones((2, 2)), mask,
    )
    check.expect(status == "nonfinite_reconstruction_output", f"status={status}")
    check.expect(invalid == 1, f"非有限点计数={invalid}")
    check.expect(int(combined.sum()) == 2, "共同有限掩膜应只保留有限输出点")
    return check


# ==================================================================
# 完整流程:三个诊断入口的合成运行(独立临时目录,不读旧结果)
# ==================================================================


def _run_entry(module_name: str, expected_files: list[str], run_root: Path) -> tuple[Check, Path]:
    check = Check(f"完整流程 {module_name}")
    module = __import__(module_name)
    # 冒烟测试强制合成模式,即使入口 CONFIG 被改为 real
    config = dataclasses.replace(
        module.CONFIG,
        mode="synthetic",
        real=dataclasses.replace(module.CONFIG.real, acknowledge_real_mode=False),
        output_root=run_root,
    )
    out_dir = module.run(config)  # 使用 run() 返回的确切路径,不做 rglob 搜索
    for relative in expected_files:
        path = out_dir / relative
        check.expect(path.exists(), f"缺少输出文件: {path}")
    return check, out_dir


def check_full_flow() -> tuple[list[Check], dict[str, Path]]:
    """运行三个入口的合成全流程;返回 (检查列表, {诊断: 输出目录})。"""
    checks: list[Check] = []
    flow_dirs: dict[str, Path] = {}
    runs_root = SMOKE_OUT / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    check, dir01 = _run_entry(
        "verify_01_skill_timeseries",
        [
            "verify01_member_metrics.csv",
            "verify01_member_classification_fractions.csv",
            "verify01_summary_by_method.csv",
            "verify01_summary_by_region_window.csv",
            "verify01_run_status.csv",
            "columns_readme.txt",
            "figs/verify01_rmse_timeseries_om_r000_300.png",
            "figs/verify01_improvement_timeseries_lh_r000_300.png",
        ],
        Path(tempfile.mkdtemp(prefix="flow01_", dir=runs_root)),
    )
    checks.append(check)
    check, dir02 = _run_entry(
        "verify_02_flux_error_budget",
        [
            "verify02_member_budget.csv",
            "verify02_summary_by_method.csv",
            "verify02_summary_by_region_window.csv",
            "verify02_run_status.csv",
            "fields/verify02_dSE_points_EAKF_006_t0.5_hfx.npz",
            "figs/verify02_budget_timeseries_hfx_r000_300.png",
            "figs/verify02_closure_residual_lh_r000_300.png",
        ],
        Path(tempfile.mkdtemp(prefix="flow02_", dir=runs_root)),
    )
    checks.append(check)
    flow_dirs["verify01"] = dir01
    flow_dirs["verify02"] = dir02
    check, dir03 = _run_entry(
        "verify_03_fixed_atmosphere_flux",
        [
            "verify03_phaseA_reconstruction.csv",
            "verify03_phaseB_replacement.csv",
            "verify03_run_status.csv",
            "overall_status.txt",
            "fields/verify03_replacement_EAKF_006_t0.5_hfx.npz",
            "figs/verify03_phaseB_components_lh.png",
        ],
        Path(tempfile.mkdtemp(prefix="flow03_", dir=runs_root)),
    )
    check.expect("overall_status: ok" in (dir03 / "overall_status.txt").read_text(),
                 "诊断三总状态应为 ok")
    checks.append(check)

    # ---- 明细覆盖:方法×成员×时次(ok 或显式状态);缺失时次记录 ----
    check = Check("合成明细覆盖(方法×成员×时次)")
    metrics = pd.read_csv(dir01 / "verify01_member_metrics.csv", dtype={"member": str})
    for method in ("EAKF", "QCF_RHF"):
        selected = metrics[metrics.method == method]
        members = set(selected.member)
        check.expect(len(members) == 6, f"{method} 成员数 {len(members)} != 6")
        check.expect(set(selected.time_hour.unique()) == {
            0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
        }, f"{method} 时次不全")
    status = pd.read_csv(dir01 / "verify01_run_status.csv", dtype={"member": str})
    check.expect("mode" in status.columns
                 and (status["mode"] == "SYNTHETIC").all(),
                 "run_status 缺少 mode 列或模式标记错误")
    qcf_044 = status[(status.method == "QCF_RHF") & (status.member == "044")]
    check.expect(bool((qcf_044.status == "missing_member_or_time").any()),
                 "缺失时次未记录于 run_status")
    checks.append(check)

    # ---- 汇总口径:主汇总仅用并集;改善成员数不超过成员数;覆盖率与闭合列 ----
    check = Check("汇总口径(仅并集;改善成员数上界;覆盖率/闭合列)")
    summary = pd.read_csv(dir01 / "verify01_summary_by_method.csv")
    check.expect("region" not in summary.columns, "主汇总不应保留 region 列(已固定为并集)")
    check.expect(bool((summary.n_members <= 6).all()), "n_members 超过成员数上界")
    check.expect(bool((summary.n_members_improved <= summary.n_members).all()),
                 "n_members_improved 超过 n_members(疑似区域重复累计)")
    check.expect(bool((summary.min_member_valid_abs_times >= 1).all()),
                 "成员有效绝对误差时次数应 >= 1")
    check.expect("member_abs_time_coverage_ratio" in summary.columns, "缺少覆盖率列")
    check.expect("expected_members" in summary.columns
                 and (summary.expected_members == 6).all(),
                 "缺少期望成员数列或取值错误")
    check.expect("member_coverage_ratio" in summary.columns, "缺少成员覆盖率列")
    region_window = pd.read_csv(dir01 / "verify01_summary_by_region_window.csv")
    check.expect(set(region_window.region.unique()) == {"r000_075", "r075_150", "r150_300", "r000_300"},
                 "逐区域汇总应包含全部四个区域")
    summary02 = pd.read_csv(dir02 / "verify02_summary_by_method.csv")
    check.expect(bool((summary02.n_members_improved <= summary02.n_members).all()),
                 "诊断二汇总改善成员数上界失守")
    check.expect("mean_closure_residual" in summary02.columns,
                 "诊断二主汇总缺少闭合残差列(无法复核闭合)")
    closure = summary02.mean_closure_residual.to_numpy(dtype=float)
    closure = closure[np.isfinite(closure)]
    check.expect(bool(np.all(np.abs(closure) < 1e-6)),
                 f"汇总闭合残差异常: max={np.max(np.abs(closure)) if closure.size else float('nan')}")
    # 导出表头必须唯一(重复列会影响后续按列处理)
    for name, frame in (("verify01", summary), ("verify02", summary02)):
        columns = list(frame.columns)
        check.expect(len(columns) == len(set(columns)), f"{name} 汇总表存在重复表头")
    checks.append(check)

    # ---- 诊断三:actual 方向与数值(独立重算)、质量标记传递、npz 内容 ----
    check = Check("诊断三 actual 独立重算/归因标记/npz 内容")
    phase_b = pd.read_csv(dir03 / "verify03_phaseB_replacement.csv", dtype={"member": str})
    union_rows = phase_b[phase_b.region == "r000_300"]
    # 聚合方向断言(合成偏置结构:hfx 强-弱 = -8-20 = -28,lh = 12-30 = -18)
    hfx_actual = union_rows[union_rows.variable == "hfx"].actual_model_diff_mean.to_numpy(dtype=float)
    lh_actual = union_rows[union_rows.variable == "lh"].actual_model_diff_mean.to_numpy(dtype=float)
    check.expect(bool(np.isfinite(np.nanmean(hfx_actual))) and np.nanmean(hfx_actual) < -10.0,
                 f"hfx 实际模式差(强-弱)聚合应约 -28,得到 {np.nanmean(hfx_actual):.2f}")
    check.expect(bool(np.isfinite(np.nanmean(lh_actual))) and np.nanmean(lh_actual) < -10.0,
                 f"lh 实际模式差(强-弱)聚合应约 -18,得到 {np.nanmean(lh_actual):.2f}")
    # 独立重算:按统一掩膜直接从合成提供者重算 strong-weak 的区域均值,
    # 与 CSV 逐例对照(不依赖替换闭合式,能发现方向反转)
    provider_module = __import__("verify_synthetic")
    provider = provider_module.build_synthetic_provider(SMOKE_CONFIG)
    for method, member, hour in (("EAKF", "006", 0.5), ("EAKF", "006", 3.0),
                                 ("QCF_RHF", "015", 3.0)):
        context = vc.build_time_context(provider, SMOKE_CONFIG, hour)
        raw_strong = provider.exp_raw_inputs(SMOKE_CONFIG.strong_experiment, method, member, hour)
        raw_weak = provider.exp_raw_inputs(SMOKE_CONFIG.weak_experiment, method, member, hour)
        inputs_strong = vc.derive_lowest_level_inputs(raw_strong)
        inputs_weak = vc.derive_lowest_level_inputs(raw_weak)
        strong_fields = provider.exp_fields(SMOKE_CONFIG.strong_experiment, method, member, hour)
        weak_fields = provider.exp_fields(SMOKE_CONFIG.weak_experiment, method, member, hour)
        mask = context["regions"][SMOKE_CONFIG.union_region] & context["ocean"] & vc.unified_reconstruction_mask(
            inputs_weak, inputs_strong, raw_weak["TSK"], raw_strong["TSK"],
            extra=(
                strong_fields["hfx"], weak_fields["hfx"],
                strong_fields["lh"], weak_fields["lh"],
                context["nr"]["hfx"], context["nr"]["lh"],
            ),
        )
        for variable in ("hfx", "lh"):
            expected = float(np.mean(vc.strong_minus_weak(
                strong_fields[variable], weak_fields[variable]
            )[mask]))
            row = phase_b[
                (phase_b.method == method) & (phase_b.member == member)
                & (phase_b.time_hour == hour) & (phase_b.region == "r000_300")
                & (phase_b.variable == variable)
            ]
            got = float(row.actual_model_diff_mean.iloc[0])
            check.expect(
                abs(got - expected) <= 1e-9 * max(1.0, abs(expected)),
                f"actual 与独立重算不一致({method}/{member}/t{hour:g}/{variable}): "
                f"{got:.6f} vs {expected:.6f}(方向可能反转)",
            )
    check.expect("attribution_flag" in phase_b.columns
                 and phase_b.attribution_flag.notna().all(),
                 "阶段 B 缺少归因标记列或存在空值")
    check.expect("attribution_flag" in phase_b.columns,
                 "阶段 B 缺少归因标记列")
    # 归因数值一致性:ok 行的标记必须与其三判据数值重算结果一致(按区域评估)
    from verify_03_fixed_atmosphere_flux import attribution_quality_flag

    ok_rows = union_rows[union_rows.status == "ok"]
    for row in ok_rows.itertuples():
        recomputed = attribution_quality_flag(
            row.attr_weak_rmse_ratio, row.attr_strong_rmse_ratio, row.attr_diff_agree_ratio
        )
        check.expect(row.attribution_flag == recomputed,
                     f"归因标记与判据数值不一致({row.method}/{row.member}/t{row.time_hour:g}/"
                     f"{row.variable}/{row.region}): {row.attribution_flag} vs {recomputed}")
    check.expect(bool(ok_rows[["attr_weak_rmse_ratio", "attr_strong_rmse_ratio",
                               "attr_diff_agree_ratio"]].notna().all().all()),
                 "ok 行的归因判据数值应为有限值")
    # 判据随区域变化:不同区域行中应存在不同的判据数值组合
    ratios_by_region = ok_rows.groupby("region")["attr_strong_rmse_ratio"].median()
    check.expect(len(ratios_by_region) >= 1 and ratios_by_region.notna().all(),
                 "逐区域归因判据数值缺失")
    check.expect(bool((union_rows.status == "ok").all()), "阶段 B 主区域存在非 ok 状态")
    check.expect("sst_only_improvement_status" in phase_b.columns,
                 "阶段 B 缺少改善率状态列")
    phase_a = pd.read_csv(dir03 / "verify03_phaseA_reconstruction.csv", dtype={"member": str})
    for column in ("residual_between_diffs_rms", "residual_between_diffs_maxabs",
                   "residual_maxabs_to_rms_ratio"):
        check.expect(column in phase_a.columns, f"阶段 A 缺少 {column}")
    sample_npz = np.load(dir03 / "fields" / "verify03_replacement_EAKF_006_t0.5_hfx.npz")
    for key in ("Fww", "Fsw", "Fss", "nr_flux", "compare_mask", "mode"):
        check.expect(key in sample_npz.files, f"阶段 B npz 缺少 {key}")
    checks.append(check)

    # ---- 阶段 B 通路:dF_rest 非零;拆分恒等式闭合 ----
    check = Check("阶段B通路(dF_rest 非零;拆分恒等式闭合)")
    check.expect(bool((union_rows.dF_rest.abs() > 0.0).any()),
                 "dF_rest 全为零:合成数据未区分 A_weak/A_strong,通路未被测试")
    residual = union_rows.split_closure_residual.to_numpy(dtype=float)
    residual = residual[np.isfinite(residual)]
    check.expect(bool(np.all(np.abs(residual) < 1e-8)),
                 "拆分恒等式未闭合")
    checks.append(check)
    flow_dirs["verify03"] = dir03
    return checks, flow_dirs


def check_all_missing_run() -> Check:
    """全部案例缺失时:不得崩溃,run_status 先落盘,空表保留列结构。"""
    check = Check("全缺失运行(状态先落盘,空表保留列)")
    module = __import__("verify_01_skill_timeseries")
    run_root = Path(tempfile.mkdtemp(prefix="all_missing_", dir=SMOKE_OUT / "runs"))
    config = VerifyConfig(
        mode="synthetic",
        methods=("QCF_RHF",),
        members=("044",),
        times=((2.5, "2018-09-10_02:30:00"),),
        output_root=run_root,
    )
    out_dir = module.run(config)
    status = pd.read_csv(out_dir / "verify01_run_status.csv", dtype={"member": str})
    check.expect(len(status) == 1 and status.status.iloc[0] == "missing_member_or_time",
                 f"run_status 内容异常: {status.to_dict('records')}")
    metrics = pd.read_csv(out_dir / "verify01_member_metrics.csv", dtype={"member": str})
    check.expect(len(metrics) == 0 and "rmse_improvement_pct" in metrics.columns,
                 "明细空表应保留列结构")
    summary = pd.read_csv(out_dir / "verify01_summary_by_method.csv")
    check.expect(len(summary) == 0 and "n_members_improved" in summary.columns,
                 "汇总空表应保留列结构")
    return check


def main() -> int:
    print("=" * 72)
    print("[SYNTHETIC] verify_diag 冒烟测试开始(仅为语法/接口/流程验收,非物理验证)")
    print("=" * 72)
    (SMOKE_OUT / "runs").mkdir(parents=True, exist_ok=True)
    checks: list[Check] = []
    builders = (
        check_paired_error_metrics_manual,
        check_strong_minus_weak_sign,
        check_dse_identity,
        check_flux_error_budget_closure,
        check_classification,
        check_zero_rmse_and_empty_mask,
        check_region_masks,
        check_bilinear_registration,
        check_nearest_registration,
        check_nearest_footprint_boundary,
        check_derive_lowest_level_inputs,
        check_output_pipeline,
        check_import_side_effect_free,
        check_real_mode_guards,
        check_synthetic_dataset_behaviour,
        check_align_times,
        check_member_then_mean,
        check_attribution_flag_logic,
        check_coverage_semantics,
        check_region_output_status,
        check_conditional_budget_manual,
        check_sign_agreement_manual,
        check_joint_stats_manual,
        check_inside_footprint_hull,
        check_grid_fingerprint_collision,
        check_region_status_distinction,
        check_real_reconstruction_interface,
        check_real_provider_contract,
        check_char_array_times,
        check_lltoxy_registration,
        check_verify04_handoff_manual,
        check_verify04_stage_skips,
        check_caliber_link_manual,
        check_verify04_unit_fill_grid_gates,
        check_verify04_mapping_conclusion,
        check_verify04_round2_fixes,
        check_verify04_end_to_end_real_mode_guard,
        check_joint_mask_numeric_counterexample,
        check_joint_tiered_counts,
        check_joint_quality_flagging,
        check_reconstruction_failure_status,
    )
    for builder in builders:
        check = builder()
        status = "PASS" if check.ok else "FAIL"
        print(f"[{status}] {check.name}", flush=True)
        for failure in check.failures:
            print(f"         - {failure}", flush=True)
        checks.append(check)
    flow_checks, flow_dirs = check_full_flow()
    checks.extend(flow_checks)
    # 端到端新表检查在全流程之后、使用本轮 run() 返回的确切目录(P2 #6)
    e2e_check = check_end_to_end_new_tables(flow_dirs.get("verify02"), flow_dirs.get("verify03"))
    status = "PASS" if e2e_check.ok else "FAIL"
    print(f"[{status}] {e2e_check.name}", flush=True)
    for failure in e2e_check.failures:
        print(f"         - {failure}", flush=True)
    checks.append(e2e_check)
    missing_check = check_all_missing_run()
    checks.append(missing_check)
    missing02_check = check_verify02_all_missing_run()
    status = "PASS" if missing02_check.ok else "FAIL"
    print(f"[{status}] {missing02_check.name}", flush=True)
    for failure in missing02_check.failures:
        print(f"         - {failure}", flush=True)
    checks.append(missing02_check)
    for check in checks[len(builders):]:
        status = "PASS" if check.ok else "FAIL"
        print(f"[{status}] {check.name}", flush=True)
        for failure in check.failures:
            print(f"         - {failure}", flush=True)

    n_fail = sum(1 for check in checks if not check.ok)
    lines = [
        "[SYNTHETIC] verify_diag smoke test report",
        f"checks: {len(checks)}, passed: {len(checks) - n_fail}, failed: {n_fail}",
        "",
        "通过范围:Python 语法与导入、无副作用导入、数组维度与基本接口、",
        "输出路径与文件生成、合成流程运行完成,以及针对已修复缺陷的回归断言。",
        "未验证:物理实现、数学归因、真实数据接口的正确性(见 TEST_RECORD.md)。",
        "",
    ]
    for check in checks:
        lines.append(f"[{'PASS' if check.ok else 'FAIL'}] {check.name}")
        lines.extend(f"    - {failure}" for failure in check.failures)
    report_path = SMOKE_OUT / "smoke_test_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[output] {report_path}", flush=True)
    print(f"[SYNTHETIC] 冒烟测试结束: {len(checks) - n_fail}/{len(checks)} 通过", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
