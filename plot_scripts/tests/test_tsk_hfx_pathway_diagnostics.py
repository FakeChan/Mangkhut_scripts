"""tsk_hfx_pathway_diagnostics 的数学、掩膜、插值与中心对齐检查。

计划要求的测试(全部使用合成数据,不读 wrfout、不访问服务器路径):
  - test_budget_sign_and_closure            预算符号与闭合
  - test_memberwise_average_order           成员内先算后平均的顺序
  - test_common_mask_excludes_fill_and_nan  fill/NaN 剔除,真 0 保留
  - test_reconstruction_near_zero_gate      响应 RMS 近零时不生成比率/pass
  - test_reconstruction_quality_states      strict/limited/fail 三态
  - test_storm_translation_recovery         整格平移恢复 ≤1e-8;非整格报插值误差
  - test_zero_shift_and_no_extrapolation    零平移评分相同;凸包外 NaN 不外推

补充检查:
  - test_theta_surface_and_destaggering     θ_surface 公式与去交错
  - test_original_mask_memberwise_rmse_order 原掩膜下逐成员 RMSE 后平均 ≠ 集合平均场 RMSE
  - test_three_term_identity                温差误差恒等式闭合(summarize_case 端到端)
  - test_classify_spatial_counts            四类空间对应计数
  - test_reconstruction_floor_consistency   本封装与 reconstruct_ocean_fluxes 一致,−250 下限计数
  - test_config_snapshot_serializable       配置快照可 JSON 化

运行:
  python tests/test_tsk_hfx_pathway_diagnostics.py      (自带 runner,无需 pytest)
  pytest tests/test_tsk_hfx_pathway_diagnostics.py      (安装 pytest 后同样可用)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_PLOT_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_PLOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PLOT_SCRIPTS))

import tsk_hfx_pathway_diagnostics as tp  # noqa: E402


# =====================================================================
# 合成数据构造
# =====================================================================

def _synthetic_case(values_w, values_s, values_nr, variable: str = "hfx",
                    ny: int = 1, nx: int = 2) -> xr.Dataset:
    """最小案例:指定变量的三方场,其余变量全部有限且一致(掩膜只由目标变量决定)。"""
    idx = np.arange(ny * nx, dtype=float).reshape(ny, nx)

    def same(value):
        return np.full((ny, nx), float(value))

    tsk_w = 300.0 + 0.01 * idx
    tsk_s = tsk_w - 0.05
    tsk_nr = 300.0 + 0.02 * idx
    t_air_w = 299.0 + 0.01 * idx
    t_air_s = t_air_w + 0.02
    t_air_nr = 299.0 + 0.02 * idx
    theta_air_w = 300.0 + 0.01 * idx
    theta_sfc_w = theta_air_w + 2.0  # dtheta_w = 2.0 处处一致
    theta_air_s = theta_air_w + 0.01
    theta_sfc_s = theta_air_s + 2.0
    theta_air_nr = 300.0 + 0.02 * idx
    theta_sfc_nr = theta_air_nr + 2.0

    fields = {
        "tsk_w": tsk_w, "tsk_s": tsk_s, "tsk_nr": tsk_nr,
        "t_air_w": t_air_w, "t_air_s": t_air_s, "t_air_nr": t_air_nr,
        "d_w": tsk_w - t_air_w, "d_s": tsk_s - t_air_s, "d_nr": tsk_nr - t_air_nr,
        "theta_air_w": theta_air_w, "theta_air_s": theta_air_s, "theta_air_nr": theta_air_nr,
        "theta_sfc_w": theta_sfc_w, "theta_sfc_s": theta_sfc_s, "theta_sfc_nr": theta_sfc_nr,
        "dtheta_w": theta_sfc_w - theta_air_w,
        "dtheta_s": theta_sfc_s - theta_air_s,
        "dtheta_nr": theta_sfc_nr - theta_air_nr,
        "hfx_w": same(100.0), "hfx_s": same(102.0), "hfx_nr": same(98.0),
        "landmask_w": np.zeros((ny, nx)), "landmask_s": np.zeros((ny, nx)),
        "landmask_nr": np.zeros((ny, nx)),
    }
    fields[f"{variable}_w"] = np.asarray(values_w, dtype=float).reshape(ny, nx)
    fields[f"{variable}_s"] = np.asarray(values_s, dtype=float).reshape(ny, nx)
    fields[f"{variable}_nr"] = np.asarray(values_nr, dtype=float).reshape(ny, nx)

    data = {name: (("y", "x"), values) for name, values in fields.items()}
    data["radius_km"] = (("y", "x"), np.full((ny, nx), 20.0))
    data["mask_radius"] = (("y", "x"), np.ones((ny, nx), dtype=bool))
    data["mask_ocean_common"] = (("y", "x"), np.ones((ny, nx), dtype=bool))
    return xr.Dataset(data, coords={"y": np.arange(ny), "x": np.arange(nx)})


def _synthetic_grid(n: int, half_extent_km: float, step_km: float,
                    clat: float = 18.0, clon: float = 140.0):
    """局地等距投影 <-> 经纬度 的合成网格(与 LocalMapper 同投影)。"""
    axis = (np.arange(n) - (n - 1) / 2.0) * step_km
    gx, gy = np.meshgrid(axis, axis)
    radius_km = tp.CONFIG["earth_radius_km"]
    lat = clat + np.rad2deg(gy / radius_km)
    lon = clon + np.rad2deg(gx / (radius_km * np.cos(np.deg2rad(clat))))
    return lat, lon, gx, gy


def _gaussian(gx, gy, sigma_km=30.0):
    return np.exp(-(gx * gx + gy * gy) / (2.0 * sigma_km * sigma_km))


# =====================================================================
# 计划要求的检查
# =====================================================================

def test_budget_sign_and_closure():
    """e_W=[1,1]、δH=[−1,−2] 应得 C=−3、S=2.5、ΔMSE=−0.5(计划数值)。"""
    budget = tp.member_budget(np.array([1.0, 1.0]), np.array([-1.0, -2.0]))
    assert budget["status"] == "ok"
    assert abs(budget["cross"] - (-3.0)) < 1e-12
    assert abs(budget["square"] - 2.5) < 1e-12
    assert abs(budget["delta_mse"] - (-0.5)) < 1e-12
    assert abs(budget["closure"]) < 1e-12
    assert budget["closure_maxabs"] < 1e-12
    # 逐点恒等式对 NaN 掩膜同样成立
    e = np.array([1.0, np.nan, 2.0])
    d = np.array([-1.0, 1.0, -2.0])
    budget2 = tp.member_budget(e, d)
    assert budget2["n"] == 2
    assert abs(budget2["closure"]) < 1e-12


def test_memberwise_average_order():
    """两个成员 e_W=[1]、[−1],δH=[1]、[−1]:成员交叉项平均 = 2,先平均场再算得 0。"""
    c1 = tp.member_budget(np.array([1.0]), np.array([1.0]))["cross"]
    c2 = tp.member_budget(np.array([-1.0]), np.array([-1.0]))["cross"]
    memberwise_mean = 0.5 * (c1 + c2)
    assert abs(memberwise_mean - 2.0) < 1e-12
    # 错误顺序:先平均误差场与响应场,再算交叉项
    e_all = np.array([1.0, -1.0])
    d_all = np.array([1.0, -1.0])
    wrong = 2.0 * float(np.mean(e_all)) * float(np.mean(d_all))
    assert abs(wrong) < 1e-12
    assert abs(memberwise_mean - wrong) > 1.0


def test_common_mask_excludes_fill_and_nan():
    """任一侧 NaN 或 |x|≥1e30 剔除;真 0 保留;原掩膜与海洋掩膜分开定义。"""
    a = np.array([0.0, 1.0, np.nan, 5.0, 3.0])
    b = np.array([0.0, np.nan, 2.0, 1e30, -3.0])
    mask = tp.common_finite_mask(a, b)
    assert mask.tolist() == [True, False, False, False, True]  # 真 0 保留
    # 原掩膜(仅变量三方有限)与海洋公共掩膜是两层独立约束
    landmask = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    ocean = tp.common_finite_mask(a, b, landmask) & (landmask < 0.5)
    assert ocean.tolist() == [True, False, False, False, True]
    assert np.count_nonzero(mask & (landmask >= 0.5)) == 0  # 陆地只由 ocean 掩膜剔除


def test_reconstruction_near_zero_gate():
    """响应 RMS=0.1 (< 0.5 W m-2) 时不得生成比率或 pass;保留响应绝对值并标 near_zero。"""
    quality = tp.reconstruction_quality(
        0.05, 0.05, 0.05, response_rms=0.1,
        strict=tp.CONFIG["recon_ratio_strict"],
        limited=tp.CONFIG["recon_ratio_limited"],
        floor=tp.CONFIG["recon_response_floor_wm2"],
    )
    assert quality["status"] == "near_zero"
    assert np.isnan(quality["ratio_w"]) and np.isnan(quality["ratio_s"])
    assert np.isnan(quality["ratio_diff"])
    assert quality["response_rms"] == 0.1  # 绝对值保留
    sensitivity = tp.quality_sensitivity(quality["ratio_w"], quality["ratio_s"],
                                         quality["ratio_diff"],
                                         thresholds=tp.CONFIG["recon_ratio_sensitivity"])
    assert all(value == "not_evaluable" for value in sensitivity.values())
    # 边界:响应 RMS 恰好等于下限时不触发 near_zero(下限为"低于则置 NaN")
    edge = tp.reconstruction_quality(0.1, 0.1, 0.1, response_rms=0.5,
                                     strict=0.2, limited=0.5, floor=0.5)
    assert edge["status"] == "strict"


def test_reconstruction_quality_states():
    """三项比值均 0.1 → strict;存在 0.3 且均 ≤0.5 → limited;存在 0.6 → fail。"""
    strict = tp.reconstruction_quality(0.1, 0.1, 0.1, response_rms=1.0,
                                       strict=0.2, limited=0.5, floor=0.5)
    limited = tp.reconstruction_quality(0.3, 0.1, 0.1, response_rms=1.0,
                                        strict=0.2, limited=0.5, floor=0.5)
    fail = tp.reconstruction_quality(0.6, 0.1, 0.1, response_rms=1.0,
                                     strict=0.2, limited=0.5, floor=0.5)
    assert strict["status"] == "strict"
    assert limited["status"] == "limited"
    assert fail["status"] == "fail"
    # 敏感性表:固定比值,对 0.1/0.2/0.5 阈值逐一给 pass/fail
    sens = tp.quality_sensitivity(0.3, 0.1, 0.1, thresholds=(0.1, 0.2, 0.5))
    assert sens == {"pass_at_0.1": "fail", "pass_at_0.2": "fail", "pass_at_0.5": "pass"}


def test_storm_translation_recovery():
    """1.5 km 网格上整 20 格(30 km)平移,正确居中后重合有效点误差 ≤1e-8;
    非整格平移报告插值误差。"""
    n = 201
    step = 1.5
    clat, clon = 18.0, 140.0
    lat, lon, gx, gy = _synthetic_grid(n, 150.0, step, clat, clon)
    v1 = _gaussian(gx, gy)
    v2 = _gaussian(gx - 30.0, gy)  # 台风东移 30 km = 20 格

    axis_t, gx_t, gy_t = tp.LocalMapper(lat, lon, 0.0, 0.0).grid(150.0, step)
    mapped1 = tp.LocalMapper(lat, lon, clat, clon).map(v1, gx_t, gy_t)
    dlon = np.rad2deg(30.0 / (tp.CONFIG["earth_radius_km"] * np.cos(np.deg2rad(clat))))
    mapped2 = tp.LocalMapper(lat, lon, clat, clon + dlon).map(v2, gx_t, gy_t)
    common = np.isfinite(mapped1) & np.isfinite(mapped2)
    assert np.any(common)
    max_error = float(np.max(np.abs(mapped1[common] - mapped2[common])))
    assert max_error <= 1.0e-8, f"整格平移恢复误差 {max_error:.3g} 超过 1e-8"

    # 非整格平移:误差必须为正(存在插值误差)且对光滑场保持小量;数值打印留档
    shift = 30.7
    v3 = _gaussian(gx - shift, gy)
    dlon3 = np.rad2deg(shift / (tp.CONFIG["earth_radius_km"] * np.cos(np.deg2rad(clat))))
    mapped3 = tp.LocalMapper(lat, lon, clat, clon + dlon3).map(v3, gx_t, gy_t)
    common3 = np.isfinite(mapped1) & np.isfinite(mapped3)
    interp_error = float(np.max(np.abs(mapped1[common3] - mapped3[common3])))
    print(f"[info] 非整格平移 {shift} km 的最大插值误差 = {interp_error:.3e}")
    assert 0.0 < interp_error < 0.1


def test_zero_shift_and_no_extrapolation():
    """零平移两种评分完全相同;边缘无源数据的目标格点必须 NaN 且覆盖数减少。"""
    clat, clon = 18.0, 140.0
    lat_s, lon_s, gx_s, gy_s = _synthetic_grid(161, 120.0, 1.5, clat, clon)  # 源 ±120 km
    lat_t, lon_t, gx_t, gy_t = _synthetic_grid(201, 150.0, 1.5, clat, clon)  # 目标 ±150 km
    v1 = _gaussian(gx_s, gy_s)

    mapper_a = tp.LocalMapper(lat_s, lon_s, clat, clon)
    mapper_b = tp.LocalMapper(lat_s, lon_s, clat, clon)
    mapped_a = mapper_a.map(v1, gx_t, gy_t)
    mapped_b = mapper_b.map(v1, gx_t, gy_t)
    assert np.array_equal(mapped_a, mapped_b, equal_nan=True)  # 零平移:评分输入逐位相同
    finite_a = np.isfinite(mapped_a)
    analytic = _gaussian(gx_t, gy_t)
    rmse_a = float(np.sqrt(np.mean((mapped_a[finite_a] - analytic[finite_a]) ** 2)))
    rmse_b = float(np.sqrt(np.mean((mapped_b[finite_a] - analytic[finite_a]) ** 2)))
    assert rmse_a == rmse_b

    # 不外推:源凸包(±120 km)之外必须为 NaN,覆盖数减少
    corner_out = (np.abs(gx_t) > 120.0 + 0.75) | (np.abs(gy_t) > 120.0 + 0.75)
    assert np.all(~finite_a[corner_out]), "凸包外出现非 NaN 值,存在外推"
    assert np.all(finite_a[~corner_out & (np.abs(gx_t) < 118.0) & (np.abs(gy_t) < 118.0)])
    assert int(np.count_nonzero(finite_a)) < gx_t.size


# =====================================================================
# 补充检查
# =====================================================================

def test_theta_surface_and_destaggering():
    """θ_surface = TSK·(p0/PSFC)^(Rd/Cp);U/V 去交错为相邻平均。"""
    assert tp.theta_surface(np.array(300.0), np.array(100000.0)) == 300.0
    expected = 300.0 * (100000.0 / 95000.0) ** (287.0 / 1004.0)
    assert abs(float(tp.theta_surface(np.array(300.0), np.array(95000.0))) - expected) < 1e-12
    u_stag = np.arange(6.0, dtype=float).reshape(1, 6)  # 值 = 下标
    assert tp.destagger_u(u_stag).tolist() == [[0.5, 1.5, 2.5, 3.5, 4.5]]
    v_stag = np.arange(6.0, dtype=float).reshape(6, 1)
    assert tp.destagger_v(v_stag).ravel().tolist() == [0.5, 1.5, 2.5, 3.5, 4.5]


def test_original_mask_memberwise_rmse_order():
    """原掩膜评分 = 逐成员各自有效点 RMSE 后平均,不等于集合平均场的 RMSE。"""
    case_a = _synthetic_case([1.0, np.nan], [0.0, 0.0], [0.0, 0.0])  # rmse_w = 1
    case_b = _synthetic_case([np.nan, 3.0], [0.0, 0.0], [0.0, 0.0])  # rmse_w = 3
    row_a = tp.summarize_case(case_a, "all", "original")
    row_b = tp.summarize_case(case_b, "all", "original")
    assert row_a["n_points"] == 1 and row_b["n_points"] == 1
    member_mean = 0.5 * (row_a["hfx_rmse_w"] + row_b["hfx_rmse_w"])
    assert abs(member_mean - 2.0) < 1e-12
    pooled = float(np.sqrt(np.nanmean((np.array([1.0, 3.0])) ** 2)))  # 集合平均场口径
    assert abs(pooled - np.sqrt(5.0)) < 1e-12
    assert abs(member_mean - pooled) > 0.1  # 两种顺序不可混用


def test_three_term_identity():
    """温差误差恒等式:MSE(e_Dθ) = 〈e_θs²〉+〈e_θa²〉−2〈e_θs e_θa〉(summarize 端到端)。"""
    ny, nx = 3, 4
    idx = np.arange(ny * nx, dtype=float).reshape(ny, nx)
    case = _synthetic_case(100.0 + idx, 100.0 + 2.0 * idx, 98.0 + idx, variable="hfx",
                           ny=ny, nx=nx)
    row = tp.summarize_case(case, "all", "ocean_common")
    total = row["dtheta_rmse_w"] ** 2
    parts = row["dtheta3_surface_w"] + row["dtheta3_air_w"] + row["dtheta3_cross_w"]
    assert abs(total - parts) < 1e-9 * max(1.0, abs(total))
    total_s = row["dtheta_rmse_s"] ** 2
    parts_s = row["dtheta3_surface_s"] + row["dtheta3_air_s"] + row["dtheta3_cross_s"]
    assert abs(total_s - parts_s) < 1e-9 * max(1.0, abs(total_s))
    # 预算闭合残差在浮点意义上为 0
    assert abs(row["hfx_closure_mean"]) < 1e-8
    assert row["hfx_closure_maxabs"] < 1e-8


def test_classify_spatial_counts():
    """四类空间对应计数(默认容差与零容差一致于构造场)。"""
    ny, nx = 1, 4
    idx = np.arange(4, dtype=float).reshape(1, 4)
    case = _synthetic_case(100.0 + idx, 100.0 + 2.0 * idx, 98.0 + idx, variable="hfx",
                           ny=ny, nx=nx)
    # e_H,W = [0, 2, 1, 0],e_H,S = [−2, −1, 3, 0] → ΔSE_H = [4, −3, 8, 0]
    case["hfx_nr"].values[...] = 0.0
    case["hfx_w"].values[...] = np.array([0.0, 2.0, 1.0, 0.0]).reshape(1, 4)
    case["hfx_s"].values[...] = np.array([-2.0, -1.0, 3.0, 0.0]).reshape(1, 4)
    # TSK 侧三方完全一致 → ΔSE_TSK = 0(全部 sst_unchanged)
    case["tsk_s"].values[...] = case["tsk_w"].values[...]
    counts = tp.classify_spatial(case, "all", "ocean_common",
                                 tp.CONFIG["dse_tsk_tol_k2"], tp.CONFIG["dse_hfx_tol_wm22"])
    # TSK 侧:三方场一致 → ΔSE_TSK = 0,全部 sst_unchanged
    assert counts["n_sst_unchanged"] == 4
    assert counts["n_sst_improve"] == 0 and counts["n_sst_worsen"] == 0
    # HFX 侧:ΔSE = [4, −3, 8, 0] → improve=1, worsen=2, unchanged=1
    assert counts["n_hfx_improve"] == 1
    assert counts["n_hfx_worsen"] == 2
    assert counts["n_hfx_unchanged"] == 1
    assert counts["n_common"] == 4
    # 零容差与默认容差在此构造下同值(ΔSE 远离容差量级)
    zero = tp.classify_spatial(case, "all", "ocean_common", 0.0, 0.0)
    assert zero["n_hfx_improve"] == counts["n_hfx_improve"]
    assert zero["n_hfx_worsen"] == counts["n_hfx_worsen"]


def test_reconstruction_floor_consistency():
    """本封装的 HFX 与 reconstruct_ocean_fluxes 逐点一致;−250 下限被显式计数。"""
    from omtmp_flux_decomposition import reconstruct_ocean_fluxes

    ny, nx = 3, 4
    idx = np.arange(ny * nx, dtype=float).reshape(ny, nx)
    ta = 300.0 + 0.05 * idx
    ts = 302.0 + 0.05 * idx
    qa = 0.015 + 1.0e-4 * idx
    pa = 99900.0 + 2.0 * idx
    ps = 100100.0 + 2.0 * idx
    z = 50.0 + 0.1 * idx
    u = 10.0 + 0.1 * idx
    v = 2.0 - 0.05 * idx
    ustar = 0.3 + 0.01 * idx
    z0m = 2.0e-4 + 1.0e-6 * idx
    mask = np.ones((ny, nx), dtype=bool)
    inputs = {
        "air_temperature_k": ta, "vapor_mixing_ratio": qa, "air_pressure_pa": pa,
        "surface_pressure_pa": ps, "height_agl_m": z, "u_ms": u, "v_ms": v,
        "initial_friction_velocity_ms": ustar, "initial_momentum_roughness_m": z0m,
    }
    out = tp._sfclay_hfx(inputs, ts, mask, tp.CONFIG)
    reference = reconstruct_ocean_fluxes(
        air_temperature_k=ta.reshape(1, -1), surface_temperature_k=ts.reshape(1, -1),
        vapor_mixing_ratio=qa.reshape(1, -1), air_pressure_pa=pa.reshape(1, -1),
        surface_pressure_pa=ps.reshape(1, -1), height_agl_m=z.reshape(1, -1),
        u_ms=u.reshape(1, -1), v_ms=v.reshape(1, -1),
        initial_friction_velocity_ms=ustar.reshape(1, -1),
        initial_momentum_roughness_m=z0m.reshape(1, -1),
        dx_m=tp.CONFIG["dx_m"], isftcflx=tp.CONFIG["isftcflx"],
    )
    assert np.allclose(out["hfx"], reference["hfx"].reshape(ny, nx),
                       rtol=0, atol=1e-9)
    assert not np.any(out["floor_bind"])  # 常规海温下限不触发

    # 下限触发:海面显著偏冷使 HFX_raw < −250 W m-2(纯数学检查,非真实海况)
    ts_cold = ta - 30.0
    inputs_cold = {**inputs, "u_ms": 25.0 - 0.05 * idx,
                   "initial_friction_velocity_ms": 0.5 + 0.01 * idx}
    out_cold = tp._sfclay_hfx(inputs_cold, ts_cold, mask, tp.CONFIG)
    assert np.any(out_cold["floor_bind"])
    bound = out_cold["floor_bind"]
    assert np.all(out_cold["hfx"][bound] == tp.CONFIG["hfx_floor_wm2"])
    assert np.all(out_cold["hfx_raw"][bound] < tp.CONFIG["hfx_floor_wm2"])
    reference_cold = reconstruct_ocean_fluxes(
        air_temperature_k=ta.reshape(1, -1), surface_temperature_k=ts_cold.reshape(1, -1),
        vapor_mixing_ratio=qa.reshape(1, -1), air_pressure_pa=pa.reshape(1, -1),
        surface_pressure_pa=ps.reshape(1, -1), height_agl_m=z.reshape(1, -1),
        u_ms=inputs_cold["u_ms"].reshape(1, -1), v_ms=v.reshape(1, -1),
        initial_friction_velocity_ms=ustar.reshape(1, -1),
        initial_momentum_roughness_m=z0m.reshape(1, -1),
        dx_m=tp.CONFIG["dx_m"], isftcflx=tp.CONFIG["isftcflx"],
    )
    assert np.allclose(out_cold["hfx"], reference_cold["hfx"].reshape(ny, nx),
                       rtol=0, atol=1e-9)


def test_config_snapshot_serializable():
    snapshot = tp.config_snapshot(tp.CONFIG)
    text = json.dumps(snapshot, default=str)
    assert "6mem_oceanAssim1Run1" in text
    assert json.loads(text)["radius_km"] == 150.0


def test_validate_time_char_array():
    """WRF Times 为 (Time, DateStrLen) 二维字符数组时,时间校验应正确解码。"""
    class _FakeDataset:
        variables = {"Times": np.array([[b"2", b"0", b"1", b"8", b"-", b"0", b"9", b"-",
                                         b"1", b"0", b"_", b"0", b"0", b":", b"3", b"0",
                                         b":", b"0", b"0"]], dtype="S1")}
    stamp = "2018-09-10_00:30:00"
    tp._validate_time(_FakeDataset(), Path("fake"), stamp)  # 不抛错即通过
    try:
        tp._validate_time(_FakeDataset(), Path("fake"), "2018-09-10_01:00:00")
    except ValueError as error:
        assert "file time mismatch" in str(error)
    else:
        raise AssertionError("mismatched stamp should raise ValueError")


# =====================================================================
# 自带 runner(无 pytest 环境)
# =====================================================================

def _run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001 - runner 需要收集全部失败
            failures.append((name, error))
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"\ntotal={len(tests)}, failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
