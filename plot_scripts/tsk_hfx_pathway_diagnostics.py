"""TSK-HFX 误差传递诊断(Task 2-4 共用库;本地编写,服务器执行)。

依据 docs/superpowers/plans/2026-10-31-tsk-hfx-error-pathway.md:

  - extract_case            NR 精确时刻文件 -> 经纬度 Delaunay 线性插值到成员
                            d02 网格子窗;三角网按案例构建一次、全变量复用,
                            凸包外一律 NaN(不外推)。
  - derive_error_fields     逐点 δTSK、e_*、ΔSE_*、交叉项 C=2 e_W δH、平方项
                            S=δH²;先成员内逐点计算,才允许跨成员平均。
  - summarize_case          成员级指标:等格点权重;原脚本掩膜(original)与
                            三方海洋公共掩膜(ocean_common)分开保存,不混用。
  - evaluate_reconstruction 固定其余输入的 TSK 替换。物理实现复用
                            wrf41_sfclayrev + omtmp_flux_decomposition,不修改;
                            −250 W m-2 下限做显式计数,不静默启用/禁用。
  - compare_storm_relative  台风居中结构比较:局地 1.5 km 网格双映射
                            (各自中心 / NR 中心),匹配公共掩膜,无外推。

运行方式与产物清单见 analysis/tsk_hfx_pathway_20261031/README.md。
综合机制解释(final_report.md)不由本模块产出。
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:  # 与 omtmp_flux_decomposition.py 相同的双路径导入模式
    from omtmp_flux_decomposition import reconstruct_ocean_fluxes  # noqa: F401
    from wrf41_sfclayrev import (
        EP1,
        KARMAN,
        P0_PA,
        R_OVER_CP,
        SfclayOptions,
        SurfaceState,
        XLV_J_KG,
        _integrated_psi,
        revised_mm5_ocean_flux,
    )
except ImportError:  # pragma: no cover - 包路径导入回退
    from plot_scripts.omtmp_flux_decomposition import reconstruct_ocean_fluxes  # type: ignore
    from plot_scripts.wrf41_sfclayrev import (  # type: ignore
        EP1,
        KARMAN,
        P0_PA,
        R_OVER_CP,
        SfclayOptions,
        SurfaceState,
        XLV_J_KG,
        _integrated_psi,
        revised_mm5_ocean_flux,
    )

# =====================================================================
# User configuration(所有可编辑参数集中在此;无命令行参数)
# =====================================================================

CONFIG: dict = {
    # "server": 读 wrfout 的正式运行; "local_smoke": 本地只跑纯数学检查,不读任何数据文件
    "run_environment": "server",
    "paths": {
        "server": {
            "member_base": "/scratch/lililei1/kcfu/tc_mangkhut/cycle_test",
            "nr_base": "/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain",
            "baseline_tsk_csv": "/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/figs/tsk_rmse_timeseries.csv",
            "baseline_hfx_csv": "/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/figs/hfx_rmse_timeseries.csv",
            "output_root": "/share/home/lililei1/kcfu/tc_mangkhut/analysis/tsk_hfx_pathway_20261031",
        },
        "local_smoke": {
            "member_base": "",
            "nr_base": "",
            "baseline_tsk_csv": "/Volumes/ssd_storage/nju_works/tc_mangkhut/analysis/tsk_hfx_20261031/tsk_rmse_timeseries.csv",
            "baseline_hfx_csv": "/Volumes/ssd_storage/nju_works/tc_mangkhut/analysis/tsk_hfx_20261031/hfx_rmse_timeseries.csv",
            "output_root": "/Volumes/ssd_storage/nju_works/tc_mangkhut/analysis/tsk_hfx_pathway_20261031",
        },
    },
    "server_python": "/share/home/lililei1/kcfu/anaconda/envs/wrf/bin/python",
    # 试验与方法(同一方法内按成员配对,不合并 EAKF/QCF_RHF)
    "weak_experiment": "6mem_oceanAssim0Run1",
    "strong_experiment": "6mem_oceanAssim1Run1",
    "methods": ("EAKF", "QCF_RHF"),
    "members": ("006", "015", "029", "037", "043", "044"),
    "domain": "d02",
    "start_time": "2018-09-10_00:00:00",
    "main_hours": (0.5, 1.0, 1.5, 2.0),
    "contrast_hours": (3.0, 6.0),
    "initial_hours": (0.0,),
    # 区域与配准
    "radius_km": 150.0,
    "inner_radius_km": 75.0,
    "bbox_pad": 3,
    "nr_margin_deg": 0.06,
    "grid_tol": 1.0e-6,
    "fill_threshold": 1.0e30,
    "earth_radius_km": 6371.0,
    # 分类容差(作用于逐点 ΔSE;不是温度/通量增量阈值)
    "dse_tsk_tol_k2": 1.0e-4,
    "dse_hfx_tol_wm22": 1.0,
    # 重建(口径沿 verify_03 / omtmp_pathway_extract;isftcflx 与 dx 待服务器 namelist 核实)
    "dx_m": 1500.0,
    "isftcflx": 0,
    "ustar_floor": 1.0e-4,
    "z0m_floor": 1.27e-7,
    "hfx_floor_wm2": -250.0,
    "recon_ratio_strict": 0.2,
    "recon_ratio_limited": 0.5,
    "recon_ratio_sensitivity": (0.1, 0.2, 0.5),
    "recon_response_floor_wm2": 0.5,
    # "diagnostic" 沿用 omtmp_pathway_extract 的诊断式 z0m;"znt_variable" 需文件含 ZNT
    "recon_z0m_source": "diagnostic",
    # 台风居中
    "storm_grid_extent_km": 150.0,
    "storm_grid_step_km": 1.5,
    "storm_radial_bin_km": 10.0,
    "center_ambiguity_pressure_pa": 1.0,
    "center_ambiguity_sep_km": 50.0,
    "center_perturb_enabled": True,
    # 输出
    "figure_dpi": 200,
    "figure_formats": ("png", "pdf"),
}

REGIONS: tuple[str, ...] = ("all", "inner75", "outer75_150")
MASK_KINDS: tuple[str, ...] = ("original", "ocean_common")

MEMBER_METRIC_COLUMNS = [
    "method", "member", "lead_hour", "valid_time", "region", "mask_kind",
    "n_points", "region_points", "coverage_fraction", "ocean_fraction", "status",
    "tsk_rmse_w", "tsk_rmse_s", "tsk_bias_w", "tsk_bias_s",
    "hfx_rmse_w", "hfx_rmse_s", "hfx_bias_w", "hfx_bias_s",
    "tair_rmse_w", "tair_rmse_s", "d_rmse_w", "d_rmse_s",
    "dtheta_rmse_w", "dtheta_rmse_s",
    "hfx_delta_mse", "hfx_cross", "hfx_square", "hfx_closure_mean", "hfx_closure_maxabs",
    "tsk_delta_mse", "d_delta_mse", "dtheta_delta_mse",
    "dtheta3_surface_w", "dtheta3_air_w", "dtheta3_cross_w",
    "dtheta3_surface_s", "dtheta3_air_s", "dtheta3_cross_s",
    "d3_surface_w", "d3_air_w", "d3_cross_w",
    "d3_surface_s", "d3_air_s", "d3_cross_s",
    "delta_tsk_mean", "delta_hfx_mean",
]


# =====================================================================
# 常量与基础工具
# =====================================================================

def paths(config: dict) -> dict:
    return config["paths"][config["run_environment"]]


def stamp_from_lead(lead_hour: float) -> str:
    """起报时刻 + 预报时效 -> wrfout 文件名中的时间串。"""
    start = pd.Timestamp(CONFIG["start_time"].replace("_", " "))
    stamp = start + pd.Timedelta(minutes=int(round(lead_hour * 60.0)))
    return stamp.strftime("%Y-%m-%d_%H:%M:%S")


def case_id(method: str, member: str, lead_hour: float) -> str:
    return f"{method}_{member}_t{lead_hour:g}"


def to_fill_nan(values) -> np.ndarray:
    """netCDF4 masked array -> float ndarray,掩膜/fill 统一为 NaN。"""
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.ma.asarray(values)), np.nan), dtype=float)


def great_circle_distance_km(lat, lon, center_lat, center_lon, radius_km: float = 6371.0) -> np.ndarray:
    """Haversine 距离(km);与服务器基线脚本同式(R=6371)。"""
    a = np.deg2rad(np.asarray(lat, dtype=float))
    b = np.deg2rad(np.asarray(center_lat, dtype=float))
    dl = np.deg2rad(np.asarray(lon, dtype=float) - np.asarray(center_lon, dtype=float))
    hav = np.sin((a - b) / 2.0) ** 2 + np.cos(a) * np.cos(b) * np.sin(dl / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def common_finite_mask(*arrays, threshold: float | None = None) -> np.ndarray:
    """所有数组均有限且 |x| < threshold 的联合掩膜(NaN 与 ±fill 都剔除,真 0 保留)。"""
    if threshold is None:
        threshold = CONFIG["fill_threshold"]
    result = None
    for array in arrays:
        array = np.asarray(array, dtype=float)
        good = np.isfinite(array) & (np.abs(array) < threshold)
        result = good if result is None else (result & good)
    if result is None:
        raise ValueError("common_finite_mask requires at least one array")
    return result


def bbox_of(mask: np.ndarray, pad: int = 3) -> tuple[slice, slice]:
    """布尔掩膜外扩 pad 圈后的 (jslice, islice);空掩膜报错。"""
    jj, ii = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(jj):
        raise ValueError("bbox_of: empty mask")
    nj, ni = mask.shape
    return (slice(max(0, int(jj.min()) - pad), min(nj, int(jj.max()) + pad + 1)),
            slice(max(0, int(ii.min()) - pad), min(ni, int(ii.max()) + pad + 1)))


def destagger_u(u_stag: np.ndarray) -> np.ndarray:
    """U(x-staggered) 去交错 -> 质量点。"""
    return 0.5 * (u_stag[..., :-1] + u_stag[..., 1:])


def destagger_v(v_stag: np.ndarray) -> np.ndarray:
    """V(y-staggered) 去交错 -> 质量点。"""
    return 0.5 * (v_stag[..., :-1, :] + v_stag[..., 1:, :])


def theta_surface(tsk_k: np.ndarray, psfc_pa: np.ndarray, p0_pa: float = P0_PA) -> np.ndarray:
    """海面位温 θ_surface = TSK·(p0/PSFC)^(Rd/Cp)(Rd=287, cp=1004)。"""
    return np.asarray(tsk_k, dtype=float) * (p0_pa / np.asarray(psfc_pa, dtype=float)) ** R_OVER_CP


def initial_momentum_roughness(ustar, isftcflx: int = 0) -> np.ndarray:
    """诊断式初始动力粗糙度(沿用 omtmp_pathway_extract / verify_common 口径,isftcflx=0)。

    TODO(待核实):需对照试验 namelist 与 wrfout 是否输出 ZNT;见 reconstruction_audit。
    """
    if isftcflx != 0:
        raise NotImplementedError("仅实现 isftcflx=0 的诊断式 z0m")
    ustar = np.maximum(np.asarray(ustar, dtype=float), 1.0e-4)
    return np.minimum(0.0185 * ustar**2 / 9.81 + 0.11 * 1.5e-5 / ustar, 2.85e-3)


def member_budget(e_weak, delta, mask=None) -> dict:
    """单成员精确误差预算:e = X_W − X_NR,δ = X_S − X_W。

    ΔMSE = 〈(e+δ)² − e²〉 ≡ C + S,C = 〈2eδ〉,S = 〈δ²〉。
    只在共同有效点平均(先成员内,后跨成员,顺序由调用方保证)。
    """
    e = np.asarray(e_weak, dtype=float)
    d = np.asarray(delta, dtype=float)
    if mask is None:
        mask = common_finite_mask(e, d)
    mask = np.asarray(mask, dtype=bool) & common_finite_mask(e, d)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return {"n": 0, "status": "empty_mask", "cross": np.nan, "square": np.nan,
                "delta_mse": np.nan, "closure": np.nan, "closure_maxabs": np.nan}
    ev, dv = e[mask], d[mask]
    cross = float(np.mean(2.0 * ev * dv))
    square = float(np.mean(dv * dv))
    delta_mse = float(np.mean((ev + dv) ** 2 - ev ** 2))
    # 逐点恒等式残差(浮点意义上应为 0):ΔSE − (2eδ + δ²)
    closure_field = ((ev + dv) ** 2 - ev ** 2) - (2.0 * ev * dv + dv * dv)
    return {
        "n": n, "status": "ok",
        "cross": cross, "square": square, "delta_mse": delta_mse,
        "closure": float(delta_mse - (cross + square)),
        "closure_maxabs": float(np.max(np.abs(closure_field))),
    }


def three_term_decomposition(e_s, e_a, mask=None) -> dict:
    """温差误差恒等式:MSE(e_s−e_a) = 〈e_s²〉 + 〈e_a²〉 − 2〈e_s e_a〉。

    乘积项不是去均值协方差;输出仅供误差预算,不得称"解释方差"。
    """
    es = np.asarray(e_s, dtype=float)
    ea = np.asarray(e_a, dtype=float)
    if mask is None:
        mask = common_finite_mask(es, ea)
    mask = np.asarray(mask, dtype=bool) & common_finite_mask(es, ea)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return {"n": 0, "status": "empty_mask", "surface": np.nan, "air": np.nan,
                "cross": np.nan, "total": np.nan, "closure": np.nan}
    sv, av = es[mask], ea[mask]
    surface = float(np.mean(sv * sv))
    air = float(np.mean(av * av))
    cross = float(np.mean(-2.0 * sv * av))
    total = float(np.mean((sv - av) ** 2))
    return {"n": n, "status": "ok", "surface": surface, "air": air,
            "cross": cross, "total": total, "closure": float(total - (surface + air + cross))}


# =====================================================================
# wrfout 读取与配准(服务器专用;本地 smoke 模式不调用)
# =====================================================================

def member_wrfout_path(config: dict, experiment: str, method: str, member: str, stamp: str) -> Path:
    return (Path(paths(config)["member_base"]) / experiment / method / member
            / f"wrfout_{config['domain']}_{stamp}")


def nr_wrfout_path(config: dict, stamp: str) -> Path:
    return Path(paths(config)["nr_base"]) / f"wrfout_{config['domain']}_{stamp}"


def _validate_time(dataset, path: Path, stamp: str) -> None:
    """校验 wrfout 文件时间;兼容标量 bytes 与 (DateStrLen,) 二维字符数组两种 Times 布局。"""
    times = dataset.variables["Times"][:]
    first = np.asarray(times[0])
    if first.ndim == 0:
        item = first.item()
        text = item.decode() if isinstance(item, (bytes, np.bytes_)) else str(item)
    else:
        text = "".join(
            c.decode() if isinstance(c, bytes) else str(c) for c in first.ravel().tolist()
        ).strip()
    if text != stamp:
        raise ValueError(f"file time mismatch: {path} has {text!r}, expected {stamp!r}")


def _global_attrs(dataset) -> dict:
    keys = ("DX", "DY", "VERSION", "TITLE", "SF_SFCLAY_PHYSICS", "SF_SURFACE_PHYSICS",
            "BL_PBL_PHYSICS", "SF_OCEAN_PHYSICS", "CEN_LAT", "CEN_LON")
    out = {}
    for key in keys:
        if key in dataset.ncattrs():
            value = getattr(dataset, key)
            out[key] = value.decode() if isinstance(value, bytes) else value
    return out


def find_center(psfc: np.ndarray, lat: np.ndarray, lon: np.ndarray, config: dict) -> dict:
    """全域最低 PSFC 格点为主中心;记录边界/多低压中心歧义标记,不静默改中心。"""
    ps = to_fill_nan(psfc)
    j, i = np.unravel_index(int(np.nanargmin(ps)), ps.shape)
    pmin = float(ps[j, i])
    nj, ni = ps.shape
    near = common_finite_mask(ps) & (ps <= pmin + config["center_ambiguity_pressure_pa"])
    dist = great_circle_distance_km(lat, lon, lat[j, i], lon[j, i], config["earth_radius_km"])
    ambiguous = int(np.count_nonzero(near & (dist >= config["center_ambiguity_sep_km"])))
    return {
        "j": int(j), "i": int(i),
        "lat": float(lat[j, i]), "lon": float(lon[j, i]), "pressure_pa": pmin,
        "boundary_flag": bool(j in (0, nj - 1) or i in (0, ni - 1)),
        "ambiguous_flag": ambiguous > 0,
        "n_near_min_far_points": ambiguous,
    }


class NrGeometry:
    """NR 子窗几何:三角网构建一次,全变量复用;凸包外 NaN(不外推)。"""

    def __init__(self, nr_lat: np.ndarray, nr_lon: np.ndarray):
        from scipy.spatial import Delaunay

        lon = np.asarray(nr_lon, dtype=float)
        lat = np.asarray(nr_lat, dtype=float)
        points = np.column_stack([lon.ravel(), lat.ravel()])
        good = np.isfinite(points).all(axis=1)
        self._flat_index = np.nonzero(good)[0]
        self.points = points[good]
        self.tri = Delaunay(self.points)

    def interp(self, values: np.ndarray, query_lon: np.ndarray, query_lat: np.ndarray) -> np.ndarray:
        from scipy.interpolate import LinearNDInterpolator

        flat = np.asarray(values, dtype=float).ravel()[self._flat_index]
        if not np.isfinite(flat).any():
            return np.full(np.shape(query_lon), np.nan)
        # LinearNDInterpolator 对 NaN 顶点会传播 NaN,与"该 NR 点无效"语义一致,保留。
        interpolator = LinearNDInterpolator(self.tri, flat, fill_value=np.nan)
        queries = np.column_stack([np.asarray(query_lon, dtype=float).ravel(),
                                   np.asarray(query_lat, dtype=float).ravel()])
        return np.asarray(interpolator(queries), dtype=float).reshape(np.shape(query_lon))


def _read_member_state(dataset, ys: slice, xs: slice, ys_v: slice, xs_u: slice) -> dict:
    """成员 wrfout 子窗原始场(含交错网格的正确切片)。"""

    def g2(name):
        return to_fill_nan(dataset.variables[name][0, ys, xs])

    def g3(name):
        return to_fill_nan(dataset.variables[name][0, 0, ys, xs])

    def gph(name):
        return to_fill_nan(dataset.variables[name][0, 0:2, ys, xs])

    return {
        "TSK": g2("TSK"), "HFX": g2("HFX"), "LH": g2("LH"), "PSFC": g2("PSFC"),
        "UST": g2("UST"), "T2": g2("T2"), "U10": g2("U10"), "V10": g2("V10"),
        "LANDMASK": g2("LANDMASK"), "HGT": g2("HGT"),
        "T": g3("T"), "P": g3("P"), "PB": g3("PB"), "QVAPOR": g3("QVAPOR"),
        "PH": gph("PH"), "PHB": gph("PHB"),
        "U": to_fill_nan(dataset.variables["U"][0, 0, ys, xs_u]),
        "V": to_fill_nan(dataset.variables["V"][0, 0, ys_v, xs]),
    }


def derive_party_fields(raw: dict, config: dict) -> dict:
    """最低层诊断量(口径 = verify_common.derive_lowest_level_inputs,先原生计算)。"""
    theta_air = raw["T"] + 300.0
    p_air = raw["P"] + raw["PB"]
    t_air = theta_air * (p_air / P0_PA) ** R_OVER_CP
    theta_sfc = theta_surface(raw["TSK"], raw["PSFC"])
    geopotential = raw["PH"] + raw["PHB"]
    z_agl = 0.5 * (geopotential[0] + geopotential[1]) / 9.81 - raw["HGT"]
    return {
        "theta_air": theta_air, "p_air": p_air, "t_air": t_air,
        "theta_sfc": theta_sfc, "z_agl": z_agl,
        "d": raw["TSK"] - t_air, "dtheta": theta_sfc - theta_air,
        "u_level": destagger_u(raw["U"]), "v_level": destagger_v(raw["V"]),
        "z0m0": initial_momentum_roughness(raw["UST"], config["isftcflx"]),
    }


def extract_case(config: dict, method: str, member: str, lead_hour: float) -> xr.Dataset:
    """单案例提取:共同网格(成员子窗)、三方状态、中心、掩膜、来源属性。

    NR 用精确时刻文件;W/S 网格逐点一致性校验;掩膜只定义,评分在 summarize。
    """
    from netCDF4 import Dataset

    stamp = stamp_from_lead(lead_hour)
    nr_file = nr_wrfout_path(config, stamp)
    weak_file = member_wrfout_path(config, config["weak_experiment"], method, member, stamp)
    strong_file = member_wrfout_path(config, config["strong_experiment"], method, member, stamp)
    for path in (nr_file, weak_file, strong_file):
        if not path.exists():
            raise FileNotFoundError(f"missing wrfout: {path}")

    with Dataset(nr_file, "r") as n:
        _validate_time(n, nr_file, stamp)
        nr_meta = _global_attrs(n)
        psfc_nr_full = to_fill_nan(n.variables["PSFC"][0])
        nr_lat_full = to_fill_nan(n.variables["XLAT"][0])
        nr_lon_full = to_fill_nan(n.variables["XLONG"][0])
    nr_center = find_center(psfc_nr_full, nr_lat_full, nr_lon_full, config)

    member_raw = {}
    member_meta = {}
    centers = {}
    for tag, path in (("w", weak_file), ("s", strong_file)):
        with Dataset(path, "r") as d:
            _validate_time(d, path, stamp)
            member_meta[tag] = _global_attrs(d)
            mlat = to_fill_nan(d.variables["XLAT"][0])
            mlon = to_fill_nan(d.variables["XLONG"][0])
            psfc_full = to_fill_nan(d.variables["PSFC"][0])
        member_raw[tag] = {"lat": mlat, "lon": mlon, "psfc_full": psfc_full}
        centers[tag] = find_center(psfc_full, mlat, mlon, config)

    lat_w = member_raw["w"]["lat"]
    lon_w = member_raw["w"]["lon"]
    for tag in ("s",):
        if not (np.allclose(member_raw[tag]["lat"], lat_w, atol=config["grid_tol"], rtol=0.0)
                and np.allclose(member_raw[tag]["lon"], lon_w, atol=config["grid_tol"], rtol=0.0)):
            raise ValueError(f"member grid mismatch between weak and strong ({method}/{member})")

    dist = great_circle_distance_km(lat_w, lon_w, nr_center["lat"], nr_center["lon"],
                                    config["earth_radius_km"])
    radius_full = dist <= config["radius_km"]
    ys, xs = bbox_of(radius_full, config["bbox_pad"])
    lat = lat_w[ys, xs]
    lon = lon_w[ys, xs]
    radius = dist[ys, xs]
    mask_radius = radius <= config["radius_km"]

    # NR 子窗:成员子窗经纬度范围外扩 margin,几何只建一次
    margin = config["nr_margin_deg"]
    nr_select = ((nr_lat_full >= lat.min() - margin) & (nr_lat_full <= lat.max() + margin)
                 & (nr_lon_full >= lon.min() - margin) & (nr_lon_full <= lon.max() + margin))
    nys, nxs = bbox_of(nr_select, config["bbox_pad"])
    geometry = NrGeometry(nr_lat_full[nys, nxs], nr_lon_full[nys, nxs])

    with Dataset(nr_file, "r") as n:
        nr_native = {
            "TSK": to_fill_nan(n.variables["TSK"][0, nys, nxs]),
            "HFX": to_fill_nan(n.variables["HFX"][0, nys, nxs]),
            "PSFC": to_fill_nan(n.variables["PSFC"][0, nys, nxs]),
            "T2": to_fill_nan(n.variables["T2"][0, nys, nxs]),
            "U10": to_fill_nan(n.variables["U10"][0, nys, nxs]),
            "V10": to_fill_nan(n.variables["V10"][0, nys, nxs]),
            "LANDMASK": to_fill_nan(n.variables["LANDMASK"][0, nys, nxs]),
        }
        nr_t = to_fill_nan(n.variables["T"][0, 0, nys, nxs])
        nr_p = (to_fill_nan(n.variables["P"][0, 0, nys, nxs])
                + to_fill_nan(n.variables["PB"][0, 0, nys, nxs]))
        nr_native["theta_air"] = nr_t + 300.0
        nr_native["t_air"] = nr_native["theta_air"] * (nr_p / P0_PA) ** R_OVER_CP
        nr_native["theta_sfc"] = theta_surface(nr_native["TSK"], nr_native["PSFC"])
        nr_native["d"] = nr_native["TSK"] - nr_native["t_air"]
        nr_native["dtheta"] = nr_native["theta_sfc"] - nr_native["theta_air"]

    data: dict[str, np.ndarray] = {
        "radius_km": radius,
        "mask_radius": mask_radius,
    }
    for name in ("TSK", "HFX", "PSFC", "T2", "U10", "V10", "LANDMASK",
                 "theta_air", "t_air", "theta_sfc", "d", "dtheta"):
        data[f"{_NR_TAG[name]}"] = geometry.interp(nr_native[name], lon, lat)

    party_inputs: dict[str, dict] = {}
    for tag, path in (("w", weak_file), ("s", strong_file)):
        # U/V 各比质量点多一个边界点;子窗碰到域边界时也必须保留它。
        xs_u = slice(xs.start, xs.stop + 1)
        ys_v = slice(ys.start, ys.stop + 1)
        with Dataset(path, "r") as d:
            raw = _read_member_state(d, ys, xs, ys_v, xs_u)
        derived = derive_party_fields(raw, config)
        for name, values in raw.items():
            if name in ("U", "V"):
                continue
            data[f"{name.lower()}_{tag}"] = values
        for name, values in derived.items():
            data[f"{name}_{tag}"] = values
        party_inputs[tag] = {**raw, **derived}

    landmask_w = data["landmask_w"]
    landmask_s = data["landmask_s"]
    landmask_nr = data["landmask_nr"]
    data["mask_ocean_common"] = (common_finite_mask(landmask_w, landmask_s, landmask_nr)
                                 & (landmask_w < 0.5) & (landmask_s < 0.5) & (landmask_nr < 0.5))

    meta = {
        "case_id": case_id(method, member, lead_hour),
        "method": method, "member": member, "lead_hour": float(lead_hour),
        "valid_time": stamp,
        "nr_file": str(nr_file), "weak_file": str(weak_file), "strong_file": str(strong_file),
        "nr_center": nr_center, "center_w": centers["w"], "center_s": centers["s"],
        "subwindow": {"ys": [ys.start, ys.stop], "xs": [xs.start, xs.stop]},
        "nr_subwindow": {"ys": [nys.start, nys.stop], "xs": [nxs.start, nxs.stop]},
        "nr_attrs": nr_meta, "weak_attrs": member_meta["w"], "strong_attrs": member_meta["s"],
        "region_points": int(np.count_nonzero(mask_radius)),
        "ocean_common_points": int(np.count_nonzero(data["mask_ocean_common"] & mask_radius)),
        "center_error_km": {
            "w": float(great_circle_distance_km(centers["w"]["lat"], centers["w"]["lon"],
                                                nr_center["lat"], nr_center["lon"],
                                                config["earth_radius_km"])),
            "s": float(great_circle_distance_km(centers["s"]["lat"], centers["s"]["lon"],
                                                nr_center["lat"], nr_center["lon"],
                                                config["earth_radius_km"])),
        },
    }

    coords = {"y": np.arange(lat.shape[0]), "x": np.arange(lat.shape[1])}
    interface_fields = {"ph_w", "ph_s", "phb_w", "phb_s"}
    case = xr.Dataset(
        {name: (("bottom_top_stag", "y", "x") if name in interface_fields else ("y", "x"), values)
         for name, values in data.items()},
        coords={**coords, "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )
    case.attrs["meta_json"] = json.dumps(meta, default=str)
    case.attrs["config_json"] = json.dumps(config_snapshot(config), default=str)
    return case


_NR_TAG = {
    "TSK": "tsk_nr", "HFX": "hfx_nr", "PSFC": "psfc_nr", "T2": "t2_nr",
    "U10": "u10_nr", "V10": "v10_nr", "LANDMASK": "landmask_nr",
    "theta_air": "theta_air_nr", "t_air": "t_air_nr", "theta_sfc": "theta_sfc_nr",
    "d": "d_nr", "dtheta": "dtheta_nr",
}


# =====================================================================
# 逐点误差场与成员级指标
# =====================================================================

def field(case: xr.Dataset, name: str) -> np.ndarray:
    return np.asarray(case[name].values, dtype=float)


def derive_error_fields(case: xr.Dataset) -> xr.Dataset:
    """逐点误差/响应/预算场(全部成员内逐点定义,禁止先用集合平均场替代)。"""
    g = lambda name: field(case, name)
    out = {}
    for var in ("tsk", "tair", "hfx", "d", "dtheta"):
        src = "t_air" if var == "tair" else var
        out[f"e_{var}_w"] = g(f"{src}_w") - g(f"{src}_nr")
        out[f"e_{var}_s"] = g(f"{src}_s") - g(f"{src}_nr")
    out["delta_tsk"] = g("tsk_s") - g("tsk_w")
    out["delta_hfx"] = g("hfx_s") - g("hfx_w")
    for var in ("tsk", "tair", "hfx", "d", "dtheta"):
        ew, es = out[f"e_{var}_w"], out[f"e_{var}_s"]
        out[f"se_{var}_w"] = ew * ew
        out[f"se_{var}_s"] = es * es
        out[f"dse_{var}"] = es * es - ew * ew
    out["cross_h"] = 2.0 * out["e_hfx_w"] * out["delta_hfx"]
    out["square_h"] = out["delta_hfx"] ** 2
    out["dse_h_direct"] = out["se_hfx_s"] - out["se_hfx_w"]
    out["dse_h_closure_field"] = out["dse_h_direct"] - (out["cross_h"] + out["square_h"])
    # 温差误差恒等式三项(位温差;实际温差 d 同理,e_d = e_tsk − e_tair 逐点成立)
    for tag, gap in (("dtheta", "dtheta"), ("d", "d")):
        if gap == "dtheta":
            es, ea = (g("theta_sfc_w") - g("theta_sfc_nr"),
                      g("theta_air_w") - g("theta_air_nr"))
            ss, sa = (g("theta_sfc_s") - g("theta_sfc_nr"),
                      g("theta_air_s") - g("theta_air_nr"))
        else:
            es, ea = out["e_tsk_w"], out["e_tair_w"]
            ss, sa = out["e_tsk_s"], out["e_tair_s"]
        out[f"gap3_{gap}_surface_w"] = es * es
        out[f"gap3_{gap}_air_w"] = ea * ea
        out[f"gap3_{gap}_cross_w"] = -2.0 * es * ea
        out[f"gap3_{gap}_surface_s"] = ss * ss
        out[f"gap3_{gap}_air_s"] = sa * sa
        out[f"gap3_{gap}_cross_s"] = -2.0 * ss * sa
    error = xr.Dataset(
        {name: (("y", "x"), values) for name, values in out.items()},
        coords={k: case.coords[k] for k in ("y", "x", "lat", "lon") if k in case.coords},
    )
    error.attrs["meta_json"] = case.attrs.get("meta_json", "")
    return error


def region_mask(case: xr.Dataset, region: str) -> np.ndarray:
    """all: r≤150;inner75: r<75;outer75_150: 75≤r≤150(边界不重复计数)。"""
    radius = field(case, "radius_km")
    inside = radius <= CONFIG["radius_km"]
    if region == "all":
        return inside
    if region == "inner75":
        return inside & (radius < CONFIG["inner_radius_km"])
    if region == "outer75_150":
        return inside & (radius >= CONFIG["inner_radius_km"])
    raise KeyError(f"unknown region: {region}")


_VARIABLE_FIELDS = {
    "tsk": ("tsk_w", "tsk_s", "tsk_nr"),
    "hfx": ("hfx_w", "hfx_s", "hfx_nr"),
    "t_air": ("t_air_w", "t_air_s", "t_air_nr"),
    "d": ("d_w", "d_s", "d_nr"),
    "dtheta": ("dtheta_w", "dtheta_s", "dtheta_nr"),
}


def _variable_mask(case: xr.Dataset, var: str, region: str, mask_kind: str) -> np.ndarray:
    base = region_mask(case, region)
    if mask_kind == "ocean_common":
        base = base & field(case, "mask_ocean_common").astype(bool)
    elif mask_kind != "original":
        raise KeyError(f"unknown mask_kind: {mask_kind}")
    return base & common_finite_mask(*(field(case, name) for name in _VARIABLE_FIELDS[var]))


def summarize_case(case: xr.Dataset, region: str, mask_kind: str) -> dict:
    """成员级指标(等格点权重);预算闭合残差与覆盖率一并返回。"""
    err = derive_error_fields(case)
    fe = lambda name: np.asarray(err[name].values, dtype=float)
    row: dict = {"region": region, "mask_kind": mask_kind, "status": "ok"}
    region_points = int(np.count_nonzero(region_mask(case, region)))
    ocean_in_region = int(np.count_nonzero(region_mask(case, region)
                                           & field(case, "mask_ocean_common").astype(bool)))
    budget_mask = _variable_mask(case, "hfx", region, mask_kind)
    row["region_points"] = region_points
    row["ocean_fraction"] = (ocean_in_region / region_points) if region_points else np.nan
    row["n_points"] = int(np.count_nonzero(budget_mask))
    row["coverage_fraction"] = (row["n_points"] / region_points) if region_points else np.nan
    row["status"] = "ok" if row["n_points"] else "empty_mask"

    for var in ("tsk", "hfx", "t_air", "d", "dtheta"):
        mask = _variable_mask(case, var, region, mask_kind)
        n = int(np.count_nonzero(mask))
        if n == 0:
            for tag in ("w", "s"):
                row[f"{var}_rmse_{tag}"] = np.nan
                row[f"{var}_bias_{tag}"] = np.nan
            continue
        for tag in ("w", "s"):
            e = field(case, f"{var}_{tag}")[mask] - field(case, f"{var}_nr")[mask]
            row[f"{var}_rmse_{tag}"] = float(np.sqrt(np.mean(e * e)))
            row[f"{var}_bias_{tag}"] = float(np.mean(e))

    budget = member_budget(field(case, "hfx_w") - field(case, "hfx_nr"),
                           field(case, "hfx_s") - field(case, "hfx_w"), budget_mask)
    row["hfx_delta_mse"] = budget["delta_mse"]
    row["hfx_cross"] = budget["cross"]
    row["hfx_square"] = budget["square"]
    row["hfx_closure_mean"] = budget["closure"]
    row["hfx_closure_maxabs"] = budget["closure_maxabs"]
    for var in ("tsk", "d", "dtheta"):
        mask = _variable_mask(case, var, region, mask_kind)
        delta_var = fe(f"e_{var}_s") - fe(f"e_{var}_w")  # δ = e_S − e_W = X_S − X_W
        sub = member_budget(fe(f"e_{var}_w"), delta_var, mask)
        row[f"{var}_delta_mse"] = sub["delta_mse"]

    for gap in ("dtheta", "d"):
        mask = _variable_mask(case, gap, region, mask_kind)
        for tag in ("w", "s"):
            if gap == "dtheta":
                es = field(case, f"theta_sfc_{tag}") - field(case, "theta_sfc_nr")
                ea = field(case, f"theta_air_{tag}") - field(case, "theta_air_nr")
            else:
                es = fe(f"e_tsk_{tag}")
                ea = fe(f"e_tair_{tag}")
            terms = three_term_decomposition(es, ea, mask)
            row[f"{gap}3_surface_{tag}"] = terms["surface"]
            row[f"{gap}3_air_{tag}"] = terms["air"]
            row[f"{gap}3_cross_{tag}"] = terms["cross"]

    row["delta_tsk_mean"] = float(np.mean(fe("delta_tsk")[budget_mask])) if row["n_points"] else np.nan
    row["delta_hfx_mean"] = float(np.mean(fe("delta_hfx")[budget_mask])) if row["n_points"] else np.nan
    return row


def classify_spatial(case: xr.Dataset, region: str, mask_kind: str,
                     tol_tsk: float, tol_hfx: float) -> dict:
    """四类空间对应 + 近零类计数;容差作用于逐点 ΔSE(非增量阈值)。"""
    err = derive_error_fields(case)
    dse_tsk = np.asarray(err["dse_tsk"].values, dtype=float)
    dse_hfx = np.asarray(err["dse_hfx"].values, dtype=float)
    dse_d = np.asarray(err["dse_d"].values, dtype=float)
    dse_dtheta = np.asarray(err["dse_dtheta"].values, dtype=float)
    base = region_mask(case, region)
    if mask_kind == "ocean_common":
        base = base & field(case, "mask_ocean_common").astype(bool)
    mask = base & common_finite_mask(dse_tsk, dse_hfx)
    n = int(np.count_nonzero(mask))
    sst_imp = mask & (dse_tsk < -tol_tsk)
    sst_wor = mask & (dse_tsk > tol_tsk)
    sst_unch = mask & (np.abs(dse_tsk) <= tol_tsk)
    hfx_imp = mask & (dse_hfx < -tol_hfx)
    hfx_wor = mask & (dse_hfx > tol_hfx)
    hfx_unch = mask & (np.abs(dse_hfx) <= tol_hfx)

    def mean_in(field_array, m):
        return float(np.mean(field_array[m])) if int(np.count_nonzero(m)) else np.nan

    return {
        "region": region, "mask_kind": mask_kind,
        "tol_tsk_k2": float(tol_tsk), "tol_hfx_wm22": float(tol_hfx),
        "n_common": n,
        "n_sst_improve": int(np.count_nonzero(sst_imp)),
        "n_sst_worsen": int(np.count_nonzero(sst_wor)),
        "n_sst_unchanged": int(np.count_nonzero(sst_unch)),
        "n_hfx_improve": int(np.count_nonzero(hfx_imp)),
        "n_hfx_worsen": int(np.count_nonzero(hfx_wor)),
        "n_hfx_unchanged": int(np.count_nonzero(hfx_unch)),
        "n_sst_improve_hfx_improve": int(np.count_nonzero(sst_imp & hfx_imp)),
        "n_sst_improve_hfx_worsen": int(np.count_nonzero(sst_imp & hfx_wor)),
        "n_sst_worsen_hfx_improve": int(np.count_nonzero(sst_wor & hfx_imp)),
        "n_sst_worsen_hfx_worsen": int(np.count_nonzero(sst_wor & hfx_wor)),
        "mean_dse_d": mean_in(dse_d, mask),
        "mean_dse_dtheta": mean_in(dse_dtheta, mask),
    }


def member_dse_h_ranking(member_metrics: pd.DataFrame) -> dict:
    """正文案例成员选择:0.5–2 h、region=all、ocean_common 掩膜下按成员平均
    ΔMSE_H 取最大/最小;规则写死于此,图注引用。"""
    sub = member_metrics[
        (member_metrics["mask_kind"] == "ocean_common")
        & (member_metrics["region"] == "all")
        & (member_metrics["lead_hour"] >= 0.5)
        & (member_metrics["lead_hour"] <= 2.0)
    ]
    per_member = sub.groupby("member")["hfx_delta_mse"].mean()
    if per_member.empty:
        return {"strongest_member": None, "weakest_member": None, "rule": "no data"}
    return {
        "strongest_member": str(per_member.idxmax()),
        "weakest_member": str(per_member.idxmin()),
        "per_member_mean_dMSE_H": {str(k): float(v) for k, v in per_member.items()},
        "rule": ("mean ΔMSE_H over 0.5–2 h, region=all, mask=ocean_common, "
                 "member-internal budget then equal-weight members; "
                 "max -> strongest case member, min -> weakest"),
    }


# =====================================================================
# Task 3:通量重建与固定其余输入的 TSK 替换
# =====================================================================

def _recon_inputs(case: xr.Dataset, tag: str, config: dict) -> dict:
    """重建输入 A(除 TSK 外全部输入;含 UST 初值、诊断 z0m、高度、气压等)。"""
    znt = None
    if config["recon_z0m_source"] == "znt_variable":
        if "znt_" + tag in case:
            znt = field(case, "znt_" + tag)
        else:
            raise KeyError("recon_z0m_source='znt_variable' 但案例缓存无 ZNT;请改回 diagnostic 或重新提取")
    ustar = field(case, f"ust_{tag}")
    z0m = (znt if znt is not None
           else initial_momentum_roughness(ustar, config["isftcflx"]))
    return {
        "air_temperature_k": field(case, f"t_air_{tag}"),
        "vapor_mixing_ratio": field(case, f"qvapor_{tag}"),
        "air_pressure_pa": field(case, f"p_air_{tag}"),
        "surface_pressure_pa": field(case, f"psfc_{tag}"),
        "height_agl_m": field(case, f"z_agl_{tag}"),
        "u_ms": field(case, f"u_level_{tag}"),
        "v_ms": field(case, f"v_level_{tag}"),
        "initial_friction_velocity_ms": ustar,
        "initial_momentum_roughness_m": z0m,
    }


_RECON_INPUT_KEYS = (
    "air_temperature_k", "vapor_mixing_ratio", "air_pressure_pa", "surface_pressure_pa",
    "height_agl_m", "u_ms", "v_ms", "initial_friction_velocity_ms",
    "initial_momentum_roughness_m",
)


def _sfclay_hfx(inputs: dict, ts: np.ndarray, mask: np.ndarray, config: dict) -> dict:
    """在统一掩膜上调用 wrf41_sfclayrev(复用物理,不修改),输出 HFX 与下限计数。

    flhc/HFX 公式逐行取自 omtmp_flux_decomposition.reconstruct_ocean_fluxes
    (lines 71-94);与该函数的差异仅在于显式保留下限前的原始值用于审计。
    """
    index = {key: np.asarray(inputs[key], dtype=float)[mask] for key in _RECON_INPUT_KEYS}
    surface = np.asarray(ts, dtype=float)[mask]
    for key, values in index.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"reconstruction input {key} has non-finite values")
    if not np.all(np.isfinite(surface)):
        raise ValueError("surface temperature has non-finite values")
    for key in ("air_pressure_pa", "surface_pressure_pa", "height_agl_m",
                "initial_friction_velocity_ms", "initial_momentum_roughness_m"):
        if np.any(index[key] <= 0.0):
            raise ValueError(f"reconstruction input {key} must be positive")

    n_clamp_ustar = int(np.count_nonzero(index["initial_friction_velocity_ms"] < config["ustar_floor"]))
    n_clamp_z0m = int(np.count_nonzero(index["initial_momentum_roughness_m"] < config["z0m_floor"]))
    state = SurfaceState(
        air_temperature_k=index["air_temperature_k"][None, :],
        surface_temperature_k=surface[None, :],
        vapor_mixing_ratio=index["vapor_mixing_ratio"][None, :],
        air_pressure_pa=index["air_pressure_pa"][None, :],
        surface_pressure_pa=index["surface_pressure_pa"][None, :],
        height_agl_m=index["height_agl_m"][None, :],
        u_ms=index["u_ms"][None, :],
        v_ms=index["v_ms"][None, :],
        initial_friction_velocity_ms=np.maximum(
            index["initial_friction_velocity_ms"], config["ustar_floor"])[None, :],
        initial_momentum_roughness_m=np.maximum(
            index["initial_momentum_roughness_m"], config["z0m_floor"])[None, :],
        dx_m=float(config["dx_m"]),
    )
    moisture = revised_mm5_ocean_flux(state, SfclayOptions(isftcflx=config["isftcflx"]))

    ta = state.air_temperature_k
    ts_ = state.surface_temperature_k
    qa = state.vapor_mixing_ratio
    pa = state.air_pressure_pa
    ps = state.surface_pressure_pa
    z = state.height_agl_m
    theta_air = ta * (P0_PA / pa) ** R_OVER_CP
    theta_sfc = ts_ * (P0_PA / ps) ** R_OVER_CP
    z_over_l = moisture.inverse_obukhov_length * z
    heat_correction = _integrated_psi(z_over_l, z, moisture.moisture_roughness_m, "heat")
    heat_denominator = (np.log((z + moisture.moisture_roughness_m) / moisture.moisture_roughness_m)
                        - heat_correction)
    virtual_temperature = ta * (1.0 + EP1 * qa)
    density = ps / (287.0 * virtual_temperature)
    moist_cp = 1004.0 * (1.0 + 0.8 * qa)
    flhc = moist_cp * density * moisture.friction_velocity_ms * KARMAN / heat_denominator
    hfx_raw = flhc * (theta_sfc - theta_air)
    floor = float(config["hfx_floor_wm2"])
    hfx = np.maximum(hfx_raw, floor)
    floor_bind = hfx_raw < floor
    def full(values):
        out = np.full(mask.shape, np.nan)
        out[mask] = np.asarray(values, dtype=float).ravel()
        return out

    floor_full = np.zeros(mask.shape, dtype=bool)
    floor_full[mask] = np.asarray(floor_bind, dtype=bool).ravel()
    return {
        "hfx": full(hfx), "hfx_raw": full(hfx_raw), "lh": full(moisture.lh),
        "qfx": full(moisture.qfx), "ust": full(moisture.friction_velocity_ms),
        "floor_bind": floor_full,
        "n_clamp_ustar": n_clamp_ustar, "n_clamp_z0m": n_clamp_z0m,
    }


def reconstruction_quality(ratio_w, ratio_s, ratio_diff, response_rms,
                           strict: float, limited: float, floor: float) -> dict:
    """重建质量状态(近零分母先行;阈值是本任务数值诊断标准,非物理常数)。

    response_rms < floor(默认 0.5 W m-2)时比率置 NaN 并标 near_zero,
    绝对残差由调用方原样保留,不得生成贡献比或 pass。
    """
    ratios = np.asarray([ratio_w, ratio_s, ratio_diff], dtype=float)
    if response_rms is None or not np.isfinite(response_rms) or response_rms < floor:
        return {"status": "near_zero", "ratio_w": np.nan, "ratio_s": np.nan,
                "ratio_diff": np.nan, "response_rms": (
                    float(response_rms) if response_rms is not None and np.isfinite(response_rms)
                    else np.nan)}
    if not np.all(np.isfinite(ratios)):
        return {"status": "not_evaluable", "ratio_w": float(ratios[0]),
                "ratio_s": float(ratios[1]), "ratio_diff": float(ratios[2]),
                "response_rms": float(response_rms)}
    rw, rs, rd = float(ratios[0]), float(ratios[1]), float(ratios[2])
    if max(rw, rs, rd) <= strict:
        status = "strict"
    elif max(rw, rs, rd) <= limited:
        status = "limited"
    else:
        status = "fail"
    return {"status": status, "ratio_w": rw, "ratio_s": rs, "ratio_diff": rd,
            "response_rms": float(response_rms)}


def quality_sensitivity(ratio_w, ratio_s, ratio_diff,
                        thresholds=(0.1, 0.2, 0.5)) -> dict:
    """固定三判据数值,对阈值 0.1/0.2/0.5 逐一给 pass/fail;不得事后改阈值筛结果。"""
    ratios = np.asarray([ratio_w, ratio_s, ratio_diff], dtype=float)
    out = {}
    for t in thresholds:
        if not np.all(np.isfinite(ratios)):
            out[f"pass_at_{t:g}"] = "not_evaluable"
        else:
            out[f"pass_at_{t:g}"] = "pass" if float(np.max(ratios)) <= float(t) else "fail"
    return out


def evaluate_reconstruction(case: xr.Dataset, config: dict) -> tuple[xr.Dataset, dict]:
    """四组合重建 + 双路径分解 + 三判据质量状态。

    F_WW=f(TSK_W,A_W)、F_SW=f(TSK_S,A_W)、F_SS=f(TSK_S,A_S)、F_WS=f(TSK_W,A_S)。
    正向 dF_T=F_SW−F_WW、dF_rest=F_SS−F_SW;反向 dF_T_rev=F_SS−F_WS、
    dF_rest_rev=F_WS−F_WW;两路径均闭合到 F_SS−F_WW。掩膜外 NaN。
    """
    mask = (region_mask(case, "all") & field(case, "mask_ocean_common").astype(bool)
            & common_finite_mask(field(case, "hfx_w"), field(case, "hfx_s"), field(case, "hfx_nr")))
    inputs_w = _recon_inputs(case, "w", config)
    inputs_s = _recon_inputs(case, "s", config)
    combined = mask & common_finite_mask(
        *(inputs_w[key] for key in _RECON_INPUT_KEYS),
        *(inputs_s[key] for key in _RECON_INPUT_KEYS),
        field(case, "tsk_w"), field(case, "tsk_s"),
    )
    ts_w = field(case, "tsk_w")
    ts_s = field(case, "tsk_s")
    calls = {
        "F_ww": _sfclay_hfx(inputs_w, ts_w, combined, config),
        "F_sw": _sfclay_hfx(inputs_w, ts_s, combined, config),
        "F_ss": _sfclay_hfx(inputs_s, ts_s, combined, config),
        "F_ws": _sfclay_hfx(inputs_s, ts_w, combined, config),
    }
    floor_bind = {name: values["floor_bind"] for name, values in calls.items()}
    clamps = {f"n_clamp_ustar_{name}": values["n_clamp_ustar"]
              for name, values in calls.items()}
    clamps.update({f"n_clamp_z0m_{name}": values["n_clamp_z0m"]
                   for name, values in calls.items()})

    out: dict[str, np.ndarray] = {}
    for name, values in calls.items():
        out[f"{name}_hfx"] = values["hfx"]
        out[f"{name}_lh"] = values["lh"]
    out["dF_T"] = out["F_sw_hfx"] - out["F_ww_hfx"]
    out["dF_rest"] = out["F_ss_hfx"] - out["F_sw_hfx"]
    out["dF_total"] = out["F_ss_hfx"] - out["F_ww_hfx"]
    out["dF_T_rev"] = out["F_ss_hfx"] - out["F_ws_hfx"]
    out["dF_rest_rev"] = out["F_ws_hfx"] - out["F_ww_hfx"]
    out["dF_T_avg"] = 0.5 * (out["dF_T"] + out["dF_T_rev"])
    out["dF_rest_avg"] = 0.5 * (out["dF_rest"] + out["dF_rest_rev"])
    out["path_difference"] = out["dF_T"] - out["dF_T_rev"]
    out["closure_forward"] = out["dF_total"] - (out["dF_T"] + out["dF_rest"])
    out["closure_reverse"] = out["dF_total"] - (out["dF_T_rev"] + out["dF_rest_rev"])
    out["residual_w"] = out["F_ww_hfx"] - field(case, "hfx_w")
    out["residual_s"] = out["F_ss_hfx"] - field(case, "hfx_s")
    delta_actual = field(case, "hfx_s") - field(case, "hfx_w")
    out["delta_actual"] = delta_actual
    out["total_minus_actual"] = out["dF_total"] - delta_actual
    out["floor_bind_any"] = floor_bind["F_ww"] | floor_bind["F_sw"] | floor_bind["F_ss"] | floor_bind["F_ws"]

    ds = xr.Dataset(
        {name: (("y", "x"), np.asarray(values, dtype=float)) for name, values in out.items()},
        coords={k: case.coords[k] for k in ("y", "x", "lat", "lon") if k in case.coords},
    )
    ds["mask_recon"] = (("y", "x"), combined)
    ds["floor_bind_any"] = (("y", "x"), np.asarray(out["floor_bind_any"], dtype=bool))
    ds.attrs["meta_json"] = case.attrs.get("meta_json", "")

    n = int(np.count_nonzero(combined))
    if n == 0:
        metrics = {"status": "empty_mask", "n_valid": 0, **clamps}
        return ds, metrics

    def rms(array):
        values = np.asarray(array, dtype=float)[combined]
        return float(np.sqrt(np.mean(values * values)))

    response_rms = rms(delta_actual)
    ratio_w = rms(out["residual_w"]) / response_rms
    ratio_s = rms(out["residual_s"]) / response_rms
    ratio_diff = rms(out["total_minus_actual"]) / response_rms
    quality = reconstruction_quality(
        ratio_w, ratio_s, ratio_diff, response_rms,
        strict=config["recon_ratio_strict"], limited=config["recon_ratio_limited"],
        floor=config["recon_response_floor_wm2"],
    )
    sensitivity = quality_sensitivity(
        quality["ratio_w"], quality["ratio_s"], quality["ratio_diff"],
        thresholds=config["recon_ratio_sensitivity"],
    )
    e_h_w = field(case, "hfx_w") - field(case, "hfx_nr")
    metrics = {
        "n_valid": n,
        "response_rms_wm2": response_rms,
        "residual_w_rms": rms(out["residual_w"]),
        "residual_s_rms": rms(out["residual_s"]),
        "total_minus_actual_rms": rms(out["total_minus_actual"]),
        "ratio_w": quality["ratio_w"], "ratio_s": quality["ratio_s"],
        "ratio_diff": quality["ratio_diff"],
        "quality_status": quality["status"],
        **sensitivity,
        "cross_2e_deltaF_T": float(np.mean(2.0 * e_h_w[combined] * out["dF_T"][combined])),
        "rmse_F_sw_minus_nr": rms(out["F_sw_hfx"] - field(case, "hfx_nr")),
        "rmse_F_ww_minus_nr": rms(out["F_ww_hfx"] - field(case, "hfx_nr")),
        "rmse_improvement_recon_baseline": (
            rms(out["F_ww_hfx"] - field(case, "hfx_nr")) - rms(out["F_sw_hfx"] - field(case, "hfx_nr"))
        ),
        "path_difference_rms": rms(out["path_difference"]),
        "closure_forward_maxabs": float(np.nanmax(np.abs(out["closure_forward"][combined]))),
        "closure_reverse_maxabs": float(np.nanmax(np.abs(out["closure_reverse"][combined]))),
        "n_floor_bind": int(np.count_nonzero(np.asarray(out["floor_bind_any"])[combined])),
        **clamps,
    }
    metrics["status"] = "ok"
    return ds, metrics


# =====================================================================
# Task 4:台风居中结构比较
# =====================================================================

class LocalMapper:
    """成员网格 -> 局地等距投影网格;Delaunay 线性插值,凸包外 NaN(不外推)。

    投影:equirectangular 局地近似 x=R·cosφ_c·Δλ, y=R·Δφ(单位 km)。
    源网格点在节点处的插值是精确的(测试 test_storm_translation_recovery)。
    """

    def __init__(self, src_lat: np.ndarray, src_lon: np.ndarray,
                 center_lat: float, center_lon: float, earth_radius_km: float = 6371.0):
        from scipy.spatial import Delaunay

        self.center = (float(center_lat), float(center_lon))
        lat = np.asarray(src_lat, dtype=float)
        lon = np.asarray(src_lon, dtype=float)
        x = np.deg2rad(lon - center_lon) * np.cos(np.deg2rad(center_lat)) * earth_radius_km
        y = np.deg2rad(lat - center_lat) * earth_radius_km
        points = np.column_stack([x.ravel(), y.ravel()])
        good = np.isfinite(points).all(axis=1)
        self.points = points[good]
        self._flat_index = np.nonzero(good.ravel())[0]
        self.tri = Delaunay(self.points)

    def grid(self, extent_km: float, step_km: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        axis = np.linspace(-extent_km, extent_km, int(round(2 * extent_km / step_km)) + 1)
        gx, gy = np.meshgrid(axis, axis)
        return axis, gx, gy

    def map(self, values: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        from scipy.interpolate import LinearNDInterpolator

        flat = np.asarray(values, dtype=float).ravel()[self._flat_index]
        interpolator = LinearNDInterpolator(self.tri, flat, fill_value=np.nan)
        result = interpolator(gx.ravel(), gy.ravel())
        return np.asarray(result, dtype=float).reshape(gx.shape)


_STORM_VARIABLES = ("tsk", "hfx", "d")


def compare_storm_relative(case: xr.Dataset, config: dict) -> tuple[xr.Dataset, dict, list[dict]]:
    """双映射(各自中心 / NR 中心)结构比较;匹配公共掩膜上计算全部评分。

    插值后的评分是结构诊断,不能替代原表原掩膜评分,也不能解释为严格的位置
    误差因果分解(位移、插值、掩膜影响混在一起)。
    """
    extent = config["storm_grid_extent_km"]
    step = config["storm_grid_step_km"]
    axis, gx, gy = LocalMapper(field(case, "lat"), field(case, "lon"), 0.0, 0.0).grid(extent, step)
    radius_grid = np.sqrt(gx * gx + gy * gy)
    target_in = radius_grid <= extent

    meta = json.loads(case.attrs.get("meta_json", "{}"))
    centers = {
        "w": (meta["center_w"]["lat"], meta["center_w"]["lon"]),
        "s": (meta["center_s"]["lat"], meta["center_s"]["lon"]),
        "nr": (meta["nr_center"]["lat"], meta["nr_center"]["lon"]),
    }
    lat = field(case, "lat")
    lon = field(case, "lon")

    # 变量场:先在原生网格算 d,再映射(与主口径一致)
    sources = {
        ("tsk", "w"): field(case, "tsk_w"), ("tsk", "s"): field(case, "tsk_s"),
        ("tsk", "nr"): field(case, "tsk_nr"),
        ("hfx", "w"): field(case, "hfx_w"), ("hfx", "s"): field(case, "hfx_s"),
        ("hfx", "nr"): field(case, "hfx_nr"),
        ("d", "w"): field(case, "d_w"), ("d", "s"): field(case, "d_s"),
        ("d", "nr"): field(case, "d_nr"),
        ("landmask", "w"): field(case, "landmask_w"),
        ("landmask", "s"): field(case, "landmask_s"),
        ("landmask", "nr"): field(case, "landmask_nr"),
    }

    mapped: dict[tuple[str, str, str], np.ndarray] = {}
    for mapping, center_key in (("centered", None), ("fixed", "nr")):
        for party in ("w", "s", "nr"):
            center = centers[party] if mapping == "centered" else centers[center_key]
            mapper = LocalMapper(lat, lon, center[0], center[1], config["earth_radius_km"])
            for name in ("tsk", "hfx", "d", "landmask"):
                mapped[(name, party, mapping)] = mapper.map(sources[(name, party)], gx, gy)

    matched: dict[str, np.ndarray] = {}
    for name in ("tsk", "hfx", "d"):
        m = target_in.copy()
        for party in ("w", "s", "nr"):
            for mapping in ("centered", "fixed"):
                m &= np.isfinite(mapped[(name, party, mapping)])
        matched[name] = m
    ocean_matched = target_in.copy()
    for party in ("w", "s", "nr"):
        for mapping in ("centered", "fixed"):
            ocean_matched &= (mapped[("landmask", party, mapping)] < 0.5)
            ocean_matched &= np.isfinite(mapped[("landmask", party, mapping)])

    storm = xr.Dataset(
        {f"{name}_{party}_{mapping}": (("gy", "gx"), mapped[(name, party, mapping)].astype(np.float32))
         for (name, party, mapping) in mapped},
        coords={"gx": axis, "gy": axis},
    )
    for name in matched:
        storm[f"matched_{name}"] = (("gy", "gx"), matched[name])
    storm["ocean_matched"] = (("gy", "gx"), ocean_matched)
    storm["target_in"] = (("gy", "gx"), target_in)
    storm.attrs["meta_json"] = case.attrs.get("meta_json", "")

    metrics: dict = {"status": "ok"}
    bin_km = config["storm_radial_bin_km"]
    edges = np.arange(0.0, extent + bin_km, bin_km)
    radial_rows: list[dict] = []

    def rmse_on(a, b, m):
        diff = np.asarray(a, dtype=float)[m] - np.asarray(b, dtype=float)[m]
        return float(np.sqrt(np.mean(diff * diff))) if diff.size else np.nan

    def bias_on(a, b, m):
        diff = np.asarray(a, dtype=float)[m] - np.asarray(b, dtype=float)[m]
        return float(np.mean(diff)) if diff.size else np.nan

    for name in _STORM_VARIABLES:
        m = matched[name]
        metrics[f"n_matched_{name}"] = int(np.count_nonzero(m))
        metrics[f"n_matched_ocean_{name}"] = int(np.count_nonzero(m & ocean_matched))
        for party in ("w", "s"):
            for mapping in ("centered", "fixed"):
                key = f"{name}_{party}_{mapping}"
                metrics[f"{key}_rmse"] = rmse_on(mapped[(name, party, mapping)],
                                                 mapped[(name, "nr", mapping)], m)
                metrics[f"{key}_bias"] = bias_on(mapped[(name, party, mapping)],
                                                 mapped[(name, "nr", mapping)], m)
                # 非轴对称残差 RMS(径向 10 km 环带的方位平均后的剩余场)
                field_values = np.asarray(mapped[(name, party, mapping)], dtype=float)
                binned = np.full_like(field_values, np.nan)
                for k in range(len(edges) - 1):
                    in_bin = m & (radius_grid >= edges[k]) & (radius_grid < edges[k + 1])
                    if np.any(in_bin):
                        binned[in_bin] = np.nanmean(field_values[in_bin])
                residual = field_values - binned
                metrics[f"{key}_nonaxial_rms"] = (
                    float(np.sqrt(np.nanmean(residual[m] ** 2)))
                    if np.any(np.isfinite(residual[m])) else np.nan
                )
                # 径向廓线(仅居中映射,逐 10 km 环带)
                if mapping == "centered":
                    for k in range(len(edges) - 1):
                        in_bin = m & (radius_grid >= edges[k]) & (radius_grid < edges[k + 1])
                        radial_rows.append({
                            "variable": name, "party": party,
                            "bin_center_km": 0.5 * (edges[k] + edges[k + 1]),
                            "mean": float(np.nanmean(field_values[in_bin]))
                            if np.any(in_bin) else np.nan,
                            "n": int(np.count_nonzero(in_bin & np.isfinite(field_values))),
                        })
        for mapping in ("centered", "fixed"):
            metrics[f"{name}_paired_rmse_change_{mapping}"] = (
                metrics[f"{name}_s_{mapping}_rmse"] - metrics[f"{name}_w_{mapping}_rmse"]
            )

    # ±1 网格中心扰动(定位敏感性):成员侧 ±i/±j 邻居作为替中心,记录评分极差
    if config.get("center_perturb_enabled"):
        sub = meta.get("subwindow", {"ys": [0, lat.shape[0]], "xs": [0, lat.shape[1]]})
        for party in ("w", "s"):
            j = int(meta[f"center_{party}"]["j"]) - sub["ys"][0]
            i = int(meta[f"center_{party}"]["i"]) - sub["xs"][0]
            neighbors = []
            for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jj, ii = j + dj, i + di
                if 0 <= jj < lat.shape[0] and 0 <= ii < lat.shape[1]:
                    neighbors.append((float(lat[jj, ii]), float(lon[jj, ii])))
            for name in _STORM_VARIABLES:
                rmse_values = []
                for clat, clon in neighbors:
                    mapper = LocalMapper(lat, lon, clat, clon, config["earth_radius_km"])
                    field_values = mapper.map(sources[(name, party)], gx, gy)
                    nr_values = mapped[(name, "nr", "centered")]
                    both = np.isfinite(field_values) & np.isfinite(nr_values)
                    if np.any(both):
                        rmse_values.append(rmse_on(field_values, nr_values, both))
                metrics[f"{name}_{party}_centered_perturb_rmse_spread"] = (
                    float(np.max(rmse_values) - np.min(rmse_values)) if rmse_values else np.nan
                )
    return storm, metrics, radial_rows


def center_diagnostics_row(case: xr.Dataset, config: dict) -> dict:
    """中心定义与歧义检查(主中心 = 全域最低 PSFC;不静默改中心)。"""
    meta = json.loads(case.attrs.get("meta_json", "{}"))
    row: dict = {
        "case_id": meta.get("case_id"),
        "nr_center_lat": meta["nr_center"]["lat"], "nr_center_lon": meta["nr_center"]["lon"],
        "nr_center_j": meta["nr_center"]["j"], "nr_center_i": meta["nr_center"]["i"],
        "nr_center_pressure_pa": meta["nr_center"]["pressure_pa"],
        "nr_center_boundary_flag": meta["nr_center"]["boundary_flag"],
        "nr_center_ambiguous_flag": meta["nr_center"]["ambiguous_flag"],
        "w_center_lat": meta["center_w"]["lat"], "w_center_lon": meta["center_w"]["lon"],
        "w_center_boundary_flag": meta["center_w"]["boundary_flag"],
        "w_center_ambiguous_flag": meta["center_w"]["ambiguous_flag"],
        "s_center_lat": meta["center_s"]["lat"], "s_center_lon": meta["center_s"]["lon"],
        "s_center_boundary_flag": meta["center_s"]["boundary_flag"],
        "s_center_ambiguous_flag": meta["center_s"]["ambiguous_flag"],
        "w_center_error_km": meta["center_error_km"]["w"],
        "s_center_error_km": meta["center_error_km"]["s"],
        "ws_center_separation_km": float(great_circle_distance_km(
            meta["center_w"]["lat"], meta["center_w"]["lon"],
            meta["center_s"]["lat"], meta["center_s"]["lon"], config["earth_radius_km"])),
    }
    return row


# =====================================================================
# 缓存与配置快照
# =====================================================================

def config_snapshot(config: dict) -> dict:
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    return clean(config)


def _bool_to_int8(dataset: xr.Dataset) -> xr.Dataset:
    out = dataset.copy(deep=False)
    for name in out.data_vars:
        if out[name].dtype == bool:
            out[name] = out[name].astype("int8")
    return out


def _int8_to_bool(dataset: xr.Dataset) -> xr.Dataset:
    out = dataset.copy(deep=False)
    for name in out.data_vars:
        if (name.startswith("mask_") or name.startswith("matched_")
                or name in ("target_in", "ocean_matched", "floor_bind_any")):
            if out[name].dtype == np.int8:
                out[name] = out[name].astype(bool)
    return out


def save_case_cache(case: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in _bool_to_int8(case).data_vars}
    _bool_to_int8(case).to_netcdf(path, encoding=encoding)


def load_case_cache(path: Path) -> xr.Dataset:
    case = xr.load_dataset(path)
    return _int8_to_bool(case)


def save_recon_cache(recon: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in recon.data_vars}
    recon.to_netcdf(path, encoding=encoding)


def load_recon_cache(path: Path) -> xr.Dataset:
    return _int8_to_bool(xr.load_dataset(path))


def save_storm_cache(storm: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _bool_to_int8(storm)
    encoding = {name: {"zlib": True, "complevel": 4} for name in out.data_vars}
    out.to_netcdf(path, encoding=encoding)


def load_storm_cache(path: Path) -> xr.Dataset:
    return _int8_to_bool(xr.load_dataset(path))


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


# =====================================================================
# 基线复现
# =====================================================================

def _baseline_lookup(config: dict, variable: str) -> pd.DataFrame:
    key = "baseline_tsk_csv" if variable == "tsk" else "baseline_hfx_csv"
    frame = pd.read_csv(paths(config)[key])
    value_cols = [c for c in frame.columns if c.startswith("mean_member_rmse")]
    if not value_cols:
        raise ValueError(f"baseline CSV {key} lacks mean_member_rmse column")
    return frame[["time", "experiment", "filter", value_cols[0]]].rename(
        columns={value_cols[0]: "baseline_mean_member_rmse"})


def baseline_reproduction_table(member_metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    """原掩膜(region=all, mask_kind=original)成员平均 RMSE 与服务器原表逐项对比。

    验收:|Δ| ≤ 1e-3 W m-2(HFX)、≤ 1e-5 K(TSK);超限先查数据/插值/掩膜。
    """
    sub = member_metrics[(member_metrics["mask_kind"] == "original")
                         & (member_metrics["region"] == "all")]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for variable, tol in (("tsk", 1.0e-5), ("hfx", 1.0e-3)):
        baseline = _baseline_lookup(config, variable)
        for experiment, exp_tag in ((config["weak_experiment"], "w"),
                                    (config["strong_experiment"], "s")):
            for method in config["methods"]:
                for hour in sorted(sub["lead_hour"].unique()):
                    stamp = stamp_from_lead(float(hour))
                    selected = sub[(sub["method"] == method) & (sub["lead_hour"] == hour)]
                    if selected.empty:
                        continue
                    computed = float(selected[f"{variable}_rmse_{exp_tag}"].mean())
                    hit = baseline[(baseline["time"] == stamp)
                                   & (baseline["experiment"] == experiment)
                                   & (baseline["filter"] == method)]
                    if hit.empty:
                        rows.append({"variable": variable, "experiment": experiment,
                                     "method": method, "valid_time": stamp,
                                     "computed_mean_member_rmse": computed,
                                     "baseline_mean_member_rmse": np.nan,
                                     "absdiff": np.nan, "tolerance": tol,
                                     "status": "baseline_row_missing"})
                        continue
                    reference = float(hit["baseline_mean_member_rmse"].iloc[0])
                    diff = abs(computed - reference)
                    rows.append({"variable": variable, "experiment": experiment,
                                 "method": method, "valid_time": stamp,
                                 "computed_mean_member_rmse": computed,
                                 "baseline_mean_member_rmse": reference,
                                 "absdiff": diff, "tolerance": tol,
                                 "status": "pass" if diff <= tol else "fail"})
    return pd.DataFrame(rows)


# =====================================================================
# 批处理入口(Task 2 / 3 / 4;服务器执行)
# =====================================================================

def _ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {name: root / name for name in ("code", "cache", "tables", "figures", "logs", "source_data")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _check_environment(config: dict) -> None:
    if config["run_environment"] == "local_smoke":
        return
    for key in ("member_base", "nr_base"):
        if not Path(paths(config)[key]).exists():
            raise FileNotFoundError(f"{key} 不存在: {paths(config)[key]};"
                                    "请确认在 lililei1 上以 server 环境运行")


def _hours(config: dict) -> tuple[float, ...]:
    return tuple(config["main_hours"]) + tuple(config["contrast_hours"])


def run_task2(config: dict = CONFIG) -> Path:
    """提取 + 逐点误差 + 成员指标 + 分类 + 基线复现 + 案例缓存。"""
    _check_environment(config)
    root = Path(paths(config)["output_root"])
    dirs = _ensure_dirs(root)
    cache_dir = dirs["cache"] / "case"
    manifest_rows, member_rows, coverage_rows, class_rows, case_meta_rows = [], [], [], [], []
    for method in config["methods"]:
        for member in config["members"]:
            for hour in _hours(config):
                cid = case_id(method, member, hour)
                print(f"[task2] {cid}", flush=True)
                try:
                    case = extract_case(config, method, member, hour)
                except Exception as error:  # 缺文件/坏文件逐条登记,不静默丢弃
                    manifest_rows.append({"case_id": cid, "task": "task2",
                                          "status": f"extract_failed: {type(error).__name__}: {error}"})
                    print(f"[task2] {cid} FAILED: {type(error).__name__}: {error}",
                          file=sys.stderr, flush=True)
                    traceback.print_exc()
                    continue
                cache_path = cache_dir / f"case_{cid}.nc"
                save_case_cache(case, cache_path)
                manifest_rows.append({"case_id": cid, "task": "task2", "status": "ok",
                                      "cache": str(cache_path)})
                for region in REGIONS:
                    for mask_kind in MASK_KINDS:
                        member_rows.append({**summarize_case(case, region, mask_kind),
                                            "method": method, "member": member,
                                            "lead_hour": hour,
                                            "valid_time": stamp_from_lead(hour)})
                        for tol_name, tol_tsk, tol_hfx in (
                            ("default", config["dse_tsk_tol_k2"], config["dse_hfx_tol_wm22"]),
                            ("zero", 0.0, 0.0),
                        ):
                            class_rows.append({**classify_spatial(case, region, mask_kind,
                                                                  tol_tsk, tol_hfx),
                                               "method": method, "member": member,
                                               "lead_hour": hour, "tol_name": tol_name,
                                               "valid_time": stamp_from_lead(hour)})
                meta = json.loads(case.attrs["meta_json"])
                coverage_rows.append({
                    "case_id": cid, "method": method, "member": member, "lead_hour": hour,
                    "region_points": meta["region_points"],
                    "ocean_common_points": meta["ocean_common_points"],
                    "nr_center_lat": meta["nr_center"]["lat"], "nr_center_lon": meta["nr_center"]["lon"],
                    "w_center_error_km": meta["center_error_km"]["w"],
                    "s_center_error_km": meta["center_error_km"]["s"],
                    "dx_member": meta["weak_attrs"].get("DX"),
                    "nr_attrs": json.dumps(meta["nr_attrs"], default=str),
                })
                case_meta_rows.append({"case_id": cid, "cache": str(cache_path),
                                       "valid_time": stamp_from_lead(hour)})
    member_metrics = pd.DataFrame(member_rows, columns=MEMBER_METRIC_COLUMNS)
    member_metrics.to_csv(dirs["tables"] / "member_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(dirs["tables"] / "spatial_classes.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(dirs["tables"] / "coverage.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(dirs["tables"] / "case_manifest.csv", index=False)
    baseline = baseline_reproduction_table(member_metrics, config)
    baseline.to_csv(dirs["tables"] / "baseline_reproduction.csv", index=False)
    write_json(config_snapshot(config), dirs["tables"] / "config_snapshot.json")
    succeeded = sum(row["status"] == "ok" for row in manifest_rows)
    failed = len(manifest_rows) - succeeded
    if succeeded == 0:
        raise RuntimeError(
            f"[task2] no case extracted successfully ({failed} failed); "
            f"see {dirs['tables'] / 'case_manifest.csv'} and the extraction tracebacks"
        )
    print(f"[task2] done: {succeeded} succeeded, {failed} failed -> {dirs['tables']}", flush=True)
    return root


def run_initial_check(config: dict = CONFIG) -> Path:
    """0 h 初值核验:强弱响应 RMS 应接近 0(HFX/UST 为未初始化态,不作物理解释)。"""
    from netCDF4 import Dataset

    _check_environment(config)
    root = Path(paths(config)["output_root"])
    dirs = _ensure_dirs(root)
    rows = []
    for method in config["methods"]:
        for member in config["members"]:
            for hour in config["initial_hours"]:
                stamp = stamp_from_lead(hour)
                case_arrays = {}
                try:
                    for tag, experiment in (("w", config["weak_experiment"]),
                                            ("s", config["strong_experiment"])):
                        path = member_wrfout_path(config, experiment, method, member, stamp)
                        with Dataset(path, "r") as d:
                            case_arrays[tag] = {
                                name: to_fill_nan(d.variables[name][0])
                                for name in ("TSK", "HFX", "UST", "U10", "V10", "PSFC")
                            }
                            t = to_fill_nan(d.variables["T"][0, 0])
                            p = (to_fill_nan(d.variables["P"][0, 0])
                                 + to_fill_nan(d.variables["PB"][0, 0]))
                            case_arrays[tag]["T_air"] = (t + 300.0) * (p / P0_PA) ** R_OVER_CP
                except Exception as error:
                    rows.append({"method": method, "member": member,
                                 "status": f"read_failed: {type(error).__name__}: {error}"})
                    continue
                w, s = case_arrays["w"], case_arrays["s"]
                row = {"method": method, "member": member, "lead_hour": hour, "status": "ok"}
                for name in ("TSK", "T_air", "U10", "V10", "PSFC"):
                    delta = s[name] - w[name]
                    row[f"{name}_response_rms"] = float(np.sqrt(np.nanmean(delta * delta)))
                row["HFX_response_rms"] = float(np.sqrt(np.nanmean((s["HFX"] - w["HFX"]) ** 2)))
                row["UST_max"] = float(np.nanmax(np.abs(w["UST"])))
                rows.append(row)
    pd.DataFrame(rows).to_csv(dirs["tables"] / "initial_handoff_check.csv", index=False)
    print(f"[initial] done -> {dirs['tables'] / 'initial_handoff_check.csv'}", flush=True)
    return root


def run_task3(config: dict = CONFIG) -> Path:
    """重建一致性 + 固定其余输入的 TSK 替换(消费 Task 2 缓存)。"""
    root = Path(paths(config)["output_root"])
    dirs = _ensure_dirs(root)
    cache_dir = dirs["cache"] / "case"
    quality_rows, replacement_rows, audit_rows, manifest_rows = [], [], [], []
    for method in config["methods"]:
        for member in config["members"]:
            for hour in _hours(config):
                cid = case_id(method, member, hour)
                path = cache_dir / f"case_{cid}.nc"
                if not path.exists():
                    manifest_rows.append({"case_id": cid, "task": "task3", "status": "cache_missing"})
                    continue
                print(f"[task3] {cid}", flush=True)
                case = load_case_cache(path)
                try:
                    recon, metrics = evaluate_reconstruction(case, config)
                except Exception as error:
                    manifest_rows.append({"case_id": cid, "task": "task3",
                                          "status": f"recon_failed: {type(error).__name__}: {error}"})
                    print(f"[task3] {cid} FAILED: {type(error).__name__}: {error}",
                          file=sys.stderr, flush=True)
                    traceback.print_exc()
                    continue
                save_recon_cache(recon, dirs["cache"] / "recon" / f"recon_{cid}.nc")
                manifest_rows.append({"case_id": cid, "task": "task3", "status": "ok",
                                      "cache": str(dirs["cache"] / "recon" / f"recon_{cid}.nc")})
                meta = json.loads(case.attrs["meta_json"])
                quality_rows.append({"method": method, "member": member, "lead_hour": hour,
                                     "valid_time": stamp_from_lead(hour), **metrics})
                for region in REGIONS:
                    replacement_rows.append({
                        "method": method, "member": member, "lead_hour": hour,
                        "valid_time": stamp_from_lead(hour), "region": region,
                        "mask_kind": "ocean_common",
                        "n_valid": metrics.get("n_valid"),
                        "quality_status": metrics.get("quality_status"),
                        "dF_T_mean": float(np.nanmean(recon["dF_T"].values[region_mask(case, region)
                                                                                   & np.asarray(recon["mask_recon"].values, bool)])),
                        "dF_rest_mean": float(np.nanmean(recon["dF_rest"].values[region_mask(case, region)
                                                                                          & np.asarray(recon["mask_recon"].values, bool)])),
                        "dF_T_rev_mean": float(np.nanmean(recon["dF_T_rev"].values[region_mask(case, region)
                                                                                               & np.asarray(recon["mask_recon"].values, bool)])),
                        "dF_rest_rev_mean": float(np.nanmean(recon["dF_rest_rev"].values[region_mask(case, region)
                                                                                                      & np.asarray(recon["mask_recon"].values, bool)])),
                        "dF_T_avg_mean": float(np.nanmean(recon["dF_T_avg"].values[region_mask(case, region)
                                                                                                   & np.asarray(recon["mask_recon"].values, bool)])),
                        "dF_rest_avg_mean": float(np.nanmean(recon["dF_rest_avg"].values[region_mask(case, region)
                                                                                                          & np.asarray(recon["mask_recon"].values, bool)])),
                        "delta_actual_mean": float(np.nanmean(recon["delta_actual"].values[region_mask(case, region)
                                                                                                        & np.asarray(recon["mask_recon"].values, bool)])),
                        "path_difference_rms": metrics.get("path_difference_rms"),
                        "n_floor_bind": metrics.get("n_floor_bind"),
                        "status": metrics.get("status"),
                    })
                audit_rows.append({
                    "case_id": cid, "sf_sfclay_physics": meta["weak_attrs"].get("SF_SFCLAY_PHYSICS"),
                    "sf_surface_physics": meta["weak_attrs"].get("SF_SURFACE_PHYSICS"),
                    "sf_ocean_physics": meta["weak_attrs"].get("SF_OCEAN_PHYSICS"),
                    "wrf_version": meta["weak_attrs"].get("VERSION"),
                    "isftcflx_configured": config["isftcflx"],
                    "z0m_source": config["recon_z0m_source"],
                    "n_floor_bind": metrics.get("n_floor_bind"),
                    "quality_status": metrics.get("quality_status"),
                })
    pd.DataFrame(quality_rows).to_csv(dirs["tables"] / "reconstruction_quality.csv", index=False)
    pd.DataFrame(replacement_rows).to_csv(dirs["tables"] / "replacement_metrics.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(dirs["tables"] / "reconstruction_audit_cases.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(dirs["tables"] / "task3_manifest.csv", index=False)
    _write_reconstruction_audit(dirs["tables"], quality_rows, audit_rows, config)
    print(f"[task3] done -> {dirs['tables']}", flush=True)
    return root


def _write_reconstruction_audit(tables_dir: Path, quality_rows: list[dict],
                                audit_rows: list[dict], config: dict) -> None:
    frame = pd.DataFrame(quality_rows)
    counts = frame["quality_status"].value_counts(dropna=False).to_dict() if len(frame) else {}
    lines = [
        "# 通量重建审计(自动生成,数值由 reconstruction_quality.csv / reconstruction_audit_cases.csv 支撑)",
        "",
        "## 方案核对",
        "",
        f"- 重建实现:wrf41_sfclayrev.revised_mm5_ocean_flux + omtmp_flux_decomposition 的 FLHC/HFX 公式,",
        "  未修改物理实现;−250 W m-2 下限保留并逐案例计数 n_floor_bind。",
        f"- isftcflx 配置值 = {config['isftcflx']}(TODO(待核实):需对照试验 namelist;wrfout 属性不含 ISFTCFLX)。",
        f"- dx_m = {config['dx_m']}(TODO(待核实):读取文件属性 DX 核实)。",
        f"- z0m 来源 = {config['recon_z0m_source']}(diagnostic 为 omtmp_pathway_extract 口径;"
        "若 wrfout 输出 ZNT,建议改用 znt_variable 复核)。",
        "- 时序语义:同一 wrfout 时刻的 HFX 是模式该时刻输出通量;重建使用同时刻状态,",
        "  存在单步滞后/迭代语义差异,残差即包含该影响,不能归零解释。",
        "",
        "## 质量状态分布",
        "",
        f"- 状态计数:{counts}",
        f"- 严格阈值 strict ≤ {config['recon_ratio_strict']},limited ≤ {config['recon_ratio_limited']},",
        f"  响应尺度下限 {config['recon_response_floor_wm2']} W m-2(低于记 near_zero,保留绝对残差)。",
        f"- 敏感性阈值 {list(config['recon_ratio_sensitivity'])} 逐案例见 reconstruction_quality.csv。",
        "",
        "## 逐案例方案属性(前 12 行)",
        "",
        pd.DataFrame(audit_rows).head(12).to_string(index=False) if audit_rows else "(无案例)",
        "",
    ]
    (tables_dir / "reconstruction_audit.md").write_text("\n".join(lines), encoding="utf-8")


def run_task4(config: dict = CONFIG) -> Path:
    """台风居中结构比较(消费 Task 2 缓存)。"""
    root = Path(paths(config)["output_root"])
    dirs = _ensure_dirs(root)
    cache_dir = dirs["cache"] / "case"
    center_rows, storm_rows, radial_rows, manifest_rows = [], [], [], []
    for method in config["methods"]:
        for member in config["members"]:
            for hour in _hours(config):
                cid = case_id(method, member, hour)
                path = cache_dir / f"case_{cid}.nc"
                if not path.exists():
                    manifest_rows.append({"case_id": cid, "task": "task4", "status": "cache_missing"})
                    continue
                print(f"[task4] {cid}", flush=True)
                case = load_case_cache(path)
                storm, metrics, radial = compare_storm_relative(case, config)
                save_storm_cache(storm, dirs["cache"] / "storm" / f"storm_{cid}.nc")
                manifest_rows.append({"case_id": cid, "task": "task4", "status": "ok",
                                      "cache": str(dirs["cache"] / "storm" / f"storm_{cid}.nc")})
                center_rows.append(center_diagnostics_row(case, config))
                storm_rows.append({"method": method, "member": member, "lead_hour": hour,
                                   "valid_time": stamp_from_lead(hour), **metrics})
                for row in radial:
                    radial_rows.append({"method": method, "member": member, "lead_hour": hour, **row})
    pd.DataFrame(center_rows).to_csv(dirs["tables"] / "center_diagnostics.csv", index=False)
    pd.DataFrame(storm_rows).to_csv(dirs["tables"] / "storm_relative_metrics.csv", index=False)
    pd.DataFrame(radial_rows).to_csv(dirs["tables"] / "radial_profiles.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(dirs["tables"] / "task4_manifest.csv", index=False)
    print(f"[task4] done -> {dirs['tables']}", flush=True)
    return root


def main() -> None:
    config = dict(CONFIG)
    run_task2(config)
    run_task3(config)
    run_task4(config)
    run_initial_check(config)


if __name__ == "__main__":
    main()
