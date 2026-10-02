"""verify_diag 公共模块:配置、纯数组诊断、NR 配准、汇总与绘图工具。

本模块供三个诊断入口脚本(verify_01/02/03)复用,设计原则:

1. "数据读取与配准 / 纯数组诊断计算 / 结果保存与绘图" 三层分离;
   纯数组诊断函数(paired_error_metrics、flux_error_budget、classify_change_categories)
   直接接收 numpy 数组,可脱离任何数据文件单独测试。
2. 默认合成模式(SYNTHETIC)。真实模式必须由用户显式修改脚本配置区,并在
   RealPathConfig 中将 acknowledge_real_mode 置 True;合成模式绝不探测真实路径。
3. 导入本模块无任何副作用:不读文件、不建目录、不画图。

符号约定(与用户核验口径一致):
    bias_exp      = mean(X_exp - X_NR)                (偏差)
    MSE_exp       = mean((X_exp - X_NR)^2)
    RMSE_exp      = sqrt(MSE_exp)
    *_change_strong_minus_weak = 强试验减弱试验的误差差值,正值表示强耦合恶化
    rmse_improvement_pct = 100*(RMSE_weak - RMSE_strong)/RMSE_weak,正值表示改善
    dSE(逐点)    = (X_strong-X_NR)^2 - (X_weak-X_NR)^2,正值表示强耦合该点恶化
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 常量
# ------------------------------------------------------------------

#: WRF unpack 后 |value| >= 1e30 视为缺测(与 omtmp_skill_extract.py 一致)
FILL_THRESHOLD = 1.0e30

#: 主结果权重方式(输出 CSV 中 weight_method 列的取值)
WEIGHT_LABEL = "equal_weight_valid_points"

#: 变量单位表
VARIABLE_UNITS = {"om": "K", "tsk": "K", "hfx": "W m-2", "lh": "W m-2"}

#: 变量 -> wrfout 变量名
VARIABLE_SOURCE_NAMES = {"om": "OM_TMP", "tsk": "TSK", "hfx": "HFX", "lh": "LH"}

#: 逐点分类标签(诊断一空间对应关系)
CATEGORY_LABELS = {
    -1: "one_side_unchanged",
    0: "both_unchanged",
    1: "sst_improve_flux_improve",
    2: "sst_improve_flux_worsen",
    3: "sst_worsen_flux_improve",
    4: "sst_worsen_flux_worsen",
}

#: 时间窗口(衔接 analyze_omtmp_skill.py 的 early/late 口径;成员汇总先做窗口内时间平均)
WINDOWS = (
    ("all_0_6h", 0.0, 6.0),
    ("early_0p5_2h", 0.5, 2.0),
    ("late_3_6h", 3.0, 6.0),
)

#: 本目录(Mangkhut_scripts/plot_scripts/verify_diag)的绝对位置;输出默认写在本目录下
VERIFY_DIAG_DIR = Path(__file__).resolve().parent

#: 真实数据路径线索(本次测试禁止访问;仅作为将来真实模式的默认配置记录)
REAL_FORECAST_BASE_DIR = Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test")
REAL_NR_DIR = Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain")


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------


@dataclass(frozen=True)
class RealPathConfig:
    """真实数据模式专用配置。本次合成测试不使用、不探测其中任何路径。

    TODO(待核实):以下每项都需在真实数据上确认后才能启用真实模式,
    详见 MATH_REVIEW_CHECKLIST.md。
    """

    #: 预报根目录;目录结构(由 omtmp_skill_extract.py 推断):
    #: {forecast_base_dir}/{experiment}/{method}/{member}/wrfout_{domain}_{time}
    forecast_base_dir: Path = REAL_FORECAST_BASE_DIR  # TODO(待核实)
    #: NR 目录;文件名同样为 wrfout_{domain}_{time}
    nr_dir: Path = REAL_NR_DIR  # TODO(待核实)
    #: NR -> 试验网格配准方法:
    #:   "ll_to_xy_bilinear" 复用 omtmp_skill_extract.py 的 wrf.ll_to_xy + scipy 双线性
    #:   "numpy_nearest"     纯 numpy 最近邻(无需 wrf 包)
    nr_registration: str = "ll_to_xy_bilinear"  # TODO(待核实)
    #: NR 时间匹配:"exact" 要求 NR 有半小时输出;"bracket_interpolate" 按相邻整点线性
    #: 插值(omtmp_skill_extract.py 的 NR_TIME_BRACKETS 口径)。NR 是否有半小时输出待核实。
    nr_time_strategy: str = "bracket_interpolate"  # TODO(待核实)
    #: 防呆开关:真实模式要求用户显式改为 True
    acknowledge_real_mode: bool = False


@dataclass(frozen=True)
class VerifyConfig:
    """三个诊断共用的顶层配置。所有可编辑参数集中在各脚本顶部。"""

    #: "synthetic"(默认)或 "real"。合成模式不探测任何真实路径。
    mode: str = "synthetic"
    domain: str = "d02"
    strong_experiment: str = "6mem_oceanAssim1Run1"
    weak_experiment: str = "6mem_oceanAssim0Run1"
    methods: tuple[str, ...] = ("EAKF", "QCF_RHF")
    #: 六个配对预报成员(非同化集合规模;两方法使用对应成员,不合并为独立样本)
    members: tuple[str, ...] = ("006", "015", "029", "037", "043", "044")
    #: 0-6 h 半小时一次共 13 个时刻。
    #: TODO(待核实):已有 omtmp_skill_extract.py 仅使用其中 8 个时次;
    #: 真实模式按此 13 个时次逐一检查文件是否存在,缺失记入 run_status。
    times: tuple[tuple[float, str], ...] = (
        (0.0, "2018-09-10_00:00:00"),
        (0.5, "2018-09-10_00:30:00"),
        (1.0, "2018-09-10_01:00:00"),
        (1.5, "2018-09-10_01:30:00"),
        (2.0, "2018-09-10_02:00:00"),
        (2.5, "2018-09-10_02:30:00"),
        (3.0, "2018-09-10_03:00:00"),
        (3.5, "2018-09-10_03:30:00"),
        (4.0, "2018-09-10_04:00:00"),
        (4.5, "2018-09-10_04:30:00"),
        (5.0, "2018-09-10_05:00:00"),
        (5.5, "2018-09-10_05:30:00"),
        (6.0, "2018-09-10_06:00:00"),
    )
    #: 距离环带 (lo_km, hi_km, 名称);区间半开 [lo, hi),最后一段闭 [lo, hi],
    #: 边界不重复计数。union 为完整 0-300 km。
    annulus_edges_km: tuple[tuple[float, float, str], ...] = (
        (0.0, 75.0, "r000_075"),
        (75.0, 150.0, "r075_150"),
        (150.0, 300.0, "r150_300"),
    )
    union_region: str = "r000_300"
    max_radius_km: float = 300.0
    #: NR 台风中心搜索框(与 omtmp_skill_extract.py 一致)
    search_lat: tuple[float, float] = (10.0, 25.0)
    search_lon: tuple[float, float] = (135.0, 155.0)
    #: OM_TMP 海洋表层层数索引(取第 0 层;TODO(待核实):维度顺序 time/ocean/y/x)
    surface_ocean_layer: int = 0
    #: 判定容差(可调,非物理常数)。注意:分类直接作用于逐点 dSE
    #: (dSE = 2e·dF + dF²),容差单位为 dSE 的单位;由于交叉项 2e·dF 的存在,
    #: dSE 容差不能开方解释为普适的海温/通量增量阈值。
    #: 海温侧(tsk 单位 K)dSE 单位 K^2;通量侧 dSE 单位 (W m-2)^2。
    sst_se_tol: float = 1.0e-4  # K^2
    flux_se_tol: float = 1.0  # (W m-2)^2
    #: 输出根目录;各脚本在其下建 SYNTHETIC_<name> / REAL_<name> 子目录
    output_root: Path = VERIFY_DIAG_DIR / "outputs"
    output_dirname: str = ""
    #: 绘图区域(时间序列图只画这些区域,减少文件数)
    plot_regions: tuple[str, ...] = ("r000_300",)
    real: RealPathConfig = field(default_factory=RealPathConfig)

    def out_dir(self, script_name: str) -> Path:
        name = self.output_dirname or script_name
        return self.output_root / f"{self.mode.upper()}_{name}"


# ------------------------------------------------------------------
# 模式防呆
# ------------------------------------------------------------------


def check_mode(config: VerifyConfig) -> None:
    """合成模式直接通过;真实模式必须显式确认。"""
    if config.mode == "synthetic":
        return
    if config.mode == "real" and config.real.acknowledge_real_mode:
        return
    raise RuntimeError(
        "真实模式被拒绝:请在脚本配置区将 mode 改为 'real',并将 "
        "RealPathConfig.acknowledge_real_mode 置为 True,同时逐项核对 TODO(待核实) 配置。"
    )


# ------------------------------------------------------------------
# 掩膜与区域
# ------------------------------------------------------------------


def finite_mask(*arrays) -> np.ndarray:
    """所有数组均有限且 |x| < FILL_THRESHOLD 的联合掩膜(2-D)。"""
    result = None
    for array in arrays:
        array = np.asarray(array, dtype=float)
        good = np.isfinite(array) & (np.abs(array) < FILL_THRESHOLD)
        result = good if result is None else (result & good)
    if result is None:
        raise ValueError("finite_mask requires at least one array")
    return result


def common_valid_mask(*arrays, base_mask=None) -> tuple[np.ndarray, int]:
    """参与该项比较的所有数组的共同有效值掩膜;base_mask(如区域&海洋)先行约束。"""
    mask = finite_mask(*arrays)
    if base_mask is not None:
        mask = mask & np.asarray(base_mask, dtype=bool)
    return mask, int(np.count_nonzero(mask))


def region_masks(distance_km, config: VerifyConfig) -> dict[str, np.ndarray]:
    """由距 NR 中心距离构造环带掩膜。

    区间为半开 [lo, hi),最后一段闭 [lo, hi],边界点只归入较高的环带,
    不重复计数;另给完整 0-300 km 并集。超出 max_radius 的点不属于任何区域。
    """
    distance = np.asarray(distance_km, dtype=float)
    masks: dict[str, np.ndarray] = {}
    edges = config.annulus_edges_km
    for index, (lo, hi, name) in enumerate(edges):
        upper_closed = index == len(edges) - 1
        masks[name] = (distance >= lo) & ((distance <= hi) if upper_closed else (distance < hi))
        # 相邻环带边界不重复计数
        if index > 0:
            prev_lo, prev_hi, prev_name = edges[index - 1]
            if lo != prev_hi:
                raise ValueError(f"annulus edges not contiguous: {prev_hi} != {lo}")
            overlap = masks[name] & masks[prev_name]
            if np.any(overlap):
                raise ValueError(f"annulus overlap between {prev_name} and {name}")
    union_lo = edges[0][0]
    masks[config.union_region] = (distance >= union_lo) & (distance <= config.max_radius_km)
    for name, mask in masks.items():
        if name == config.union_region:
            continue
        if np.any(mask & ~masks[config.union_region]):
            raise ValueError(f"annulus {name} not contained in {config.union_region}")
    return masks


def window_of(hour: float) -> tuple[str, ...]:
    """返回该预报时效所属的所有时间窗口名(按 WINDOWS 定义)。"""
    return tuple(name for name, lo, hi in WINDOWS if lo <= hour <= hi)


# ------------------------------------------------------------------
# 台风中心与距离
# ------------------------------------------------------------------


def haversine_km(lat, lon, center_lat, center_lon):
    """相对单一中心的大圆距离,km(与 omtmp_skill_extract.haversine_km 一致)。"""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dlat = np.deg2rad(lat - center_lat)
    dlon = np.deg2rad(lon - center_lon)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.deg2rad(center_lat))
        * np.cos(np.deg2rad(lat))
        * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * 6371.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def find_storm_center(
    psfc, lat, lon, landmask, search_lat, search_lon
) -> tuple[float, float, int, int]:
    """在海面气压最低的海洋格点处取 NR 台风中心(omtmp_skill_extract._storm_center 口径)。

    返回 (center_lat, center_lon, iy, ix);找不到有限海洋点时抛 RuntimeError,
    由调用方记入 run_status,不静默跳过。
    """
    psfc = np.asarray(psfc, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    landmask = np.asarray(landmask, dtype=float)
    search = (
        (landmask < 0.5)
        & (lat >= search_lat[0])
        & (lat <= search_lat[1])
        & (lon >= search_lon[0])
        & (lon <= search_lon[1])
        & np.isfinite(psfc)
        & (np.abs(psfc) < FILL_THRESHOLD)
    )
    work = np.where(search, psfc, np.inf)
    flat = int(np.argmin(work))
    iy, ix = np.unravel_index(flat, work.shape)
    if not np.isfinite(work[iy, ix]):
        raise RuntimeError("no finite ocean PSFC point found inside the storm search box")
    return float(lat[iy, ix]), float(lon[iy, ix]), int(iy), int(ix)


# ------------------------------------------------------------------
# NR -> 试验网格配准(纯 numpy 两条路径;ll_to_xy 路径在 RealWrfProvider 内)
# ------------------------------------------------------------------


def _extract_regular_axis(coord2d: np.ndarray, vary_along: int, name: str) -> tuple[np.ndarray, bool]:
    """从二维坐标场提取一维规则轴;要求另一维上坐标恒定且严格单调。

    返回 (升序一维轴, 是否发生了反转);调用方必须据反转标志同步翻转数据场,
    否则降序坐标会得到错误插值。
    """
    coord2d = np.asarray(coord2d, dtype=float)
    if vary_along == 1:  # 坐标沿第 1 维(列)变化,如经度
        line = coord2d[0, :]
        constant_ok = np.allclose(coord2d, line[None, :], atol=1.0e-9)
    else:
        line = coord2d[:, 0]
        constant_ok = np.allclose(coord2d, line[:, None], atol=1.0e-9)
    if not constant_ok:
        raise ValueError(
            f"numpy_bilinear registration requires a regular {name} axis; "
            "use the ll_to_xy_bilinear or numpy_nearest method for curvilinear grids"
        )
    if not (np.all(np.diff(line) > 0.0) or np.all(np.diff(line) < 0.0)):
        raise ValueError(f"{name} axis is not strictly monotonic")
    reversed_axis = bool(line[0] > line[-1])
    if reversed_axis:
        line = line[::-1].copy()
    return line, reversed_axis


def resample_bilinear_regular(field, src_lat2d, src_lon2d, dst_lat2d, dst_lon2d) -> np.ndarray:
    """规则经纬网格上的双线性重采样(合成模式默认;线性场可精确复原)。

    目标点落在源网格外时填 NaN;纬度/经度任一为降序时同步翻转数据场。
    曲线网格请使用 ll_to_xy_bilinear / numpy_nearest。
    """
    field = np.asarray(field, dtype=float)
    lat_axis, lat_reversed = _extract_regular_axis(src_lat2d, vary_along=0, name="latitude")
    lon_axis, lon_reversed = _extract_regular_axis(src_lon2d, vary_along=1, name="longitude")
    if lat_reversed:
        field = field[::-1, :]
    if lon_reversed:
        field = field[:, ::-1]

    dst_lat = np.asarray(dst_lat2d, dtype=float)
    dst_lon = np.asarray(dst_lon2d, dtype=float)

    def float_index(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
        position = np.interp(values.ravel(), axis, np.arange(axis.size))
        return position.reshape(values.shape)

    fy = float_index(dst_lat, lat_axis)
    fx = float_index(dst_lon, lon_axis)
    inside = (
        (dst_lat >= lat_axis[0])
        & (dst_lat <= lat_axis[-1])
        & (dst_lon >= lon_axis[0])
        & (dst_lon <= lon_axis[-1])
    )
    fy = np.clip(fy, 0.0, lat_axis.size - 1.0)
    fx = np.clip(fx, 0.0, lon_axis.size - 1.0)
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    y1 = np.minimum(y0 + 1, lat_axis.size - 1)
    x1 = np.minimum(x0 + 1, lon_axis.size - 1)
    wy = fy - y0
    wx = fx - x0
    out = (
        field[y0, x0] * (1.0 - wy) * (1.0 - wx)
        + field[y1, x0] * wy * (1.0 - wx)
        + field[y0, x1] * (1.0 - wy) * wx
        + field[y1, x1] * wy * wx
    )
    return np.where(inside, out, np.nan)


def estimate_source_spacing_km(src_lat2d, src_lon2d, sample: int = 1024, seed: int = 0) -> float:
    """估计源网格相邻点间距(km):抽样格点与其索引相邻格点 (i±1,j)/(i,j±1) 的
    真实球面距离的中位数。

    相邻索引对就是结构化网格的真实相邻关系(规则与曲线网格均适用),
    避免了"抽样点间最近距离随网格变大而被高估"的偏差。
    """
    lat2d = np.asarray(src_lat2d, dtype=float)
    lon2d = np.asarray(src_lon2d, dtype=float)
    ny, nx = lat2d.shape
    if ny * nx < 2:
        raise ValueError("need at least two source points to estimate spacing")
    rng = np.random.default_rng(seed)
    offsets = []
    if ny > 1:
        offsets.append((1, 0))
    if nx > 1:
        offsets.append((0, 1))
    if not offsets:
        raise ValueError("degenerate grid: no adjacent index pairs")
    i_indices = rng.integers(0, ny, size=min(sample, ny * nx))
    j_indices = rng.integers(0, nx, size=i_indices.size)
    distances: list[float] = []
    for di, dj in offsets:
        src_i = np.clip(i_indices, 0, ny - 1 - di)
        src_j = np.clip(j_indices, 0, nx - 1 - dj)
        dst_i = src_i + di
        dst_j = src_j + dj
        lat1 = np.deg2rad(lat2d[src_i, src_j])
        lon1 = np.deg2rad(lon2d[src_i, src_j])
        lat2 = np.deg2rad(lat2d[dst_i, dst_j])
        lon2 = np.deg2rad(lon2d[dst_i, dst_j])
        dist_km = 6371.0 * 2.0 * np.arcsin(
            np.clip(
                np.sqrt(
                    np.sin(0.5 * (lat2 - lat1)) ** 2
                    + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * (lon2 - lon1)) ** 2
                ),
                0.0,
                1.0,
            )
        )
        finite = dist_km[np.isfinite(dist_km) & (dist_km > 0.0)]
        distances.append(finite)
    pooled = np.concatenate([d for d in distances if d.size]) if distances else np.array([])
    if pooled.size == 0:
        raise ValueError("could not estimate source grid spacing")
    return float(np.median(pooled))


def _sphere_coords(lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """经纬度(弧度)映射到单位球三维坐标,供 cKDTree 做球面最近邻。"""
    cos_lat = np.cos(lat_rad)
    return np.stack(
        [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)],
        axis=-1,
    )


#: 源网格空间索引缓存(几何固定时复用;键为网格指纹,容量很小)
_SOURCE_INDEX_CACHE: dict = {}
_SOURCE_INDEX_CACHE_MAX = 4


def _grid_fingerprint(lat2d: np.ndarray, lon2d: np.ndarray):
    lat = np.asarray(lat2d, dtype=float)
    lon = np.asarray(lon2d, dtype=float)
    flat_lat = lat.ravel()
    flat_lon = lon.ravel()
    middle = flat_lat.size // 2
    # 加权求和作为弱校验和:即使角点/中点相同、内部不同的网格也不会碰撞
    weights = np.arange(1.0, flat_lat.size + 1.0)
    lat_checksum = float(np.sum(flat_lat * weights))
    lon_checksum = float(np.sum(flat_lon * weights))
    return (
        lat.shape,
        float(flat_lat[0]), float(flat_lat[-1]),
        float(flat_lon[0]), float(flat_lon[-1]),
        float(flat_lat[middle]), float(flat_lon[middle]),
        lat_checksum, lon_checksum,
    )


def _build_source_footprint(lat2d: np.ndarray, lon2d: np.ndarray, tol_deg: float):
    """源网格有效足迹:规则网格用包围盒,曲线网格用凸包(均带半个格距容差)。"""
    lat2d = np.asarray(lat2d, dtype=float)
    lon2d = np.asarray(lon2d, dtype=float)
    try:
        lat_axis, _ = _extract_regular_axis(lat2d, vary_along=0, name="latitude")
        lon_axis, _ = _extract_regular_axis(lon2d, vary_along=1, name="longitude")
        return (
            "bbox",
            float(lat_axis[0]), float(lat_axis[-1]),
            float(lon_axis[0]), float(lon_axis[-1]),
            tol_deg,
        )
    except ValueError:
        pass
    points = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    points = points[np.isfinite(points).all(axis=1)]
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        # hull.equations: [a, b, c] 满足 a*x + b*y + c <= 0 为内部
        return ("hull", np.asarray(hull.equations, dtype=float), tol_deg)
    except Exception:
        return (
            "bbox",
            float(np.nanmin(lat2d)), float(np.nanmax(lat2d)),
            float(np.nanmin(lon2d)), float(np.nanmax(lon2d)),
            tol_deg,
        )


def _inside_footprint(footprint, dst_lat_deg: np.ndarray, dst_lon_deg: np.ndarray) -> np.ndarray:
    kind = footprint[0]
    if kind == "bbox":
        _, lat_min, lat_max, lon_min, lon_max, tol = footprint
        return (
            (dst_lat_deg >= lat_min - tol)
            & (dst_lat_deg <= lat_max + tol)
            & (dst_lon_deg >= lon_min - tol)
            & (dst_lon_deg <= lon_max + tol)
        )
    _, equations, tol = footprint
    if dst_lat_deg.size == 0:
        return np.zeros(dst_lat_deg.shape, dtype=bool)
    homogeneous = np.column_stack(
        [dst_lat_deg, dst_lon_deg, np.ones(dst_lat_deg.size)]
    )
    signed = homogeneous @ equations.T  # (n_dst, n_faces)
    # 逐点布尔:与查询点一一对应(单点/多点/零点均成立);
    # 不能对整个数组调用 bool(...)(多点时抛 ValueError)
    inside_flat = np.all(signed <= tol, axis=1)
    return np.asarray(inside_flat, dtype=bool).reshape(dst_lat_deg.shape)


def _grids_identical(lat_a, lon_a, lat_b, lon_b, atol: float = 1.0e-12) -> bool:
    """完整几何一致性校验:形状与全部经纬度逐点一致。"""
    lat_a = np.asarray(lat_a, dtype=float)
    lon_a = np.asarray(lon_a, dtype=float)
    lat_b = np.asarray(lat_b, dtype=float)
    lon_b = np.asarray(lon_b, dtype=float)
    if lat_a.shape != lat_b.shape or lon_a.shape != lon_b.shape:
        return False
    return bool(
        np.array_equal(lat_a, lat_b, equal_nan=True)
        and np.array_equal(lon_a, lon_b, equal_nan=True)
    )


def _build_source_index(src_lat2d, src_lon2d) -> dict:
    """构建/复用源网格空间索引(cKDTree + 足迹 + 间距),按网格指纹缓存。

    指纹命中后仍做完整经纬度逐点校验(_grids_identical):采样点/加权求和
    不能可靠区分完整网格,校验失败则按碰撞处理并新建条目,
    绝不错误复用其他网格的索引。
    """
    key = _grid_fingerprint(src_lat2d, src_lon2d)
    cached = _SOURCE_INDEX_CACHE.get(key)
    if cached is not None:
        if _grids_identical(
            cached.get("fingerprint_lat"), cached.get("fingerprint_lon"),
            src_lat2d, src_lon2d,
        ):
            return cached
        # 指纹碰撞但几何不同:移除旧条目,按新网格重建
        _SOURCE_INDEX_CACHE.pop(key, None)
    lat2d = np.asarray(src_lat2d, dtype=float)
    lon2d = np.asarray(src_lon2d, dtype=float)
    flat_lat = lat2d.ravel()
    flat_lon = lon2d.ravel()
    good = np.isfinite(flat_lat) & np.isfinite(flat_lon)
    coords = _sphere_coords(
        np.deg2rad(flat_lat[good]), np.deg2rad(flat_lon[good])
    )
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
    except ImportError:
        tree = None  # 无 scipy 时回退到分块全对搜索(见 _resample_nearest_blocked)
    spacing_deg = _spacing_deg_from_index_pairs(lat2d, lon2d)
    entry = {
        "tree": tree,
        "coords": coords,
        "valid_positions": np.flatnonzero(good),
        "spacing_km": estimate_source_spacing_km(lat2d, lon2d),
        "footprint": _build_source_footprint(lat2d, lon2d, 0.5 * spacing_deg),
        "fingerprint_lat": lat2d.copy(),
        "fingerprint_lon": lon2d.copy(),
    }
    if len(_SOURCE_INDEX_CACHE) >= _SOURCE_INDEX_CACHE_MAX:
        _SOURCE_INDEX_CACHE.pop(next(iter(_SOURCE_INDEX_CACHE)))
    _SOURCE_INDEX_CACHE[key] = entry
    return entry


def _spacing_deg_from_index_pairs(lat2d: np.ndarray, lon2d: np.ndarray) -> float:
    """相邻索引对的坐标差中位数(度),用于足迹容差。"""
    lat2d = np.asarray(lat2d, dtype=float)
    lon2d = np.asarray(lon2d, dtype=float)
    diffs = []
    if lat2d.shape[0] > 1:
        diffs.append(np.abs(np.diff(lat2d, axis=0)).ravel())
    if lat2d.shape[1] > 1:
        diffs.append(np.abs(np.diff(lon2d, axis=1)).ravel())
    pooled = np.concatenate([d[np.isfinite(d)] for d in diffs]) if diffs else np.array([])
    if pooled.size == 0:
        return 0.0
    return float(np.median(pooled))


def _resample_nearest_blocked(
    field,
    src_lat2d,
    src_lon2d,
    dst_lat2d,
    dst_lon2d,
    max_radius_km,
    entry: dict,
    dst_chunk: int = 4096,
    src_tile: int = 4096,
) -> np.ndarray:
    """无 scipy 时的回退实现:双向分块全对搜索(内存有界,
    复杂度 O(N_source*N_target);大网格请安装 scipy 走空间索引)。"""
    src_lat = np.deg2rad(np.asarray(src_lat2d, dtype=float).ravel())
    src_lon = np.deg2rad(np.asarray(src_lon2d, dtype=float).ravel())
    dst_lat = np.deg2rad(np.asarray(dst_lat2d, dtype=float).ravel())
    dst_lon = np.deg2rad(np.asarray(dst_lon2d, dtype=float).ravel())
    n_src, n_dst = src_lat.size, dst_lat.size
    flat_src = np.asarray(field, dtype=float).ravel()

    best_dist2 = np.full(n_dst, np.inf)
    best_index = np.full(n_dst, -1, dtype=np.int64)
    for d_start in range(0, n_dst, dst_chunk):
        d_stop = min(d_start + dst_chunk, n_dst)
        block_lat = dst_lat[d_start:d_stop]
        block_lon = dst_lon[d_start:d_stop]
        for s_start in range(0, n_src, src_tile):
            s_stop = min(s_start + src_tile, n_src)
            tile_lat = src_lat[s_start:s_stop][:, None]
            tile_lon = src_lon[s_start:s_stop][:, None]
            dy = tile_lat - block_lat[None, :]
            dx = tile_lon - block_lon[None, :]
            cos_lat = np.cos(0.5 * (tile_lat + block_lat[None, :]))
            dist2 = dy * dy + cos_lat * cos_lat * dx * dx
            local = np.argmin(dist2, axis=0)
            local_best = np.take_along_axis(dist2, local[None, :], axis=0)[0]
            better = local_best < best_dist2[d_start:d_stop]
            best_dist2[d_start:d_stop][better] = local_best[better]
            best_index[d_start:d_stop][better] = local[better] + s_start

    out = np.full(n_dst, np.nan)
    found = best_index >= 0
    out[found] = flat_src[best_index[found]]
    if max_radius_km is not None:
        too_far = found & (
            6371.0 * np.sqrt(np.clip(best_dist2, 0.0, None)) > float(max_radius_km)
        )
        out[too_far] = np.nan
    return out.reshape(np.asarray(dst_lat2d).shape)


def resample_nearest(
    field,
    src_lat2d,
    src_lon2d,
    dst_lat2d,
    dst_lon2d,
    max_radius_km=None,
) -> np.ndarray:
    """最近邻重采样:可复用空间索引(cKDTree,O(N log N))+ 源网格足迹覆盖判断。

    - 覆盖判断 = 足迹内(规则网格包围盒/曲线网格凸包,容差半个格距)
      且(可选)距最近源点不超过 max_radius_km;"贴着边界但在域外"的
      目标点会被足迹判断排除,不会套用边缘最近邻。
    - 索引按源网格指纹缓存(几何固定的网格跨时次复用)。
    - 无 scipy 时回退到分块全对搜索(内存有界但复杂度 O(Ns*Nt),仅小网格可用)。
    - 源点值为 NaN 时按原样返回该 NaN(由调用方掩膜处理)。
    """
    field = np.asarray(field, dtype=float)
    dst_lat_deg = np.asarray(dst_lat2d, dtype=float)
    dst_lon_deg = np.asarray(dst_lon2d, dtype=float)
    flat_src = field.ravel()
    entry = _build_source_index(src_lat2d, src_lon2d)

    dst_lat_rad = np.deg2rad(dst_lat_deg.ravel())
    dst_lon_rad = np.deg2rad(dst_lon_deg.ravel())
    dst_coords = _sphere_coords(dst_lat_rad, dst_lon_rad)
    n_dst = dst_coords.shape[0]
    out = np.full(n_dst, np.nan)

    cutoff_chord = None
    if max_radius_km is not None:
        theta = float(max_radius_km) / 6371.0
        cutoff_chord = 2.0 * np.sin(min(theta, np.pi) / 2.0)

    if entry["tree"] is not None:
        bound = cutoff_chord if cutoff_chord is not None else np.inf
        dist, idx = entry["tree"].query(
            dst_coords, k=1, distance_upper_bound=bound
        )
        found = np.isfinite(dist) & (idx < entry["coords"].shape[0])
        out[found] = flat_src[entry["valid_positions"][idx[found]]]
    else:
        out = _resample_nearest_blocked(
            field, src_lat2d, src_lon2d, dst_lat_deg, dst_lon_deg,
            max_radius_km, entry,
        ).ravel()

    inside = _inside_footprint(entry["footprint"], dst_lat_deg.ravel(), dst_lon_deg.ravel())
    out[~inside] = np.nan
    return out.reshape(dst_lat_deg.shape)


def register_nr_fields(
    nr_fields: dict,
    nr_static: dict,
    exp_static: dict,
    config: VerifyConfig,
    provider=None,
) -> dict:
    """三个诊断共用的 NR→试验网格配准单一入口(策略一致,杜绝各脚本各自实现)。

    - 掩膜类字段(landmask):真实模式默认 ll_to_xy 投影索引 order=0
      (provider.register_mask_lltoxy,不做全点对搜索);否则用带足迹覆盖与
      覆盖半径的空间索引最近邻。两种方式下域外目标均为 NaN。
    - 物理场:真实模式默认 "ll_to_xy_bilinear"(provider 提供投影插值),
      可选 "numpy_nearest";合成模式用规则网格双线性
      (real.nr_registration 为 numpy_nearest 时同样走最近邻;
      ll_to_xy 名称仅对真实 provider 有意义)。
    覆盖半径 = 2×源网格相邻间距估计(estimate_source_spacing_km)。
    """
    exp_lat, exp_lon = exp_static["lat"], exp_static["lon"]
    nr_lat, nr_lon = nr_static["lat"], nr_static["lon"]
    cutoff_km = estimate_source_spacing_km(nr_lat, nr_lon) * 2.0
    use_lltoxy = (
        config.mode == "real"
        and config.real.nr_registration == "ll_to_xy_bilinear"
    )
    out: dict = {}
    for name, field in nr_fields.items():
        if name in ("landmask",):
            if use_lltoxy and provider is not None and hasattr(provider, "register_mask_lltoxy"):
                out[name] = provider.register_mask_lltoxy(name, field, exp_lat, exp_lon)
            else:
                out[name] = resample_nearest(
                    field, nr_lat, nr_lon, exp_lat, exp_lon, max_radius_km=cutoff_km
                )
            continue
        registration = config.real.nr_registration
        if config.mode == "real" and registration == "ll_to_xy_bilinear":
            if provider is None or not hasattr(provider, "register_field_lltoxy"):
                raise RuntimeError(
                    "ll_to_xy_bilinear 配准需要 RealWrfProvider;"
                    "合成模式请使用规则网格配准(默认配置)"
                )
            out[name] = provider.register_field_lltoxy(name, field, exp_lat, exp_lon)
        elif registration == "numpy_nearest":
            out[name] = resample_nearest(
                field, nr_lat, nr_lon, exp_lat, exp_lon, max_radius_km=cutoff_km
            )
        else:
            out[name] = resample_bilinear_regular(field, nr_lat, nr_lon, exp_lat, exp_lon)
    return out


def build_time_context(provider, config: VerifyConfig, time_hour: float) -> dict:
    """单时次配准上下文(诊断一/二/三共用,消除各脚本的重复实现)。

    每个时刻所有方法/成员共用同一 NR 中心、验证格点、区域与海洋掩膜:
    NR 中心取 NR PSFC 海面最低点;NR 场按 register_nr_fields 配准;
    海洋掩膜 = 试验 LANDMASK<0.5 且 NR(最近邻)LANDMASK<0.5。
    """
    nr_static = provider.nr_static()
    nr_fields = provider.nr_fields(time_hour)
    center_lat, center_lon, _, _ = find_storm_center(
        nr_fields["psfc"], nr_static["lat"], nr_static["lon"], nr_static["landmask"],
        config.search_lat, config.search_lon,
    )
    exp_static = provider.static_fields()
    distance = haversine_km(exp_static["lat"], exp_static["lon"], center_lat, center_lon)
    regions = region_masks(distance, config)
    registered = register_nr_fields(
        {name: nr_fields[name] for name in ("om", "tsk", "hfx", "lh")},
        nr_static, exp_static, config, provider,
    )
    nr_land_on_exp = register_nr_fields(
        {"landmask": nr_static["landmask"]}, nr_static, exp_static, config, provider
    )["landmask"]
    ocean = (exp_static["landmask"] < 0.5) & (nr_land_on_exp < 0.5)
    return {
        "center": (center_lat, center_lon),
        "distance": distance,
        "regions": regions,
        "ocean": ocean,
        "nr": registered,
        "static": exp_static,
    }


def strong_minus_weak(strong, weak) -> np.ndarray:
    """强试验减弱试验的逐点差值(正值 = 强试验更大)。

    统一符号入口:所有"strong_minus_weak"语义的差值必须经过本函数,
    防止 (strong, weak) 元组下标误用导致的方向反转。
    """
    return np.asarray(strong, dtype=float) - np.asarray(weak, dtype=float)


def align_times_to_hours(frame: pd.DataFrame, hours) -> pd.DataFrame:
    """把逐时刻序列对齐到配置时刻集合;缺失时刻补 NaN(绘图断线,不跨接)。"""
    if frame.empty:
        return pd.DataFrame({"time_hour": list(hours)}).set_index("time_hour")
    indexed = frame.set_index("time_hour")
    return indexed.reindex(pd.Index(list(hours), name="time_hour"))


def member_then_mean(frame: pd.DataFrame, column: str, members) -> float:
    """先成员内对时间等权平均,再跨成员等权平均(缺失时次不改变成员权重)。"""
    per_member = []
    for member in members:
        member_values = frame[frame.member == member][column].to_numpy(dtype=float)
        finite = member_values[np.isfinite(member_values)]
        if finite.size:
            per_member.append(float(np.mean(finite)))
    return float(np.mean(per_member)) if per_member else np.nan

# ------------------------------------------------------------------
# 条件子集与离线响应-实际变化联合诊断(诊断二/三新增)
# ------------------------------------------------------------------

#: 海温改善子集名称(与 dSE_T 阈值判定对应)
SUBSET_NAMES = ("all_common", "sst_improved", "sst_worsened", "sst_unchanged")


def sst_subset_masks(
    dse_sst: np.ndarray, base_mask: np.ndarray, tol: float
) -> dict[str, np.ndarray]:
    """按海温误差变化构造条件子集掩膜(诊断二/三共用)。

    dSE_T = (T_s - T_NR)^2 - (T_w - T_NR)^2(正值 = 强耦合该点海温误差更大)。
    子集只依据海温侧信息筛选,不同时要求通量变化:
        all_common     base_mask & dSE_T 有限
        sst_improved   dSE_T < -tol
        sst_worsened   dSE_T > +tol
        sst_unchanged  |dSE_T| <= tol
    改善 + 恶化 + 未变化 = all_common(容差只用于分类,不改数值)。
    """
    dse = np.asarray(dse_sst, dtype=float)
    base = np.asarray(base_mask, dtype=bool)
    all_common = base & np.isfinite(dse)
    return {
        "all_common": all_common,
        "sst_improved": all_common & (dse < -float(tol)),
        "sst_worsened": all_common & (dse > float(tol)),
        "sst_unchanged": all_common & (np.abs(dse) <= float(tol)),
    }


def cross_term_stats(error: np.ndarray, increment: np.ndarray, mask: np.ndarray) -> dict:
    """统一掩膜上的实际通量误差预算分量(诊断二/三共用口径)。

    e = F_weak_model - F_NR(实际弱误差;不使用离线重建弱误差)
    dF = F_strong_model - F_weak_model(实际强弱增量)
    C = 2<e*dF>(逐点乘积的平均,不是 <e><dF>,也不是协方差)
    S = <dF^2>
    delta_mse_direct = MSE_strong - MSE_weak = C + S
    closure_residual = delta_mse_direct - (C + S)(浮点意义上的恒等式残差)

    空掩膜返回 NaN + empty_mask,不填零。
    """
    error = np.asarray(error, dtype=float)
    increment = np.asarray(increment, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n = int(np.count_nonzero(mask))
    nan_result = {
        "n_valid": 0, "status": "empty_mask",
        "mse_weak": np.nan, "mse_strong": np.nan,
        "rmse_weak": np.nan, "rmse_strong": np.nan,
        "delta_mse_direct": np.nan, "cross_term": np.nan,
        "increment_square_term": np.nan, "closure_residual": np.nan,
    }
    if n == 0:
        return nan_result
    e = error[mask]
    d = increment[mask]
    good = np.isfinite(e) & np.isfinite(d)
    if not np.any(good):
        # 契约:n_valid = 实际参与统计的有效点数;全 NaN 时为 0,
        # 失败原因保留在 status,不把掩膜点数冒充有效点数
        return {**nan_result, "status": "all_nan_inside_mask"}
    mse_weak = float(np.mean(e[good] ** 2))
    mse_strong = float(np.mean((e[good] + d[good]) ** 2))
    cross = float(np.mean(2.0 * e[good] * d[good]))
    square = float(np.mean(d[good] ** 2))
    delta_direct = mse_strong - mse_weak
    return {
        "n_valid": int(np.count_nonzero(good)),
        "status": "ok",
        "mse_weak": mse_weak,
        "mse_strong": mse_strong,
        "rmse_weak": float(np.sqrt(mse_weak)),
        "rmse_strong": float(np.sqrt(mse_strong)),
        "delta_mse_direct": delta_direct,
        "cross_term": cross,
        "increment_square_term": square,
        "closure_residual": delta_direct - (cross + square),
    }


def cross_term_direction(cross_term: float, tol: float) -> str:
    """交叉项方向状态(独立容差,单位 = 变量单位^2;容差只影响分类)。

    C > +tol  "positive_cross_term"        (区域整体变化方向不利)
    C < -tol  "negative_cross_term"        (存在纠错倾向)
    其余       "cross_term_neutral"        (接近零/不可分辨,不归入方向不利)
    """
    value = float(cross_term) if np.isfinite(cross_term) else np.nan
    if not np.isfinite(value):
        return "cross_term_not_evaluable"
    if value > float(tol):
        return "positive_cross_term"
    if value < -float(tol):
        return "negative_cross_term"
    return "cross_term_neutral"


def net_error_direction(delta_mse: float, tol: float) -> str:
    """净误差变化状态(独立容差,单位 = 变量单位^2)。"""
    value = float(delta_mse) if np.isfinite(delta_mse) else np.nan
    if not np.isfinite(value):
        return "net_not_evaluable"
    if value < -float(tol):
        return "net_improved"
    if value > float(tol):
        return "net_worsened"
    return "net_unchanged"


def rms(values: np.ndarray, mask: np.ndarray) -> float:
    """统一掩膜上的 RMS(掩膜外/非有限点剔除;空集返回 NaN)。"""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean(values[mask] ** 2)))


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """统一掩膜上的等权平均(空集返回 NaN)。"""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(mask):
        return np.nan
    return float(np.mean(values[mask]))


def sign_agreement_stats(
    first_change: np.ndarray,
    second_change: np.ndarray,
    mask: np.ndarray,
    abs_tol: float,
) -> dict:
    """两种变化的符号一致率(仅在双方 |增量| 均 > abs_tol 的格点上计算)。

    返回:n_valid_subset(掩膜内有效点数)、n_eligible(参与点数)、
    n_agree(同号数)、sign_agreement(一致率 = n_agree / n_eligible)、
    eligible_fraction(参与点数 / 掩膜内有效点数)、status。
    阈值单位 = 增量单位(通量侧为 W m-2;与 dSE 的 (W m-2)^2 阈值不同)。
    参与点数为 0 时指标置 NaN 并说明原因(no_points_above_threshold)。
    """
    first = np.asarray(first_change, dtype=float)
    second = np.asarray(second_change, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n_valid = int(np.count_nonzero(mask & np.isfinite(first) & np.isfinite(second)))
    result = {
        "n_valid_subset": n_valid,
        "n_eligible": 0,
        "n_agree": 0,
        "sign_agreement": np.nan,
        "eligible_fraction": np.nan,
        "status": "empty_mask",
    }
    if n_valid == 0:
        return result
    result["status"] = "ok"
    eligible = mask & np.isfinite(first) & np.isfinite(second) & (
        (np.abs(first) > float(abs_tol)) & (np.abs(second) > float(abs_tol))
    )
    n_eligible = int(np.count_nonzero(eligible))
    result["n_eligible"] = n_eligible
    result["eligible_fraction"] = float(n_eligible) / n_valid
    if n_eligible == 0:
        result["status"] = "no_points_above_threshold"
        return result
    agree = np.sign(first[eligible]) == np.sign(second[eligible])
    result["n_agree"] = int(np.count_nonzero(agree))
    result["sign_agreement"] = result["n_agree"] / n_eligible
    return result


def response_vs_actual_stats(
    error_weak_model: np.ndarray,
    d_sst: np.ndarray,
    d_actual: np.ndarray,
    mask: np.ndarray,
    flux_increment_tol: float,
    ratio_denominator_floor: float | None = None,
) -> dict:
    """离线 SST 响应与实际通量变化的联合统计(诊断三新增,单案例单子集)。

    全部指标在同一最终掩膜(mask:区域&海洋&TSK/通量/NR/重建输出共同有限)
    上计算,不允许各指标自行删点:
        cross_model_error_sst_response = 2<e_model * dF_SST>
            (e_model = F_weak_model - F_NR,实际弱误差;
             仅是离线响应相对实际误差方向的诊断,不是因果贡献,
             也不与实际增量平方项相加闭合实际 dMSE);
        rms_sst_response / rms_actual_response;
        rms_sst_minus_actual = sqrt(<(dF_SST - dF_actual)^2>);
        sst_to_actual_rms_ratio = RMS(dF_SST)/RMS(dF_actual)
            (实际响应为零时置 NaN,状态 actual_response_zero);
        sign agreement 两增量一致率(阈值 flux_increment_tol,单位 W m-2);
        实际通量 C/S/dMSE/闭合残差(cross_term_stats,同掩膜)。
    """
    mask = np.asarray(mask, dtype=bool)
    d_sst_arr = np.asarray(d_sst, dtype=float)
    d_actual_arr = np.asarray(d_actual, dtype=float)
    error_arr = np.asarray(error_weak_model, dtype=float)
    # 共同有效掩膜:全部参与数组(e_model、dF_SST、dF_actual)都有限;
    # 不允许各指标自行删点
    joint_mask = mask & np.isfinite(error_arr) & np.isfinite(d_sst_arr) & np.isfinite(d_actual_arr)
    budget = cross_term_stats(error_weak_model, d_actual, joint_mask)
    rms_sst = rms(d_sst, joint_mask)
    rms_actual = rms(d_actual, joint_mask)
    diff = d_sst_arr - d_actual_arr
    # 差值 RMS 必须与全部指标使用同一 joint_mask(P2 #2 反例:
    # e=[1,NaN]、dSST=[1,100]、actual=[1,0] 时,旧掩膜会得到 70.7 而非 0)
    result = {
        "n_valid": budget["n_valid"],
        "status": budget["status"],
        "cross_model_error_sst_response": np.nan,
        "rms_sst_response": rms_sst,
        "rms_actual_response": rms_actual,
        "rms_sst_minus_actual": rms(diff, joint_mask),
        "sst_to_actual_rms_ratio": np.nan,
        "actual_response_status": "ok",
        # 实际通量预算分量统一带 actual_ 前缀(与联合表列名一致)
        "actual_mse_weak": budget["mse_weak"],
        "actual_mse_strong": budget["mse_strong"],
        "actual_rmse_weak": budget["rmse_weak"],
        "actual_rmse_strong": budget["rmse_strong"],
        "actual_delta_mse_direct": budget["delta_mse_direct"],
        "actual_cross_term": budget["cross_term"],
        "actual_increment_square_term": budget["increment_square_term"],
        "actual_closure_residual": budget["closure_residual"],
    }
    if result["status"] != "ok":
        result["actual_response_status"] = result["status"]
        result.update({
            "n_sign_eligible": 0,
            "n_sign_agree": 0,
            "sign_agreement": np.nan,
            "sign_eligible_fraction": np.nan,
            "sign_status": result["status"],
        })
        return result
    e = error_arr[joint_mask]
    d_sst_in = d_sst_arr[joint_mask]
    result["cross_model_error_sst_response"] = float(
        np.mean(2.0 * e * d_sst_in)
    )
    # 比率分母保护:严格零分母 -> actual_response_zero;
    # 若配置了诊断尺度的分母下限,低于下限同样不给比率(状态单列),
    # 避免近零分母的极大比率主导后续平均。下限只影响比率,不改原始 RMS。
    denominator_floor = (
        float(ratio_denominator_floor) if ratio_denominator_floor is not None else 0.0
    )
    if not (np.isfinite(rms_actual) and rms_actual > 0.0):
        result["actual_response_status"] = "actual_response_zero"
    elif np.isfinite(rms_sst) and rms_actual <= denominator_floor:
        result["actual_response_status"] = "actual_response_below_floor"
    else:
        result["sst_to_actual_rms_ratio"] = (
            rms_sst / rms_actual if np.isfinite(rms_sst) else np.nan
        )
    agreement = sign_agreement_stats(d_sst, d_actual, joint_mask, flux_increment_tol)
    result.update({
        "n_sign_eligible": agreement["n_eligible"],
        "n_sign_agree": agreement["n_agree"],
        "sign_agreement": agreement["sign_agreement"],
        "sign_eligible_fraction": agreement["eligible_fraction"],
        "sign_status": agreement["status"],
    })
    return result


def config_snapshot(config: VerifyConfig) -> dict:
    """随结果保存的配置快照(配准方式、时间匹配方式等,JSON 可序列化)。"""
    return {
        "mode": config.mode,
        "domain": config.domain,
        "strong_experiment": config.strong_experiment,
        "weak_experiment": config.weak_experiment,
        "methods": list(config.methods),
        "members": list(config.members),
        "times": [[hour, name] for hour, name in config.times],
        "annulus_edges_km": [[lo, hi, name] for lo, hi, name in config.annulus_edges_km],
        "union_region": config.union_region,
        "max_radius_km": config.max_radius_km,
        "surface_ocean_layer": config.surface_ocean_layer,
        "sst_se_tol_k2": config.sst_se_tol,
        "flux_se_tol_w2": config.flux_se_tol,
        "nr_registration": config.real.nr_registration,
        "nr_time_strategy": config.real.nr_time_strategy,
        "weight_method": WEIGHT_LABEL,
        "fill_threshold": FILL_THRESHOLD,
    }


def save_config_snapshot(out_dir: Path, config: VerifyConfig, name: str = "config_snapshot.json") -> None:
    """把配置快照写入输出目录(供结果表复核配准/时间匹配方式)。"""
    import json

    path = out_dir / name
    path.write_text(
        json.dumps(config_snapshot(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[output] {path}", flush=True)


# ------------------------------------------------------------------
# 纯数组诊断
# ------------------------------------------------------------------


def paired_error_metrics(strong, weak, truth, base_mask=None) -> dict:
    """强弱试验相对 NR 的配对误差指标(等权有效格点平均)。

    参与比较的全部数组(strong/weak/truth)在共同的有限值掩膜上统计,
    掩膜为空时 status="empty_mask" 并返回 NaN(不静默跳过);
    RMSE_weak 为零时分母保护:improvement 置 NaN,status 记 "zero_rmse_weak"。

    返回键:n_valid, weak_bias, strong_bias, mse_weak, mse_strong, rmse_weak,
    rmse_strong, mse_change_strong_minus_weak, rmse_change_strong_minus_weak,
    rmse_improvement_pct, status
    """
    strong = np.asarray(strong, dtype=float)
    weak = np.asarray(weak, dtype=float)
    truth = np.asarray(truth, dtype=float)
    mask, n = common_valid_mask(strong, weak, truth, base_mask=base_mask)

    nan_keys = (
        "weak_bias", "strong_bias", "mse_weak", "mse_strong", "rmse_weak",
        "rmse_strong", "mse_change_strong_minus_weak", "rmse_change_strong_minus_weak",
        "rmse_improvement_pct",
    )
    if n == 0:
        return {"n_valid": 0, "status": "empty_mask", **{key: np.nan for key in nan_keys}}

    err_strong = strong[mask] - truth[mask]
    err_weak = weak[mask] - truth[mask]
    mse_strong = float(np.mean(err_strong**2))
    mse_weak = float(np.mean(err_weak**2))
    rmse_strong = float(np.sqrt(mse_strong))
    rmse_weak = float(np.sqrt(mse_weak))
    status = "ok"
    if rmse_weak > 0.0:
        improvement = 100.0 * (rmse_weak - rmse_strong) / rmse_weak
    else:
        improvement = np.nan
        status = "zero_rmse_weak"
    return {
        "n_valid": n,
        "weak_bias": float(np.mean(err_weak)),
        "strong_bias": float(np.mean(err_strong)),
        "mse_weak": mse_weak,
        "mse_strong": mse_strong,
        "rmse_weak": rmse_weak,
        "rmse_strong": rmse_strong,
        "mse_change_strong_minus_weak": mse_strong - mse_weak,
        "rmse_change_strong_minus_weak": rmse_strong - rmse_weak,
        "rmse_improvement_pct": improvement,
        "status": status,
    }


def se_change(strong, weak, truth) -> np.ndarray:
    """逐点 dSE = (X_strong - X_NR)^2 - (X_weak - X_NR)^2(正值=强耦合该点恶化)。

    任一输入无效的格点返回 NaN,由调用方掩膜处理。
    """
    strong = np.asarray(strong, dtype=float)
    weak = np.asarray(weak, dtype=float)
    truth = np.asarray(truth, dtype=float)
    with np.errstate(invalid="ignore"):
        return (strong - truth) ** 2 - (weak - truth) ** 2


def classify_change_categories(dse_sst, dse_flux, tol_sst, tol_flux) -> np.ndarray:
    """按海温与通量的逐点误差变化分类(诊断一空间对应关系)。

    dSE < -tol 判为改善,> +tol 判为恶化,|dSE| <= tol 判为未变化:
        1 sst_improve_flux_improve   2 sst_improve_flux_worsen
        3 sst_worsen_flux_improve    4 sst_worsen_flux_worsen
        0 both_unchanged            -1 one_side_unchanged(仅一侧未变化,单独处理)
    输入 NaN 的格点返回 -99(须由掩膜剔除,不参与占比)。
    """
    dse_sst = np.asarray(dse_sst, dtype=float)
    dse_flux = np.asarray(dse_flux, dtype=float)
    category = np.full(dse_sst.shape, -99, dtype=int)

    def side(values: np.ndarray, tol: float) -> np.ndarray:
        result = np.zeros(values.shape, dtype=int)
        result[values < -tol] = -1  # 改善
        result[values > tol] = 1  # 恶化
        return result

    valid = np.isfinite(dse_sst) & np.isfinite(dse_flux)
    sst_side = side(dse_sst, tol_sst)
    flux_side = side(dse_flux, tol_flux)
    both_changed = valid & (sst_side != 0) & (flux_side != 0)
    sst_improve = sst_side < 0
    flux_improve = flux_side < 0
    category[valid & (sst_side == 0) & (flux_side == 0)] = 0
    category[valid & ((sst_side == 0) ^ (flux_side == 0))] = -1
    category[both_changed & sst_improve & flux_improve] = 1
    category[both_changed & sst_improve & ~flux_improve] = 2
    category[both_changed & ~sst_improve & flux_improve] = 3
    category[both_changed & ~sst_improve & ~flux_improve] = 4
    return category


def category_fractions(category: np.ndarray, mask=None) -> dict:
    """五类占比(等权格点占比,分母为共同有效格点数)。"""
    category = np.asarray(category)
    if mask is not None:
        category = category[np.asarray(mask, dtype=bool)]
    n = int(category.size)
    result: dict = {"n_valid": n}
    if n == 0:
        result["status"] = "empty_mask"
        for label in CATEGORY_LABELS.values():
            result[f"frac_{label}"] = np.nan
        return result
    result["status"] = "ok"
    for code, label in CATEGORY_LABELS.items():
        result[f"frac_{label}"] = float(np.mean(category == code))
    return result


def flux_error_budget(flux_strong, flux_weak, flux_nr, base_mask=None) -> dict:
    """通量误差变化的精确分解(诊断二)。

        e  = F_weak - F_NR          (未去均值的误差乘积,不是协方差)
        dF = F_strong - F_weak
        dSE(逐点) = 2 e dF + (dF)^2
        <dSE> = delta_mse_direct = MSE_strong - MSE_weak
              = cross_term(2<e dF>) + increment_square_term(<dF^2>)
        closure_residual = delta_mse_direct - (cross_term + increment_square_term)

    逐点乘积先平均,严禁用 <e><dF> 代替 <e dF>。
    掩膜为统一掩膜(区域 & 海洋 & 全部数组共同有效)。
    """
    strong = np.asarray(flux_strong, dtype=float)
    weak = np.asarray(flux_weak, dtype=float)
    truth = np.asarray(flux_nr, dtype=float)
    mask, n = common_valid_mask(strong, weak, truth, base_mask=base_mask)

    nan_keys = (
        "mse_weak", "mse_strong", "delta_mse_direct", "cross_term",
        "increment_square_term", "closure_residual",
    )
    if n == 0:
        return {"n_valid": 0, "status": "empty_mask", **{key: np.nan for key in nan_keys}}

    error = weak[mask] - truth[mask]          # e
    increment = strong[mask] - weak[mask]     # dF
    mse_weak = float(np.mean(error**2))
    mse_strong = float(np.mean((error + increment) ** 2))
    delta_mse_direct = mse_strong - mse_weak
    cross_term = float(np.mean(2.0 * error * increment))
    increment_square_term = float(np.mean(increment**2))
    closure_residual = delta_mse_direct - (cross_term + increment_square_term)
    return {
        "n_valid": n,
        "mse_weak": mse_weak,
        "mse_strong": mse_strong,
        "delta_mse_direct": delta_mse_direct,
        "cross_term": cross_term,
        "increment_square_term": increment_square_term,
        "closure_residual": closure_residual,
        "status": "ok",
    }


# ------------------------------------------------------------------
# 最低模式层输入推导(合成与真实模式共用;口径取自 omtmp_pathway_extract.py)
# ------------------------------------------------------------------


def initial_momentum_roughness(ustar, isftcflx: int = 0) -> np.ndarray:
    """初始摩阻速度对应的动力粗糙度(诊断式,与 omtmp_pathway_extract.py z0m0 一致)。

    TODO(待核实):该式即 wrf41_sfclayrev._momentum_roughness 的 isftcflx=0 形式
    (CZO*u*^2/g + 0.11*nu/u*,上限 2.85e-3);真实运行前需确认试验 namelist 的
    isftcflx 与 sfclay 方案版本。
    """
    if isftcflx != 0:
        raise NotImplementedError("仅实现 isftcflx=0 的诊断式 z0m;其余选项待核实")
    ustar = np.maximum(np.asarray(ustar, dtype=float), 1.0e-4)
    return np.minimum(0.0185 * ustar**2 / 9.81 + 0.11 * 1.5e-5 / ustar, 2.85e-3)


def derive_lowest_level_inputs(raw: dict) -> dict:
    """由 wrfout 原始场推导 sfclay 重建所需的最低模式层输入。

    输入 raw 键(wrfout 变量,均取当个时次、去 time 维):
        T(Perturbation potential temperature, level 0)
        P, PB(level 0)、QVAPOR(level 0)、PSFC、HGT、PH, PHB(level 0:2)
        U, V(第 0 层,交错网格)、UST

    输出键(口径与 omtmp_pathway_extract.py 第 224-245 行一致,待审查):
        air_temperature_k : theta0*(p/p0)^(Rd/cp),Rd=287, cp=1004
        vapor_mixing_ratio: QVAPOR level 0
        air_pressure_pa   : P+PB level 0
        surface_pressure_pa: PSFC
        height_agl_m      : 0.5*((PH+PHB)_0+(PH+PHB)_1)/9.81 - HGT(第 1 层中点离地高度)
        u_ms, v_ms        : 第 0 层 U/V 去交错
        initial_friction_velocity_ms: UST
        initial_momentum_roughness_m: initial_momentum_roughness(UST)
    """
    theta0 = np.asarray(raw["T"], dtype=float) + 300.0
    pressure0 = np.asarray(raw["P"], dtype=float) + np.asarray(raw["PB"], dtype=float)
    geopotential = np.asarray(raw["PH"], dtype=float) + np.asarray(raw["PHB"], dtype=float)
    height_agl = 0.5 * (geopotential[0] + geopotential[1]) / 9.81 - np.asarray(raw["HGT"], dtype=float)
    u_staggered = np.asarray(raw["U"], dtype=float)
    v_staggered = np.asarray(raw["V"], dtype=float)
    ustable = np.asarray(raw["UST"], dtype=float)
    return {
        "air_temperature_k": theta0 * (pressure0 / 100000.0) ** (287.0 / 1004.0),
        "vapor_mixing_ratio": np.asarray(raw["QVAPOR"], dtype=float),
        "air_pressure_pa": pressure0,
        "surface_pressure_pa": np.asarray(raw["PSFC"], dtype=float),
        "height_agl_m": height_agl,
        "u_ms": 0.5 * (u_staggered[:, :-1] + u_staggered[:, 1:]),
        "v_ms": 0.5 * (v_staggered[:-1, :] + v_staggered[1:, :]),
        "initial_friction_velocity_ms": ustable,
        "initial_momentum_roughness_m": initial_momentum_roughness(ustable),
    }


#: 重建函数的输入键(不含表面温度与 dx/isftcflx)
RECONSTRUCTION_INPUT_KEYS = (
    "air_temperature_k",
    "vapor_mixing_ratio",
    "air_pressure_pa",
    "surface_pressure_pa",
    "height_agl_m",
    "u_ms",
    "v_ms",
    "initial_friction_velocity_ms",
    "initial_momentum_roughness_m",
)


def unified_reconstruction_mask(
    inputs_a: dict, inputs_b: dict, ts_a, ts_b, extra=()
) -> np.ndarray:
    """对参与重建比较的全部数组取共同有限掩膜。

    A/B 两套输入(如 weak/strong)与两套表面温度,以及 extra 中的数组
    (如 NR 通量)全部要求有限,保证比较格点一致。
    """
    arrays = [inputs_a[key] for key in RECONSTRUCTION_INPUT_KEYS]
    arrays += [inputs_b[key] for key in RECONSTRUCTION_INPUT_KEYS]
    arrays += [ts_a, ts_b]
    arrays += [np.asarray(item, dtype=float) for item in extra]
    return finite_mask(*arrays)


# ------------------------------------------------------------------
# 成员汇总(区分两种改善口径)
# ------------------------------------------------------------------


#: summarize_member_metrics 支持的数值列(存在才参与汇总)
SUMMARIZE_METRIC_COLUMNS = (
    "rmse_weak", "rmse_strong", "mse_weak", "mse_strong",
    "weak_bias", "strong_bias",
    "mse_change_strong_minus_weak", "rmse_change_strong_minus_weak",
    "delta_mse_direct", "cross_term", "increment_square_term", "closure_residual",
    "rmse_improvement_pct", "n_valid",
)


def summarize_member_metrics(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
    expected_times: dict | None = None,
    expected_members: int | None = None,
) -> pd.DataFrame:
    """将逐成员明细汇总为方法级表格,明确区分两种改善口径。

    步骤(与 analyze_omtmp_skill.py 的"先在 case 内做时间平均再跨成员统计"一致):
      1. 按 keys+member+window 分组,窗口内对时间求平均 -> 每成员每窗口一行;
         分别统计该成员窗口内的:已读取时次数(n_times_read)、
         有效绝对误差时次数(status 可用或 rmse_weak/strong 有限;零基准 RMSE
         的绝对误差视为有效)、有效改善率时次数(rmse_improvement_pct 有限);
      2. 跨成员(等权)求平均:
           mean_member_rmse_improvement_pct : 各成员相对改善率的平均值;
           n_members_improved               : 窗口平均改善率>0 的唯一成员数;
      3. pooled(平均 RMSE 的相对变化):
           mean_rmse_weak / mean_rmse_strong 为成员窗口平均 RMSE 的再平均,
           pooled_rmse_improvement_pct = 100*(mean_rmse_weak-mean_rmse_strong)/mean_rmse_weak。
    两种口径不可混用。expected_times 给出各窗口期望时次数;
    expected_members 给出配置期望成员数(覆盖率分母显式化)。
    输入为空时返回带完整列结构的空表(不抛异常,run_status 先行落盘)。
    """
    metric_columns = [column for column in SUMMARIZE_METRIC_COLUMNS if column in frame.columns]
    group_extra = [column for column in ("pair",) if column in frame.columns]
    group_columns = list(keys) + ["window"] + group_extra
    coverage_columns = [
        "mean_member_times_read", "min_member_times_read", "max_member_times_read",
        "mean_member_valid_abs_times", "min_member_valid_abs_times",
        "mean_member_valid_improvement_times",
        "expected_times", "member_abs_time_coverage_ratio",
        "expected_members", "member_coverage_ratio",
    ]
    explicit_mean_columns = ["mean_rmse_weak", "mean_rmse_strong"]
    output_columns = list(dict.fromkeys(
        group_columns
        + [
            "n_members", "n_members_improved", "n_members_finite_improvement",
            "mean_member_rmse_improvement_pct", "mean_rmse_weak", "mean_rmse_strong",
            "pooled_rmse_improvement_pct",
        ]
        + coverage_columns
        + [f"mean_{column}" for column in metric_columns
           if column not in ("rmse_improvement_pct", "rmse_weak", "rmse_strong")]
    ))
    has_abs_columns = "rmse_weak" in frame.columns and "rmse_strong" in frame.columns
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    frame = frame.copy()
    frame["window"] = [window_of(hour) for hour in frame["time_hour"]]
    frame = frame.explode("window", ignore_index=True)
    # 逐行有效性:绝对误差(status 可用,含 zero_rmse_weak)与改善率(分母有限)
    if "status" in frame.columns:
        frame["_abs_valid"] = frame["status"].isin(("ok", "zero_rmse_weak"))
    elif has_abs_columns:
        frame["_abs_valid"] = frame["rmse_weak"].notna() & frame["rmse_strong"].notna()
    else:
        frame["_abs_valid"] = True
    if "rmse_improvement_pct" in frame.columns:
        frame["_impr_valid"] = frame["rmse_improvement_pct"].notna()
    else:
        frame["_impr_valid"] = False
    per_member_agg = {
        **{column: (column, "mean") for column in metric_columns},
        "n_times_read": ("time_hour", "count"),
        "n_times_abs_valid": ("_abs_valid", "sum"),
        "n_times_impr_valid": ("_impr_valid", "sum"),
    }
    per_member = (
        frame.groupby(list(keys) + ["member", "window"] + group_extra, sort=True)
        .agg(**per_member_agg)
        .reset_index()
    )
    rows = []
    for group_keys, group in per_member.groupby(group_columns, sort=True):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = dict(zip(group_columns, group_keys))
        has_improvement = "rmse_improvement_pct" in group.columns
        has_rmse = "rmse_weak" in group.columns and "rmse_strong" in group.columns
        if has_improvement:
            improvements = group["rmse_improvement_pct"].to_numpy(dtype=float)
            finite_improvements = improvements[np.isfinite(improvements)]
            row["n_members_improved"] = int(np.count_nonzero(finite_improvements > 0.0))
            row["n_members_finite_improvement"] = int(finite_improvements.size)
            row["mean_member_rmse_improvement_pct"] = (
                float(np.mean(finite_improvements)) if finite_improvements.size else np.nan
            )
        else:
            row["n_members_improved"] = 0
            row["n_members_finite_improvement"] = 0
            row["mean_member_rmse_improvement_pct"] = np.nan
        row["n_members"] = int(group["member"].nunique())
        if has_rmse:
            mean_rmse_weak = float(np.nanmean(group["rmse_weak"].to_numpy(dtype=float)))
            mean_rmse_strong = float(np.nanmean(group["rmse_strong"].to_numpy(dtype=float)))
            row["mean_rmse_weak"] = mean_rmse_weak
            row["mean_rmse_strong"] = mean_rmse_strong
            row["pooled_rmse_improvement_pct"] = (
                100.0 * (mean_rmse_weak - mean_rmse_strong) / mean_rmse_weak
                if mean_rmse_weak > 0.0
                else np.nan
            )
        else:
            row["mean_rmse_weak"] = np.nan
            row["mean_rmse_strong"] = np.nan
            row["pooled_rmse_improvement_pct"] = np.nan
        times_read = group["n_times_read"].to_numpy(dtype=float)
        abs_valid = group["n_times_abs_valid"].to_numpy(dtype=float)
        impr_valid = group["n_times_impr_valid"].to_numpy(dtype=float)
        row["mean_member_times_read"] = float(np.mean(times_read))
        row["min_member_times_read"] = int(np.min(times_read))
        row["max_member_times_read"] = int(np.max(times_read))
        row["mean_member_valid_abs_times"] = float(np.mean(abs_valid))
        row["min_member_valid_abs_times"] = int(np.min(abs_valid))
        row["mean_member_valid_improvement_times"] = float(np.mean(impr_valid))
        if expected_times is not None:
            window_name = row["window"]
            row["expected_times"] = int(expected_times.get(window_name, 0))
            # 覆盖率分母 = 配置期望时次数;分子 = 有效绝对误差时次数
            row["member_abs_time_coverage_ratio"] = (
                row["mean_member_valid_abs_times"] / row["expected_times"]
                if row["expected_times"]
                else np.nan
            )
        if expected_members is not None:
            row["expected_members"] = int(expected_members)
            row["member_coverage_ratio"] = (
                row["n_members"] / expected_members if expected_members else np.nan
            )
        for column in metric_columns:
            if column in ("rmse_improvement_pct", "rmse_weak", "rmse_strong"):
                continue  # 已按专用口径单列输出,避免重复表头
            row[f"mean_{column}"] = float(np.nanmean(group[column].to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


# ------------------------------------------------------------------
# 输出(CSV / npz / 绘图)
# ------------------------------------------------------------------


def ensure_output_dir(config: VerifyConfig, script_name: str) -> Path:
    check_mode(config)
    out_dir = config.out_dir(script_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(exist_ok=True)
    (out_dir / "fields").mkdir(exist_ok=True)
    return out_dir


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    print(f"[output] {path} ({len(frame)} rows)", flush=True)


def write_columns_readme(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[output] {path}", flush=True)


def write_overall_status(out_dir: Path, n_valid_rows: int, n_skipped: int, mode_tag: str) -> str:
    """写出批次总状态:有效结果行为零时显式标记 no_valid_cases,避免
    "程序正常结束"被误读为"诊断有有效结果"。"""
    overall = "ok" if n_valid_rows > 0 else "no_valid_cases"
    text = (
        f"overall_status: {overall}\n"
        f"valid_result_rows: {n_valid_rows}\n"
        f"skipped_or_failed_cases: {n_skipped}\n"
    )
    (out_dir / "overall_status.txt").write_text(text, encoding="utf-8")
    print(f"[output] {out_dir / 'overall_status.txt'} ({overall})", flush=True)
    if overall != "ok":
        print(
            f"[{mode_tag}] WARNING: no valid result rows; "
            "check verify0X_run_status.csv for the failure reasons",
            flush=True,
        )
    return overall


def save_point_fields(path: Path, **arrays) -> None:
    """保存逐点中间场(.npz),供复查图件和与诊断一的分类场对应。"""
    np.savez_compressed(path, **arrays)


def setup_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"[output] {path}", flush=True)


def plot_metric_timeseries(series: dict, title: str, ylabel: str, path: Path) -> None:
    """时间序列图:series = {label: (times, values)};NaN 处断线。"""
    import matplotlib.pyplot as plt

    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for label, (x, y) in series.items():
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ax.plot(x, y, marker="o", markersize=3.0, linewidth=1.2, label=label)
    ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle=":")
    ax.set_xlabel("forecast hour (h)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.0, ncol=2)
    save_figure(fig, path)


def plot_category_map(
    category, distance_km, title: str, path: Path, annulus_edges=(75.0, 150.0, 300.0)
) -> None:
    """逐点分类图:五类离散色标 + 环带距离等值线。"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    setup_matplotlib()
    codes = [-1, 0, 1, 2, 3, 4]
    colors = ["#bdbdbd", "#d9d9d9", "#1a9850", "#fdae61", "#74add1", "#d73027"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([c - 0.5 for c in codes] + [4.5], cmap.N)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    masked = np.ma.masked_equal(category, -99)
    image = ax.imshow(masked, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    contour = np.asarray(distance_km, dtype=float)
    ax.contour(contour, levels=list(annulus_edges), colors="k", linewidths=0.6, linestyles="--")
    cbar = fig.colorbar(image, ax=ax, ticks=codes, fraction=0.046)
    cbar.ax.set_yticklabels(
        ["one\nunchanged", "both\nunchanged", "SST+ F+", "SST+ F-", "SST- F+", "SST- F-"],
        fontsize=7.0,
    )
    ax.set_xlabel("x (grid index)")
    ax.set_ylabel("y (grid index)")
    ax.set_title(title, fontsize=9.0)
    save_figure(fig, path)


def plot_budget_bars(labels, values: dict, title: str, ylabel: str, path: Path) -> None:
    """误差预算分量柱状图(交叉项/平方项/直接变化)。"""
    import matplotlib.pyplot as plt

    setup_matplotlib()
    width = 0.8 / max(len(values), 1)
    positions = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for offset, (name, vals) in enumerate(values.items()):
        vals = np.asarray(vals, dtype=float)
        ax.bar(positions + offset * width, vals, width=width, label=name)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xticks(positions + width * (len(values) - 1) / 2.0)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8.0)
    save_figure(fig, path)


# ------------------------------------------------------------------
# 真实数据提供者(本次未执行;接口与字段约定待审查)
# ------------------------------------------------------------------


class RealWrfProvider:
    """真实 wrfout 数据提供者(入口契约与合成 provider 一致)。

    公开接口:static_fields / nr_static / has_case / exp_fields /
    exp_raw_inputs / nr_fields / register_field_lltoxy。
    路径与字段约定均来自本地源码(omtmp_skill_extract.py / omtmp_pathway_extract.py),
    标注 TODO(待核实) 的项目必须先人工确认;读取适配层已用本地合成 NetCDF
    文件做过接口测试(见 verify_smoke_test),但从未在真实数据上执行。
    """

    def __init__(self, config: VerifyConfig):
        if config.mode != "real" or not config.real.acknowledge_real_mode:
            raise RuntimeError("RealWrfProvider 仅能在显式确认的真实模式下构造")
        self.config = config
        self.base_dir = Path(config.real.forecast_base_dir)
        self.nr_dir = Path(config.real.nr_dir)
        self._static_ref: dict | None = None
        self._nr_static_ref: dict | None = None

    # ------------------------------------------------------------------
    # 路径与时间
    # ------------------------------------------------------------------

    def wrfout_path(self, experiment: str, method: str, member: str, time_name: str) -> Path:
        # TODO(待核实):目录结构 BASE/EXP/METHOD/MEMBER/wrfout_d02_<time>
        return self.base_dir / experiment / method / member / f"wrfout_{self.config.domain}_{time_name}"

    def nr_path(self, time_name: str) -> Path:
        return self.nr_dir / f"wrfout_{self.config.domain}_{time_name}"

    def time_name(self, time_hour: float) -> str:
        for hour, name in self.config.times:
            if hour == time_hour:
                return name
        raise KeyError(f"time_hour {time_hour} not in configuration")

    def _open_any_nr(self):
        """打开任一存在的 NR 文件(仅用于投影/静态信息)。"""
        from netCDF4 import Dataset

        for _, time_name in self.config.times:
            path = self.nr_path(time_name)
            if path.exists():
                return Dataset(path, "r")
        raise FileNotFoundError(f"no NR wrfout found under {self.nr_dir}")

    @staticmethod
    def _decode_wrf_time(stamp) -> str:
        """解码 WRF Times 变量的三种存储形式:VLEN 字符串、bytes、S1 字符数组。

        对 (Time, DateStrLen) 的 S1 字符数组,Times[0] 是 bytes/str 元素数组,
        必须逐元素拼接;直接 str() 会得到 "[b'2' b'0' ...]" 而永远匹配失败。
        """
        if isinstance(stamp, np.ndarray):
            parts = []
            for item in stamp.tolist():
                if isinstance(item, (bytes, np.bytes_)):
                    parts.append(item.decode("utf-8", errors="replace"))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        if isinstance(stamp, (bytes, np.bytes_)):
            return stamp.decode("utf-8", errors="replace").strip()
        return str(stamp).strip()

    def _validate_file_time(self, dataset, path: Path, time_name: str) -> None:
        """校验文件内 Times 属性与请求时刻一致(不能只靠文件名)。"""
        if "Times" not in dataset.variables:
            raise RuntimeError(f"{path} 缺少 Times 变量,无法校验有效时间")
        stamp = dataset.variables["Times"][0]
        text = self._decode_wrf_time(stamp)
        if text != time_name:
            raise RuntimeError(f"{path} 有效时间 {text!r} 与请求时刻 {time_name!r} 不符")

    # ------------------------------------------------------------------
    # 入口契约方法(与合成 provider 对齐)
    # ------------------------------------------------------------------

    def has_case(self, method: str, member: str, time_hour: float) -> bool:
        """强、弱配对成员的当个时次文件都必须存在。TODO(待核实):未在真实数据执行。"""
        time_name = self.time_name(time_hour)
        return all(
            self.wrfout_path(experiment, method, member, time_name).exists()
            for experiment in (self.config.strong_experiment, self.config.weak_experiment)
        )

    def static_fields(self) -> dict:
        """试验网格静态场(取弱试验首个可用时次),并缓存为网格一致性参考。"""
        for _, time_name in self.config.times:
            for method in self.config.methods:
                for member in self.config.members:
                    path = self.wrfout_path(
                        self.config.weak_experiment, method, member, time_name
                    )
                    if path.exists():
                        static = self.read_static(path)
                        self._static_ref = static
                        return static
        raise FileNotFoundError("no experiment wrfout found for static fields")

    def nr_static(self) -> dict:
        """NR 网格静态场(XLAT/XLONG/LANDMASK),并缓存为 NR 网格一致性参考。

        TODO(待核实):未在真实数据执行。几何固定假设在此显式断言:
        其他时次 NR 文件读取时会与该参考逐点核对(_check_nr_grid)。
        """
        with self._open_any_nr() as dataset:
            static = {
                "lat": self._read_2d(dataset, "XLAT"),
                "lon": self._read_2d(dataset, "XLONG"),
                "landmask": self._read_2d(dataset, "LANDMASK"),
            }
        self._nr_static_ref = {"lat": static["lat"], "lon": static["lon"]}
        return static

    def exp_fields(self, experiment: str, method: str, member: str, time_hour: float) -> dict:
        """试验侧单时次分析场;校验文件有效时间与网格一致性。

        强/弱试验网格必须与 static_fields 参考网格一致(经纬度逐点核对),
        不允许仅因维度相同就逐下标比较。TODO(待核实):未在真实数据执行。
        """
        from netCDF4 import Dataset

        time_name = self.time_name(time_hour)
        path = self.wrfout_path(experiment, method, member, time_name)
        ocean_layer = self.config.surface_ocean_layer
        with Dataset(path, "r") as dataset:
            self._validate_file_time(dataset, path, time_name)
            fields = {
                "om": np.asarray(
                    np.ma.filled(dataset.variables["OM_TMP"][0, ocean_layer], np.nan), dtype=float
                ),
                "tsk": self._read_2d(dataset, "TSK"),
                "hfx": self._read_2d(dataset, "HFX"),
                "lh": self._read_2d(dataset, "LH"),
                "psfc": self._read_2d(dataset, "PSFC"),
            }
            lat = self._read_2d(dataset, "XLAT")
            lon = self._read_2d(dataset, "XLONG")
        if self._static_ref is not None:
            if not (
                np.allclose(lat, self._static_ref["lat"], atol=1.0e-6, rtol=0.0)
                and np.allclose(lon, self._static_ref["lon"], atol=1.0e-6, rtol=0.0)
            ):
                raise RuntimeError(
                    f"experiment grid mismatch between {path} and the static reference grid"
                )
        return fields

    def exp_raw_inputs(self, experiment: str, method: str, member: str, time_hour: float) -> dict:
        """wrfout 风格的最低层原始场,供 derive_lowest_level_inputs 推导。

        TODO(待核实):最低模式层输入定义(T+300、P+PB、QVAPOR、
        0.5*((PH+PHB)_0+(PH+PHB)_1)/9.81-HGT、U/V 去交错、UST、诊断式 z0m)
        复用 omtmp_pathway_extract.py 口径,真实数据上未测试。
        """
        from netCDF4 import Dataset

        time_name = self.time_name(time_hour)
        path = self.wrfout_path(experiment, method, member, time_name)
        with Dataset(path, "r") as dataset:
            self._validate_file_time(dataset, path, time_name)
            return {
                "T": np.asarray(np.ma.filled(dataset.variables["T"][0, 0], np.nan), dtype=float),
                "P": np.asarray(np.ma.filled(dataset.variables["P"][0, 0], np.nan), dtype=float),
                "PB": np.asarray(np.ma.filled(dataset.variables["PB"][0, 0], np.nan), dtype=float),
                "QVAPOR": np.asarray(
                    np.ma.filled(dataset.variables["QVAPOR"][0, 0], np.nan), dtype=float
                ),
                "PSFC": self._read_2d(dataset, "PSFC"),
                "HGT": self._read_2d(dataset, "HGT"),
                "PH": np.asarray(np.ma.filled(dataset.variables["PH"][0, 0:2], np.nan), dtype=float),
                "PHB": np.asarray(np.ma.filled(dataset.variables["PHB"][0, 0:2], np.nan), dtype=float),
                "U": np.asarray(np.ma.filled(dataset.variables["U"][0, 0], np.nan), dtype=float),
                "V": np.asarray(np.ma.filled(dataset.variables["V"][0, 0], np.nan), dtype=float),
                "UST": self._read_2d(dataset, "UST"),
                "TSK": self._read_2d(dataset, "TSK"),
            }

    def nr_fields(self, time_hour: float) -> dict:
        """NR 单时次场;按 nr_time_strategy 处理半小时时刻,并校验时间与网格。

        TODO(待核实):NR 是否有半小时输出未确认;"bracket_interpolate" 复用
        omtmp_skill_extract.py 的相邻整点线性插值口径,真实数据上未测试。
        """
        from netCDF4 import Dataset

        time_name = self.time_name(time_hour)
        exact = self.nr_path(time_name)
        if exact.exists() or self.config.real.nr_time_strategy == "exact":
            with Dataset(exact, "r") as dataset:
                return self._read_nr_fields_validated(dataset, exact, time_name)
        # 相邻整点线性插值(整点文件名由时刻向最近整点取整)
        lower_hour = float(np.floor(time_hour))
        upper_hour = float(np.ceil(time_hour))
        alpha = time_hour - lower_hour
        if alpha == 0.0:
            with Dataset(self.nr_path(self.time_name(lower_hour)), "r") as dataset:
                return self._read_nr_fields_validated(
                    dataset, exact, self.time_name(lower_hour)
                )

        def hour_name(hour: float) -> str:
            for config_hour, name in self.config.times:
                if config_hour == hour:
                    return name
            raise KeyError(hour)

        lower_name, upper_name = hour_name(lower_hour), hour_name(upper_hour)
        with Dataset(self.nr_path(lower_name), "r") as low, Dataset(
            self.nr_path(upper_name), "r"
        ) as high:
            low_fields = self._read_nr_fields_validated(low, self.nr_path(lower_name), lower_name)
            high_fields = self._read_nr_fields_validated(high, self.nr_path(upper_name), upper_name)
            return {
                name: (1.0 - alpha) * low_fields[name] + alpha * high_fields[name]
                for name in low_fields
            }

    def _read_nr_fields_validated(self, dataset, path: Path, expected_time_name: str) -> dict:
        """读取 NR 场并校验文件内部时间与网格一致性(不能只靠文件名/维度)。"""
        self._validate_file_time(dataset, path, expected_time_name)
        self._check_nr_grid(dataset, path)
        return self._read_nr_dataset_fields(dataset)

    def _check_nr_grid(self, dataset, path: Path) -> None:
        """NR 各时次网格必须与参考网格一致(几何固定假设的显式断言)。"""
        if self._nr_static_ref is None:
            return
        lat = self._read_2d(dataset, "XLAT")
        lon = self._read_2d(dataset, "XLONG")
        if not (
            np.allclose(lat, self._nr_static_ref["lat"], atol=1.0e-6, rtol=0.0)
            and np.allclose(lon, self._nr_static_ref["lon"], atol=1.0e-6, rtol=0.0)
        ):
            raise RuntimeError(f"NR grid mismatch between {path} and the NR static reference")

    # ------------------------------------------------------------------
    # 底层读取与配准
    # ------------------------------------------------------------------

    def _read_nr_dataset_fields(self, dataset) -> dict:
        return {
            "om": np.asarray(
                np.ma.filled(
                    dataset.variables["OM_TMP"][0, self.config.surface_ocean_layer], np.nan
                ),
                dtype=float,
            ),
            "tsk": self._read_2d(dataset, "TSK"),
            "hfx": self._read_2d(dataset, "HFX"),
            "lh": self._read_2d(dataset, "LH"),
            "psfc": self._read_2d(dataset, "PSFC"),
            "landmask": self._read_2d(dataset, "LANDMASK"),
        }

    @staticmethod
    def _read_2d(dataset, name: str) -> np.ndarray:
        return np.asarray(np.ma.filled(dataset.variables[name][0], np.nan), dtype=float)

    def read_static(self, path: Path) -> dict:
        from netCDF4 import Dataset

        with Dataset(path, "r") as dataset:
            return {
                "lat": self._read_2d(dataset, "XLAT"),
                "lon": self._read_2d(dataset, "XLONG"),
                "landmask": self._read_2d(dataset, "LANDMASK"),
            }

    def register_field_lltoxy(self, name: str, field, exp_lat, exp_lon) -> np.ndarray:
        """ll_to_xy + 双线性把 NR 场插到试验网格(omtmp_skill_extract.py 口径)。

        仅使用 NR 网格的投影信息(与具体时次无关),域外目标点为 NaN。
        TODO(待核实):该路径从未在真实数据上执行;半小时插值时刻与
        ll_to_xy 组合时使用最近整点文件的投影。
        """
        return self._lltoxy_resample(field, exp_lat, exp_lon, order=1)

    def register_mask_lltoxy(self, name: str, field, exp_lat, exp_lon) -> np.ndarray:
        """海陆掩膜的投影索引 order=0 重采样(WRF 曲线网格默认,避免全点对搜索)。

        与 register_field_lltoxy 同一投影索引,仅插值阶数为 0(最近邻),
        域外目标点为 NaN。TODO(待核实):未在真实数据上执行。
        """
        return self._lltoxy_resample(field, exp_lat, exp_lon, order=0)

    def _lltoxy_resample(self, field, exp_lat, exp_lon, order: int) -> np.ndarray:
        from scipy.ndimage import map_coordinates
        from wrf import ll_to_xy

        with self._open_any_nr() as nr_dataset:
            xy = ll_to_xy(nr_dataset, exp_lat, exp_lon, as_int=False, meta=False)
        x_index = np.asarray(xy[0], dtype=float).reshape(np.asarray(exp_lat).shape)
        y_index = np.asarray(xy[1], dtype=float).reshape(np.asarray(exp_lat).shape)
        coordinates = np.vstack((y_index.ravel(), x_index.ravel()))
        return map_coordinates(
            np.asarray(field, dtype=float),
            coordinates,
            order=order,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        ).reshape(np.asarray(exp_lat).shape)
