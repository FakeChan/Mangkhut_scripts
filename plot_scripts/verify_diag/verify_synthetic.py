"""合成数据生成器(SYNTHETIC ONLY)——为 verify_01/02/03 提供小规模内存数据。

要点:
1. 固定随机种子,可复现;绝不读取任何真实 WRF/NR/缓存文件,也绝不探测真实路径。
2. 合成"NR"位于较粗的规则经纬网格上,试验场位于细网格上,用于真实地走一遍
   NR->试验网格的配准接口(bilinear / nearest)。
3. 台风中心精确放在 NR 网格节点上,PSFC 解析低压中心,检验中心识别逻辑。
4. 海洋温度误差强耦合明显减小(hfx/lh 误差不同步减小),复现待核验的现象形态;
   这只是现象形态的合成,不代表任何物理归因。
5. 构造陆地格点、NaN 缺测、超 300 km 区域、缺失成员时次等边界情形。

重要(生成方式与限制):合成"分析场"= 解析真值场 + 统计误差形态(光滑模态
+ 偏置 + 白噪声),合成"低层大气输入"另按解析廓线加小扰动生成。两组数据
之间没有任何通量方案层面的公式联系,因此本生成器只用于接口与流程测试,
不能用于检验通量物理机制或重建一致性(阶段 A 的残差天然不可忽略)。
诊断三的被测物理函数 reconstruct_ocean_fluxes()/wrf41_sfclayrev 与本生成器
完全独立,不存在循环论证。
"""

from __future__ import annotations

import math
import zlib

import numpy as np

from verify_common import VerifyConfig, check_mode

SYNTHETIC_SEED = 20260930

#: 刻意缺失的 (方法, 成员, 预报时效),用于测试缺失时次的状态报告
MISSING_CASE = ("QCF_RHF", "044", 2.5)

#: NaN 缺测补丁 (方法, 成员, 预报时效, 变量, 强弱标记, y0:y1, x0:x1)
NAN_PATCHES = (
    ("EAKF", "006", 0.5, "hfx", "weak", slice(5, 9), slice(3, 6)),
    ("QCF_RHF", "015", 1.0, "om", "strong", slice(20, 23), slice(8, 11)),
)

#: 强/弱耦合误差幅度因子(强/弱);om 明显减小、hfx/lh 基本不减小 -> 复现现象形态
ERROR_FACTOR_STRONG = {"om": 0.35, "tsk": 0.60, "hfx": 1.05, "lh": 0.95}
BIAS_WEAK = {"om": 0.35, "tsk": 0.25, "hfx": 20.0, "lh": 30.0}
BIAS_STRONG = {"om": 0.10, "tsk": 0.08, "hfx": -8.0, "lh": 12.0}
NOISE_STD_WEAK = {"om": 0.05, "tsk": 0.15, "hfx": 12.0, "lh": 18.0}
NOISE_STD_STRONG = {"om": 0.05, "tsk": 0.15, "hfx": 12.0, "lh": 18.0}

#: 合成网格设置:细网格(试验)0.25 度,粗网格(NR)0.5 度,中心 (14.0N, 138.0E)
GRID_LAT0, GRID_LON0 = 10.0, 135.0
EXP_DDEG, NR_DDEG = 0.25, 0.5
EXP_N, NR_N = 33, 17
CENTER_LAT, CENTER_LON = 14.0, 138.0
#: 陆地经度界限(试验与 NR 略有错位,检验双重海洋掩膜)
LAND_LON_EXP, LAND_LON_NR = 141.25, 141.5


def _smooth_modes(rng: np.random.Generator, shape: tuple[int, int], n_modes: int = 4) -> np.ndarray:
    """随机低阶三角模态叠加,生成空间光滑的无量纲场(均值约 0,标准差约 1)。"""
    ny, nx = shape
    v = np.linspace(0.0, 1.0, ny)[:, None]
    u = np.linspace(0.0, 1.0, nx)[None, :]
    field = np.zeros(shape, dtype=float)
    for _ in range(n_modes):
        k = int(rng.integers(1, 4))
        l = int(rng.integers(1, 4))
        amplitude = float(rng.uniform(-1.0, 1.0))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        field += amplitude * np.sin(math.pi * k * v + phase) * np.sin(math.pi * l * u)
    return field / max(np.std(field), 1.0e-12)


class SyntheticDataset:
    """内存中的合成数据集,接口与 RealWrfProvider 对齐。"""

    def __init__(
        self,
        config: VerifyConfig,
        members_by_method: dict[str, tuple[str, ...]],
        seed: int = SYNTHETIC_SEED,
    ):
        check_mode(config)
        if config.mode != "synthetic":
            raise RuntimeError("SyntheticDataset 仅用于合成模式")
        self.config = config
        self.members_by_method = {
            method: tuple(member for member in members if member in set(config.members))
            for method, members in members_by_method.items()
        }
        self.times = dict(config.times)
        self.missing_case = MISSING_CASE

        # ---- 网格(试验细网格、NR 粗网格) ----
        exp_lat_1d = GRID_LAT0 + EXP_DDEG * np.arange(EXP_N)
        exp_lon_1d = GRID_LON0 + EXP_DDEG * np.arange(EXP_N)
        nr_lat_1d = GRID_LAT0 + NR_DDEG * np.arange(NR_N)
        nr_lon_1d = GRID_LON0 + NR_DDEG * np.arange(NR_N)
        self.exp_lat, self.exp_lon = np.meshgrid(exp_lat_1d, exp_lon_1d, indexing="ij")
        self.nr_lat, self.nr_lon = np.meshgrid(nr_lat_1d, nr_lon_1d, indexing="ij")
        self.exp_land = np.where(self.exp_lon >= LAND_LON_EXP, 1.0, 0.0)
        self.nr_land = np.where(self.nr_lon >= LAND_LON_NR, 1.0, 0.0)

        # ---- 台风中心(精确位于 NR 网格节点) ----
        self.center_lat, self.center_lon = CENTER_LAT, CENTER_LON
        self.center_offset_km = 0.35  # 真实中心强度半径尺度

        self._rng = np.random.default_rng(seed)
        self._base_seed = seed
        # 预生成误差场缓存,保证同一 (var, method, member) 在各时次形态连续
        self._error_cache: dict[tuple, dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # 解析"真值"场(任意经纬度可评估,细/粗网格共用同一函数)
    # ------------------------------------------------------------------

    def _center_distance_deg(self, lat, lon) -> float:
        return np.hypot(lat - self.center_lat, lon - self.center_lon)

    def truth_field(self, variable: str, time_hour: float, lat, lon) -> np.ndarray:
        """合成 NR"真值"(模式通量视角),随时间缓慢演变。"""
        r = self._center_distance_deg(lat, lon) / self.center_offset_km
        core = np.exp(-(r**2))
        drift = 1.0 + 0.05 * time_hour
        if variable == "om":
            return 302.0 - 1.5 * core + 0.1 * time_hour + 0.3 * np.sin(np.deg2rad(lon * 8.0))
        if variable == "tsk":
            return 303.0 - 1.2 * core + 0.1 * time_hour
        if variable == "hfx":
            return 30.0 + 180.0 * core * drift
        if variable == "lh":
            return 80.0 + 260.0 * core * drift
        raise KeyError(variable)

    def psfc_field(self, lat, lon) -> np.ndarray:
        """海平面气压:含解析低压中心,供中心识别逻辑检验。"""
        r = self._center_distance_deg(lat, lon) / 1.8
        return 100800.0 - 1800.0 * np.exp(-(r**2))

    # ------------------------------------------------------------------
    # 误差场(成员/方法相关,时次间连续)
    # ------------------------------------------------------------------

    def _error_fields(self, variable: str, method: str, member: str) -> dict[str, np.ndarray]:
        """返回该 (variable, method, member) 在试验网格上的 weak/strong 误差形态(模板)。"""
        key = (variable, method, member)
        if key not in self._error_cache:
            # zlib.crc32 保证跨进程可复现(Python 内置 hash 受 PYTHONHASHSEED 随机化)
            digest = zlib.crc32("&".join(key).encode("utf-8"))
            seed = (self._base_seed + digest) % (2**32)
            rng = np.random.default_rng(seed)
            shape = self.exp_lat.shape
            self._error_cache[key] = {
                "weak": _smooth_modes(rng, shape),
                "strong": _smooth_modes(rng, shape),
                "noise_seed": int(rng.integers(0, 2**32)),
            }
        return self._error_cache[key]

    def _experiment_field(
        self, variable: str, experiment: str, method: str, member: str, time_hour: float
    ) -> np.ndarray:
        truth = self.truth_field(variable, time_hour, self.exp_lat, self.exp_lon)
        template = self._error_fields(variable, method, member)
        growth = 1.0 + 0.2 * time_hour
        noise_rng = np.random.default_rng(template["noise_seed"] + int(round(time_hour * 2)))
        white_weak = noise_rng.standard_normal(self.exp_lat.shape)
        white_strong = np.random.default_rng(template["noise_seed"] + 7 + int(round(time_hour * 2))).standard_normal(
            self.exp_lat.shape
        )
        if experiment == self.config.weak_experiment:
            error = (
                BIAS_WEAK[variable]
                + NOISE_STD_WEAK[variable] * template["weak"] * growth
                + 0.5 * NOISE_STD_WEAK[variable] * white_weak
            )
        elif experiment == self.config.strong_experiment:
            error = (
                BIAS_STRONG[variable]
                + ERROR_FACTOR_STRONG[variable]
                * NOISE_STD_STRONG[variable]
                * template["strong"]
                * growth
                + 0.5 * NOISE_STD_STRONG[variable] * white_strong
            )
        else:
            raise KeyError(experiment)
        field = truth + error
        # 陆地格点:om 与通量缺测;tsk 仍有值(与真实 wrfout 行为类似)
        if variable in ("om", "hfx", "lh"):
            field = np.where(self.exp_land < 0.5, field, np.nan)
        return field

    def _apply_nan_patches(self, field, method: str, member: str, time_hour: float, variable: str, side: str):
        for spec_method, spec_member, spec_time, spec_var, spec_side, ys, xs in NAN_PATCHES:
            # 必须逐项比较 spec_* 与请求的 (method, member, ...),否则补丁会
            # 误施加到其他方法/成员的同名时次上
            if (spec_method, spec_member, float(spec_time), spec_var, spec_side) == (
                method,
                member,
                time_hour,
                variable,
                side,
            ):
                field = field.copy()
                field[ys, xs] = np.nan
        return field

    # ------------------------------------------------------------------
    # 对外接口(与 RealWrfProvider 对齐)
    # ------------------------------------------------------------------

    def static_fields(self) -> dict:
        return {"lat": self.exp_lat, "lon": self.exp_lon, "landmask": self.exp_land}

    def nr_static(self) -> dict:
        return {"lat": self.nr_lat, "lon": self.nr_lon, "landmask": self.nr_land}

    def has_case(self, method: str, member: str, time_hour: float) -> bool:
        """成员在两方法下成对存在;缺失时次按 MISSING_CASE 判定。"""
        if member not in self.members_by_method.get(method, ()):
            return False
        if (method, member, time_hour) == self.missing_case:
            return False
        return time_hour in self.times

    def exp_fields(self, experiment: str, method: str, member: str, time_hour: float) -> dict:
        """试验侧分析场:om/tsk/hfx/lh + psfc(细网格)。"""
        side = "strong" if experiment == self.config.strong_experiment else "weak"
        fields = {}
        for variable in ("om", "tsk", "hfx", "lh"):
            field = self._experiment_field(variable, experiment, method, member, time_hour)
            fields[variable] = self._apply_nan_patches(field, method, member, time_hour, variable, side)
        fields["psfc"] = self.psfc_field(self.exp_lat, self.exp_lon)
        return fields

    def nr_fields(self, time_hour: float) -> dict:
        """NR 侧场(粗网格):om/tsk/hfx/lh 真值 + psfc + landmask。

        om/通量在 NR 陆地缺测(与真实 NR 一致),tsk/psfc 全场有效。
        """
        fields = {}
        for variable in ("om", "tsk", "hfx", "lh"):
            value = self.truth_field(variable, time_hour, self.nr_lat, self.nr_lon)
            if variable in ("om", "hfx", "lh"):
                value = np.where(self.nr_land < 0.5, value, np.nan)
            fields[variable] = value
        fields["psfc"] = self.psfc_field(self.nr_lat, self.nr_lon)
        fields["landmask"] = self.nr_land
        return fields

    # ------------------------------------------------------------------
    # 诊断三所需的最低模式层原始场(wrfout 风格,推导口径由
    # verify_common.derive_lowest_level_inputs 完成)
    # ------------------------------------------------------------------

    def _wind_profile(self, lat, lon, time_hour: float) -> tuple[float, float]:
        r = self._center_distance_deg(lat, lon) / self.center_offset_km
        speed = 6.0 + 16.0 * np.exp(-(r**2)) + 0.5 * time_hour
        return speed, 0.25 * speed  # 简化的切向风合成

    def exp_raw_inputs(self, experiment: str, method: str, member: str, time_hour: float) -> dict:
        """wrfout 风格的最低层原始场(T/P/PB/QVAPOR/PH/PHB/HGT/U/V/UST/PSFC/TSK)。

        低层大气包含与试验(强/弱)和成员相关的小扰动(量级 ~0.5 K、~4% 风速),
        使诊断三的"替换其余输入"通路(A_weak != A_strong)被真实测试;
        成员间还叠加不同的海温-气温关系,覆盖弱不稳定/近中性等不同稳定度形态。
        扰动量级刻意保持小幅度,不代表强耦合试验的真实大气差异。
        """
        fields = self.exp_fields(experiment, method, member, time_hour)
        del fields["hfx"], fields["lh"]
        tsk = fields["tsk"]
        r = self._center_distance_deg(self.exp_lat, self.exp_lon) / self.center_offset_km
        # 试验与成员相关的大气扰动(固定种子,可复现)
        digest = zlib.crc32(f"{experiment}&{method}&{member}".encode("utf-8"))
        rng = np.random.default_rng(digest)
        is_strong = experiment == self.config.strong_experiment
        strong_shift = 0.5 if is_strong else 0.0  # 强耦合大气偏暖 ~0.5 K(小扰动)
        wind_mod = 1.0 + 0.04 * _smooth_modes(rng, self.exp_lat.shape) + (0.04 if is_strong else 0.0)
        theta_mod = 0.4 * _smooth_modes(rng, self.exp_lat.shape)
        qv_mod = 0.0012 * _smooth_modes(rng, self.exp_lat.shape)
        # 成员相关的稳定度形态:海温-气温差在弱不稳定到近中性间变化
        stability_offset = float(rng.uniform(-0.8, 0.3))
        theta0 = (
            300.0
            + 2.0 * np.exp(-(r**2))
            + strong_shift
            + stability_offset
            + theta_mod
        )
        psfc = fields["psfc"]
        pressure0 = psfc - 350.0
        qv0 = 0.016 + 0.004 * np.exp(-(r**2)) + qv_mod
        speed, angle = self._wind_profile(self.exp_lat, self.exp_lon, time_hour)
        speed = speed * wind_mod
        u_center = speed * np.cos(np.deg2rad(angle * 10.0))
        v_center = speed * np.sin(np.deg2rad(angle * 10.0))
        ustable = 0.15 + 0.35 * np.exp(-(r**2)) + (0.02 if is_strong else 0.0)

        ny, nx = self.exp_lat.shape
        surface_height = np.where(self.exp_land < 0.5, 0.0, 10.0)  # 陆地低地形,保证高度为正
        raw = {
            "T": theta0 - 300.0,
            "P": pressure0 * 0.6,
            "PB": pressure0 * 0.4,
            "QVAPOR": qv0,
            "PSFC": psfc,
            "HGT": surface_height,
            "PH": np.stack([surface_height * 9.81, np.full((ny, nx), 40.0 * 9.81)]),
            "PHB": np.zeros((2, ny, nx)),
            "U": np.zeros((ny, nx + 1)),
            "V": np.zeros((ny + 1, nx)),
            "UST": ustable,
            "TSK": tsk,
        }
        # 交错网格风场:填充 U/V 使去交错(相邻交错点平均)后等于
        # 0.5*(u_center[:, :-1] + u_center[:, 1:]) 的邻点平均,
        # 即对预期风廓线做轻微平滑;只有常数场才严格等于 u_center 本身。
        raw["U"][:, :-1] = u_center
        raw["U"][:, -1] = u_center[:, -1]
        raw["V"][:-1, :] = v_center
        raw["V"][-1, :] = v_center[-1, :]
        # 陆地:输入场仍给有限值(诊断三按海洋掩膜取子集;掩膜由 TSK/通量缺失控制)
        return raw


def build_synthetic_provider(config: VerifyConfig):
    """按脚本配置构造合成数据集(成员按方法对齐:两方法使用相同六个成员)。"""
    members_by_method = {method: tuple(config.members) for method in config.methods}
    return SyntheticDataset(config, members_by_method)
