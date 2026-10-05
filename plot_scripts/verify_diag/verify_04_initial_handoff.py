"""诊断四:初值传递核验(B → A → I;A 即试验 0h 输出)(默认真实模式)。

最小实现,复用 verify_common 与 pathway/verify_diag 已有能力。核验三组显式
区分的差异(字段名不混用):

    d_assim   = A - B              同化增量(A 为试验 0h 输出)
    d_handoff = I - A              分析到 WRF 实际启动状态的差
    d_pair    = X_strong - X_weak  强弱试验差(各阶段分别计算)

阶段契约(P1 #3 修正):B/A/I 属于**起报循环**,只在该循环的有效时间
(INIT_TIME_NAME)核验;lead time 不充当 cycle。无 Times 的文件必须显式配置
记录号(缺省 None,不盲目取第 0 条)。

计算资格与结论分离(审查 P1 #1/#5):
- 单位:两侧实际 units 都存在且与配置一致才计算;缺失 → unit_unknown,
  不符 → unit_mismatch,均跳过并保留两侧原始单位;别名/转换须显式配置
  (UNIT_ALIASES,默认空);
- 网格:同记录读取、位置对应、NaN 坐标拒绝;交错变量(U/V)保留原生
  交错网格,detail=grid_unverified_staggered(不冒充质量点核验);
- 缺测/填充:基础掩膜独立于数值有效性;相减前按侧识别缺失与
  |x|>=1e30 的填充值(相消填充不再伪装为零差);
- 结论:status=ok 只代表数值可计算;handoff_conclusion 列单独给出
  交接一致性结论,需来源确认 + 逐试验映射核实,否则 not_concluded_*。

输出(空输入也生成带列名的表):
    verify04_source_manifest.csv      逐读取请求的来源与元数据
    verify04_pairwise_checks.csv      阶段差(d_assim/d_handoff)
    verify04_initial_pair_checks.csv  强弱初值差(d_pair)
    verify04_coverage.csv             预期/实际/缺失覆盖
    verify04_status.csv               成功/跳过/失败及原因
    verify04_config_snapshot.json     本诊断实际生效配置
    caliber_compat_table.csv          pathway vs verify_diag 口径判定
    columns_readme.txt

运行: python verify_04_initial_handoff.py(默认真实数据模式;合成模式下
合成输入生成到输出目录 synthetic_inputs/,不读取任何真实文件)。
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd

import verify_common as vc
from verify04_readers import FieldRequest, grids_match, read_stage_field, read_stage_grid
from verify_common import VerifyConfig

SCRIPT_NAME = "verify04_initial_handoff"

# =====================
# 可编辑配置区
# =====================
# 默认真实数据模式(2026-10-05 起,与服务器上 verify_01/02/03 一致)。
# 注意:真实模式下 B/A/I 阶段路径默认未配置(STAGE_SOURCE_DEFAULTS),
# 对应比较将按 source_not_configured 跳过并逐条记录;F0(0h 预报输出)
# 路径已按服务器核实布局预填,d_pair@F0 可直接计算。配置 B/A/I 后
# d_assim/d_handoff 才会参与计算。
CONFIG = VerifyConfig(
    mode="real",
    output_dirname=SCRIPT_NAME,
    real=vc.RealPathConfig(acknowledge_real_mode=True),
    plot_regions=(),  # 本诊断不绘图
)

#: 起报循环的有效时间(B/A/I 的核验时刻;lead time 不充当 cycle)
INIT_TIME_NAME = "2018-09-10_00:00:00"
#: 循环标识(写入输出,便于多循环扩展;每循环需独立运行并配置阶段来源)
CYCLE_ID = "cycle_2018-09-10_00"

#: 传递映射的逐试验定义(证据见 REVIEW_SOURCES_AND_CALIBER.md)。
#: update_ocean=1 走 ncks.sh(海洋+大气变量清单);update_ocean=0 走
#: ncks_air.sh——该文件本地缺失,弱试验映射**未知**(variables=None)。
@dataclasses.dataclass(frozen=True)
class MappingSpec:
    experiment: str
    variables: tuple[str, ...] | None   # None = 映射未知
    basis: str
    confirmed: bool = False             # 仅在服务器运行确认后置 True


MAPPING_SPECS = {
    "6mem_oceanAssim1Run1": MappingSpec(
        experiment="6mem_oceanAssim1Run1",
        variables=("MU", "OM_TMP", "OM_U", "OM_V", "OM_S", "P", "PH",
                   "QVAPOR", "THM", "U", "V", "W"),
        basis="ncks_updateDARTvar.sh variable list (update_ocean=1; inferred, not confirmed)",
        confirmed=False,
    ),
    "6mem_oceanAssim0Run1": MappingSpec(
        experiment="6mem_oceanAssim0Run1",
        variables=None,  # ncks_air.sh 在本地缺失(update_ocean=0 分支)
        basis="ncks_air.sh referenced by driver but ABSENT locally; mapping unknown",
        confirmed=False,
    ),
}
#: 映射之外的已知后处理(记录用,不参与判定)
POST_MAPPING_STEPS = ("update_wrf_bc (boundary); "
                      "ad_omini_d01/d02.py (OM_TINI/OM_SINI profiles from GFS)")

#: 单位登记表(R4 修订:区分拼写别名与数值转换,默认空 = 严格门控)。
#: 键 = (变量名, 文件单位, 期望单位);值 = 登记项:
#:   {"type": "alias", "basis": ...}
#:     数值恒等的拼写别名(如 K 与 Kelvin),放行且不改变数值;
#:   {"type": "conversion", "scale": s, "offset": o, "basis": ...}
#:     数值转换:file_unit 值 * s + o = expected_unit 值,先转换再比较,
#:     并在 detail 记录依据;
#:   其他类型在运行时拒绝(不为未实现的转换放行)。
#: 已登记:OM_TMP 的 units 在 DART/海洋文件与 wrfinput/wrfout 中写作小写 "k",
#: 与期望 "K" 为大小写拼写差异、数值恒等(alias 不改变数值,仅放行)。
UNIT_ALIASES: dict[tuple[str, str, str], dict] = {
    ("OM_TMP", "k", "K"): {"type": "alias",
                           "basis": "case spelling: all DART/ocean files write "
                                    "lowercase 'k'; numerically identical to K"},
}


#: 检查变量配置:name, unit(配置期望单位), layer(None=表面/2D),
#: layer_dim, tolerance, role, staggered(交错网格标记)
@dataclasses.dataclass(frozen=True)
class CheckVariable:
    name: str
    unit: str
    layer: int | None
    layer_dim: str | None
    tolerance: float
    role: str
    staggered: bool = False


CHECK_VARIABLES = (
    CheckVariable("OM_TMP", "K", 0, "ocean_layer_stag", 1.0e-3, "ocean"),
    CheckVariable("TSK", "K", None, None, 1.0e-3, "tsk"),
    CheckVariable("SST", "K", None, None, 1.0e-3, "optional_sst"),
    CheckVariable("T", "K", 0, "bottom_top", 1.0e-6, "atmosphere_pair"),
    CheckVariable("QVAPOR", "kg kg-1", 0, "bottom_top", 1.0e-8, "atmosphere_pair"),
    CheckVariable("U", "m s-1", 0, "bottom_top", 1.0e-6, "atmosphere_pair", staggered=True),
    CheckVariable("V", "m s-1", 0, "bottom_top", 1.0e-6, "atmosphere_pair", staggered=True),
    CheckVariable("PSFC", "Pa", None, None, 1.0e-3, "atmosphere_pair"),
)

#: 真实模式各阶段路径配置(path_template=None = 未配置,比较按状态跳过;
#: inferred_template 为文件名推断,依据见 REVIEW_SOURCES_AND_CALIBER.md;
#: time_record 仅用于无 Times 文件的显式回退,缺省 None = 不盲目取第 0 条;
#: 阶段 A 默认指「后验合并文件」,原始 DART 后验是 output_d0X,两者不同)
@dataclasses.dataclass(frozen=True)
class StageSource:
    stage: str
    path_template: str | None
    time_in_path: bool
    mapping_basis: str
    confirmed: bool
    format: str
    inferred_template: str = ""
    time_record: int | None = None
    time_record_basis: str = ""


#: 2018091000 循环(10_00_00)的阶段路径,已于 2026-10-05 在服务器上以只读
#: 方式核验:B 由 DART input_list_d01/d02.txt 确认(用户确认 + 清单核对);
#: A/I 由探测确认(OM_TMP 表层 I-A 全域精确为 0;文件日期与重跑链一致)。
#: 注意:该配置绑定 10_00_00 单一循环;多循环需按循环分别配置。
STAGE_SOURCE_DEFAULTS = {
    "B": StageSource(
        stage="B",
        path_template="/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/"
                      "0mem_all_time/cyclingDA/10_00_00/firstguess_{domain}.mem{member}",
        time_in_path=False,
        mapping_basis="DART prior per input_list_d01/d02.txt (user-confirmed; "
                      "this copy has TSK==OM_TMP surface on ocean, i.e. "
                      "update_tsk_from_omtmp was applied)",
        confirmed=True, format="wrfinput",
        inferred_template="同路径(cyclingDA/10_00_00;EAKF/ 子目录的 07-06 副本"
                          " TSK 未同步,非 input_list 指向的先验)",
    ),
    "A": StageSource(
        stage="A",
        path_template=str(vc.REAL_FORECAST_BASE_DIR / "{experiment}" / "{method}"
                          / "{member}" / "wrfout_{domain}_{time}"),
        time_in_path=True,
        mapping_basis="the experiment's 0h forecast output = the post-"
                      "assimilation launch state (analysis renamed to wrfinput, "
                      "then output by WRF at t=0); verified 2026-10-05: "
                      "OM_TMP/TSK I-A=0 exactly on all 800000 points "
                      "(user-confirmed semantics)",
        confirmed=True, format="wrfout",
        inferred_template="即 cycle_test/{experiment}/{method}/{member}/ 的 "
                          "wrfout_{domain} 起报时刻输出",
    ),
    "I": StageSource(
        stage="I",
        path_template="/share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA/run_wrf/"
                      "10_00_00/{member}/wrfinput_{domain}",
        time_in_path=True,
        mapping_basis="ncks-merged firstguess moved to wrfinput + update_wrf_bc "
                      "+ ad_omini (run_driver_6mem_cyclingDA.sh:307-320); "
                      "verified exists (2026-10-04 14:29, from the Oct-4 rerun)",
        confirmed=True, format="wrfinput",
        inferred_template="run_wrf/{cycletime}/{member}/wrfinput_d{domain}"
                          "(目录名 10_00_00 为 MM_DD_HH 格式;注意该目录按"
                          "成员共享,后运行的试验会覆盖先前的 wrfinput)",
    ),
}

PAIRWISE_COLUMNS = [
    "mode", "cycle_id", "experiment", "method", "member",
    "init_time", "valid_time", "time_basis",
    "stage_pair", "variable", "layer", "unit_expected",
    "unit_left", "unit_right", "tolerance", "role", "staggered",
    "mapping_basis", "source_confirmed", "status", "detail",
    "handoff_conclusion",
    "n_base", "n_valid", "n_missing", "n_missing_left", "n_missing_right",
    "n_fill",
    "mean_diff", "rms_diff", "max_abs_diff",
    "n_over_tolerance", "frac_over_tolerance",
]
PAIR_COLUMNS = [
    "mode", "cycle_id", "method", "member",
    "init_time", "valid_time", "time_basis",
    "stage", "variable", "layer", "unit_expected",
    "unit_strong", "unit_weak", "tolerance", "role", "staggered",
    "status", "detail", "handoff_conclusion",
    "n_base", "n_valid", "n_missing", "n_missing_left", "n_missing_right",
    "n_fill",
    "mean_pair_diff", "rms_pair_diff", "max_abs_pair_diff",
    "n_over_tolerance", "frac_over_tolerance",
]
MANIFEST_COLUMNS = [
    "mode", "cycle_id", "stage", "experiment", "method", "member",
    "init_time", "variable", "layer",
    "path", "exists", "time_found", "time_source", "time_record_used",
    "unit_expected", "unit_found", "unit_matches",
    "dims_order", "mapping_basis", "mapping_confirmed", "mapping_known",
    "source_confirmed", "format", "detail",
]
STATUS_COLUMNS = ["mode", "scope", "status", "detail", "count"]
COVERAGE_COLUMNS = [
    "mode", "cycle_id", "comparison", "variable", "method",
    "expected", "attempted", "completed", "skipped", "not_attempted",
    "skipped_reasons",
]


def _stage_source_configs(config: VerifyConfig) -> dict[str, StageSource]:
    """各阶段路径配置:synthetic 模式指向输出目录内生成的合成文件;
    real 模式使用 STAGE_SOURCE_DEFAULTS(None=未配置,比较按状态跳过)。"""
    if config.mode == "synthetic":
        synth_root = config.out_dir(SCRIPT_NAME) / "synthetic_inputs"
        # 合成布局本身已知;confirmed 保留 STAGE_SOURCE_DEFAULTS 的缺省(False),
        # 测试可显式置 True 以演练结论链(结论仍需映射确认,见 _handoff_conclusion)
        return {
            stage: dataclasses.replace(
                info,
                path_template=(
                    synth_root / "{stage}_{experiment}_{method}_{member}_{domain}.nc"
                ).as_posix().replace("{stage}", stage),
            )
            for stage, info in STAGE_SOURCE_DEFAULTS.items()
        }
    return {stage: info for stage, info in STAGE_SOURCE_DEFAULTS.items()}


def _resolve_stage_path(template: str | None, domain: str, member: str,
                        time_name: str, experiment: str = "",
                        method: str = "") -> str | None:
    if template is None:
        return None
    return template.format(
        domain=domain, member=member, time=time_name,
        experiment=experiment, method=method,
    )


def _unit_check(read_left: object, read_right: object,
                var: CheckVariable) -> tuple[str | None, str | None, str | None,
                                             tuple | None, tuple | None]:
    """单位门控(P1 #1;R4 修订):返回 (status, unit_left, unit_right,
    transform_left, transform_right)。

    status None 表示通过;unit_unknown(任一侧未提供 units)或
    unit_mismatch(与配置期望不符)时比较被跳过。
    UNIT_ALIASES 登记项:
      - type=alias:数值恒等的拼写别名,放行、不改变数值;
      - type=conversion:按 (scale, offset) 把该侧值变换到期望单位
        (file_value * scale + offset),变换依据写入 detail;
      - 未实现的登记类型在运行时拒绝(R4:不为未实现的转换放行)。
    transform 为 None 表示该侧不变换。
    """
    unit_left = read_left.unit
    unit_right = read_right.unit

    def lookup(unit_found: str) -> tuple[str | None, tuple | None]:
        if unit_found == var.unit:
            return None, None  # 已是期望单位
        spec = UNIT_ALIASES.get((var.name, unit_found, var.unit))
        if spec is None:
            return "unregistered", None
        kind = spec.get("type")
        if kind == "alias":
            return None, None  # 拼写别名:数值恒等,不变换
        if kind == "conversion":
            return None, (float(spec["scale"]), float(spec["offset"]))
        raise ValueError(
            f"UNIT_ALIASES: unsupported registration type {kind!r} for "
            f"({var.name}, {unit_found!r} -> {var.unit!r}); 未实现的转换不放行"
        )

    if read_left.unit_matches is None or read_right.unit_matches is None:
        return "unit_unknown", unit_left, unit_right, None, None
    status_l, transform_l = lookup(unit_left)
    status_r, transform_r = lookup(unit_right)
    if status_l or status_r:
        mismatch = (not read_left.unit_matches) or (not read_right.unit_matches)
        status = "unit_mismatch" if mismatch else "unit_unknown"
        return status, unit_left, unit_right, None, None
    return None, unit_left, unit_right, transform_l, transform_r


def _handoff_conclusion(stage_pair: str, stats_status: str,
                        frac_over: float, source_confirmed: bool,
                        mapping: MappingSpec | None, variable: str,
                        grid_verified: bool = True,
                        n_valid: int | None = None,
                        n_base: int | None = None) -> str:
    """I-A 的交接一致性结论(与数值可计算 status 分离;P1 #5;R1 修订)。

    前提链(任一不满足即 not_concluded_*):
      数值可计算 -> 网格已核验 -> 来源已确认 -> 映射已知且已核实;
    覆盖分层(R1):n_valid < n_base 时结论为 *on_valid_subset
    (共同有效点内一致),不把未核验点算作已验证一致。
    """
    if stage_pair != "I-A":
        return "n/a (not a handoff comparison)"
    if stats_status != "ok":
        return f"not_concluded:{stats_status}"
    if not grid_verified:
        return "not_concluded_grid_unverified"
    if not source_confirmed:
        return "not_concluded_source_unconfirmed"
    if mapping is None or mapping.variables is None:
        return "not_concluded_mapping_unknown"
    if variable not in mapping.variables:
        return "not_concluded_variable_outside_mapping"
    if not mapping.confirmed:
        return "not_concluded_mapping_inferred_not_confirmed"
    partial = n_valid is not None and n_base is not None and n_valid < n_base
    if frac_over == 0.0:
        return ("consistent_on_valid_subset" if partial
                else "consistent_within_tolerance")
    return ("differs_on_valid_subset" if partial
            else "differs_outside_tolerance")


def _write_synthetic_inputs(config: VerifyConfig) -> Path:
    """生成合成 B/A/I NetCDF(固定种子;Times 含全部配置时次)。

    场景(供手算核验,与冒烟测试断言对应):
    - OM_TMP:B 为基准;A = B + 同化增量(部分格点 ±0.5,均值 5/48);
      I = A(映射 M 内变量交接精确);
    - TSK:各阶段同值(模拟「OM_TMP 被同化改变而 TSK 未被更新」);
    - 大气变量:强弱试验的 I 完全相同(一致性检查预期≈0),B/A 各自不同;
    - SST 刻意缺失(检验 missing_variable 状态);
    - U/V 写在交错维上(检验 grid_unverified_staggered);
    - Times 含全部配置时次(检验按属性匹配;比较只用起报时次记录)。
    """
    from netCDF4 import Dataset

    synth_root = config.out_dir(SCRIPT_NAME) / "synthetic_inputs"
    synth_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261003)
    ny, nx = 6, 8
    lat = 13.0 + 0.25 * np.arange(ny)[:, None] * np.ones((1, nx))
    lon = 137.0 + 0.25 * np.arange(nx)[None, :] * np.ones((ny, 1))
    times = [name for _, name in config.times]
    n_times = len(times)
    ocean = np.ones((ny, nx), dtype=bool)

    pattern = rng.normal(0.0, 1.0, (ny, nx))
    inc = np.zeros((ny, nx))
    inc[:3, :4] = 0.5  # 同化增量仅部分格点非零(含 -0.5 抵消点)
    inc[0, 0] = -0.5

    for method in config.methods:
        for member in config.members:
            for experiment in (config.strong_experiment, config.weak_experiment):
                tag = "strong" if experiment == config.strong_experiment else "weak"
                offset = 0.6 if tag == "strong" else 0.0
                om_b = 300.0 + offset + 0.1 * pattern
                om_a = om_b + inc
                tsk = 302.0 + offset + 0.05 * pattern
                atmo_b = 2.0 + 0.01 * pattern
                atmo_a = atmo_b + 0.02
                atmo_i = 3.0 + 0.01 * pattern  # 强弱大气启动状态完全相同
                stages = {
                    "B": {"OM_TMP": om_b, "TSK": tsk, "atmo": atmo_b},
                    "A": {"OM_TMP": om_a, "TSK": tsk, "atmo": atmo_a},
                    "I": {"OM_TMP": om_a, "TSK": tsk, "atmo": atmo_i},
                }
                for stage, fields in stages.items():
                    path = synth_root / (
                        f"{stage}_{experiment}_{method}_{member}_{config.domain}.nc"
                    )
                    with Dataset(path, "w", format="NETCDF4") as ds:
                        ds.createDimension("Time", None)
                        ds.createDimension("ocean_layer_stag", 2)
                        ds.createDimension("bottom_top", 3)
                        ds.createDimension("south_north", ny)
                        ds.createDimension("west_east", nx)
                        ds.createDimension("south_north_stag", ny + 1)
                        ds.createDimension("west_east_stag", nx + 1)
                        times_var = ds.createVariable("Times", str, ("Time",))
                        for record, stamp in enumerate(times):
                            times_var[record] = stamp
                        v = ds.createVariable("XLAT", "f8", ("Time", "south_north", "west_east"))
                        v[0] = lat
                        v = ds.createVariable("XLONG", "f8", ("Time", "south_north", "west_east"))
                        v[0] = lon
                        v = ds.createVariable("LANDMASK", "f8", ("Time", "south_north", "west_east"))
                        v[0] = np.where(ocean, 0.0, 1.0)

                        def write_field(name, data, dims, unit):
                            var = ds.createVariable(name, "f8", dims)
                            var.units = unit
                            var[:] = data

                        om = np.zeros((n_times, 2, ny, nx))
                        for record in range(n_times):
                            om[record, 0] = fields["OM_TMP"] + 0.01 * record
                            om[record, 1] = fields["OM_TMP"] + 0.3
                        write_field("OM_TMP", om,
                                    ("Time", "ocean_layer_stag", "south_north", "west_east"), "K")
                        write_field("TSK", np.repeat(fields["TSK"][None], n_times, axis=0),
                                    ("Time", "south_north", "west_east"), "K")
                        atmo = np.zeros((3, ny, nx))
                        atmo[0] = fields["atmo"]
                        atmo[1:] = fields["atmo"] + 1.0
                        atmo_t = np.repeat(atmo[None], n_times, axis=0)
                        write_field("T", atmo_t, ("Time", "bottom_top", "south_north", "west_east"), "K")
                        write_field("QVAPOR", atmo_t * 0.001,
                                    ("Time", "bottom_top", "south_north", "west_east"), "kg kg-1")
                        u_t = np.zeros((n_times, 3, ny, nx + 1))
                        u_t[:, :, :, :-1] = atmo_t
                        u_t[:, :, :, -1] = u_t[:, :, :, -2]
                        write_field("U", u_t, ("Time", "bottom_top", "south_north", "west_east_stag"), "m s-1")
                        v_t = np.zeros((n_times, 3, ny + 1, nx))
                        v_t[:, :, :-1, :] = atmo_t
                        v_t[:, :, -1, :] = v_t[:, :, -2, :]
                        write_field("V", v_t, ("Time", "bottom_top", "south_north_stag", "west_east"), "m s-1")
                        write_field("PSFC", np.repeat(
                            (fields["atmo"] * 100.0 + 100000.0)[None], n_times, axis=0),
                            ("Time", "south_north", "west_east"), "Pa")
    return synth_root


def _variable_request(var: CheckVariable, time_name: str,
                      time_record: int | None) -> FieldRequest:
    return FieldRequest(
        variable=var.name, time_name=time_name, time_record=time_record,
        layer=var.layer, layer_dim=var.layer_dim, expected_unit=var.unit,
    )


def _verify04_config_snapshot(config: VerifyConfig,
                              stage_sources: dict[str, StageSource]) -> dict:
    """诊断四实际生效配置(P2 #8):阶段路径/确认标记/记录号、变量与容差、
    逐试验映射、单位别名、起报时次契约。"""
    return {
        "mode": config.mode,
        "cycle_id": CYCLE_ID,
        "init_time_name": INIT_TIME_NAME,
        "domain": config.domain,
        "strong_experiment": config.strong_experiment,
        "weak_experiment": config.weak_experiment,
        "methods": list(config.methods),
        "members": list(config.members),
        "stage_sources": {
            stage: {
                "path_template": info.path_template,
                "time_in_path": info.time_in_path,
                "time_record": info.time_record,
                "time_record_basis": info.time_record_basis,
                "mapping_basis": info.mapping_basis,
                "confirmed": info.confirmed,
                "format": info.format,
                "inferred_template": info.inferred_template,
            }
            for stage, info in stage_sources.items()
        },
        "check_variables": [dataclasses.asdict(var) for var in CHECK_VARIABLES],
        "mapping_specs": [
            {"experiment": spec.experiment,
             "variables": list(spec.variables) if spec.variables else None,
             "basis": spec.basis, "confirmed": spec.confirmed}
            for spec in MAPPING_SPECS.values()
        ],
        "post_mapping_steps": POST_MAPPING_STEPS,
        "unit_aliases": {f"{k[0]}:{k[1]}->{k[2]}": v for k, v in UNIT_ALIASES.items()},
        "fill_threshold": vc.FILL_THRESHOLD,
        "weight_method": vc.WEIGHT_LABEL,
    }


def run(config: VerifyConfig = CONFIG) -> Path:
    """执行初值传递核验;返回输出目录。合成模式自带合成输入。"""
    vc.check_mode(config)
    mode_tag = config.mode.upper()
    out_dir = vc.ensure_output_dir(config, SCRIPT_NAME)
    if config.mode == "synthetic":
        synth_root = _write_synthetic_inputs(config)
        print(f"[{mode_tag}] synthetic stage files -> {synth_root}", flush=True)

    stage_sources = _stage_source_configs(config)
    manifest_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    pair_rows: list[dict] = []
    status_rows: list[dict] = []
    coverage_counter: dict[tuple, dict] = {}

    def note(scope: str, status: str, detail: str = "") -> None:
        status_rows.append({"mode": mode_tag, "scope": scope, "status": status,
                            "detail": detail, "count": 1})

    def coverage_key(comparison: str, variable: str, method: str) -> tuple:
        return (comparison, variable, method)

    def bump_coverage(key: tuple, attempted: bool, completed: bool,
                      reason: str = "") -> None:
        entry = coverage_counter.setdefault(
            key, {"expected": 0, "attempted": 0, "completed": 0, "reasons": {}}
        )
        if attempted:
            entry["attempted"] += 1
        if completed:
            entry["completed"] += 1
        if not completed and reason:
            entry["reasons"][reason] = entry["reasons"].get(reason, 0) + 1

    def register_expected(comparison: str, variable: str, method: str,
                          units: int) -> None:
        key = coverage_key(comparison, variable, method)
        entry = coverage_counter.setdefault(
            key, {"expected": 0, "attempted": 0, "completed": 0, "reasons": {}}
        )
        entry["expected"] += units * n_members_cfg

    def read_stage_values(stage: str, var: CheckVariable, experiment: str,
                          method: str, member: str,
                          time_name: str) -> tuple[object, dict]:
        """读取一个 (阶段, 变量) 请求;返回 (FieldRead, manifest 行)。"""
        info = stage_sources[stage]
        path = _resolve_stage_path(
            info.path_template, config.domain, member, time_name,
            experiment, method,
        ) if info.path_template else None
        manifest = {
            "mode": mode_tag, "cycle_id": CYCLE_ID, "stage": stage,
            "experiment": experiment, "method": method, "member": member,
            "init_time": INIT_TIME_NAME, "variable": var.name,
            "layer": -1 if var.layer is None else var.layer,
            "path": path if path else "(not_configured)",
            "exists": "", "time_found": "", "time_source": "",
            "time_record_used": "", "unit_expected": var.unit,
            "unit_found": "", "unit_matches": "",
            "dims_order": "", "mapping_basis": info.mapping_basis,
            "mapping_confirmed": (
                MAPPING_SPECS[experiment].confirmed
                if experiment in MAPPING_SPECS else False
            ),
            "mapping_known": bool(
                experiment in MAPPING_SPECS
                and MAPPING_SPECS[experiment].variables
            ),
            "source_confirmed": info.confirmed, "format": info.format,
            "detail": "",
        }
        if path is None:
            manifest["detail"] = "source_not_configured"
            return None, manifest
        read = read_stage_field(
            path, _variable_request(var, time_name, time_record=info.time_record)
        )
        manifest["exists"] = "no" if read.status == "file_missing" else "yes"
        manifest["time_found"] = read.time_found or ""
        manifest["time_source"] = read.time_source
        manifest["time_record_used"] = (
            read.time_index if read.time_index is not None else ""
        )
        manifest["unit_found"] = read.unit or ""
        manifest["unit_matches"] = (
            "" if read.unit_matches is None else str(read.unit_matches)
        )
        manifest["dims_order"] = read.dims_order
        if read.status != "ok":
            manifest["detail"] = f"{read.status}: {read.detail}"[:200]
        return read, manifest

    seen_names: set[str] = set()
    for var in CHECK_VARIABLES:
        if var.name in seen_names and var.layer is None:
            raise ValueError(
                f"CHECK_VARIABLES: duplicate non-layered variable {var.name!r}; "
                "多层请使用不同 layer 配置"
            )
        seen_names.add(var.name)
    time_name = INIT_TIME_NAME
    init_hour = [hour for hour, name in config.times if name == time_name]
    if not init_hour:
        raise ValueError(
            f"INIT_TIME_NAME {INIT_TIME_NAME!r} 不在 config.times 中,无法确定 time_hour"
        )
    n_members_cfg = len(config.members)
    stages = ("B", "A", "I")

    for method in config.methods:
        # 预期比较数(起报时次契约:每方法每成员各一次;A-B/I-A 含强弱两试验)
        for var in CHECK_VARIABLES:
            for stage_pair in ("A-B", "I-A"):
                register_expected(stage_pair, var.name, method, units=2)
            for stage in stages:
                register_expected(f"pair@{stage}", var.name, method, units=1)

        for member in config.members:
            # ---- 逐试验逐阶段读取(每个文件一次;清单逐读取请求记录)----
            reads: dict[tuple[str, str, str], object] = {}
            grids: dict[tuple[str, str], tuple] = {}
            for experiment in (config.strong_experiment, config.weak_experiment):
                for stage in stages:
                    info = stage_sources[stage]
                    if info.path_template is None:
                        note(f"stage/{stage}/{experiment}/{method}/{member}",
                             "source_not_configured")
                        for var in CHECK_VARIABLES:
                            manifest_rows.append({
                                "mode": mode_tag, "cycle_id": CYCLE_ID,
                                "stage": stage, "experiment": experiment,
                                "method": method, "member": member,
                                "init_time": INIT_TIME_NAME,
                                "variable": var.name,
                                "layer": -1 if var.layer is None else var.layer,
                                "path": "(not_configured)", "exists": "",
                                "time_found": "", "time_source": "",
                                "time_record_used": "",
                                "unit_expected": var.unit, "unit_found": "",
                                "unit_matches": "", "dims_order": "",
                                "mapping_basis": info.mapping_basis,
                                "mapping_confirmed": False,
                                "mapping_known": False,
                                "source_confirmed": info.confirmed,
                                "format": info.format,
                                "detail": "source_not_configured",
                            })
                        continue
                    path = _resolve_stage_path(
                        info.path_template, config.domain, member,
                        time_name, experiment, method,
                    )
                    lat, lon, grid_status = read_stage_grid(
                        path, time_name, time_record=info.time_record
                    )
                    if grid_status != "ok":
                        note(f"stage/{stage}/{experiment}/{method}/{member}",
                             grid_status.split(":")[0], grid_status)
                    grids[(stage, experiment)] = (lat, lon)
                    for var in CHECK_VARIABLES:
                        read, manifest = read_stage_values(
                            stage, var, experiment, method, member, time_name
                        )
                        manifest_rows.append(manifest)
                        reads[(stage, experiment, var.name, var.layer)] = read

            # ---- d_assim = A - B / d_handoff = I - A(逐试验)----
            for experiment in (config.strong_experiment, config.weak_experiment):
                mapping = MAPPING_SPECS.get(experiment)
                for var in CHECK_VARIABLES:
                    for stage_pair, later, earlier in (
                        ("A-B", "A", "B"),
                        ("I-A", "I", "A"),
                    ):
                        cov = coverage_key(stage_pair, var.name, method)
                        read_later = reads.get((later, experiment, var.name, var.layer))
                        read_earlier = reads.get((earlier, experiment, var.name, var.layer))
                        source_confirmed = (
                            stage_sources[later].confirmed
                            and stage_sources[earlier].confirmed
                        )
                        non_handoff_conclusion = (
                            "n/a (not a handoff comparison)" if stage_pair == "A-B"
                            else "descriptive_not_a_handoff_check"
                        )
                        base = {
                            "mode": mode_tag, "cycle_id": CYCLE_ID,
                            "experiment": experiment, "method": method,
                            "member": member, "init_time": INIT_TIME_NAME,
                            "valid_time": time_name,
                            "time_basis": "cycle_initial_time",
                            "stage_pair": stage_pair, "variable": var.name,
                            "layer": -1 if var.layer is None else var.layer,
                            "unit_expected": var.unit,
                            "tolerance": var.tolerance, "role": var.role,
                            "staggered": var.staggered,
                            "mapping_basis": (
                                mapping.basis
                                if stage_pair == "I-A" and mapping else stage_pair
                            ),
                            "source_confirmed": source_confirmed,
                        }
                        if read_later is None or read_earlier is None:
                            pairwise_rows.append({
                                **base, "status": "source_not_configured",
                                "detail": "stage path not configured",
                                "handoff_conclusion": (
                                    "not_concluded:source_not_configured"
                                    if stage_pair == "I-A" else non_handoff_conclusion
                                ),
                            })
                            bump_coverage(cov, True, False, "source_not_configured")
                            continue
                        status_fail = None
                        for read in (read_later, read_earlier):
                            if read.status != "ok":
                                status_fail = (read.status, read.detail)
                                break
                        if status_fail:
                            pairwise_rows.append({
                                **base, "status": status_fail[0],
                                "detail": status_fail[1][:200],
                                "handoff_conclusion": (
                                    f"not_concluded:{status_fail[0]}"
                                    if stage_pair == "I-A" else non_handoff_conclusion
                                ),
                            })
                            bump_coverage(cov, True, False, status_fail[0])
                            continue
                        lat_l, lon_l = grids.get((later, experiment), (None, None))
                        lat_e, lon_e = grids.get((earlier, experiment), (None, None))
                        grid_ok, grid_detail, grid_verified = grids_match(
                            read_later, read_earlier, lat_l, lon_l, lat_e, lon_e,
                            staggered=var.staggered,
                        )
                        if not grid_ok:
                            pairwise_rows.append({
                                **base, "status": grid_detail.split(":")[0],
                                "detail": grid_detail,
                                "handoff_conclusion": (
                                    f"not_concluded:{grid_detail.split(':')[0]}"
                                    if stage_pair == "I-A" else non_handoff_conclusion
                                ),
                            })
                            bump_coverage(cov, True, False, grid_detail.split(":")[0])
                            continue
                        # ---- 单位门控(P1 #1;R4:登记的数值转换先变换)----
                        unit_status, unit_left, unit_right, tf_l, tf_r = _unit_check(
                            read_later, read_earlier, var
                        )
                        if unit_status is not None:
                            pairwise_rows.append({
                                **base, "status": unit_status,
                                "detail": f"units left={unit_left!r} right={unit_right!r}",
                                "unit_left": unit_left, "unit_right": unit_right,
                                "handoff_conclusion": (
                                    f"not_concluded:{unit_status}"
                                    if stage_pair == "I-A" else non_handoff_conclusion
                                ),
                            })
                            bump_coverage(cov, True, False, unit_status)
                            continue
                        values_l = read_later.values
                        values_e = read_earlier.values
                        if tf_l is not None:
                            values_l = values_l * tf_l[0] + tf_l[1]
                        if tf_r is not None:
                            values_e = values_e * tf_r[0] + tf_r[1]
                        stats = vc.stage_difference_stats(
                            values_l, values_e,
                            np.ones(values_l.shape, dtype=bool),
                            var.tolerance,
                        )
                        detail = ""
                        if tf_l is not None or tf_r is not None:
                            detail = "unit conversion applied (see UNIT_ALIASES)"
                        if stage_pair == "I-A":
                            mapping_vars = (
                                mapping.variables
                                if mapping and mapping.variables else ()
                            )
                            if var.name not in mapping_vars:
                                detail = (detail + ";" if detail else "") + \
                                    "mapping_unverified: variable mapping " \
                                    "unknown or outside M list"
                            if stats["n_fill"]:
                                detail = (detail + ";" if detail else "") + \
                                    "fill-magnitude values detected"
                        if var.staggered:
                            # R1:网格未核验时不给一致性结论,仅保留标记
                            detail = "grid_unverified_staggered"
                        conclusion = _handoff_conclusion(
                            stage_pair, stats["status"],
                            stats["frac_over_tolerance"],
                            source_confirmed, mapping, var.name,
                            grid_verified=grid_verified,
                            n_valid=stats["n_valid"], n_base=stats["n_base"],
                        )
                        pairwise_rows.append({
                            **base, "status": stats["status"], "detail": detail,
                            "unit_left": unit_left, "unit_right": unit_right,
                            "handoff_conclusion": conclusion,
                            **{k: stats[k] for k in (
                                "n_base", "n_valid", "n_missing",
                                "n_missing_left", "n_missing_right", "n_fill",
                                "mean_diff", "rms_diff", "max_abs_diff",
                                "n_over_tolerance", "frac_over_tolerance",
                            )},
                        })
                        bump_coverage(cov, True, stats["status"] == "ok",
                                      stats["status"] if stats["status"] != "ok" else "")

            # ---- d_pair = strong - weak(逐阶段;字段名与 d_assim 严格区分)----
            for stage in stages:
                for var in CHECK_VARIABLES:
                    cov = coverage_key(f"pair@{stage}", var.name, method)
                    read_s = reads.get((stage, config.strong_experiment, var.name, var.layer))
                    read_w = reads.get((stage, config.weak_experiment, var.name, var.layer))
                    base = {
                        "mode": mode_tag, "cycle_id": CYCLE_ID,
                        "method": method, "member": member,
                        "init_time": INIT_TIME_NAME, "valid_time": time_name,
                        "time_basis": "cycle_initial_time", "stage": stage,
                        "variable": var.name,
                        "layer": -1 if var.layer is None else var.layer,
                        "unit_expected": var.unit,
                        "tolerance": var.tolerance, "role": var.role,
                        "staggered": var.staggered,
                        "mapping_basis": (
                            MAPPING_SPECS[config.strong_experiment].basis
                            if stage == "I" else stage
                        ),
                    }
                    if read_s is None or read_w is None:
                        pair_rows.append({
                            **base, "status": "source_not_configured",
                            "detail": "stage path not configured",
                            "handoff_conclusion": "n/a (pair difference)",
                        })
                        bump_coverage(cov, True, False, "source_not_configured")
                        continue
                    if read_s.status != "ok" or read_w.status != "ok":
                        fail = read_s if read_s.status != "ok" else read_w
                        pair_rows.append({
                            **base, "status": fail.status,
                            "detail": fail.detail[:200],
                            "handoff_conclusion": "n/a (pair difference)",
                        })
                        bump_coverage(cov, True, False, fail.status)
                        continue
                    (unit_status, unit_s, unit_w,
                     tf_s, tf_w) = _unit_check(read_s, read_w, var)
                    if unit_status is not None:
                        pair_rows.append({
                            **base, "status": unit_status,
                            "detail": f"units strong={unit_s!r} weak={unit_w!r}",
                            "unit_strong": unit_s, "unit_weak": unit_w,
                            "handoff_conclusion": "n/a (pair difference)",
                        })
                        bump_coverage(cov, True, False, unit_status)
                        continue
                    lat_s, lon_s = grids.get((stage, config.strong_experiment), (None, None))
                    lat_w, lon_w = grids.get((stage, config.weak_experiment), (None, None))
                    grid_ok, grid_detail, grid_verified = grids_match(
                        read_s, read_w, lat_s, lon_s, lat_w, lon_w,
                        staggered=var.staggered,
                    )
                    if not grid_ok:
                        pair_rows.append({
                            **base, "status": grid_detail.split(":")[0],
                            "detail": grid_detail,
                            "handoff_conclusion": "n/a (pair difference)",
                        })
                        bump_coverage(cov, True, False, grid_detail.split(":")[0])
                        continue
                    values_s = read_s.values
                    values_w = read_w.values
                    if tf_s is not None:
                        values_s = values_s * tf_s[0] + tf_s[1]
                    if tf_w is not None:
                        values_w = values_w * tf_w[0] + tf_w[1]
                    stats = vc.stage_difference_stats(
                        values_s, values_w,
                        np.ones(values_s.shape, dtype=bool), var.tolerance,
                    )
                    # R1:大气一致性描述只在网格已核验时给出
                    detail = "grid_unverified_staggered" if var.staggered else ""
                    if (var.role == "atmosphere_pair" and stats["status"] == "ok"
                            and grid_verified):
                        detail = (
                            "atmosphere_pair_consistent_within_tolerance"
                            if stats["frac_over_tolerance"] == 0.0
                            else "atmosphere_pair_differs"
                        )
                        if (stats["frac_over_tolerance"] == 0.0
                                and stats["n_valid"] < stats["n_base"]):
                            detail = "atmosphere_pair_consistent_on_valid_subset"
                    pair_rows.append({
                        **base, "status": stats["status"], "detail": detail,
                        "unit_strong": unit_s, "unit_weak": unit_w,
                        "handoff_conclusion": "n/a (pair difference)",
                        **{
                            "n_base": stats["n_base"],
                            "n_valid": stats["n_valid"],
                            "n_missing": stats["n_missing"],
                            "n_missing_left": stats["n_missing_left"],
                            "n_missing_right": stats["n_missing_right"],
                            "n_fill": stats["n_fill"],
                            "mean_pair_diff": stats["mean_diff"],
                            "rms_pair_diff": stats["rms_diff"],
                            "max_abs_pair_diff": stats["max_abs_diff"],
                            "n_over_tolerance": stats["n_over_tolerance"],
                            "frac_over_tolerance": stats["frac_over_tolerance"],
                        },
                    })
                    bump_coverage(cov, True, stats["status"] == "ok",
                                  stats["status"] if stats["status"] != "ok" else "")

    # =====================
    # 输出(空输入也生成带列名的表)
    # =====================
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    pairwise = pd.DataFrame(pairwise_rows, columns=PAIRWISE_COLUMNS)
    pairs = pd.DataFrame(pair_rows, columns=PAIR_COLUMNS)
    vc.write_csv(manifest, out_dir / "verify04_source_manifest.csv")
    vc.write_csv(pairwise, out_dir / "verify04_pairwise_checks.csv")
    vc.write_csv(pairs, out_dir / "verify04_initial_pair_checks.csv")

    coverage_rows = []
    for (comparison, variable, method), entry in sorted(coverage_counter.items()):
        coverage_rows.append({
            "mode": mode_tag, "cycle_id": CYCLE_ID,
            "comparison": comparison, "variable": variable, "method": method,
            "expected": entry["expected"],
            "attempted": entry["attempted"], "completed": entry["completed"],
            "skipped": entry["attempted"] - entry["completed"],
            "not_attempted": entry["expected"] - entry["attempted"],
            "skipped_reasons": ";".join(
                f"{reason}:{count}" for reason, count in sorted(entry["reasons"].items())
            ),
        })
    coverage = pd.DataFrame(coverage_rows, columns=COVERAGE_COLUMNS)
    vc.write_csv(coverage, out_dir / "verify04_coverage.csv")

    status_summary = pd.DataFrame(status_rows, columns=STATUS_COLUMNS)
    if len(status_summary):
        status_summary = (
            status_summary.groupby(["mode", "scope", "status", "detail"], as_index=False)
            .agg(count=("count", "sum"))
        )
    vc.write_csv(status_summary, out_dir / "verify04_status.csv")

    # ---- 模块 B:口径对照表与兼容性结果 ----
    import caliber_link

    caliber_link.write_caliber_table(out_dir)

    vc.write_columns_readme(out_dir / "columns_readme.txt", _columns_readme_lines(config))
    n_ok = int((pairwise.status == "ok").sum()) if len(pairwise) else 0
    vc.write_overall_status(out_dir, n_ok, len(status_summary), mode_tag)
    snapshot_path = out_dir / "verify04_config_snapshot.json"
    snapshot_path.write_text(
        json.dumps(_verify04_config_snapshot(config, stage_sources),
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[output] {snapshot_path}", flush=True)
    vc.save_config_snapshot(out_dir, config)
    _print_summary(pairwise, pairs, manifest, mode_tag)
    return out_dir


def _columns_readme_lines(config: VerifyConfig) -> list[str]:
    return [
        "verify04 输出列说明(初值传递核验;SYNTHETIC/REAL 由 mode 列标记)",
        "",
        f"- 起报时次契约:cycle_id={CYCLE_ID},B/A/I 与 F0 只在 "
        f"INIT_TIME_NAME={INIT_TIME_NAME} 核验;lead time 不充当 cycle。",
        "三组差值的字段名严格区分,不混称「分析增量」:",
        "  d_assim  = A - B(stage_pair=A-B;同化增量)",
        "  d_handoff = I - A(stage_pair=I-A;交接差;映射 M 未核实的变量",
        "    记 mapping_unverified,不判通过/失败)",
        "  d_pair   = X_strong - X_weak(逐阶段;initial_pair 表,stage 列)",
        "- 单位门控:两侧实际 units 都与配置一致才计算;缺失=unit_unknown、",
        "  不符=unit_mismatch,均跳过并保留 unit_left/unit_right 原始单位;",
        "  隐式换算默认禁用(UNIT_ALIASES 须显式登记)。",
        "- 缺测/填充:基础掩膜 n_base 不随缺测收缩;n_missing 按侧分解",
        "  (n_missing_left/right);n_fill 为 |x|>=1e30 的填充量级点数",
        "  (相消填充不再伪装为零差)。",
        "- 网格:与字段同记录读取;质量点变量要求坐标形状=字段形状、",
        "  坐标无 NaN、两侧一致;交错变量(U/V)detail=grid_unverified_",
        "  staggered(质量点坐标不冒充交错位置核验)。",
        "- handoff_conclusion 与 status 分离:status=ok 只代表数值可计算;",
        "  结论需 source_confirmed 与逐试验 mapping(MappingSpec)核实,",
        "  否则 not_concluded_source_unconfirmed/mapping_unknown/",
        "  variable_outside_mapping/mapping_inferred_not_confirmed。",
        "- 阶段 A 默认指「后验合并文件」(firstguess 格式 + ncks 更新变量);",
        "  原始 DART 后验是 output_d0X,两者不同。",
        "- 状态:source_not_configured/file_missing/missing_variable/",
        "  time_not_found/time_ambiguous/invalid_time_record/",
        "  missing_dimension/grid_mismatch/grid_contains_nan/",
        "  grid_field_shape_mismatch/grid_unavailable/unit_unknown/",
        "  unit_mismatch/empty_mask/all_nan_inside_mask/ok;",
        "  不可评估不等于零差异或通过。",
        "- coverage 表:expected 为配置预期比较数(含强弱两试验),",
        "  attempted/completed/skipped/not_attempted 及原因逐条记录。",
        "- 映射 M 逐试验定义:强试验(oceanAssim1)=ncks.sh 12 变量清单;",
        "  弱试验(oceanAssim0)=ncks_air.sh 本地缺失,映射未知。",
        "- SST 默认在检查变量清单中但 wrfout/firstguess 通常无此变量,",
        "  将以 missing_variable 状态记录,不是错误。",
    ]


def _print_summary(pairwise, pairs, manifest, mode_tag: str) -> None:
    statuses = pairwise.status.value_counts().to_dict() if len(pairwise) else {}
    print(
        f"[{mode_tag}] verify04: {len(manifest)} manifest rows, "
        f"{len(pairwise)} pairwise rows (status: {statuses}), "
        f"{len(pairs)} pair rows",
        flush=True,
    )


if __name__ == "__main__":
    output_dir = run(CONFIG)
    print(f"[{CONFIG.mode.upper()}] verify04 done -> {output_dir}", flush=True)
