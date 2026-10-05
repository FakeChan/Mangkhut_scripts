"""verify_04 小型 NetCDF 读取适配器(初值传递核验专用)。

设计要点(对应任务书"读取和状态处理"):
- 按维度**名称**定位空间维(south_north/west_east)、海洋层维
  (ocean_layer_stag/ocean_layer)与垂直维(bottom_top/bottom_top_stag),
  不依赖维度顺序,不盲目取第 0 条时间记录;
- 时间记录:优先用 Times 属性逐文件匹配请求的有效时间;Times 缺失时才
  回退到显式提供的记录号,并标记 time_source=record_index;
- 网格校验:读取 XLAT/XLONG,提供 grids_match 判定;**不做任何插值**,
  网格不同即报告 grid_mismatch,由调用方跳过逐点比较;
- 单位:读取变量 units 属性;缺失或与期望不符只记录状态,不改数值;
- 只读打开("r"),绝不修改输入文件。

纯读取模块:导入无副作用,不探测任何真实路径。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import verify_common as vc

#: 空间维与层维的候选名称(按名称定位,与维度顺序无关)
Y_DIM_CANDIDATES = ("south_north", "y")
X_DIM_CANDIDATES = ("west_east", "x")
OCEAN_LAYER_CANDIDATES = ("ocean_layer_stag", "ocean_layer", "z_ocean")
VERT_DIM_CANDIDATES = ("bottom_top", "bottom_top_stag")
TIME_DIM_CANDIDATES = ("Time", "time")


@dataclass(frozen=True)
class FieldRequest:
    """一次字段读取请求。"""

    variable: str
    time_name: str                 # 请求的有效时间字符串(WRF 格式)
    time_record: int | None = None # Times 缺失时的回退记录号(须显式配置)
    layer: int | None = None       # 层维上的层号(OM_TMP 表层=0 等)
    layer_dim: str | None = None   # 期望的层维名称(约束校验;None=不校验)
    expected_unit: str | None = None


@dataclass(frozen=True)
class FieldRead:
    """一次字段读取的结果(values=None 时 status 说明原因)。"""

    values: np.ndarray | None
    status: str                    # ok / file_missing / missing_variable /
    detail: str                    # time_not_found / missing_dimension / unreadable
    dims: tuple = ()
    shape: tuple = ()
    unit: str | None = None
    unit_matches: bool | None = None   # None=文件未提供 units
    time_source: str = "none"      # times_attribute / record_index /
                                   # no_time_dimension / none
    time_found: str | None = None
    time_index: int | None = None  # 实际选中的时间记录号(R6 追溯)
    dims_order: str = ""           # 实际维度顺序,便于追溯


def _find_dim(dimensions, candidates) -> str | None:
    for name in candidates:
        if name in dimensions:
            return name
    return None


def read_stage_field(path, request: FieldRequest) -> FieldRead:
    """按维度名称与显式时间记录读取一个阶段字段,压成 (ny, nx) 或原维度。

    只读;任何失败都以状态返回,不抛异常(由调用方记录到 status 表)。
    """
    from netCDF4 import Dataset

    path = str(path)
    try:
        dataset = Dataset(path, "r")
    except (OSError, FileNotFoundError) as error:
        return FieldRead(None, "file_missing", str(error)[:200])

    try:
        if request.variable not in dataset.variables:
            return FieldRead(
                None, "missing_variable",
                f"{path}: variable {request.variable} absent",
                dims=tuple(dataset.dimensions),
            )
        variable = dataset.variables[request.variable]
        dims = variable.dimensions
        dims_order = ",".join(dims)
        shape = variable.shape

        # ---- 时间维定位与记录选择 ----
        time_dim = _find_dim(dims, TIME_DIM_CANDIDATES)
        time_index = None
        time_source = "none"
        time_found = None
        if time_dim is not None:
            time_size = dataset.dimensions[time_dim].size
            if "Times" in dataset.variables:
                times_var = dataset.variables["Times"]
                stamps = []
                for record in range(time_size):
                    stamps.append(vc.decode_wrf_time_stamp(times_var[record]))
                matches = [i for i, s in enumerate(stamps) if s == request.time_name]
                if len(matches) > 1:
                    return FieldRead(
                        None, "time_ambiguous",
                        f"{path}: {request.time_name!r} appears {len(matches)} "
                        f"times in Times {stamps}",
                        dims=dims, shape=shape, dims_order=dims_order,
                    )
                if matches:
                    time_index = matches[0]
                    time_source = "times_attribute"
                    time_found = stamps[time_index]
                else:
                    return FieldRead(
                        None, "time_not_found",
                        f"{path}: requested {request.time_name!r} not in Times "
                        f"{stamps}",
                        dims=dims, shape=shape, dims_order=dims_order,
                    )
            elif request.time_record is not None:
                if request.time_record < 0:
                    return FieldRead(
                        None, "invalid_time_record",
                        f"{path}: negative time_record {request.time_record}",
                        dims=dims, shape=shape, dims_order=dims_order,
                    )
                if request.time_record >= time_size:
                    return FieldRead(
                        None, "time_not_found",
                        f"{path}: record {request.time_record} >= {time_size}",
                        dims=dims, shape=shape, dims_order=dims_order,
                    )
                time_index = request.time_record
                time_source = "record_index"
            else:
                return FieldRead(
                    None, "time_not_found",
                    f"{path}: no Times variable and no explicitly configured "
                    "time_record (no blind record-0 default)",
                    dims=dims, shape=shape, dims_order=dims_order,
                )
        else:
            # 字段本身无时间维:允许,显式标记(静态初值文件可能如此)
            time_source = "no_time_dimension"

        # ---- 索引组装:仅按维度名称取固定层/时间,其余维度保留 ----
        index: list = []
        squeeze_axes: list[int] = []
        for axis, dim in enumerate(dims):
            if dim == time_dim and time_index is not None:
                index.append(time_index)
                squeeze_axes.append(axis)
            elif dim in OCEAN_LAYER_CANDIDATES:
                if request.layer is None:
                    return FieldRead(
                        None, "missing_dimension",
                        f"{path}: layer dimension {dim} present but no layer "
                        "requested",
                        dims=dims, shape=shape, dims_order=dims_order,
                    )
                if (request.layer_dim is not None
                        and dim != request.layer_dim):
                    return FieldRead(
                        None, "layer_dim_mismatch",
                        f"{path}: layer dim {dim!r} != expected "
                        f"{request.layer_dim!r}",
                        dims=dims, shape=shape, dims_order=dims_order,
                    )
                index.append(request.layer)
                squeeze_axes.append(axis)
            elif dim in VERT_DIM_CANDIDATES:
                if request.layer is None:
                    return FieldRead(
                        None, "missing_dimension",
                        f"{path}: vertical dim {dim} present but no level "
                        "requested",
                        dims=dims, shape=shape, dims_order=dims_order,
                    )
                if (request.layer_dim is not None
                        and dim != request.layer_dim):
                    return FieldRead(
                        None, "layer_dim_mismatch",
                        f"{path}: vertical dim {dim!r} != expected "
                        f"{request.layer_dim!r}",
                        dims=dims, shape=shape, dims_order=dims_order,
                    )
                index.append(request.layer)
                squeeze_axes.append(axis)
            else:
                index.append(slice(None))

        values = variable[tuple(index)]
        values = np.asarray(np.ma.filled(values, np.nan), dtype=float)

        # ---- 空间维归一化:按名称把 (west_east, south_north) 顺序转正为
        # (south_north, west_east),使维度换序文件与其他阶段可比 ----
        remaining = [d for axis, d in enumerate(dims) if axis not in squeeze_axes]
        if "south_north" in remaining and "west_east" in remaining:
            if remaining.index("west_east") < remaining.index("south_north"):
                axis_y = remaining.index("south_north")
                axis_x = remaining.index("west_east")
                # 文件顺序为 (west_east, south_north):转置为 (south_north, west_east)
                values = np.transpose(values, (axis_y, axis_x))

        # ---- 单位 ----
        unit = getattr(variable, "units", None)
        unit_matches = (
            None if unit is None or request.expected_unit is None
            else str(unit).strip() == request.expected_unit
        )

        return FieldRead(
            values, "ok", "", dims=dims, shape=shape,
            unit=None if unit is None else str(unit),
            unit_matches=unit_matches,
            time_source=time_source, time_found=time_found,
            time_index=time_index,
            dims_order=dims_order,
        )
    except Exception as error:  # 读取异常统一转为状态,不中断整个诊断
        return FieldRead(None, "unreadable", f"{type(error).__name__}: {error}")
    finally:
        dataset.close()


def read_stage_grid(path, time_name: str | None = None,
                    time_record: int | None = None) -> tuple[
        np.ndarray | None, np.ndarray | None, str
]:
    """读取阶段的 XLAT/XLONG 网格,与字段绑定同一时间记录(P1 #4)。

    - 时间选择与字段相同:优先 Times 属性匹配;缺失时用显式记录号
      (time_record,须由配置提供);两者都无 → time_not_found;
      网格无时间维 → 原样使用(no_time_dimension);
    - 按维度名称定位并归一化为 (south_north, west_east);
    - 不检查 NaN(由 grids_match 统一判定并报告)。
    失败返回 (None, None, 状态)。
    """
    from netCDF4 import Dataset

    path = str(path)
    try:
        dataset = Dataset(path, "r")
    except (OSError, FileNotFoundError) as error:
        return None, None, f"file_missing: {error}"[:200]
    try:
        missing = [name for name in ("XLAT", "XLONG") if name not in dataset.variables]
        if missing:
            return None, None, f"missing_variable: {','.join(missing)}"

        def read_coord(name: str) -> np.ndarray | None:
            variable = dataset.variables[name]
            dims = variable.dimensions
            time_dim = _find_dim(dims, TIME_DIM_CANDIDATES)
            # 先按与字段相同的时间规则解析记录号(R3:时间维可在任意轴位)
            record = None
            has_time = time_dim is not None
            if has_time:
                time_size = dataset.dimensions[time_dim].size
                if "Times" in dataset.variables:
                    stamps = [
                        vc.decode_wrf_time_stamp(dataset.variables["Times"][r])
                        for r in range(time_size)
                    ]
                    if time_name is None:
                        return None
                    matches = [i for i, s in enumerate(stamps) if s == time_name]
                    if len(matches) > 1:
                        raise ValueError(f"time_ambiguous: {name}")
                    if not matches:
                        raise ValueError(f"time_not_found: {name}")
                    record = matches[0]
                elif time_record is not None:
                    if time_record < 0 or time_record >= time_size:
                        raise ValueError(f"time_not_found: {name}")
                    record = time_record
                else:
                    return None
            # 逐轴构造索引(按维度名称,不假设时间维在第 0 轴)
            index: list = []
            squeeze: list[int] = []
            for axis, dim in enumerate(dims):
                if has_time and dim == time_dim:
                    index.append(record)
                    squeeze.append(axis)
                elif dim in ("south_north", "west_east",
                             "south_north_stag", "west_east_stag"):
                    index.append(slice(None))
                else:
                    raise ValueError(f"unexpected_dimension: {name}:{dim}")
            values = np.asarray(
                np.ma.filled(variable[tuple(index)], np.nan), dtype=float
            )
            remaining = [d for axis, d in enumerate(dims)
                         if axis not in squeeze
                         and d in ("south_north", "west_east",
                                    "south_north_stag", "west_east_stag")]
            if len(remaining) == 2 and remaining.index("west_east") < remaining.index("south_north"):
                values = np.transpose(values)
            return values

        lat = read_coord("XLAT")
        lon = read_coord("XLONG")
        if lat is None or lon is None:
            return None, None, "time_not_found: no Times and no explicit record"
        return lat, lon, "ok"
    except Exception as error:
        return None, None, f"unreadable: {type(error).__name__}: {error}"[:200]
    finally:
        dataset.close()


def grids_match(read_a: FieldRead, read_b: FieldRead, lat_a, lon_a, lat_b, lon_b,
                staggered: bool = False) -> tuple[bool, str, bool]:
    """比较两侧读取结果的形状与网格(R1 修订)。

    返回 (allowed, detail, grid_verified):
    - 形状不一致 → (False, shape_mismatch, False);
    - 质量点变量:坐标形状必须与字段形状一致(位置对应),坐标含 NaN 或
      两侧不一致 → (False, grid_contains_nan/grid_mismatch, False);
      全部通过 → (True, "ok", True);
    - 交错变量(U/V):保留原生交错网格,质量点坐标不能冒充交错位置核验
      → (True, "grid_unverified_staggered", False):允许描述统计,
      但 grid_verified=False 阻止任何交接/一致性结论(R1);
    - 坐标缺失 → (False, grid_unavailable, False)。
    """
    if read_a.values is None or read_b.values is None:
        return False, "missing_values", False
    if read_a.values.shape != read_b.values.shape:
        return False, (
            f"shape_mismatch:{read_a.values.shape} vs {read_b.values.shape}"
        ), False
    if staggered:
        # 两侧同为交错场且形状一致:几何无法用质量点坐标核验,显式标记;
        # verified=False 由调用方阻止一致性结论(R1)
        return True, "grid_unverified_staggered", False
    if lat_a is None or lat_b is None or lon_a is None or lon_b is None:
        return False, "grid_unavailable", False
    if lat_a.shape != read_a.values.shape:
        return False, (
            f"grid_field_shape_mismatch: coord{lat_a.shape} vs field{read_a.values.shape}"
        ), False
    if lat_a.shape != lat_b.shape:
        return False, f"grid_shape_mismatch:{lat_a.shape} vs {lat_b.shape}", False
    if bool(np.any(~np.isfinite(lat_a))) or bool(np.any(~np.isfinite(lon_a))) \
            or bool(np.any(~np.isfinite(lat_b))) or bool(np.any(~np.isfinite(lon_b))):
        return False, "grid_contains_nan", False
    if not vc.grids_identical(lat_a, lon_a, lat_b, lon_b):
        return False, "grid_mismatch", False
    return True, "ok", True
