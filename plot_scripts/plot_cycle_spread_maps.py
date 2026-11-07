"""计算并绘制六组 d02 全域逐格点集合离散度，并叠加 NR 最小 PSFC 位置。

编辑下方 CONFIG 后直接运行本文件，不使用命令行参数。
行顺序：强耦合 EAKF/QCF_RHF、弱耦合 EAKF/QCF_RHF、无耦合 EAKF/QCF_RHF。
列顺序：2018-09-10 00–06 UTC，每小时一次，包含起止时间。

spread(y,x) = sqrt(sum_m((X_m(y,x) - mean_m(X_m(y,x)))**2) / (N-1))

先逐格点沿集合维计算标准差，保留整个 d02 二维场（包含陆地和边界）。
不对每个成员先做空间平均，不拼接成员网格，不使用台风半径裁剪。
某格点必须具有全部成员的有效值，否则该点为 NaN；缺成员文件直接报错。
全 NaN 面板不绘制，全时段没有有效数据的行从图中移除；部分 NaN 格点透明。
无耦合试验不运行海洋模式，OM_TMP 对应两行只保留缺测数据记录，不绘图。
每个试验/filter 独立取集合平均 PSFC 的最小值作为中心；当前时刻叠加从
起始时刻到当前时刻的中心轨迹。历史经纬度映射到当前网格，适用于移动 d02。
每个面板另用橙色小叉标出该时刻 NR（自然运行）PSFC 最小值的位置，
NR 经纬度同样映射到当前面板网格。
额外输出全域 RMS spread = sqrt(mean_yx(spread**2))，仅对有效格点等权平均。

支持 wrfout 中原始二维变量或原始三维变量的某个模式层；vertical_level
从 0 开始，不是 hPa 插值层。THM/T 等原始变量保持其文件内定义，不转换为
实际温度；U/V 等交错变量保留原网格。色标单位取自变量的 units 属性。

输出：PNG/PDF 对比图、含二维 spread/有效成员数/经纬度/台风中心/NR 最小 PSFC
位置的 NetCDF、逐试验逐时刻的全域 RMS spread、有效格点数与中心位置 CSV。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr


# =====================================================================
# 用户配置：仅需修改此处
# =====================================================================
CONFIG = {
    "base_dir": Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test"),
    "output_dir": Path(__file__).resolve().parent / "figs" / "d02_spread_maps",
    "domain": "d02",
    "groups": (
        {"label": "Strong coupling", "directory": "6mem_oceanAssim1Run1"},  # 强耦合
        {"label": "Weak coupling", "directory": "6mem_oceanAssim0Run1"},    # 弱耦合
        {"label": "Uncoupled", "directory": "6mem_oceanAssim0Run0",
         "unavailable_variables": ("OM_TMP",)},  # 不运行海洋模式；此变量两行不绘图
    ),
    "filters": ("EAKF", "QCF_RHF"),
    "members": ("006", "015", "029", "037", "043", "044"),
    # NR 真值：同一 d02 域，每个绘制时刻一个文件；缺文件直接报错。
    "nr_base_dir": Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain"),
    "nr_file_template": "wrfout_{domain}_{time}",  # 也接受同名 .nc
    "nr_variable": "PSFC",
    # 文件路径：base_dir/试验/filter/成员/wrfout_d02_时间。
    # 默认直接查找，再递归查找；若多个 cycle 中存在同名文件则报错。
    # 此时将 member_subdir 改成相对成员目录的指定预报子目录。
    "member_subdir": "",
    "start_time": "2018-09-10_00:00:00",  # WRF UTC 时间
    "end_time": "2018-09-10_06:00:00",
    "step_minutes": 60,
    "variable": "TSK",                # 二维例：TSK、T2、PSFC、HFX
    "vertical_level": 0,              # 三维例：QVAPOR、OM_TMP；二维变量忽略此项
    "scale": 1.0,                     # 单位换算，如 PSFC: 0.01 (Pa -> hPa)
    "unit": None,                     # None 自动读取 units；换算后应显式写如 'hPa'
    "fill_threshold": 1e30,
    "grid_tolerance_deg": 1e-6,        # 同一集合内经纬度一致性检查
    "cmap": "YlGnBu",                # 非负连续色标，所有子图共用
    "vmax": None,                     # None 采用全部子图最大值；亦可指定固定上限
    "panel_size": (2.6, 2.6),          # 每个子图的宽、高（英寸）
    "figure_formats": ("png", "pdf"),
    "dpi": 300,
    "track_color": "#222222",         # 历史中心：圆点和连线
    "current_center_color": "#d73027", # 当前中心：红色星号
    "track_linewidth": 1.0,
    "history_marker_size": 3.5,
    "current_marker_size": 10.0,
    "nr_marker": "x",                  # NR 最小 PSFC：橙色小叉
    "nr_marker_color": "#ff7f00",
    "nr_marker_size": 6.0,
    "nr_marker_width": 1.3,
}

TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


def build_times(config: dict) -> list[datetime]:
    start = datetime.strptime(config["start_time"], TIME_FORMAT)
    end = datetime.strptime(config["end_time"], TIME_FORMAT)
    minutes = config["step_minutes"]
    if minutes <= 0 or end < start or (end - start).total_seconds() % (60 * minutes):
        raise ValueError("Time range must be nonnegative and divisible by step_minutes > 0")
    steps = int((end - start).total_seconds() / (minutes * 60))
    return [start + timedelta(minutes=i * minutes) for i in range(steps + 1)]


def find_member_file(config: dict, experiment: str, method: str,
                     member: str, time: datetime) -> Path:
    root = (Path(config["base_dir"]) / experiment / method / member
            / config["member_subdir"])
    name = f"wrfout_{config['domain']}_{time.strftime(TIME_FORMAT)}"
    names = (name, name + ".nc")
    matches = [root / n for n in names if (root / n).is_file()]
    if not matches and root.is_dir():
        matches = sorted(p for n in names for p in root.rglob(n) if p.is_file())
    if not matches:
        raise FileNotFoundError(f"Missing member forecast: {root / name}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous forecast files for {root / name}: {matches}. "
                         "Set CONFIG['member_subdir'] to one forecast directory.")
    return matches[0]


def _time_index(ds: xr.Dataset, time: datetime, path: Path) -> int:
    if "Times" not in ds:
        if ds.sizes.get("Time", 1) != 1:
            raise ValueError(f"Multiple Time records without WRF Times: {path}")
        return 0  # 单时次文件可通过文件名识别时间
    stamps = []
    raw = np.asarray(ds["Times"].values)
    for record in raw:
        if raw.dtype.kind == "S":
            stamp = np.asarray(record).tobytes().decode("ascii")
        else:
            stamp = "".join(np.asarray(record).ravel().astype(str))
        stamps.append(stamp.rstrip("\x00 "))
    matches = [i for i, stamp in enumerate(stamps) if stamp == time.strftime(TIME_FORMAT)]
    if len(matches) != 1:
        raise ValueError(f"Expected one WRF Times record for {time} in {path}, got {stamps}")
    return matches[0]


def _without_time(arr: xr.DataArray, index: int) -> xr.DataArray:
    return arr.isel(Time=index) if "Time" in arr.dims else arr


def read_member_field(path: Path, time: datetime, config: dict):
    """只读取指定层，返回 field、lat、lon、原始单位；检查水平维度。"""
    with xr.open_dataset(path, decode_times=False) as ds:
        if config["variable"] not in ds:
            ocean_variables = [name for name in ds.data_vars if name.startswith("OM_")]
            raise KeyError(f"Variable {config['variable']!r} not found in file: {path}. "
                           f"Available OM_ variables: {ocean_variables}. "
                           "Check the variable name and the experiment's ocean output.")
        index = _time_index(ds, time, path)
        arr = _without_time(ds[config["variable"]], index)
        yd = next((d for d in arr.dims if d in ("south_north", "south_north_stag")), None)
        xd = next((d for d in arr.dims if d in ("west_east", "west_east_stag")), None)
        if yd is None or xd is None:
            raise ValueError(f"Expected WRF horizontal dimensions in {path}, got {arr.dims}")
        vertical_dims = [d for d in arr.dims if d not in (yd, xd)]
        if len(vertical_dims) == 1:
            z = vertical_dims[0]
            level = config["vertical_level"]
            if (isinstance(level, bool) or not isinstance(level, (int, np.integer))
                    or not 0 <= level < arr.sizes[z]):
                raise ValueError(f"vertical_level={level} outside [0, {arr.sizes[z]-1}] "
                                 f"for {z} in {path}")
            arr = arr.isel({z: level})
        elif vertical_dims:
            raise ValueError(f"Expected 2-D or one vertical dimension, got {arr.dims}: {path}")
        arr = arr.transpose(yd, xd)
        suffix = "_U" if xd.endswith("_stag") else "_V" if yd.endswith("_stag") else ""
        lat = _without_time(ds[f"XLAT{suffix}"], index).transpose(yd, xd).values.astype(float)
        lon = _without_time(ds[f"XLONG{suffix}"], index).transpose(yd, xd).values.astype(float)
        field = arr.values.astype(float)
        unit = str(arr.attrs.get("units", "unknown"))
    if field.shape != lat.shape or field.shape != lon.shape:
        raise ValueError(f"Field/grid shape mismatch: {path}")
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError(f"Invalid geographic grid: {path}")
    valid = np.isfinite(field) & (np.abs(field) < config["fill_threshold"])
    return np.where(valid, field, np.nan), lat, lon, unit


def find_nr_file(config: dict, time: datetime) -> Path:
    """Return the NR file for one valid time; a missing file is fatal."""
    root = Path(config["nr_base_dir"])
    name = config["nr_file_template"].format(
        domain=config["domain"], time=time.strftime(TIME_FORMAT))
    for candidate in (root / name, root / (name + ".nc")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing NR file: {root / name}")


def nr_min_psfc(config: dict, time: datetime):
    """Location, value and unit of the NR minimum PSFC at one valid time."""
    path = find_nr_file(config, time)
    nr_config = dict(config, variable=config["nr_variable"], vertical_level=0)
    field, lat, lon, unit = read_member_field(path, time, nr_config)
    if not np.isfinite(field).any():
        raise ValueError(f"No valid {config['nr_variable']} in NR file: {path}")
    j, i = np.unravel_index(np.nanargmin(field), field.shape)
    return float(lat[j, i]), float(lon[j, i]), float(field[j, i]), unit, path


def calc_ensemble_spread(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """沿成员维计算样本标准差；仅保留全部成员有效的格点。"""
    fields = np.asarray(fields, dtype=float)
    if fields.ndim != 3 or fields.shape[0] < 2:
        raise ValueError("Expected (N_members >= 2, Ny, Nx)")
    counts = np.isfinite(fields).sum(axis=0)
    complete = counts == fields.shape[0]
    spread = np.full(fields.shape[1:], np.nan)
    # 不使用 nanstd，以免缺测时悄悄改变集合大小；避免全 NaN 的运行警告。
    spread[complete] = np.std(fields[:, complete], axis=0, ddof=1)
    return spread, counts


def ensemble_mean_center(pressures, lat, lon) -> tuple[float, float, float]:
    """先求六成员逐格点 PSFC 平均，再取最小值；不是平均各成员的中心。"""
    pressures = np.stack(pressures)
    complete = np.isfinite(pressures).all(axis=0)
    if not complete.any():
        return np.nan, np.nan, np.nan
    mean_pressure = np.full(complete.shape, np.nan)
    mean_pressure[complete] = pressures[:, complete].mean(axis=0)
    j, i = np.unravel_index(np.nanargmin(mean_pressure), mean_pressure.shape)
    return float(lat[j, i]), float(lon[j, i]), float(mean_pressure[j, i])


def centers_on_grid(lat, lon, center_lats, center_lons) -> np.ndarray:
    """将历史经纬度映射到当前模型网格的 (x,y)；域外点保持 NaN。

    最近格点附近用经纬度的局部二维线性变换求小数下标，可处理移动嵌套、
    曲线网格和 U/V 交错网格；不会将远离当前区域的历史中心强行画在边界。
    """
    ny, nx = lat.shape
    positions = np.full((len(center_lats), 2), np.nan)
    for k, (clat, clon) in enumerate(zip(center_lats, center_lons)):
        if not np.isfinite([clat, clon]).all():
            continue
        east = ((lon - clon + 180) % 360 - 180) * np.cos(np.deg2rad(clat))
        north = lat - clat
        j, i = np.unravel_index(np.argmin(east**2 + north**2), lat.shape)
        if ny < 2 or nx < 2:
            if east[j, i]**2 + north[j, i]**2 < 1e-16:
                positions[k] = i, j
            continue
        i0, i1, j0, j1 = max(0, i-1), min(nx-1, i+1), max(0, j-1), min(ny-1, j+1)
        jacobian = np.array([
            [(east[j, i1]-east[j, i0])/(i1-i0), (east[j1, i]-east[j0, i])/(j1-j0)],
            [(north[j, i1]-north[j, i0])/(i1-i0), (north[j1, i]-north[j0, i])/(j1-j0)],
        ])
        try:
            dx, dy = np.linalg.solve(jacobian, -np.array([east[j, i], north[j, i]]))
        except np.linalg.LinAlgError:
            continue
        x, y = i + dx, j + dy
        if -0.5 <= x <= nx-0.5 and -0.5 <= y <= ny-0.5:
            positions[k] = x, y
    return positions


def calculate(config: dict) -> xr.Dataset:
    times = build_times(config)
    members = config["members"]
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("Configure at least two distinct ensemble members")
    if not np.isfinite(config["scale"]) or config["scale"] <= 0:
        raise ValueError("scale must be finite and positive")
    nr_centers, nr_units = [], set()
    for time in times:
        nr_lat, nr_lon, nr_p, nr_unit, nr_path = nr_min_psfc(config, time)
        nr_centers.append((nr_lat, nr_lon, nr_p))
        nr_units.add(nr_unit)
        print(f"NR / {time:%Y-%m-%d %H:%M} UTC: min {config['nr_variable']} = "
              f"{nr_p:.1f} {nr_unit} at {nr_lat:.3f}N, {nr_lon:.3f}E ({nr_path.name})")
    if len(nr_units) != 1:
        raise ValueError(f"Inconsistent NR units across times: {nr_units}")
    rows = [(group, method) for group in config["groups"] for method in config["filters"]]
    records, source_files, statuses, reasons, units = [], [], [], [], set()
    centers, pressure_units = [], set()
    pressure_config = dict(config, variable="PSFC", vertical_level=0)
    for group, method in rows:
        if config["variable"] in group.get("unavailable_variables", ()):
            reason = f"{config['variable']} unavailable: ocean model not run"
            records.append([None] * len(times))
            source_files.append([[""] * len(members) for _ in times])
            statuses.append(["not_applicable"] * len(times))
            reasons.append([reason] * len(times))
            centers.append([(np.nan, np.nan, np.nan)] * len(times))
            print(f"N/A: {group['label']} / {method}: {reason}")
            continue
        row_records, row_files, row_centers, row_status, row_reason = [], [], [], [], []
        for time in times:
            fields, paths, pressures = [], [], []
            for member in members:
                path = find_member_file(config, group["directory"], method, member, time)
                field, lat, lon, unit = read_member_field(path, time, config)
                units.add(unit)
                if not fields:
                    ref_lat, ref_lon = lat, lon
                elif (field.shape != fields[0].shape
                      or not np.allclose(lat, ref_lat, atol=config["grid_tolerance_deg"], rtol=0)
                      or not np.allclose(lon, ref_lon, atol=config["grid_tolerance_deg"], rtol=0)):
                    raise ValueError(f"Ensemble grid mismatch: {path}")
                if config["variable"] == "PSFC":
                    pressure, plat, plon, pressure_unit = field, lat, lon, unit
                else:
                    pressure, plat, plon, pressure_unit = read_member_field(path, time, pressure_config)
                pressure_units.add(pressure_unit)
                if not pressures:
                    pressure_lat, pressure_lon = plat, plon
                elif (pressure.shape != pressures[0].shape
                      or not np.allclose(plat, pressure_lat, atol=config["grid_tolerance_deg"], rtol=0)
                      or not np.allclose(plon, pressure_lon, atol=config["grid_tolerance_deg"], rtol=0)):
                    raise ValueError(f"PSFC ensemble grid mismatch: {path}")
                pressures.append(pressure)
                fields.append(field)
                paths.append(str(path))
            spread, counts = calc_ensemble_spread(np.stack(fields))
            spread *= config["scale"]
            has_data = np.isfinite(spread).any()
            row_status.append("ok" if has_data else "no_valid_data")
            row_reason.append("" if has_data else "No complete-member grid points; panel not plotted")
            row_centers.append(ensemble_mean_center(pressures, pressure_lat, pressure_lon))
            row_records.append((spread, counts, ref_lat, ref_lon))
            row_files.append(paths)
            print(f"{group['label']} / {method} / {time:%Y-%m-%d %H:%M} UTC: "
                  f"N={len(fields)}, valid grid points={np.isfinite(spread).sum()}/{spread.size}")
        records.append(row_records)
        source_files.append(row_files)
        statuses.append(row_status)
        reasons.append(row_reason)
        centers.append(row_centers)
    if not units:
        raise ValueError(f"No applicable experiments for {config['variable']}")
    if len(units) != 1:
        raise ValueError(f"Inconsistent input units across members/experiments: {units}")
    if len(pressure_units) != 1:
        raise ValueError(f"Inconsistent PSFC units across members/experiments: {pressure_units}")
    # 同一变量在各试验/时次的 d02 水平形状应一致；经纬度随时次可变化。
    shapes = {record[0].shape for row in records for record in row if record is not None}
    if len(shapes) != 1:
        raise ValueError(f"Different d02 grid shapes across panels: {shapes}")
    dims = ("row", "time", "y", "x")
    shape = next(iter(shapes))
    empty = (np.full(shape, np.nan), np.zeros(shape, dtype=np.int16),
             np.full(shape, np.nan), np.full(shape, np.nan))
    variables = {name: (dims, np.stack([[(r if r is not None else empty)[k]
                                       for r in row] for row in records]))
                 for k, name in enumerate(("spread", "valid_member_count", "latitude", "longitude"))}
    variables["source_file"] = (("row", "time", "member"), np.asarray(source_files))
    variables["panel_status"] = (("row", "time"), np.asarray(statuses))
    variables["panel_reason"] = (("row", "time"), np.asarray(reasons))
    variables["members_used"] = (("row", "time"),
                                  np.where(np.asarray(statuses) != "not_applicable", len(members), 0))
    for k, name in enumerate(("center_latitude", "center_longitude", "center_psfc")):
        variables[name] = (("row", "time"), np.asarray(centers)[:, :, k])
    for k, name in enumerate(("nr_min_latitude", "nr_min_longitude", "nr_min_psfc")):
        variables[name] = (("time",), np.asarray(nr_centers)[:, k])
    ds = xr.Dataset(variables, coords={
        "row": np.arange(len(rows)), "time": np.array(times, dtype="datetime64[ns]"),
        "member": list(members),
        "coupling": ("row", [group["label"] for group, _ in rows]),
        "experiment": ("row", [group["directory"] for group, _ in rows]),
        "filter": ("row", [method for _, method in rows]),
    })
    unit = config["unit"] if config["unit"] is not None else units.pop()
    if config["scale"] != 1.0 and config["unit"] is None:
        raise ValueError("Set CONFIG['unit'] explicitly when scale changes the input units")
    ds.spread.attrs = {"units": unit, "long_name": "Pointwise ensemble sample standard deviation",
                       "ddof": 1, "required_members": len(members)}
    ds["valid_member_count"] = ds.valid_member_count.astype(np.int16)
    ds["valid_grid_points"] = ds.spread.count(("y", "x"))
    ds["domain_rms_spread"] = np.sqrt((ds.spread ** 2).mean(("y", "x"), skipna=True))
    ds.domain_rms_spread.attrs = {"units": unit, "definition": "sqrt(mean_xy(spread**2)); equal grid weights"}
    ds.latitude.attrs["units"] = "degrees_north"
    ds.longitude.attrs["units"] = "degrees_east"
    ds.center_latitude.attrs["units"] = "degrees_north"
    ds.center_longitude.attrs["units"] = "degrees_east"
    ds.center_psfc.attrs = {"units": pressure_units.pop(),
                            "definition": "Minimum of ensemble-mean PSFC on full d02 mass grid; "
                                          "all members required at each point"}
    ds.nr_min_latitude.attrs["units"] = "degrees_north"
    ds.nr_min_longitude.attrs["units"] = "degrees_east"
    ds.nr_min_psfc.attrs = {"units": nr_units.pop(),
                            "definition": "Minimum of NR PSFC on the NR d02 mass grid at each time"}
    ds.attrs = {"variable": config["variable"], "domain": config["domain"],
                "vertical_level": config["vertical_level"], "time_zone": "UTC",
                "spatial_support": "Whole d02 including land and boundary points; no cropping",
                "missing_policy": "All members required at each point; otherwise spread is NaN. "
                                  "Configured unavailable variables are N/A, with members_used=0.",
                "scale": config["scale"]}
    return ds


def plot_spread(ds: xr.Dataset, config: dict):
    """只画有效数据行/面板；叠加该行从起始时刻到当前时刻的中心轨迹。"""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import Normalize

    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 9})
    panel_valid = np.isfinite(ds.spread.values).any(axis=(2, 3))
    row_indices = np.flatnonzero(panel_valid.any(axis=1))
    if not len(row_indices):
        raise ValueError("No finite spread data to plot")
    nrows, ncols = len(row_indices), ds.sizes["time"]
    shown_panels = panel_valid[row_indices]
    width, height = config["panel_size"]
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(width * ncols + 1.4, height * nrows + 0.8),
                             constrained_layout=True)
    vmax = config["vmax"]
    if vmax is None:
        vmax = float(ds.spread.max(skipna=True)) or 1.0  # 零离散度亦能绘图
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("vmax must be finite and positive")
    norm = Normalize(0, vmax)
    cmap = plt.get_cmap(config["cmap"]).with_extremes(bad=(1, 1, 1, 0))
    active_axes = []
    for i, row in enumerate(row_indices):
        for j in range(ncols):
            ax = axes[i, j]
            if not shown_panels[i, j]:
                ax.set_visible(False)
                continue
            active_axes.append(ax)
            image = ax.imshow(ds.spread.values[row, j], origin="lower", aspect="equal",
                              interpolation="nearest", cmap=cmap, norm=norm)
            if "center_latitude" in ds and "center_longitude" in ds:
                positions = centers_on_grid(ds.latitude.values[row, j], ds.longitude.values[row, j],
                                            ds.center_latitude.values[row, :j+1],
                                            ds.center_longitude.values[row, :j+1])
                trail, = ax.plot(positions[:, 0], positions[:, 1], "o-",
                                 color=config["track_color"],
                                 linewidth=config["track_linewidth"],
                                 markersize=config["history_marker_size"],
                                 markerfacecolor="white", label="TC track", zorder=4)
                trail.set_path_effects([pe.Stroke(linewidth=config["track_linewidth"]+1.5,
                                                  foreground="white"), pe.Normal()])
                if np.isfinite(positions[-1]).all():
                    ax.plot(*positions[-1], marker="*", linestyle="none",
                            color=config["current_center_color"], markeredgecolor="white",
                            markeredgewidth=0.5, markersize=config["current_marker_size"],
                            label="Current TC center", zorder=5)
            nr_pos = centers_on_grid(ds.latitude.values[row, j], ds.longitude.values[row, j],
                                     [ds.nr_min_latitude.values[j]],
                                     [ds.nr_min_longitude.values[j]])[0]
            if np.isfinite(nr_pos).all():
                nr_artist, = ax.plot(nr_pos[0], nr_pos[1],
                                     marker=config["nr_marker"], linestyle="none",
                                     color=config["nr_marker_color"],
                                     markersize=config["nr_marker_size"],
                                     markeredgewidth=config["nr_marker_width"],
                                     label="NR min PSFC", zorder=6)
                nr_artist.set_path_effects([
                    pe.Stroke(linewidth=config["nr_marker_width"] + 1.5, foreground="white"),
                    pe.Normal()])
            first_in_column, last_in_column = np.flatnonzero(shown_panels[:, j])[[0, -1]]
            first_in_row = np.flatnonzero(shown_panels[i])[0]
            if i == first_in_column:
                stamp = ds.time.values[j].astype("datetime64[s]").astype(datetime)
                ax.set_title(stamp.strftime("%m-%d\n%H:%M UTC"), fontsize=10)
            if j == first_in_row:
                ax.set_ylabel(f"{ds.coupling.values[row]}\n{ds['filter'].values[row]}\nsouth_north index")
            if i == last_in_column:
                ax.set_xlabel("west_east index")
            ax.tick_params(labelsize=7, labelbottom=(i == last_in_column), labelleft=(j == first_in_row))
    extend = "max" if float(ds.spread.max()) > vmax else "neither"
    cbar = fig.colorbar(image, ax=active_axes, shrink=0.85, pad=0.015,
                        aspect=45, extend=extend)
    cbar.set_label(f"{config['variable']} ensemble spread ({ds.spread.attrs['units']})")
    level = f"; 3-D variables use model level {config['vertical_level']} (0-based)"
    fig.suptitle(f"{config['variable']} | {config['domain']} full-domain ensemble spread | "
                 f"N = {ds.sizes['member']}, ddof = 1\n"
                 f"{config['start_time'][:10]} UTC{level}", fontsize=11)
    fig.supxlabel("TC centers: black circles = ensemble-mean PSFC history; red star = current time; "
                  "orange cross = NR minimum PSFC", fontsize=9)
    return fig, axes


def save_outputs(ds: xr.Dataset, config: dict) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.strptime(config["start_time"], TIME_FORMAT).strftime("%Y%m%d%H%M")
    end = datetime.strptime(config["end_time"], TIME_FORMAT).strftime("%Y%m%d%H%M")
    stem = f"{config['variable']}_{config['domain']}_lev{config['vertical_level']}_spread_{start}_{end}"
    nc_path, csv_path = output_dir / f"{stem}.nc", output_dir / f"{stem}.csv"
    ds.to_netcdf(nc_path, engine="netcdf4", encoding={"spread": {"zlib": True, "complevel": 4}})
    summary = ds[["domain_rms_spread", "valid_grid_points", "members_used",
                  "panel_status", "panel_reason", "center_latitude", "center_longitude",
                  "center_psfc", "nr_min_latitude", "nr_min_longitude",
                  "nr_min_psfc"]].to_dataframe().reset_index()
    summary["ensemble_members"] = ds.sizes["member"]
    summary["total_grid_points"] = ds.sizes["y"] * ds.sizes["x"]
    summary["units"] = ds.spread.attrs["units"]
    summary["center_psfc_units"] = ds.center_psfc.attrs["units"]
    summary["nr_min_psfc_units"] = ds.nr_min_psfc.attrs["units"]
    summary.to_csv(csv_path, index=False)
    paths = [nc_path, csv_path]
    if not np.isfinite(ds.spread.values).any():
        print("No finite spread data: figure export skipped.")
        for path in paths:
            print(f"Saved: {path}")
        return paths
    fig, _ = plot_spread(ds, config)
    try:
        for fmt in config["figure_formats"]:
            path = output_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=config["dpi"], bbox_inches="tight")
            paths.append(path)
    finally:
        plt.close(fig)
    for path in paths:
        print(f"Saved: {path}")
    return paths


def main():
    save_outputs(calculate(CONFIG), CONFIG)


if __name__ == "__main__":
    main()
