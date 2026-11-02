"""计算并绘制六组 d02 全域逐格点集合离散度（无需 NR 真值）。

编辑下方 CONFIG 后直接运行本文件，不使用命令行参数。
行顺序：强耦合 EAKF/QCF_RHF、弱耦合 EAKF/QCF_RHF、无耦合 EAKF/QCF_RHF。
列顺序：2018-09-10 00–06 UTC，每小时一次，包含起止时间。

spread(y,x) = sqrt(sum_m((X_m(y,x) - mean_m(X_m(y,x)))**2) / (N-1))

先逐格点沿集合维计算标准差，保留整个 d02 二维场（包含陆地和边界）。
不对每个成员先做空间平均，不拼接成员网格，不使用台风半径裁剪。
某格点必须具有全部成员的有效值，否则该点为 NaN；缺成员文件直接报错。
无耦合试验不运行海洋模式，OM_TMP 对应两行保留并标为 N/A，不填成零离散度。
额外输出全域 RMS spread = sqrt(mean_yx(spread**2))，仅对有效格点等权平均。

支持 wrfout 中原始二维变量或原始三维变量的某个模式层；vertical_level
从 0 开始，不是 hPa 插值层。THM/T 等原始变量保持其文件内定义，不转换为
实际温度；U/V 等交错变量保留原网格。色标单位取自变量的 units 属性。

输出：PNG/PDF 六行对比图、含二维 spread/有效成员数/经纬度的 NetCDF、
逐试验逐时刻的全域 RMS spread 与有效格点数 CSV。
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
         "unavailable_variables": ("OM_TMP",)},  # 不运行海洋模式；此变量两行显示 N/A
    ),
    "filters": ("EAKF", "QCF_RHF"),
    "members": ("006", "015", "029", "037", "043", "044"),
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


def calculate(config: dict) -> xr.Dataset:
    times = build_times(config)
    members = config["members"]
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("Configure at least two distinct ensemble members")
    if not np.isfinite(config["scale"]) or config["scale"] <= 0:
        raise ValueError("scale must be finite and positive")
    rows = [(group, method) for group in config["groups"] for method in config["filters"]]
    records, source_files, statuses, reasons, units = [], [], [], [], set()
    for group, method in rows:
        if config["variable"] in group.get("unavailable_variables", ()):
            reason = f"{config['variable']} unavailable: ocean model not run"
            records.append([None] * len(times))
            source_files.append([[""] * len(members) for _ in times])
            statuses.append(["not_applicable"] * len(times))
            reasons.append([reason] * len(times))
            print(f"N/A: {group['label']} / {method}: {reason}")
            continue
        row_records, row_files = [], []
        for time in times:
            fields, paths = [], []
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
                fields.append(field)
                paths.append(str(path))
            spread, counts = calc_ensemble_spread(np.stack(fields))
            spread *= config["scale"]
            if not np.isfinite(spread).any():
                raise ValueError(f"No complete-member grid points: {group['label']} {method} {time}")
            row_records.append((spread, counts, ref_lat, ref_lon))
            row_files.append(paths)
            print(f"{group['label']} / {method} / {time:%Y-%m-%d %H:%M} UTC: "
                  f"N={len(fields)}, valid grid points={np.isfinite(spread).sum()}/{spread.size}")
        records.append(row_records)
        source_files.append(row_files)
        statuses.append(["ok"] * len(times))
        reasons.append([""] * len(times))
    if not units:
        raise ValueError(f"No applicable experiments for {config['variable']}")
    if len(units) != 1:
        raise ValueError(f"Inconsistent input units across members/experiments: {units}")
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
                                  np.where(np.asarray(statuses) == "ok", len(members), 0))
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
    ds.attrs = {"variable": config["variable"], "domain": config["domain"],
                "vertical_level": config["vertical_level"], "time_zone": "UTC",
                "spatial_support": "Whole d02 including land and boundary points; no cropping",
                "missing_policy": "All members required at each point; otherwise spread is NaN. "
                                  "Configured unavailable variables are N/A, with members_used=0.",
                "scale": config["scale"]}
    return ds


def plot_spread(ds: xr.Dataset, config: dict):
    """六行（或配置的行数）×时刻列，共享从零开始的统一色标。"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 9})
    nrows, ncols = ds.sizes["row"], ds.sizes["time"]
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
    cmap = plt.get_cmap(config["cmap"]).with_extremes(bad="#dddddd")
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            image = ax.imshow(ds.spread.values[i, j], origin="lower", aspect="equal",
                              interpolation="nearest", cmap=cmap, norm=norm)
            if "panel_status" in ds and ds.panel_status.values[i, j] == "not_applicable":
                ax.text(0.5, 0.5, f"N/A\n{config['variable']} unavailable\nOcean model not run",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9)
            if i == 0:
                stamp = ds.time.values[j].astype("datetime64[s]").astype(datetime)
                ax.set_title(stamp.strftime("%m-%d\n%H:%M UTC"), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{ds.coupling.values[i]}\n{ds['filter'].values[i]}\nsouth_north index")
            if i == nrows - 1:
                ax.set_xlabel("west_east index")
            ax.tick_params(labelsize=7, labelbottom=(i == nrows - 1), labelleft=(j == 0))
    extend = "max" if float(ds.spread.max()) > vmax else "neither"
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, pad=0.015,
                        aspect=45, extend=extend)
    cbar.set_label(f"{config['variable']} ensemble spread ({ds.spread.attrs['units']})")
    level = f"; 3-D variables use model level {config['vertical_level']} (0-based)"
    fig.suptitle(f"{config['variable']} | {config['domain']} full-domain ensemble spread | "
                 f"N = {ds.sizes['member']}, ddof = 1\n"
                 f"{config['start_time'][:10]} UTC{level}; gray = missing / N/A", fontsize=11)
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
                  "panel_status", "panel_reason"]].to_dataframe().reset_index()
    summary["ensemble_members"] = ds.sizes["member"]
    summary["total_grid_points"] = ds.sizes["y"] * ds.sizes["x"]
    summary["units"] = ds.spread.attrs["units"]
    summary.to_csv(csv_path, index=False)
    paths = [nc_path, csv_path]
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
