"""
Visualize ensemble correlation between satellite Hx and TC center position.

For each observation pixel in a DART obs_seq.out external_FO file, compute:

    corr(Hx(member), TC_center_lon(member))
    corr(Hx(member), TC_center_lat(member))

The TC center is diagnosed from each member wrfout using the minimum SLP/slp
if present, otherwise minimum PSFC.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import netCDF4 as nc
except ImportError:
    nc = None


# =========================
# User configuration
# =========================
CONFIG = {
    "obs_seq_path": r"./obs_seq.out",
    # Directory containing mem001, mem002, ..., mem050.
    "member_base_dir": r"/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst",
    "domain": "d01",
    "wrf_time": "2018-09-10_00:00:00",
    "member_start": 1,
    "member_end": 50,
    "lat_var": "XLAT",
    "lon_var": "XLONG",
    # Center variable priority. Use SLP/slp if available; PSFC fallback.
    "center_var_priority": ["SLP", "slp", "PSFC"],
    # Exclude members whose TC center is farther than this standardized
    # distance from the ensemble-mean center:
    #   sqrt(((lon-lon_mean)/lon_std)^2 + ((lat-lat_mean)/lat_std)^2)
    # Set to None to disable filtering.
    "tc_center_sigma_filter": 3.0,
    "output_csv": r"./diag/hx_tc_center_corr.csv",
    "output_png": r"./diag/hx_tc_center_corr.png",
    "output_regression_png": r"./diag/hx_tc_center_regression_r2.png",
}


def parse_obs_seq_external_fo(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="replace").splitlines()
    rows = []
    i = 0
    while i < len(lines):
        if not re.match(r"\s*OBS\s+\d+", lines[i]):
            i += 1
            continue

        obs_id = int(lines[i].split()[1])
        obs_value = float(lines[i + 1].split()[0])
        data_qc = float(lines[i + 2].split()[0])

        lon_rad = lat_rad = vert_value = np.nan
        vert_type = -999
        kind = -999
        visir_float = None
        visir_int = None
        hx = None
        errvar = np.nan
        obs_time = None

        j = i + 3
        complete = False
        while j < len(lines):
            if re.match(r"\s*OBS\s+\d+", lines[j]):
                break

            stripped = lines[j].strip()
            if stripped == "loc3d":
                vals = lines[j + 1].split()
                lon_rad = float(vals[0])
                lat_rad = float(vals[1])
                vert_value = float(vals[2])
                vert_type = int(float(vals[3]))
            elif stripped == "kind":
                kind = int(lines[j + 1].split()[0])
            elif stripped == "visir":
                visir_float = [float(x) for x in lines[j + 1].split()]
                visir_int = [int(x) for x in lines[j + 2].split()]
            elif stripped.startswith("external_FO"):
                parts = stripped.split()
                nmem = int(parts[1])
                vals = []
                k = j + 1
                while len(vals) < nmem and k < len(lines):
                    if re.match(r"\s*OBS\s+\d+", lines[k]):
                        break
                    vals.extend(float(x) for x in lines[k].split())
                    k += 1
                if len(vals) >= nmem and k + 1 < len(lines):
                    hx = np.asarray(vals[:nmem], dtype=float)
                    obs_time = tuple(int(x) for x in lines[k].split())
                    errvar = float(lines[k + 1].split()[0])
                    complete = True
                j = k
                continue
            j += 1

        if complete:
            row = {
                "obs_id": obs_id,
                "obs": obs_value,
                "data_qc": data_qc,
                "lon": math.degrees(lon_rad),
                "lat": math.degrees(lat_rad),
                "vert_value": vert_value,
                "vert_type": vert_type,
                "kind": kind,
                "errvar": errvar,
                "obs_time_days_seconds": obs_time,
                "hx": hx,
            }
            if visir_float is not None:
                row.update(
                    {
                        "sat_zenith": visir_float[0],
                        "sat_azimuth": visir_float[1],
                        "platform": visir_int[1] if len(visir_int) > 1 else np.nan,
                        "sensor": visir_int[2] if len(visir_int) > 2 else np.nan,
                        "channel": visir_int[3] if len(visir_int) > 3 else np.nan,
                    }
                )
            rows.append(row)

        i = j

    return pd.DataFrame(rows)


def read_nc_header(path: Path) -> dict:
    with path.open("rb") as f:
        data = f.read(256_000)

    pos = 0

    def read(n: int) -> bytes:
        nonlocal pos
        out = data[pos : pos + n]
        pos += n
        return out

    def u32() -> int:
        return struct.unpack(">I", read(4))[0]

    def u64() -> int:
        return struct.unpack(">Q", read(8))[0]

    def align4() -> None:
        nonlocal pos
        if pos % 4:
            pos += 4 - (pos % 4)

    def name() -> str:
        n = u32()
        out = read(n).decode("utf-8", "replace")
        align4()
        return out

    def values(nc_type: int, count: int) -> None:
        size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8}[nc_type]
        read(size * count)
        align4()

    def attrs() -> None:
        tag = u32()
        if tag == 0:
            _ = u32()
            return
        if tag != 12:
            raise ValueError(f"Unexpected NetCDF attr tag {tag}")
        nattrs = u32()
        for _ in range(nattrs):
            _ = name()
            nc_type = u32()
            count = u32()
            values(nc_type, count)

    magic = read(4)
    if magic not in (b"CDF\x01", b"CDF\x02"):
        raise ValueError(
            f"{path} is not a NetCDF classic/64-bit offset file, and netCDF4 is unavailable."
        )
    _ = u32()

    dim_tag = u32()
    dims = []
    if dim_tag == 10:
        ndims = u32()
        for _ in range(ndims):
            dims.append((name(), u32()))
    elif dim_tag != 0:
        raise ValueError(f"Unexpected NetCDF dim tag {dim_tag}")

    attrs()

    var_tag = u32()
    variables = {}
    if var_tag == 11:
        nvars = u32()
        for _ in range(nvars):
            var_name = name()
            ndims = u32()
            dimids = [u32() for _ in range(ndims)]
            attrs()
            nc_type = u32()
            vsize = u32()
            begin = u64() if magic == b"CDF\x02" else u32()
            variables[var_name] = {
                "dims": [dims[i] for i in dimids],
                "shape": tuple(dims[i][1] for i in dimids),
                "type": nc_type,
                "vsize": vsize,
                "begin": begin,
            }
    elif var_tag != 0:
        raise ValueError(f"Unexpected NetCDF var tag {var_tag}")

    return {"dims": dims, "variables": variables}


def nc_dtype(nc_type: int) -> str:
    if nc_type == 5:
        return ">f4"
    if nc_type == 6:
        return ">f8"
    if nc_type == 4:
        return ">i4"
    if nc_type == 3:
        return ">i2"
    if nc_type == 2:
        return "S1"
    raise ValueError(f"Unsupported NetCDF type {nc_type}")


def read_var_classic(path: Path, header: dict, varname: str) -> np.ndarray:
    meta = header["variables"][varname]
    arr = np.memmap(
        path,
        dtype=nc_dtype(meta["type"]),
        mode="r",
        offset=meta["begin"],
        shape=meta["shape"],
    )
    return np.asarray(arr)


def read_dataset_arrays(path: Path, varnames: list[str]) -> dict[str, np.ndarray]:
    if nc is not None:
        with nc.Dataset(path) as ds:
            return {name: np.asarray(ds.variables[name][:]) for name in varnames}

    header = read_nc_header(path)
    return {name: read_var_classic(path, header, name) for name in varnames}


def available_vars(path: Path) -> set[str]:
    if nc is not None:
        with nc.Dataset(path) as ds:
            return set(ds.variables.keys())
    return set(read_nc_header(path)["variables"].keys())


def squeeze_time(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def wrfout_path(base_dir: Path, member: int, domain: str, wrf_time: str) -> Path:
    mem_name = f"mem{member:03d}"
    return base_dir / mem_name / f"wrfout_{domain}_{wrf_time}"


def choose_center_var(path: Path, priority: list[str]) -> str:
    names = available_vars(path)
    for varname in priority:
        if varname in names:
            return varname
    raise KeyError(f"None of center_var_priority {priority} was found in {path}")


def find_member_tc_centers(config: dict) -> pd.DataFrame:
    base_dir = Path(config["member_base_dir"])
    members = list(range(int(config["member_start"]), int(config["member_end"]) + 1))
    rows = []

    for member in members:
        path = wrfout_path(base_dir, member, config["domain"], config["wrf_time"])
        if not path.exists():
            raise FileNotFoundError(path)
        center_var = choose_center_var(path, list(config["center_var_priority"]))
        arrays = read_dataset_arrays(path, [config["lat_var"], config["lon_var"], center_var])
        lat = squeeze_time(arrays[config["lat_var"]])
        lon = squeeze_time(arrays[config["lon_var"]])
        center_field = squeeze_time(arrays[center_var])
        if center_field.ndim != 2:
            raise ValueError(f"{center_var} must be 2D after removing Time, got {center_field.shape}")
        j, i = np.unravel_index(np.nanargmin(center_field), center_field.shape)
        rows.append(
            {
                "member": member,
                "center_var": center_var,
                "tc_i": int(i),
                "tc_j": int(j),
                "tc_lon": float(lon[j, i]),
                "tc_lat": float(lat[j, i]),
                "tc_min_value": float(center_field[j, i]),
            }
        )

    return pd.DataFrame(rows)


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    xv = x[valid]
    yv = y[valid]
    if np.std(xv, ddof=1) == 0.0 or np.std(yv, ddof=1) == 0.0:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def linear_regression_hx_on_center(
    hx_members: np.ndarray, tc_lon: np.ndarray, tc_lat: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Return intercept, beta_lon, beta_lat, R2, and fitted standard deviation."""
    valid = np.isfinite(hx_members) & np.isfinite(tc_lon) & np.isfinite(tc_lat)
    if valid.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    y = hx_members[valid]
    xlon = tc_lon[valid]
    xlat = tc_lat[valid]
    if np.std(y, ddof=1) == 0.0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Center predictors so intercept is the Hx value at mean TC center.
    xlon_anom = xlon - np.mean(xlon)
    xlat_anom = xlat - np.mean(xlat)
    x = np.column_stack([np.ones_like(y), xlon_anom, xlat_anom])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    residual = y - fitted
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(residual**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    return float(coef[0]), float(coef[1]), float(coef[2]), float(r2), float(np.std(fitted, ddof=1))


def calculate(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_df = parse_obs_seq_external_fo(Path(config["obs_seq_path"]))
    if obs_df.empty:
        raise RuntimeError("No complete external_FO observations were parsed.")

    centers = find_member_tc_centers(config)
    hx = np.vstack(obs_df["hx"].to_numpy())
    if hx.shape[1] != len(centers):
        raise ValueError(f"Hx has {hx.shape[1]} members, but found {len(centers)} wrfout members.")

    centers["use_member"] = True
    centers["tc_center_sigma_distance"] = 0.0
    sigma_filter = config.get("tc_center_sigma_filter")
    if sigma_filter is not None:
        lon_mean = float(centers["tc_lon"].mean())
        lat_mean = float(centers["tc_lat"].mean())
        lon_std = float(centers["tc_lon"].std(ddof=1))
        lat_std = float(centers["tc_lat"].std(ddof=1))
        if lon_std == 0.0 or lat_std == 0.0:
            raise ValueError("Cannot apply TC center sigma filter because lon or lat spread is zero.")
        sigma_distance = np.sqrt(
            ((centers["tc_lon"] - lon_mean) / lon_std) ** 2
            + ((centers["tc_lat"] - lat_mean) / lat_std) ** 2
        )
        centers["tc_center_sigma_distance"] = sigma_distance
        centers["use_member"] = sigma_distance <= float(sigma_filter)

    use_member = centers["use_member"].to_numpy(dtype=bool)
    if use_member.sum() < 4:
        raise ValueError(f"Only {use_member.sum()} members remain after TC center filtering.")

    tc_lon = centers.loc[use_member, "tc_lon"].to_numpy()
    tc_lat = centers.loc[use_member, "tc_lat"].to_numpy()
    tc_i = centers.loc[use_member, "tc_i"].to_numpy(dtype=float)
    tc_j = centers.loc[use_member, "tc_j"].to_numpy(dtype=float)
    hx_used = hx[:, use_member]

    rows = []
    for iobs, obs_row in obs_df.iterrows():
        hx_members = hx_used[iobs, :]
        rows.append(
            {
                "obs_id": int(obs_row["obs_id"]),
                "lat": float(obs_row["lat"]),
                "lon": float(obs_row["lon"]),
                "obs": float(obs_row["obs"]),
                "hx_mean": float(np.mean(hx_members)),
                "hx_sd": float(np.std(hx_members, ddof=1)),
                "innovation": float(obs_row["obs"] - np.mean(hx_members)),
                "corr_hx_tc_lon": pearsonr(hx_members, tc_lon),
                "corr_hx_tc_lat": pearsonr(hx_members, tc_lat),
                "corr_hx_tc_i": pearsonr(hx_members, tc_i),
                "corr_hx_tc_j": pearsonr(hx_members, tc_j),
                "cov_hx_tc_lon": float(np.cov(hx_members, tc_lon, ddof=1)[0, 1]),
                "cov_hx_tc_lat": float(np.cov(hx_members, tc_lat, ddof=1)[0, 1]),
                "n_members_used": int(use_member.sum()),
                "n_members_excluded": int((~use_member).sum()),
            }
        )
        intercept, beta_lon, beta_lat, r2, fitted_sd = linear_regression_hx_on_center(
            hx_members, tc_lon, tc_lat
        )
        rows[-1].update(
            {
                "reg_intercept": intercept,
                "reg_beta_tc_lon": beta_lon,
                "reg_beta_tc_lat": beta_lat,
                "reg_r2_tc_lon_lat": r2,
                "reg_fitted_sd": fitted_sd,
            }
        )

    return pd.DataFrame(rows), centers


def _plot_obs_field(
    ax,
    plot_df: pd.DataFrame,
    field: str,
    title: str,
    levels: np.ndarray,
    cmap: str,
    label: str,
):
    nobs = len(plot_df)
    side = int(round(math.sqrt(nobs)))
    square_grid = side * side == nobs

    if square_grid:
        lon = plot_df["lon"].to_numpy().reshape(side, side)
        lat = plot_df["lat"].to_numpy().reshape(side, side)
        val = plot_df[field].to_numpy().reshape(side, side)
        cf = ax.contourf(lon, lat, val, levels=levels, cmap=cmap, extend="both")
        if np.nanmin(val) <= 0.0 <= np.nanmax(val):
            ax.contour(lon, lat, val, levels=[0.0], colors="k", linewidths=0.7)
    else:
        lon = plot_df["lon"].to_numpy()
        lat = plot_df["lat"].to_numpy()
        val = plot_df[field].to_numpy()
        cf = ax.tricontourf(lon, lat, val, levels=levels, cmap=cmap, extend="both")
        if np.nanmin(val) <= 0.0 <= np.nanmax(val):
            ax.tricontour(lon, lat, val, levels=[0.0], colors="k", linewidths=0.7)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    return cf, label


def plot_corr_maps(result: pd.DataFrame, centers: pd.DataFrame, output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    plot_df = result.sort_values("obs_id")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=160, constrained_layout=True)
    specs = [
        ("corr_hx_tc_lon", "corr(Hx, TC longitude)"),
        ("corr_hx_tc_lat", "corr(Hx, TC latitude)"),
    ]

    for ax, (field, title) in zip(axes, specs):
        cf, label = _plot_obs_field(
            ax,
            plot_df,
            field,
            title,
            np.linspace(-1.0, 1.0, 21),
            "RdBu_r",
            "Correlation",
        )

        ax.scatter(
            centers["tc_lon"],
            centers["tc_lat"],
            s=12,
            c="0.15",
            alpha=0.55,
            linewidths=0.0,
            label="member TC center",
        )
        ax.scatter(
            [centers["tc_lon"].mean()],
            [centers["tc_lat"].mean()],
            s=46,
            c="yellow",
            edgecolors="k",
            linewidths=0.7,
            marker="*",
            label="mean center",
        )
        fig.colorbar(cf, ax=ax, label=label)

    axes[0].legend(loc="best", fontsize=8)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def plot_regression_r2_map(result: pd.DataFrame, centers: pd.DataFrame, output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    plot_df = result.sort_values("obs_id")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=160, constrained_layout=True)
    specs = [
        ("reg_r2_tc_lon_lat", "R2: Hx ~ TC lon + TC lat", np.linspace(0.0, 1.0, 21), "viridis", "R2"),
        (
            "reg_beta_tc_lon",
            "Regression beta: TC longitude",
            np.linspace(
                -np.nanpercentile(np.abs(plot_df["reg_beta_tc_lon"]), 98),
                np.nanpercentile(np.abs(plot_df["reg_beta_tc_lon"]), 98),
                21,
            ),
            "RdBu_r",
            "K per degree lon",
        ),
        (
            "reg_beta_tc_lat",
            "Regression beta: TC latitude",
            np.linspace(
                -np.nanpercentile(np.abs(plot_df["reg_beta_tc_lat"]), 98),
                np.nanpercentile(np.abs(plot_df["reg_beta_tc_lat"]), 98),
                21,
            ),
            "RdBu_r",
            "K per degree lat",
        ),
    ]

    for ax, (field, title, levels, cmap, label) in zip(axes, specs):
        if not np.all(np.isfinite(levels)) or np.nanmax(levels) == np.nanmin(levels):
            levels = np.linspace(-1.0, 1.0, 21) if "beta" in field else np.linspace(0.0, 1.0, 21)
        cf, cbar_label = _plot_obs_field(ax, plot_df, field, title, levels, cmap, label)
        ax.scatter(
            centers["tc_lon"],
            centers["tc_lat"],
            s=10,
            c="0.15",
            alpha=0.45,
            linewidths=0.0,
        )
        ax.scatter(
            [centers["tc_lon"].mean()],
            [centers["tc_lat"].mean()],
            s=46,
            c="yellow",
            edgecolors="k",
            linewidths=0.7,
            marker="*",
        )
        fig.colorbar(cf, ax=ax, label=cbar_label)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-seq-path", default=None)
    parser.add_argument("--member-base-dir", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--wrf-time", default=None)
    parser.add_argument("--member-start", type=int, default=None)
    parser.add_argument("--member-end", type=int, default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-png", default=None)
    parser.add_argument("--output-regression-png", default=None)
    parser.add_argument("--tc-center-sigma-filter", type=float, default=None)
    parser.add_argument("--no-tc-center-filter", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict:
    config = CONFIG.copy()
    for key, value in {
        "obs_seq_path": args.obs_seq_path,
        "member_base_dir": args.member_base_dir,
        "domain": args.domain,
        "wrf_time": args.wrf_time,
        "member_start": args.member_start,
        "member_end": args.member_end,
        "output_csv": args.output_csv,
        "output_png": args.output_png,
        "output_regression_png": args.output_regression_png,
        "tc_center_sigma_filter": args.tc_center_sigma_filter,
    }.items():
        if value is not None:
            config[key] = value
    if args.no_tc_center_filter:
        config["tc_center_sigma_filter"] = None
    if args.no_plot:
        config["output_png"] = None
        config["output_regression_png"] = None
    return config


def main() -> None:
    config = merge_config(parse_args())
    result, centers = calculate(config)

    output_csv = Path(config["output_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    centers.to_csv(output_csv.with_name(output_csv.stem + "_tc_centers.csv"), index=False)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_csv.with_name(output_csv.stem + '_tc_centers.csv')}")
    print("TC center spread:")
    print(centers[["tc_lon", "tc_lat", "tc_i", "tc_j"]].agg(["mean", "std", "min", "max"]).to_string())
    if "use_member" in centers:
        used = centers[centers["use_member"]]
        excluded = centers[~centers["use_member"]]
        print(f"TC center filter: used {len(used)} members, excluded {len(excluded)} members")
        if len(excluded) > 0:
            print("Excluded members:")
            print(
                excluded[
                    ["member", "tc_lon", "tc_lat", "tc_i", "tc_j", "tc_center_sigma_distance"]
                ].to_string(index=False)
            )
        print("TC center spread after filtering:")
        print(used[["tc_lon", "tc_lat", "tc_i", "tc_j"]].agg(["mean", "std", "min", "max"]).to_string())
    print("Correlation map summary:")
    print(result[["corr_hx_tc_lon", "corr_hx_tc_lat", "corr_hx_tc_i", "corr_hx_tc_j"]].describe().to_string())
    print("Regression R2 summary:")
    print(result["reg_r2_tc_lon_lat"].describe().to_string())
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        count = int((result["reg_r2_tc_lon_lat"] > threshold).sum())
        print(f"R2 > {threshold:.1f}: {count} / {len(result)}")

    if config["output_png"]:
        output_png = Path(config["output_png"])
        plot_corr_maps(result, centers, output_png)
        print(f"Wrote {output_png}")

    if config["output_regression_png"]:
        output_regression_png = Path(config["output_regression_png"])
        plot_regression_r2_map(result, centers, output_regression_png)
        print(f"Wrote {output_regression_png}")


if __name__ == "__main__":
    main()
