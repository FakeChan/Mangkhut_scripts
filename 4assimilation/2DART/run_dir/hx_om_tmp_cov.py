"""
Diagnose ensemble correlation between OM_TMP and satellite-observation Hx.

The script reads DART obs_seq.out files containing an external_FO block with
member Hx values, reads member wrfout files from mem001...mem050 directories,
and computes ensemble correlations:

    corr(OM_TMP(member, level, nearby grid), Hx(member, obs))

This is intended to diagnose whether a positive brightness-temperature
innovation should produce warming or cooling in the EAKF regression.
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
    "obs_seq_path": r"/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir/obs_seq.out_kctest1_d01_10_00_00_LACC_ch2",
    # Directory containing mem001, mem002, ..., mem050.
    "member_base_dir": r"/scratch/lililei1/kcfu/tc_mangkhut/2ens_free_fcst",
    "domain": "d01",
    "wrf_time": "2018-09-10_00:00:00",
    "member_start": 1,
    "member_end": 50,
    "omtmp_var": "OM_TMP",
    "lat_var": "XLAT",
    "lon_var": "XLONG",
    # Ocean levels are 0-based Python indices. Use None for all levels.
    "levels": [0, 5, 10, 14, 20, 29],
    # If radius_km is 0 or None, use nearest grid point only.
    # If > 0, average OM_TMP within this radius around each obs location for
    # each member before computing correlation.
    "radius_km": 0.0,
    "output_csv": r"./diag/cov_hxom_tmp.csv",
    # Plot only this level's covariance. Use None to skip plotting.
    "plot_cov_level": 0,
    "output_png": r"./diag/cov_hxom_tmp.png",
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
        raise ValueError(f"{path} is not a NetCDF classic/64-bit offset file.")
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


def read_dataset_arrays(path: Path, varnames: list[str]) -> dict[str, np.ndarray]:
    """Read variables from NetCDF classic or NetCDF-4 WRF files."""
    if nc is not None:
        with nc.Dataset(path) as ds:
            return {name: np.asarray(ds.variables[name][:]) for name in varnames}

    header = read_nc_header(path)
    return {name: read_var(path, header, name) for name in varnames}


def wrfout_path(base_dir: Path, member: int, domain: str, wrf_time: str) -> Path:
    mem_name = f"mem{member:03d}"
    return base_dir / mem_name / f"wrfout_{domain}_{wrf_time}"


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


def read_var(path: Path, header: dict, varname: str) -> np.ndarray:
    meta = header["variables"][varname]
    arr = np.memmap(
        path,
        dtype=nc_dtype(meta["type"]),
        mode="r",
        offset=meta["begin"],
        shape=meta["shape"],
    )
    return np.asarray(arr)


def squeeze_time(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def haversine_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    earth_radius_km = 6371.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat0_rad) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    xv = x[valid]
    yv = y[valid]
    if np.std(xv, ddof=1) == 0.0 or np.std(yv, ddof=1) == 0.0:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def load_omtmp_ensemble(config: dict, obs_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_dir = Path(config["member_base_dir"])
    members = list(range(int(config["member_start"]), int(config["member_end"]) + 1))
    first_path = wrfout_path(base_dir, members[0], config["domain"], config["wrf_time"])
    if not first_path.exists():
        raise FileNotFoundError(first_path)

    first_arrays = read_dataset_arrays(
        first_path, [config["lat_var"], config["lon_var"], config["omtmp_var"]]
    )
    lat = squeeze_time(first_arrays[config["lat_var"]])
    lon = squeeze_time(first_arrays[config["lon_var"]])
    om_shape = squeeze_time(first_arrays[config["omtmp_var"]]).shape
    if len(om_shape) != 3:
        raise ValueError(f"{config['omtmp_var']} must be 3D after removing Time, got {om_shape}")

    levels = config["levels"]
    if levels is None:
        levels = list(range(om_shape[0]))
    else:
        levels = [int(x) for x in levels]

    obs_points = list(zip(obs_df["lat"].to_numpy(), obs_df["lon"].to_numpy()))
    radius_km = float(config["radius_km"] or 0.0)
    nearest_ji = []
    masks = []
    for lat0, lon0 in obs_points:
        d2 = (lat - lat0) ** 2 + (lon - lon0) ** 2
        j, i = np.unravel_index(np.argmin(d2), d2.shape)
        nearest_ji.append((int(j), int(i)))
        if radius_km > 0.0:
            masks.append(haversine_km(lat, lon, lat0, lon0) <= radius_km)

    nmem = len(members)
    nobs = len(obs_df)
    nlev = len(levels)
    om_values = np.full((nmem, nlev, nobs), np.nan, dtype=float)

    for im, member in enumerate(members):
        path = wrfout_path(base_dir, member, config["domain"], config["wrf_time"])
        if not path.exists():
            raise FileNotFoundError(path)
        if member == members[0]:
            omtmp = squeeze_time(first_arrays[config["omtmp_var"]])
        else:
            omtmp = squeeze_time(read_dataset_arrays(path, [config["omtmp_var"]])[config["omtmp_var"]])
        for il, level in enumerate(levels):
            field = omtmp[level, :, :]
            if radius_km > 0.0:
                om_values[im, il, :] = [float(np.nanmean(field[mask])) for mask in masks]
            else:
                om_values[im, il, :] = [float(field[j, i]) for j, i in nearest_ji]

    obs_df["nearest_j"] = [ji[0] for ji in nearest_ji]
    obs_df["nearest_i"] = [ji[1] for ji in nearest_ji]
    return om_values, np.asarray(levels, dtype=int), obs_df


def calculate(config: dict) -> pd.DataFrame:
    obs_df = parse_obs_seq_external_fo(Path(config["obs_seq_path"]))
    if obs_df.empty:
        raise RuntimeError("No complete external_FO observations were parsed.")

    expected_nmem = int(config["member_end"]) - int(config["member_start"]) + 1
    hx = np.vstack(obs_df["hx"].to_numpy())
    if hx.shape[1] != expected_nmem:
        raise ValueError(f"Hx has {hx.shape[1]} members, expected {expected_nmem}.")

    om_values, levels, obs_df = load_omtmp_ensemble(config, obs_df)

    rows = []
    for iobs, obs_row in obs_df.iterrows():
        hx_members = hx[iobs, :]
        hx_mean = float(np.mean(hx_members))
        hx_sd = float(np.std(hx_members, ddof=1))
        innovation = float(obs_row["obs"] - hx_mean)
        for il, level in enumerate(levels):
            om_members = om_values[:, il, iobs]
            corr = pearsonr(om_members, hx_members)
            cov = float(np.cov(om_members, hx_members, ddof=1)[0, 1])
            beta = cov / float(np.var(hx_members, ddof=1)) if hx_sd > 0.0 else np.nan
            rows.append(
                {
                    "obs_id": int(obs_row["obs_id"]),
                    "lat": float(obs_row["lat"]),
                    "lon": float(obs_row["lon"]),
                    "nearest_i": int(obs_row["nearest_i"]),
                    "nearest_j": int(obs_row["nearest_j"]),
                    "level": int(level),
                    "obs": float(obs_row["obs"]),
                    "hx_mean": hx_mean,
                    "hx_sd": hx_sd,
                    "innovation": innovation,
                    "om_mean": float(np.mean(om_members)),
                    "om_sd": float(np.std(om_members, ddof=1)),
                    "corr_om_hx": corr,
                    "cov_om_hx": cov,
                    "eakf_regression_beta": beta,
                    "predicted_increment_sign": np.sign(beta * innovation)
                    if np.isfinite(beta)
                    else np.nan,
                    "radius_km": float(config["radius_km"] or 0.0),
                }
            )

    return pd.DataFrame(rows)


def plot_covariance_contour(result: pd.DataFrame, level: int, output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install matplotlib or skip plotting.") from exc

    level_df = result[result["level"] == int(level)].sort_values("obs_id")
    if level_df.empty:
        raise ValueError(f"No rows found for level {level}.")

    nobs = len(level_df)
    side = int(round(math.sqrt(nobs)))
    if side * side == nobs:
        lon = level_df["lon"].to_numpy().reshape(side, side)
        lat = level_df["lat"].to_numpy().reshape(side, side)
        cov = level_df["cov_om_hx"].to_numpy().reshape(side, side)
    else:
        # Fallback for non-square obs layouts: tricontourf on scattered points.
        lon = level_df["lon"].to_numpy()
        lat = level_df["lat"].to_numpy()
        cov = level_df["cov_om_hx"].to_numpy()

    finite_cov = cov[np.isfinite(cov)]
    if finite_cov.size == 0:
        raise ValueError("No finite covariance values to plot.")
    limit = np.nanpercentile(np.abs(finite_cov), 98.0)
    if limit == 0.0 or not np.isfinite(limit):
        limit = np.nanmax(np.abs(finite_cov))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=160)
    levels = np.linspace(-limit, limit, 21)

    if np.ndim(cov) == 2:
        cf = ax.contourf(lon, lat, cov, levels=levels, cmap="RdBu_r", extend="both")
        ax.contour(lon, lat, cov, levels=[0.0], colors="k", linewidths=0.8)
    else:
        cf = ax.tricontourf(lon, lat, cov, levels=levels, cmap="RdBu_r", extend="both")
        ax.tricontour(lon, lat, cov, levels=[0.0], colors="k", linewidths=0.8)

    ax.set_title(f"cov(OM_TMP level {level}, Hx)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.colorbar(cf, ax=ax, label="Covariance")
    fig.tight_layout()
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
    parser.add_argument("--levels", nargs="+", type=int, default=None)
    parser.add_argument("--all-levels", action="store_true")
    parser.add_argument("--radius-km", type=float, default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--plot-cov-level", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-png", default=None)
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
        "levels": args.levels,
        "radius_km": args.radius_km,
        "output_csv": args.output_csv,
        "plot_cov_level": args.plot_cov_level,
        "output_png": args.output_png,
    }.items():
        if value is not None:
            config[key] = value
    if args.all_levels:
        config["levels"] = None
    if args.no_plot:
        config["plot_cov_level"] = None
    return config


def main() -> None:
    config = merge_config(parse_args())
    out = Path(config["output_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    result = calculate(config)
    result.to_csv(out, index=False)

    print(f"Wrote {out}")
    print(f"Rows: {len(result)}")
    print("Correlation summary by level:")
    summary = result.groupby("level")["corr_om_hx"].agg(["count", "mean", "median", "min", "max"])
    print(summary.to_string())
    print("Predicted increment sign counts by level:")
    signs = result.groupby(["level", "predicted_increment_sign"]).size()
    print(signs.to_string())

    if config["plot_cov_level"] is not None:
        output_png = Path(config["output_png"])
        plot_covariance_contour(result, int(config["plot_cov_level"]), output_png)
        print(f"Wrote covariance contour: {output_png}")


if __name__ == "__main__":
    main()
