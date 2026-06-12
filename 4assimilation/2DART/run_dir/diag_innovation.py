"""
Plot LACC averaged innovation from one or more DART obs_seq.out files.

For each observation point, this script computes

    averaged innovation = mean_t(obs_t) - mean_t(mean_member(Hx_t))

using obs_seq.out files that contain external_FO blocks.

The obs_seq files should have the same observation layout/order for a direct
LACC-style average. If positions differ slightly, use --match-by-nearest with
caution; the default is to match by obs_id/order.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# User configuration
# =========================
CONFIG = {
    # Put the files in the same order as the LACC window, e.g.
    # [t-12h, t-9h, t-6h, t-3h, t].
    "obs_seq_paths": [
        r"/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir/obs_seq.out_kctest1_d01_10_00_00_LACC_ch2",
    ],
    "output_csv": r"./diag/lacc_averaged_innovation.csv",
    "output_png": r"./diag/lacc_averaged_innovation.png",
    # If True, also write per-file innovation columns.
    "write_each_time": True,
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
                "hx_mean": float(np.mean(hx)),
                "hx_sd": float(np.std(hx, ddof=1)),
                "innovation": float(obs_value - np.mean(hx)),
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


def calculate(config: dict) -> pd.DataFrame:
    paths = [Path(p) for p in config["obs_seq_paths"]]
    if not paths:
        raise ValueError("CONFIG['obs_seq_paths'] is empty.")

    frames = []
    for itime, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(path)
        df = parse_obs_seq_external_fo(path)
        if df.empty:
            raise RuntimeError(f"No complete external_FO observations parsed from {path}.")
        df = df.sort_values("obs_id").reset_index(drop=True)
        df["window_index"] = itime
        df["source_file"] = str(path)
        frames.append(df)

    nobs = len(frames[0])
    obs_ids = frames[0]["obs_id"].to_numpy()
    for frame, path in zip(frames[1:], paths[1:]):
        if len(frame) != nobs:
            raise ValueError(f"{path} has {len(frame)} obs, expected {nobs}.")
        if not np.array_equal(frame["obs_id"].to_numpy(), obs_ids):
            raise ValueError(f"{path} obs_id layout differs from the first file.")

    base = frames[0][["obs_id", "lat", "lon", "vert_value", "vert_type", "kind"]].copy()
    obs_stack = np.vstack([frame["obs"].to_numpy(dtype=float) for frame in frames])
    hx_stack = np.vstack([frame["hx_mean"].to_numpy(dtype=float) for frame in frames])
    innov_stack = obs_stack - hx_stack

    base["obs_avg"] = np.nanmean(obs_stack, axis=0)
    base["hx_mean_avg"] = np.nanmean(hx_stack, axis=0)
    base["avg_innovation"] = base["obs_avg"] - base["hx_mean_avg"]
    base["mean_of_innovation"] = np.nanmean(innov_stack, axis=0)
    base["innovation_std_over_window"] = np.nanstd(innov_stack, axis=0, ddof=1) if len(frames) > 1 else 0.0
    base["n_times"] = len(frames)

    if config.get("write_each_time", True):
        for itime, path in enumerate(paths):
            tag = f"t{itime:02d}"
            base[f"obs_{tag}"] = obs_stack[itime]
            base[f"hx_mean_{tag}"] = hx_stack[itime]
            base[f"innovation_{tag}"] = innov_stack[itime]
            base[f"source_{tag}"] = str(path)

    return base


def plot_averaged_innovation(result: pd.DataFrame, output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    plot_df = result.sort_values("obs_id")
    nobs = len(plot_df)
    side = int(round(math.sqrt(nobs)))
    square_grid = side * side == nobs

    values = plot_df["avg_innovation"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite averaged innovation values to plot.")
    limit = np.nanpercentile(np.abs(finite), 98.0)
    if not np.isfinite(limit) or limit == 0.0:
        limit = np.nanmax(np.abs(finite))
    levels = np.linspace(-limit, limit, 21)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=160)
    if square_grid:
        lon = plot_df["lon"].to_numpy(dtype=float).reshape(side, side)
        lat = plot_df["lat"].to_numpy(dtype=float).reshape(side, side)
        val = values.reshape(side, side)
        cf = ax.contourf(lon, lat, val, levels=levels, cmap="RdBu_r", extend="both")
        if np.nanmin(val) <= 0.0 <= np.nanmax(val):
            ax.contour(lon, lat, val, levels=[0.0], colors="k", linewidths=0.8)
    else:
        lon = plot_df["lon"].to_numpy(dtype=float)
        lat = plot_df["lat"].to_numpy(dtype=float)
        cf = ax.tricontourf(lon, lat, values, levels=levels, cmap="RdBu_r", extend="both")
        if np.nanmin(values) <= 0.0 <= np.nanmax(values):
            ax.tricontour(lon, lat, values, levels=[0.0], colors="k", linewidths=0.8)

    ax.set_title("LACC averaged innovation: mean(obs) - mean(Hx)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.colorbar(cf, ax=ax, label="Averaged innovation")
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-seq-paths", nargs="+", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-png", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict:
    config = CONFIG.copy()
    if args.obs_seq_paths is not None:
        config["obs_seq_paths"] = args.obs_seq_paths
    if args.output_csv is not None:
        config["output_csv"] = args.output_csv
    if args.output_png is not None:
        config["output_png"] = args.output_png
    if args.no_plot:
        config["output_png"] = None
    return config


def main() -> None:
    config = merge_config(parse_args())
    result = calculate(config)

    output_csv = Path(config["output_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    print("Averaged innovation summary:")
    print(result["avg_innovation"].describe().to_string())
    for threshold in [0.25, 0.5, 1.0, 2.0, 3.0]:
        count = int((np.abs(result["avg_innovation"]) > threshold).sum())
        print(f"|avg innovation| > {threshold}: {count} / {len(result)}")

    if config["output_png"]:
        output_png = Path(config["output_png"])
        plot_averaged_innovation(result, output_png)
        print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
