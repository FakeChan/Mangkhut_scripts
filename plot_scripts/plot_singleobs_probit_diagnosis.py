#!/usr/bin/env python3
"""
Plot single-observation probit/update diagnostics for EAKF and QCF_RHF.

The figure is designed to explain why one single-observation assimilation
improves or degrades the state estimate:

1. observation prior and transform context,
2. observation-space update,
3. probit/observation increment versus state response at one grid point,
4. physical analysis increment with NR-prior contours overlaid.

Most file discovery, obs_seq parsing, interpolation, and member extraction are
reused from plot_singleobs_nr_compare.py.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"),
)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import plot_singleobs_nr_compare as base


# =============================================================================
# User configuration
# =============================================================================
DATA_ROOT = base.DATA_ROOT
FILTERS = ["EAKF", "QCF_RHF"]
OBS_POINTS = base.OBS_POINTS
DOMAINS = base.DOMAINS
MEMBERS = base.MEMBERS

OBS_SOURCE_PATH = base.OBS_SOURCE_PATH
FIRSTGUESS_DIR = base.FIRSTGUESS_DIR

VAR_NAME = base.VAR_NAME
LEVEL = base.LEVEL
SCALE = base.SCALE

NR_FILE = base.NR_FILE
NR_BASE = base.NR_BASE
NR_DOMAIN = base.NR_DOMAIN
TIME_STRING = base.TIME_STRING

TC_HALF_WIDTH_KM = base.TC_HALF_WIDTH_KM
STATE_SELECTION = base.STATE_SELECTION
STATE_LAT = base.STATE_LAT
STATE_LON = base.STATE_LON

OUTPUT_DIR = Path(__file__).resolve().parent / "figs" / "singleobs_probit_diagnosis"
FIG_FORMAT = "png"
DPI = 300
MAP_COLOR_LEVELS = 21
NR_PRIOR_CONTOURS = 9

FILTER_LABELS = base.FILTER_LABELS
VAR_LABELS = base.VAR_LABELS
OUTPUT_PREFIXES = base.OUTPUT_PREFIXES
FIRSTGUESS_PREFIXES = base.FIRSTGUESS_PREFIXES


@dataclass
class PlotBundle:
    result: base.RunResult
    obs_prior: np.ndarray
    obs_increment: np.ndarray
    obs_post: np.ndarray
    probit_prior: np.ndarray | None
    probit_increment: np.ndarray | None
    probit_post: np.ndarray | None
    state_prior: np.ndarray
    state_post: np.ndarray
    state_increment: np.ndarray


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def align_member_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Trim member-wise arrays to the shortest length without reordering."""
    if not arrays:
        return ()
    lengths = [len(arr) for arr in arrays]
    n = min(lengths)
    return tuple(np.asarray(arr, dtype=float)[:n] for arr in arrays)


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = align_member_arrays(x, y)
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return slope, intercept, and Pearson r for finite paired values."""
    x, y = finite_pair(x, y)
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        r_value = float("nan")
    else:
        r_value = float(np.corrcoef(x, y)[0, 1])
    return float(slope), float(intercept), r_value


def error_change(analysis: np.ndarray, prior: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Negative values indicate the analysis is closer to the NR than the prior."""
    return np.abs(analysis - truth) - np.abs(prior - truth)


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if not np.isfinite(std) or std <= 0:
        return np.full_like(x, np.nan, dtype=float)
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def rank_fraction(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    sorted_values = np.sort(values[valid])
    if sorted_values.size == 0:
        return sorted_values, sorted_values
    ranks = (np.arange(sorted_values.size, dtype=float) + 1.0) / (sorted_values.size + 1.0)
    return sorted_values, ranks


def get_firstguess_dir(domain: str) -> Path:
    if FIRSTGUESS_DIR is None:
        raise ValueError("Set FIRSTGUESS_DIR near the top of this script before running.")
    if isinstance(FIRSTGUESS_DIR, dict):
        return Path(FIRSTGUESS_DIR[domain])
    return Path(FIRSTGUESS_DIR)


def get_obs_source(obs_point: int, fallback_run_dir: Path) -> Path:
    old = base.OBS_SOURCE_PATH
    try:
        base.OBS_SOURCE_PATH = OBS_SOURCE_PATH
        return base.resolve_obs_source(obs_point, fallback_run_dir)
    finally:
        base.OBS_SOURCE_PATH = old


def bundle_for_result(
    result: base.RunResult,
    firstguess_dir: Path,
    domain: str,
    members: list[int],
    var_name: str,
    level: int | None,
    scale: float,
    state_lat: float,
    state_lon: float,
) -> PlotBundle | None:
    obs_prior = result.obs_space.get("obs_prior")
    obs_increment = result.obs_space.get("obs_increment")
    if obs_prior is None or obs_increment is None:
        return None

    state_prior = base.member_values_at_point(
        firstguess_dir,
        domain,
        members,
        FIRSTGUESS_PREFIXES,
        var_name,
        level,
        scale,
        state_lat,
        state_lon,
    )
    state_post = base.member_values_at_point(
        result.run_dir,
        domain,
        members,
        OUTPUT_PREFIXES,
        var_name,
        level,
        scale,
        state_lat,
        state_lon,
    )
    if state_prior is None or state_post is None:
        return None

    obs_prior, obs_increment, state_prior, state_post = align_member_arrays(
        obs_prior,
        obs_increment,
        state_prior,
        state_post,
    )
    state_increment = state_post - state_prior
    obs_post = obs_prior + obs_increment

    probit_prior = result.obs_space.get("probit_obs_prior")
    probit_increment = result.obs_space.get("probit_obs_increment")
    probit_post = None
    if probit_prior is not None and probit_increment is not None:
        probit_prior, probit_increment = align_member_arrays(probit_prior, probit_increment)
        n = min(len(obs_prior), len(probit_prior), len(probit_increment))
        obs_prior, obs_increment, obs_post = obs_prior[:n], obs_increment[:n], obs_post[:n]
        state_prior, state_post, state_increment = state_prior[:n], state_post[:n], state_increment[:n]
        probit_prior, probit_increment = probit_prior[:n], probit_increment[:n]
        probit_post = probit_prior + probit_increment

    return PlotBundle(
        result=result,
        obs_prior=obs_prior,
        obs_increment=obs_increment,
        obs_post=obs_post,
        probit_prior=probit_prior,
        probit_increment=probit_increment,
        probit_post=probit_post,
        state_prior=state_prior,
        state_post=state_post,
        state_increment=state_increment,
    )


def annotate_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = "0.25") -> None:
    x_finite, y_finite = finite_pair(x, y)
    if x_finite.size < 2:
        return
    slope, intercept, r_value = linear_fit(x_finite, y_finite)
    xmin, xmax = float(np.nanmin(x_finite)), float(np.nanmax(x_finite))
    if xmin == xmax or not np.isfinite(slope):
        return
    xx = np.linspace(xmin, xmax, 50)
    ax.plot(xx, slope * xx + intercept, color=color, lw=1.2, ls="--")
    ax.text(
        0.03,
        0.96,
        f"slope={slope:.3g}\nr={r_value:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.82", "lw": 0.5, "alpha": 0.9},
    )


def plot_prior_transform_panel(ax: plt.Axes, bundle: PlotBundle, obs_value: float | None) -> None:
    filt = bundle.result.filt
    x = bundle.obs_prior[np.isfinite(bundle.obs_prior)]
    if x.size == 0:
        ax.text(0.5, 0.5, "No obs prior values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.hist(x, bins=min(12, max(4, x.size // 4)), color="#8b79c9", alpha=0.45, density=True, label="obs prior")
    xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
    if xs[0] == xs[-1]:
        xs = np.linspace(xs[0] - 0.5, xs[0] + 0.5, 200)

    if filt.upper() == "EAKF":
        pdf = normal_pdf(xs, float(np.nanmean(x)), float(np.nanstd(x, ddof=1)))
        ax.plot(xs, pdf, color="#4c5f8f", lw=1.5, label="Gaussian fit")
        note = "Gaussian CDF"
    else:
        sorted_values, ranks = rank_fraction(bundle.obs_prior)
        if sorted_values.size:
            twin = ax.twinx()
            twin.step(sorted_values, ranks, where="post", color="#b85c45", lw=1.2, label="rank CDF")
            twin.set_ylabel("rank CDF")
            twin.set_ylim(0, 1)
            twin.tick_params(labelsize=6)
        note = "rank CDF + probit"

    if obs_value is not None and np.isfinite(obs_value):
        ax.axvline(obs_value, color="#d73027", lw=1.2, label="obs")

    ax.set_title(f"{FILTER_LABELS.get(filt, filt)}: {note}")
    ax.set_xlabel("obs prior H(x)")
    ax.set_ylabel("density")
    ax.grid(True, color="0.9", lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=6)


def plot_obs_update_panel(ax: plt.Axes, bundle: PlotBundle) -> None:
    ax.scatter(bundle.obs_prior, bundle.obs_increment, s=16, color="#5f6f91", alpha=0.78)
    annotate_fit(ax, bundle.obs_prior, bundle.obs_increment)
    ax.axhline(0, color="0.75", lw=0.8)
    ax.set_title("Observation-space update")
    ax.set_xlabel("obs prior H(x)")
    ax.set_ylabel("obs increment")
    ax.grid(True, color="0.9", lw=0.5)


def plot_state_response_panel(ax: plt.Axes, bundle: PlotBundle) -> None:
    filt = bundle.result.filt.upper()
    if filt == "QCF_RHF" and bundle.probit_increment is not None:
        x = bundle.probit_increment
        xlabel = "probit obs increment"
        title = "Transformed-space driver vs state response"
        color = "#b85c45"
    else:
        x = bundle.obs_increment
        xlabel = "obs increment"
        title = "Physical-space linear response"
        color = "#5f6f91"

    y = bundle.state_increment
    ax.scatter(x, y, s=16, color=color, alpha=0.78)
    annotate_fit(ax, x, y)
    ax.axhline(0, color="0.75", lw=0.8)
    ax.axvline(0, color="0.75", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{VAR_LABELS.get(VAR_NAME, VAR_NAME)} increment")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(True, color="0.9", lw=0.5)


def plot_physical_panel(
    ax: plt.Axes,
    increment: np.ndarray,
    nr_prior_difference: np.ndarray,
    nr_lats: np.ndarray,
    nr_lons: np.ndarray,
    region_mask: np.ndarray,
    tc_lat: float,
    tc_lon: float,
    obs_info: base.ObsSeqInfo | None,
    state_lat: float,
    state_lon: float,
    increment_levels: np.ndarray,
    nr_prior_levels: np.ndarray,
) -> object:
    plot_lons, plot_lats, plot_inc, plot_prior = base.crop_to_mask(
        nr_lons,
        nr_lats,
        np.where(region_mask, increment, np.nan),
        np.where(region_mask, nr_prior_difference, np.nan),
        mask=region_mask,
    )
    pcm = ax.contourf(plot_lons, plot_lats, plot_inc, levels=increment_levels, cmap="RdBu_r", extend="both")
    if np.any(np.isfinite(plot_prior)):
        contours = ax.contour(
            plot_lons,
            plot_lats,
            plot_prior,
            levels=nr_prior_levels,
            colors="0.15",
            linewidths=0.55,
            alpha=0.78,
        )
        ax.clabel(contours, inline=True, fontsize=5, fmt="%.2g")

    ax.scatter(tc_lon, tc_lat, marker="+", s=62, lw=1.4, c="black", label="TC center")
    if obs_info is not None and obs_info.lat is not None and obs_info.lon is not None:
        ax.scatter(obs_info.lon, obs_info.lat, marker="x", s=38, lw=1.2, c="black", label="obs")
    ax.scatter(state_lon, state_lat, marker="v", s=40, c="white", edgecolors="black", linewidths=0.8, label="state")
    ax.set_title("Physical increment; contours = NR - prior")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.nanmin(plot_lons)), float(np.nanmax(plot_lons)))
    ax.set_ylim(float(np.nanmin(plot_lats)), float(np.nanmax(plot_lats)))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.88", lw=0.4)
    return pcm


def symmetric_levels(values: list[np.ndarray], nlevels: int) -> np.ndarray:
    finite = [arr[np.isfinite(arr)].ravel() for arr in values if np.any(np.isfinite(arr))]
    if not finite:
        vlim = 1.0
    else:
        joined = np.concatenate(finite)
        vlim = float(np.nanpercentile(np.abs(joined), 98))
        if not np.isfinite(vlim) or vlim == 0:
            vlim = 1.0
    return np.linspace(-vlim, vlim, nlevels)


def make_figure(obs_point: int, domain: str, nr_file: Path, members: list[int], scale: float, data_root: Path) -> Path:
    nr_lats, nr_lons = base.read_grid_for_field(nr_file, VAR_NAME)
    nr_truth = base.read_field(nr_file, VAR_NAME, LEVEL, scale)
    tc_lat, tc_lon, tc_pressure, center_name = base.tc_center_from_nr(nr_file)
    region_mask = base.tc_square_mask(nr_lats, nr_lons, tc_lat, tc_lon, TC_HALF_WIDTH_KM)
    if not np.any(region_mask):
        raise ValueError(f"No NR grid points inside half-width {TC_HALF_WIDTH_KM:g} km")

    run_dirs = {filt: base.resolve_run_dir(data_root, filt, obs_point) for filt in FILTERS}
    obs_source = get_obs_source(obs_point, next(iter(run_dirs.values())))
    obs_info = base.find_obs_seq_info(obs_source, obs_point)

    firstguess_dir = get_firstguess_dir(domain)
    prior_mean_file = base.find_firstguess_mean_file(firstguess_dir, domain)
    prior_mean_on_nr = base.interp_file_to_nr(prior_mean_file, VAR_NAME, LEVEL, scale, nr_lats, nr_lons)
    nr_prior_difference = np.where(region_mask, nr_truth - prior_mean_on_nr, np.nan)

    results = [
        base.calculate_run(
            filt,
            run_dir,
            domain,
            members,
            OUTPUT_PREFIXES,
            VAR_NAME,
            LEVEL,
            scale,
            nr_lats,
            nr_lons,
            prior_mean_on_nr,
            region_mask,
        )
        for filt, run_dir in run_dirs.items()
    ]

    state_lat, state_lon, state_truth = base.select_state_point(
        STATE_SELECTION,
        nr_lats,
        nr_lons,
        nr_truth,
        results[0].increment_on_nr,
        region_mask,
        obs_info,
        STATE_LAT,
        STATE_LON,
        tc_lat,
        tc_lon,
    )

    bundles = [
        bundle_for_result(
            result,
            firstguess_dir,
            domain,
            members,
            VAR_NAME,
            LEVEL,
            scale,
            state_lat,
            state_lon,
        )
        for result in results
    ]
    if any(bundle is None for bundle in bundles):
        missing = [result.filt for result, bundle in zip(results, bundles) if bundle is None]
        raise ValueError(f"Missing obs/state member diagnostics for: {', '.join(missing)}")
    bundles = [bundle for bundle in bundles if bundle is not None]

    increment_levels = symmetric_levels([b.result.increment_on_nr for b in bundles], MAP_COLOR_LEVELS)
    nr_prior_levels = symmetric_levels([nr_prior_difference], NR_PRIOR_CONTOURS)

    configure_matplotlib()
    ncols = len(bundles)
    fig, axs = plt.subplots(4, ncols, figsize=(4.15 * ncols, 11.2), squeeze=False, constrained_layout=True)

    for col, bundle in enumerate(bundles):
        plot_prior_transform_panel(axs[0, col], bundle, obs_info.obs_value if obs_info is not None else None)
        plot_obs_update_panel(axs[1, col], bundle)
        plot_state_response_panel(axs[2, col], bundle)
        pcm = plot_physical_panel(
            axs[3, col],
            bundle.result.increment_on_nr,
            nr_prior_difference,
            nr_lats,
            nr_lons,
            region_mask,
            tc_lat,
            tc_lon,
            obs_info,
            state_lat,
            state_lon,
            increment_levels,
            nr_prior_levels,
        )

    fig.colorbar(
        pcm,
        ax=axs[3, :],
        shrink=0.9,
        ticks=np.linspace(increment_levels[0], increment_levels[-1], 5),
        label=f"{VAR_LABELS.get(VAR_NAME, VAR_NAME)} analysis increment",
    )
    handles, labels = axs[3, 0].get_legend_handles_labels()
    if handles:
        axs[3, 0].legend(handles, labels, loc="best", fontsize=6)

    fig.suptitle(
        (
            f"Single obs {obs_point}, {domain}, {VAR_NAME}"
            f"{'' if LEVEL is None else f' level {LEVEL}'} | "
            f"state=({state_lat:.3f}, {state_lon:.3f}), NR={state_truth:.3g}; "
            f"NR center from {center_name}={tc_pressure:.1f}"
        ),
        fontsize=9,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / f"singleobs{obs_point}_{domain}_{VAR_NAME}_lev{LEVEL if LEVEL is not None else '2d'}_probit_diagnosis"
    fig.savefig(f"{stem}.{FIG_FORMAT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(f".{FIG_FORMAT}")


def main() -> None:
    if STATE_SELECTION not in {"max_abs_error", "obs_nearest", "tc_center"}:
        raise ValueError("STATE_SELECTION must be max_abs_error, obs_nearest, or tc_center")

    data_root = base.default_data_root() if DATA_ROOT is None else Path(DATA_ROOT)
    nr_file_config = None if NR_FILE is None else Path(NR_FILE)
    scale = base.auto_scale(VAR_NAME) if SCALE == "auto" else float(SCALE)

    first_run = base.resolve_run_dir(data_root, FILTERS[0], OBS_POINTS[0])
    first_mean = base.find_mean_file(first_run, DOMAINS[0])
    inferred_time = TIME_STRING or base.read_time_string(first_mean)
    nr_file = base.find_nr_file(nr_file_config, Path(NR_BASE), NR_DOMAIN, inferred_time)

    print(f"Using NR file: {nr_file}")
    wrote: list[Path] = []
    for obs_point in OBS_POINTS:
        for domain in DOMAINS:
            print(f"Plotting probit diagnosis for obs_seq{obs_point}, {domain}")
            wrote.append(make_figure(obs_point, domain, nr_file, MEMBERS, scale, data_root))
    print("Wrote figures:")
    for path in wrote:
        print(f"  {path}")


if __name__ == "__main__":
    main()
