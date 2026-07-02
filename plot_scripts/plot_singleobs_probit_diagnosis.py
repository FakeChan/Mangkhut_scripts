#!/usr/bin/env python3
"""
Plot Jeff Anderson-style single-observation PPI diagnostics.

The figure is designed to explain why one single-observation assimilation
improves or degrades the state estimate:

1. physical-space joint update: obs increment and state-variable increment,
2. joint PPI/probit-space prior and posterior distribution,
3. observation marginal: prior/posterior H(x) distributions and likelihood.

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
from scipy.special import ndtr, ndtri

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

# Observation likelihood standard deviation.  If None, use sqrt(errvar) from
# obs_seq when available; otherwise fall back to 1.0.
OBS_LIKELIHOOD_STD = None

# PPI transform used by each filter in the diagnostic plot.  EAKF uses a
# Gaussian prior CDF; QCF_RHF uses the RHF/BNRH prior CDF.  Both prior and
# posterior values are transformed through the same prior CDF, following
# Jeff Anderson's ppi_update.m.
FILTER_PPI_DISTS = {
    "EAKF": "Normal",
    "QCF_RHF": "RHF",
}

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


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if not np.isfinite(std) or std <= 0:
        return np.full_like(x, np.nan, dtype=float)
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def bnrh_sorted_quantiles(sorted_values: np.ndarray) -> np.ndarray:
    """DART_LAB ens_quantiles.m for the unbounded RHF/BNRH case."""
    n = sorted_values.size
    q = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        series_start = i + 1.0
        series_length = j - i
        q[i:j] = series_start / (n + 1.0) + (series_length - 1.0) / (2.0 * (n + 1.0))
        i = j
    return q


def bnrh_cdf_unbounded(values: np.ndarray, prior_values: np.ndarray) -> np.ndarray:
    """
    Unbounded BNRH/RHF CDF from DART_LAB bnrh_cdf_initialized.m.

    The CDF is built from the prior ensemble: uniform mass between neighboring
    sorted members and normal tails outside the ensemble range.
    """
    values = np.asarray(values, dtype=float)
    prior = np.asarray(prior_values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    prior = prior[np.isfinite(prior)]
    if prior.size < 2:
        return out

    sort_ens = np.sort(prior)
    n = sort_ens.size
    q_exact = bnrh_sorted_quantiles(sort_ens)
    del_q = 1.0 / (n + 1.0)
    tail_sd = float(np.std(prior, ddof=1))
    if not np.isfinite(tail_sd) or tail_sd <= 0:
        return out

    tail_del_q = 1.0 / (n + 1.8)
    dist_for_unit_sd = -float(ndtri(tail_del_q))
    tail_mean_left = sort_ens[0] + dist_for_unit_sd * tail_sd
    tail_mean_right = sort_ens[-1] - dist_for_unit_sd * tail_sd
    left_edge_cdf = ndtr((sort_ens[0] - tail_mean_left) / tail_sd)
    right_edge_cdf = ndtr((sort_ens[-1] - tail_mean_right) / tail_sd)

    vals = values[valid]
    q = np.empty(vals.shape, dtype=float)
    for idx, value in enumerate(vals):
        if value < sort_ens[0]:
            q[idx] = ndtr((value - tail_mean_left) / tail_sd) / left_edge_cdf * del_q
            q[idx] = min(q[idx], q_exact[0])
        elif value == sort_ens[0]:
            q[idx] = q_exact[0]
        elif value > sort_ens[-1]:
            fract = (ndtr((value - tail_mean_right) / tail_sd) - right_edge_cdf) / (1.0 - right_edge_cdf)
            q[idx] = n * del_q + fract * del_q
            q[idx] = min(q[idx], 1.0)
        elif value == sort_ens[-1]:
            q[idx] = q_exact[-1]
        else:
            upper = int(np.searchsorted(sort_ens, value, side="right"))
            if upper > 0 and sort_ens[upper - 1] == value:
                q[idx] = q_exact[upper - 1]
            else:
                lower = upper - 1
                q[idx] = (lower + 1.0) * del_q + (
                    (value - sort_ens[lower]) / (sort_ens[upper] - sort_ens[lower])
                ) * del_q
    out[valid] = q
    return out


def empirical_normal_scores(values: np.ndarray) -> np.ndarray:
    """Map one ensemble to normal scores using its own empirical ranks."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return out
    order = np.argsort(values[valid])
    ranks = np.empty(valid_idx.size, dtype=float)
    ranks[order] = (np.arange(valid_idx.size, dtype=float) + 1.0) / (valid_idx.size + 1.0)
    out[valid_idx] = ndtri(ranks)
    return out


def values_to_ppi_from_prior(values: np.ndarray, prior_values: np.ndarray, dist_type: str) -> np.ndarray:
    """
    Transform values to PPI/probit space using the prior distribution CDF.

    This mirrors the plotting geometry in DART_LAB's ppi_update.m: both prior
    and posterior state values are mapped through the prior state CDF, then
    through the inverse standard normal CDF.
    """
    values = np.asarray(values, dtype=float)
    prior_values = np.asarray(prior_values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid_values = np.isfinite(values)
    prior = prior_values[np.isfinite(prior_values)]
    if prior.size == 0:
        return out

    dist_name = dist_type.upper()
    if dist_name == "NORMAL":
        mean = float(np.nanmean(prior))
        std = float(np.nanstd(prior, ddof=1))
        if not np.isfinite(std) or std <= 0:
            return out
        q = ndtr((values[valid_values] - mean) / std)
    elif dist_name == "RHF":
        q = bnrh_cdf_unbounded(values[valid_values], prior)
    else:
        raise ValueError("STATE_PPI_DIST must be 'RHF' or 'Normal'")

    eps = np.finfo(float).eps
    q = np.clip(q, eps, 1.0 - eps)
    out[valid_values] = ndtri(q)
    return out


def gaussian_kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Small dependency-light KDE for ensemble marginal display."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    if values.size == 1:
        bandwidth = max(abs(values[0]) * 0.05, 1.0)
    else:
        std = float(np.nanstd(values, ddof=1))
        bandwidth = 1.06 * std * values.size ** (-1.0 / 5.0)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = max(std, 1.0)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.nanmean(np.exp(-0.5 * z * z), axis=1) / (bandwidth * math.sqrt(2.0 * math.pi))


def observation_likelihood_std(obs_info: base.ObsSeqInfo | None) -> float:
    if OBS_LIKELIHOOD_STD is not None:
        obs_std = float(OBS_LIKELIHOOD_STD)
        if not np.isfinite(obs_std) or obs_std <= 0:
            raise ValueError("OBS_LIKELIHOOD_STD must be a positive finite number")
        return obs_std
    if obs_info is not None and obs_info.errvar is not None and obs_info.errvar > 0:
        return math.sqrt(float(obs_info.errvar))
    return 1.0


def ppi_dist_for_filter(filt: str) -> str:
    return FILTER_PPI_DISTS.get(filt, "RHF")


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


def plot_physical_joint_update_panel(
    ax: plt.Axes,
    bundle: PlotBundle,
    obs_value: float | None,
    state_truth: float,
) -> None:
    """Jeff-style physical-space prior/posterior joint ensemble plot."""
    prior_color = "#7b5fc8"
    post_color = "#57a773"
    line_color = "#48b9c7"
    truth_color = "#d73027"

    for xo, xp, yo, yp in zip(bundle.obs_prior, bundle.obs_post, bundle.state_prior, bundle.state_post):
        if np.all(np.isfinite([xo, xp, yo, yp])):
            ax.plot([xo, xp], [yo, yp], color=line_color, lw=0.65, alpha=0.65, zorder=1)

    ax.scatter(bundle.obs_prior, bundle.state_prior, s=16, color=prior_color, alpha=0.78, label="prior", zorder=2)
    ax.scatter(bundle.obs_post, bundle.state_post, s=16, color=post_color, alpha=0.78, label="posterior", zorder=3)
    annotate_fit(ax, bundle.obs_prior, bundle.state_prior, color=prior_color)
    if obs_value is not None and np.isfinite(obs_value) and np.isfinite(state_truth):
        ax.scatter(obs_value, state_truth, s=58, marker="v", color=truth_color, label="obs/NR", zorder=4)

    ax.set_title(f"{FILTER_LABELS.get(bundle.result.filt, bundle.result.filt)} physical-space joint update")
    ax.set_xlabel("Observed quantity H(x)")
    ax.set_ylabel(VAR_LABELS.get(VAR_NAME, VAR_NAME))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(True, color="0.9", lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=6)


def plot_ppi_joint_panel(ax: plt.Axes, bundle: PlotBundle) -> None:
    """Plot prior/posterior joint ensemble in probit probability integral space."""
    ppi_dist = ppi_dist_for_filter(bundle.result.filt)
    if bundle.probit_prior is not None and bundle.probit_post is not None:
        obs_prior_ppi = bundle.probit_prior
        obs_post_ppi = bundle.probit_post
        obs_note = "logged obs probit"
    else:
        obs_prior_ppi = values_to_ppi_from_prior(bundle.obs_prior, bundle.obs_prior, ppi_dist)
        obs_post_ppi = values_to_ppi_from_prior(bundle.obs_post, bundle.obs_prior, ppi_dist)
        obs_note = f"{ppi_dist} prior CDF fallback"

    state_prior_ppi = values_to_ppi_from_prior(bundle.state_prior, bundle.state_prior, ppi_dist)
    obs_increment_ppi = obs_post_ppi - obs_prior_ppi
    valid_reg = np.isfinite(obs_prior_ppi) & np.isfinite(obs_increment_ppi) & np.isfinite(state_prior_ppi)
    state_post_ppi = np.full_like(state_prior_ppi, np.nan, dtype=float)
    if np.count_nonzero(valid_reg) >= 2:
        covar = np.cov(obs_prior_ppi[valid_reg], state_prior_ppi[valid_reg])
        obs_var = covar[0, 0]
        reg_coef = covar[0, 1] / obs_var if np.isfinite(obs_var) and obs_var > 0 else np.nan
        state_post_ppi[valid_reg] = state_prior_ppi[valid_reg] + reg_coef * obs_increment_ppi[valid_reg]
        reg_note = f"PPI reg={reg_coef:.3g}"
    else:
        reg_note = "PPI reg unavailable"

    prior_color = "#7b5fc8"
    post_color = "#57a773"
    line_color = "#48b9c7"
    for xo, xp, yo, yp in zip(obs_prior_ppi, obs_post_ppi, state_prior_ppi, state_post_ppi):
        if np.all(np.isfinite([xo, xp, yo, yp])):
            ax.plot([xo, xp], [yo, yp], color=line_color, lw=0.65, alpha=0.7, zorder=1)
    ax.scatter(obs_prior_ppi, state_prior_ppi, s=16, color=prior_color, alpha=0.78, label="prior", zorder=2)
    ax.scatter(obs_post_ppi, state_post_ppi, s=16, color=post_color, alpha=0.78, label="posterior", zorder=3)
    annotate_fit(ax, obs_prior_ppi, state_prior_ppi, color=prior_color)

    ax.axhline(0, color="0.82", lw=0.7)
    ax.axvline(0, color="0.82", lw=0.7)
    ax.set_title(f"Joint PPI space distribution ({reg_note})")
    ax.set_xlabel(f"PPI transformed observed ({obs_note})")
    ax.set_ylabel(f"PPI transformed state ({ppi_dist} prior CDF)")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.grid(True, color="0.9", lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=6)


def plot_obs_marginal_panel(ax: plt.Axes, bundle: PlotBundle, obs_value: float | None, obs_std: float) -> None:
    """Plot prior/posterior H(x) marginal distributions and observation likelihood."""
    all_values = np.concatenate([bundle.obs_prior[np.isfinite(bundle.obs_prior)], bundle.obs_post[np.isfinite(bundle.obs_post)]])
    if obs_value is not None and np.isfinite(obs_value):
        all_values = np.concatenate([all_values, np.asarray([obs_value])])
    if all_values.size == 0:
        ax.text(0.5, 0.5, "No obs marginal values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    spread = float(np.nanmax(all_values) - np.nanmin(all_values))
    pad = max(0.15 * spread, obs_std * 3.0, 1.0)
    grid = np.linspace(float(np.nanmin(all_values) - pad), float(np.nanmax(all_values) + pad), 300)
    prior_pdf = gaussian_kde_curve(bundle.obs_prior, grid)
    post_pdf = gaussian_kde_curve(bundle.obs_post, grid)

    prior_color = "#7b5fc8"
    post_color = "#57a773"
    like_color = "#d73027"
    ax.plot(grid, prior_pdf, color=prior_color, lw=1.5, label="prior H(x)")
    ax.plot(grid, post_pdf, color=post_color, lw=1.5, label="posterior H(x)")
    y0 = -0.06 * float(np.nanmax([np.nanmax(prior_pdf), np.nanmax(post_pdf), 1.0]))
    ax.scatter(bundle.obs_prior, np.full_like(bundle.obs_prior, y0), marker="*", s=28, color=prior_color, alpha=0.75)
    ax.scatter(bundle.obs_post, np.full_like(bundle.obs_post, y0 * 1.7), marker="*", s=28, color=post_color, alpha=0.75)

    if obs_value is not None and np.isfinite(obs_value):
        likelihood = normal_pdf(grid, float(obs_value), obs_std)
        ax.plot(grid, likelihood, color=like_color, lw=1.5, ls="--", label=f"likelihood std={obs_std:g}")
        ax.scatter(obs_value, 0.0, marker="*", s=70, color=like_color, zorder=4, label="obs")

    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_title("Marginal distribution of observation")
    ax.set_xlabel("Observed quantity H(x)")
    ax.set_ylabel("density / likelihood")
    ax.grid(True, color="0.9", lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=6)


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

    obs_std = observation_likelihood_std(obs_info)
    obs_value = obs_info.obs_value if obs_info is not None else None

    configure_matplotlib()
    ncols = len(bundles)
    fig, axs = plt.subplots(3, ncols, figsize=(4.35 * ncols, 9.2), squeeze=False, constrained_layout=True)

    for col, bundle in enumerate(bundles):
        plot_physical_joint_update_panel(axs[0, col], bundle, obs_value, state_truth)
        plot_ppi_joint_panel(axs[1, col], bundle)
        plot_obs_marginal_panel(axs[2, col], bundle, obs_value, obs_std)

    fig.suptitle(
        (
            f"Single obs {obs_point}, {domain}, {VAR_NAME}"
            f"{'' if LEVEL is None else f' level {LEVEL}'} | "
            f"state=({state_lat:.3f}, {state_lon:.3f}), NR={state_truth:.3g}; "
            f"obs std={obs_std:g}; NR center from {center_name}={tc_pressure:.1f}"
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
