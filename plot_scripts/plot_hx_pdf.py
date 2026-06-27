#!/usr/bin/env python3
"""
Plot probability density distributions of ensemble Hx values from obs_seq.out.

The script reads two DART obs_seq.out files with external_FO blocks.  For the
usual 50-member, 676-observation case it flattens each Hx matrix into 50*676
samples, plots density histograms with configurable bar width, and overlays
Gaussian PDFs estimated from each file's samples.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"),
)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# User configuration
# =============================================================================
OBS_SEQ_PATHS = [
    Path("/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir/obs_seq.out_kctest1_d01_10_00_00_quantile_ch4_clrsky"),
    Path("/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/1convert_obs/run_dir/obs_seq.out_kctest1_d01_10_00_00_LACC_ch4"),
]
DATASET_LABELS = ["Hx", "Hx_LACC"]
DATASET_COLORS = ["#4f7fb8", "#d08a37"]

OUTPUT_DIR = Path(__file__).resolve().parent / "figs" / "hx_pdf"
OUTPUT_NAME = "hx_pdf_compare_1000"

EXPECTED_NOBS = 676
EXPECTED_MEMBERS = 50

# Histogram bar width in Hx units.  Change this value to adjust bar width.
BIN_WIDTH = 0.25

# Extend Gaussian PDF beyond histogram limits so the tails approach zero.
PDF_TAIL_STD = 4.0

FIGSIZE = (4.8, 3.4)
DPI = 450

HX_LABEL = "Hx"
FIG_TITLE = "Ensemble Hx distributions"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def parse_obs_seq_hx(path: Path) -> np.ndarray:
    """Return Hx values as an array with shape (nobs, nmembers)."""
    lines = path.read_text(errors="replace").splitlines()
    hx_rows: list[np.ndarray] = []
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped.startswith("external_FO"):
            i += 1
            continue

        parts = stripped.split()
        if len(parts) < 2:
            raise ValueError(f"external_FO block without member count at line {i + 1}: {path}")
        nmem = int(parts[1])

        values: list[float] = []
        i += 1
        while i < len(lines) and len(values) < nmem:
            if re.match(r"\s*OBS\s+\d+", lines[i]):
                break
            try:
                values.extend(float(x) for x in lines[i].split())
            except ValueError:
                break
            i += 1

        if len(values) < nmem:
            raise ValueError(
                f"external_FO block at {path}:{i + 1} has {len(values)} values, expected {nmem}"
            )
        hx_rows.append(np.asarray(values[:nmem], dtype=float))

    if not hx_rows:
        raise RuntimeError(f"No external_FO blocks found in {path}")

    member_counts = {row.size for row in hx_rows}
    if len(member_counts) != 1:
        raise ValueError(f"Inconsistent external_FO member counts in {path}: {sorted(member_counts)}")

    return np.vstack(hx_rows)


def validate_hx_shape(hx: np.ndarray, expected_nobs: int | None, expected_members: int | None) -> None:
    if expected_nobs is not None and hx.shape[0] != expected_nobs:
        raise ValueError(f"Parsed {hx.shape[0]} observations, expected {expected_nobs}")
    if expected_members is not None and hx.shape[1] != expected_members:
        raise ValueError(f"Parsed {hx.shape[1]} members, expected {expected_members}")


def make_bins(samples: np.ndarray, bin_width: float) -> np.ndarray:
    if bin_width <= 0:
        raise ValueError("BIN_WIDTH must be positive")
    xmin = math.floor(float(np.nanmin(samples)) / bin_width) * bin_width
    xmax = math.ceil(float(np.nanmax(samples)) / bin_width) * bin_width
    if xmin == xmax:
        xmax = xmin + bin_width
    return np.arange(xmin, xmax + bin_width * 1.0001, bin_width)


def gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std <= 0 or not np.isfinite(std):
        return np.full_like(x, np.nan, dtype=float)
    z = (x - mean) / std
    return np.exp(-0.5 * z**2) / (std * math.sqrt(2.0 * math.pi))


def make_pdf_x(samples_list: list[np.ndarray], bins: np.ndarray, bin_width: float, tail_std: float) -> np.ndarray:
    means = np.asarray([np.mean(samples) for samples in samples_list], dtype=float)
    stds = np.asarray([np.std(samples, ddof=1) for samples in samples_list], dtype=float)
    finite = np.isfinite(means) & np.isfinite(stds) & (stds > 0)
    if np.any(finite):
        pdf_min = float(np.min(means[finite] - tail_std * stds[finite]))
        pdf_max = float(np.max(means[finite] + tail_std * stds[finite]))
    else:
        pdf_min = float(bins[0])
        pdf_max = float(bins[-1])
    pad = max(bin_width, 0.02 * (float(bins[-1]) - float(bins[0])))
    x_min = min(float(bins[0]), pdf_min) - pad
    x_max = max(float(bins[-1]), pdf_max) + pad
    return np.linspace(x_min, x_max, 1000)


def finite_samples(hx: np.ndarray) -> np.ndarray:
    samples = hx.reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError("No finite Hx samples available for plotting")
    return samples


def plot_hx_pdf_comparison(
    hx_list: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    bin_width: float,
    output_dir: Path,
    output_name: str,
) -> Path:
    if len(hx_list) != 2:
        raise ValueError("This comparison plot expects exactly two Hx arrays")
    if len(labels) != len(hx_list):
        raise ValueError("DATASET_LABELS length must match OBS_SEQ_PATHS")
    if len(colors) != len(hx_list):
        raise ValueError("DATASET_COLORS length must match OBS_SEQ_PATHS")

    sample_sets = [finite_samples(hx) for hx in hx_list]
    all_samples = np.concatenate(sample_sets)
    bins = make_bins(all_samples, bin_width)
    x_pdf = make_pdf_x(sample_sets, bins, bin_width, PDF_TAIL_STD)

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    stats_lines = []
    for hx, samples, label, color in zip(hx_list, sample_sets, labels, colors):
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1))
        y_pdf = gaussian_pdf(x_pdf, mean, std)
        ax.hist(
            samples,
            bins=bins,
            density=True,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.42,
            label=label,
        )
        ax.plot(
            x_pdf,
            y_pdf,
            color=color,
            lw=1.9,
            label="_nolegend_",
        )
        ax.axvline(mean, color=color, lw=1.0, ls="--", alpha=0.9)
        stats_lines.extend(
            [
                f"[{label}]",
                f"n_obs = {hx.shape[0]}",
                f"n_members = {hx.shape[1]}",
                f"n_samples = {samples.size}",
                f"mean = {mean:.12g}",
                f"std_ddof1 = {std:.12g}",
                "",
            ]
        )

    ax.set_title(FIG_TITLE)
    ax.set_xlabel(HX_LABEL)
    ax.set_ylabel("Probability density")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(loc="best", fontsize=7)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"{output_name}.png"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    stats_path = output_dir / f"{output_name}_stats.txt"
    stats_path.write_text(
        "\n".join(
            [
                "obs_seq_paths:",
                *[f"  {label}: {path}" for label, path in zip(labels, OBS_SEQ_PATHS)],
                f"bin_width = {bin_width:.12g}",
                "",
                *stats_lines,
            ]
        )
        + "\n"
    )
    return out_png


def main() -> None:
    if len(OBS_SEQ_PATHS) != 2:
        raise ValueError("OBS_SEQ_PATHS must contain exactly two obs_seq.out files")
    hx_list = []
    for path in OBS_SEQ_PATHS:
        hx = parse_obs_seq_hx(path)
        validate_hx_shape(hx, EXPECTED_NOBS, EXPECTED_MEMBERS)
        hx_list.append(hx)
    out_png = plot_hx_pdf_comparison(hx_list, DATASET_LABELS, DATASET_COLORS, BIN_WIDTH, OUTPUT_DIR, OUTPUT_NAME)
    for label, hx in zip(DATASET_LABELS, hx_list):
        print(f"{label}: parsed Hx shape {hx.shape}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
