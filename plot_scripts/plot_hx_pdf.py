#!/usr/bin/env python3
"""
Plot the probability density distribution of ensemble Hx values from obs_seq.out.

The script reads DART obs_seq.out files with external_FO blocks.  For the usual
50-member, 676-observation case it flattens the Hx matrix into 50*676 samples,
plots a density histogram with configurable bar width, and overlays the
Gaussian PDF estimated from all samples.
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
OBS_SEQ_PATH = Path("/share/home/lililei1/kcfu/tc_mangkhut/4assimilation/2DART/run_dir/obs_seq.out")
OUTPUT_DIR = Path(__file__).resolve().parent / "figs" / "hx_pdf"
OUTPUT_NAME = "hx_pdf"

EXPECTED_NOBS = 676
EXPECTED_MEMBERS = 50

# Histogram bar width in Hx units.  Change this value to adjust bar width.
BIN_WIDTH = 0.25

FIGSIZE = (4.8, 3.4)
DPI = 450

HX_LABEL = "Hx"
FIG_TITLE = "Ensemble Hx distribution"


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


def plot_hx_pdf(hx: np.ndarray, bin_width: float, output_dir: Path, output_name: str) -> Path:
    samples = hx.reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError("No finite Hx samples available for plotting")

    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    bins = make_bins(samples, bin_width)
    x_pdf = np.linspace(bins[0], bins[-1], 800)
    y_pdf = gaussian_pdf(x_pdf, mean, std)

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(
        samples,
        bins=bins,
        density=True,
        color="#6f8fbf",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.82,
        label=f"Hx histogram (bin={bin_width:g})",
    )
    ax.plot(
        x_pdf,
        y_pdf,
        color="#c23b3b",
        lw=1.8,
        label=rf"Gaussian $\mu$={mean:.3g}, $\sigma$={std:.3g}",
    )
    ax.axvline(mean, color="#2f2f2f", lw=1.0, ls="--", label="Mean")

    ax.set_title(FIG_TITLE)
    ax.set_xlabel(HX_LABEL)
    ax.set_ylabel("Probability density")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(loc="best", fontsize=7)

    ax.text(
        0.98,
        0.96,
        f"n_obs={hx.shape[0]}\nn_mem={hx.shape[1]}\nn={samples.size}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9, "pad": 3},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"{output_name}.png"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    stats_path = output_dir / f"{output_name}_stats.txt"
    stats_path.write_text(
        "\n".join(
            [
                f"obs_seq_path = {OBS_SEQ_PATH}",
                f"n_obs = {hx.shape[0]}",
                f"n_members = {hx.shape[1]}",
                f"n_samples = {samples.size}",
                f"mean = {mean:.12g}",
                f"std_ddof1 = {std:.12g}",
                f"bin_width = {bin_width:.12g}",
            ]
        )
        + "\n"
    )
    return out_png


def main() -> None:
    hx = parse_obs_seq_hx(OBS_SEQ_PATH)
    validate_hx_shape(hx, EXPECTED_NOBS, EXPECTED_MEMBERS)
    out_png = plot_hx_pdf(hx, BIN_WIDTH, OUTPUT_DIR, OUTPUT_NAME)
    print(f"Parsed Hx shape: {hx.shape}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
