"""Publication-style evidence figure for strong-versus-weak coupling skill."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


# =====================
# User configuration
# =====================
PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_DIR / "tmp" / "omtmp_skill_cache"
OUTPUT_STEM = PROJECT_DIR / "Mangkhut_scripts" / "plot_scripts" / "figs" / "omtmp_skill_evidence"
PANEL_OUTPUT_DIR = OUTPUT_STEM.parent / f"{OUTPUT_STEM.name}_panels"
OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)  # ensure figs/ exists before savefig
EXPORT_FORMAT = "png"  # single export format for the full figure and panels: "png", "svg", or "pdf"
assert EXPORT_FORMAT in ("png", "svg", "pdf"), f"unsupported EXPORT_FORMAT: {EXPORT_FORMAT}"
FIGURE_SIZE_INCHES = (7.2, 7.45)
PNG_DPI = 400
BOOTSTRAP_SAMPLES = 20_000
RANDOM_SEED = 20260804
EARLY_TIMES = (0.5, 1.0, 1.5, 2.0)


# Mandatory editable-text settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 6.3,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.0,
    }
)

COLORS = {
    "ocean": "#0F4D92",
    "skin": "#3775BA",
    "moisture": "#42949E",
    "moisture_2": "#77B8B2",
    "temperature": "#9A4D8E",
    "temperature_2": "#C08AB4",
    "flux": "#B64342",
    "flux_2": "#E28E86",
    "neutral": "#606060",
    "gain_fill": "#EAF5EC",
    "loss_fill": "#FBEDEC",
    "zero": "#4D4D4D",
}


def bootstrap_ci(values, samples=BOOTSTRAP_SAMPLES, seed=RANDOM_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.size, size=(samples, values.size))
    means = values[index].mean(axis=1)
    return np.quantile(means, (0.025, 0.975))


def add_panel_label(ax, label, x=-0.10, y=1.04):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def panel_bbox_inches(fig, artists, pad=0.04):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = []
    for artist in artists:
        bbox = artist.get_tightbbox(renderer)
        if bbox is not None:
            bboxes.append(bbox)
    if not bboxes:
        raise ValueError("No drawable artists were provided for panel export.")
    bbox = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
    return bbox.padded(pad)


def save_individual_panels(fig, panel_artists):
    PANEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for panel_name, artists in panel_artists.items():
        bbox = panel_bbox_inches(fig, artists)
        kwargs = {"bbox_inches": bbox, "facecolor": "white"}
        if EXPORT_FORMAT == "png":
            kwargs["dpi"] = PNG_DPI
        fig.savefig(PANEL_OUTPUT_DIR / f"{panel_name}.{EXPORT_FORMAT}", **kwargs)


def selected_metrics(metrics):
    return metrics[
        (metrics.region == "r000_300")
        & (metrics.mask_type == "ocean")
        & (metrics.time_hour > 0.0)
    ].copy()


def hero_ratios(metrics):
    variables = ["om", "tsk", "q2", "qv_l1", "t2", "theta_l1", "hfx", "lh"]
    early = metrics[metrics.time_hour.isin(EARLY_TIMES) & metrics.variable.isin(variables)]
    cases = early.groupby(["method", "member", "variable"], as_index=False).agg(
        strong=("strong_rmse", "mean"), weak=("weak_rmse", "mean")
    )
    cases["ratio"] = cases.strong / cases.weak
    rows = []
    for variable, group in cases.groupby("variable"):
        low, high = bootstrap_ci(group.ratio)
        rows.append(
            {
                "variable": variable,
                "mean": group.ratio.mean(),
                "low": low,
                "high": high,
                "n": len(group),
            }
        )
    return pd.DataFrame(rows).set_index("variable").loc[variables].reset_index(), cases


def time_statistics(metrics, variables):
    rows = []
    data = metrics[metrics.variable.isin(variables)]
    for (variable, time_hour), group in data.groupby(["variable", "time_hour"]):
        values = group.rmse_improvement_pct.to_numpy(float)
        low, high = bootstrap_ci(values, seed=RANDOM_SEED + int(time_hour * 10))
        rows.append(
            {
                "variable": variable,
                "time_hour": time_hour,
                "mean": values.mean(),
                "low": low,
                "high": high,
            }
        )
    return pd.DataFrame(rows)


def plot_hero(ax, metrics):
    ratios, _ = hero_ratios(metrics)
    labels = {
        "om": "OM_TMP(0)",
        "tsk": "TSK",
        "q2": "Q2",
        "qv_l1": "QVAPOR L1",
        "t2": "T2",
        "theta_l1": r"$\theta$ L1",
        "hfx": "HFX",
        "lh": "LH",
    }
    colors = {
        "om": COLORS["ocean"],
        "tsk": COLORS["skin"],
        "q2": COLORS["moisture"],
        "qv_l1": COLORS["moisture_2"],
        "t2": COLORS["temperature"],
        "theta_l1": COLORS["temperature_2"],
        "hfx": COLORS["flux"],
        "lh": COLORS["flux_2"],
    }
    y = np.arange(len(ratios))[::-1]
    ax.axvspan(0.70, 1.0, color=COLORS["gain_fill"], zorder=0)
    ax.axvspan(1.0, 1.02, color=COLORS["loss_fill"], zorder=0)
    ax.axvline(1.0, color=COLORS["zero"], lw=0.9, ls="--", zorder=1)
    for yi, row in zip(y, ratios.itertuples()):
        error = np.array([[row.mean - row.low], [row.high - row.mean]])
        ax.errorbar(
            row.mean,
            yi,
            xerr=error,
            fmt="o",
            color=colors[row.variable],
            ecolor=colors[row.variable],
            elinewidth=1.1,
            capsize=2.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
        delta = 100.0 * (1.0 - row.mean)
        ax.text(
            1.012,
            yi,
            f"{delta:+.3g}%",
            ha="right",
            va="center",
            fontsize=6.2,
            color=colors[row.variable],
        )
    ax.set_yticks(y, [labels[value] for value in ratios.variable])
    ax.set_xlim(0.68, 1.015)
    ax.set_xlabel("RMSE ratio, strong / weak  (lower is better)")
    ax.set_title("Early forecast skill (0.5–2 h, ocean, 0–300 km)", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D8D8D8", lw=0.45, alpha=0.8)
    add_panel_label(ax, "a", x=-0.07)


def plot_ocean_time(ax, metrics):
    statistics = time_statistics(metrics, ["om", "tsk"])
    for variable, label, color, marker in (
        ("om", "OM_TMP(0)", COLORS["ocean"], "o"),
        ("tsk", "TSK", COLORS["skin"], "s"),
    ):
        data = statistics[statistics.variable == variable].sort_values("time_hour")
        ax.plot(data.time_hour, data["mean"], marker=marker, color=color, label=label)
        ax.fill_between(data.time_hour, data.low, data.high, color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.0, color=COLORS["zero"], ls="--", lw=0.8)
    ax.set_xticks([0.5, 1, 2, 3, 4, 6])
    ax.set_xlabel("Forecast hour")
    ax.set_ylabel("RMSE improvement (%)")
    ax.set_title("Ocean/skin advantage persists", loc="left", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#D8D8D8", lw=0.45, alpha=0.8)
    add_panel_label(ax, "b")


def plot_atmosphere_time(ax, metrics):
    variables = ["q2", "qv_l1", "t2", "theta_l1", "hfx", "lh"]
    statistics = time_statistics(metrics, variables)
    styles = {
        "q2": ("Q2", COLORS["moisture"], "o"),
        "qv_l1": ("QVAPOR L1", COLORS["moisture_2"], "s"),
        "t2": ("T2", COLORS["temperature"], "^"),
        "theta_l1": (r"$\theta$ L1", COLORS["temperature_2"], "v"),
        "hfx": ("HFX", COLORS["flux"], "D"),
        "lh": ("LH", COLORS["flux_2"], "P"),
    }
    for variable in variables:
        label, color, marker = styles[variable]
        data = statistics[statistics.variable == variable].sort_values("time_hour")
        ax.plot(data.time_hour, data["mean"], marker=marker, color=color, label=label)
    ax.axhline(0.0, color=COLORS["zero"], ls="--", lw=0.8)
    ax.set_xticks([0.5, 1, 2, 3, 4, 6])
    ax.set_xlabel("Forecast hour")
    ax.set_ylabel("RMSE improvement (%)")
    ax.set_ylim(-0.72, 0.30)
    ax.set_title("Atmospheric gains are small and variable-dependent", loc="left", fontweight="bold")
    ax.legend(ncol=3, loc="lower left", columnspacing=0.9, handletextpad=0.35)
    ax.grid(axis="y", color="#D8D8D8", lw=0.45, alpha=0.8)
    add_panel_label(ax, "c", x=-0.07)


def plot_vertical(ax, vertical):
    data = vertical[
        (vertical.window == "early_0p5_2h")
        & (vertical.method_group == "ALL")
        & (vertical.region == "r000_300")
        & (vertical.mask_type == "ocean")
    ]
    for variable, label, color, marker in (
        ("qv", "QVAPOR", COLORS["moisture"], "o"),
        ("theta", r"$\theta$", COLORS["temperature"], "s"),
    ):
        group = data[data.variable == variable].sort_values("level")
        x = group.mean_rmse_improvement_pct.to_numpy()
        low = group.ci025_rmse_improvement_pct.to_numpy()
        high = group.ci975_rmse_improvement_pct.to_numpy()
        y = group.level.to_numpy()
        ax.plot(x, y, marker=marker, color=color, label=label)
        ax.fill_betweenx(y, low, high, color=color, alpha=0.12, linewidth=0)
    ax.axvline(0.0, color=COLORS["zero"], ls="--", lw=0.8)
    ax.set_ylim(0.7, 10.3)
    ax.set_yticks([1, 3, 5, 7, 9, 10])
    ax.set_xlabel("RMSE improvement (%)")
    ax.set_ylabel("Model level")
    ax.set_title("Moisture signal decays upward", loc="left", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="x", color="#D8D8D8", lw=0.45, alpha=0.8)
    add_panel_label(ax, "d")


def plot_method_split(ax, summary):
    data = summary[
        (summary.window == "early_0p5_2h")
        & (summary.region == "r000_300")
        & (summary.mask_type == "ocean")
        & summary.method_group.isin(["QCF_RHF", "EAKF"])
        & summary.variable.isin(["om", "tsk"])
    ]
    x = np.array([0.0, 1.0])
    width = 0.28
    for offset, variable, label, color in (
        (-width / 2, "om", "OM_TMP(0)", COLORS["ocean"]),
        (width / 2, "tsk", "TSK", COLORS["skin"]),
    ):
        values = []
        lows = []
        highs = []
        for method in ("QCF_RHF", "EAKF"):
            row = data[(data.method_group == method) & (data.variable == variable)].iloc[0]
            values.append(row.mean_rmse_improvement_pct)
            lows.append(row.ci025_rmse_improvement_pct)
            highs.append(row.ci975_rmse_improvement_pct)
        values = np.asarray(values)
        ax.bar(x + offset, values, width=width, color=color, alpha=0.86, label=label)
        ax.errorbar(
            x + offset,
            values,
            yerr=np.vstack((values - lows, np.asarray(highs) - values)),
            fmt="none",
            ecolor="#303030",
            elinewidth=0.8,
            capsize=2,
        )
    ax.axhline(0.0, color=COLORS["zero"], ls="--", lw=0.8)
    ax.set_xticks(x, ["QCF_RHF", "EAKF"])
    ax.set_ylabel("RMSE improvement (%)")
    ax.set_title("Ocean advantage is method-robust", loc="left", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#D8D8D8", lw=0.45, alpha=0.8)
    add_panel_label(ax, "e")


def plot_intensity_axis(ax, mean, low, high, title, unit, panel_label=None):
    color = COLORS["moisture"] if low > 0.0 else COLORS["neutral"]
    span = max(abs(low), abs(high), 1.0e-6)
    ax.axvline(0.0, color=COLORS["zero"], ls="--", lw=0.8)
    ax.errorbar(
        mean,
        0.0,
        xerr=np.array([[mean - low], [high - mean]]),
        fmt="o",
        color=color,
        ecolor=color,
        capsize=2.5,
        markeredgecolor="white",
        markeredgewidth=0.4,
    )
    ax.set_xlim(-1.25 * span, 1.25 * span)
    ax.set_yticks([])
    ax.set_xlabel(unit)
    ax.set_title(title, loc="center", fontsize=7, fontweight="bold")
    ax.text(
        0.5,
        0.78,
        f"{mean:+.3g}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        color=color,
    )
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#D8D8D8", lw=0.4, alpha=0.7)
    if panel_label:
        add_panel_label(ax, panel_label, x=-0.25, y=1.15)


def write_caption():
    text = """# Figure caption

Strong coupling robustly improves the ocean surface and skin temperature, but the atmospheric and tropical-cyclone forecast benefits are strongly attenuated. (a) Paired RMSE ratios during 0.5–2 h over ocean points within 300 km of the NR storm center; points and 95% bootstrap confidence intervals use 12 method/member paired cases after averaging time within case. Values below one favor strong coupling. (b) Time evolution of the ocean and skin-temperature RMSE improvement. (c) Atmospheric and surface-flux improvements on an expanded scale. (d) Early-window vertical profiles of QVAPOR and perturbation-potential-temperature skill. (e) Ocean/skin improvements separated by assimilation method. (f–h) Early-window changes in track, minimum-pressure and maximum-wind absolute errors. Positive improvement is weak-error minus strong-error. NR fields were bilinearly interpolated to the experiment grid; 00:30 and 01:30 NR truth fields were linearly interpolated in time between hourly outputs.
"""
    (OUTPUT_STEM.parent / f"{OUTPUT_STEM.name}_caption.md").write_text(text, encoding="utf-8")


def main():
    metrics_raw = pd.read_csv(CACHE_DIR / "omtmp_skill_metrics.csv", dtype={"member": str})
    metrics = selected_metrics(metrics_raw)
    summary = pd.read_csv(CACHE_DIR / "omtmp_skill_summary.csv")
    vertical = pd.read_csv(CACHE_DIR / "omtmp_skill_vertical_summary.csv")
    intensity = pd.read_csv(CACHE_DIR / "omtmp_skill_intensity_summary.csv")

    fig = plt.figure(figsize=FIGURE_SIZE_INCHES, constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.20, 1.05, 0.78],
        width_ratios=[1.05, 1.00, 0.92],
        left=0.09,
        right=0.985,
        bottom=0.07,
        top=0.91,
        hspace=0.55,
        wspace=0.48,
    )
    ax_a = fig.add_subplot(grid[0, 0:2])
    ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, 0:2])
    ax_d = fig.add_subplot(grid[1, 2])
    ax_e = fig.add_subplot(grid[2, 0])
    intensity_grid = grid[2, 1:3].subgridspec(1, 3, wspace=0.55)
    ax_f = fig.add_subplot(intensity_grid[0, 0])
    ax_g = fig.add_subplot(intensity_grid[0, 1])
    ax_h = fig.add_subplot(intensity_grid[0, 2])

    plot_hero(ax_a, metrics)
    plot_ocean_time(ax_b, metrics)
    plot_atmosphere_time(ax_c, metrics)
    plot_vertical(ax_d, vertical)
    plot_method_split(ax_e, summary)

    row = intensity[
        (intensity.window == "early_0p5_2h") & (intensity.method_group == "ALL")
    ].iloc[0]
    plot_intensity_axis(
        ax_f,
        row.mean_track_improvement_km,
        row.ci025_track_improvement_km,
        row.ci975_track_improvement_km,
        "Track error",
        "Improvement (km)",
        panel_label="f",
    )
    plot_intensity_axis(
        ax_g,
        row.mean_min_psfc_improvement_hpa,
        row.ci025_min_psfc_improvement_hpa,
        row.ci975_min_psfc_improvement_hpa,
        "Min PSFC error",
        "Improvement (hPa)",
        panel_label="g",
    )
    plot_intensity_axis(
        ax_h,
        row.mean_max_ws10_improvement,
        row.ci025_max_ws10_improvement,
        row.ci975_max_ws10_improvement,
        "Max 10-m wind error",
        r"Improvement (m s$^{-1}$)",
        panel_label="h",
    )

    fig.suptitle(
        "Strong coupling improves ocean/skin temperature, but atmospheric and TC gains are limited",
        x=0.09,
        y=0.972,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.944,
        "Positive improvement favors strong coupling; shading/error bars show 95% paired-case bootstrap intervals (n = 12).",
        ha="left",
        va="top",
        fontsize=6.8,
        color=COLORS["neutral"],
    )

    save_individual_panels(
        fig,
        {
            "a_rmse_ratio": [ax_a],
            "b_ocean_skin_time": [ax_b],
            "c_atmosphere_flux_time": [ax_c],
            "d_vertical_profiles": [ax_d],
            "e_method_split": [ax_e],
            "f_track_error": [ax_f],
            "g_min_psfc_error": [ax_g],
            "h_max_wind_error": [ax_h],
        },
    )
    kwargs = {}
    if EXPORT_FORMAT == "png":
        kwargs["dpi"] = PNG_DPI
    fig.savefig(OUTPUT_STEM.with_suffix(f".{EXPORT_FORMAT}"), bbox_inches="tight", **kwargs)
    plt.close(fig)
    write_caption()
    print(f"saved {OUTPUT_STEM}.{EXPORT_FORMAT}")
    print(f"saved individual panels under {PANEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
