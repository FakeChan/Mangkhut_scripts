"""Create a six-panel evidence figure for the OM_TMP-atmosphere pathway."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


# =====================
# User configuration
# =====================
PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_DIR / "tmp" / "omtmp_pathway_cache"
OUTPUT_BASE = PROJECT_DIR / "Mangkhut_scripts" / "plot_scripts" / "figs" / "omtmp_pathway_evidence"
PANEL_OUTPUT_DIR = OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_panels"
OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)  # ensure figs/ exists before savefig
EXPORT_FORMAT = "png"  # single export format for the full figure and panels: "png", "svg", or "pdf"
assert EXPORT_FORMAT in ("png", "svg", "pdf"), f"unsupported EXPORT_FORMAT: {EXPORT_FORMAT}"
FIGURE_SIZE_INCH = (7.2, 8.1)
PNG_DPI = 400
REGION = "0-150"
VERTICAL_ANNULUS = 1  # 75–150 km, where the boundary-layer signal is clearest.
METHODS = ("EAKF", "QCF_RHF")
METHOD_LABELS = {"EAKF": "EAKF", "QCF_RHF": "QCF-RHF"}
METHOD_COLORS = {"EAKF": "#42949E", "QCF_RHF": "#0F4D92"}
TIMES = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
EARLY_TIMES = np.array([0.5, 1.0, 1.5, 2.0])
LEVEL_HEIGHTS_M = np.array([13.0, 47.6, 99.6, 160.7, 230.8, 310.1, 394.3, 492.6, 605.3, 732.7])


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
    }
)


def panel_label(ax, label):
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
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


surface = pd.read_csv(
    CACHE_DIR / "omtmp_pathway_surface_blocks.csv",
    dtype={"method": str, "member": str},
)
links = pd.read_csv(CACHE_DIR / "omtmp_pathway_link_summary.csv")
direct_member = pd.read_csv(
    CACHE_DIR / "omtmp_pathway_direct_flux_member.csv",
    dtype={"method": str, "member": str},
)
vertical = pd.read_csv(CACHE_DIR / "omtmp_pathway_vertical_summary.csv")

surface["member"] = surface["member"].str.zfill(3)
direct_member["member"] = direct_member["member"].str.zfill(3)
expected_members = {"006", "015", "029", "037", "043", "044"}
assert set(surface["method"]) == set(METHODS)
assert set(surface["member"]) == expected_members
assert set(TIMES).issubset(set(surface["time_hour"]))

fig = plt.figure(figsize=FIGURE_SIZE_INCH, layout="constrained")
gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.1), hspace=0.18, wspace=0.16)
axes = [fig.add_subplot(gs[row, col]) for row in range(3) for col in range(2)]
ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes

# a — member-level source increments and ensemble cancellation.
source = surface[
    (surface["time_hour"] == 0.0) & (surface["annulus"].isin((0, 1)))
]
source_member = (
    source.groupby(["method", "member"], as_index=False)["dom0"].mean()
)
member_order = sorted(expected_members)
x = np.arange(len(member_order), dtype=float)
width = 0.34
for offset, method in zip((-width / 2, width / 2), METHODS):
    values = (
        source_member[source_member["method"] == method]
        .set_index("member")
        .loc[member_order, "dom0"]
        .to_numpy()
    )
    ax_a.bar(
        x + offset,
        values,
        width=width,
        color=METHOD_COLORS[method],
        edgecolor="white",
        linewidth=0.5,
        label=METHOD_LABELS[method],
    )
    ensemble_mean = float(np.mean(values))
    ax_a.axhline(ensemble_mean, color=METHOD_COLORS[method], lw=0.9, ls=(0, (3, 2)), alpha=0.8)
ax_a.axhline(0.0, color="#4D4D4D", lw=0.7)
ax_a.set_xticks(x, member_order)
ax_a.set_xlabel("Paired ensemble member")
ax_a.set_ylabel(r"Area-mean $\Delta$OM$_0$ (K)")
ax_a.set_title("Member cancellation hides the ocean increment", loc="left", pad=4)
ax_a.legend(ncol=2, loc="upper right", handlelength=1.4, columnspacing=0.8)
ax_a.text(
    0.02,
    0.04,
    "3 warming / 3 cooling members\nensemble mean ≈ 0",
    transform=ax_a.transAxes,
    color="#4D4D4D",
    fontsize=6.4,
    va="bottom",
)
panel_label(ax_a, "a")

# b — fixed-atmosphere direct response versus the 30-min model response.
flux_specs = (
    ("EAKF", "direct_dhfx", "EAKF\nHFX"),
    ("EAKF", "direct_dlh", "EAKF\nLH"),
    ("QCF_RHF", "direct_dhfx", "QCF-RHF\nHFX"),
    ("QCF_RHF", "direct_dlh", "QCF-RHF\nLH"),
)
x_b = np.arange(len(flux_specs), dtype=float)
bar_width = 0.34
rng = np.random.default_rng(20260803)
for i, (method, flux_name, _) in enumerate(flux_specs):
    rows = direct_member[
        (direct_member["method"] == method)
        & (direct_member["region"] == REGION)
        & (direct_member["flux"] == flux_name)
    ].sort_values("member")
    direct_values = rows["slope"].to_numpy()
    actual_values = rows["actual_slope_0p5h"].to_numpy()
    ax_b.bar(
        i - bar_width / 2,
        np.mean(direct_values),
        width=bar_width,
        color="#D8D8D8",
        edgecolor=METHOD_COLORS[method],
        linewidth=0.8,
        hatch="///",
        label="Fixed atmosphere" if i == 0 else "_nolegend_",
    )
    ax_b.bar(
        i + bar_width / 2,
        np.mean(actual_values),
        width=bar_width,
        color=METHOD_COLORS[method],
        edgecolor="white",
        linewidth=0.5,
        label="WRF at 0.5 h" if i == 0 else "_nolegend_",
    )
    jitter = rng.uniform(-0.055, 0.055, len(rows))
    ax_b.scatter(
        i - bar_width / 2 + jitter,
        direct_values,
        s=8,
        facecolor="white",
        edgecolor=METHOD_COLORS[method],
        linewidth=0.55,
        zorder=4,
    )
    ax_b.scatter(
        i + bar_width / 2 + jitter,
        actual_values,
        s=8,
        facecolor=METHOD_COLORS[method],
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
    )
    ratio = np.mean(actual_values / direct_values)
    ax_b.text(
        i + bar_width / 2,
        max(np.mean(actual_values), np.max(actual_values)) + 4.0,
        f"×{ratio:.2f}",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=METHOD_COLORS[method],
    )
ax_b.set_xticks(x_b, [spec[2] for spec in flux_specs])
ax_b.set_ylabel(r"Flux sensitivity (W m$^{-2}$ K$^{-1}$)")
ax_b.set_ylim(0.0, 175.0)
ax_b.set_title("OM alone explains the main flux response", loc="left", pad=4)
ax_b.legend(loc="upper left", handlelength=1.5)
panel_label(ax_b, "b")

# c — complete 30-min pathway with member-bootstrap intervals.
chain = (
    ("dom0_to_dhfx", r"OM$_0$→HFX"),
    ("dom0_to_dlh", r"OM$_0$→LH"),
    ("dhfx_to_dt2", "HFX→T2"),
    ("dlh_to_dq2", "LH→Q2"),
    ("dt2_to_dtheta_l1", r"T2→$\theta_1$"),
    ("dq2_to_dqv_l1", r"Q2→q$_{v,1}$"),
)
x_c = np.arange(len(chain), dtype=float)
for offset, method, marker in zip((-0.08, 0.08), METHODS, ("o", "s")):
    selected = links[
        (links["method"] == method)
        & (links["time_hour"] == 0.5)
        & (links["region"] == REGION)
    ].set_index("link")
    values = np.array([selected.loc[key, "mean_corr"] for key, _ in chain])
    low = np.array([selected.loc[key, "mean_corr_ci025"] for key, _ in chain])
    high = np.array([selected.loc[key, "mean_corr_ci975"] for key, _ in chain])
    ax_c.errorbar(
        x_c + offset,
        values,
        yerr=np.vstack((values - low, high - values)),
        fmt=marker,
        ms=4.0,
        color=METHOD_COLORS[method],
        mfc=METHOD_COLORS[method],
        mec="white",
        mew=0.4,
        capsize=2.0,
        elinewidth=0.8,
        label=METHOD_LABELS[method],
    )
ax_c.axhline(0.8, color="#A8A8A8", lw=0.7, ls=(0, (2, 2)), zorder=0)
ax_c.set_xticks(x_c, [label for _, label in chain], rotation=28, ha="right")
ax_c.set_ylabel("Spatial correlation")
ax_c.set_ylim(0.65, 1.015)
ax_c.set_title("The full lower-boundary pathway is coherent at 0.5 h", loc="left", pad=4)
ax_c.legend(loc="lower left", ncol=2, handletextpad=0.4, columnspacing=0.8)
ax_c.text(
    0.98,
    0.04,
    "95% member-bootstrap CI",
    transform=ax_c.transAxes,
    ha="right",
    fontsize=6.1,
    color="#4D4D4D",
)
panel_label(ax_c, "c")

# d — time evolution of the interface and air-side links.
for method, marker in zip(METHODS, ("o", "s")):
    method_rows = links[
        (links["method"] == method)
        & (links["region"] == REGION)
        & (links["time_hour"].isin(TIMES))
    ]
    flux_corr = []
    air_corr = []
    for time_hour in TIMES:
        at_time = method_rows[method_rows["time_hour"] == time_hour].set_index("link")
        flux_corr.append(np.mean([at_time.loc["dom0_to_dhfx", "mean_corr"], at_time.loc["dom0_to_dlh", "mean_corr"]]))
        air_corr.append(np.mean([at_time.loc["dhfx_to_dt2", "mean_corr"], at_time.loc["dlh_to_dq2", "mean_corr"]]))
    ax_d.plot(
        TIMES,
        flux_corr,
        color=METHOD_COLORS[method],
        marker=marker,
        ms=3.5,
        label=f"{METHOD_LABELS[method]}: OM→flux",
    )
    ax_d.plot(
        TIMES,
        air_corr,
        color=METHOD_COLORS[method],
        marker=marker,
        ms=3.5,
        ls="--",
        alpha=0.85,
        label=f"{METHOD_LABELS[method]}: flux→2 m",
    )
ax_d.axhline(0.0, color="#767676", lw=0.7)
ax_d.axvspan(0.5, 2.0, color="#DDF3DE", alpha=0.55, zorder=0)
ax_d.set_xticks(TIMES)
ax_d.set_xlabel("Forecast time (h)")
ax_d.set_ylabel("Mean spatial correlation")
ax_d.set_ylim(-1.0, 1.03)
ax_d.set_title("Atmospheric feedback dominates after ~2–3 h", loc="left", pad=4)
method_handles = [
    Line2D([0], [0], color=METHOD_COLORS[method], marker=marker, ms=3.5, label=METHOD_LABELS[method])
    for method, marker in zip(METHODS, ("o", "s"))
]
stage_handles = [
    Line2D([0], [0], color="#4D4D4D", ls="-", label="OM→flux"),
    Line2D([0], [0], color="#4D4D4D", ls="--", label="flux→2 m"),
]
method_legend = ax_d.legend(
    handles=method_handles,
    loc="lower left",
    ncol=2,
    columnspacing=0.7,
    handlelength=1.4,
    fontsize=5.8,
)
ax_d.add_artist(method_legend)
ax_d.legend(
    handles=stage_handles,
    loc="lower right",
    ncol=2,
    columnspacing=0.7,
    handlelength=1.7,
    fontsize=5.8,
)
panel_label(ax_d, "d")

# e — vertical penetration during the early response window.
vsel = vertical[
    (vertical["annulus"] == VERTICAL_ANNULUS)
    & (vertical["source"] == "dom0")
    & (vertical["time_hour"].isin(EARLY_TIMES))
]
matrices = []
for response in ("dtheta", "dqv"):
    matrix = np.full((10, len(EARLY_TIMES)), np.nan)
    for ilevel in range(1, 11):
        for itime, time_hour in enumerate(EARLY_TIMES):
            rows = vsel[
                (vsel["response"] == response)
                & (vsel["level"] == ilevel)
                & (vsel["time_hour"] == time_hour)
            ]
            matrix[ilevel - 1, itime] = rows["mean_corr"].mean()
    matrices.append(matrix)
heatmap = np.concatenate((matrices[0], np.full((10, 1), np.nan), matrices[1]), axis=1)
cmap = mpl.colormaps["RdBu_r"].copy()
cmap.set_bad("white")
im = ax_e.imshow(heatmap[::-1], cmap=cmap, vmin=-0.2, vmax=0.85, aspect="auto", interpolation="nearest")
ax_e.axvline(4.0, color="white", lw=3.0)
ax_e.set_yticks(np.arange(10), [f"{height:.0f}" for height in LEVEL_HEIGHTS_M[::-1]])
ax_e.set_ylabel("Height AGL (m)")
x_labels = [
    "$\\theta$\n0.5",
    "$\\theta$\n1",
    "$\\theta$\n1.5",
    "$\\theta$\n2",
    "",
    "q$_v$\n0.5",
    "q$_v$\n1",
    "q$_v$\n1.5",
    "q$_v$\n2",
]
ax_e.set_xticks(np.arange(9), x_labels)
ax_e.set_xlabel("Variable and forecast time (h)")
ax_e.set_title("The early signal penetrates the lower 0.5–0.7 km", loc="left", pad=4)
cbar = fig.colorbar(im, ax=ax_e, location="right", fraction=0.045, pad=0.025)
cbar.set_label(r"Correlation $r$", fontsize=6.8)
cbar.ax.tick_params(labelsize=6.0, width=0.6)
panel_label(ax_e, "e")

# f — absence of a coherent downstream convective/radiative response.
downstream = (
    ("dom0_to_dw_l5", "W, level 5", "#767676", "o"),
    ("dom0_to_dolr", "OLR", "#9A4D8E", "s"),
    ("dom0_to_drain", "Accumulated rain", "#D28E2C", "^"),
)
all_downstream_values = []
for link_name, label, color, marker in downstream:
    means = []
    lows = []
    highs = []
    for time_hour in TIMES:
        values = links[
            (links["region"] == REGION)
            & (links["time_hour"] == time_hour)
            & (links["link"] == link_name)
        ]["mean_corr"].to_numpy()
        means.append(np.mean(values))
        lows.append(np.min(values))
        highs.append(np.max(values))
    means = np.asarray(means)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    all_downstream_values.extend(lows.tolist() + highs.tolist())
    ax_f.fill_between(TIMES, lows, highs, color=color, alpha=0.12, linewidth=0)
    ax_f.plot(TIMES, means, color=color, marker=marker, ms=3.3, label=label)
ax_f.axhline(0.0, color="#4D4D4D", lw=0.7)
y_limit = max(0.16, float(np.max(np.abs(all_downstream_values))) + 0.035)
ax_f.set_ylim(-y_limit, y_limit)
ax_f.set_xticks(TIMES)
ax_f.set_xlabel("Forecast time (h)")
ax_f.set_ylabel(r"Correlation with $\Delta$OM$_0$")
ax_f.set_title("No robust deep-convective or radiative pathway", loc="left", pad=4)
ax_f.legend(loc="upper left", ncol=3, columnspacing=0.7, handlelength=1.5)
ax_f.text(
    0.98,
    0.04,
    "line: method mean; shading: method range",
    transform=ax_f.transAxes,
    ha="right",
    fontsize=6.0,
    color="#4D4D4D",
)
panel_label(ax_f, "f")

fig.suptitle(
    "Small ocean increments drive a coherent but short-lived boundary-layer response",
    fontsize=10.5,
    fontweight="bold",
    y=1.015,
)
fig.text(
    0.5,
    -0.012,
    "Strong-minus-weak pairs; 0–150 km ocean blocks (~15 km), unless noted. "
    "n=6 paired members per method; panel e uses 75–150 km and averages method-level member statistics.",
    ha="center",
    va="top",
    fontsize=6.2,
    color="#4D4D4D",
)

save_individual_panels(
    fig,
    {
        "a_member_cancellation": [ax_a],
        "b_flux_sensitivity": [ax_b],
        "c_lower_boundary_pathway": [ax_c],
        "d_time_evolution": [ax_d],
        "e_vertical_penetration": [ax_e, cbar.ax],
        "f_downstream_response": [ax_f],
    },
)

kwargs = {}
if EXPORT_FORMAT == "png":
    kwargs["dpi"] = PNG_DPI
fig.savefig(
    OUTPUT_BASE.with_suffix(f".{EXPORT_FORMAT}"),
    bbox_inches="tight",
    facecolor="white",
    **kwargs,
)
plt.close(fig)
print(OUTPUT_BASE.with_suffix(f".{EXPORT_FORMAT}"))
print(PANEL_OUTPUT_DIR)
