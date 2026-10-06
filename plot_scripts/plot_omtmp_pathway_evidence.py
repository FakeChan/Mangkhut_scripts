"""Create a six-panel evidence figure for the OM_TMP-atmosphere pathway."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from omtmp_evidence_cache import (
    FORECAST_BASE_DIR as DEFAULT_FORECAST_BASE_DIR,
    METHODS as DEFAULT_METHODS,
    NR_DIR as DEFAULT_NR_DIR,
    STRONG_EXPERIMENT as DEFAULT_STRONG_EXPERIMENT,
    WEAK_EXPERIMENT as DEFAULT_WEAK_EXPERIMENT,
    ensure_pathway_cache,
    member_sort_key,
    paired_case_count,
)


# =====================
# User configuration
# =====================
SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_PROJECT_DIR = SCRIPT_DIR.parents[1] if SCRIPT_DIR.parent.name == "Mangkhut_scripts" else SCRIPT_DIR.parent
DEFAULT_CACHE_DIR = LOCAL_PROJECT_DIR / "tmp" / "omtmp_pathway_cache"
if not DEFAULT_CACHE_DIR.exists():
    DEFAULT_CACHE_DIR = SCRIPT_DIR / "omtmp_pathway_cache"
CACHE_DIR = Path(os.environ.get("OMTMP_CACHE_DIR", DEFAULT_CACHE_DIR))
OUTPUT_BASE = SCRIPT_DIR / "figs" / "omtmp_pathway_evidence"
PANEL_OUTPUT_DIR = OUTPUT_BASE.parent / f"{OUTPUT_BASE.name}_panels"
OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)  # ensure figs/ exists before savefig
EXPORT_FORMAT = "png"  # single export format for the full figure and panels: "png", "svg", or "pdf"
assert EXPORT_FORMAT in ("png", "svg", "pdf"), f"unsupported EXPORT_FORMAT: {EXPORT_FORMAT}"
FIGURE_SIZE_INCH = (7.2, 8.1)
PNG_DPI = 400
REGION = "0-150"
VERTICAL_ANNULUS = 1  # 75–150 km, where the boundary-layer signal is clearest.
FORECAST_BASE_DIR = DEFAULT_FORECAST_BASE_DIR
NR_DIR = DEFAULT_NR_DIR
STRONG_EXPERIMENT = DEFAULT_STRONG_EXPERIMENT
WEAK_EXPERIMENT = DEFAULT_WEAK_EXPERIMENT
METHODS = DEFAULT_METHODS
METHOD_LABELS = {"EAKF": "EAKF", "QCF_RHF": "QCF-RHF"}
METHOD_COLORS = {"EAKF": "#42949E", "QCF_RHF": "#0F4D92"}
TIMES = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
EARLY_TIMES = np.array([0.5, 1.0, 1.5, 2.0])
LEVEL_HEIGHTS_M = np.array([13.0, 47.6, 99.6, 160.7, 230.8, 310.1, 394.3, 492.6, 605.3, 732.7])
MAX_MEMBERS_PER_METHOD = None
CACHE_POLICY = "auto"  # "auto" builds missing cache; "refresh" rebuilds; "reuse" requires existing cache.


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



# =============================================================================
# 新增图配置(基于 2026-10-06 最新诊断 CSV:20261006_verify_diag_rerun_review)
# =============================================================================
RUN_LEGACY_SIX_PANEL = False   # 原六联图:需先另行刷新 2026-08-03 旧缓存,默认关闭
RUN_UPDATED_EVIDENCE = True    # 三张新图(图 1/2/3);图 4 见 RUN_OPTIONAL_FIG4
RUN_OPTIONAL_FIG4 = False      # 可选图 4:TSK×通量改善分类比例(默认关闭)

REVIEW_DIR = LOCAL_PROJECT_DIR / "diagnostic_reports" / "20261006_verify_diag_rerun_review"
LATEST_OUTPUTS = REVIEW_DIR / "server_snapshot" / "outputs"
ANALYSIS_D03 = REVIEW_DIR / "analysis_d03"
MEMBER_METRICS_CSV = LATEST_OUTPUTS / "REAL_verify01_skill_timeseries" / "verify01_member_metrics.csv"
CONDITIONAL_BUDGET_CSV = LATEST_OUTPUTS / "REAL_verify02_flux_error_budget" / "verify02_conditional_budget.csv"
FOCUS_QUALITY_CSV = ANALYSIS_D03 / "11_report_focus_quality.csv"
FOCUS_METRICS_CSV = ANALYSIS_D03 / "12_report_focus_metrics.csv"

UPDATED_FIG_DIR = SCRIPT_DIR / "figs" / "verify_diag_updated_evidence"
UPDATED_PANEL_DIR = SCRIPT_DIR / "figs" / "verify_diag_updated_evidence_panels"
UPDATED_SOURCE_DIR = SCRIPT_DIR / "figs" / "verify_diag_updated_evidence_source_data"
UPDATED_EXPORT_FORMAT = "png"  # "png" / "svg" / "pdf"
UPDATED_PNG_DPI = 400
UPDATED_FIG_SIZE_INCH = (7.2, 8.1)

# 新图空间口径与窗口(NR 中心、0-300 km 海洋逐点;与原六联图的
# 配对中心、0-150 km 块平均不是同一口径,两者不可直接混合平均)
EVIDENCE_REGION = "r000_300"
EVIDENCE_WINDOWS = (("early_0p5_2h", 0.5, 2.0), ("late_3_6h", 3.0, 6.0))
EVIDENCE_VARIABLES = ("om", "tsk", "hfx", "lh")
VARIABLE_LABELS = {"om": "OM_TMP (surface)", "tsk": "TSK",
                   "hfx": "HFX (sensible heat)", "lh": "LH (latent heat)"}

# =============================================================================
# 原六联图(2026-08-03 旧 pathway 缓存口径;配对中心、0-150 km 块平均)。
# 默认不运行:旧缓存需另行刷新,且不得与本批新图(最新诊断 CSV、
# NR 中心、海洋逐点统计)作为同一批次汇报。开启前请先确认缓存版本。
# =============================================================================
def build_legacy_six_panel():
    """生成原六联图(依赖 2026-08-03 旧 pathway 缓存,另行刷新后使用)。"""
    ensure_pathway_cache(
        cache_dir=CACHE_DIR,
        forecast_base_dir=FORECAST_BASE_DIR,
        nr_dir=NR_DIR,
        strong_experiment=STRONG_EXPERIMENT,
        weak_experiment=WEAK_EXPERIMENT,
        methods=METHODS,
        max_members_per_method=MAX_MEMBERS_PER_METHOD,
        cache_policy=CACHE_POLICY,
    )

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
    members_by_method = {
        method: tuple(sorted(surface.loc[surface["method"] == method, "member"].unique(), key=member_sort_key))
        for method in METHODS
    }
    missing_methods = [method for method, members in members_by_method.items() if not members]
    if missing_methods:
        raise ValueError(f"No pathway cache rows for methods: {missing_methods}")
    member_order = tuple(sorted(set(surface["member"]), key=member_sort_key))
    n_cases = paired_case_count(members_by_method)
    assert set(surface["method"]) == set(METHODS)
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
    x = np.arange(len(member_order), dtype=float)
    width = 0.34
    for offset, method in zip((-width / 2, width / 2), METHODS):
        values = (
            source_member[source_member["method"] == method]
            .set_index("member")
            .reindex(member_order)["dom0"]
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
        ensemble_mean = float(np.nanmean(values))
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
        "paired members shown individually\nmethod means shown dashed",
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
        f"n={n_cases} paired method/member cases; panel e uses 75–150 km and averages method-level member statistics.",
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


# =============================================================================
# 新增图(图 1/2/3 与可选图 4):基于 2026-10-06 最新诊断 CSV
# (diagnostic_reports/20261006_verify_diag_rerun_review),独立于旧缓存,
# 不调用 ensure_pathway_cache 或任何提取入口。
# 空间口径:NR 中心、0-300 km 海洋逐点(与原六联图的配对中心、
# 0-150 km 块平均不是同一口径,两者不可直接混合平均)。
# =============================================================================

import plot_evidence_updated_lib as lib

VAR_ORDER = ("om", "tsk", "hfx", "lh")


def _export_source_csv(frame, name):
    UPDATED_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = UPDATED_SOURCE_DIR / name
    frame.to_csv(path, index=False, float_format="%.12g")
    print(f"[updated-evidence] source data: {path}")


def load_latest_diagnostics():
    """读取 2026-10-06 重跑的最新诊断 CSV(不触发任何缓存/提取)。"""
    member_metrics = pd.read_csv(MEMBER_METRICS_CSV, dtype={"member": str})
    conditional = pd.read_csv(CONDITIONAL_BUDGET_CSV, dtype={"member": str})
    focus_quality = pd.read_csv(FOCUS_QUALITY_CSV)
    focus_metrics = pd.read_csv(FOCUS_METRICS_CSV)
    return member_metrics, conditional, focus_quality, focus_metrics


def fig1_rmse_improvement(member_metrics):
    """图 1:四变量相对 RMSE 改善时序(0-300 km,0-6 h 全部半小时)。

    误差参照 NR;细线=成员,粗线=方法平均(成员等权);正值=改善。
    """
    ts = lib.skill_member_timeseries(member_metrics, region=EVIDENCE_REGION)
    summary = lib.skill_summary_by_window(
        member_metrics, region=EVIDENCE_REGION, windows=lib.WINDOWS
    )
    fig, axes = plt.subplots(2, 2, figsize=UPDATED_FIG_SIZE_INCH, sharex=True,
                             constrained_layout=True)
    axes = axes.ravel()
    for ax, var in zip(axes, VAR_ORDER):
        for method in METHODS:
            color = METHOD_COLORS[method]
            member_ts = ts[(ts.method == method) & (ts.variable == var)]
            method_mean = (
                member_ts.groupby("time_hour", as_index=False)
                .rmse_improvement_pct.mean()
            )
            for member, g in member_ts.groupby("member"):
                ax.plot(g.time_hour, g.rmse_improvement_pct, color=color,
                        lw=0.7, alpha=0.45, label="_nolegend_")
            ax.plot(method_mean.time_hour, method_mean.rmse_improvement_pct,
                    color=color, lw=1.8, marker="o", ms=3,
                    label=METHOD_LABELS[method])
        ax.axhline(0.0, color="#4D4D4D", lw=0.7)
        ax.axvspan(0.5, 2.0, color="#DDF3DE", alpha=0.5, zorder=0)
        ax.axvspan(3.0, 6.0, color="#FDE8E8", alpha=0.45, zorder=0)
        ax.set_title(VARIABLE_LABELS[var], loc="left", fontsize=9)
        if var in ("om", "tsk"):
            ax.set_ylim(-5, 45)
        else:
            ax.set_ylim(-2.0, 2.0)
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=7, frameon=False)
    axes[2].set_ylabel("paired RMSE improvement vs NR (%)")
    axes[2].set_xlabel("forecast time (h)")
    axes[3].set_xlabel("forecast time (h)")
    fig.suptitle("Ocean/TSK errors improved; flux errors did not "
                 "(0-300 km ocean, every 0.5 h, 0-6 h)", fontsize=10.5,
                 fontweight="bold")
    fig.text(0.5, -0.01,
             "Error reference: NR. Thin lines: six paired members; thick: "
             "member-equal mean. Members and consecutive times are not "
             "independent typhoon samples.",
             ha="center", fontsize=6.5, color="#4D4D4D")
    return fig, summary


def _budget_groups(budget, flux, window):
    """图 2 单面板的分组(方法 × 子集),顺序固定。"""
    groups = []
    for method in METHODS:
        for subset, tag in (("all_common", "all"), ("sst_improved", "TSK-imp")):
            row = budget[(budget.window == window)
                         & (budget.method == method)
                         & (budget.variable == flux)
                         & (budget.subset == subset)]
            if len(row):
                c = float(row.cross_term.iloc[0])
                s = float(row.increment_square_term.iloc[0])
                dm = float(row.delta_mse_direct.iloc[0])
                n_members = int(row.n_members.iloc[0])
                period = "early" if window.startswith("early") else "late"
                label = f"{METHOD_LABELS[method]}\n{tag}\n{period}"
                groups.append({"method": method, "subset": subset,
                               "window": window, "label": label,
                               "c": c, "s": s, "dm": dm,
                               "n_members": n_members})
            else:
                groups.append({"method": method, "subset": subset,
                               "window": window,
                               "label": f"{METHOD_LABELS[method]}\n{tag}\n{period}",
                               "c": np.nan, "s": np.nan, "dm": np.nan,
                               "n_members": 0})
    return groups


def fig2_budget_windows(conditional):
    """图 2:实际通量误差预算 ΔMSE=C+S,早期 vs 后期,全域 vs TSK 改善子集。

    先成员内对时次等权平均,再跨成员等权;正 ΔMSE=误差增加。
    """
    budget = lib.budget_summary_by_window(
        conditional, region=EVIDENCE_REGION,
        subsets=("all_common", "sst_improved"),
        windows=lib.WINDOWS,
    )
    budget = budget[budget.variable.isin(("hfx", "lh"))]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), constrained_layout=True)
    windows_order = ("early_0p5_2h", "late_3_6h")
    for ax_row, flux in zip(axes, ("hfx", "lh")):
        for ax_col, window in zip(ax_row, windows_order):
            ax.clear()
            groups = _budget_groups(budget, flux, window)
            x = np.arange(len(groups))
            width = 0.25
            for offset, (metric, label, hatch) in enumerate((
                    ("c", "C = 2<e·δF>", ""),
                    ("s", "S = <δF²>", ""),
                    ("dm", "ΔMSE = C+S", "//"))):
                vals = [g[metric] for g in groups]
                colors = [METHOD_COLORS[g["method"]] for g in groups]
                ax.bar(x + (offset - 1) * width, vals, width, label=label,
                       color=colors, alpha=0.75 if metric != "dm" else 1.0,
                       hatch=hatch if metric == "dm" else "",
                       edgecolor="white")
            ax.axhline(0.0, color="#4D4D4D", lw=0.8)
            ax.set_xticks(x, [g["label"] for g in groups], fontsize=6,
                          rotation=30, ha="right")
            ax.set_title(f"{flux.upper()}  (W m$^{{-2}})^2$", loc="left",
                         fontsize=9)
            ax.grid(alpha=0.25, axis="y")
            ax.annotate(
                "n_members: "
                + ", ".join(f"{g['method']}-{g['subset'].replace('_', '-')}="
                            f"{g['n_members']}" for g in groups),
                (0.5, -0.42), xycoords="axes fraction", ha="center",
                fontsize=5.5, color="#4D4D4D")
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.suptitle("Actual flux error budget ΔMSE = C + S: early vs late, "
                 "domain vs TSK-improved subset", fontsize=10.5,
                 fontweight="bold")
    fig.text(0.5, -0.02,
             "Member-window means, then equal-weight across members; "
             "(W m$^{-2})^2$. Early loss is direction-dominated (C>0); "
             "late C turns negative but S offsets it.",
             ha="center", fontsize=6.5, color="#4D4D4D")
    return fig, budget


def fig3_sst_response_lh(focus_quality, focus_metrics):
    """图 3:LH 海温替换响应与质量覆盖(0-300 km TSK 改善子集,早 vs 后)。

    A 面板:联合/保守交集通过案例数;B 面板:SST/实际响应 RMS 比;
    C 面板:C_SST。0 h 实际响应为零,不纳入(11/12 表只含早晚窗口)。
    """
    cov = lib.focus_coverage(focus_quality, subset="sst_improved")
    cov = cov[cov.variable == "lh"]
    met = lib.focus_metrics_by_gate(
        focus_metrics,
        gates=(lib.GATE_JOINT, lib.GATE_CONSERVATIVE),
        variables=("lh",),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), constrained_layout=True)
    x_labels = ["EAKF\nearly", "EAKF\nlate", "QCF-RHF\nearly", "QCF-RHF\nlate"]
    x = np.arange(4)

    ax = axes[0]
    joint = cov.n_pass.to_numpy()
    cons = cov.n_joint_and_regionA_pass.to_numpy()
    ax.bar(x - 0.2, joint, 0.4, color="#9EC9E2", label="joint screen")
    ax.bar(x + 0.2, cons, 0.4, color="#2C7FB8", label="conservative")
    for xi, (jv, cv) in enumerate(zip(joint, cons)):
        ax.text(xi - 0.2, jv + 0.3, str(int(jv)), ha="center", fontsize=6)
        ax.text(xi + 0.2, cv + 0.3, str(int(cv)), ha="center", fontsize=6)
    ax.set_xticks(x, x_labels, fontsize=7)
    ax.set_ylabel("member-time cases (of 24)")
    ax.set_title("A  Quality-screened cases (LH)", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    met_j = met[met.gate == lib.GATE_JOINT]
    met_c = met[met.gate == lib.GATE_CONSERVATIVE]
    ratio_j = met_j.mean_member_sst_to_actual_rms_ratio.to_numpy()
    ratio_c = met_c.mean_member_sst_to_actual_rms_ratio.to_numpy()
    ax.bar(x - 0.2, ratio_j, 0.4, color="#9EC9E2", label="joint screen")
    ax.bar(x + 0.2, ratio_c, 0.4, color="#2C7FB8", label="conservative")
    ax.axhline(1.0, color="#B2182B", lw=0.8, ls="--")
    ax.text(3.35, 1.04, "ratio = 1", fontsize=6, color="#B2182B")
    ax.set_xticks(x, x_labels, fontsize=7)
    ax.set_ylabel("RMS(δF_SST) / RMS(δF_actual)")
    ax.set_title("B  SST response vs actual response", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    csst_j = met_j.mean_member_cross_model_error_sst_response.to_numpy()
    csst_c = met_c.mean_member_cross_model_error_sst_response.to_numpy()
    ax.bar(x - 0.2, csst_j, 0.4, color="#9EC9E2", label="joint screen")
    ax.bar(x + 0.2, csst_c, 0.4, color="#2C7FB8", label="conservative")
    ax.axhline(0.0, color="#4D4D4D", lw=0.8)
    ax.set_xticks(x, x_labels, fontsize=7)
    ax.set_ylabel("C_SST = 2<e·δF_SST>  ((W m$^{-2})^2$)")
    ax.set_title("C  C_SST vs actual error (LH)", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Offline SST-substitution response vs actual LH change "
                 "(0-300 km, TSK-improved subset)", fontsize=10,
                 fontweight="bold")
    fig.text(0.5, -0.03,
             "δF_SST = f(TSK_strong, weak inputs) - f(TSK_weak, weak inputs); "
             "TSK substituted, not OM_TMP. C_SST is diagnostic, not a causal "
             "fraction. 0 h has zero actual response (excluded).",
             ha="center", fontsize=6.2, color="#4D4D4D")
    return fig, {"coverage": cov, "metrics_conservative": met_c,
                 "metrics_joint": met_j}


def fig4_classification(classification_fractions):
    """可选图 4:TSK×通量改善四类比例(早晚 × 方法 × 通量)。

    比例归一化到「TSK 与通量双方均有变化」的格点(分母明确标注);
    单侧未变化与双侧未变化不计入分母,其覆盖比例另行标注。
    """
    df = classification_fractions.copy()
    df = df[(df.region == EVIDENCE_REGION) & df.status.eq("ok")]
    df = df[df.pair.isin(("tsk-hfx", "tsk-lh"))]
    df = lib.add_window_column(df)
    df = df[df.window.notna()]
    rows = []
    for (window, method, pair, variable), g in df.groupby(
        ["window", "method", "pair", "variable"], sort=True
    ):
        denom = float(
            g.frac_sst_improve_flux_improve.sum()
            + g.frac_sst_improve_flux_worsen.sum()
            + g.frac_sst_worsen_flux_improve.sum()
            + g.frac_sst_worsen_flux_worsen.sum()
        )
        rows.append({
            "window": window, "method": method, "pair": pair,
            "frac_improve_improve": (
                float(g.frac_sst_improve_flux_improve.mean()) / denom
                if denom else np.nan),
            "frac_improve_worsen": (
                float(g.frac_sst_improve_flux_worsen.mean()) / denom
                if denom else np.nan),
            "frac_worsen_improve": (
                float(g.frac_sst_worsen_flux_improve.mean()) / denom
                if denom else np.nan),
            "frac_worsen_worsen": (
                float(g.frac_sst_worsen_flux_worsen.mean()) / denom
                if denom else np.nan),
            "both_changed_fraction_of_grid": (
                denom / float(g.n_valid.mean()) if float(g.n_valid.mean())
                else np.nan),
        })
    summary = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    cat_labels = ["TSK+ F+", "TSK+ F-", "TSK- F+", "TSK- F-"]
    cat_cols = ["frac_improve_improve", "frac_improve_worsen",
                "frac_worsen_improve", "frac_worsen_worsen"]
    cat_colors = ["#1A9850", "#FDAE61", "#74ADD1", "#D73027"]
    for ax, (pair, group) in zip(axes.ravel(), summary.groupby("pair", sort=True)):
        x = np.arange(len(group))
        bottom = np.zeros(len(group))
        for col, label, color in zip(cat_cols, cat_labels, cat_colors):
            vals = np.nan_to_num(group[col].to_numpy(dtype=float))
            ax.bar(x, vals, 0.6, bottom=bottom, color=color, label=label)
            bottom = bottom + np.nan_to_num(vals)
        ax.set_xticks(x, [f"{m}\n{w.replace('_0p5_2h', ' early').replace('_3_6h', ' late')}"
                          for m, w in zip(group.method, group.window)],
                      fontsize=6.5, rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("fraction of both-changed points")
        ax.set_title(pair.replace("tsk-", "TSK-").replace("-hfx", " vs HFX")
                     .replace("-lh", " vs LH"), loc="left", fontsize=9)
        ax.grid(alpha=0.25, axis="y")
    axes[0, 0].legend(fontsize=6, ncol=2, frameon=False)
    fig.suptitle("TSK-improvement vs flux-change categories "
                 "(normalized to both-changed points)", fontsize=10,
                 fontweight="bold")
    fig.text(0.5, -0.01,
             "Single-side unchanged and both-unchanged points are excluded "
             "from the denominator; their grid fraction is shown separately.",
             ha="center", fontsize=6.3, color="#4D4D4D")
    return fig, summary


def build_updated_evidence():
    """生成三张新图(+可选图 4)与源数据汇总表(不触旧缓存/服务器)。"""
    UPDATED_FIG_DIR.mkdir(parents=True, exist_ok=True)
    UPDATED_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    UPDATED_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    member_metrics, conditional, focus_quality, focus_metrics = \
        load_latest_diagnostics()

    fmt = UPDATED_EXPORT_FORMAT
    dpi = UPDATED_PNG_DPI

    # ---- 图 1 ----
    fig1, summary1 = fig1_rmse_improvement(member_metrics)
    summary1["region"] = EVIDENCE_REGION
    summary1["aggregation"] = "member-time mean, then equal-weight members"
    summary1["error_reference"] = "NR"
    _export_source_csv(summary1, "fig1_skill_improvement_source.csv")
    fig1.savefig(UPDATED_FIG_DIR / f"fig1_skill_improvement.{fmt}",
                 dpi=dpi if fmt == "png" else None,
                 bbox_inches="tight", facecolor="white")
    save_individual_panels(fig1, {
        "u1a_om_improvement": [fig1.axes[0]],
        "u1b_tsk_improvement": [fig1.axes[1]],
        "u1c_hfx_improvement": [fig1.axes[2]],
        "u1d_lh_improvement": [fig1.axes[3]],
    })
    plt.close(fig1)

    # ---- 图 2 ----
    fig2, budget = fig2_budget_windows(conditional)
    budget["region"] = EVIDENCE_REGION
    budget["aggregation"] = "member-time mean, then equal-weight members"
    budget["note"] = "delta_mse = cross_term + increment_square_term"
    _export_source_csv(budget, "fig2_budget_windows_source.csv")
    fig2.savefig(UPDATED_FIG_DIR / f"fig2_budget_windows.{fmt}",
                 dpi=dpi if fmt == "png" else None,
                 bbox_inches="tight", facecolor="white")
    save_individual_panels(fig2, {
        "u2_hfx_early": [fig2.axes[0]],
        "u2_lh_early": [fig2.axes[1]],
        "u2_hfx_late": [fig2.axes[2]],
        "u2_lh_late": [fig2.axes[3]],
    })
    plt.close(fig2)

    # ---- 图 3 ----
    fig3, joint_and_cov = fig3_sst_response_lh(focus_quality, focus_metrics)
    joint_and_cov["region"] = EVIDENCE_REGION
    _export_source_csv(joint_and_cov, "fig3_sst_response_source.csv")
    fig3.savefig(UPDATED_FIG_DIR / f"fig3_sst_response_lh.{fmt}",
                 dpi=dpi if fmt == "png" else None,
                 bbox_inches="tight", facecolor="white")
    save_individual_panels(fig3, {
        "u3a_quality_cases": [fig3.axes[0]],
        "u3b_rms_ratio": [fig3.axes[1]],
        "u3c_csst": [fig3.axes[2]],
    })
    plt.close(fig3)

    # ---- 可选图 4(默认关闭)----
    if RUN_OPTIONAL_FIG4:
        classification = pd.read_csv(
            LATEST_OUTPUTS / "REAL_verify01_skill_timeseries" /
            "verify01_member_classification_fractions.csv",
            dtype={"member": str},
        )
        fig4, class_summary = fig4_classification(classification)
        class_summary["region"] = EVIDENCE_REGION
        class_summary["denominator"] = "points with both TSK and flux changes"
        _export_source_csv(class_summary, "fig4_classification_source.csv")
        fig4.savefig(UPDATED_FIG_DIR / f"fig4_classification.{fmt}",
                     dpi=dpi if fmt == "png" else None,
                     bbox_inches="tight", facecolor="white")
        plt.close(fig4)


def main():
    if RUN_UPDATED_EVIDENCE:
        build_updated_evidence()
    if RUN_LEGACY_SIX_PANEL:
        build_legacy_six_panel()


if __name__ == "__main__":
    main()
