#!/usr/bin/env python3
"""LACC member scatter and paired ensemble-stability diagnostics.

Run this file after editing CONFIG below. Input loading reuses the validated
lag-covariance diagnostic beside it. No assimilation or remote commands run.
The vertical regression residual of NR is a descriptive discrepancy, not a
posterior error, significance test, or direct ocean radiance sensitivity.
"""
from __future__ import annotations

import math
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

if __package__:
    from . import diag_LACC_lag_cov_innovation as lag
else:
    import diag_LACC_lag_cov_innovation as lag

# ===================== User configuration =====================
# Keep the companion diagnostic in the same directory on the server.
# Its Config also allows explicit hx_root, nr_file, state_lat/lon and member
# file patterns. The following defaults use the inspected experiment layout.
INPUTS = lag.Config(
    project_root=Path("/share/home/lililei1/kcfu/tc_mangkhut"),
    background_choice="obs_seq111",
    obs_seq111_dir=Path("/scratch/lililei1/kcfu/tc_mangkhut/4assimilation/DART/EAKF/obs_seq111"),
    center_time="2018-09-10_00:00:00",
    member_start=1,
    member_end=50,
    lag_hours=(0, 3, 6),
    output_dir=Path(__file__).resolve().parent / "figs" / "LACC_member_scatter",
    write_figures=False,  # figures are produced by this script, not lag.run()
)


@dataclass(frozen=True)
class ScatterConfig:
    inputs: lag.Config = field(default_factory=lag.Config)
    bootstrap_repeats: int = 2000
    subset_repeats: int = 500
    subset_fraction: float = 0.5
    random_seed: int = 20261010
    confidence_level: float = 0.95
    figure_formats: tuple[str, ...] = ("png", "pdf")
    figure_dpi: int = 200
    figure_width_in: float = 10.5
    panel_height_in: float = 4.0
    label_influential_members: int = 3
    strict_external_fo: bool = True


CONFIG = ScatterConfig(inputs=INPUTS)
# =================== End user configuration ===================


@dataclass(frozen=True)
class Panel:
    label: str
    lag_hours: tuple[int, ...]
    hx: np.ndarray
    ocean: np.ndarray
    nr_hx: float
    nr_ocean: float
    clear_sky_flag: str


def validate_config(config: ScatterConfig) -> None:
    lag.validate_config(config.inputs)
    if len(config.inputs.member_numbers()) < 3:
        raise ValueError("scatter stability diagnostics require at least 3 members")
    for name in ("bootstrap_repeats", "subset_repeats", "figure_dpi"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(config.random_seed, int) or isinstance(config.random_seed, bool) or config.random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    if not 0 < config.subset_fraction < 1:
        raise ValueError("subset_fraction must be between 0 and 1")
    if not 0 < config.confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if any(fmt not in {"png", "pdf", "svg"} for fmt in config.figure_formats):
        raise ValueError("figure_formats may contain png, pdf or svg only")
    for value in (config.figure_width_in, config.panel_height_in):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("figure dimensions must be finite and positive")
    if (not isinstance(config.label_influential_members, int)
            or isinstance(config.label_influential_members, bool) or config.label_influential_members < 0):
        raise ValueError("label_influential_members must be a nonnegative integer")


def pair_statistics(hx, ocean, nr_hx: float, nr_ocean: float) -> dict:
    """Regress ocean (vertical axis) ON Hx (horizontal axis), across members."""
    x, y = np.asarray(hx, dtype=float), np.asarray(ocean, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError("Hx and ocean must be matching one-dimensional member vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("member vectors must be finite; silently dropping members breaks pairing")
    if not math.isfinite(nr_hx) or not math.isfinite(nr_ocean):
        raise ValueError("NR coordinates must be finite")
    moments = lag.pair_moments(y, x)
    # Averaging identical non-binary Kelvin decimals can leave a one-ULP
    # residual, producing false spread and even r=1. Use exact element
    # equality for constants; do not erase genuinely small ensemble spread.
    constant_x, constant_y = bool(np.all(x == x[0])), bool(np.all(y == y[0]))
    if constant_x or constant_y:
        moments = replace(moments,
                          mean_x=float(y[0]) if constant_y else moments.mean_x,
                          mean_y=float(x[0]) if constant_x else moments.mean_y,
                          std_x=0.0 if constant_y else moments.std_x,
                          std_y=0.0 if constant_x else moments.std_y,
                          cov=0.0, corr=float("nan"), degenerate=True)
    variance_x = moments.std_y**2
    slope = moments.cov / variance_x if variance_x > 0 else float("nan")
    intercept = moments.mean_x - slope * moments.mean_y
    fitted = moments.mean_x + slope * (x - moments.mean_y)
    residual_std = (
        float(np.sqrt(np.sum((y-fitted)**2) / (len(x)-2)))
        if len(x) > 2 and math.isfinite(slope) else float("nan")
    )
    nr_prediction = moments.mean_x + slope * (nr_hx - moments.mean_y)
    nr_residual = nr_ocean - nr_prediction
    innovation = nr_hx - moments.mean_y
    ocean_error = nr_ocean - moments.mean_x
    product = moments.cov * innovation
    return {
        "n_members": len(x),
        "hx_mean": moments.mean_y, "hx_std": moments.std_y,
        "ocean_mean": moments.mean_x, "ocean_std": moments.std_x,
        "nr_hx": float(nr_hx), "nr_ocean": float(nr_ocean),
        "covariance": moments.cov, "correlation": moments.corr,
        "slope_ocean_on_hx": slope, "intercept": intercept,
        "regression_residual_std_K": residual_std,
        "nr_ocean_regression": nr_prediction,
        "nr_vertical_residual_K": nr_residual,
        "nr_vertical_residual_over_residual_std": (
            nr_residual / residual_std if residual_std > 1e-12 else float("nan")
        ),
        "nr_hx_outside_member_range": bool(nr_hx < np.min(x) or nr_hx > np.max(x)),
        "d_NR": innovation, "e_o": ocean_error,
        "direction_product_NR": product, "alignment_NR": product * ocean_error,
        "alignment_flag": lag.classify_alignment(product, ocean_error, moments.corr, moments.std_x, 0.0),
    }


def build_panels(context: lag.ObservationContext, series: lag.LagSeries,
                 assimilated_hours: set[int]) -> list[Panel]:
    """Full window is an average over actual assimilated lags, per member."""
    if series.hx.shape != (len(series.lag_hours), len(context.ocean_prior)):
        raise ValueError("Hx lag/member dimensions do not match the ocean ensemble")
    indices = {hour: i for i, hour in enumerate(series.lag_hours)}
    if len(indices) != len(series.lag_hours):
        raise ValueError("duplicate lag hours")
    missing = sorted(assimilated_hours - set(indices))
    if missing or not assimilated_hours:
        raise ValueError(f"full assimilated window has missing lag(s): {missing}")
    panels = [Panel(f"{hour} h", (hour,), series.hx[i], context.ocean_prior,
                    float(series.y_nr[i]), context.ocean_nr, series.clear_sky_flags[i])
              for i, hour in enumerate(series.lag_hours)]
    chosen = [indices[hour] for hour in sorted(assimilated_hours)]
    flags = [series.clear_sky_flags[i] for i in chosen]
    flag = "not_clear" if "not_clear" in flags else "unknown" if "unknown" in flags else "clear"
    panels.append(Panel("LACC [" + ",".join(map(str, sorted(assimilated_hours))) + "]",
                        tuple(sorted(assimilated_hours)), np.mean(series.hx[chosen], axis=0),
                        context.ocean_prior, float(np.mean(series.y_nr[chosen])), context.ocean_nr, flag))
    return panels


def make_resamples(n_members: int, config: ScatterConfig) -> dict[str, np.ndarray]:
    """One set of draws is reused for all panels of a given observation.

    Ocean, every individual Hx lag, and the window use identical indices.
    Bootstrap samples with replacement; subsets sample without replacement.
    """
    if n_members < 3:
        raise ValueError("at least 3 members required")
    rng = np.random.default_rng(config.random_seed)
    subset_size = min(n_members-1, max(2, math.ceil(config.subset_fraction*n_members)))
    return {
        "bootstrap": rng.integers(n_members, size=(config.bootstrap_repeats, n_members)),
        "subsets": np.stack([rng.choice(n_members, subset_size, replace=False)
                             for _ in range(config.subset_repeats)]),
    }


def sign_category(value: float) -> str:
    return "undefined" if not math.isfinite(value) else "negative" if value < 0 else "positive" if value > 0 else "zero"


def resample_summary(rows: list[dict], base_covariance: float, confidence: float) -> dict:
    # Constant paired draws have a valid covariance of zero but undefined r.
    # They MUST remain in the covariance distribution and sign denominator.
    valid = [row for row in rows if row["valid"]]
    correlation_valid = [row for row in rows if row["correlation_valid"]]
    tail = (1-confidence)/2
    covariance = [row["covariance"] for row in valid]
    correlation = [row["correlation"] for row in correlation_valid]
    base_sign = sign_category(base_covariance)
    return {
        "n_total": len(rows), "n_valid": len(valid), "n_invalid": len(rows)-len(valid),
        "n_correlation_valid": len(correlation_valid),
        "n_correlation_undefined": len(rows)-len(correlation_valid),
        "sample_size": rows[0]["sample_size"] if rows else None,
        "confidence_level": confidence,
        "covariance_percentile_interval": np.quantile(covariance, [tail, 1-tail]).tolist() if valid else None,
        "correlation_percentile_interval": np.quantile(correlation, [tail, 1-tail]).tolist() if correlation_valid else None,
        "negative_fraction_valid": sum(row["covariance"] < 0 for row in valid)/len(valid) if valid else None,
        "same_sign_fraction_valid": (
            sum(sign_category(row["covariance"]) == base_sign for row in valid)/len(valid)
            if valid and base_sign not in {"undefined", "zero"} else None
        ),
    }


def analyze_panel(panel: Panel, members: list[int], draws: dict[str, np.ndarray],
                  config: ScatterConfig) -> tuple[dict, list[dict], list[dict]]:
    if (len(members) != len(panel.hx) or len(set(members)) != len(members)
            or any(not isinstance(m, (int, np.integer)) or m < 1 for m in members)):
        raise ValueError("member IDs must be unique positive integers matching the vectors")
    if len(members) < 3:
        raise ValueError("at least 3 members required for leave-one-out")
    result = pair_statistics(panel.hx, panel.ocean, panel.nr_hx, panel.nr_ocean)
    result.update(panel_label=panel.label, lag_hours=list(panel.lag_hours), clear_sky_flag=panel.clear_sky_flag)
    loo = []
    for i, member in enumerate(members):
        retained = np.arange(len(members)) != i
        reduced = pair_statistics(panel.hx[retained], panel.ocean[retained], panel.nr_hx, panel.nr_ocean)
        cov = reduced["covariance"]
        loo.append({
            "panel_label": panel.label, "omitted_member": int(member),
            "covariance_without_member": cov,
            "correlation_without_member": reduced["correlation"],
            "covariance_change": cov-result["covariance"],
            "sign_flip": bool(cov*result["covariance"] < 0),
            "sign_zero": bool(cov == 0 and result["covariance"] != 0),
        })
    influential = sorted(loo, key=lambda row: abs(row["covariance_change"]), reverse=True)
    result["leave_one_out"] = {
        "n_sign_flips": sum(row["sign_flip"] for row in loo),
        "sign_flip_members": [row["omitted_member"] for row in loo if row["sign_flip"]],
        "n_sign_zero": sum(row["sign_zero"] for row in loo),
        "sign_zero_members": [row["omitted_member"] for row in loo if row["sign_zero"]],
        "covariance_range": [min(row["covariance_without_member"] for row in loo),
                             max(row["covariance_without_member"] for row in loo)],
        "most_influential_members": [row["omitted_member"] for row in influential[:config.label_influential_members]],
    }
    samples = []
    for method, indices in draws.items():
        if (indices.ndim != 2 or indices.shape[1] < 2 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(members))):
            raise ValueError("invalid resampling member indices")
        rows = []
        for repetition, chosen in enumerate(indices):
            stat = pair_statistics(panel.hx[chosen], panel.ocean[chosen], panel.nr_hx, panel.nr_ocean)
            row = {
                "panel_label": panel.label, "method": method, "repetition": repetition,
                "sample_size": len(chosen), "member_indices": ",".join(map(str, chosen)),
                "member_numbers": ",".join(str(members[i]) for i in chosen),
                "covariance": stat["covariance"], "correlation": stat["correlation"],
                "d_NR": stat["d_NR"], "e_o": stat["e_o"],
                "alignment_NR": stat["alignment_NR"],
                "valid": math.isfinite(stat["covariance"]),
                "correlation_valid": math.isfinite(stat["correlation"]),
            }
            rows.append(row)
        result[method] = resample_summary(rows, result["covariance"], config.confidence_level)
        samples.extend(rows)
    return result, loo, samples


def make_figure(context: lag.ObservationContext, panels: list[Panel], results: list[dict],
                members: list[int], config: ScatterConfig) -> dict[str, str]:
    """Draw member-range regression lines; NR outside that range is explicit."""
    plt = lag.plt
    from matplotlib.ticker import MaxNLocator
    nrows = math.ceil(len(panels)/2)
    style = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
             "pdf.fonttype": 42, "svg.fonttype": "none", "axes.formatter.useoffset": False,
             "figure.constrained_layout.h_pad": 0.12}
    with plt.rc_context(style):
        fig, axes = plt.subplots(nrows, 2, figsize=(config.figure_width_in, config.panel_height_in*nrows),
                                 squeeze=False, layout="constrained")
        for index, (panel, result, ax) in enumerate(zip(panels, results, axes.flat)):
            ax.scatter(panel.hx, panel.ocean, color="#537DA5", s=30, alpha=0.8, label="Members", zorder=3)
            ax.scatter(result["hx_mean"], result["ocean_mean"], marker="o", facecolors="white",
                       edgecolors="black", s=85, linewidths=1.5, label="Ensemble mean", zorder=5)
            ax.scatter(panel.nr_hx, panel.nr_ocean, marker="*", color="#B24B36", s=150,
                       edgecolors="white", linewidths=0.5, label="NR", zorder=6)
            ax.annotate("NR", (panel.nr_hx, panel.nr_ocean), xytext=(7,5), textcoords="offset points", color="#B24B36")
            if math.isfinite(result["slope_ocean_on_hx"]):
                xline = np.array([np.min(panel.hx), np.max(panel.hx)])
                yline = result["ocean_mean"] + result["slope_ocean_on_hx"]*(xline-result["hx_mean"])
                ax.plot(xline, yline, color="#525252", linewidth=1.4, label="Ocean-on-Hx fit", zorder=2)
            ax.axvline(result["hx_mean"], color="#AAAAAA", ls=":", lw=0.7)
            ax.axhline(result["ocean_mean"], color="#AAAAAA", ls=":", lw=0.7)
            offsets = ((7,-14), (7,12), (-22,20))
            for label_index, member in enumerate(result["leave_one_out"]["most_influential_members"]):
                position = members.index(member)
                ax.annotate(str(member), (panel.hx[position], panel.ocean[position]),
                            xytext=offsets[label_index % len(offsets)], textcoords="offset points",
                            fontsize=8, color="#34495E")
            boot = result["bootstrap"]["same_sign_fraction_valid"]
            half = result["subsets"]["same_sign_fraction_valid"]
            percent = lambda value: "undefined" if value is None else f"{100*value:.1f}%"
            # Keep diagnostic text outside the data area so it never masks
            # outlying members (the very points we are trying to inspect).
            ax.text(0.0, -0.29,
                    f"c = {result['covariance']:+.4g} K²; r = {result['correlation']:+.3f}\n"
                    f"LOO flip / zero: {result['leave_one_out']['n_sign_flips']} / "
                    f"{result['leave_one_out']['n_sign_zero']} (of {len(members)})\n"
                    f"Same c sign: bootstrap {percent(boot)}, subsets {percent(half)}\n"
                    f"Undefined r draws: boot {result['bootstrap']['n_correlation_undefined']}/"
                    f"{result['bootstrap']['n_total']}, subsets {result['subsets']['n_correlation_undefined']}/"
                    f"{result['subsets']['n_total']}",
                    transform=ax.transAxes, va="top", fontsize=8)
            ax.set_title(f"{chr(65+index)}  {panel.label}  [{panel.clear_sky_flag}]", loc="left", fontsize=11)
            ax.set_xlabel("Hx brightness temperature (K)")
            ax.set_ylabel(f"Center-time {config.inputs.state_var} (K)")
            ax.margins(x=0.16, y=0.28)
            ax.ticklabel_format(axis="both", style="plain", useOffset=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            if index == 0:
                ax.legend(loc="best", fontsize=8, frameon=False)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        fig.suptitle(f"LACC member scatter — observation {context.label} (raw row {context.raw_index})\n"
                     f"State ({context.state_lat:.4f}, {context.state_lon:.4f}); n={len(members)}; "
                     f"center {config.inputs.center_time}\n"
                     "Resampling measures ensemble sensitivity; it is not causal validation.", fontsize=11)
        paths = {}
        for fmt in config.figure_formats:
            path = config.inputs.output_dir / f"lacc_member_scatter_obs{context.label}_{config.inputs.center_tag()}.{fmt}"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=config.figure_dpi, bbox_inches="tight", pad_inches=0.12)
            paths[fmt] = str(path)
        plt.close(fig)
    return paths


def json_safe(value):
    """Recursively remove nonstandard NaN/Infinity BEFORE calling json.dump."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def run(config: ScatterConfig = CONFIG) -> dict:
    """Read validated input once, then export scatter, influence and resamples."""
    validate_config(config)
    inputs = config.inputs
    center, tags = lag.read_times_file(inputs.resolved_lacc_times())
    schedule_info = lag.resolve_lag_schedule(inputs, center, tags)
    schedule = schedule_info["schedule"]
    assimilated = set(schedule_info["assimilated_lag_hours"])
    if assimilated - {hour for hour, _ in schedule}:
        raise ValueError("missing assimilated lag(s); scatter requires the complete real window")
    center_tag = inputs.center_tag()
    profiles = {tag: lag.read_profile_coordinates(lag.profile_path(inputs, tag), inputs.nobs_raw, tag)
                for tag in dict.fromkeys([center_tag] + [tag for _, tag in schedule])}
    mask = lag.read_mask(inputs.resolved_unified_mask(), inputs.nobs_raw, center_tag)
    targets = [lag.resolve_obs_target(target, mask.values, inputs.nobs_raw) for target in inputs.obs_targets]
    if not targets or len({target["label"] for target in targets}) != len(targets):
        raise ValueError("observation labels must be nonempty and unique")
    members = inputs.member_numbers()
    backgrounds = [lag.read_level_zero_field(lag.background_member_path(inputs, member), inputs.state_var,
                   inputs.state_level, inputs.lat_var, inputs.lon_var, inputs.water_mask_source) for member in members]
    nr = lag.read_level_zero_field(inputs.resolved_nr_file(), inputs.state_var, inputs.state_level,
                                  inputs.lat_var, inputs.lon_var, inputs.water_mask_source)
    background_check = lag.validate_background_consistency(inputs, backgrounds, members, nr)
    files = lag.HxFileCache(inputs)
    nr_checks = {tag: files.verify_nr(tag) for _, tag in schedule}
    for _, tag in schedule:
        for member in members:
            files.verify_member(member, tag)
    member_check = files.member_report({tag: hour for hour, tag in schedule})
    warnings = []
    if member_check["n_unverified"] or any(not entry["available"] for entry in nr_checks.values()):
        warnings.append("Some BT matrices are absent: their F-order numbering is UNVERIFIED; see checks.")
    displacement_checks = {}
    for _, tag in schedule:
        check = lag.compare_profile_coordinates(profiles[center_tag], profiles[tag])["summary"]
        displacement_checks[tag] = check
        if not check["same_fixed_location"]:
            warnings.append(f"Profile {tag} moves relative to center: max {check['max_distance_km']:.3f} km.")
    draws = make_resamples(len(members), config)
    observations, points, summary_rows, loo_rows, resample_rows = {}, [], [], [], []
    for target in targets:
        context = lag.build_observation_context(inputs, target, profiles[center_tag], backgrounds, nr)
        series = lag.load_lag_series(inputs, context, schedule, profiles, files)
        external = lag.check_external_fo(context, series, assimilated)
        if external["status"] == "failed":
            if config.strict_external_fo:
                raise lag.ConsistencyError(f"observation {context.label}: external_FO disagrees with the loaded ensemble: {external}")
            warnings.append(f"Observation {context.label}: external_FO FAILED.")
        averages = lag.compute_noise_free_averages(inputs, context, series, assimilated)
        reference = lag.check_reference_values(inputs, context, series, averages)
        if reference["status"] == "failed":
            warnings.append(f"Observation {context.label}: reference values FAILED.")
        panels = build_panels(context, series, assimilated)
        results, observation_points = [], []
        for panel in panels:
            result, loo, samples = analyze_panel(panel, members, draws, config)
            result.update(obs_id=context.label, raw_index=context.raw_index)
            results.append(result)
            flat = {k: v for k,v in result.items() if not isinstance(v, (dict, list))}
            flat["lag_hours"] = ",".join(map(str, panel.lag_hours))
            for method in ("bootstrap", "subsets", "leave_one_out"):
                flat.update({f"{method}_{key}": value for key,value in result[method].items()})
            summary_rows.append(flat)
            loo_rows.extend(dict(obs_id=context.label, **row) for row in loo)
            resample_rows.extend(dict(obs_id=context.label, **row) for row in samples)
            for member, hx, ocean in zip(members, panel.hx, panel.ocean):
                observation_points.append({"obs_id":context.label, "raw_index":context.raw_index,
                    "panel_label":panel.label, "member":member, "hx":float(hx), "ocean":float(ocean),
                    "state_lat":context.ocean_sample.sample_lat, "state_lon":context.ocean_sample.sample_lon})
        points.extend(observation_points)
        observations[context.label] = {
            "member_numbers": members, "points": observation_points, "panels": results,
            "ocean_valid_times": context.background_times, "nr_valid_time": nr.time_string,
            "state_requested_lat":context.state_lat, "state_requested_lon":context.state_lon,
            "ocean_sample":asdict(context.ocean_sample), "nr_sample":asdict(context.nr_sample),
            "background_files":context.background_files, "nr_file":str(nr.path),
            "single_obs_location_check":context.single_obs_location,
            "external_fo_check":external, "reference_check":reference,
            "window_innovations":averages,
            "figures":make_figure(context, panels, results, members, config) if config.figure_formats else {},
        }
    tag = inputs.center_tag()
    outputs = {
        "members_csv":str(lag.write_csv(inputs.output_dir / f"lacc_scatter_members_{tag}.csv", points)),
        "summary_csv":str(lag.write_csv(inputs.output_dir / f"lacc_scatter_summary_{tag}.csv", summary_rows)),
        "leave_one_out_csv":str(lag.write_csv(inputs.output_dir / f"lacc_scatter_leave_one_out_{tag}.csv", loo_rows)),
        "resamples_csv":str(lag.write_csv(inputs.output_dir / f"lacc_scatter_resamples_{tag}.csv", resample_rows)),
        "json":str(inputs.output_dir / f"lacc_member_scatter_summary_{tag}.json"),
    }
    payload = {
        "config":asdict(config), "lag_schedule":schedule_info,
        "checks":{"background":background_check, "nr_f_order":nr_checks,
                  "member_hx_f_order":member_check, "profile_displacement":displacement_checks},
        "resampling":{"seed":config.random_seed, "bit_generator":"PCG64",
                      "pairing":"same member indices for ocean and every Hx lag",
                      "bootstrap":"with replacement", "subsets":"without replacement",
                      "interval":"percentile intervals; no independence or causal guarantee"},
        "interpretation":{
            "regression":"ocean_on_hx; NR residual is vertical, in K",
            "direction":"noise-free c*d_NR*e_o; not the actual noisy analysis increment",
            "limits":"Resampling assesses this ensemble, not forecast skill across independent cases. A stable negative covariance can still give an unsuitable ocean correction.",
        },
        "observations":observations, "outputs":outputs, "warnings":warnings,
    }
    with Path(outputs["json"]).open("w", encoding="utf-8") as stream:
        json.dump(json_safe(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
    return payload


if __name__ == "__main__":
    payload = run(CONFIG)
    print(f"Scatter diagnostic written to {CONFIG.inputs.output_dir}")
    for label, obs in payload["observations"].items():
        for row in obs["panels"]:
            print(f"obs {label}, {row['panel_label']}: c={row['covariance']:+.5g}, "
                  f"r={row['correlation']:+.3f}, NR residual={row['nr_vertical_residual_K']:+.4f} K, "
                  f"LOO flips={row['leave_one_out']['n_sign_flips']}")
    for warning in payload["warnings"]:
        print(f"WARNING: {warning}")
