"""Analyze the cached paired OM_TMP-to-atmosphere pathway tables."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =====================
# User configuration
# =====================
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(
    os.environ.get(
        "OMTMP_CACHE_DIR",
        SCRIPT_DIR / "omtmp_pathway_cache",
    )
)
SURFACE_CSV = CACHE_DIR / "omtmp_pathway_surface_blocks.csv"
VERTICAL_CSV = CACHE_DIR / "omtmp_pathway_vertical_stats.csv"
BOOTSTRAP_SAMPLES = 5000
RANDOM_SEED = 20260803
ISFTCFLX = 0
DX_M = 1500.0
SOURCE_THRESHOLD_K = 0.002


sys.path.insert(0, str(SCRIPT_DIR))
from omtmp_flux_decomposition import reconstruct_ocean_fluxes  # noqa: E402
from omtmp_pathway_extract import finite_linear_stats, source_composite  # noqa: E402


surface = pd.read_csv(SURFACE_CSV, dtype={"method": str, "member": str})
vertical = pd.read_csv(VERTICAL_CSV, dtype={"method": str, "member": str})
surface["member"] = surface["member"].str.zfill(3)
vertical["member"] = vertical["member"].str.zfill(3)

regions = {
    "0-75": (0,),
    "75-150": (1,),
    "0-150": (0, 1),
    "150-300": (2,),
    "0-300": (0, 1, 2),
}
links = (
    ("dom0_to_dom", "dom0", "dom"),
    ("dom0_to_dtsk", "dom0", "dtsk"),
    ("dom_to_dtsk", "dom", "dtsk"),
    ("dom0_to_dhfx", "dom0", "dhfx"),
    ("dom0_to_dqfx", "dom0", "dqfx"),
    ("dom0_to_dlh", "dom0", "dlh"),
    ("dom_to_dhfx", "dom", "dhfx"),
    ("dom_to_dqfx", "dom", "dqfx"),
    ("dom_to_dlh", "dom", "dlh"),
    ("dom0_to_dust", "dom0", "dust"),
    ("dom0_to_du10", "dom0", "du10"),
    ("dom0_to_dv10", "dom0", "dv10"),
    ("dhfx_to_dt2", "dhfx", "dt2"),
    ("dlh_to_dq2", "dlh", "dq2"),
    ("dt2_to_dtheta_l1", "dt2", "dtheta_l1"),
    ("dq2_to_dqv_l1", "dq2", "dqv_l1"),
    ("dom0_to_dpblh", "dom0", "dpblh"),
    ("dom0_to_dw_l5", "dom0", "dw_l5"),
    ("dom0_to_dolr", "dom0", "dolr"),
    ("dom0_to_drain", "dom0", "drain"),
)

member_rows = []
for (method, member, time_hour), group in surface.groupby(
    ["method", "member", "time_hour"], sort=True
):
    for region_name, annuli in regions.items():
        selected = group[group["annulus"].isin(annuli)]
        for link_name, source_name, response_name in links:
            stats = finite_linear_stats(selected[source_name], selected[response_name])
            if source_name in {"dom0", "dom"}:
                composite = source_composite(
                    selected[source_name],
                    selected[response_name],
                    threshold=SOURCE_THRESHOLD_K,
                )
            else:
                composite = {
                    "n_positive": 0,
                    "n_negative": 0,
                    "mean_response_positive": np.nan,
                    "mean_response_negative": np.nan,
                }
            member_rows.append(
                {
                    "method": method,
                    "member": member,
                    "time_hour": time_hour,
                    "region": region_name,
                    "link": link_name,
                    **stats,
                    **composite,
                }
            )

member_links = pd.DataFrame(member_rows)
member_links.to_csv(CACHE_DIR / "omtmp_pathway_link_member.csv", index=False)

rng = np.random.default_rng(RANDOM_SEED)
summary_rows = []
for keys, group in member_links.groupby(["method", "time_hour", "region", "link"], sort=True):
    method, time_hour, region_name, link_name = keys
    row = {
        "method": method,
        "time_hour": time_hour,
        "region": region_name,
        "link": link_name,
        "valid_corr_members": int(group["corr"].notna().sum()),
        "mean_corr": group["corr"].mean(),
        "median_corr": group["corr"].median(),
        "mean_slope": group["slope"].mean(),
        "median_slope": group["slope"].median(),
        "positive_slope_fraction": (group["slope"].dropna() > 0.0).mean(),
        "mean_rms_source": group["rms_source"].mean(),
        "mean_rms_response": group["rms_response"].mean(),
        "mean_response_positive_source": group["mean_response_positive"].mean(),
        "mean_response_negative_source": group["mean_response_negative"].mean(),
    }
    for metric in ("corr", "slope"):
        values = group[metric].dropna().to_numpy(dtype=float)
        if values.size >= 2:
            draws = rng.choice(values, size=(BOOTSTRAP_SAMPLES, values.size), replace=True).mean(axis=1)
            row[f"mean_{metric}_ci025"] = float(np.quantile(draws, 0.025))
            row[f"mean_{metric}_ci975"] = float(np.quantile(draws, 0.975))
        else:
            row[f"mean_{metric}_ci025"] = np.nan
            row[f"mean_{metric}_ci975"] = np.nan
    summary_rows.append(row)

link_summary = pd.DataFrame(summary_rows)
link_summary.to_csv(CACHE_DIR / "omtmp_pathway_link_summary.csv", index=False)

# Offline direct-flux response: hold the initial atmosphere fixed and change only OM_TMP.
initial = surface[surface["time_hour"] == 0.0].copy()
required = [
    "om0_strong", "om0_weak", "air_temp0", "qv0", "air_pressure0", "psfc0",
    "height0", "u0", "v0", "ust0", "z0m0",
]
initial = initial[np.isfinite(initial[required]).all(axis=1)].copy()
for name in ("direct_dhfx", "direct_dqfx", "direct_dlh", "direct_dust"):
    initial[name] = np.nan

for (method, member), group in initial.groupby(["method", "member"], sort=True):
    index = group.index
    common = {
        "air_temperature_k": group["air_temp0"].to_numpy()[None, :],
        "vapor_mixing_ratio": group["qv0"].to_numpy()[None, :],
        "air_pressure_pa": group["air_pressure0"].to_numpy()[None, :],
        "surface_pressure_pa": group["psfc0"].to_numpy()[None, :],
        "height_agl_m": group["height0"].to_numpy()[None, :],
        "u_ms": group["u0"].to_numpy()[None, :],
        "v_ms": group["v0"].to_numpy()[None, :],
        "initial_friction_velocity_ms": np.maximum(group["ust0"].to_numpy()[None, :], 1.0e-4),
        "initial_momentum_roughness_m": np.maximum(group["z0m0"].to_numpy()[None, :], 1.27e-7),
        "dx_m": DX_M,
        "isftcflx": ISFTCFLX,
    }
    strong_flux = reconstruct_ocean_fluxes(
        surface_temperature_k=group["om0_strong"].to_numpy()[None, :],
        **common,
    )
    weak_flux = reconstruct_ocean_fluxes(
        surface_temperature_k=group["om0_weak"].to_numpy()[None, :],
        **common,
    )
    initial.loc[index, "direct_dhfx"] = (strong_flux["hfx"] - weak_flux["hfx"]).ravel()
    initial.loc[index, "direct_dqfx"] = (strong_flux["qfx"] - weak_flux["qfx"]).ravel()
    initial.loc[index, "direct_dlh"] = (strong_flux["lh"] - weak_flux["lh"]).ravel()
    initial.loc[index, "direct_dust"] = (strong_flux["ust"] - weak_flux["ust"]).ravel()

direct_rows = []
for (method, member), group in initial.groupby(["method", "member"], sort=True):
    for region_name, annuli in regions.items():
        selected = group[group["annulus"].isin(annuli)]
        for flux_name in ("direct_dhfx", "direct_dqfx", "direct_dlh", "direct_dust"):
            stats = finite_linear_stats(selected["dom0"], selected[flux_name])
            composite = source_composite(
                selected["dom0"], selected[flux_name], threshold=SOURCE_THRESHOLD_K
            )
            direct_rows.append(
                {
                    "method": method,
                    "member": member,
                    "region": region_name,
                    "flux": flux_name,
                    **stats,
                    **composite,
                }
            )

direct_member = pd.DataFrame(direct_rows)
actual_half_hour = member_links[
    (member_links["time_hour"] == 0.5)
    & member_links["link"].isin(("dom0_to_dhfx", "dom0_to_dqfx", "dom0_to_dlh"))
].copy()
actual_half_hour["flux"] = "direct_" + actual_half_hour["link"].str.removeprefix("dom0_to_")
actual_half_hour = actual_half_hour.rename(
    columns={"slope": "actual_slope_0p5h", "corr": "actual_corr_0p5h"}
)
direct_member = direct_member.merge(
    actual_half_hour[["method", "member", "region", "flux", "actual_slope_0p5h", "actual_corr_0p5h"]],
    on=["method", "member", "region", "flux"],
    how="left",
)
direct_member["actual_to_direct_slope_ratio"] = (
    direct_member["actual_slope_0p5h"] / direct_member["slope"]
)
direct_member.to_csv(CACHE_DIR / "omtmp_pathway_direct_flux_member.csv", index=False)

direct_summary = (
    direct_member.groupby(["method", "region", "flux"], as_index=False)
    .agg(
        valid_members=("slope", "count"),
        mean_direct_corr=("corr", "mean"),
        mean_direct_slope=("slope", "mean"),
        positive_direct_slope_fraction=("slope", lambda values: (values.dropna() > 0.0).mean()),
        mean_actual_corr_0p5h=("actual_corr_0p5h", "mean"),
        mean_actual_slope_0p5h=("actual_slope_0p5h", "mean"),
        mean_actual_to_direct_slope_ratio=("actual_to_direct_slope_ratio", "mean"),
        mean_direct_rms=("rms_response", "mean"),
        mean_direct_positive_composite=("mean_response_positive", "mean"),
        mean_direct_negative_composite=("mean_response_negative", "mean"),
    )
)
direct_summary.to_csv(CACHE_DIR / "omtmp_pathway_direct_flux_summary.csv", index=False)

vertical_summary = (
    vertical.groupby(
        ["method", "time_hour", "annulus", "level", "source", "response"],
        as_index=False,
    )
    .agg(
        valid_members=("corr", "count"),
        mean_corr=("corr", "mean"),
        median_corr=("corr", "median"),
        mean_slope=("slope", "mean"),
        positive_slope_fraction=("slope", lambda values: (values.dropna() > 0.0).mean()),
        mean_rms_response=("rms_response", "mean"),
        mean_response_positive=("mean_response_positive", "mean"),
        mean_response_negative=("mean_response_negative", "mean"),
    )
)
vertical_summary.to_csv(CACHE_DIR / "omtmp_pathway_vertical_summary.csv", index=False)

print(f"surface rows: {len(surface):,}")
print(f"member link rows: {len(member_links):,}")
print(f"direct member rows: {len(direct_member):,}")
print(f"vertical summary rows: {len(vertical_summary):,}")
