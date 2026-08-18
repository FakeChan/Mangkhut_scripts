"""Read-only paired extraction for the OM_TMP-to-atmosphere pathway.

The script is stored locally and streamed to the remote Python interpreter.
It writes a ZIP stream containing reduced CSV tables to standard output; it
never opens a remote output file.
"""

from __future__ import annotations

import csv
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

from omtmp_evidence_cache import discover_common_members, parse_methods


# =====================
# User configuration
# =====================
BASE_DIR = Path(os.environ.get("OMTMP_FORECAST_BASE_DIR", "/scratch/lililei1/kcfu/tc_mangkhut/cycle_test"))
STRONG_EXPERIMENT = os.environ.get("OMTMP_STRONG_EXPERIMENT", "6mem_oceanAssim1Run1")
WEAK_EXPERIMENT = os.environ.get("OMTMP_WEAK_EXPERIMENT", "6mem_oceanAssim0Run1")
METHODS = parse_methods(os.environ.get("OMTMP_METHODS"))
_MAX_MEMBERS = os.environ.get("OMTMP_MAX_MEMBERS_PER_METHOD")
MAX_MEMBERS_PER_METHOD = None if not _MAX_MEMBERS else int(_MAX_MEMBERS)
DOMAIN = "d02"
TIMES = (
    (0.0, "2018-09-10_00:00:00"),
    (0.5, "2018-09-10_00:30:00"),
    (1.0, "2018-09-10_01:00:00"),
    (1.5, "2018-09-10_01:30:00"),
    (2.0, "2018-09-10_02:00:00"),
    (3.0, "2018-09-10_03:00:00"),
    (4.0, "2018-09-10_04:00:00"),
    (6.0, "2018-09-10_06:00:00"),
)
SEARCH_LAT = (10.0, 25.0)
SEARCH_LON = (135.0, 155.0)
HALF_WIDTH_GRID = 230
MAX_RADIUS_KM = 300.0
BLOCK_SIZE = 10
MIN_VALID_FRACTION = 0.5
PROFILE_LEVEL_COUNT = 10
SOURCE_COMPOSITE_THRESHOLD_K = 0.002


def get_method_members():
    return discover_common_members(
        BASE_DIR,
        strong_experiment=STRONG_EXPERIMENT,
        weak_experiment=WEAK_EXPERIMENT,
        methods=METHODS,
        max_members_per_method=MAX_MEMBERS_PER_METHOD,
    )


def haversine_km(lat, lon, center_lat, center_lon):
    """Great-circle distance from one center, in km."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dlat = np.deg2rad(lat - center_lat)
    dlon = np.deg2rad(lon - center_lon)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.deg2rad(center_lat))
        * np.cos(np.deg2rad(lat))
        * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * 6371.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def annulus_index(distance_km):
    """Map distances to 0–75, 75–150, and 150–300 km annuli."""
    distance = np.asarray(distance_km, dtype=float)
    result = np.full(distance.shape, -1, dtype=int)
    result[(distance >= 0.0) & (distance < 75.0)] = 0
    result[(distance >= 75.0) & (distance < 150.0)] = 1
    result[(distance >= 150.0) & (distance <= 300.0)] = 2
    return result


def block_mean_records(fields, valid_mask, block_size=10, min_valid_fraction=0.5):
    """Return aligned block means for blocks with sufficient valid coverage."""
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("valid_mask must be 2-D")
    arrays = {name: np.asarray(value, dtype=float) for name, value in fields.items()}
    if any(value.shape != valid.shape for value in arrays.values()):
        raise ValueError("all fields must match valid_mask")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in (0, 1]")

    records = []
    ny, nx = valid.shape
    for y0 in range(0, ny, block_size):
        for x0 in range(0, nx, block_size):
            y1 = min(y0 + block_size, ny)
            x1 = min(x0 + block_size, nx)
            block_valid = valid[y0:y1, x0:x1]
            if block_valid.mean() < min_valid_fraction:
                continue
            record = {
                "block_y": y0 // block_size,
                "block_x": x0 // block_size,
                "valid_fraction": float(block_valid.mean()),
            }
            for name, value in arrays.items():
                block = value[y0:y1, x0:x1]
                good = block_valid & np.isfinite(block) & (np.abs(block) < 1.0e30)
                record[name] = float(np.mean(block[good])) if np.any(good) else np.nan
            records.append(record)
    return records


def finite_linear_stats(source, response):
    """Finite-pair descriptive statistics and ordinary least-squares slope."""
    x = np.asarray(source, dtype=float).ravel()
    y = np.asarray(response, dtype=float).ravel()
    good = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1.0e30) & (np.abs(y) < 1.0e30)
    x = x[good]
    y = y[good]
    result = {
        "n": int(x.size),
        "mean_source": np.nan,
        "rms_source": np.nan,
        "mean_response": np.nan,
        "rms_response": np.nan,
        "corr": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
    }
    if x.size == 0:
        return result
    result.update(
        mean_source=float(np.mean(x)),
        rms_source=float(np.sqrt(np.mean(x * x))),
        mean_response=float(np.mean(y)),
        rms_response=float(np.sqrt(np.mean(y * y))),
    )
    variance = float(np.mean((x - np.mean(x)) ** 2))
    if x.size >= 3 and variance > 0.0 and float(np.std(y)) > 0.0:
        covariance = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
        slope = covariance / variance
        result.update(
            corr=float(np.corrcoef(x, y)[0, 1]),
            slope=slope,
            intercept=float(np.mean(y) - slope * np.mean(x)),
        )
    return result


def source_composite(source, response, threshold=0.0):
    """Response means in source-warming and source-cooling samples."""
    x = np.asarray(source, dtype=float).ravel()
    y = np.asarray(response, dtype=float).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    positive = good & (x > threshold)
    negative = good & (x < -threshold)
    return {
        "n_positive": int(np.sum(positive)),
        "n_negative": int(np.sum(negative)),
        "mean_response_positive": float(np.mean(y[positive])) if np.any(positive) else np.nan,
        "mean_response_negative": float(np.mean(y[negative])) if np.any(negative) else np.nan,
    }


if __name__ == "__main__":
    from netCDF4 import Dataset

    method_members = get_method_members()

    surface_columns = [
        "method", "member", "time_hour", "center_lat", "center_lon",
        "block_y", "block_x", "valid_fraction", "lat", "lon", "distance_km",
        "annulus", "dom0", "dom", "dtsk", "dhfx", "dqfx", "dlh", "dust",
        "dt2", "dq2", "dpsfc", "dpblh", "du10", "dv10", "dolr", "drain",
        "dw_l5", "dtheta_l1", "dqv_l1", "om0_strong", "om0_weak",
        "air_temp0", "qv0", "air_pressure0", "psfc0", "height0", "u0", "v0",
        "ust0", "z0m0",
    ]
    profile_columns = [
        "method", "member", "time_hour", "annulus", "level", "source",
        "response", "n", "mean_source", "rms_source", "mean_response",
        "rms_response", "corr", "slope", "intercept", "n_positive",
        "n_negative", "mean_response_positive", "mean_response_negative",
    ]
    metadata_columns = [
        "method", "member", "time_hour", "center_lat", "center_lon",
        "mean_pair_min_psfc_hpa", "block_count",
    ]

    surface_buffer = io.StringIO()
    profile_buffer = io.StringIO()
    metadata_buffer = io.StringIO()
    surface_writer = csv.DictWriter(surface_buffer, fieldnames=surface_columns, lineterminator="\n")
    profile_writer = csv.DictWriter(profile_buffer, fieldnames=profile_columns, lineterminator="\n")
    metadata_writer = csv.DictWriter(metadata_buffer, fieldnames=metadata_columns, lineterminator="\n")
    surface_writer.writeheader()
    profile_writer.writeheader()
    metadata_writer.writeheader()

    time0_name = TIMES[0][1]
    for method in METHODS:
        for member in method_members[method]:
            strong_dir = BASE_DIR / STRONG_EXPERIMENT / method / member
            weak_dir = BASE_DIR / WEAK_EXPERIMENT / method / member
            strong0_path = strong_dir / f"wrfout_{DOMAIN}_{time0_name}"
            weak0_path = weak_dir / f"wrfout_{DOMAIN}_{time0_name}"

            with Dataset(strong0_path, "r") as strong0, Dataset(weak0_path, "r") as weak0:
                lat_full = np.asarray(np.ma.filled(weak0.variables["XLAT"][0], np.nan), dtype=float)
                lon_full = np.asarray(np.ma.filled(weak0.variables["XLONG"][0], np.nan), dtype=float)
                land_full = np.asarray(np.ma.filled(weak0.variables["LANDMASK"][0], np.nan), dtype=float)
                om0_strong_full = np.asarray(np.ma.filled(strong0.variables["OM_TMP"][0, 0], np.nan), dtype=float)
                om0_weak_full = np.asarray(np.ma.filled(weak0.variables["OM_TMP"][0, 0], np.nan), dtype=float)

                initial_full = {}
                p0 = np.asarray(np.ma.filled(weak0.variables["P"][0, 0], np.nan), dtype=float)
                p0 += np.asarray(np.ma.filled(weak0.variables["PB"][0, 0], np.nan), dtype=float)
                theta0 = np.asarray(np.ma.filled(weak0.variables["T"][0, 0], np.nan), dtype=float) + 300.0
                initial_full["air_temp0"] = theta0 * (p0 / 100000.0) ** (287.0 / 1004.0)
                initial_full["qv0"] = np.asarray(np.ma.filled(weak0.variables["QVAPOR"][0, 0], np.nan), dtype=float)
                initial_full["air_pressure0"] = p0
                initial_full["psfc0"] = np.asarray(np.ma.filled(weak0.variables["PSFC"][0], np.nan), dtype=float)
                geopotential0 = np.asarray(np.ma.filled(weak0.variables["PH"][0, 0:2], np.nan), dtype=float)
                geopotential0 += np.asarray(np.ma.filled(weak0.variables["PHB"][0, 0:2], np.nan), dtype=float)
                hgt0 = np.asarray(np.ma.filled(weak0.variables["HGT"][0], np.nan), dtype=float)
                initial_full["height0"] = 0.5 * (geopotential0[0] + geopotential0[1]) / 9.81 - hgt0
                u_stag = np.asarray(np.ma.filled(weak0.variables["U"][0, 0], np.nan), dtype=float)
                v_stag = np.asarray(np.ma.filled(weak0.variables["V"][0, 0], np.nan), dtype=float)
                initial_full["u0"] = 0.5 * (u_stag[:, :-1] + u_stag[:, 1:])
                initial_full["v0"] = 0.5 * (v_stag[:-1, :] + v_stag[1:, :])
                initial_full["ust0"] = np.asarray(np.ma.filled(weak0.variables["UST"][0], np.nan), dtype=float)
                safe_ust = np.maximum(initial_full["ust0"], 1.0e-4)
                initial_full["z0m0"] = np.minimum(
                    0.0185 * safe_ust**2 / 9.81 + 0.11 * 1.5e-5 / safe_ust,
                    2.85e-3,
                )

            for time_hour, time_name in TIMES:
                strong_path = strong_dir / f"wrfout_{DOMAIN}_{time_name}"
                weak_path = weak_dir / f"wrfout_{DOMAIN}_{time_name}"
                print(f"extract {method} {member} {time_name}", file=sys.stderr, flush=True)
                with Dataset(strong_path, "r") as strong, Dataset(weak_path, "r") as weak:
                    ps_strong_full = np.asarray(np.ma.filled(strong.variables["PSFC"][0], np.nan), dtype=float)
                    ps_weak_full = np.asarray(np.ma.filled(weak.variables["PSFC"][0], np.nan), dtype=float)
                    search = (
                        (land_full < 0.5)
                        & (lat_full >= SEARCH_LAT[0]) & (lat_full <= SEARCH_LAT[1])
                        & (lon_full >= SEARCH_LON[0]) & (lon_full <= SEARCH_LON[1])
                    )
                    center_field = 0.5 * (ps_strong_full + ps_weak_full)
                    center_field = np.where(search, center_field, np.inf)
                    center_flat = int(np.argmin(center_field))
                    center_y, center_x = np.unravel_index(center_flat, center_field.shape)
                    center_lat = float(lat_full[center_y, center_x])
                    center_lon = float(lon_full[center_y, center_x])
                    y0 = max(0, center_y - HALF_WIDTH_GRID)
                    y1 = min(lat_full.shape[0], center_y + HALF_WIDTH_GRID + 1)
                    x0 = max(0, center_x - HALF_WIDTH_GRID)
                    x1 = min(lat_full.shape[1], center_x + HALF_WIDTH_GRID + 1)
                    ys = slice(y0, y1)
                    xs = slice(x0, x1)

                    lat = lat_full[ys, xs]
                    lon = lon_full[ys, xs]
                    land = land_full[ys, xs]
                    distance = haversine_km(lat, lon, center_lat, center_lon)
                    valid = (land < 0.5) & (distance <= MAX_RADIUS_KM)

                    om0_strong = om0_strong_full[ys, xs]
                    om0_weak = om0_weak_full[ys, xs]
                    om_strong = np.asarray(np.ma.filled(strong.variables["OM_TMP"][0, 0, ys, xs], np.nan), dtype=float)
                    om_weak = np.asarray(np.ma.filled(weak.variables["OM_TMP"][0, 0, ys, xs], np.nan), dtype=float)
                    fields = {
                        "lat": lat,
                        "lon": lon,
                        "distance_km": distance,
                        "dom0": om0_strong - om0_weak,
                        "dom": om_strong - om_weak,
                        "om0_strong": om0_strong,
                        "om0_weak": om0_weak,
                    }
                    for name, output_name in (
                        ("TSK", "dtsk"), ("HFX", "dhfx"), ("QFX", "dqfx"),
                        ("LH", "dlh"), ("UST", "dust"), ("T2", "dt2"),
                        ("Q2", "dq2"), ("PSFC", "dpsfc"), ("PBLH", "dpblh"),
                        ("U10", "du10"), ("V10", "dv10"), ("OLR", "dolr"),
                    ):
                        a = np.asarray(np.ma.filled(strong.variables[name][0, ys, xs], np.nan), dtype=float)
                        b = np.asarray(np.ma.filled(weak.variables[name][0, ys, xs], np.nan), dtype=float)
                        fields[output_name] = a - b
                    rain_strong = np.asarray(np.ma.filled(strong.variables["RAINC"][0, ys, xs], np.nan), dtype=float)
                    rain_strong += np.asarray(np.ma.filled(strong.variables["RAINNC"][0, ys, xs], np.nan), dtype=float)
                    rain_weak = np.asarray(np.ma.filled(weak.variables["RAINC"][0, ys, xs], np.nan), dtype=float)
                    rain_weak += np.asarray(np.ma.filled(weak.variables["RAINNC"][0, ys, xs], np.nan), dtype=float)
                    fields["drain"] = rain_strong - rain_weak
                    w_strong = np.asarray(np.ma.filled(strong.variables["W"][0, 4, ys, xs], np.nan), dtype=float)
                    w_weak = np.asarray(np.ma.filled(weak.variables["W"][0, 4, ys, xs], np.nan), dtype=float)
                    fields["dw_l5"] = w_strong - w_weak

                    theta_strong = np.asarray(
                        np.ma.filled(strong.variables["T"][0, :PROFILE_LEVEL_COUNT, ys, xs], np.nan), dtype=float
                    )
                    theta_weak = np.asarray(
                        np.ma.filled(weak.variables["T"][0, :PROFILE_LEVEL_COUNT, ys, xs], np.nan), dtype=float
                    )
                    qv_strong = np.asarray(
                        np.ma.filled(strong.variables["QVAPOR"][0, :PROFILE_LEVEL_COUNT, ys, xs], np.nan), dtype=float
                    )
                    qv_weak = np.asarray(
                        np.ma.filled(weak.variables["QVAPOR"][0, :PROFILE_LEVEL_COUNT, ys, xs], np.nan), dtype=float
                    )
                    for level in range(PROFILE_LEVEL_COUNT):
                        fields[f"dtheta_{level + 1}"] = theta_strong[level] - theta_weak[level]
                        fields[f"dqv_{level + 1}"] = qv_strong[level] - qv_weak[level]

                    for name, full in initial_full.items():
                        fields[name] = full[ys, xs]

                    records = block_mean_records(
                        fields,
                        valid,
                        block_size=BLOCK_SIZE,
                        min_valid_fraction=MIN_VALID_FRACTION,
                    )
                    metadata_writer.writerow(
                        {
                            "method": method,
                            "member": member,
                            "time_hour": time_hour,
                            "center_lat": center_lat,
                            "center_lon": center_lon,
                            "mean_pair_min_psfc_hpa": center_field[center_y, center_x] / 100.0,
                            "block_count": len(records),
                        }
                    )

                    for record in records:
                        record["method"] = method
                        record["member"] = member
                        record["time_hour"] = time_hour
                        record["center_lat"] = center_lat
                        record["center_lon"] = center_lon
                        record["annulus"] = int(annulus_index(record["distance_km"]))
                        record["dtheta_l1"] = record["dtheta_1"]
                        record["dqv_l1"] = record["dqv_1"]
                        surface_writer.writerow({name: record.get(name, np.nan) for name in surface_columns})

                    for annulus in (0, 1, 2):
                        selected = [r for r in records if int(annulus_index(r["distance_km"])) == annulus]
                        if not selected:
                            continue
                        for level in range(1, PROFILE_LEVEL_COUNT + 1):
                            for source_name in ("dom0", "dom"):
                                source_values = np.array([r[source_name] for r in selected])
                                for response_name in (f"dtheta_{level}", f"dqv_{level}"):
                                    response_values = np.array([r[response_name] for r in selected])
                                    stats = finite_linear_stats(source_values, response_values)
                                    composite = source_composite(
                                        source_values,
                                        response_values,
                                        threshold=SOURCE_COMPOSITE_THRESHOLD_K,
                                    )
                                    profile_writer.writerow(
                                        {
                                            "method": method,
                                            "member": member,
                                            "time_hour": time_hour,
                                            "annulus": annulus,
                                            "level": level,
                                            "source": source_name,
                                            "response": response_name.split("_")[0],
                                            **stats,
                                            **composite,
                                        }
                                    )

    readme = (
        "Read-only paired strong-minus-weak cache.\n"
        f"Strong={STRONG_EXPERIMENT}; weak={WEAK_EXPERIMENT}; domain={DOMAIN}.\n"
        f"Members by method={method_members}.\n"
        f"Blocks={BLOCK_SIZE}x{BLOCK_SIZE} model cells; minimum valid ocean fraction={MIN_VALID_FRACTION}.\n"
        "Annuli: 0=0-75 km, 1=75-150 km, 2=150-300 km.\n"
        "T is WRF perturbation potential temperature; dtheta columns are paired T differences.\n"
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        bundle.writestr("omtmp_pathway_surface_blocks.csv", surface_buffer.getvalue())
        bundle.writestr("omtmp_pathway_vertical_stats.csv", profile_buffer.getvalue())
        bundle.writestr("omtmp_pathway_metadata.csv", metadata_buffer.getvalue())
        bundle.writestr("README.txt", readme)
    sys.stdout.buffer.write(archive.getvalue())
