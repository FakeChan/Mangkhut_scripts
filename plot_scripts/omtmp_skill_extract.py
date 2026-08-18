"""Read-only strong/weak forecast-skill extraction against the NR truth.

This local script is streamed to the remote Python interpreter. It writes one
ZIP byte stream to stdout and never creates a file on the remote server.
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
NR_DIR = Path(os.environ.get("OMTMP_NR_DIR", "/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain"))
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
NR_TIME_BRACKETS = {
    0.0: ("2018-09-10_00:00:00", "2018-09-10_00:00:00", 0.0),
    0.5: ("2018-09-10_00:00:00", "2018-09-10_01:00:00", 0.5),
    1.0: ("2018-09-10_01:00:00", "2018-09-10_01:00:00", 0.0),
    1.5: ("2018-09-10_01:00:00", "2018-09-10_02:00:00", 0.5),
    2.0: ("2018-09-10_02:00:00", "2018-09-10_02:00:00", 0.0),
    3.0: ("2018-09-10_03:00:00", "2018-09-10_03:00:00", 0.0),
    4.0: ("2018-09-10_04:00:00", "2018-09-10_04:00:00", 0.0),
    6.0: ("2018-09-10_06:00:00", "2018-09-10_06:00:00", 0.0),
}
SEARCH_LAT = (10.0, 25.0)
SEARCH_LON = (135.0, 155.0)
MAX_RADIUS_KM = 300.0
BLOCK_SIZE = 10
MIN_VALID_FRACTION = 0.5
PROFILE_LEVEL_COUNT = 10
# Optional smoke-test guard. Leave unset/zero for the complete extraction.
MAX_CASES = int(os.environ.get("OMTMP_SKILL_MAX_CASES", "0"))
TIME_FILTER = os.environ.get("OMTMP_SKILL_TIME_FILTER")
TIME_FILTER = None if TIME_FILTER is None else float(TIME_FILTER)


def get_method_members():
    return discover_common_members(
        BASE_DIR,
        strong_experiment=STRONG_EXPERIMENT,
        weak_experiment=WEAK_EXPERIMENT,
        methods=METHODS,
        max_members_per_method=MAX_MEMBERS_PER_METHOD,
    )

SURFACE_SPECS = (
    ("om", "OM_TMP", "K"),
    ("tsk", "TSK", "K"),
    ("hfx", "HFX", "W m-2"),
    ("qfx", "QFX", "kg m-2 s-1"),
    ("lh", "LH", "W m-2"),
    ("t2", "T2", "K"),
    ("q2", "Q2", "kg kg-1"),
    ("pblh", "PBLH", "m"),
    ("ust", "UST", "m s-1"),
    ("psfc", "PSFC", "Pa"),
    ("u10", "U10", "m s-1"),
    ("v10", "V10", "m s-1"),
    ("olr", "OLR", "W m-2"),
)
DERIVED_SPECS = (
    ("ws10", "m s-1"),
    ("theta_l1", "K"),
    ("qv_l1", "kg kg-1"),
)
BLOCK_VARIABLES = tuple(name for name, _, _ in SURFACE_SPECS) + tuple(
    name for name, _ in DERIVED_SPECS
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


def paired_skill_metrics(strong, weak, truth, valid_mask=None):
    """Metrics on exactly the same finite points; positive improvement favors strong."""
    strong = np.asarray(strong, dtype=float)
    weak = np.asarray(weak, dtype=float)
    truth = np.asarray(truth, dtype=float)
    good = np.isfinite(strong) & np.isfinite(weak) & np.isfinite(truth)
    good &= (np.abs(strong) < 1.0e30) & (np.abs(weak) < 1.0e30) & (np.abs(truth) < 1.0e30)
    if valid_mask is not None:
        good &= np.asarray(valid_mask, dtype=bool)
    if not np.any(good):
        return {
            key: (0 if key == "n" else np.nan)
            for key in (
                "n", "mean_strong", "mean_weak", "mean_truth", "strong_bias",
                "weak_bias", "strong_mae", "weak_mae", "strong_rmse",
                "weak_rmse", "bias_abs_improvement", "mae_improvement",
                "rmse_improvement", "rmse_improvement_pct",
            )
        }
    s = strong[good]
    w = weak[good]
    t = truth[good]
    es = s - t
    ew = w - t
    sb = float(np.mean(es))
    wb = float(np.mean(ew))
    smae = float(np.mean(np.abs(es)))
    wmae = float(np.mean(np.abs(ew)))
    srmse = float(np.sqrt(np.mean(es * es)))
    wrmse = float(np.sqrt(np.mean(ew * ew)))
    return {
        "n": int(s.size),
        "mean_strong": float(np.mean(s)),
        "mean_weak": float(np.mean(w)),
        "mean_truth": float(np.mean(t)),
        "strong_bias": sb,
        "weak_bias": wb,
        "strong_mae": smae,
        "weak_mae": wmae,
        "strong_rmse": srmse,
        "weak_rmse": wrmse,
        "bias_abs_improvement": abs(wb) - abs(sb),
        "mae_improvement": wmae - smae,
        "rmse_improvement": wrmse - srmse,
        "rmse_improvement_pct": 100.0 * (wrmse - srmse) / wrmse if wrmse > 0.0 else np.nan,
    }


def _filled(variable, index):
    return np.asarray(np.ma.filled(variable[index], np.nan), dtype=float)


def _nr_blended(low, high, alpha, variable_name, index):
    """Read an exact NR field or linearly interpolate two surrounding outputs."""
    first = _filled(low.variables[variable_name], index)
    if alpha == 0.0:
        return first
    second = _filled(high.variables[variable_name], index)
    return (1.0 - alpha) * first + alpha * second


def _surface_fields(dataset, ys, xs):
    fields = {}
    for output_name, variable_name, _ in SURFACE_SPECS:
        if variable_name == "OM_TMP":
            fields[output_name] = _filled(dataset.variables[variable_name], (0, 0, ys, xs))
        else:
            fields[output_name] = _filled(dataset.variables[variable_name], (0, ys, xs))
    fields["ws10"] = np.hypot(fields["u10"], fields["v10"])
    fields["theta_l1"] = _filled(dataset.variables["T"], (0, 0, ys, xs)) + 300.0
    fields["qv_l1"] = _filled(dataset.variables["QVAPOR"], (0, 0, ys, xs))
    return fields


def _interpolate_2d(field, y_index, x_index, order=1):
    from scipy.ndimage import map_coordinates

    coordinates = np.vstack((y_index.ravel(), x_index.ravel()))
    return map_coordinates(
        np.asarray(field, dtype=float), coordinates, order=order,
        mode="constant", cval=np.nan, prefilter=False,
    ).reshape(y_index.shape)


def _nr_surface_fields(low, high, alpha, y_index, x_index):
    fields = {}
    for output_name, variable_name, _ in SURFACE_SPECS:
        if variable_name == "OM_TMP":
            raw = _nr_blended(low, high, alpha, variable_name, (0, 0, slice(None), slice(None)))
        else:
            raw = _nr_blended(low, high, alpha, variable_name, (0, slice(None), slice(None)))
        fields[output_name] = _interpolate_2d(raw, y_index, x_index)
    fields["ws10"] = np.hypot(fields["u10"], fields["v10"])
    theta = _nr_blended(low, high, alpha, "T", (0, 0, slice(None), slice(None))) + 300.0
    qv = _nr_blended(low, high, alpha, "QVAPOR", (0, 0, slice(None), slice(None)))
    fields["theta_l1"] = _interpolate_2d(theta, y_index, x_index)
    fields["qv_l1"] = _interpolate_2d(qv, y_index, x_index)
    return fields


def _storm_center(lat, lon, land, psfc):
    search = (
        (land < 0.5)
        & (lat >= SEARCH_LAT[0]) & (lat <= SEARCH_LAT[1])
        & (lon >= SEARCH_LON[0]) & (lon <= SEARCH_LON[1])
        & np.isfinite(psfc)
    )
    work = np.where(search, psfc, np.inf)
    flat = int(np.argmin(work))
    y, x = np.unravel_index(flat, work.shape)
    if not np.isfinite(work[y, x]):
        raise RuntimeError("No finite ocean PSFC point found in storm search box")
    return y, x, float(lat[y, x]), float(lon[y, x]), float(psfc[y, x])


def _region_masks(distance):
    return {
        "r000_075": (distance >= 0.0) & (distance < 75.0),
        "r075_150": (distance >= 75.0) & (distance < 150.0),
        "r150_300": (distance >= 150.0) & (distance <= 300.0),
        "r000_300": (distance >= 0.0) & (distance <= 300.0),
    }


def _write_row(writer, base, stats):
    writer.writerow({**base, **stats})


if __name__ == "__main__":
    from netCDF4 import Dataset
    from wrf import ll_to_xy

    method_members = get_method_members()

    metric_columns = [
        "method", "member", "time_hour", "nr_time_method", "region", "mask_type", "variable", "unit",
        "n", "mean_strong", "mean_weak", "mean_truth", "strong_bias", "weak_bias",
        "strong_mae", "weak_mae", "strong_rmse", "weak_rmse", "bias_abs_improvement",
        "mae_improvement", "rmse_improvement", "rmse_improvement_pct",
    ]
    vertical_columns = metric_columns + ["level"]
    intensity_columns = [
        "method", "member", "time_hour", "nr_time_method", "nr_center_lat", "nr_center_lon",
        "strong_center_lat", "strong_center_lon", "weak_center_lat", "weak_center_lon",
        "strong_track_error_km", "weak_track_error_km", "track_improvement_km",
        "nr_min_psfc_hpa", "strong_min_psfc_hpa", "weak_min_psfc_hpa",
        "strong_min_psfc_abs_error_hpa", "weak_min_psfc_abs_error_hpa",
        "min_psfc_improvement_hpa", "nr_max_ws10", "strong_max_ws10", "weak_max_ws10",
        "strong_max_ws10_abs_error", "weak_max_ws10_abs_error", "max_ws10_improvement",
    ]
    block_columns = [
        "method", "member", "time_hour", "nr_time_method", "nr_center_lat", "nr_center_lon",
        "block_y", "block_x", "valid_fraction", "lat", "lon", "distance_km", "annulus",
    ]
    for variable in BLOCK_VARIABLES:
        block_columns.extend((f"{variable}_s", f"{variable}_w", f"{variable}_nr", f"{variable}_es", f"{variable}_ew"))

    metric_buffer = io.StringIO()
    vertical_buffer = io.StringIO()
    intensity_buffer = io.StringIO()
    block_buffer = io.StringIO()
    metric_writer = csv.DictWriter(metric_buffer, fieldnames=metric_columns, lineterminator="\n")
    vertical_writer = csv.DictWriter(vertical_buffer, fieldnames=vertical_columns, lineterminator="\n")
    intensity_writer = csv.DictWriter(intensity_buffer, fieldnames=intensity_columns, lineterminator="\n")
    block_writer = csv.DictWriter(block_buffer, fieldnames=block_columns, lineterminator="\n")
    for writer in (metric_writer, vertical_writer, intensity_writer, block_writer):
        writer.writeheader()

    example_method = next(method for method in METHODS if method_members[method])
    example_member = method_members[example_method][0]
    example_path = BASE_DIR / STRONG_EXPERIMENT / example_method / example_member / f"wrfout_{DOMAIN}_{TIMES[0][1]}"
    nr0_path = NR_DIR / f"wrfout_{DOMAIN}_{TIMES[0][1]}"
    with Dataset(example_path, "r") as example, Dataset(nr0_path, "r") as nr0:
        lat_full = _filled(example.variables["XLAT"], (0, slice(None), slice(None)))
        lon_full = _filled(example.variables["XLONG"], (0, slice(None), slice(None)))
        land_full = _filled(example.variables["LANDMASK"], (0, slice(None), slice(None)))
        xy = ll_to_xy(nr0, lat_full, lon_full, as_int=False, meta=False)
        x_nr_full = np.asarray(xy[0], dtype=float).reshape(lat_full.shape)
        y_nr_full = np.asarray(xy[1], dtype=float).reshape(lat_full.shape)
        mapping_inside_fraction = float(np.mean(
            (x_nr_full >= 0.0) & (x_nr_full <= len(nr0.dimensions["west_east"]) - 1)
            & (y_nr_full >= 0.0) & (y_nr_full <= len(nr0.dimensions["south_north"]) - 1)
        ))

    units = {name: unit for name, _, unit in SURFACE_SPECS}
    units.update(dict(DERIVED_SPECS))
    active_times = [item for item in TIMES if TIME_FILTER is None or item[0] == TIME_FILTER]
    case_count = sum(len(method_members[method]) for method in METHODS) * len(active_times)
    completed_cases = 0

    for time_hour, time_name in TIMES:
        if TIME_FILTER is not None and time_hour != TIME_FILTER:
            continue
        if MAX_CASES and completed_cases >= MAX_CASES:
            break
        nr_low_name, nr_high_name, nr_alpha = NR_TIME_BRACKETS[time_hour]
        nr_low_path = NR_DIR / f"wrfout_{DOMAIN}_{nr_low_name}"
        nr_high_path = NR_DIR / f"wrfout_{DOMAIN}_{nr_high_name}"
        nr_time_method = "exact" if nr_alpha == 0.0 else f"linear_{nr_alpha:g}"
        print(f"load NR {time_name} ({nr_time_method})", file=sys.stderr, flush=True)
        with Dataset(nr_low_path, "r") as nr_low:
            nr_high = nr_low if nr_alpha == 0.0 else Dataset(nr_high_path, "r")
            nr_lat = _filled(nr_low.variables["XLAT"], (0, slice(None), slice(None)))
            nr_lon = _filled(nr_low.variables["XLONG"], (0, slice(None), slice(None)))
            nr_land = _filled(nr_low.variables["LANDMASK"], (0, slice(None), slice(None)))
            nr_psfc = _nr_blended(
                nr_low, nr_high, nr_alpha, "PSFC", (0, slice(None), slice(None))
            )
            _, _, center_lat, center_lon, nr_min_psfc = _storm_center(nr_lat, nr_lon, nr_land, nr_psfc)

            distance_full = haversine_km(lat_full, lon_full, center_lat, center_lon)
            near = distance_full <= (MAX_RADIUS_KM + 20.0)
            y_points, x_points = np.where(near)
            if y_points.size == 0:
                raise RuntimeError(f"NR center outside experiment domain at {time_name}")
            y0, y1 = int(y_points.min()), int(y_points.max()) + 1
            x0, x1 = int(x_points.min()), int(x_points.max()) + 1
            ys, xs = slice(y0, y1), slice(x0, x1)
            lat = lat_full[ys, xs]
            lon = lon_full[ys, xs]
            land = land_full[ys, xs]
            distance = distance_full[ys, xs]
            x_nr = x_nr_full[ys, xs]
            y_nr = y_nr_full[ys, xs]
            nr_fields = _nr_surface_fields(nr_low, nr_high, nr_alpha, y_nr, x_nr)
            nr_land_on_exp = _interpolate_2d(nr_land, y_nr, x_nr, order=0)
            ocean = (land < 0.5) & (nr_land_on_exp < 0.5)
            regions = _region_masks(distance)
            nr_profile = {}
            for level in range(PROFILE_LEVEL_COUNT):
                theta = _nr_blended(
                    nr_low, nr_high, nr_alpha, "T", (0, level, slice(None), slice(None))
                ) + 300.0
                qv = _nr_blended(
                    nr_low, nr_high, nr_alpha, "QVAPOR", (0, level, slice(None), slice(None))
                )
                nr_profile[("theta", level)] = _interpolate_2d(theta, y_nr, x_nr)
                nr_profile[("qv", level)] = _interpolate_2d(qv, y_nr, x_nr)

            for method in METHODS:
                if MAX_CASES and completed_cases >= MAX_CASES:
                    break
                for member in method_members[method]:
                    if MAX_CASES and completed_cases >= MAX_CASES:
                        break
                    strong_path = BASE_DIR / STRONG_EXPERIMENT / method / member / f"wrfout_{DOMAIN}_{time_name}"
                    weak_path = BASE_DIR / WEAK_EXPERIMENT / method / member / f"wrfout_{DOMAIN}_{time_name}"
                    with Dataset(strong_path, "r") as strong, Dataset(weak_path, "r") as weak:
                        strong_fields = _surface_fields(strong, ys, xs)
                        weak_fields = _surface_fields(weak, ys, xs)

                        for variable in BLOCK_VARIABLES:
                            for region_name, region_mask in regions.items():
                                mask_types = ("ocean",) if variable == "om" else ("ocean", "all")
                                for mask_type in mask_types:
                                    valid = region_mask & (ocean if mask_type == "ocean" else True)
                                    stats = paired_skill_metrics(
                                        strong_fields[variable], weak_fields[variable], nr_fields[variable], valid
                                    )
                                    _write_row(metric_writer, {
                                        "method": method, "member": member, "time_hour": time_hour,
                                        "nr_time_method": nr_time_method,
                                        "region": region_name, "mask_type": mask_type,
                                        "variable": variable, "unit": units[variable],
                                    }, stats)

                        block_fields = {"lat": lat, "lon": lon, "distance_km": distance}
                        for variable in BLOCK_VARIABLES:
                            s = strong_fields[variable]
                            w = weak_fields[variable]
                            t = nr_fields[variable]
                            block_fields[f"{variable}_s"] = s
                            block_fields[f"{variable}_w"] = w
                            block_fields[f"{variable}_nr"] = t
                            block_fields[f"{variable}_es"] = s - t
                            block_fields[f"{variable}_ew"] = w - t
                        records = block_mean_records(
                            block_fields, ocean & regions["r000_300"],
                            block_size=BLOCK_SIZE, min_valid_fraction=MIN_VALID_FRACTION,
                        )
                        for record in records:
                            record.update({
                                "method": method, "member": member, "time_hour": time_hour,
                                "nr_time_method": nr_time_method,
                                "nr_center_lat": center_lat, "nr_center_lon": center_lon,
                                "annulus": int(annulus_index(record["distance_km"])),
                            })
                            block_writer.writerow({name: record.get(name, np.nan) for name in block_columns})

                        search_local = regions["r000_300"] & np.isfinite(strong_fields["psfc"]) & np.isfinite(weak_fields["psfc"])
                        s_center_work = np.where(search_local & ocean, strong_fields["psfc"], np.inf)
                        w_center_work = np.where(search_local & ocean, weak_fields["psfc"], np.inf)
                        sy, sx = np.unravel_index(int(np.argmin(s_center_work)), s_center_work.shape)
                        wy, wx = np.unravel_index(int(np.argmin(w_center_work)), w_center_work.shape)
                        s_track = float(haversine_km(lat[sy, sx], lon[sy, sx], center_lat, center_lon))
                        w_track = float(haversine_km(lat[wy, wx], lon[wy, wx], center_lat, center_lon))
                        nr_ws_max = float(np.nanmax(np.where(regions["r000_300"], nr_fields["ws10"], np.nan)))
                        s_ws_max = float(np.nanmax(np.where(regions["r000_300"], strong_fields["ws10"], np.nan)))
                        w_ws_max = float(np.nanmax(np.where(regions["r000_300"], weak_fields["ws10"], np.nan)))
                        s_min = float(s_center_work[sy, sx])
                        w_min = float(w_center_work[wy, wx])
                        intensity_writer.writerow({
                            "method": method, "member": member, "time_hour": time_hour,
                            "nr_time_method": nr_time_method,
                            "nr_center_lat": center_lat, "nr_center_lon": center_lon,
                            "strong_center_lat": lat[sy, sx], "strong_center_lon": lon[sy, sx],
                            "weak_center_lat": lat[wy, wx], "weak_center_lon": lon[wy, wx],
                            "strong_track_error_km": s_track, "weak_track_error_km": w_track,
                            "track_improvement_km": w_track - s_track,
                            "nr_min_psfc_hpa": nr_min_psfc / 100.0,
                            "strong_min_psfc_hpa": s_min / 100.0, "weak_min_psfc_hpa": w_min / 100.0,
                            "strong_min_psfc_abs_error_hpa": abs(s_min - nr_min_psfc) / 100.0,
                            "weak_min_psfc_abs_error_hpa": abs(w_min - nr_min_psfc) / 100.0,
                            "min_psfc_improvement_hpa": (abs(w_min - nr_min_psfc) - abs(s_min - nr_min_psfc)) / 100.0,
                            "nr_max_ws10": nr_ws_max, "strong_max_ws10": s_ws_max, "weak_max_ws10": w_ws_max,
                            "strong_max_ws10_abs_error": abs(s_ws_max - nr_ws_max),
                            "weak_max_ws10_abs_error": abs(w_ws_max - nr_ws_max),
                            "max_ws10_improvement": abs(w_ws_max - nr_ws_max) - abs(s_ws_max - nr_ws_max),
                        })

                        for level in range(PROFILE_LEVEL_COUNT):
                            for variable, variable_name, unit in (
                                ("theta", "T", "K"), ("qv", "QVAPOR", "kg kg-1")
                            ):
                                s = _filled(strong.variables[variable_name], (0, level, ys, xs))
                                w = _filled(weak.variables[variable_name], (0, level, ys, xs))
                                if variable == "theta":
                                    s += 300.0
                                    w += 300.0
                                t = nr_profile[(variable, level)]
                                for region_name, region_mask in regions.items():
                                    for mask_type in ("ocean", "all"):
                                        valid = region_mask & (ocean if mask_type == "ocean" else True)
                                        stats = paired_skill_metrics(s, w, t, valid)
                                        _write_row(vertical_writer, {
                                            "method": method, "member": member, "time_hour": time_hour,
                                            "nr_time_method": nr_time_method,
                                            "region": region_name, "mask_type": mask_type,
                                            "variable": variable, "unit": unit, "level": level + 1,
                                        }, stats)

                    completed_cases += 1
                    print(
                        f"case {completed_cases}/{case_count} {method} {member} t={time_hour:g}h",
                        file=sys.stderr, flush=True,
                    )
            if nr_high is not nr_low:
                nr_high.close()

    readme = (
        "Read-only paired skill cache against NR truth.\n"
        f"Strong={STRONG_EXPERIMENT}; weak={WEAK_EXPERIMENT}; NR={NR_DIR}; domain={DOMAIN}.\n"
        f"Members by method={method_members}.\n"
        "NR is bilinearly interpolated to the experiment grid using wrf.ll_to_xy and scipy.map_coordinates.\n"
        "NR half-hour truth is linearly time-interpolated between the surrounding hourly NR outputs.\n"
        f"Fraction of experiment grid inside NR={mapping_inside_fraction:.8f}.\n"
        "All strong/weak metrics use identical finite points. Positive improvement means strong coupling is better.\n"
        "Ocean masks require both experiment and nearest-interpolated NR LANDMASK < 0.5.\n"
        "Annuli: r000_075, r075_150, r150_300, and their union r000_300.\n"
        f"Spatial cache uses {BLOCK_SIZE}x{BLOCK_SIZE} experiment-grid blocks (~15 km).\n"
        "OM_TMP uses ocean layer 0. theta is WRF T+300 K; qv is QVAPOR at model levels.\n"
        "TSK and OM_TMP are both retained to diagnose the model's OM_TMP(layer 0)->TSK handoff.\n"
        "HFX/QFX at forecast hour 0 may be initialization values and should be interpreted cautiously.\n"
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        bundle.writestr("omtmp_skill_metrics.csv", metric_buffer.getvalue())
        bundle.writestr("omtmp_skill_vertical.csv", vertical_buffer.getvalue())
        bundle.writestr("omtmp_skill_intensity.csv", intensity_buffer.getvalue())
        bundle.writestr("omtmp_skill_blocks.csv", block_buffer.getvalue())
        bundle.writestr("README.txt", readme)
    sys.stdout.buffer.write(archive.getvalue())
