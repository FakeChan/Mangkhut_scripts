"""Diagnose initial TC-region latent heat flux and OHC26.

User-editable experiment paths and plotting settings are defined near the top
of this module.  The numerical helpers are kept import-safe for unit testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

try:
    from plot_scripts.wrf41_sfclayrev import (
        R_OVER_CP,
        SfclayOptions,
        SurfaceFluxResult,
        SurfaceState,
        revised_mm5_ocean_flux,
    )
except ModuleNotFoundError:  # Direct execution from plot_scripts/.
    from wrf41_sfclayrev import (  # type: ignore[no-redef]
        R_OVER_CP,
        SfclayOptions,
        SurfaceFluxResult,
        SurfaceState,
        revised_mm5_ocean_flux,
    )


EARTH_RADIUS_KM = 6371.0
GRAVITY = 9.81
P0_PA = 100_000.0


@dataclass(frozen=True)
class Experiment:
    name: str
    label: str
    ocean_enabled: bool
    color: str


@dataclass(frozen=True)
class Config:
    experiment_root: Path
    nr_root: Path
    output_dir: Path
    valid_time: str
    member_domain: str
    nr_domain: str
    filters: tuple[str, ...]
    members: tuple[str, ...]
    experiments: tuple[Experiment, ...]
    radius_km: float = 150.0
    ocean_density_kg_m3: float = 1025.0
    ocean_cp_j_kg_k: float = 3985.0


# =============================================================================
# User configuration
# =============================================================================

CONFIG = Config(
    experiment_root=Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test"),
    nr_root=Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout"),
    output_dir=Path("./figs/initial_tc_lh_ohc"),
    valid_time="2018-09-10_00:00:00",
    member_domain="d02",
    nr_domain="d03",
    filters=("EAKF", "QCF_RHF"),
    members=("006", "015", "029", "037", "043", "044"),
    experiments=(
        Experiment("6mem_oceanAssim0Run0", "No DA", False, "#0072B2"),
        Experiment(
            "6mem_oceanAssim0Run1", "Weak-couple DA", True, "#D55E00"
        ),
        Experiment(
            "6mem_oceanAssim1Run1", "Strong-couple DA", True, "#009E73"
        ),
    ),
)


def haversine_distance_km(
    lats: np.ndarray,
    lons: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> np.ndarray:
    """Calculate great-circle distance from one center to a lat/lon grid."""
    latitude = np.asarray(lats, dtype=float)
    longitude = np.asarray(lons, dtype=float)
    if latitude.shape != longitude.shape or latitude.ndim != 2:
        raise ValueError("latitude and longitude must be matching 2-D arrays")
    if not np.isfinite(center_lat) or not np.isfinite(center_lon):
        raise ValueError("center latitude and longitude must be finite")
    lat1 = np.deg2rad(latitude)
    lon1 = np.deg2rad(longitude)
    lat2 = np.deg2rad(center_lat)
    lon2 = np.deg2rad(center_lon)
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2
    a += np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def tc_ocean_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    ocean_mask: np.ndarray,
) -> np.ndarray:
    """Return ocean points no farther than ``radius_km`` from the TC center."""
    if radius_km <= 0.0:
        raise ValueError("radius_km must be positive")
    ocean = np.asarray(ocean_mask, dtype=bool)
    distances = haversine_distance_km(lats, lons, center_lat, center_lon)
    if ocean.shape != distances.shape:
        raise ValueError("ocean mask shape must match the coordinate grid")
    mask = ocean & np.isfinite(distances) & (distances <= radius_km + 1.0e-9)
    if not np.any(mask):
        raise ValueError(f"no ocean grid points found within {radius_km:g} km")
    return mask


def ohc26_profile(
    temperature_c: np.ndarray,
    depth_m: np.ndarray,
    rho: float,
    cp: float,
) -> float:
    """Integrate heat above 26 C down to the first 26 C crossing."""
    temperature = np.asarray(temperature_c, dtype=float)
    depth = np.asarray(depth_m, dtype=float)
    if temperature.ndim != 1 or depth.ndim != 1 or temperature.shape != depth.shape:
        raise ValueError("temperature and depth profiles must be matching 1-D arrays")
    if temperature.size < 2:
        raise ValueError("OHC26 requires at least two ocean levels")
    if np.any(~np.isfinite(temperature)) or np.any(~np.isfinite(depth)):
        raise ValueError("temperature and depth profiles must be finite")
    if np.any(np.diff(depth) <= 0.0):
        raise ValueError("ocean depth must be strictly monotonic increasing")
    if depth[0] < 0.0:
        raise ValueError("ocean depth must be positive downward")
    if rho <= 0.0 or cp <= 0.0:
        raise ValueError("rho and cp must be positive")
    if temperature[0] <= 26.0:
        return 0.0

    crossing_candidates = np.flatnonzero(
        (temperature[:-1] > 26.0) & (temperature[1:] <= 26.0)
    )
    if crossing_candidates.size == 0:
        raise ValueError("ocean profile has no first 26 C crossing")
    crossing = int(crossing_candidates[0])
    t0 = temperature[crossing]
    t1 = temperature[crossing + 1]
    z0 = depth[crossing]
    z1 = depth[crossing + 1]
    d26 = z0 + (26.0 - t0) * (z1 - z0) / (t1 - t0)

    warm_temperature = temperature[: crossing + 1] - 26.0
    warm_depth = depth[: crossing + 1]
    if warm_depth[0] > 0.0:
        warm_depth = np.concatenate(([0.0], warm_depth))
        warm_temperature = np.concatenate(
            ([warm_temperature[0]], warm_temperature)
        )
    warm_depth = np.concatenate((warm_depth, [d26]))
    warm_temperature = np.concatenate((warm_temperature, [0.0]))
    integral_k_m = float(np.trapezoid(warm_temperature, warm_depth))
    return rho * cp * integral_k_m


def ohc26_field(
    temperature_c: np.ndarray,
    depth_m: np.ndarray,
    rho: float,
    cp: float,
) -> np.ndarray:
    """Calculate OHC26 for a ``(ocean_level, y, x)`` temperature field."""
    temperature = np.asarray(temperature_c, dtype=float)
    if temperature.ndim != 3:
        raise ValueError("ocean temperature must have dimensions (level, y, x)")
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim == 1:
        if depth.size != temperature.shape[0]:
            raise ValueError("depth level count does not match ocean temperature")
        depth_field = np.broadcast_to(
            depth[:, None, None], temperature.shape
        )
    elif depth.shape == temperature.shape:
        depth_field = depth
    else:
        raise ValueError("depth must be 1-D or match the ocean temperature field")

    output = np.empty(temperature.shape[1:], dtype=float)
    for j, i in np.ndindex(output.shape):
        output[j, i] = ohc26_profile(
            temperature[:, j, i], depth_field[:, j, i], rho, cp
        )
    return output


def _time0(data: xr.DataArray) -> xr.DataArray:
    for dim in data.dims:
        if dim.lower() == "time":
            return data.isel({dim: 0})
    return data


def _mass_2d(ds: xr.Dataset, name: str) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"required WRF variable {name} is missing")
    data = _time0(ds[name])
    for dim in tuple(data.dims):
        if dim in {"bottom_top", "bottom_top_stag"}:
            data = data.isel({dim: 0})
    values = np.asarray(data.squeeze().values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-D after slicing, got {values.shape}")
    return values


def _lowest_staggered_wind(ds: xr.Dataset, name: str) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"required WRF variable {name} is missing")
    data = _time0(ds[name])
    if "bottom_top" in data.dims:
        data = data.isel(bottom_top=0)
    values = np.asarray(data.values, dtype=float)
    if name == "U":
        if "west_east_stag" not in data.dims:
            raise ValueError("U does not have west_east_stag")
        axis = data.dims.index("west_east_stag")
    else:
        if "south_north_stag" not in data.dims:
            raise ValueError("V does not have south_north_stag")
        axis = data.dims.index("south_north_stag")
    left = np.take(values, np.arange(values.shape[axis] - 1), axis=axis)
    right = np.take(values, np.arange(1, values.shape[axis]), axis=axis)
    output = 0.5 * (left + right)
    if output.ndim != 2:
        raise ValueError(f"destaggered {name} must be 2-D, got {output.shape}")
    return output


def _ocean_surface_temperature(ds: xr.Dataset, use_ocean_sst: bool) -> np.ndarray:
    if not use_ocean_sst:
        return _mass_2d(ds, "TSK")
    if "OM_TMP" not in ds:
        raise KeyError("OM_TMP is required for an ocean-running experiment")
    data = _time0(ds["OM_TMP"])
    ocean_dims = [dim for dim in data.dims if "ocean_layer" in dim]
    if len(ocean_dims) != 1:
        raise ValueError(f"could not identify one OM_TMP ocean dimension: {data.dims}")
    values = np.asarray(data.isel({ocean_dims[0]: 0}).values, dtype=float)
    units = str(ds["OM_TMP"].attrs.get("units", "")).lower()
    if "c" in units and "k" not in units:
        values = values + 273.15
    elif "k" not in units and np.nanmedian(values) < 100.0:
        values = values + 273.15
    if values.ndim != 2:
        raise ValueError(f"OM_TMP surface must be 2-D, got {values.shape}")
    return values


def _physics_attribute(ds: xr.Dataset, name: str, default: int | None = None) -> int:
    if name not in ds.attrs:
        if default is None:
            raise KeyError(f"required WRF physics attribute {name} is missing")
        return default
    value = np.asarray(ds.attrs[name]).ravel()
    if value.size == 0:
        raise ValueError(f"WRF physics attribute {name} is empty")
    return int(value[0])


def read_surface_state(
    ds: xr.Dataset,
    use_ocean_sst: bool,
) -> tuple[SurfaceState, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the lowest-level mass-grid state needed by SFCLAYREV."""
    sfclay = _physics_attribute(ds, "SF_SFCLAY_PHYSICS")
    if sfclay != 1:
        raise ValueError(
            f"unsupported SF_SFCLAY_PHYSICS={sfclay}; only WRF 4.1 Revised MM5 (1) is implemented"
        )
    if _physics_attribute(ds, "ISFFLX", 1) != 1:
        raise ValueError("ISFFLX must be 1 to diagnose surface moisture exchange")

    u = _lowest_staggered_wind(ds, "U")
    v = _lowest_staggered_wind(ds, "V")
    perturbation_theta = _mass_2d(ds, "T")
    pressure = _mass_2d(ds, "P") + _mass_2d(ds, "PB")
    theta = perturbation_theta + 300.0
    air_temperature = theta * (pressure / P0_PA) ** R_OVER_CP

    if "PH" not in ds or "PHB" not in ds:
        raise KeyError("PH and PHB are required to reconstruct lowest-level height")
    geopotential = np.asarray((_time0(ds["PH"]) + _time0(ds["PHB"])).values, dtype=float)
    vertical_dim = _time0(ds["PH"]).dims.index("bottom_top_stag")
    lower = np.take(geopotential, 0, axis=vertical_dim)
    upper = np.take(geopotential, 1, axis=vertical_dim)
    height = 0.5 * (lower + upper) / GRAVITY - _mass_2d(ds, "HGT")

    lats = _mass_2d(ds, "XLAT")
    lons = _mass_2d(ds, "XLONG")
    if "XLAND" in ds:
        ocean = _mass_2d(ds, "XLAND") > 1.5
    elif "LANDMASK" in ds:
        ocean = _mass_2d(ds, "LANDMASK") < 0.5
    else:
        raise KeyError("XLAND or LANDMASK is required to identify ocean points")

    shape = lats.shape
    named = {
        "U": u,
        "V": v,
        "T": air_temperature,
        "QVAPOR": _mass_2d(ds, "QVAPOR"),
        "pressure": pressure,
        "height": height,
        "PSFC": _mass_2d(ds, "PSFC"),
        "surface temperature": _ocean_surface_temperature(ds, use_ocean_sst),
        "PBLH": _mass_2d(ds, "PBLH"),
        "ocean mask": ocean,
    }
    bad_shapes = {name: value.shape for name, value in named.items() if value.shape != shape}
    if bad_shapes:
        raise ValueError(f"WRF mass-grid shape mismatch: expected {shape}, got {bad_shapes}")

    state = SurfaceState(
        air_temperature_k=air_temperature,
        surface_temperature_k=named["surface temperature"],
        vapor_mixing_ratio=named["QVAPOR"],
        air_pressure_pa=pressure,
        surface_pressure_pa=named["PSFC"],
        height_agl_m=height,
        u_ms=u,
        v_ms=v,
        pbl_height_m=named["PBLH"],
        dx_m=float(ds.attrs.get("DX", np.nan)),
    )
    return state, lats, lons, ocean


def calculate_lh_field(ds: xr.Dataset, use_ocean_sst: bool) -> SurfaceFluxResult:
    state, _, _, _ = read_surface_state(ds, use_ocean_sst)
    isftcflx = _physics_attribute(ds, "ISFTCFLX", 0)
    return revised_mm5_ocean_flux(state, SfclayOptions(isftcflx=isftcflx))


def read_ohc_inputs(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Read ocean temperature in C and depth in metres positive downward."""
    if "OM_TMP" not in ds or "OM_DEPTH" not in ds:
        raise KeyError("OM_TMP and OM_DEPTH are required for OHC26")
    temperature_da = _time0(ds["OM_TMP"])
    depth_da = _time0(ds["OM_DEPTH"])
    ocean_dims = [dim for dim in temperature_da.dims if "ocean_layer" in dim]
    if len(ocean_dims) != 1:
        raise ValueError(f"could not identify one OM_TMP ocean dimension: {temperature_da.dims}")
    ocean_dim = ocean_dims[0]
    temperature_da = temperature_da.transpose(
        ocean_dim, *[dim for dim in temperature_da.dims if dim != ocean_dim]
    )
    if ocean_dim not in depth_da.dims:
        depth_ocean_dims = [dim for dim in depth_da.dims if "ocean_layer" in dim]
        if len(depth_ocean_dims) != 1:
            raise ValueError(f"could not identify OM_DEPTH ocean dimension: {depth_da.dims}")
        depth_da = depth_da.rename({depth_ocean_dims[0]: ocean_dim})
    depth_da = depth_da.transpose(
        ocean_dim, *[dim for dim in depth_da.dims if dim != ocean_dim]
    )

    temperature = np.asarray(temperature_da.values, dtype=float)
    depth = np.asarray(depth_da.values, dtype=float)
    temp_units = str(ds["OM_TMP"].attrs.get("units", "")).lower()
    if "k" in temp_units or ("c" not in temp_units and np.nanmedian(temperature) > 100.0):
        temperature = temperature - 273.15
    elif "c" not in temp_units:
        raise ValueError(f"unknown OM_TMP units: {temp_units!r}")

    depth_units = str(ds["OM_DEPTH"].attrs.get("units", "")).lower().strip()
    if depth_units not in {"m", "meter", "meters", "metre", "metres"}:
        raise ValueError(f"unknown OM_DEPTH units: {depth_units!r}")
    if depth.ndim == 1:
        depth = np.broadcast_to(depth[:, None, None], temperature.shape)
    elif depth.shape != temperature.shape:
        try:
            depth = np.broadcast_to(depth, temperature.shape)
        except ValueError as exc:
            raise ValueError(
                f"OM_DEPTH shape {depth.shape} cannot broadcast to OM_TMP {temperature.shape}"
            ) from exc
    mean_profile = np.nanmean(depth.reshape(depth.shape[0], -1), axis=1)
    if np.all(np.diff(mean_profile) < 0.0):
        depth = -depth
    if np.any(np.diff(depth, axis=0) <= 0.0):
        raise ValueError("OM_DEPTH must be strictly monotonic positive downward")
    return temperature, depth


def find_unique_wrfout(root: Path, domain: str, valid_time: str) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    basename = f"wrfout_{domain}_{valid_time}"
    exact = sorted(path for path in root.rglob(basename) if path.is_file())
    matches = exact or sorted(path for path in root.rglob(f"{basename}*") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"no {basename}* below {root}")
    if len(matches) != 1:
        raise ValueError(f"ambiguous {basename}* below {root}: {matches}")
    return matches[0]


def read_tc_center(
    nr_path: Path,
    slp_reader: Callable[[Path], np.ndarray] | None = None,
) -> tuple[float, float, float]:
    with xr.open_dataset(nr_path, decode_times=False) as ds:
        lats = _mass_2d(ds, "XLAT")
        lons = _mass_2d(ds, "XLONG")
    if slp_reader is None:
        try:
            import netCDF4
            from wrf import getvar, to_np
        except ImportError as exc:
            raise ImportError("wrf-python and netCDF4 are required to diagnose NR SLP") from exc
        with netCDF4.Dataset(nr_path) as nc:
            slp = np.asarray(to_np(getvar(nc, "slp", timeidx=0)), dtype=float)
    else:
        slp = np.asarray(slp_reader(Path(nr_path)), dtype=float)
    if slp.shape != lats.shape:
        raise ValueError(f"SLP shape {slp.shape} does not match NR grid {lats.shape}")
    if not np.isfinite(slp).any():
        raise ValueError("NR SLP contains no finite values")
    index = np.unravel_index(np.nanargmin(slp), slp.shape)
    return float(lats[index]), float(lons[index]), float(slp[index])


def attach_group_statistics(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    grouped = frame.groupby(["experiment", "filter"])[value_column]
    output = frame.copy()
    output["ensemble_mean"] = grouped.transform("mean")
    output["ensemble_std"] = grouped.transform(lambda values: values.std(ddof=1))
    return output


def _ohc_on_mask(
    temperature_c: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    rho: float,
    cp: float,
) -> tuple[float, int]:
    values = []
    for j, i in zip(*np.where(mask), strict=True):
        values.append(
            ohc26_profile(temperature_c[:, j, i], depth_m[:, j, i], rho, cp)
        )
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("OHC26 contains nonfinite selected ocean values")
    return float(array.mean()), int(array.size)


def calculate_member_records(
    config: Config,
    slp_reader: Callable[[Path], np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nr_path = find_unique_wrfout(config.nr_root, config.nr_domain, config.valid_time)
    center_lat, center_lon, center_slp = read_tc_center(nr_path, slp_reader)
    lh_records: list[dict[str, object]] = []
    ohc_records: list[dict[str, object]] = []

    for experiment in config.experiments:
        for filter_name in config.filters:
            for member in config.members:
                member_root = (
                    config.experiment_root / experiment.name / filter_name / member
                )
                member_path = find_unique_wrfout(
                    member_root, config.member_domain, config.valid_time
                )
                print(f"Reading {experiment.name}/{filter_name}/{member}: {member_path}")
                with xr.open_dataset(member_path, decode_times=False) as ds:
                    state, lats, lons, ocean = read_surface_state(
                        ds, experiment.ocean_enabled
                    )
                    mask = tc_ocean_mask(
                        lats,
                        lons,
                        center_lat,
                        center_lon,
                        config.radius_km,
                        ocean,
                    )
                    isftcflx = _physics_attribute(ds, "ISFTCFLX", 0)
                    flux = revised_mm5_ocean_flux(
                        state, SfclayOptions(isftcflx=isftcflx)
                    )
                    lh_values = flux.lh[mask]
                    if not np.isfinite(lh_values).all():
                        raise ValueError(f"nonfinite LH in selected region: {member_path}")
                    common = {
                        "experiment": experiment.name,
                        "experiment_label": experiment.label,
                        "filter": filter_name,
                        "member": member,
                        "input_path": str(member_path),
                        "center_lat": center_lat,
                        "center_lon": center_lon,
                        "center_slp_hpa": center_slp,
                        "tc_ocean_points": int(mask.sum()),
                        "sf_sfclay_physics": 1,
                        "isftcflx": isftcflx,
                        "stored_flux_used": False,
                    }
                    lh_records.append(
                        {
                            **common,
                            "lh_mean_w_m2": float(lh_values.mean()),
                            "lh_finite_points": int(np.isfinite(lh_values).sum()),
                        }
                    )
                    if experiment.ocean_enabled:
                        temperature_c, depth_m = read_ohc_inputs(ds)
                        ohc_j_m2, finite_points = _ohc_on_mask(
                            temperature_c,
                            depth_m,
                            mask,
                            config.ocean_density_kg_m3,
                            config.ocean_cp_j_kg_k,
                        )
                        ohc_records.append(
                            {
                                **common,
                                "ohc26_mean_j_m2": ohc_j_m2,
                                "ohc26_mean_kj_cm2": ohc_j_m2 / 1.0e7,
                                "ohc_finite_points": finite_points,
                            }
                        )

    lh = attach_group_statistics(pd.DataFrame(lh_records), "lh_mean_w_m2")
    ohc = attach_group_statistics(pd.DataFrame(ohc_records), "ohc26_mean_kj_cm2")
    expected_lh = len(config.experiments) * len(config.filters) * len(config.members)
    expected_ohc = (
        sum(experiment.ocean_enabled for experiment in config.experiments)
        * len(config.filters)
        * len(config.members)
    )
    if len(lh) != expected_lh or len(ohc) != expected_ohc:
        raise RuntimeError(
            f"record-count mismatch: LH={len(lh)}/{expected_lh}, OHC={len(ohc)}/{expected_ohc}"
        )
    return lh, ohc
