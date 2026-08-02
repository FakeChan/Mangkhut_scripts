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


@dataclass(frozen=True)
class StaticGrid:
    signature: tuple[object, ...]
    lats: np.ndarray
    lons: np.ndarray
    ocean: np.ndarray
    terrain_height_m: np.ndarray


# =============================================================================
# User configuration
# =============================================================================

CONFIG = Config(
    experiment_root=Path("/scratch/lililei1/kcfu/tc_mangkhut/cycle_test"),
    nr_root=Path("/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout/2domain"),
    output_dir=Path("./figs/initial_tc_lh_ohc"),
    valid_time="2018-09-10_00:00:00",
    member_domain="d02",
    nr_domain="d02",
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
    layer_thickness = np.diff(warm_depth)
    layer_mean_anomaly = 0.5 * (warm_temperature[:-1] + warm_temperature[1:])
    integral_k_m = float(np.sum(layer_mean_anomaly * layer_thickness))
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


def _mass_2d(
    ds: xr.Dataset,
    name: str,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"required WRF variable {name} is missing")
    data = _time0(ds[name])
    for dim in tuple(data.dims):
        if dim in {"bottom_top", "bottom_top_stag"}:
            data = data.isel({dim: 0})
    spatial_indexers = {}
    if "south_north" in data.dims:
        spatial_indexers["south_north"] = y_slice
    if "west_east" in data.dims:
        spatial_indexers["west_east"] = x_slice
    if spatial_indexers:
        data = data.isel(spatial_indexers)
    values = np.asarray(data.values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-D after slicing, got {values.shape}")
    return values


def _expanded_staggered_slice(selection: slice, mass_size: int) -> slice:
    start, stop, step = selection.indices(mass_size)
    if step != 1:
        raise ValueError("spatial slices must use a unit step")
    return slice(start, stop + 1, 1)


def _lowest_staggered_wind(
    ds: xr.Dataset,
    name: str,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"required WRF variable {name} is missing")
    data = _time0(ds[name])
    if "bottom_top" in data.dims:
        data = data.isel(bottom_top=0)
    if name == "U":
        if "west_east_stag" not in data.dims:
            raise ValueError("U does not have west_east_stag")
        data = data.isel(
            south_north=y_slice,
            west_east_stag=_expanded_staggered_slice(
                x_slice, int(ds.sizes["west_east"])
            ),
        )
        axis = data.dims.index("west_east_stag")
    else:
        if "south_north_stag" not in data.dims:
            raise ValueError("V does not have south_north_stag")
        data = data.isel(
            south_north_stag=_expanded_staggered_slice(
                y_slice, int(ds.sizes["south_north"])
            ),
            west_east=x_slice,
        )
        axis = data.dims.index("south_north_stag")
    values = np.asarray(data.values, dtype=float)
    left = np.take(values, np.arange(values.shape[axis] - 1), axis=axis)
    right = np.take(values, np.arange(1, values.shape[axis]), axis=axis)
    output = 0.5 * (left + right)
    if output.ndim != 2:
        raise ValueError(f"destaggered {name} must be 2-D, got {output.shape}")
    return output


def _ocean_surface_temperature(
    ds: xr.Dataset,
    use_ocean_sst: bool,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
) -> np.ndarray:
    if not use_ocean_sst:
        return _mass_2d(ds, "TSK", y_slice, x_slice)
    if "OM_TMP" not in ds:
        raise KeyError("OM_TMP is required for an ocean-running experiment")
    data = _time0(ds["OM_TMP"])
    ocean_dims = [dim for dim in data.dims if "ocean_layer" in dim]
    if len(ocean_dims) != 1:
        raise ValueError(f"could not identify one OM_TMP ocean dimension: {data.dims}")
    data = data.isel({ocean_dims[0]: 0})
    data = data.isel(south_north=y_slice, west_east=x_slice)
    values = np.asarray(data.values, dtype=float)
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


_GRID_SIGNATURE_ATTRIBUTES = (
    "DX",
    "DY",
    "MAP_PROJ",
    "CEN_LAT",
    "CEN_LON",
    "MOAD_CEN_LAT",
    "STAND_LON",
    "TRUELAT1",
    "TRUELAT2",
    "I_PARENT_START",
    "J_PARENT_START",
    "PARENT_GRID_RATIO",
)


def _hashable_attribute(value: object) -> tuple[object, ...]:
    array = np.asarray(value).ravel()
    return tuple(item.item() if hasattr(item, "item") else item for item in array)


def static_grid_signature(ds: xr.Dataset) -> tuple[object, ...]:
    """Build a metadata-only key for a fixed WRF mass grid."""
    if "south_north" not in ds.sizes or "west_east" not in ds.sizes:
        raise KeyError("south_north and west_east dimensions are required")
    if "XLAND" in ds:
        mask_variable = "XLAND"
    elif "LANDMASK" in ds:
        mask_variable = "LANDMASK"
    else:
        raise KeyError("XLAND or LANDMASK is required to identify ocean points")
    attributes = tuple(
        (name, _hashable_attribute(ds.attrs[name]))
        for name in _GRID_SIGNATURE_ATTRIBUTES
        if name in ds.attrs
    )
    return (
        int(ds.sizes["south_north"]),
        int(ds.sizes["west_east"]),
        mask_variable,
        attributes,
    )


def read_static_grid(ds: xr.Dataset) -> StaticGrid:
    """Read mass-grid fields that are invariant across ensemble members."""
    signature = static_grid_signature(ds)
    mask_variable = str(signature[2])
    lats = _mass_2d(ds, "XLAT")
    lons = _mass_2d(ds, "XLONG")
    if mask_variable == "XLAND":
        ocean = _mass_2d(ds, mask_variable) > 1.5
    else:
        ocean = _mass_2d(ds, mask_variable) < 0.5
    terrain_height_m = _mass_2d(ds, "HGT")
    return StaticGrid(signature, lats, lons, ocean, terrain_height_m)


def read_surface_state(
    ds: xr.Dataset,
    use_ocean_sst: bool,
    static_grid: StaticGrid | None = None,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
) -> tuple[SurfaceState, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the lowest-level mass-grid state needed by SFCLAYREV."""
    sfclay = _physics_attribute(ds, "SF_SFCLAY_PHYSICS")
    if sfclay != 1:
        raise ValueError(
            f"unsupported SF_SFCLAY_PHYSICS={sfclay}; only WRF 4.1 Revised MM5 (1) is implemented"
        )
    if _physics_attribute(ds, "ISFFLX") != 1:
        raise ValueError("ISFFLX must be 1 to diagnose surface moisture exchange")

    u = _lowest_staggered_wind(ds, "U", y_slice, x_slice)
    v = _lowest_staggered_wind(ds, "V", y_slice, x_slice)
    perturbation_theta = _mass_2d(ds, "T", y_slice, x_slice)
    pressure = _mass_2d(ds, "P", y_slice, x_slice) + _mass_2d(
        ds, "PB", y_slice, x_slice
    )
    theta = perturbation_theta + 300.0
    air_temperature = theta * (pressure / P0_PA) ** R_OVER_CP

    if "PH" not in ds or "PHB" not in ds:
        raise KeyError("PH and PHB are required to reconstruct lowest-level height")
    geopotential_indexers = {
        "bottom_top_stag": slice(0, 2),
        "south_north": y_slice,
        "west_east": x_slice,
    }
    ph = _time0(ds["PH"]).isel(geopotential_indexers)
    phb = _time0(ds["PHB"]).isel(geopotential_indexers)
    geopotential = np.asarray((ph + phb).values, dtype=float)
    vertical_dim = ph.dims.index("bottom_top_stag")
    lower = np.take(geopotential, 0, axis=vertical_dim)
    upper = np.take(geopotential, 1, axis=vertical_dim)
    if static_grid is None:
        static_grid = read_static_grid(ds)
    elif static_grid.signature != static_grid_signature(ds):
        raise ValueError("cached static grid does not match the WRF dataset")
    lats = static_grid.lats[y_slice, x_slice]
    lons = static_grid.lons[y_slice, x_slice]
    ocean = static_grid.ocean[y_slice, x_slice]
    terrain_height_m = static_grid.terrain_height_m[y_slice, x_slice]
    height = 0.5 * (lower + upper) / GRAVITY - terrain_height_m

    shape = lats.shape
    surface_temperature = _ocean_surface_temperature(
        ds, use_ocean_sst, y_slice, x_slice
    )
    if use_ocean_sst:
        # PWP ocean variables can be zero/uninitialized over land.  Land values
        # are never diagnosed, but must remain thermodynamically valid while the
        # vectorized surface-layer kernel evaluates the mass grid.
        surface_temperature = np.where(
            ocean,
            surface_temperature,
            _mass_2d(ds, "TSK", y_slice, x_slice),
        )
    named = {
        "U": u,
        "V": v,
        "T": air_temperature,
        "QVAPOR": _mass_2d(ds, "QVAPOR", y_slice, x_slice),
        "pressure": pressure,
        "height": height,
        "PSFC": _mass_2d(ds, "PSFC", y_slice, x_slice),
        "surface temperature": surface_temperature,
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
        dx_m=float(ds.attrs.get("DX", np.nan)),
    )
    return state, lats, lons, ocean


def calculate_lh_field(ds: xr.Dataset, use_ocean_sst: bool) -> SurfaceFluxResult:
    state, _, _, _ = read_surface_state(ds, use_ocean_sst)
    isftcflx = _physics_attribute(ds, "ISFTCFLX")
    return revised_mm5_ocean_flux(state, SfclayOptions(isftcflx=isftcflx))


def select_surface_state(state: SurfaceState, mask: np.ndarray) -> SurfaceState:
    """Pack selected 2-D points into one row for the ocean-only solver."""
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 2 or not np.any(selected):
        raise ValueError("surface-state mask must be a nonempty 2-D array")

    arrays = {
        name: np.asarray(value, dtype=float)
        for name, value in state.__dict__.items()
        if name != "dx_m"
    }
    bad_shapes = {
        name: values.shape
        for name, values in arrays.items()
        if values.shape != selected.shape
    }
    if bad_shapes:
        raise ValueError(
            f"surface-state arrays do not match mask {selected.shape}: {bad_shapes}"
        )
    packed = {name: values[selected][None, :] for name, values in arrays.items()}
    return SurfaceState(**packed, dx_m=state.dx_m)


def read_ohc_inputs(
    ds: xr.Dataset,
    y_slice: slice = slice(None),
    x_slice: slice = slice(None),
    surface_temperature_k: np.ndarray | None = None,
    depth_reference_index: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read selected ocean temperature and positive-downward PWP depth."""
    if "OM_TMP" not in ds or "OM_DEPTH" not in ds:
        raise KeyError("OM_TMP and OM_DEPTH are required for OHC26")
    temperature_da = _time0(ds["OM_TMP"])
    depth_da = _time0(ds["OM_DEPTH"])
    ocean_dims = [dim for dim in temperature_da.dims if "ocean_layer" in dim]
    if len(ocean_dims) != 1:
        raise ValueError(
            f"could not identify one OM_TMP ocean dimension: {temperature_da.dims}"
        )
    ocean_dim = ocean_dims[0]
    horizontal_dims = [dim for dim in temperature_da.dims if dim != ocean_dim]
    if len(horizontal_dims) != 2:
        raise ValueError(
            "OM_TMP must have two horizontal dimensions after time slicing: "
            f"{temperature_da.dims}"
        )
    spatial_indexers = {
        horizontal_dims[0]: y_slice,
        horizontal_dims[1]: x_slice,
    }
    temperature_da = temperature_da.isel(spatial_indexers)
    selected_surface = None
    if surface_temperature_k is not None:
        selected_surface = np.asarray(surface_temperature_k, dtype=float)
        expected_shape = tuple(temperature_da.sizes[dim] for dim in horizontal_dims)
        if selected_surface.shape != expected_shape:
            raise ValueError(
                "surface temperature shape does not match the selected OHC region: "
                f"{selected_surface.shape} != {expected_shape}"
            )
        temperature_da = temperature_da.isel({ocean_dim: slice(1, None)})
    depth_spatial_indexers = {
        dim: indexer
        for dim, indexer in spatial_indexers.items()
        if dim in depth_da.dims and depth_da.sizes[dim] > 1
    }
    depth_da = depth_da.isel(depth_spatial_indexers)
    if depth_reference_index is not None:
        # PWP OM_DEPTH is one horizontally shared vertical coordinate.  Select
        # an ocean point so uninitialized land values are never used.
        reference_y, reference_x = depth_reference_index
        selected_shape = tuple(temperature_da.sizes[dim] for dim in horizontal_dims)
        if not (
            0 <= reference_y < selected_shape[0]
            and 0 <= reference_x < selected_shape[1]
        ):
            raise ValueError(
                "depth reference index lies outside the selected OHC region"
            )
        reference_indexers = {}
        for dim, index in zip(horizontal_dims, depth_reference_index):
            if dim in depth_da.dims:
                reference_indexers[dim] = 0 if depth_da.sizes[dim] == 1 else index
        depth_da = depth_da.isel(reference_indexers)
    temperature_da = temperature_da.transpose(
        ocean_dim, *[dim for dim in temperature_da.dims if dim != ocean_dim]
    )
    if ocean_dim not in depth_da.dims:
        depth_ocean_dims = [dim for dim in depth_da.dims if "ocean_layer" in dim]
        if len(depth_ocean_dims) != 1:
            raise ValueError(
                f"could not identify OM_DEPTH ocean dimension: {depth_da.dims}"
            )
        depth_da = depth_da.rename({depth_ocean_dims[0]: ocean_dim})
    depth_da = depth_da.transpose(
        ocean_dim, *[dim for dim in depth_da.dims if dim != ocean_dim]
    )

    temperature = np.asarray(temperature_da.values, dtype=float)
    depth = np.asarray(depth_da.values, dtype=float)
    temp_units = str(ds["OM_TMP"].attrs.get("units", "")).lower()
    if "k" in temp_units or (
        "c" not in temp_units and np.nanmedian(temperature) > 100.0
    ):
        temperature = temperature - 273.15
    elif "c" not in temp_units:
        raise ValueError(f"unknown OM_TMP units: {temp_units!r}")
    if selected_surface is not None:
        temperature = np.concatenate(
            ((selected_surface - 273.15)[None, ...], temperature), axis=0
        )

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
    orientation = None
    for profile in depth.reshape(depth.shape[0], -1).T:
        if not np.isfinite(profile).all():
            continue
        differences = np.diff(profile)
        if np.all(differences > 0.0):
            orientation = 1
            break
        if np.all(differences < 0.0):
            orientation = -1
            break
    if orientation is None:
        raise ValueError("OM_DEPTH has no finite strictly monotonic ocean profile")
    if orientation < 0:
        depth = -depth
    return temperature, depth


def mask_bounding_slices(mask: np.ndarray) -> tuple[slice, slice]:
    """Return the smallest rectangular slices containing a nonempty 2-D mask."""
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 2 or not np.any(selected):
        raise ValueError("mask must be a nonempty 2-D array")
    rows, columns = np.where(selected)
    return (
        slice(int(rows.min()), int(rows.max()) + 1),
        slice(int(columns.min()), int(columns.max()) + 1),
    )


def find_unique_wrfout(root: Path, domain: str, valid_time: str) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    basename = f"wrfout_{domain}_{valid_time}"
    matches = sorted(path for path in root.rglob(f"{basename}*") if path.is_file())
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
    for j, i in zip(*np.where(mask)):
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
    static_grid_cache: dict[tuple[object, ...], StaticGrid] = {}
    mask_cache: dict[tuple[object, ...], np.ndarray] = {}

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
                    grid_signature = static_grid_signature(ds)
                    if grid_signature not in static_grid_cache:
                        static_grid_cache[grid_signature] = read_static_grid(ds)
                    static_grid = static_grid_cache[grid_signature]
                    if grid_signature not in mask_cache:
                        mask_cache[grid_signature] = tc_ocean_mask(
                            static_grid.lats,
                            static_grid.lons,
                            center_lat,
                            center_lon,
                            config.radius_km,
                            static_grid.ocean,
                        )
                    mask = mask_cache[grid_signature]
                    y_slice, x_slice = mask_bounding_slices(mask)
                    local_mask = mask[y_slice, x_slice]
                    state, _, _, _ = read_surface_state(
                        ds,
                        experiment.ocean_enabled,
                        static_grid=static_grid,
                        y_slice=y_slice,
                        x_slice=x_slice,
                    )
                    selected_state = select_surface_state(state, local_mask)
                    isftcflx = _physics_attribute(ds, "ISFTCFLX")
                    flux = revised_mm5_ocean_flux(
                        selected_state, SfclayOptions(isftcflx=isftcflx)
                    )
                    lh_values = flux.lh.ravel()
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
                        selected_rows, selected_columns = np.where(local_mask)
                        depth_reference_index = (
                            int(selected_rows[0]),
                            int(selected_columns[0]),
                        )
                        temperature_c, depth_m = read_ohc_inputs(
                            ds,
                            y_slice=y_slice,
                            x_slice=x_slice,
                            surface_temperature_k=state.surface_temperature_k,
                            depth_reference_index=depth_reference_index,
                        )
                        ohc_j_m2, finite_points = _ohc_on_mask(
                            temperature_c,
                            depth_m,
                            local_mask,
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


FILTER_MARKERS = {"EAKF": "o", "QCF_RHF": "s"}


def plot_member_comparison(
    frame: pd.DataFrame,
    value_column: str,
    output_path: Path,
    config: Config,
    ylabel: str,
    title: str,
) -> None:
    """Plot member points plus ensemble mean and sample-standard-deviation."""
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if frame.empty:
        raise ValueError(f"cannot plot empty {value_column} table")
    experiments = [
        experiment
        for experiment in config.experiments
        if experiment.name in set(frame["experiment"])
    ]
    if not experiments:
        raise ValueError("plot table contains no configured experiments")

    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=150)
    filter_offsets = np.linspace(-0.17, 0.17, len(config.filters))
    for exp_index, experiment in enumerate(experiments):
        for filter_index, filter_name in enumerate(config.filters):
            subset = frame[
                (frame["experiment"] == experiment.name)
                & (frame["filter"] == filter_name)
            ].sort_values("member")
            if subset.empty:
                raise ValueError(
                    f"no data for {experiment.name}/{filter_name} in {value_column}"
                )
            group_x = exp_index + filter_offsets[filter_index]
            jitter = np.linspace(-0.045, 0.045, len(subset))
            ax.scatter(
                group_x + jitter,
                subset[value_column],
                s=34,
                marker=FILTER_MARKERS.get(filter_name, "o"),
                facecolor=experiment.color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.82,
                zorder=3,
            )
            mean = float(subset["ensemble_mean"].iloc[0])
            std = float(subset["ensemble_std"].iloc[0])
            if not np.isfinite(std):
                std = 0.0
            ax.errorbar(
                group_x,
                mean,
                yerr=std,
                fmt="D",
                color="black",
                markerfacecolor="white",
                markersize=6,
                capsize=4,
                linewidth=1.5,
                zorder=4,
            )

    ax.set_xticks(np.arange(len(experiments)))
    ax.set_xticklabels([experiment.label for experiment in experiments])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    handles = [
        Line2D(
            [0],
            [0],
            marker=FILTER_MARKERS.get(filter_name, "o"),
            linestyle="none",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markersize=7,
            label=filter_name,
        )
        for filter_name in config.filters
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="-",
            color="black",
            markerfacecolor="white",
            markersize=6,
            label="Ensemble mean ± 1 SD",
        )
    )
    ax.legend(handles=handles, frameon=False, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    lh: pd.DataFrame,
    ohc: pd.DataFrame,
    config: Config,
) -> tuple[Path, Path, Path, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    lh_csv = config.output_dir / "initial_tc150_lh_members.csv"
    ohc_csv = config.output_dir / "initial_tc150_ohc_members.csv"
    lh_png = config.output_dir / "initial_tc150_lh.png"
    ohc_png = config.output_dir / "initial_tc150_ohc.png"
    lh.to_csv(lh_csv, index=False)
    ohc.to_csv(ohc_csv, index=False)
    plot_member_comparison(
        lh,
        "lh_mean_w_m2",
        lh_png,
        config,
        ylabel="Latent heat flux (W m$^{-2}$)",
        title="Initial TC 150-km ocean latent heat flux",
    )
    plot_member_comparison(
        ohc,
        "ohc26_mean_kj_cm2",
        ohc_png,
        config,
        ylabel="OHC26 (kJ cm$^{-2}$)",
        title="Initial TC 150-km ocean heat content",
    )
    return lh_csv, ohc_csv, lh_png, ohc_png


def main() -> None:
    lh, ohc = calculate_member_records(CONFIG)
    paths = write_outputs(lh, ohc, CONFIG)
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
