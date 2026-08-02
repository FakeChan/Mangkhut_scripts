"""Diagnose initial TC-region latent heat flux and OHC26.

User-editable experiment paths and plotting settings are defined near the top
of this module.  The numerical helpers are kept import-safe for unit testing.
"""

from __future__ import annotations

import numpy as np


EARTH_RADIUS_KM = 6371.0


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
