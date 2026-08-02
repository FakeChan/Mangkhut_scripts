"""Offline WRF 4.1 Revised-MM5 ocean surface-flux reconstruction.

The equations follow WRF v4.1 ``phys/module_sf_sfclayrev.F``.  This module is
deliberately independent of NetCDF I/O so the initial-state reconstruction can
be verified with small numerical fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


KARMAN = 0.4
GRAVITY = 9.81
RD = 287.0
R_OVER_CP = 287.0 / 1004.0
P0_PA = 100_000.0
EP1 = 0.608
EP2 = 0.622
SVP1_KPA = 0.6112
SVP2 = 17.67
SVP3_K = 29.65
SVPT0_K = 273.15
XLV_J_KG = 2.5e6
CZO = 0.0185
OZO = 1.59e-5


@dataclass(frozen=True)
class SurfaceState:
    air_temperature_k: np.ndarray
    surface_temperature_k: np.ndarray
    vapor_mixing_ratio: np.ndarray
    air_pressure_pa: np.ndarray
    surface_pressure_pa: np.ndarray
    height_agl_m: np.ndarray
    u_ms: np.ndarray
    v_ms: np.ndarray
    dx_m: float


@dataclass(frozen=True)
class SfclayOptions:
    isftcflx: int = 0
    max_iterations: int = 100
    tolerance: float = 1.0e-7
    latent_heat_j_kg: float = XLV_J_KG


@dataclass(frozen=True)
class SurfaceFluxResult:
    qfx: np.ndarray
    lh: np.ndarray
    friction_velocity_ms: np.ndarray
    momentum_roughness_m: np.ndarray
    moisture_roughness_m: np.ndarray
    bulk_richardson: np.ndarray
    inverse_obukhov_length: np.ndarray
    iterations: int


def saturation_mixing_ratio(
    temperature_k: np.ndarray, pressure_pa: np.ndarray
) -> np.ndarray:
    """Return WRF's saturation water-vapor mixing ratio in kg kg-1."""
    temperature = np.asarray(temperature_k, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    if np.any(~np.isfinite(temperature)) or np.any(~np.isfinite(pressure)):
        raise ValueError("temperature and pressure must be finite")
    if np.any(pressure <= 0.0):
        raise ValueError("pressure must be positive")
    vapor_pressure_kpa = SVP1_KPA * np.exp(
        SVP2 * (temperature - SVPT0_K) / (temperature - SVP3_K)
    )
    pressure_kpa = pressure / 1000.0
    if np.any(vapor_pressure_kpa >= pressure_kpa):
        raise ValueError("saturation vapor pressure must be below air pressure")
    return EP2 * vapor_pressure_kpa / (pressure_kpa - vapor_pressure_kpa)


def _psim_stable(zeta: np.ndarray | float) -> np.ndarray:
    zeta = np.asarray(zeta, dtype=float)
    return -6.1 * np.log(zeta + (1.0 + zeta**2.5) ** (1.0 / 2.5))


def _psih_stable(zeta: np.ndarray | float) -> np.ndarray:
    zeta = np.asarray(zeta, dtype=float)
    return -5.3 * np.log(zeta + (1.0 + zeta**1.1) ** (1.0 / 1.1))


def _psim_unstable(zeta: np.ndarray | float) -> np.ndarray:
    zeta = np.asarray(zeta, dtype=float)
    x = (1.0 - 16.0 * zeta) ** 0.25
    psim_k = (
        2.0 * np.log(0.5 * (1.0 + x))
        + np.log(0.5 * (1.0 + x * x))
        - 2.0 * np.arctan(x)
        + np.pi / 2.0
    )
    y = (1.0 - 10.0 * zeta) ** (1.0 / 3.0)
    psim_c = (
        1.5 * np.log((y * y + y + 1.0) / 3.0)
        - np.sqrt(3.0) * np.arctan((2.0 * y + 1.0) / np.sqrt(3.0))
        + np.pi / np.sqrt(3.0)
    )
    return (psim_k + zeta * zeta * psim_c) / (1.0 + zeta * zeta)


def _psih_unstable(zeta: np.ndarray | float) -> np.ndarray:
    zeta = np.asarray(zeta, dtype=float)
    y = np.sqrt(1.0 - 16.0 * zeta)
    psih_k = 2.0 * np.log((1.0 + y) / 2.0)
    yh = (1.0 - 34.0 * zeta) ** (1.0 / 3.0)
    psih_c = (
        1.5 * np.log((yh * yh + yh + 1.0) / 3.0)
        - np.sqrt(3.0) * np.arctan((2.0 * yh + 1.0) / np.sqrt(3.0))
        + np.pi / np.sqrt(3.0)
    )
    return (psih_k + zeta * zeta * psih_c) / (1.0 + zeta * zeta)


def _psi_m(zeta: np.ndarray | float) -> np.ndarray:
    zeta = np.asarray(zeta, dtype=float)
    result = np.empty_like(zeta)
    stable = zeta >= 0.0
    result[stable] = _psim_stable(zeta[stable])
    result[~stable] = _psim_unstable(zeta[~stable])
    return result


def _psi_h(zeta: np.ndarray | float) -> np.ndarray:
    zeta = np.asarray(zeta, dtype=float)
    result = np.empty_like(zeta)
    stable = zeta >= 0.0
    result[stable] = _psih_stable(zeta[stable])
    result[~stable] = _psih_unstable(zeta[~stable])
    return result


def _integrated_psi(
    z_over_l: np.ndarray,
    height_m: np.ndarray,
    roughness_m: np.ndarray,
    kind: str,
) -> np.ndarray:
    upper = z_over_l * (height_m + roughness_m) / height_m
    lower = z_over_l * roughness_m / height_m
    fn = _psi_m if kind == "momentum" else _psi_h
    return fn(upper) - fn(lower)


def _ri_residual(
    zol: np.ndarray | float,
    ri: np.ndarray | float,
    z: np.ndarray | float,
    z0: np.ndarray | float,
) -> np.ndarray:
    zol = np.asarray(zol, dtype=float)
    ri = np.asarray(ri, dtype=float)
    z = np.asarray(z, dtype=float)
    z0 = np.asarray(z0, dtype=float)
    zol = np.where(zol * ri < 0.0, 0.0, zol)
    correction_m = _integrated_psi(zol, z, z0, "momentum")
    correction_h = _integrated_psi(zol, z, z0, "heat")
    logz = np.log((z + z0) / z0)
    return zol * (logz - correction_h) / (logz - correction_m) ** 2 - ri


def _zol_from_ri(ri: np.ndarray, z: np.ndarray, z0: np.ndarray) -> np.ndarray:
    target, height, roughness = np.broadcast_arrays(
        np.clip(np.asarray(ri, dtype=float), -250.0, 250.0),
        np.asarray(z, dtype=float),
        np.asarray(z0, dtype=float),
    )
    result = np.zeros_like(target)
    active = np.abs(target) >= 1.0e-12
    unstable = target < 0.0
    lo = np.where(unstable, -5.0, 0.0)
    hi = np.where(unstable, 0.0, 5.0)
    flo = _ri_residual(lo, target, height, roughness)
    fhi = _ri_residual(hi, target, height, roughness)

    for _ in range(20):
        unbracketed = active & (flo * fhi > 0.0)
        if not np.any(unbracketed):
            break
        expand_lo = unbracketed & unstable
        expand_hi = unbracketed & ~unstable
        lo = np.where(expand_lo, lo * 2.0, lo)
        hi = np.where(expand_hi, hi * 2.0, hi)
        flo = np.where(
            expand_lo,
            _ri_residual(lo, target, height, roughness),
            flo,
        )
        fhi = np.where(
            expand_hi,
            _ri_residual(hi, target, height, roughness),
            fhi,
        )

    unbracketed = active & (flo * fhi > 0.0)
    if np.any(unbracketed):
        failed_target = float(target[unbracketed][0])
        raise RuntimeError(f"could not bracket z/L for Ri={failed_target:g}")

    working = active.copy()
    mid = np.zeros_like(target)
    for _ in range(80):
        candidate = 0.5 * (lo + hi)
        fmid = _ri_residual(candidate, target, height, roughness)
        converged = working & (
            (np.abs(fmid) < 1.0e-10) | (np.abs(hi - lo) < 0.01)
        )
        mid = np.where(working, candidate, mid)
        remaining = working & ~converged
        if not np.any(remaining):
            break
        replace_hi = remaining & (flo * fmid <= 0.0)
        replace_lo = remaining & ~replace_hi
        hi = np.where(replace_hi, candidate, hi)
        fhi = np.where(replace_hi, fmid, fhi)
        lo = np.where(replace_lo, candidate, lo)
        flo = np.where(replace_lo, fmid, flo)
        working = remaining
    result = np.where(active, mid, result)
    return result


def _validate_state(state: SurfaceState) -> tuple[int, int]:
    arrays = {
        name: np.asarray(value, dtype=float)
        for name, value in state.__dict__.items()
        if name != "dx_m"
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"surface-state arrays must have one shape, got {shapes}")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError(f"surface-state arrays must be 2-D, got shape={shape}")
    if any(np.any(~np.isfinite(array)) for array in arrays.values()):
        raise ValueError("surface-state arrays must be finite")
    if np.any(arrays["air_pressure_pa"] <= 0.0) or np.any(
        arrays["surface_pressure_pa"] <= 0.0
    ):
        raise ValueError("air and surface pressure must be positive")
    if np.any(arrays["height_agl_m"] <= 0.0):
        raise ValueError("lowest-model-level height must be positive")
    if state.dx_m <= 0.0:
        raise ValueError("dx_m must be positive")
    return shape


def _momentum_roughness(
    ustar: np.ndarray, isftcflx: int
) -> np.ndarray:
    safe_ustar = np.maximum(ustar, 0.01)
    if isftcflx == 0:
        z0 = CZO * safe_ustar**2 / GRAVITY + 0.11 * 1.5e-5 / safe_ustar
    else:
        weight = np.minimum((safe_ustar / 1.06) ** 0.3, 1.0)
        z1 = 0.011 * safe_ustar**2 / GRAVITY + OZO
        z2 = 10.0 * np.exp(-9.5 * safe_ustar ** (-1.0 / 3.0))
        z2 += 0.11 * 1.5e-5 / safe_ustar
        z0 = (1.0 - weight) * z1 + weight * z2
    return np.clip(z0, 1.27e-7, 2.85e-3)


def _moisture_roughness(
    ustar: np.ndarray,
    z0m: np.ndarray,
    air_temperature_k: np.ndarray,
    isftcflx: int,
) -> np.ndarray:
    if isftcflx == 1:
        return np.full_like(ustar, 1.0e-4)
    viscosity = (1.32 + 0.009 * (air_temperature_k - 273.15)) * 1.0e-5
    reynolds = np.maximum(ustar * z0m / viscosity, 1.0e-12)
    if isftcflx == 2:
        gz0ozq = 0.4 * (7.3 * reynolds**0.25 * np.sqrt(0.60) - 5.0)
        return np.clip(z0m / np.exp(gz0ozq), 2.0e-9, 1.0e-2)
    return np.clip(5.5e-5 * reynolds ** (-0.60), 2.0e-9, 1.0e-4)


def revised_mm5_ocean_flux(
    state: SurfaceState, options: SfclayOptions = SfclayOptions()
) -> SurfaceFluxResult:
    """Reconstruct instantaneous upward ocean moisture and latent heat flux."""
    _validate_state(state)
    if options.isftcflx not in {0, 1, 2}:
        raise ValueError("isftcflx must be 0, 1, or 2")
    if options.max_iterations < 1 or options.tolerance <= 0.0:
        raise ValueError("iteration controls must be positive")

    ta = np.asarray(state.air_temperature_k, dtype=float)
    ts = np.asarray(state.surface_temperature_k, dtype=float)
    qa = np.asarray(state.vapor_mixing_ratio, dtype=float)
    pa = np.asarray(state.air_pressure_pa, dtype=float)
    ps = np.asarray(state.surface_pressure_pa, dtype=float)
    z = np.asarray(state.height_agl_m, dtype=float)
    u = np.asarray(state.u_ms, dtype=float)
    v = np.asarray(state.v_ms, dtype=float)

    qs = saturation_mixing_ratio(ts, ps)
    theta_a = ta * (P0_PA / pa) ** R_OVER_CP
    theta_s = ts * (P0_PA / ps) ** R_OVER_CP
    theta_v_a = theta_a * (1.0 + EP1 * qa)
    theta_v_s = theta_s * (1.0 + EP1 * qs)
    dtheta_v = theta_v_a - theta_v_s

    wind = np.sqrt(u * u + v * v)
    convective = np.sqrt(np.maximum(-dtheta_v, 0.0))
    subgrid = 0.32 * max(state.dx_m / 5000.0 - 1.0, 0.0) ** (1.0 / 3.0)
    effective_wind = np.maximum(
        np.sqrt(wind * wind + convective * convective + subgrid * subgrid),
        0.1,
    )
    bulk_ri = GRAVITY / theta_a * z * dtheta_v / effective_wind**2

    ustar = np.maximum(KARMAN * effective_wind / np.log((z + 1.0e-4) / 1.0e-4), 0.001)
    z0m = _momentum_roughness(ustar, options.isftcflx)
    iterations = options.max_iterations
    for iteration in range(1, options.max_iterations + 1):
        zol = _zol_from_ri(bulk_ri, z, z0m)
        psi_m = _integrated_psi(zol, z, z0m, "momentum")
        denom_m = np.maximum(np.log((z + z0m) / z0m) - psi_m, 1.0e-6)
        target = np.maximum(KARMAN * effective_wind / denom_m, 0.001)
        new_ustar = 0.5 * ustar + 0.5 * target
        new_z0m = _momentum_roughness(new_ustar, options.isftcflx)
        change = max(
            float(np.max(np.abs(new_ustar - ustar))),
            float(np.max(np.abs(new_z0m - z0m))),
        )
        ustar, z0m = new_ustar, new_z0m
        if change <= options.tolerance:
            iterations = iteration
            break
    else:
        raise RuntimeError(
            f"Revised-MM5 surface solve failed to converge at {ustar.size} grid points"
        )

    zol = _zol_from_ri(bulk_ri, z, z0m)
    z0q = _moisture_roughness(ustar, z0m, ta, options.isftcflx)
    psi_q = _integrated_psi(zol, z, z0q, "heat")
    denom_q = np.maximum(np.log((z + z0q) / z0q) - psi_q, 1.0e-6)
    virtual_temperature = ta * (1.0 + EP1 * qa)
    density = ps / (RD * virtual_temperature)
    flqc = density * ustar * KARMAN / denom_q
    qfx = np.maximum(flqc * (qs - qa), 0.0)
    lh = options.latent_heat_j_kg * qfx

    return SurfaceFluxResult(
        qfx=qfx,
        lh=lh,
        friction_velocity_ms=ustar,
        momentum_roughness_m=z0m,
        moisture_roughness_m=z0q,
        bulk_richardson=bulk_ri,
        inverse_obukhov_length=zol / z,
        iterations=iterations,
    )
