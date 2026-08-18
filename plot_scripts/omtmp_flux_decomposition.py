"""Direct WRF 4.1 ocean flux response with a fixed atmospheric state."""

from __future__ import annotations

import numpy as np


try:
    from wrf41_sfclayrev import (
        EP1,
        KARMAN,
        P0_PA,
        RD,
        R_OVER_CP,
        SfclayOptions,
        SurfaceState,
        _integrated_psi,
        revised_mm5_ocean_flux,
    )
except ImportError:
    from plot_scripts.wrf41_sfclayrev import (
        EP1,
        KARMAN,
        P0_PA,
        RD,
        R_OVER_CP,
        SfclayOptions,
        SurfaceState,
        _integrated_psi,
        revised_mm5_ocean_flux,
    )


def reconstruct_ocean_fluxes(
    *,
    air_temperature_k,
    surface_temperature_k,
    vapor_mixing_ratio,
    air_pressure_pa,
    surface_pressure_pa,
    height_agl_m,
    u_ms,
    v_ms,
    initial_friction_velocity_ms,
    initial_momentum_roughness_m,
    dx_m,
    isftcflx=0,
):
    """Return QFX, LH and HFX for the supplied ocean temperature."""
    state = SurfaceState(
        air_temperature_k=np.asarray(air_temperature_k, dtype=float),
        surface_temperature_k=np.asarray(surface_temperature_k, dtype=float),
        vapor_mixing_ratio=np.asarray(vapor_mixing_ratio, dtype=float),
        air_pressure_pa=np.asarray(air_pressure_pa, dtype=float),
        surface_pressure_pa=np.asarray(surface_pressure_pa, dtype=float),
        height_agl_m=np.asarray(height_agl_m, dtype=float),
        u_ms=np.asarray(u_ms, dtype=float),
        v_ms=np.asarray(v_ms, dtype=float),
        initial_friction_velocity_ms=np.asarray(initial_friction_velocity_ms, dtype=float),
        initial_momentum_roughness_m=np.asarray(initial_momentum_roughness_m, dtype=float),
        dx_m=float(dx_m),
    )
    moisture = revised_mm5_ocean_flux(state, SfclayOptions(isftcflx=isftcflx))

    ta = state.air_temperature_k
    ts = state.surface_temperature_k
    qa = state.vapor_mixing_ratio
    pa = state.air_pressure_pa
    ps = state.surface_pressure_pa
    z = state.height_agl_m
    theta_air = ta * (P0_PA / pa) ** R_OVER_CP
    theta_surface = ts * (P0_PA / ps) ** R_OVER_CP
    z_over_l = moisture.inverse_obukhov_length * z
    heat_correction = _integrated_psi(
        z_over_l,
        z,
        moisture.moisture_roughness_m,
        "heat",
    )
    heat_denominator = (
        np.log((z + moisture.moisture_roughness_m) / moisture.moisture_roughness_m)
        - heat_correction
    )
    virtual_temperature = ta * (1.0 + EP1 * qa)
    density = ps / (RD * virtual_temperature)
    moist_cp = 1004.0 * (1.0 + 0.8 * qa)
    flhc = (
        moist_cp
        * density
        * moisture.friction_velocity_ms
        * KARMAN
        / heat_denominator
    )
    hfx = np.maximum(flhc * (theta_surface - theta_air), -250.0)
    return {
        "qfx": moisture.qfx,
        "lh": moisture.lh,
        "hfx": hfx,
        "ust": moisture.friction_velocity_ms,
        "bulk_richardson": moisture.bulk_richardson,
    }
