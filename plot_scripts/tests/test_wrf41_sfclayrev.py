from __future__ import annotations

import unittest

import numpy as np

from plot_scripts.wrf41_sfclayrev import (
    SfclayOptions,
    SurfaceState,
    revised_mm5_ocean_flux,
    saturation_mixing_ratio,
)


def uniform_surface_state(
    *,
    air_temperature_k: float = 299.0,
    surface_temperature_k: float = 301.0,
    vapor_mixing_ratio: float = 0.018,
    u_ms: float = 8.0,
    v_ms: float = 0.0,
) -> SurfaceState:
    shape = (1, 1)

    def field(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=float)

    return SurfaceState(
        air_temperature_k=field(air_temperature_k),
        surface_temperature_k=field(surface_temperature_k),
        vapor_mixing_ratio=field(vapor_mixing_ratio),
        air_pressure_pa=field(95_000.0),
        surface_pressure_pa=field(100_000.0),
        height_agl_m=field(25.0),
        u_ms=field(u_ms),
        v_ms=field(v_ms),
        pbl_height_m=field(800.0),
        dx_m=1_500.0,
    )


class Wrf41SfclayrevTests(unittest.TestCase):
    def test_saturation_mixing_ratio_matches_wrf_expression(self):
        got = saturation_mixing_ratio(np.array([300.0]), np.array([100_000.0]))
        self.assertAlmostEqual(got[0], 0.02279024, places=7)

    def test_warm_ocean_reconstructs_lh_when_stored_flux_is_zero(self):
        state = uniform_surface_state()
        result = revised_mm5_ocean_flux(state, SfclayOptions(isftcflx=0))
        self.assertGreater(result.qfx[0, 0], 0.0)
        self.assertAlmostEqual(result.lh[0, 0], 2.5e6 * result.qfx[0, 0])

    def test_stable_surface_has_positive_bulk_richardson_number(self):
        state = uniform_surface_state(surface_temperature_k=297.0)
        result = revised_mm5_ocean_flux(state, SfclayOptions(isftcflx=0))
        self.assertGreater(result.bulk_richardson[0, 0], 0.0)
        self.assertTrue(np.isfinite(result.inverse_obukhov_length).all())

    def test_unstable_surface_has_negative_bulk_richardson_number(self):
        state = uniform_surface_state(surface_temperature_k=303.0)
        result = revised_mm5_ocean_flux(state, SfclayOptions(isftcflx=0))
        self.assertLess(result.bulk_richardson[0, 0], 0.0)
        self.assertTrue(np.isfinite(result.inverse_obukhov_length).all())

    def test_isftcflx_one_uses_fixed_moisture_roughness(self):
        result = revised_mm5_ocean_flux(
            uniform_surface_state(), SfclayOptions(isftcflx=1)
        )
        self.assertAlmostEqual(result.moisture_roughness_m[0, 0], 1.0e-4)

    def test_isftcflx_two_produces_finite_garratt_roughness(self):
        result = revised_mm5_ocean_flux(
            uniform_surface_state(), SfclayOptions(isftcflx=2)
        )
        self.assertGreater(result.moisture_roughness_m[0, 0], 0.0)
        self.assertTrue(np.isfinite(result.moisture_roughness_m).all())

    def test_condensation_is_clipped_to_zero(self):
        state = uniform_surface_state(
            surface_temperature_k=296.0,
            vapor_mixing_ratio=0.024,
        )
        result = revised_mm5_ocean_flux(state, SfclayOptions(isftcflx=0))
        self.assertEqual(result.qfx[0, 0], 0.0)
        self.assertEqual(result.lh[0, 0], 0.0)

    def test_shape_mismatch_is_rejected(self):
        state = uniform_surface_state()
        bad = SurfaceState(
            **{
                **state.__dict__,
                "u_ms": np.ones((2, 1)),
            }
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            revised_mm5_ocean_flux(bad, SfclayOptions())


if __name__ == "__main__":
    unittest.main()
