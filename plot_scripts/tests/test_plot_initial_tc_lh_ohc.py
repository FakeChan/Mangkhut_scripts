from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

from plot_scripts import plot_initial_tc_lh_ohc as diag


class RecordingBackendArray(BackendArray):
    """Lazy in-memory array that records the slices requested by xarray."""

    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values)
        self.requested_keys: list[tuple[object, ...]] = []

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key):
        self.requested_keys.append(key)
        return self.values[key]


def lazy_recording_data_array(
    values: np.ndarray, dims: tuple[str, ...]
) -> tuple[xr.DataArray, RecordingBackendArray]:
    backend = RecordingBackendArray(values)
    data = xr.DataArray(indexing.LazilyIndexedArray(backend), dims=dims)
    return data, backend


class TcMaskTests(unittest.TestCase):
    def test_tc_mask_includes_boundary_and_excludes_land(self):
        lats = np.array([[0.0, 0.0, 0.0]])
        lons = np.array([[0.0, 150.0 / 111.195, 2.0]])
        ocean = np.array([[True, True, False]])
        mask = diag.tc_ocean_mask(lats, lons, 0.0, 0.0, 150.0, ocean)
        self.assertEqual(mask.tolist(), [[True, True, False]])

    def test_tc_mask_rejects_empty_ocean_selection(self):
        with self.assertRaisesRegex(ValueError, "ocean"):
            diag.tc_ocean_mask(
                np.array([[10.0]]),
                np.array([[10.0]]),
                0.0,
                0.0,
                150.0,
                np.array([[True]]),
            )


class Ohc26Tests(unittest.TestCase):
    def test_ohc26_integrates_partial_crossing_interval(self):
        got = diag.ohc26_profile(
            np.array([29.0, 27.0, 25.0]),
            np.array([0.0, 10.0, 20.0]),
            rho=1025.0,
            cp=3985.0,
        )
        self.assertAlmostEqual(got, 1025.0 * 3985.0 * 22.5)

    def test_ohc26_is_zero_for_cool_surface(self):
        got = diag.ohc26_profile(
            np.array([25.5, 24.0]),
            np.array([0.0, 10.0]),
            rho=1025.0,
            cp=3985.0,
        )
        self.assertEqual(got, 0.0)

    def test_ohc26_rejects_missing_crossing(self):
        with self.assertRaisesRegex(ValueError, "26"):
            diag.ohc26_profile(
                np.array([29.0, 28.0]),
                np.array([0.0, 10.0]),
                rho=1025.0,
                cp=3985.0,
            )

    def test_ohc26_rejects_nonmonotonic_depth(self):
        with self.assertRaisesRegex(ValueError, "monotonic"):
            diag.ohc26_profile(
                np.array([29.0, 27.0, 25.0]),
                np.array([0.0, 20.0, 10.0]),
                rho=1025.0,
                cp=3985.0,
            )

    def test_ohc26_field_calculates_each_profile(self):
        temperature = np.array(
            [
                [[29.0, 25.0]],
                [[27.0, 24.0]],
                [[25.0, 23.0]],
            ]
        )
        got = diag.ohc26_field(
            temperature,
            np.array([0.0, 10.0, 20.0]),
            rho=1025.0,
            cp=3985.0,
        )
        self.assertAlmostEqual(got[0, 0], 1025.0 * 3985.0 * 22.5)
        self.assertEqual(got[0, 1], 0.0)


def synthetic_member_dataset(include_ocean: bool = True) -> xr.Dataset:
    ny, nx = 2, 3
    coords = {
        "Time": [0],
        "bottom_top": [0],
        "bottom_top_stag": [0, 1],
        "south_north": np.arange(ny),
        "south_north_stag": np.arange(ny + 1),
        "west_east": np.arange(nx),
        "west_east_stag": np.arange(nx + 1),
    }
    lat = np.array([[19.9, 19.9, 19.9], [20.1, 20.1, 20.1]])
    lon = np.array([[119.9, 120.0, 120.1], [119.9, 120.0, 120.1]])
    ds = xr.Dataset(
        {
            "XLAT": (("Time", "south_north", "west_east"), lat[None]),
            "XLONG": (("Time", "south_north", "west_east"), lon[None]),
            "U": (
                ("Time", "bottom_top", "south_north", "west_east_stag"),
                np.full((1, 1, ny, nx + 1), 8.0),
            ),
            "V": (
                ("Time", "bottom_top", "south_north_stag", "west_east"),
                np.zeros((1, 1, ny + 1, nx)),
            ),
            "T": (
                ("Time", "bottom_top", "south_north", "west_east"),
                np.full((1, 1, ny, nx), 4.0),
            ),
            "P": (
                ("Time", "bottom_top", "south_north", "west_east"),
                np.full((1, 1, ny, nx), 5_000.0),
            ),
            "PB": (
                ("Time", "bottom_top", "south_north", "west_east"),
                np.full((1, 1, ny, nx), 90_000.0),
            ),
            "QVAPOR": (
                ("Time", "bottom_top", "south_north", "west_east"),
                np.full((1, 1, ny, nx), 0.018),
            ),
            "PH": (
                ("Time", "bottom_top_stag", "south_north", "west_east"),
                np.stack(
                    [np.zeros((ny, nx)), np.full((ny, nx), 9.81 * 50.0)]
                )[None],
            ),
            "PHB": (
                ("Time", "bottom_top_stag", "south_north", "west_east"),
                np.zeros((1, 2, ny, nx)),
            ),
            "HGT": (("Time", "south_north", "west_east"), np.zeros((1, ny, nx))),
            "PSFC": (
                ("Time", "south_north", "west_east"),
                np.full((1, ny, nx), 100_000.0),
            ),
            "TSK": (
                ("Time", "south_north", "west_east"),
                np.full((1, ny, nx), 301.0),
            ),
            "PBLH": (
                ("Time", "south_north", "west_east"),
                np.full((1, ny, nx), 800.0),
            ),
            "XLAND": (
                ("Time", "south_north", "west_east"),
                np.full((1, ny, nx), 2.0),
            ),
            "QFX": (("Time", "south_north", "west_east"), np.zeros((1, ny, nx))),
            "LH": (("Time", "south_north", "west_east"), np.zeros((1, ny, nx))),
            "UST": (
                ("Time", "south_north", "west_east"),
                np.full((1, ny, nx), 1.0e-4),
            ),
            "XTIME": (("Time",), np.array([0.0])),
        },
        coords=coords,
        attrs={
            "DX": 1500.0,
            "SF_SFCLAY_PHYSICS": 1,
            "ISFTCFLX": 0,
            "ISFFLX": 1,
        },
    )
    if include_ocean:
        ds = ds.assign_coords(ocean_layer_stag=np.arange(3))
        ocean_temperature = np.empty((1, 3, ny, nx))
        ocean_temperature[:, 0] = 302.15
        ocean_temperature[:, 1] = 300.15
        ocean_temperature[:, 2] = 298.15
        ds["OM_TMP"] = (
            ("Time", "ocean_layer_stag", "south_north", "west_east"),
            ocean_temperature,
            {"units": "K"},
        )
        ds["OM_DEPTH"] = (
            ("Time", "ocean_layer_stag", "south_north", "west_east"),
            np.broadcast_to(
                np.array([0.0, 10.0, 20.0])[None, :, None, None],
                (1, 3, ny, nx),
            ),
            {"units": "m"},
        )
    return ds


class WrfReaderAndWorkflowTests(unittest.TestCase):
    def test_initial_zero_stored_flux_is_reconstructed(self):
        ds = synthetic_member_dataset(include_ocean=True)
        mask = np.array([[True, True, False], [False, False, False]])

        result = diag.diagnose_lh_on_mask(ds, mask, use_ocean_sst=True)

        self.assertEqual(result.source, "reconstructed_initial")
        self.assertFalse(result.stored_flux_used)
        self.assertEqual(result.xtime_minutes, 0.0)
        self.assertGreater(result.mean_w_m2, 0.0)
        self.assertEqual(result.finite_points, 2)

    def test_initial_nonzero_lh_uses_stored_lh(self):
        ds = synthetic_member_dataset(include_ocean=True)
        ds["LH"][:] = np.array([[[100.0, 110.0, 120.0], [130.0, 140.0, 150.0]]])
        mask = np.array([[True, True, False], [False, False, False]])

        result = diag.diagnose_lh_on_mask(ds, mask, use_ocean_sst=True)

        self.assertEqual(result.source, "stored_LH")
        self.assertTrue(result.stored_flux_used)
        self.assertAlmostEqual(result.mean_w_m2, 105.0)

    def test_initial_nonzero_qfx_overrides_zero_lh(self):
        ds = synthetic_member_dataset(include_ocean=True)
        ds["QFX"][:] = np.array(
            [[[4.0e-5, 8.0e-5, 0.0], [0.0, 0.0, 0.0]]]
        )
        mask = np.array([[True, True, False], [False, False, False]])

        with self.assertWarnsRegex(RuntimeWarning, "LH.*zero.*QFX.*nonzero"):
            result = diag.diagnose_lh_on_mask(ds, mask, use_ocean_sst=True)

        self.assertEqual(result.source, "derived_QFX")
        self.assertTrue(result.stored_flux_used)
        self.assertAlmostEqual(result.mean_w_m2, 150.0)

    def test_integrated_output_uses_stored_lh(self):
        ds = synthetic_member_dataset(include_ocean=True)
        ds["XTIME"][:] = 5.0
        ds["LH"][:] = 175.0
        ds["QFX"][:] = 175.0 / diag.XLV_J_KG
        mask = np.ones((2, 3), dtype=bool)

        result = diag.diagnose_lh_on_mask(ds, mask, use_ocean_sst=True)

        self.assertEqual(result.source, "stored_LH")
        self.assertTrue(result.stored_flux_used)
        self.assertEqual(result.xtime_minutes, 5.0)
        self.assertAlmostEqual(result.mean_w_m2, 175.0)

    def test_integrated_output_falls_back_to_qfx(self):
        ds = synthetic_member_dataset(include_ocean=True).drop_vars("LH")
        ds["XTIME"][:] = 5.0
        ds["QFX"][:] = 4.0e-5
        mask = np.ones((2, 3), dtype=bool)

        result = diag.diagnose_lh_on_mask(ds, mask, use_ocean_sst=True)

        self.assertEqual(result.source, "derived_QFX")
        self.assertAlmostEqual(result.mean_w_m2, 100.0)

    def test_integrated_zero_flux_is_preserved_with_warning(self):
        ds = synthetic_member_dataset(include_ocean=True)
        ds["XTIME"][:] = 5.0
        mask = np.ones((2, 3), dtype=bool)

        with self.assertWarnsRegex(RuntimeWarning, "XTIME > 0.*zero"):
            result = diag.diagnose_lh_on_mask(ds, mask, use_ocean_sst=True)

        self.assertEqual(result.source, "stored_LH")
        self.assertTrue(result.stored_flux_used)
        self.assertEqual(result.mean_w_m2, 0.0)

    def test_integrated_output_requires_stored_flux(self):
        ds = synthetic_member_dataset(include_ocean=True).drop_vars(["LH", "QFX"])
        ds["XTIME"][:] = 5.0
        mask = np.ones((2, 3), dtype=bool)

        with self.assertRaisesRegex(KeyError, "XTIME > 0.*LH or QFX"):
            diag.diagnose_lh_on_mask(ds, mask, use_ocean_sst=True)

    def test_lh_diagnosis_requires_one_finite_xtime(self):
        mask = np.ones((2, 3), dtype=bool)
        for ds in (
            synthetic_member_dataset(include_ocean=True).drop_vars("XTIME"),
            synthetic_member_dataset(include_ocean=True).assign(
                XTIME=(("Time",), [np.nan])
            ),
        ):
            with self.subTest(variables=tuple(ds.data_vars)):
                with self.assertRaisesRegex((KeyError, ValueError), "XTIME"):
                    diag.diagnose_lh_on_mask(ds, mask, use_ocean_sst=True)

    def test_stored_lh_warns_when_nonzero_qfx_disagrees(self):
        ds = synthetic_member_dataset(include_ocean=True)
        ds["LH"][:] = 100.0
        ds["QFX"][:] = 1.0e-3
        mask = np.ones((2, 3), dtype=bool)

        with self.assertWarnsRegex(RuntimeWarning, "LH.*QFX"):
            result = diag.diagnose_lh_on_mask(ds, mask, use_ocean_sst=True)

        self.assertEqual(result.source, "stored_LH")
        self.assertAlmostEqual(result.mean_w_m2, 100.0)

    def test_nr_reference_uses_nr_native_mask_for_lh_and_ohc(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(
                diag.CONFIG,
                nr_root=root,
                nr_domain="d03",
                valid_time="2018-09-10_00:00:00",
                radius_km=150.0,
            )
            nr_path = root / f"wrfout_d03_{config.valid_time}"
            nr = synthetic_member_dataset(include_ocean=True)
            nr["LH"][:] = 175.0
            nr["QFX"][:] = 175.0 / 2.5e6
            nr.to_netcdf(nr_path)

            reference = diag.calculate_nr_reference(
                config,
                nr_path,
                center_lat=20.0,
                center_lon=120.0,
            )

        self.assertEqual(len(reference), 1)
        row = reference.iloc[0]
        self.assertEqual(row["input_path"], str(nr_path))
        self.assertEqual(row["lh_source"], "stored_LH")
        self.assertTrue(row["stored_flux_used"])
        self.assertEqual(row["xtime_minutes"], 0.0)
        self.assertEqual(row["tc_ocean_points"], 6)
        self.assertEqual(row["lh_finite_points"], 6)
        self.assertEqual(row["ohc_finite_points"], 6)
        self.assertAlmostEqual(row["lh_mean_w_m2"], 175.0)
        expected_ohc = 1025.0 * 3985.0 * 22.5 / 1.0e7
        self.assertAlmostEqual(row["ohc26_mean_kj_cm2"], expected_ohc)

    def test_surface_solver_excludes_unselected_invalid_land_point(self):
        def field(ocean_value: float, land_value: float) -> np.ndarray:
            return np.array([[ocean_value, land_value]], dtype=float)

        state = diag.SurfaceState(
            air_temperature_k=field(299.0, 260.0),
            surface_temperature_k=field(301.0, 280.0),
            vapor_mixing_ratio=field(0.018, 0.001),
            air_pressure_pa=field(95_000.0, 95_000.0),
            surface_pressure_pa=field(100_000.0, 100_000.0),
            height_agl_m=field(25.0, 100.0),
            u_ms=field(8.0, 10.0),
            v_ms=field(0.0, 0.0),
            initial_friction_velocity_ms=field(1.0e-4, 1.0e-4),
            initial_momentum_roughness_m=field(1.0e-4, 1.0e-4),
            dx_m=1_500.0,
        )

        selected = diag.select_surface_state(
            state, np.array([[True, False]])
        )
        result = diag.revised_mm5_ocean_flux(selected, diag.SfclayOptions())

        self.assertEqual(selected.air_temperature_k.shape, (1, 1))
        self.assertEqual(selected.air_temperature_k[0, 0], 299.0)
        self.assertTrue(np.isfinite(result.lh).all())

    def test_surface_reader_only_loads_two_lowest_geopotential_levels(self):
        ds = synthetic_member_dataset(include_ocean=False)
        dims = ds["PH"].dims
        ph_values = np.concatenate(
            [
                ds["PH"].values,
                np.full((1, 3, 2, 3), 9.81 * 9999.0),
            ],
            axis=1,
        )
        phb_values = np.zeros_like(ph_values)
        ph, ph_backend = lazy_recording_data_array(ph_values, dims)
        phb, phb_backend = lazy_recording_data_array(phb_values, dims)
        ds = ds.drop_vars(["PH", "PHB", "bottom_top_stag"]).assign(
            PH=ph, PHB=phb
        )

        state, _, _, _ = diag.read_surface_state(ds, use_ocean_sst=False)

        np.testing.assert_allclose(state.height_agl_m, 25.0)
        expected = (0, slice(0, 2, 1), slice(None), slice(None))
        self.assertTrue(ph_backend.requested_keys)
        self.assertTrue(phb_backend.requested_keys)
        self.assertTrue(all(key == expected for key in ph_backend.requested_keys))
        self.assertTrue(all(key == expected for key in phb_backend.requested_keys))

    def test_reader_destaggers_wind_and_ignores_zero_stored_flux(self):
        ds = synthetic_member_dataset(include_ocean=True)
        state, lats, lons, ocean = diag.read_surface_state(ds, use_ocean_sst=True)
        np.testing.assert_allclose(state.u_ms, 8.0)
        np.testing.assert_allclose(state.v_ms, 0.0)
        np.testing.assert_allclose(state.height_agl_m, 25.0)
        np.testing.assert_allclose(state.initial_friction_velocity_ms, 1.0e-4)
        np.testing.assert_allclose(state.initial_momentum_roughness_m, 1.0e-4)
        self.assertEqual(lats.shape, (2, 3))
        self.assertTrue(ocean.all())
        result = diag.calculate_lh_field(ds, use_ocean_sst=True)
        self.assertTrue((result.lh > 0.0).all())

    def test_surface_reader_rejects_missing_znt_after_initial_time(self):
        ds = synthetic_member_dataset(include_ocean=False)
        ds["XTIME"][:] = 5.0

        with self.assertRaisesRegex(KeyError, "ZNT.*XTIME=0"):
            diag.read_surface_state(ds, use_ocean_sst=False)

    def test_surface_reader_does_not_require_unused_pblh(self):
        ds = synthetic_member_dataset(include_ocean=False).drop_vars("PBLH")

        state, _, _, _ = diag.read_surface_state(ds, use_ocean_sst=False)

        self.assertFalse(hasattr(state, "pbl_height_m"))

    def test_surface_reader_reuses_matching_static_grid_without_variable_io(self):
        first = synthetic_member_dataset(include_ocean=False)
        static_grid = diag.read_static_grid(first)
        second = synthetic_member_dataset(include_ocean=False)
        backends = {}
        replacements = {}
        for name in ("XLAT", "XLONG", "XLAND", "HGT"):
            data, backend = lazy_recording_data_array(
                second[name].values.copy(), second[name].dims
            )
            replacements[name] = data
            backends[name] = backend
        second = second.drop_vars(list(replacements)).assign(replacements)

        state, lats, lons, ocean = diag.read_surface_state(
            second, use_ocean_sst=False, static_grid=static_grid
        )

        np.testing.assert_allclose(state.height_agl_m, 25.0)
        np.testing.assert_allclose(lats, static_grid.lats)
        np.testing.assert_allclose(lons, static_grid.lons)
        np.testing.assert_array_equal(ocean, static_grid.ocean)
        self.assertTrue(
            all(not backend.requested_keys for backend in backends.values())
        )

    def test_surface_reader_only_loads_requested_spatial_box(self):
        ds = synthetic_member_dataset(include_ocean=False)
        static_grid = diag.read_static_grid(ds)
        backends = {}
        replacements = {}
        for name in (
            "U",
            "V",
            "T",
            "P",
            "PB",
            "QVAPOR",
            "PH",
            "PHB",
            "PSFC",
            "TSK",
            "UST",
        ):
            data, backend = lazy_recording_data_array(
                ds[name].values.copy(), ds[name].dims
            )
            replacements[name] = data
            backends[name] = backend
        ds = ds.drop_vars(list(replacements)).assign(replacements)

        state, lats, _, _ = diag.read_surface_state(
            ds,
            use_ocean_sst=False,
            static_grid=static_grid,
            y_slice=slice(0, 1),
            x_slice=slice(1, 3),
        )

        self.assertEqual(state.air_temperature_k.shape, (1, 2))
        self.assertEqual(lats.shape, (1, 2))
        mass_request = (0, 0, slice(0, 1, 1), slice(1, 3, 1))
        for name in ("T", "P", "PB", "QVAPOR"):
            self.assertTrue(
                all(key == mass_request for key in backends[name].requested_keys)
            )
        surface_request = (0, slice(0, 1, 1), slice(1, 3, 1))
        for name in ("PSFC", "TSK", "UST"):
            self.assertTrue(
                all(key == surface_request for key in backends[name].requested_keys)
            )
        self.assertTrue(
            all(
                key == (0, 0, slice(0, 1, 1), slice(1, 4, 1))
                for key in backends["U"].requested_keys
            )
        )
        self.assertTrue(
            all(
                key == (0, 0, slice(0, 2, 1), slice(1, 3, 1))
                for key in backends["V"].requested_keys
            )
        )
        geopotential_request = (
            0,
            slice(0, 2, 1),
            slice(0, 1, 1),
            slice(1, 3, 1),
        )
        for name in ("PH", "PHB"):
            self.assertTrue(
                all(
                    key == geopotential_request
                    for key in backends[name].requested_keys
                )
            )

    def test_read_ohc_inputs_converts_kelvin_and_preserves_depth(self):
        temperature_c, depth_m = diag.read_ohc_inputs(
            synthetic_member_dataset(include_ocean=True)
        )
        np.testing.assert_allclose(temperature_c[:, 0, 0], [29.0, 27.0, 25.0])
        np.testing.assert_allclose(depth_m[:, 0, 0], [0.0, 10.0, 20.0])

    def test_ohc_reader_only_loads_requested_spatial_box(self):
        ds = synthetic_member_dataset(include_ocean=True)
        temperature_values = ds["OM_TMP"].values.copy()
        depth_values = ds["OM_DEPTH"].values.copy()
        temperature, temperature_backend = lazy_recording_data_array(
            temperature_values, ds["OM_TMP"].dims
        )
        temperature.attrs["units"] = "K"
        depth, depth_backend = lazy_recording_data_array(
            depth_values, ds["OM_DEPTH"].dims
        )
        depth.attrs["units"] = "m"
        ds = ds.drop_vars(["OM_TMP", "OM_DEPTH"]).assign(
            OM_TMP=temperature, OM_DEPTH=depth
        )

        temperature_c, depth_m = diag.read_ohc_inputs(
            ds, y_slice=slice(0, 1), x_slice=slice(1, 3)
        )

        self.assertEqual(temperature_c.shape, (3, 1, 2))
        self.assertEqual(depth_m.shape, (3, 1, 2))
        expected = (0, slice(None), slice(0, 1, 1), slice(1, 3, 1))
        self.assertTrue(temperature_backend.requested_keys)
        self.assertTrue(depth_backend.requested_keys)
        self.assertTrue(
            all(key == expected for key in temperature_backend.requested_keys)
        )
        self.assertTrue(all(key == expected for key in depth_backend.requested_keys))

    def test_ohc_reader_reuses_surface_temperature_loaded_for_lh(self):
        ds = synthetic_member_dataset(include_ocean=True)
        temperature, temperature_backend = lazy_recording_data_array(
            ds["OM_TMP"].values.copy(), ds["OM_TMP"].dims
        )
        temperature.attrs["units"] = "K"
        ds = ds.drop_vars("OM_TMP").assign(OM_TMP=temperature)

        state, _, _, _ = diag.read_surface_state(ds, use_ocean_sst=True)
        temperature_c, _ = diag.read_ohc_inputs(
            ds,
            y_slice=slice(0, 1),
            x_slice=slice(1, 3),
            surface_temperature_k=state.surface_temperature_k[0:1, 1:3],
        )

        np.testing.assert_allclose(temperature_c[:, 0, 0], [29.0, 27.0, 25.0])
        surface_request = (0, 0, slice(None), slice(None))
        subsurface_request = (0, slice(1, 3, 1), slice(0, 1, 1), slice(1, 3, 1))
        self.assertIn(surface_request, temperature_backend.requested_keys)
        self.assertIn(subsurface_request, temperature_backend.requested_keys)
        self.assertNotIn(
            (0, slice(None), slice(0, 1, 1), slice(1, 3, 1)),
            temperature_backend.requested_keys,
        )

    def test_ohc_reader_loads_one_shared_depth_profile(self):
        ds = synthetic_member_dataset(include_ocean=True)
        depth, depth_backend = lazy_recording_data_array(
            ds["OM_DEPTH"].values.copy(), ds["OM_DEPTH"].dims
        )
        depth.attrs["units"] = "m"
        ds = ds.drop_vars("OM_DEPTH").assign(OM_DEPTH=depth)

        _, depth_m = diag.read_ohc_inputs(
            ds,
            y_slice=slice(0, 1),
            x_slice=slice(1, 3),
            depth_reference_index=(0, 0),
        )

        self.assertEqual(depth_m.shape, (3, 1, 2))
        np.testing.assert_allclose(depth_m[:, 0, 1], [0.0, 10.0, 20.0])
        expected = (0, slice(None), 0, 1)
        self.assertTrue(depth_backend.requested_keys)
        self.assertTrue(all(key == expected for key in depth_backend.requested_keys))

    def test_invalid_land_ocean_state_is_ignored_before_lh_and_ohc(self):
        ds = synthetic_member_dataset(include_ocean=True)
        ds["XLAND"][0, 0, 0] = 1.0
        ds["OM_TMP"][0, :, 0, 0] = 0.0
        depth = ds["OM_DEPTH"].values.copy()
        depth[0, :, 0, 0] = 0.0
        ds["OM_DEPTH"] = (ds["OM_DEPTH"].dims, depth, {"units": "m"})
        flux = diag.calculate_lh_field(ds, use_ocean_sst=True)
        self.assertTrue(np.isfinite(flux.lh).all())
        temperature_c, depth_m = diag.read_ohc_inputs(ds)
        mask = np.asarray(ds["XLAND"][0].values > 1.5)
        mean, count = diag._ohc_on_mask(
            temperature_c, depth_m, mask, 1025.0, 3985.0
        )
        self.assertTrue(np.isfinite(mean))
        self.assertEqual(count, int(mask.sum()))

    def test_missing_surface_physics_metadata_is_rejected(self):
        for attribute in ("ISFFLX", "ISFTCFLX"):
            with self.subTest(attribute=attribute):
                ds = synthetic_member_dataset(include_ocean=True)
                del ds.attrs[attribute]
                with self.assertRaisesRegex(KeyError, attribute):
                    diag.calculate_lh_field(ds, use_ocean_sst=True)

    def test_exact_and_suffixed_wrfout_matches_are_ambiguous(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            basename = "wrfout_d02_2018-09-10_00:00:00"
            (root / basename).touch()
            (root / f"{basename}.backup").touch()
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                diag.find_unique_wrfout(root, "d02", "2018-09-10_00:00:00")

    def test_group_statistics_use_sample_standard_deviation(self):
        frame = pd.DataFrame(
            {
                "experiment": ["x", "x"],
                "filter": ["EAKF", "EAKF"],
                "value": [10.0, 14.0],
            }
        )
        got = diag.attach_group_statistics(frame, "value")
        np.testing.assert_allclose(got["ensemble_mean"], 12.0)
        np.testing.assert_allclose(got["ensemble_std"], np.sqrt(8.0))

    def test_workflow_omits_no_da_from_ohc(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(
                diag.CONFIG,
                experiment_root=root / "experiments",
                nr_root=root / "nr",
                output_dir=root / "output",
                member_domain="d02",
                nr_domain="d03",
                filters=("EAKF",),
                members=("001", "002"),
                experiments=(
                    diag.Experiment("noda", "No DA", False, "#0072B2"),
                    diag.Experiment("weak", "Weak", True, "#D55E00"),
                ),
            )
            nr_path = config.nr_root / f"wrfout_d03_{config.valid_time}"
            nr_path.parent.mkdir(parents=True)
            nr = synthetic_member_dataset(include_ocean=False)[["XLAT", "XLONG"]]
            nr.to_netcdf(nr_path)

            for experiment in config.experiments:
                for member in config.members:
                    member_dir = (
                        config.experiment_root / experiment.name / "EAKF" / member
                    )
                    member_dir.mkdir(parents=True)
                    path = member_dir / f"wrfout_d02_{config.valid_time}"
                    synthetic_member_dataset(experiment.ocean_enabled).to_netcdf(path)

            slp = np.array([[1005.0, 1004.0, 1003.0], [1002.0, 990.0, 1001.0]])
            lh, ohc = diag.calculate_member_records(
                config,
                slp_reader=lambda _: slp,
            )

        self.assertEqual(set(lh["experiment"]), {"noda", "weak"})
        self.assertEqual(set(ohc["experiment"]), {"weak"})
        self.assertEqual(len(lh), 4)
        self.assertEqual(len(ohc), 2)
        self.assertTrue((lh["stored_flux_used"] == False).all())  # noqa: E712
        self.assertTrue((lh["lh_source"] == "reconstructed_initial").all())
        self.assertTrue((lh["xtime_minutes"] == 0.0).all())


def cached_member_frames(config: diag.Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    lh_rows = []
    ohc_rows = []
    for experiment in config.experiments:
        for filter_name in config.filters:
            for member in config.members:
                common = {
                    "experiment": experiment.name,
                    "experiment_label": experiment.label,
                    "filter": filter_name,
                    "member": member,
                    "ensemble_mean": 125.0,
                    "ensemble_std": 5.0,
                    "xtime_minutes": 0.0,
                    "lh_source": "reconstructed_initial",
                    "stored_flux_used": False,
                }
                lh_rows.append({**common, "lh_mean_w_m2": 125.0})
                if experiment.ocean_enabled:
                    ohc_rows.append({**common, "ohc26_mean_kj_cm2": 82.0})
    return pd.DataFrame(lh_rows), pd.DataFrame(ohc_rows)


class CacheInputTests(unittest.TestCase):
    def cache_config(self, root: Path) -> diag.Config:
        return replace(
            diag.CONFIG,
            output_dir=root,
            filters=("EAKF", "QCF_RHF"),
            members=("001", "002"),
            experiments=(
                diag.Experiment("noda", "No DA", False, "#0072B2"),
                diag.Experiment("weak", "Weak", True, "#D55E00"),
            ),
        )

    def test_load_nr_cache_reads_one_finite_reference_row(self):
        with TemporaryDirectory() as tmp:
            config = self.cache_config(Path(tmp))
            expected = pd.DataFrame(
                {
                    "lh_mean_w_m2": [175.0],
                    "ohc26_mean_kj_cm2": [90.0],
                    "xtime_minutes": [5.0],
                    "lh_source": ["stored_LH"],
                    "stored_flux_used": [True],
                }
            )
            expected.to_csv(
                config.output_dir / "initial_tc150_nr_reference.csv", index=False
            )

            got = diag.load_nr_cache(config)

        pd.testing.assert_frame_equal(got, expected)

    def test_load_member_cache_reads_complete_lh_and_ohc_tables(self):
        with TemporaryDirectory() as tmp:
            config = self.cache_config(Path(tmp))
            expected_lh, expected_ohc = cached_member_frames(config)
            expected_lh.to_csv(
                config.output_dir / "initial_tc150_lh_members.csv", index=False
            )
            expected_ohc.to_csv(
                config.output_dir / "initial_tc150_ohc_members.csv", index=False
            )

            got_lh, got_ohc = diag.load_member_cache(config)

        pd.testing.assert_frame_equal(got_lh, expected_lh)
        pd.testing.assert_frame_equal(got_ohc, expected_ohc)

    def test_load_member_cache_rejects_missing_paired_ohc_csv(self):
        with TemporaryDirectory() as tmp:
            config = self.cache_config(Path(tmp))
            lh, _ = cached_member_frames(config)
            lh.to_csv(config.output_dir / "initial_tc150_lh_members.csv", index=False)

            with self.assertRaisesRegex(FileNotFoundError, "ohc_members"):
                diag.load_member_cache(config)

    def test_load_member_cache_rejects_duplicate_member_key(self):
        with TemporaryDirectory() as tmp:
            config = self.cache_config(Path(tmp))
            lh, ohc = cached_member_frames(config)
            lh = pd.concat([lh, lh.iloc[[0]]], ignore_index=True)
            lh.to_csv(config.output_dir / "initial_tc150_lh_members.csv", index=False)
            ohc.to_csv(config.output_dir / "initial_tc150_ohc_members.csv", index=False)

            with self.assertRaisesRegex(ValueError, "duplicate"):
                diag.load_member_cache(config)

    def test_load_member_cache_rejects_nonfinite_metric(self):
        with TemporaryDirectory() as tmp:
            config = self.cache_config(Path(tmp))
            lh, ohc = cached_member_frames(config)
            lh.loc[0, "lh_mean_w_m2"] = np.nan
            lh.to_csv(config.output_dir / "initial_tc150_lh_members.csv", index=False)
            ohc.to_csv(config.output_dir / "initial_tc150_ohc_members.csv", index=False)

            with self.assertRaisesRegex(ValueError, "nonfinite"):
                diag.load_member_cache(config)

    def test_load_member_cache_requires_flux_source_audit_columns(self):
        with TemporaryDirectory() as tmp:
            config = self.cache_config(Path(tmp))
            lh, ohc = cached_member_frames(config)
            lh = lh.drop(columns="lh_source")
            lh.to_csv(config.output_dir / "initial_tc150_lh_members.csv", index=False)
            ohc.to_csv(config.output_dir / "initial_tc150_ohc_members.csv", index=False)

            with self.assertRaisesRegex(ValueError, "lh_source"):
                diag.load_member_cache(config)

    def test_load_member_cache_rejects_incomplete_configured_coverage(self):
        with TemporaryDirectory() as tmp:
            config = self.cache_config(Path(tmp))
            lh, ohc = cached_member_frames(config)
            lh.iloc[:-1].to_csv(
                config.output_dir / "initial_tc150_lh_members.csv", index=False
            )
            ohc.to_csv(config.output_dir / "initial_tc150_ohc_members.csv", index=False)

            with self.assertRaisesRegex(ValueError, "configured coverage"):
                diag.load_member_cache(config)


class CacheSwitchTests(unittest.TestCase):
    def test_cache_switches_default_to_recalculation(self):
        self.assertFalse(diag.CONFIG.read_nr_from_csv)
        self.assertFalse(diag.CONFIG.read_members_from_csv)

    def test_resolve_results_can_cache_nr_and_calculate_members(self):
        nr = pd.DataFrame(
            {"lh_mean_w_m2": [175.0], "ohc26_mean_kj_cm2": [90.0]}
        )
        lh = pd.DataFrame({"lh_mean_w_m2": [120.0]})
        ohc = pd.DataFrame({"ohc26_mean_kj_cm2": [80.0]})
        config = replace(
            diag.CONFIG,
            read_nr_from_csv=True,
            read_members_from_csv=False,
        )
        with (
            patch.object(diag, "load_nr_cache", return_value=nr),
            patch.object(diag, "calculate_member_records", return_value=(lh, ohc)),
            patch.object(
                diag,
                "find_unique_wrfout",
                side_effect=AssertionError("NR calculation must be skipped"),
            ),
        ):
            got_lh, got_ohc, got_nr = diag.resolve_diagnostic_tables(config)

        self.assertIs(got_lh, lh)
        self.assertIs(got_ohc, ohc)
        self.assertIs(got_nr, nr)

    def test_resolve_results_can_calculate_nr_and_cache_members(self):
        nr = pd.DataFrame(
            {"lh_mean_w_m2": [175.0], "ohc26_mean_kj_cm2": [90.0]}
        )
        lh = pd.DataFrame({"lh_mean_w_m2": [120.0]})
        ohc = pd.DataFrame({"ohc26_mean_kj_cm2": [80.0]})
        nr_path = Path("nr.nc")
        config = replace(
            diag.CONFIG,
            read_nr_from_csv=False,
            read_members_from_csv=True,
        )
        with (
            patch.object(diag, "load_member_cache", return_value=(lh, ohc)),
            patch.object(diag, "find_unique_wrfout", return_value=nr_path),
            patch.object(diag, "read_tc_center", return_value=(20.0, 120.0, 950.0)),
            patch.object(diag, "calculate_nr_reference", return_value=nr),
            patch.object(
                diag,
                "calculate_member_records",
                side_effect=AssertionError("member calculation must be skipped"),
            ),
        ):
            got_lh, got_ohc, got_nr = diag.resolve_diagnostic_tables(config)

        self.assertIs(got_lh, lh)
        self.assertIs(got_ohc, ohc)
        self.assertIs(got_nr, nr)


class OutputTests(unittest.TestCase):
    def test_plot_member_comparison_draws_solid_red_nr_reference_line(self):
        with TemporaryDirectory() as tmp:
            import matplotlib.pyplot as plt

            config = replace(
                diag.CONFIG,
                output_dir=Path(tmp),
                filters=("EAKF",),
                experiments=(diag.Experiment("weak", "Weak", True, "#D55E00"),),
            )
            frame = pd.DataFrame(
                {
                    "experiment": ["weak", "weak"],
                    "experiment_label": ["Weak", "Weak"],
                    "filter": ["EAKF", "EAKF"],
                    "member": ["001", "002"],
                    "lh_mean_w_m2": [120.0, 130.0],
                    "ensemble_mean": [125.0, 125.0],
                    "ensemble_std": [np.sqrt(50.0), np.sqrt(50.0)],
                }
            )
            with patch("matplotlib.pyplot.close"):
                diag.plot_member_comparison(
                    frame,
                    "lh_mean_w_m2",
                    Path(tmp) / "lh.png",
                    config,
                    ylabel="LH",
                    title="LH comparison",
                    nr_value=175.0,
                )
                figure = plt.gcf()

            nr_lines = [
                line for line in figure.axes[0].lines if line.get_label() == "NR"
            ]
            self.assertEqual(len(nr_lines), 1)
            np.testing.assert_allclose(nr_lines[0].get_ydata(), [175.0, 175.0])
            self.assertEqual(nr_lines[0].get_color().lower(), "#d73027")
            self.assertEqual(nr_lines[0].get_linestyle(), "-")
            plt.close(figure)

    def test_write_outputs_creates_nr_csv_and_two_separate_pngs(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = replace(
                diag.CONFIG,
                output_dir=output_dir,
                filters=("EAKF", "QCF_RHF"),
                members=("001", "002"),
                experiments=(
                    diag.Experiment("noda", "No DA", False, "#0072B2"),
                    diag.Experiment("weak", "Weak", True, "#D55E00"),
                ),
            )
            lh = pd.DataFrame(
                {
                    "experiment": ["noda", "noda", "weak", "weak"],
                    "experiment_label": ["No DA", "No DA", "Weak", "Weak"],
                    "filter": ["EAKF", "QCF_RHF", "EAKF", "QCF_RHF"],
                    "member": ["001", "002", "001", "002"],
                    "lh_mean_w_m2": [120.0, 125.0, 130.0, 135.0],
                    "ensemble_mean": [120.0, 125.0, 130.0, 135.0],
                    "ensemble_std": [0.0, 0.0, 0.0, 0.0],
                }
            )
            ohc = pd.DataFrame(
                {
                    "experiment": ["weak", "weak"],
                    "experiment_label": ["Weak", "Weak"],
                    "filter": ["EAKF", "QCF_RHF"],
                    "member": ["001", "002"],
                    "ohc26_mean_kj_cm2": [80.0, 82.0],
                    "ensemble_mean": [80.0, 82.0],
                    "ensemble_std": [0.0, 0.0],
                }
            )
            nr_reference = pd.DataFrame(
                {
                    "lh_mean_w_m2": [175.0],
                    "ohc26_mean_kj_cm2": [90.0],
                    "lh_source": ["LH"],
                }
            )
            paths = diag.write_outputs(lh, ohc, nr_reference, config)
            self.assertEqual(
                [path.name for path in paths],
                [
                    "initial_tc150_lh_members.csv",
                    "initial_tc150_ohc_members.csv",
                    "initial_tc150_nr_reference.csv",
                    "initial_tc150_lh.png",
                    "initial_tc150_ohc.png",
                ],
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            self.assertEqual(len(list(output_dir.glob("*.png"))), 2)

    def test_write_outputs_does_not_rewrite_csvs_loaded_as_cache(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = replace(
                diag.CONFIG,
                output_dir=output_dir,
                filters=("EAKF", "QCF_RHF"),
                members=("001", "002"),
                experiments=(
                    diag.Experiment("noda", "No DA", False, "#0072B2"),
                    diag.Experiment("weak", "Weak", True, "#D55E00"),
                ),
            )
            lh, ohc = cached_member_frames(config)
            nr = pd.DataFrame(
                {"lh_mean_w_m2": [175.0], "ohc26_mean_kj_cm2": [90.0]}
            )
            paths = {
                "lh": output_dir / "initial_tc150_lh_members.csv",
                "ohc": output_dir / "initial_tc150_ohc_members.csv",
                "nr": output_dir / "initial_tc150_nr_reference.csv",
            }
            lh.to_csv(paths["lh"], index=False)
            ohc.to_csv(paths["ohc"], index=False)
            nr.to_csv(paths["nr"], index=False)
            before = {name: path.read_bytes() for name, path in paths.items()}

            outputs = diag.write_outputs(
                lh,
                ohc,
                nr,
                config,
                write_member_csv=False,
                write_nr_csv=False,
            )

            after = {name: path.read_bytes() for name, path in paths.items()}
            self.assertEqual(after, before)
            self.assertTrue(outputs[-2].is_file())
            self.assertTrue(outputs[-1].is_file())


if __name__ == "__main__":
    unittest.main()
