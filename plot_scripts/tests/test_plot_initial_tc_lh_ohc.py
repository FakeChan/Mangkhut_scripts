from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

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
        self.assertEqual(lats.shape, (2, 3))
        self.assertTrue(ocean.all())
        result = diag.calculate_lh_field(ds, use_ocean_sst=True)
        self.assertTrue((result.lh > 0.0).all())

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
        for name in ("PSFC", "TSK"):
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


class OutputTests(unittest.TestCase):
    def test_write_outputs_creates_two_csvs_and_two_separate_pngs(self):
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
            paths = diag.write_outputs(lh, ohc, config)
            self.assertEqual(
                [path.name for path in paths],
                [
                    "initial_tc150_lh_members.csv",
                    "initial_tc150_ohc_members.csv",
                    "initial_tc150_lh.png",
                    "initial_tc150_ohc.png",
                ],
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            self.assertEqual(len(list(output_dir.glob("*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
