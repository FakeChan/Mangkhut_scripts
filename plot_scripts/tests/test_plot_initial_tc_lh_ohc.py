from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

from plot_scripts import plot_initial_tc_lh_ohc as diag


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

    def test_read_ohc_inputs_converts_kelvin_and_preserves_depth(self):
        temperature_c, depth_m = diag.read_ohc_inputs(
            synthetic_member_dataset(include_ocean=True)
        )
        np.testing.assert_allclose(temperature_c[:, 0, 0], [29.0, 27.0, 25.0])
        np.testing.assert_allclose(depth_m[:, 0, 0], [0.0, 10.0, 20.0])

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


if __name__ == "__main__":
    unittest.main()
