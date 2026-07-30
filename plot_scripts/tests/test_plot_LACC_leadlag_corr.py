from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import netCDF4 as nc
import numpy as np

from plot_scripts import plot_LACC_leadlag_corr as lacc


class CoreLaccTests(unittest.TestCase):
    def make_config(self, root: Path, **overrides):
        values = {
            "hx_dir": root / "hx",
            "mem_dir": root / "members",
            "profile_path": root / "profile.dat",
            "output_dir": root / "output",
            "current_time": "2018-09-10_00:00:00",
            "max_lag_hours": 12,
            "lag_interval_hours": 3,
            "member_start": 1,
            "member_end": 4,
            "expected_obs_count": 2,
        }
        values.update(overrides)
        return lacc.Config(**values)

    def test_build_lag_hours_includes_every_three_hours_to_maximum(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            self.assertEqual(lacc.build_lag_hours(config), [0, 3, 6, 9, 12])

    def test_validate_config_rejects_nonintegral_lag_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(
                Path(tmp),
                max_lag_hours=10,
                lag_interval_hours=3,
            )
            with self.assertRaisesRegex(ValueError, "integer multiple"):
                lacc.validate_config(config)

    def test_hx_path_formats_member_and_crosses_day_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            path = lacc.hx_path(config, member=7, lag_hours=12)
            expected = (
                config.hx_dir
                / "mem007"
                / "AMSUA"
                / "BT_09_12_00"
                / "obs_d01_ch4_totalline.txt"
            )
            self.assertEqual(path, expected)

    def test_read_profile_coordinates_uses_marker_and_preserves_order(self):
        text = "\n".join(
            [
                "! Gas units",
                "1",
                "! Elevation (km), latitude and longitude (degrees)",
                "0.1 10.5 140.25",
                "! Pressure levels (hPa)",
                "1000",
                "! Elevation (km), latitude and longitude (degrees)",
                "0.2 11.5 141.25",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.dat"
            path.write_text(text)
            result = lacc.read_profile_coordinates(path, expected_count=2)

        np.testing.assert_allclose(result["elevation_km"], [0.1, 0.2])
        np.testing.assert_allclose(result["lat"], [10.5, 11.5])
        np.testing.assert_allclose(result["lon"], [140.25, 141.25])
        self.assertEqual(result["obs_index"].tolist(), [1, 2])

    def test_read_profile_coordinates_rejects_wrong_count(self):
        text = "\n".join(
            [
                "! Elevation (km), latitude and longitude (degrees)",
                "0.1 10.5 140.25",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.dat"
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "expected 2"):
                lacc.read_profile_coordinates(path, expected_count=2)

    def test_read_hx_file_requires_expected_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_path = root / "valid.txt"
            valid_path.write_text("250.0\n251.5\n")
            np.testing.assert_allclose(
                lacc.read_hx_file(valid_path, expected_count=2),
                [250.0, 251.5],
            )

            short_path = root / "short.txt"
            short_path.write_text("250.0\n")
            with self.assertRaisesRegex(ValueError, "expected 2"):
                lacc.read_hx_file(short_path, expected_count=2)

            nan_path = root / "nan.txt"
            nan_path.write_text("250.0\nnan\n")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                lacc.read_hx_file(nan_path, expected_count=2)

    def test_pointwise_correlations_use_members_not_spatial_points(self):
        omtmp = np.array(
            [
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [4.0, 1.0],
            ]
        )
        hx = np.array(
            [
                [10.0, 10.0],
                [20.0, 20.0],
                [30.0, 30.0],
                [40.0, 40.0],
            ]
        )
        result = lacc.pointwise_correlations(omtmp, hx)
        np.testing.assert_allclose(result, [1.0, -1.0], atol=1.0e-12)

    def test_pointwise_correlations_return_nan_for_zero_member_variance(self):
        omtmp = np.array([[1.0], [2.0], [3.0], [4.0]])
        hx = np.array([[8.0], [8.0], [8.0], [8.0]])
        result = lacc.pointwise_correlations(omtmp, hx)
        self.assertTrue(np.isnan(result[0]))

    def test_build_averaged_hx_uses_cumulative_windows_including_current(self):
        hx = np.array(
            [
                [[2.0], [4.0]],
                [[4.0], [8.0]],
                [[9.0], [12.0]],
            ]
        )
        result = lacc.build_averaged_hx(hx)
        expected = np.array(
            [
                [[2.0], [4.0]],
                [[3.0], [6.0]],
                [[5.0], [8.0]],
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_summarize_pointwise_keeps_signed_correlations(self):
        summary = lacc.summarize_pointwise(np.array([-1.0, 0.0, 1.0, np.nan]))
        self.assertEqual(summary["valid_point_count"], 3)
        self.assertEqual(summary["total_point_count"], 4)
        self.assertAlmostEqual(summary["mean_corr"], 0.0)
        self.assertAlmostEqual(summary["median_corr"], 0.0)
        self.assertAlmostEqual(summary["q25_corr"], -0.5)
        self.assertAlmostEqual(summary["q75_corr"], 0.5)


class InterpolationAndWorkflowTests(unittest.TestCase):
    @staticmethod
    def write_member_file(path: Path, member_value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with nc.Dataset(path, "w") as dataset:
            dataset.createDimension("Time", 1)
            dataset.createDimension("ocean_layer", 1)
            dataset.createDimension("south_north", 2)
            dataset.createDimension("west_east", 2)
            lat = dataset.createVariable(
                "XLAT",
                "f8",
                ("Time", "south_north", "west_east"),
            )
            lon = dataset.createVariable(
                "XLONG",
                "f8",
                ("Time", "south_north", "west_east"),
            )
            omtmp = dataset.createVariable(
                "OM_TMP",
                "f8",
                ("Time", "ocean_layer", "south_north", "west_east"),
            )
            lat[0] = [[0.0, 0.0], [1.0, 1.0]]
            lon[0] = [[0.0, 1.0], [0.0, 1.0]]
            omtmp[0, 0] = np.full((2, 2), member_value)

    @staticmethod
    def write_profile(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "! Elevation (km), latitude and longitude (degrees)",
                    "0.0 0.25 0.25",
                    "! Elevation (km), latitude and longitude (degrees)",
                    "0.0 0.75 0.75",
                ]
            )
        )

    @staticmethod
    def write_hx_files(config) -> None:
        for member in range(config.member_start, config.member_end + 1):
            lag0 = lacc.hx_path(config, member, 0)
            lag3 = lacc.hx_path(config, member, 3)
            lag0.parent.mkdir(parents=True, exist_ok=True)
            lag3.parent.mkdir(parents=True, exist_ok=True)
            lag0.write_text(f"{250.0 + member}\n{260.0 - member}\n")
            lag3.write_text(f"{255.0 - member}\n{270.0 - member}\n")

    def make_workflow_config(self, root: Path):
        return lacc.Config(
            hx_dir=root / "hx",
            mem_dir=root / "members",
            profile_path=root / "profile.dat",
            output_dir=root / "output",
            current_time="2018-09-10_00:00:00",
            max_lag_hours=3,
            lag_interval_hours=3,
            member_start=1,
            member_end=4,
            expected_obs_count=2,
        )

    def test_linear_interpolation_matches_planar_field(self):
        lat = np.array([[0.0, 0.0], [1.0, 1.0]])
        lon = np.array([[0.0, 1.0], [0.0, 1.0]])
        field = 10.0 + 2.0 * lat + 3.0 * lon
        result = lacc.interpolate_to_observations(
            field,
            lat,
            lon,
            obs_lat=np.array([0.25, 0.75]),
            obs_lon=np.array([0.75, 0.25]),
        )
        np.testing.assert_allclose(result, [12.75, 12.25], atol=1.0e-12)

    def test_linear_interpolation_rejects_points_outside_domain(self):
        lat = np.array([[0.0, 0.0], [1.0, 1.0]])
        lon = np.array([[0.0, 1.0], [0.0, 1.0]])
        field = np.ones((2, 2))
        with self.assertRaisesRegex(ValueError, "outside"):
            lacc.interpolate_to_observations(
                field,
                lat,
                lon,
                obs_lat=np.array([2.0]),
                obs_lon=np.array([2.0]),
            )

    def test_complete_workflow_writes_expected_tables_and_two_panel_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_workflow_config(root)
            self.write_profile(config.profile_path)
            for member in range(config.member_start, config.member_end + 1):
                self.write_member_file(
                    lacc.member_path(config, member),
                    member_value=float(member),
                )
            self.write_hx_files(config)

            results, outputs = lacc.run(config)

            self.assertEqual(results.single_pointwise.shape[0], 4)
            self.assertEqual(results.single_summary["lag_hours"].tolist(), [0, 3])
            self.assertEqual(results.averaged_pointwise.shape[0], 4)
            self.assertEqual(
                results.averaged_summary["window_size"].tolist(),
                [1, 2],
            )
            self.assertEqual(results.omtmp_interpolated.shape[0], 8)

            lag0 = results.single_pointwise.query("lag_hours == 0")
            np.testing.assert_allclose(
                lag0.sort_values("obs_index")["corr"].to_numpy(),
                [1.0, -1.0],
                atol=1.0e-12,
            )
            ave2 = results.averaged_pointwise.query("window_size == 2")
            ave2_corr = ave2.sort_values("obs_index")["corr"].to_numpy()
            self.assertTrue(np.isnan(ave2_corr[0]))
            self.assertAlmostEqual(ave2_corr[1], -1.0)

            self.assertEqual(
                set(outputs),
                {
                    "single_pointwise",
                    "single_summary",
                    "averaged_pointwise",
                    "averaged_summary",
                    "omtmp_interpolated",
                    "figure",
                },
            )
            for output_path in outputs.values():
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
