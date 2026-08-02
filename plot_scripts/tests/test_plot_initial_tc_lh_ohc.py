from __future__ import annotations

import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
