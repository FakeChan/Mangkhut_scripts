"""Local synthetic tests: no server connections or experiment inputs."""
import importlib
import importlib.util
import itertools
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from plot_scripts import diag_LACC_lag_cov_innovation as lag


class ScatterTests(unittest.TestCase):
    def setUp(self):
        name = "plot_scripts.diag_LACC_member_scatter"
        self.assertIsNotNone(importlib.util.find_spec(name), "scatter diagnostic not implemented")
        self.s = importlib.import_module(name)

    def settings(self, inputs=None, **kwargs):
        return self.s.ScatterConfig(
            inputs=inputs or lag.Config(), bootstrap_repeats=60,
            subset_repeats=30, **kwargs,
        )

    def test_regression_is_ocean_on_hx_and_nr_residual_is_vertical(self):
        # y = 10 - 2*x. NR=(2,8) is 2 K above this regression.
        result = self.s.pair_statistics([0, 1, 2, 3], [10, 8, 6, 4], 2, 8)
        self.assertAlmostEqual(result["covariance"], -10 / 3)
        self.assertAlmostEqual(result["slope_ocean_on_hx"], -2)
        self.assertAlmostEqual(result["nr_ocean_regression"], 6)
        self.assertAlmostEqual(result["nr_vertical_residual_K"], 2)
        self.assertAlmostEqual(result["d_NR"], 0.5)
        self.assertAlmostEqual(result["e_o"], 1)
        self.assertEqual(result["alignment_flag"], "AWAY_FROM_NR")

    def test_full_window_averages_hx_by_member_before_regression(self):
        context = lag._make_context()
        context.ocean_prior = np.array([10., 8., 6., 4.])
        context.ocean_nr = 9.
        # 9h must not be included in the real [0,3] window.
        series = lag._make_series([0, 3, 9], [[0, 1, 2, 3], [4, 3, 2, 1], [90]*4], [2, 6, 100])
        panels = self.s.build_panels(context, series, {0, 3})
        np.testing.assert_allclose(panels[-1].hx, [2, 2, 2, 2])
        self.assertEqual(panels[-1].nr_hx, 4)
        self.assertEqual(panels[-1].lag_hours, (0, 3))
        self.assertEqual(len(panels), 4)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.s.build_panels(context, series, {0, 3, 6})

    def test_paired_resampling_keeps_perfect_negative_relationship(self):
        cfg = self.settings()
        draws = self.s.make_resamples(10, cfg)
        panel = self.s.Panel("0h", (0,), np.arange(10.), 20.-2*np.arange(10.), 5., 12., "clear")
        result, loo, samples = self.s.analyze_panel(panel, list(range(2, 12)), draws, cfg)
        self.assertEqual(result["bootstrap"]["same_sign_fraction_valid"], 1.)
        self.assertEqual(result["subsets"]["same_sign_fraction_valid"], 1.)
        self.assertTrue(all(abs(row["correlation"] + 1) < 1e-12 for row in samples if row["valid"]))
        self.assertEqual([row["omitted_member"] for row in loo], list(range(2, 12)))
        self.assertEqual(result["leave_one_out"]["n_sign_flips"], 0)
        self.assertTrue(all(len(row["member_indices"].split(",")) == 5 for row in samples if row["method"] == "subsets"))

    def test_leave_one_out_identifies_a_member_that_reverses_covariance(self):
        panel = self.s.Panel("0h", (0,), np.array([0.,1.,2.,3.,10.]),
                             np.array([3.,2.,1.,0.,10.]), 5., 5., "clear")
        cfg = self.settings()
        result, loo, _ = self.s.analyze_panel(panel, [11,12,13,14,15], self.s.make_resamples(5,cfg), cfg)
        self.assertGreater(result["covariance"], 0)
        last = next(row for row in loo if row["omitted_member"] == 15)
        self.assertLess(last["covariance_without_member"], 0)
        self.assertTrue(last["sign_flip"])
        self.assertIn(15, result["leave_one_out"]["sign_flip_members"])

    def test_degenerate_resamples_are_counted_without_fabricating_stability(self):
        cfg = self.settings()
        panel = self.s.Panel("0h", (0,), np.ones(4), np.arange(4.), 1., 4., "unknown")
        result, _, _ = self.s.analyze_panel(panel, [1,2,3,4], self.s.make_resamples(4,cfg), cfg)
        self.assertTrue(np.isnan(result["correlation"]))
        self.assertTrue(np.isnan(result["slope_ocean_on_hx"]))
        self.assertEqual(result["bootstrap"]["n_valid"], 60)
        self.assertEqual(result["bootstrap"]["n_correlation_valid"], 0)
        self.assertIsNone(result["bootstrap"]["same_sign_fraction_valid"])
        self.assertEqual(result["bootstrap"]["n_invalid"], 0)

    def test_decimal_kelvin_constants_have_exactly_zero_spread(self):
        cfg = self.settings()
        for n in (25, 50):
            with self.subTest(n=n):
                panel = self.s.Panel("0h", (0,), np.full(n,250.0001),
                                     np.full(n,290.0003), 251., 291., "clear")
                result, _, _ = self.s.analyze_panel(panel, list(range(1,n+1)),
                                                  self.s.make_resamples(n,cfg), cfg)
                self.assertEqual(result["covariance"], 0.)
                self.assertEqual(result["hx_std"], 0.)
                self.assertEqual(result["ocean_std"], 0.)
                self.assertTrue(np.isnan(result["correlation"]))
                self.assertTrue(np.isnan(result["slope_ocean_on_hx"]))
                self.assertEqual(result["bootstrap"]["n_correlation_valid"], 0)
                self.assertIsNone(result["bootstrap"]["same_sign_fraction_valid"])
        for x, y in ((np.full(50,250.0001), np.arange(50.)),
                     (np.arange(50.), np.full(50,290.0003))):
            result = self.s.pair_statistics(x,y,251.,291.)
            self.assertEqual(result["covariance"], 0.)
            self.assertTrue(np.isnan(result["correlation"]))

    def test_zero_covariance_bootstrap_draws_remain_in_sign_denominator(self):
        cfg = self.settings()
        panel = self.s.Panel("0h", (0,), np.arange(3.), -np.arange(3.), 1., 0., "clear")
        draws = {"bootstrap": np.array(list(itertools.product(range(3), repeat=3)))}
        result, _, _ = self.s.analyze_panel(panel, [1,2,3], draws, cfg)
        boot = result["bootstrap"]
        self.assertEqual(boot["n_valid"], 27)
        self.assertEqual(boot["n_correlation_valid"], 24)
        self.assertAlmostEqual(boot["same_sign_fraction_valid"], 24/27)
        self.assertAlmostEqual(boot["negative_fraction_valid"], 24/27)

    def test_single_supporting_member_does_not_look_bootstrap_stable(self):
        cfg = self.s.ScatterConfig(figure_formats=())
        hx = np.r_[np.zeros(49), 1.]
        panel = self.s.Panel("0h", (0,), hx, 1-hx, 1., 1., "clear")
        result, _, _ = self.s.analyze_panel(panel, list(range(1,51)), self.s.make_resamples(50,cfg), cfg)
        self.assertEqual(result["bootstrap"]["n_valid"], 2000)
        self.assertEqual(result["bootstrap"]["n_correlation_valid"], 1296)
        self.assertAlmostEqual(result["bootstrap"]["same_sign_fraction_valid"], 0.648)
        self.assertEqual(result["bootstrap"]["covariance_percentile_interval"][1], 0.)
        self.assertEqual(result["leave_one_out"]["sign_zero_members"], [50])

    def test_rejects_nonfinite_inputs_and_wrong_member_mapping(self):
        with self.assertRaises(ValueError):
            self.s.pair_statistics([0,1,np.nan], [0,1,2], 1, 1)
        cfg = self.settings()
        panel = self.s.Panel("0h", (0,), np.arange(4.), np.arange(4.), 2., 2., "clear")
        with self.assertRaises(ValueError):
            self.s.analyze_panel(panel, [1,1,3,4], self.s.make_resamples(4,cfg), cfg)

    def test_local_synthetic_run_exports_nr_members_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = replace(lag._build_synthetic_dataset(Path(tmp)), write_figures=False)
            cfg = self.settings(inputs, figure_formats=("png", "pdf"), figure_dpi=90)
            payload = self.s.run(cfg)
            self.assertEqual(payload["checks"]["member_hx_f_order"]["n_verified"], 12)
            report = payload["observations"]["66"]
            self.assertEqual(len(report["panels"]), 4)
            full = report["panels"][-1]
            self.assertEqual(full["lag_hours"], [0,3,6])
            self.assertAlmostEqual(full["covariance"], -0.0023611111111111)
            self.assertAlmostEqual(full["d_NR"], 0.5916666666666667)
            self.assertEqual(full["nr_ocean"], 291.5)
            self.assertEqual(report["member_numbers"], [1,2,3,4])
            self.assertEqual(len(report["points"]), 16)
            for paths in report["figures"].values():
                self.assertGreater(Path(paths).stat().st_size, 1000)
            saved = json.loads(Path(payload["outputs"]["json"]).read_text())
            self.assertEqual(saved["resampling"]["pairing"], "same member indices for ocean and every Hx lag")
            for p in payload["outputs"].values():
                self.assertTrue(Path(p).is_file())

    def test_export_is_strict_json_even_when_statistics_are_undefined(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = replace(lag._build_synthetic_dataset(Path(tmp)), write_figures=False)
            for member in inputs.member_numbers():
                with lag.nc.Dataset(lag.background_member_path(inputs, member), "r+") as dataset:
                    dataset.variables[inputs.state_var][:] = 290.0
            payload = self.s.run(self.settings(inputs, figure_formats=()))
            text = Path(payload["outputs"]["json"]).read_text()
            def reject_constant(value):
                raise ValueError(f"nonstandard JSON constant: {value}")
            json.loads(text, parse_constant=reject_constant)

    def test_fifty_member_end_to_end_and_common_draws_across_panels(self):
        ocean = np.linspace(-0.4, 0.4, 50)
        hx = 250 + np.array([0.8, -0.6, -1.2])[:,None]*ocean[None,:]
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(
                lag, SYNTHETIC_MEMBERS=50, SYNTHETIC_OCEAN_BY_MEMBER=ocean,
                SYNTHETIC_HX_BY_LAG_MEMBER=hx):
            inputs = replace(lag._build_synthetic_dataset(Path(tmp)), write_figures=False)
            cfg = self.settings(inputs, figure_formats=())
            payload = self.s.run(cfg)
            self.assertEqual(payload["checks"]["member_hx_f_order"]["n_verified"], 150)
            obs = payload["observations"]["66"]
            self.assertEqual(obs["member_numbers"], list(range(1,51)))
            self.assertEqual(obs["panels"][-1]["subsets"]["sample_size"], 25)
            self.assertAlmostEqual(obs["panels"][-1]["covariance"], -np.var(ocean,ddof=1)/3)
            import csv
            with Path(payload["outputs"]["resamples_csv"]).open() as stream:
                rows = list(csv.DictReader(stream))
            indices = [row["member_indices"] for row in rows
                       if row["obs_id"] == "66" and row["method"] == "bootstrap" and row["repetition"] == "0"]
            self.assertEqual(len(indices), 4)
            self.assertEqual(len(set(indices)), 1)

    def test_member_subset_and_f_order_gate_survive_in_new_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = replace(lag._build_synthetic_dataset(Path(tmp)), member_start=2, write_figures=False)
            cfg = self.settings(inputs, figure_formats=())
            p = self.s.run(cfg)
            self.assertEqual(p["observations"]["66"]["member_numbers"], [2,3,4])
            self.assertEqual(p["observations"]["66"]["external_fo_check"]["status"], "passed")
            hx = lag.member_hx_path(inputs, 2, "09_21_00")
            a = np.loadtxt(hx)
            np.savetxt(hx, a.reshape(26,26,order="F").reshape(-1,order="C"))
            with self.assertRaises(lag.ConsistencyError):
                self.s.run(cfg)


if __name__ == "__main__":
    unittest.main()
