"""Spread script: domain mode stays legacy; region modes follow the NR center."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_cycle_meanspread_times as spread


T0 = datetime(2018, 9, 10, 0, 0, 0)
TIMES = [T0, T0 + timedelta(minutes=30)]


class SpreadRegionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

        self.saved = {name: getattr(spread, name)
                      for name in ("EXPERIMENTS", "FILTERS", "MEMBERS", "VARIABLES")}
        self.addCleanup(self.restore)

        spread.EXPERIMENTS = ["6mem_oceanAssim0Run0", "6mem_oceanAssim0Run1"]
        spread.FILTERS = ["EAKF", "QCF_RHF"]
        spread.MEMBERS = ["001", "002"]
        spread.VARIABLES = [{
            "name": "HFX",
            "vertical_level": None,
            "scale": 1.0,
            "unit": "W m-2",
            "experiments": spread.EXPERIMENTS,
            "out_png": self.root / "figs" / "spread_hfx.png",
        }]
        self.variable = spread.VARIABLES[0]
        self.column = spread.spread_column(self.variable)

    def restore(self):
        for name, value in self.saved.items():
            setattr(spread, name, value)

    # -- synthetic wrfout -------------------------------------------------
    def write_file(self, path, hour, member_index=0, nr=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        ny, nx = (9, 11) if nr else (7, 9)
        j, i = np.indices((ny, nx)).astype(float)
        lat = (17.98 if nr else 18.0) + j * 0.02
        lon = (139.98 if nr else 140.0) + i * 0.02
        with Dataset(path, "w") as d:
            for dim, size in [("Time", 1), ("south_north", ny), ("west_east", nx)]:
                d.createDimension(dim, size)
            fields = {"XLAT": lat, "XLONG": lon}
            if nr:
                center_j, center_i = 4, 4 + int(round(6 * hour))
                fields["PSFC"] = 98000.0 + 10.0 * ((j - center_j) ** 2 + (i - center_i) ** 2)
            else:
                hfx = 100.0 + 10.0 * i + 5.0 * j + member_index * (3.0 + 5.0 * (i >= 5))
                hfx[0, 0] = 1.0e4 * (member_index + 1)  # land point inflates domain spread
                land = np.zeros((ny, nx))
                land[0, 0] = 1.0
                fields["HFX"] = hfx
                fields["LANDMASK"] = land
                fields["OM_TMP"] = 300.0 + member_index
            for name, value in fields.items():
                d.createVariable(name, "f8", ("Time", "south_north", "west_east"),
                                 fill_value=1e36)[0] = value

    def prepare(self):
        for t in TIMES:
            hour = (t - T0).total_seconds() / 3600
            stamp = t.strftime("%Y-%m-%d_%H:%M:%S")
            self.write_file(self.root / "nr" / f"wrfout_d02_{stamp}", hour, nr=True)
            for exp in spread.EXPERIMENTS:
                for filt in spread.FILTERS:
                    for mi, member in enumerate(spread.MEMBERS):
                        self.write_file(
                            self.root / "members" / exp / filt / member / f"wrfout_d02_{stamp}",
                            hour, member_index=mi,
                        )

    def expected_grid_spread(self):
        j, i = np.indices((7, 9)).astype(float)
        fields = []
        for m in range(len(spread.MEMBERS)):
            field = 100.0 + 10.0 * i + 5.0 * j + m * (3.0 + 5.0 * (i >= 5))
            field[0, 0] = 1.0e4 * (m + 1)
            fields.append(field)
        return np.nanstd(np.stack(fields, axis=0), axis=0, ddof=1)

    def row(self, df, mode, time, experiment=None, filt="EAKF"):
        sub = df[(df.region_mode == mode) & (df.time == time) & (df["filter"] == filt)]
        if experiment is not None:
            sub = sub[sub.experiment == experiment]
        return sub.iloc[0]

    # -- tests ------------------------------------------------------------
    def test_domain_mode_is_legacy_domain_mean(self):
        self.prepare()
        df = spread.calculate(self.root / "members", TIMES,
                              region_modes=("domain",), nr_base=self.root / "missing")
        expected = float(np.nanmean(self.expected_grid_spread()))
        row = self.row(df, "domain", T0.strftime("%Y-%m-%d_%H:%M:%S"))
        self.assertAlmostEqual(row[self.column], expected, places=6)
        self.assertEqual(int(row[spread.n_points_column(self.variable)]), 63)
        self.assertTrue(np.isnan(row.center_lat))

    def test_region_mode_uses_ocean_only_and_moves_with_nr(self):
        self.prepare()
        df = spread.calculate(self.root / "members", TIMES,
                              region_modes=("domain", "nr_initial", "nr_each_time"),
                              radius_km=3.0, nr_base=self.root / "nr")
        late = TIMES[1].strftime("%Y-%m-%d_%H:%M:%S")
        initial = self.row(df, "nr_initial", late)
        moving = self.row(df, "nr_each_time", late)

        ocean_spread = float(np.std([0.0, 3.0], ddof=1))
        moved_spread = float(np.std([0.0, 8.0], ddof=1))
        self.assertAlmostEqual(initial[self.column], ocean_spread, places=6)
        self.assertAlmostEqual(moving[self.column], moved_spread, places=6)
        self.assertNotAlmostEqual(initial.center_lon, moving.center_lon, places=6)
        self.assertEqual(int(initial[spread.n_points_column(self.variable)]), 5)
        self.assertEqual(int(moving[spread.n_points_column(self.variable)]), 5)

        domain = self.row(df, "domain", late)
        self.assertGreater(domain[self.column], 100.0)
        self.assertLess(initial[self.column], 3.0)

    def test_lost_common_support_raises(self):
        self.prepare()
        late = TIMES[1].strftime("%Y-%m-%d_%H:%M:%S")
        path = (self.root / "members" / spread.EXPERIMENTS[0] / "EAKF"
                / spread.MEMBERS[0] / f"wrfout_d02_{late}")
        with Dataset(path, "r+") as d:
            d["HFX"][0, 3, 3] = 1e36
        with self.assertRaisesRegex(ValueError, "support"):
            spread.calculate(self.root / "members", TIMES,
                             region_modes=("nr_initial",),
                             radius_km=3.0, nr_base=self.root / "nr")

    def test_member_grid_mismatch_raises(self):
        self.prepare()
        late = TIMES[1].strftime("%Y-%m-%d_%H:%M:%S")
        path = (self.root / "members" / spread.EXPERIMENTS[1] / "QCF_RHF"
                / spread.MEMBERS[1] / f"wrfout_d02_{late}")
        with Dataset(path, "r+") as d:
            d["XLAT"][0, 0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "grid"):
            spread.calculate(self.root / "members", TIMES,
                             region_modes=("nr_initial",),
                             radius_km=3.0, nr_base=self.root / "nr")

    def test_unknown_region_mode_raises(self):
        self.prepare()
        with self.assertRaises(ValueError):
            spread.calculate(self.root / "members", TIMES,
                             region_modes=("bogus",), nr_base=self.root / "nr")

    def test_region_mode_requires_positive_radius(self):
        self.prepare()
        with self.assertRaises(ValueError):
            spread.calculate(self.root / "members", TIMES,
                             region_modes=("nr_initial",), radius_km=0.0,
                             nr_base=self.root / "nr")

    def test_variable_limited_to_one_experiment_is_skipped_for_the_other(self):
        om = {
            "name": "OM_TMP",
            "vertical_level": None,
            "scale": 1.0,
            "unit": "K",
            "experiments": [spread.EXPERIMENTS[1]],
            "out_png": self.root / "figs" / "spread_omtmp.png",
        }
        spread.VARIABLES = [self.variable, om]
        self.prepare()
        df = spread.calculate(self.root / "members", TIMES,
                              region_modes=("nr_initial",), radius_km=3.0,
                              nr_base=self.root / "nr")
        om_column = spread.spread_column(om)
        missing = df[df.experiment == spread.EXPERIMENTS[0]].iloc[0]
        present = df[df.experiment == spread.EXPERIMENTS[1]].iloc[0]
        self.assertTrue(np.isnan(missing[om_column]))
        self.assertEqual(int(missing[spread.n_points_column(om)]), 0)
        self.assertAlmostEqual(present[om_column], float(np.std([0.0, 1.0], ddof=1)),
                               places=6)

    def test_mode_output_path_keeps_domain_name(self):
        path = Path("figs/spread_hfx.png")
        self.assertEqual(spread.mode_output_path(path, "domain"), path)
        self.assertEqual(spread.mode_output_path(path, "nr_initial"),
                         Path("figs/spread_hfx_nr_initial.png"))

    def test_plot_writes_one_figure_per_mode(self):
        self.prepare()
        df = spread.calculate(self.root / "members", TIMES,
                              region_modes=("domain", "nr_initial"),
                              radius_km=3.0, nr_base=self.root / "nr")
        spread.plot(df, region_modes=("domain", "nr_initial"), radius_km=3.0)
        self.assertTrue((self.root / "figs" / "spread_hfx.png").is_file())
        self.assertTrue((self.root / "figs" / "spread_hfx_nr_initial.png").is_file())


if __name__ == "__main__":
    unittest.main()
