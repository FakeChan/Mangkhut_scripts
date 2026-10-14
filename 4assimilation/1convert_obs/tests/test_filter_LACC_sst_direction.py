"""Local synthetic tests for filter_LACC_sst_direction.py.

Everything here runs against synthetic files written into a temporary
directory: no server path is read, nothing is downloaded, and no RTTOV / MPI /
DART program is executed (``text_to_obs`` is replaced by a local stub).  The
observation values, the member Hx values and the two ocean fields are written so
that every expected number can be stated in closed form.

    python -m unittest discover -s 4assimilation/1convert_obs/tests -t . -v

The fixture reproduces the deployed layout, including the marker-based profile
blocks with the 185-line stride the text converters rely on, so the same tree
drives the filter, ``obs2DART_LACC.py``, ``merge_FO_in_one_file.py`` and the
shell driver ``run_single_assimilation_test.sh``.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import netCDF4 as nc
import numpy as np

# matplotlib writes its font cache on first use; keep it out of HOME.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

REPO_ROOT = Path(__file__).resolve().parents[3]
CONVERT_DIR = REPO_ROOT / "4assimilation" / "1convert_obs"
FILTER_PATH = CONVERT_DIR / "filter_LACC_sst_direction.py"
DRIVER_PATH = CONVERT_DIR / "run_single_assimilation_test.sh"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


filt = _load_module(FILTER_PATH, "filter_LACC_sst_direction_under_test")
lag = filt.lag


# =============================================================================
# The synthetic experiment
# =============================================================================
CENTER_TIME = "2018-09-10_00:00:00"
CENTER_TAG = "10_00_00"
LAG_TAGS = ("10_00_00", "09_21_00", "09_18_00")
LAG_HOURS = (0, 3, 6)

LAT_CENTER, LON_CENTER = 14.3, 147.4
LAT_STEP, LON_STEP = 0.024, 0.032

BACKGROUND_BASE_K = 300.0
BACKGROUND_LAT_SLOPE, BACKGROUND_LON_SLOPE = 0.5, 0.25
# e = x_NR - x_bar carries (nr_lat_slope - BACKGROUND_LAT_SLOPE) * dlat, so its
# sign flips across the observation box (and is never accidentally zero).
NR_LAT_SLOPE, NR_LON_SLOPE = -1.5, 0.25
NR_BASE_K = 300.0

BACKGROUND_LAT = np.linspace(13.5, 15.1, 9)
BACKGROUND_LON = np.linspace(146.5, 148.3, 9)
# A deliberately DIFFERENT grid for the NR: different shape AND extent.
NR_LAT = np.linspace(13.2, 15.4, 12)
NR_LON = np.linspace(146.2, 148.6, 14)

FILLER_LEVEL_K = 999.0  # levels 1..29; visible if anything ever read them
NOISE_FREE_LAG_OFFSET_K = 1000.0  # makes a lag re-average obvious in the audit
PROFILE_MARKER = lag.PROFILE_COORDINATE_MARKER


def _state_field(lat: np.ndarray, lon: np.ndarray, base: float, dlat: float, dlon: float):
    """A field exactly linear in (lat, lon): linear interpolation is exact."""
    return base + dlat * (lat[:, None] - LAT_CENTER) + dlon * (lon[None, :] - LON_CENTER)


def write_state_file(
    path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    field: np.ndarray,
    time_string: str = CENTER_TIME,
    units: str = "k",
    landmask: "np.ndarray | None" = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nlat, nlon = field.shape
    with nc.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("ocean_layer_stag", 30)
        dataset.createDimension("south_north", nlat)
        dataset.createDimension("west_east", nlon)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:] = np.array([[char.encode("ascii") for char in time_string]], dtype="S1")
        xlat = dataset.createVariable("XLAT", "f8", ("Time", "south_north", "west_east"))
        xlon = dataset.createVariable("XLONG", "f8", ("Time", "south_north", "west_east"))
        xlat[:] = np.repeat(lat[:, None], nlon, axis=1)
        xlon[:] = np.repeat(lon[None, :], nlat, axis=0)
        mask = dataset.createVariable("LANDMASK", "f8", ("Time", "south_north", "west_east"))
        mask[:] = (
            np.zeros((nlat, nlon)) if landmask is None else np.asarray(landmask, dtype=float)
        )
        omtmp = dataset.createVariable(
            "OM_TMP", "f8", ("Time", "ocean_layer_stag", "south_north", "west_east")
        )
        omtmp.units = units
        full = np.full((30, nlat, nlon), FILLER_LEVEL_K, dtype="f8")
        full[0, :, :] = field
        omtmp[0, :, :, :] = full
    return path


def write_profile_file(
    path: Path, lat: np.ndarray, lon: np.ndarray
) -> Path:
    """The RTTOV profile layout: 185-line stride, marker on the row before each point.

    Line 186 + 185*k carries "elev lat lon" and line 188 + 185*k carries the
    angles -- exactly what ``read_every_nth_line(start_line=186|188, step=185)``
    in the text converters expects, and what the marker-based reader in the
    companion diagnostic finds.
    """
    nobs = int(lat.size)
    lines = ["! filler line keeping the block length at 185"] * (184 + 185 * nobs)
    for index in range(nobs):
        lines[185 + 185 * index - 1] = PROFILE_MARKER
        lines[186 + 185 * index - 1] = f"  0.0100  {lat[index]:.6f}  {lon[index]:.6f}"
        lines[188 + 185 * index - 1] = "  45.0000  120.0000"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


class Fixture:
    """A complete synthetic experiment shaped like the deployed one.

    ``write()`` creates the files; the analytic accessors (``x_bar``, ``h_bar``,
    ``covariance``, ...) are computed from the same design arrays and stay valid
    for a copy of the tree, so a test can build a template once and copy it.
    """

    def __init__(
        self,
        root: Path,
        members: int = 4,
        side: int = 26,
        kept: "int | None" = None,
        write_per_lag: bool = True,
    ) -> None:
        self.root = Path(root)
        self.members = int(members)
        self.side = int(side)
        self.nobs = self.side * self.side
        self.kept = int(kept) if kept is not None else max(2, self.nobs // 3)
        self.write_per_lag = bool(write_per_lag)

        self.hx_root = self.root / "3create_obs" / "hx_rttov"
        self.obs_bt_dir = self.hx_root / "3obs_BT_LACC" / "AMSUA"
        self.ens_bt_dir = self.hx_root / "4ens_BT_LACC"
        self.profile_file = (
            self.hx_root / "profile" / f"profile_d01_LACC_{CENTER_TAG}" / "prof10_00:00.dat"
        )
        self.background_dir = (
            self.root / "4assimilation" / "0mem_all_time" / "cyclingDA" / CENTER_TAG
        )
        self.nr_file = self.root / "NR_wrfout" / "2domain" / f"wrfout_d02_{CENTER_TIME}"
        self.output_dir = self.root / "filter_out"

        rows = self.side
        row_of = np.arange(self.nobs) % rows
        col_of = np.arange(self.nobs) // rows
        self.row_of = row_of
        self.col_of = col_of
        self.latitude = LAT_CENTER + (row_of - (rows - 1) / 2.0) * LAT_STEP
        self.longitude = LON_CENTER + (col_of - (rows - 1) / 2.0) * LON_STEP

        # Member offsets of the background SST, centred on zero, so that
        # mean_m(x_m) carries no member-offset term: e is then the difference of
        # the two linear fields alone.
        self.sst_base = np.linspace(-0.15, 0.15, self.members)

        # Designed per-observation fields.
        self.slope = np.where(row_of < rows / 2.0, -1.0, 1.0)
        self.hx_base = 250.0 + 0.5 * row_of + 0.01 * col_of  # unique per raw obs
        self.d_target = np.where((row_of + col_of) % 2 == 0, 0.4, -0.4)
        self.obs_value = self.hx_base + self.d_target
        # The noise-free centre-lag BT carries the OPPOSITE innovation, so a
        # reader that picked the wrong file would flip the decision.
        self.nr_bt_by_lag = np.stack(
            [
                self.hx_base - self.d_target,
                self.hx_base + NOISE_FREE_LAG_OFFSET_K,
                self.hx_base + 2.0 * NOISE_FREE_LAG_OFFSET_K,
            ]
        )

        self.original_mask = np.zeros(self.nobs, dtype=int)
        self.original_mask[
            np.unique(np.round(np.linspace(0, self.nobs - 1, self.kept)).astype(int))
        ] = 1
        # An NR offset that flips every e at once (used by the all-deleted case).
        self.nr_offset = 0.0
        self.nr_lat_slope = NR_LAT_SLOPE

    # ---------------------------------------------------------------- config
    @property
    def config(self) -> "filt.Config":
        """Only the essentials are set: every path default stays under test."""
        return filt.Config(
            project_root=self.root,
            output_dir=self.output_dir,
            center_time=CENTER_TIME,
            domain="d01",
            sensor="AMSUA",
            channel=4,
            member_start=1,
            member_end=self.members,
            nobs_raw=self.nobs,
            matrix_side=self.side,
            expected_lag_hours=LAG_HOURS,
        )

    def with_output_dir(self, directory: Path) -> "Fixture":
        self.output_dir = Path(directory)
        return self

    # ------------------------------------------------------------ analytics
    def x_members(self, raw_index: int) -> np.ndarray:
        j = raw_index - 1
        base = (
            BACKGROUND_BASE_K
            + BACKGROUND_LAT_SLOPE * (self.latitude[j] - LAT_CENTER)
            + BACKGROUND_LON_SLOPE * (self.longitude[j] - LON_CENTER)
        )
        return base + self.sst_base

    def nr_value(self, raw_index: int) -> float:
        j = raw_index - 1
        return (
            NR_BASE_K
            + self.nr_offset
            + self.nr_lat_slope * (self.latitude[j] - LAT_CENTER)
            + NR_LON_SLOPE * (self.longitude[j] - LON_CENTER)
        )

    def hx_members(self, raw_index: int) -> np.ndarray:
        j = raw_index - 1
        return self.hx_base[j] + self.slope[j] * (self.sst_base - self.sst_base.mean())

    def lag_hx_members(self, raw_index: int, lag_index: int) -> np.ndarray:
        """Per-lag member Hx: a different data set, offset by ~1000 K per lag."""
        j = raw_index - 1
        return (
            self.hx_base[j]
            + NOISE_FREE_LAG_OFFSET_K * (lag_index + 1)
            + self.slope[j] * (self.sst_base - self.sst_base.mean())
        )

    @staticmethod
    def _mean(values: np.ndarray) -> float:
        """The filter's rule for identical vectors, restated independently."""
        return float(values[0]) if bool(np.all(values == values[0])) else float(values.mean())

    def x_bar(self, raw_index: int) -> float:
        return self._mean(self.x_members(raw_index))

    def h_bar(self, raw_index: int) -> float:
        return self._mean(self.hx_members(raw_index))

    def covariance(self, raw_index: int) -> float:
        x = self.x_members(raw_index)
        h = self.hx_members(raw_index)
        x_const = bool(np.all(x == x[0]))
        h_const = bool(np.all(h == h[0]))
        x_bar = float(x[0]) if x_const else float(x.mean())
        h_bar = float(h[0]) if h_const else float(h.mean())
        dx = np.zeros_like(x) if x_const else x - x_bar
        dh = np.zeros_like(h) if h_const else h - h_bar
        return float(np.sum(dx * dh) / (x.size - 1))

    def innovation(self, raw_index: int) -> float:
        return float(self.obs_value[raw_index - 1]) - self.h_bar(raw_index)

    def sst_error(self, raw_index: int) -> float:
        return self.nr_value(raw_index) - self.x_bar(raw_index)

    def score(self, raw_index: int) -> float:
        return (
            self.covariance(raw_index) * self.innovation(raw_index) * self.sst_error(raw_index)
        )

    @staticmethod
    def _status_from_signs(cov: float, innovation: float, sst_error: float) -> str:
        signs = (np.sign(cov), np.sign(innovation), np.sign(sst_error))
        if any(sign == 0.0 for sign in signs):
            return filt.STATUS_KEPT_ZERO_DIRECTION
        if sum(1 for sign in signs if sign < 0.0) % 2 == 1:
            return filt.STATUS_REJECTED_REVERSE
        return filt.STATUS_KEPT_NONREVERSE

    def expected_status(self, raw_index: int) -> str:
        if int(self.original_mask[raw_index - 1]) == 0:
            return filt.STATUS_EXCLUDED_BY_ORIGINAL_MASK
        return self._status_from_signs(
            self.covariance(raw_index), self.innovation(raw_index), self.sst_error(raw_index)
        )

    def expected_keep(self, raw_index: int) -> bool:
        """True only for rows the ORIGINAL mask keeps AND the screen does not delete."""
        return self.expected_status(raw_index) in (
            filt.STATUS_KEPT_NONREVERSE,
            filt.STATUS_KEPT_ZERO_DIRECTION,
        )

    def kept_raw_ids(self) -> list[int]:
        return [raw for raw in range(1, self.nobs + 1) if self.expected_keep(raw)]

    def clear_sky_raw_ids(self) -> list[int]:
        return [raw for raw in range(1, self.nobs + 1) if int(self.original_mask[raw - 1])]

    # ------------------------------------------------------------- writing
    def write(self) -> "Fixture":
        self._write_centre_files()
        write_profile_file(self.profile_file, self.latitude, self.longitude)
        self._write_member_hx()
        self._write_state()
        return self

    def _write_centre_files(self) -> None:
        centre = self.obs_bt_dir / f"BT_LACC_{CENTER_TAG}"
        centre.mkdir(parents=True, exist_ok=True)
        body = [f"center_time={CENTER_TAG}"] + [f"lag_time={tag}" for tag in LAG_TAGS]
        (centre / "LACC_times.txt").write_text("\n".join(body) + "\n")
        np.savetxt(centre / "clear_sky_mask.txt", self.original_mask, fmt="%d")
        np.savetxt(centre / "obs_d01_ch4_totalline_withpert.txt", self.obs_value, fmt="%.10f")
        if not self.write_per_lag:
            return
        for lag_index, tag in enumerate(LAG_TAGS):
            directory = self.obs_bt_dir / f"BT_{tag}"
            directory.mkdir(parents=True, exist_ok=True)
            column = self.nr_bt_by_lag[lag_index].copy()
            np.savetxt(directory / "obs_d01_ch4_totalline.txt", column, fmt="%.10f")
            np.savetxt(
                directory / "obs_d01_ch4.txt",
                column.reshape(self.side, self.side, order="F"),
                fmt="%.10f",
            )
            np.savetxt(directory / "clear_sky_mask.txt", self.original_mask, fmt="%d")

    def _write_member_hx(self) -> None:
        for member in range(1, self.members + 1):
            member_root = self.ens_bt_dir / f"mem{member:03d}" / "AMSUA"
            if self.write_per_lag:
                for lag_index, tag in enumerate(LAG_TAGS):
                    directory = member_root / f"BT_{tag}"
                    directory.mkdir(parents=True, exist_ok=True)
                    column = np.array(
                        [
                            self.lag_hx_members(raw, lag_index)[member - 1]
                            for raw in range(1, self.nobs + 1)
                        ]
                    )
                    np.savetxt(directory / "obs_d01_ch4_totalline.txt", column, fmt="%.10f")
                    np.savetxt(
                        directory / "obs_d01_ch4.txt",
                        column.reshape(self.side, self.side, order="F"),
                        fmt="%.10f",
                    )
            directory = member_root / f"BT_LACC_{CENTER_TAG}"
            directory.mkdir(parents=True, exist_ok=True)
            column = np.array(
                [self.hx_members(raw)[member - 1] for raw in range(1, self.nobs + 1)]
            )
            np.savetxt(directory / "obs_d01_ch4_totalline.txt", column, fmt="%.10f")
            np.savetxt(
                directory / "obs_d01_ch4.txt",
                column.reshape(self.side, self.side, order="F"),
                fmt="%.10f",
            )

    def _write_state(self) -> None:
        for member in range(1, self.members + 1):
            write_state_file(
                self.background_dir / f"firstguess_d01.mem{member:03d}",
                BACKGROUND_LAT,
                BACKGROUND_LON,
                _state_field(
                    BACKGROUND_LAT,
                    BACKGROUND_LON,
                    BACKGROUND_BASE_K + float(self.sst_base[member - 1]),
                    BACKGROUND_LAT_SLOPE,
                    BACKGROUND_LON_SLOPE,
                ),
            )
        write_state_file(
            self.nr_file,
            NR_LAT,
            NR_LON,
            _state_field(
                NR_LAT, NR_LON, NR_BASE_K + self.nr_offset, self.nr_lat_slope, NR_LON_SLOPE
            ),
        )

    # ------------------------------------------------------------- readers
    def find_row(self, **criteria) -> int:
        """The first raw observation (1-based) matching a designed case."""
        for raw in np.flatnonzero(self.original_mask) + 1:
            j = raw - 1
            if criteria.get("c_sign") is not None and int(np.sign(self.covariance(raw))) != criteria["c_sign"]:
                continue
            if criteria.get("d_sign") is not None and int(np.sign(self.innovation(raw))) != criteria["d_sign"]:
                continue
            if criteria.get("e_sign") is not None and int(np.sign(self.sst_error(raw))) != criteria["e_sign"]:
                continue
            if criteria.get("below_centre") is not None and bool(self.latitude[j] < LAT_CENTER) != criteria["below_centre"]:
                continue
            return int(raw)
        raise AssertionError(f"no observation matches {criteria}")

    def audit_rows(self) -> list[dict]:
        with (self.output_dir / filt.AUDIT_FILENAME).open() as stream:
            return list(csv.DictReader(stream))

    def audit_row(self, raw_index: int) -> dict:
        return self.audit_rows()[raw_index - 1]

    def mask_values(self) -> np.ndarray:
        text = (self.output_dir / filt.MASK_FILENAME).read_text()
        return np.array([int(line) for line in text.split()], dtype=int)

    def summary(self) -> dict:
        def reject(value):  # strict JSON: no NaN / Infinity tokens
            raise AssertionError(f"non-standard JSON constant {value}")

        return json.loads(
            (self.output_dir / filt.SUMMARY_FILENAME).read_text(), parse_constant=reject
        )

    def outputs(self) -> list[Path]:
        return sorted(p for p in self.output_dir.glob("*") if p.is_file())


# =============================================================================
# Small mutators, one per failure mode
# =============================================================================
def set_times(path: Path, stamp: str) -> None:
    with nc.Dataset(path, "r+") as dataset:
        dataset.variables["Times"][:] = np.array(
            [[char.encode("ascii") for char in stamp]], dtype="S1"
        )


def set_units(path: Path, text: str) -> None:
    with nc.Dataset(path, "r+") as dataset:
        dataset.variables["OM_TMP"].units = text


def drop_units(path: Path) -> None:
    with nc.Dataset(path, "r+") as dataset:
        dataset.variables["OM_TMP"].delncattr("units")


def fill_level0(path: Path, value: float) -> None:
    with nc.Dataset(path, "r+") as dataset:
        dataset.variables["OM_TMP"][0, 0, :, :] = value


def bump_xlat(path: Path, delta: float) -> None:
    with nc.Dataset(path, "r+") as dataset:
        dataset.variables["XLAT"][0, 0, 0] = float(dataset.variables["XLAT"][0, 0, 0]) + delta


def set_landmask(path: Path, lat_box, lon_box) -> None:
    with nc.Dataset(path, "r+") as dataset:
        lat = np.asarray(dataset.variables["XLAT"][0])
        lon = np.asarray(dataset.variables["XLONG"][0])
        inside = (
            (lat >= lat_box[0]) & (lat <= lat_box[1]) & (lon >= lon_box[0]) & (lon <= lon_box[1])
        )
        mask = np.asarray(dataset.variables["LANDMASK"][0])
        mask[inside] = 1.0
        dataset.variables["LANDMASK"][0] = mask


def write_c_order_column(path: Path, side: int) -> None:
    """Rewrite a one-column Hx file as the C-order flattening of its own matrix.

    The matched layout is the F-order one; the C-order vector is a permutation of
    it, so the values stay plausible while every index refers to another pixel.
    """
    values = np.loadtxt(path)
    matrix = values.reshape(side, side, order="F")
    wrong = matrix.reshape(-1, order="C")
    assert not np.array_equal(wrong, values)
    path.write_text("\n".join(f"{value:.10f}" for value in wrong) + "\n")


def write_matrix_at_column_path(path: Path, side: int) -> None:
    """Put the 2D matrix where the one-column file belongs."""
    matrix = np.loadtxt(path).reshape(side, side, order="F")
    np.savetxt(path, matrix, fmt="%.10f")


def shift_one_profile_point(path: Path, raw_index: int, lat: float, lon: float) -> None:
    lines = path.read_text().splitlines()
    lines[186 + 185 * (raw_index - 1) - 1] = f"  0.0100  {lat:.6f}  {lon:.6f}"
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# 1-5: the formula, exercised directly
# =============================================================================
class DirectionFormulaTests(unittest.TestCase):
    def test_finite_inputs_with_nonfinite_computed_moments_are_refused(self):
        cases = (
            ([-1e308, 1e308], [-1e308, 1e308], 1., 1.),
            ([-1e200, -1e200, 1e200, 1e200],
             [-1e200, 1e200, -1e200, 1e200], 1., 1.),
            ([1e308, 1e308], [0., 1.], 1., -1e308),
        )
        for x, h, y, nr in cases:
            with self.subTest(x=x, h=h):
                with self.assertRaises(filt.DirectionFilterError):
                    filt.evaluate_direction(np.array(x), np.array(h), y, nr)

    def test_every_sign_combination_of_c_d_e(self):
        """Keep iff c*d*e >= 0, including the c<0, d<0 "correct direction" case."""
        # mean([-1, 0, 1, 0]) is exactly 0.0, so h_bar and x_bar are exact.
        x_varying = np.array([-1.0, 0.0, 1.0, 0.0])
        h_varying = np.array([-1.0, 0.0, 1.0, 0.0])
        x_constant = np.full(4, 250.0001)
        h_constant = np.full(4, 251.0)
        for sign_c in (+1, 0, -1):
            for sign_d in (+1, 0, -1):
                for sign_e in (+1, 0, -1):
                    with self.subTest(c=sign_c, d=sign_d, e=sign_e):
                        x = x_varying if sign_c != 0 else x_constant
                        h = h_varying * sign_c if sign_c != 0 else h_constant
                        h_bar = 0.0 if sign_c != 0 else 251.0
                        y = h_bar + sign_d * 1.0
                        x_nr = float(x.mean()) + sign_e * 1.0
                        result = filt.evaluate_direction(x, h, y, x_nr)
                        observed = (
                            np.sign(result.cov_sst_hx),
                            np.sign(result.innovation),
                            np.sign(result.sst_error),
                        )
                        self.assertEqual(
                            observed,
                            (sign_c, sign_d, sign_e),
                            "the constructed case must reach the intended signs",
                        )
                        negatives = sum(
                            1 for sign in (sign_c, sign_d, sign_e) if sign < 0
                        )
                        expected_keep = not (
                            all(sign != 0 for sign in (sign_c, sign_d, sign_e))
                            and negatives % 2 == 1
                        )
                        self.assertEqual(result.keep, expected_keep)
                        if any(sign == 0 for sign in (sign_c, sign_d, sign_e)):
                            self.assertEqual(result.status, filt.STATUS_KEPT_ZERO_DIRECTION)
                        elif expected_keep:
                            self.assertEqual(result.status, filt.STATUS_KEPT_NONREVERSE)
                        else:
                            self.assertEqual(result.status, filt.STATUS_REJECTED_REVERSE)

    def test_the_explicit_c_negative_d_negative_cases(self):
        x = np.array([-2.0, -1.0, 1.0, 2.0])
        h = np.array([-2.0, -1.0, 1.0, 2.0])
        # c < 0 (the coupling is reversed), d < 0, e > 0 -> KEEP.
        kept = filt.evaluate_direction(x, -h, -0.5, 2.0)
        self.assertLess(kept.cov_sst_hx, 0.0)
        self.assertLess(kept.innovation, 0.0)
        self.assertGreater(kept.sst_error, 0.0)
        self.assertTrue(kept.keep)
        self.assertEqual(kept.status, filt.STATUS_KEPT_NONREVERSE)
        # c < 0, d < 0, e < 0 -> the update points away from the NR: DELETE.
        deleted = filt.evaluate_direction(x, -h, -0.5, -2.0)
        self.assertLess(deleted.cov_sst_hx, 0.0)
        self.assertLess(deleted.innovation, 0.0)
        self.assertLess(deleted.sst_error, 0.0)
        self.assertFalse(deleted.keep)
        self.assertEqual(deleted.status, filt.STATUS_REJECTED_REVERSE)

    def test_identical_decimal_members_have_exactly_zero_covariance(self):
        decimal = 250.0001
        x = np.full(5, decimal)
        self.assertNotEqual(float(x.mean()), decimal, "the naive mean is not exact here")
        h = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = filt.evaluate_direction(x, h, 260.0, 251.0)
        self.assertEqual(result.cov_sst_hx, 0.0)
        self.assertEqual(result.sst_background_mean, decimal)
        self.assertEqual(result.status, filt.STATUS_KEPT_ZERO_DIRECTION)
        self.assertTrue(result.keep)
        # ... and the same the other way round: identical Hx members.
        result = filt.evaluate_direction(h, x, decimal, 251.0)
        self.assertEqual(result.cov_sst_hx, 0.0)
        self.assertEqual(result.hx_mean, decimal)
        self.assertEqual(result.innovation, 0.0)
        self.assertEqual(result.status, filt.STATUS_KEPT_ZERO_DIRECTION)

    def test_zero_factors_are_kept(self):
        x = np.array([-1.0, 0.0, 1.0, 0.0])
        h = np.array([-1.0, 0.0, 1.0, 0.0])
        # e == 0 exactly (x_nr == x_bar) while c and d are non-zero -> keep.
        result = filt.evaluate_direction(x, h, 1.0, 0.0)
        self.assertEqual(result.sst_error, 0.0)
        self.assertNotEqual(result.cov_sst_hx, 0.0)
        self.assertTrue(result.keep)
        self.assertEqual(result.status, filt.STATUS_KEPT_ZERO_DIRECTION)
        # d == 0 exactly (constant Hx members, y == h_bar) -> keep.
        result = filt.evaluate_direction(x, np.full(4, 3.5), 3.5, 1.0)
        self.assertEqual(result.innovation, 0.0)
        self.assertTrue(result.keep)
        self.assertEqual(result.status, filt.STATUS_KEPT_ZERO_DIRECTION)

    def test_an_underflowed_product_cannot_hide_a_reverse_direction(self):
        """c, d and e tiny but non-zero: c*d*e underflows, the sign test must not."""
        x = np.array([-1.0e-100, 0.0, 1.0e-100])
        h = np.array([-2.0e-100, 0.0, 2.0e-100])
        # c > 0, d < 0, e > 0: one negative factor, so the product is negative --
        # and it underflows to zero.
        result = filt.evaluate_direction(x, h, -1.0e-100, 1.0e-100)
        self.assertGreater(result.cov_sst_hx, 0.0)
        self.assertLess(result.innovation, 0.0)
        self.assertGreater(result.sst_error, 0.0)
        self.assertEqual(result.score, 0.0, "the product underflows to exactly zero")
        # A "score < 0" test would have kept this observation.
        self.assertFalse(result.keep)
        self.assertTrue(result.reverse)
        self.assertEqual(result.status, filt.STATUS_REJECTED_REVERSE)

    def test_a_same_direction_overshoot_is_kept(self):
        x = np.array([-10.0, -5.0, 5.0, 10.0])
        h = np.array([-10.0, -5.0, 5.0, 10.0])
        result = filt.evaluate_direction(x, h, 1000.0, 0.001)
        # Huge coupling and a huge innovation against a near-zero NR error: the
        # implied increment overshoots by orders of magnitude, and no amplitude,
        # overshoot or correlation threshold exists to delete it.
        self.assertLess(abs(result.sst_error), 1.0e-2)
        self.assertGreater(abs(result.cov_sst_hx * result.innovation), 1.0e4)
        self.assertGreater(abs(result.cov_sst_hx * result.innovation / result.sst_error), 1.0e7)
        self.assertTrue(result.keep)
        self.assertEqual(result.status, filt.STATUS_KEPT_NONREVERSE)

    def test_refuses_non_finite_and_unpaired_member_vectors(self):
        x = np.array([-1.0, 0.0, 1.0, 0.0])
        h = np.array([-1.0, 0.0, 1.0, 0.0])
        for bad_x, bad_h, bad_y, bad_nr in (
            (np.array([1.0, np.nan]), h, 0.0, 0.0),
            (x, h, float("nan"), 0.0),
            (x, h, 0.0, float("inf")),
            (x[:3], h, 0.0, 0.0),
            (np.array([1.0]), np.array([2.0]), 0.0, 0.0),
        ):
            with self.subTest(bad_x=bad_x.shape, bad_y=bad_y):
                with self.assertRaises(filt.DirectionFilterError):
                    filt.evaluate_direction(bad_x, bad_h, bad_y, bad_nr)


# =============================================================================
# The filter against the synthetic files
# =============================================================================
class FilterFixtureCase(unittest.TestCase):
    """A fresh four-member synthetic experiment per test (no test methods here)."""

    members = 4

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.fx = Fixture(self.root, members=self.members).write()

    def run_filter(self, fixture: "Fixture | None" = None, **overrides) -> dict:
        fixture = fixture or self.fx
        config = fixture.config
        if overrides:
            config = dataclasses.replace(config, **overrides)
        return filt.run(config)


class FilterRunTests(FilterFixtureCase):
    # -------------------------------------------------------------- basics
    def test_analytic_fields_are_sampled_at_the_observation_points(self):
        """Background and NR live on different grids; both are interpolated."""
        payload = self.run_filter()
        self.assertEqual(payload["counts"]["n_raw"], self.fx.nobs)
        with nc.Dataset(self.fx.nr_file) as dataset:
            nr_shape = tuple(dataset.variables["OM_TMP"][0, 0].shape)
        with nc.Dataset(self.fx.background_dir / "firstguess_d01.mem001") as dataset:
            background_shape = tuple(dataset.variables["OM_TMP"][0, 0].shape)
        self.assertNotEqual(nr_shape, background_shape, "the two grids must differ")
        self.assertEqual(
            payload["checks"]["times_and_grid"]["latitude_grid"], list(background_shape)
        )

        clear = self.fx.clear_sky_raw_ids()
        mismatches = []
        for raw in (clear[0], clear[len(clear) // 3], clear[-2], clear[-1]):
            row = self.fx.audit_row(raw)
            self.assertAlmostEqual(float(row["sst_background_mean"]), self.fx.x_bar(raw), places=9)
            self.assertAlmostEqual(float(row["sst_nr"]), self.fx.nr_value(raw), places=9)
            self.assertAlmostEqual(
                float(row["sst_error_nr_minus_background"]), self.fx.sst_error(raw), places=9
            )
            self.assertAlmostEqual(float(row["hx_mean"]), self.fx.h_bar(raw), places=9)
            self.assertAlmostEqual(float(row["innovation"]), self.fx.innovation(raw), places=9)
            self.assertAlmostEqual(float(row["cov_sst_hx"]), self.fx.covariance(raw), places=9)
            self.assertAlmostEqual(float(row["direction_score"]), self.fx.score(raw), places=9)
            self.assertAlmostEqual(
                float(row["actual_observation"]), float(self.fx.obs_value[raw - 1]), places=9
            )
            self.assertAlmostEqual(float(row["latitude"]), self.fx.latitude[raw - 1], places=9)
            self.assertAlmostEqual(float(row["longitude"]), self.fx.longitude[raw - 1], places=9)
            # The NR value at the same ARRAY INDEX is a different physical point.
            with nc.Dataset(self.fx.nr_file) as dataset:
                indexed = float(
                    dataset.variables["OM_TMP"][0, 0, (raw - 1) % nr_shape[0], (raw - 1) // nr_shape[0] % nr_shape[1]]
                )
            mismatches.append(abs(indexed - float(row["sst_nr"])))
        self.assertGreater(max(mismatches), 1.0, "array indices are not used for the NR")

    def test_the_mask_and_the_counts_follow_the_designed_decisions(self):
        original = self.fx.obs_bt_dir / f"BT_LACC_{CENTER_TAG}" / "clear_sky_mask.txt"
        original_bytes = original.read_bytes()
        payload = self.run_filter()
        expected = np.array(
            [int(self.fx.expected_keep(raw)) for raw in range(1, self.fx.nobs + 1)]
        )
        np.testing.assert_array_equal(self.fx.mask_values(), expected)
        counts = payload["counts"]
        self.assertEqual(counts["final_kept"], int(expected.sum()))
        self.assertEqual(
            counts["excluded_by_original_mask"], int((self.fx.original_mask == 0).sum())
        )
        self.assertEqual(
            counts["rejected_reverse"],
            sum(
                1
                for raw in range(1, self.fx.nobs + 1)
                if self.fx.expected_status(raw) == filt.STATUS_REJECTED_REVERSE
            ),
        )
        self.assertEqual(
            counts["kept_nonreverse"] + counts["kept_zero_direction"], counts["final_kept"]
        )
        self.assertGreater(counts["rejected_reverse"], 0)
        self.assertGreater(counts["final_kept"], 0)
        # The mask file: fixed nobs rows, 0/1 only, trailing newline.
        text = (self.fx.output_dir / filt.MASK_FILENAME).read_text()
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(len(text.splitlines()), self.fx.nobs)
        self.assertTrue(set(text.split()) <= {"0", "1"})
        # The original clear-sky mask is never modified.
        self.assertEqual(original.read_bytes(), original_bytes)
        # Every audit row agrees with the designed status.
        self.assertEqual(len(self.fx.audit_rows()), self.fx.nobs)
        for row in self.fx.audit_rows():
            raw = int(row["raw_obs_id_1based"])
            self.assertEqual(row["status"], self.fx.expected_status(raw))
            self.assertEqual(int(row["keep_final"]), int(self.fx.expected_keep(raw)))

    def test_running_twice_from_the_original_mask_does_not_compound(self):
        first = self.run_filter()
        first_mask = self.fx.mask_values().copy()
        first_audit = (self.fx.output_dir / filt.AUDIT_FILENAME).read_bytes()
        second = self.run_filter()
        self.assertEqual(first["counts"], second["counts"])
        np.testing.assert_array_equal(self.fx.mask_values(), first_mask)
        self.assertEqual((self.fx.output_dir / filt.AUDIT_FILENAME).read_bytes(), first_audit)
        self.assertEqual(
            first["counts"]["n_original_clear_sky"], int((self.fx.original_mask == 1).sum())
        )

    def test_zero_direction_observations_are_kept_with_their_own_status(self):
        raw = self.fx.find_row(c_sign=-1)
        self.fx.slope[raw - 1] = 0.0  # constant Hx members -> c is exactly 0
        self.fx.obs_value[raw - 1] = float(self.fx.hx_base[raw - 1])  # d is exactly 0
        self.fx.write()
        self.assertEqual(self.fx.expected_status(raw), filt.STATUS_KEPT_ZERO_DIRECTION)
        self.run_filter()
        row = self.fx.audit_row(raw)
        self.assertEqual(row["status"], filt.STATUS_KEPT_ZERO_DIRECTION)
        self.assertEqual(row["cov_sst_hx"], "0")
        self.assertEqual(int(row["keep_final"]), 1)

    def test_a_same_direction_overshoot_keeps_the_observation(self):
        """A correctly directed but oversized implied increment is not deleted."""
        # c > 0 with e < 0 needs d < 0 to point toward the NR; the innovation is
        # then fifty times the NR error, i.e. an update that overshoots wildly.
        raw = self.fx.find_row(c_sign=1, e_sign=-1)
        self.assertLess(self.fx.innovation(raw), 0.0)
        self.fx.obs_value[raw - 1] = float(self.fx.hx_base[raw - 1]) - 50.0
        self.fx.write()
        self.assertEqual(self.fx.expected_status(raw), filt.STATUS_KEPT_NONREVERSE)
        self.run_filter()
        row = self.fx.audit_row(raw)
        self.assertEqual(row["status"], filt.STATUS_KEPT_NONREVERSE)
        self.assertEqual(int(row["keep_final"]), 1)
        # The screen applied no amplitude test: the reported score stays that of
        # a large coupling times a large innovation, and the NR error is small.
        self.assertGreater(
            abs(float(row["cov_sst_hx"]) * float(row["innovation"])),
            abs(float(row["sst_error_nr_minus_background"])),
        )

    def test_the_standalone_entry_point_runs_and_refuses_cleanly(self):
        # The file's own entry point runs the top-of-file CONFIG (the driver uses
        # its bridge instead) and returns 0 / non-zero rather than raising.
        with mock.patch.object(filt, "CONFIG", self.fx.config):
            self.assertEqual(filt.main(), 0)
        broken = dataclasses.replace(
            self.fx.config, clear_sky_mask_file=self.root / "does_not_exist.txt"
        )
        with mock.patch.object(filt, "CONFIG", broken):
            self.assertEqual(filt.main(), 2)

    def test_a_single_member_configuration_is_refused(self):
        config = dataclasses.replace(self.fx.config, member_end=1)
        with self.assertRaises(filt.DirectionFilterError) as caught:
            filt.run(config)
        self.assertIn("at least 2 members", str(caught.exception))
        self.assertEqual(self.fx.outputs(), [])

    def test_the_noisy_observation_is_used_even_when_the_noise_flips_the_sign(self):
        """The noise-free centre-lag file carries the opposite innovation."""
        raw = self.fx.find_row(c_sign=-1, d_sign=-1, e_sign=1)
        noise_free = float(self.fx.nr_bt_by_lag[0][raw - 1])
        self.assertGreater(
            noise_free - self.fx.h_bar(raw), 0.0, "the noise-free innovation is positive"
        )
        self.assertEqual(self.fx.expected_status(raw), filt.STATUS_KEPT_NONREVERSE)
        self.run_filter()
        row = self.fx.audit_row(raw)
        self.assertAlmostEqual(
            float(row["actual_observation"]), float(self.fx.obs_value[raw - 1]), places=9
        )
        self.assertLess(float(row["innovation"]), 0.0)
        self.assertEqual(row["status"], filt.STATUS_KEPT_NONREVERSE)
        # A reader that had taken that noise-free value would have rejected this
        # observation (c < 0, d > 0, e > 0).
        self.assertLess(
            self.fx.covariance(raw)
            * (noise_free - self.fx.h_bar(raw))
            * self.fx.sst_error(raw),
            0.0,
        )

    def test_the_final_hx_is_used_not_a_lag_re_average(self):
        self.run_filter()
        for raw in self.fx.clear_sky_raw_ids()[:2] + self.fx.clear_sky_raw_ids()[-1:]:
            row = self.fx.audit_row(raw)
            self.assertAlmostEqual(float(row["hx_mean"]), self.fx.h_bar(raw), places=9)
            for lag_index in range(len(LAG_TAGS)):
                lag_mean = float(np.mean(self.fx.lag_hx_members(raw, lag_index)))
                self.assertGreater(
                    abs(lag_mean - float(row["hx_mean"])),
                    NOISE_FREE_LAG_OFFSET_K / 2.0,
                    "the per-lag data differ by ~1000 K, so a re-average would be obvious",
                )

    def test_members_are_paired_by_member_number(self):
        raw = self.fx.find_row(c_sign=-1, below_centre=True)
        # Guard: an implementation that sorted the two member vectors
        # independently would get the opposite sign here.
        x = self.fx.x_members(raw)
        h = self.fx.hx_members(raw)
        self.assertGreater(float(np.corrcoef(np.sort(x), np.sort(h))[0, 1]), 0.0)
        self.run_filter()
        self.assertLess(float(self.fx.audit_row(raw)["cov_sst_hx"]), 0.0)

        # Swap member 001's and the last member's Hx files: the audit must follow
        # the swapped vectors, which is only true if each value is read from the
        # file named for that member number.
        tag_dir = f"BT_LACC_{CENTER_TAG}"
        first_dir = self.fx.ens_bt_dir / "mem001" / "AMSUA" / tag_dir
        last_dir = self.fx.ens_bt_dir / f"mem{self.fx.members:03d}" / "AMSUA" / tag_dir
        # Swap the whole member, matrix file included: the F-order gate compares
        # each column against its own matrix and would (correctly) refuse a
        # half-swapped pair.
        for name in ("obs_d01_ch4_totalline.txt", "obs_d01_ch4.txt"):
            first_bytes, last_bytes = (first_dir / name).read_bytes(), (last_dir / name).read_bytes()
            (first_dir / name).write_bytes(last_bytes)
            (last_dir / name).write_bytes(first_bytes)
        swapped = self.fx.hx_members(raw).copy()
        swapped[0], swapped[-1] = swapped[-1], swapped[0]
        self.run_filter()
        row = self.fx.audit_row(raw)
        expected_cov = float(
            np.sum((self.fx.x_members(raw) - self.fx.x_bar(raw)) * (swapped - swapped.mean()))
            / (swapped.size - 1)
        )
        self.assertAlmostEqual(float(row["cov_sst_hx"]), expected_cov, places=9)
        self.assertAlmostEqual(float(row["hx_mean"]), float(swapped.mean()), places=9)

    def test_original_mask_zero_rows_cannot_be_re_kept_and_ids_map(self):
        # Force a row the original mask excludes into a "keep" direction; it must
        # still be excluded, with empty statistics and no filtered ids.
        excluded = int(np.flatnonzero(self.fx.original_mask == 0)[0])
        self.fx.slope[excluded] = 1.0
        self.fx.obs_value[excluded] = float(self.fx.hx_base[excluded]) + 0.4
        self.fx.write()
        payload = self.run_filter()
        rows = self.fx.audit_rows()
        statistics = (
            "sst_background_mean",
            "sst_nr",
            "sst_error_nr_minus_background",
            "hx_mean",
            "actual_observation",
            "innovation",
            "cov_sst_hx",
            "direction_score",
        )
        for row in rows:
            if int(row["original_mask"]) == 0:
                self.assertEqual(row["status"], filt.STATUS_EXCLUDED_BY_ORIGINAL_MASK)
                self.assertEqual(row["keep_final"], "0")
                self.assertEqual(row["base_filtered_id_1based"], "")
                self.assertEqual(row["final_filtered_id_1based"], "")
                for column in statistics:
                    self.assertEqual(row[column], "", f"{column} must stay empty")
        # The three numbering conventions are consistent for every other row.
        base_ids = (np.flatnonzero(self.fx.original_mask) + 1).tolist()
        final_ids = self.fx.kept_raw_ids()
        for row in rows:
            raw = int(row["raw_obs_id_1based"])
            if int(row["original_mask"]) == 0:
                continue
            self.assertEqual(int(row["base_filtered_id_1based"]), base_ids.index(raw) + 1)
            if int(row["keep_final"]):
                self.assertEqual(int(row["final_filtered_id_1based"]), final_ids.index(raw) + 1)
            else:
                self.assertEqual(row["final_filtered_id_1based"], "")
        self.assertEqual(payload["counts"]["final_kept"], len(final_ids))
        self.assertIn(filt.STATUS_EXCLUDED_BY_ORIGINAL_MASK, {row["status"] for row in rows})

    def test_the_summary_json_is_strict_and_records_the_contract(self):
        payload = self.run_filter()
        saved = self.fx.summary()
        self.assertEqual(saved["counts"], payload["counts"])
        self.assertEqual(
            saved["static_filter"]["final_mask"], "original_clear_sky_mask AND direction_keep"
        )
        self.assertEqual(saved["config"]["state_var"], "OM_TMP")
        self.assertEqual(saved["config"]["state_level"], 0)
        self.assertEqual(saved["config"]["sample_mode"], "linear")
        self.assertEqual(saved["config"]["local_window_deg"], 1.0)
        self.assertEqual(saved["config"]["max_local_window_deg"], 4.0)
        self.assertEqual(saved["checks"]["window"]["lag_hours"], [0, 3, 6])
        self.assertEqual(saved["checks"]["member_hx_f_order"]["n_verified"], self.fx.members)
        self.assertTrue(
            all(entry["f_order_matches"] for entry in saved["checks"]["member_hx_f_order"]["entries"])
        )
        self.assertEqual(saved["checks"]["temperature_units"]["nr"]["value"], "k")
        self.assertEqual(
            saved["inputs"]["observation_file"], payload["inputs"]["observation_file"]
        )
        self.assertEqual(saved["outputs"]["mask"], payload["outputs"]["mask"])
        self.assertEqual(
            saved["members"]["numbers"], list(range(1, self.fx.members + 1))
        )
        self.assertTrue(saved["members"]["paired_by_member_number"])
        self.assertTrue(
            any("single-observation" in limit for limit in saved["static_filter"]["scope_limits"])
        )
        self.assertIn("underflow", saved["static_filter"]["decision_implementation"])
        self.assertIsInstance(saved["counts"]["final_kept"], int)
        # The audit, the mask and the JSON are the only outputs.
        self.assertEqual(
            {path.name for path in self.fx.outputs()},
            {filt.MASK_FILENAME, filt.AUDIT_FILENAME, filt.SUMMARY_FILENAME},
        )

    def test_all_deleted_still_writes_the_audit_and_refuses(self):
        # c < 0 everywhere, d < 0 everywhere and an NR offset that makes e < 0
        # everywhere: the product is negative for every clear-sky observation.
        self.fx.nr_offset = -5.0
        self.fx.slope[:] = -1.0
        self.fx.obs_value = self.fx.hx_base - 0.4
        self.fx.write()
        for raw in np.flatnonzero(self.fx.original_mask) + 1:
            self.assertEqual(self.fx.expected_status(raw), filt.STATUS_REJECTED_REVERSE)
        with self.assertRaises(filt.NoObservationSurvivesError) as caught:
            self.run_filter()
        payload = caught.exception.payload
        self.assertEqual(payload["counts"]["final_kept"], 0)
        self.assertEqual(payload["counts"]["rejected_reverse"], self.fx.kept)
        for key in ("audit_csv", "mask", "summary_json"):
            self.assertTrue(Path(payload["outputs"][key]).is_file(), key)
        self.assertEqual(int(self.fx.mask_values().sum()), 0)
        self.assertEqual(len(self.fx.audit_rows()), self.fx.nobs)
        self.assertEqual(self.fx.summary()["counts"]["final_kept"], 0)

    def test_a_single_surviving_observation_still_has_correct_files(self):
        # One clear-sky row out of a full-size file set: the observation file and
        # every member Hx file keep their shape, and the screen needs at least
        # two MEMBERS (not two observations) to be defined.
        small = Fixture(self.root / "small", members=3, side=5, kept=1)
        small.original_mask[:] = 0
        small.original_mask[6] = 1  # one single clear-sky observation
        small.slope[:] = 1.0
        small.obs_value = small.hx_base + 0.4
        small.with_output_dir(small.root / "out").write()
        self.assertTrue(small.expected_keep(7))
        payload = self.run_filter(small)
        self.assertEqual(payload["counts"]["n_original_clear_sky"], 1)
        self.assertEqual(payload["counts"]["final_kept"], 1)
        expected_mask = np.zeros(25, dtype=int)
        expected_mask[6] = 1
        np.testing.assert_array_equal(small.mask_values(), expected_mask)
        self.assertEqual(small.audit_row(7)["final_filtered_id_1based"], "1")
        self.assertEqual(
            (small.obs_bt_dir / f"BT_LACC_{CENTER_TAG}" / "obs_d01_ch4_totalline_withpert.txt")
            .read_text()
            .count("\n"),
            25,
        )

    def test_a_one_observation_matrix_cannot_pass_the_shared_f_order_gate(self):
        """nobs == 1 is refused, loudly, instead of being waved through.

        A 1x1 matrix file round-trips through ``np.loadtxt`` as a 0-d array, so
        the companion diagnostic's ``check_matrix_flattening`` (which this task
        deliberately does not relax) cannot confirm the numbering.  The deployed
        size is 26x26, and the filter must say so rather than accept an
        unverifiable layout.
        """
        one = Fixture(self.root / "one", members=2, side=1, kept=1)
        one.with_output_dir(one.root / "out").write()
        self.assertEqual(
            len(
                (one.obs_bt_dir / f"BT_LACC_{CENTER_TAG}" / "obs_d01_ch4_totalline_withpert.txt")
                .read_text()
                .split()
            ),
            1,
        )
        with self.assertRaises(filt.DirectionFilterError) as caught:
            self.run_filter(one)
        self.assertIn("matrix", str(caught.exception))
        self.assertEqual(one.outputs(), [])


# =============================================================================
# 8-11: refusals
# =============================================================================
class RefusalTests(FilterFixtureCase):
    def setUp(self):
        super().setUp()
        self.centre = self.fx.obs_bt_dir / f"BT_LACC_{CENTER_TAG}"
        self.member_dir = self.fx.ens_bt_dir / "mem002" / "AMSUA" / f"BT_LACC_{CENTER_TAG}"

    def assert_refuses(self, fragment: str, **overrides):
        with self.assertRaises(lag.DiagnosticError) as caught:
            self.run_filter(**overrides)
        self.assertIn(fragment, str(caught.exception))
        self.assertEqual(self.fx.outputs(), [], "a refusal must publish nothing")

    def test_fixed_scientific_settings_cannot_be_overridden(self):
        for changes in (
            {"state_var": "TSK"}, {"state_level": 1},
            {"sample_mode": "nearest"}, {"sample_require_water": False},
            {"require_center_time_match": False},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(filt.DirectionFilterError):
                    filt.validate_config(dataclasses.replace(self.fx.config, **changes))

    def test_a_template_repeating_one_background_member_is_refused(self):
        self.assert_refuses("duplicate", background_pattern="firstguess_d01.mem001")

    def test_background_members_linking_to_the_same_file_are_refused(self):
        first = self.fx.background_dir / "firstguess_d01.mem001"
        other = self.fx.background_dir / "firstguess_d01.mem002"
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                other.unlink()
                if kind == "symlink":
                    other.symlink_to(first)
                else:
                    os.link(first, other)
                self.assert_refuses("duplicate")

    def test_nan_at_the_nearest_water_cell_is_refused(self):
        raw = int(np.flatnonzero(self.fx.original_mask)[0])
        path = self.fx.background_dir / "firstguess_d01.mem002"
        with nc.Dataset(path, "r+") as dataset:
            lat, lon = dataset["XLAT"][0], dataset["XLONG"][0]
            distance = (lat-self.fx.latitude[raw])**2 + (lon-self.fx.longitude[raw])**2
            iy, ix = np.unravel_index(np.argmin(distance), lat.shape)
            dataset["OM_TMP"][0, 0, iy, ix] = np.nan
        self.assert_refuses("non-finite")

    def test_nan_in_a_non_nearest_interpolation_vertex_is_refused(self):
        from scipy.spatial import Delaunay
        grid = lag.read_level_zero_field(
            self.fx.nr_file, "OM_TMP", 0, "XLAT", "XLONG", "auto"
        )
        raw = int(np.flatnonzero(self.fx.original_mask)[0])
        latitude, longitude = self.fx.latitude[raw], self.fx.longitude[raw]
        support = grid.water & (abs(grid.lat-latitude) <= 1.) & (abs(grid.lon-longitude) <= 1.)
        flat_ids = np.flatnonzero(support)
        points = np.column_stack((grid.lon[support], grid.lat[support]))
        triangulation = Delaunay(points)
        simplex = triangulation.find_simplex([longitude, latitude])
        triangle = triangulation.simplices[simplex]
        transform = triangulation.transform[simplex]
        leading_weights = transform[:2].dot(np.array([longitude, latitude])-transform[2])
        weights = np.append(leading_weights, 1.-leading_weights.sum())
        nearest = np.argmin((points[:, 0]-longitude)**2 + (points[:, 1]-latitude)**2)
        # The target may lie on a triangle edge: its zero-weight third vertex
        # does not contribute and must not be used to manufacture this case.
        vertex = next(int(v) for v, w in zip(triangle, weights) if v != nearest and w > 1e-8)
        field = grid.field.copy()
        field.flat[flat_ids[vertex]] = np.nan
        grid = dataclasses.replace(grid, field=field)
        with self.assertRaises(filt.DirectionFilterError):
            filt.sample_or_refuse(grid, latitude, longitude, "NR", raw+1, self.fx.config)

    def test_nan_outside_the_interpolation_support_does_not_change_sampling(self):
        grid = lag.read_level_zero_field(
            self.fx.nr_file, "OM_TMP", 0, "XLAT", "XLONG", "auto"
        )
        raw = int(np.flatnonzero(self.fx.original_mask)[0])
        latitude, longitude = self.fx.latitude[raw], self.fx.longitude[raw]
        expected = filt.sample_or_refuse(grid, latitude, longitude, "NR", raw+1, self.fx.config)
        field = grid.field.copy()
        field[-1, -1] = np.nan
        # Also cover a missing vertex inside the search window which carries
        # no weight at this target, rather than only a remote missing cell.
        local = grid.water & (abs(grid.lat-latitude) <= 1.) & (abs(grid.lon-longitude) <= 1.)
        field.flat[np.flatnonzero(local)[0]] = np.nan
        changed = dataclasses.replace(grid, field=field)
        actual = filt.sample_or_refuse(changed, latitude, longitude, "NR", raw+1, self.fx.config)
        self.assertEqual(actual, expected)

    def test_f_order_mismatch_is_refused(self):
        write_c_order_column(self.member_dir / "obs_d01_ch4_totalline.txt", self.fx.side)
        self.assert_refuses("member 002")

    def test_a_matrix_at_the_column_path_is_refused(self):
        write_matrix_at_column_path(self.member_dir / "obs_d01_ch4_totalline.txt", self.fx.side)
        self.assert_refuses("member 002")

    def test_a_missing_member_column_is_refused(self):
        (self.member_dir / "obs_d01_ch4_totalline.txt").unlink()
        self.assert_refuses("member 002")

    def test_a_missing_member_matrix_is_refused(self):
        (self.member_dir / "obs_d01_ch4.txt").unlink()
        self.assert_refuses("member 002")

    def test_a_missing_background_member_file_is_refused(self):
        (self.fx.background_dir / "firstguess_d01.mem002").unlink()
        self.assert_refuses("firstguess_d01.mem002")

    def test_a_wrong_background_valid_time_is_refused(self):
        set_times(self.fx.background_dir / "firstguess_d01.mem002", "2018-09-09_00:00:00")
        self.assert_refuses("background member 2")

    def test_a_wrong_nr_valid_time_is_refused(self):
        set_times(self.fx.nr_file, "2018-09-11_00:00:00")
        self.assert_refuses("NR")

    def test_a_member_grid_mismatch_is_refused(self):
        bump_xlat(self.fx.background_dir / "firstguess_d01.mem003", 0.5)
        self.assert_refuses("background member 3")

    def test_non_finite_state_values_are_refused(self):
        fill_level0(self.fx.background_dir / "firstguess_d01.mem002", float("nan"))
        self.assert_refuses("member 002")

    def test_a_non_kelvin_unit_is_refused(self):
        set_units(self.fx.background_dir / "firstguess_d01.mem001", "celsius")
        self.assert_refuses("celsius")

    def test_a_missing_unit_attribute_is_refused(self):
        drop_units(self.fx.background_dir / "firstguess_d01.mem001")
        self.assert_refuses("units")

    def test_a_non_ocean_observation_point_is_refused(self):
        for member in range(1, self.fx.members + 1):
            set_landmask(
                self.fx.background_dir / f"firstguess_d01.mem{member:03d}",
                (LAT_CENTER - 0.2, LAT_CENTER + 0.2),
                (LON_CENTER - 0.2, LON_CENTER + 0.2),
            )
        with self.assertRaises(filt.DirectionFilterError) as caught:
            self.run_filter()
        message = str(caught.exception)
        self.assertIn("not water", message)
        self.assertIn("raw observation", message)
        self.assertIn("background member", message)
        self.assertEqual(self.fx.outputs(), [])

    def test_an_observation_outside_the_grid_is_refused_without_fallback(self):
        raw = int(np.flatnonzero(self.fx.original_mask)[0]) + 1
        shift_one_profile_point(self.fx.profile_file, raw, 10.0, 140.0)
        with self.assertRaises(filt.DirectionFilterError) as caught:
            self.run_filter()
        self.assertIn(f"raw observation {raw}", str(caught.exception))
        self.assertIn("NR", str(caught.exception))
        self.assertEqual(self.fx.outputs(), [])

    def test_a_non_finite_profile_coordinate_is_refused(self):
        shift_one_profile_point(self.fx.profile_file, 1, float("nan"), 147.4)
        self.assert_refuses("non-finite")

    def test_a_wrong_window_or_centre_time_is_refused(self):
        path = self.centre / "LACC_times.txt"
        path.write_text("center_time=10_00_00\nlag_time=10_00_00\nlag_time=09_21_00\n")
        self.assert_refuses("window")
        path.write_text(
            "center_time=09_00_00\nlag_time=10_00_00\nlag_time=09_21_00\nlag_time=09_18_00\n"
        )
        self.assert_refuses("center_time")
        path.write_text("center_time=10_00_00\nlag_time=10_00_00\n")
        self.assert_refuses("window")

    def test_a_repeated_lag_time_is_refused(self):
        path = self.centre / "LACC_times.txt"
        path.write_text(
            "center_time=10_00_00\nlag_time=10_00_00\nlag_time=09_21_00\n"
            "lag_time=09_18_00\nlag_time=10_00_00\n"
        )
        self.assert_refuses("repeats")


# =============================================================================
# 15: the two text converters consume the merged mask
# =============================================================================
class ConverterIntegrationTests(unittest.TestCase):
    members = 50

    @classmethod
    def setUpClass(cls):
        cls._template = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._template.cleanup)
        cls.template_root = Path(cls._template.name) / "experiment"
        Fixture(cls.template_root, members=cls.members, write_per_lag=False).write()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        shutil.copytree(self.template_root, self.root / "experiment", symlinks=True)
        # Analytics only: the files come from the template copy.
        self.fx = Fixture(self.root / "experiment", members=self.members, write_per_lag=False)
        self.payload = filt.run(self.fx.config)
        self.kept_raw = self.fx.kept_raw_ids()
        self.assertEqual(self.payload["counts"]["final_kept"], len(self.kept_raw))
        self.assertLess(len(self.kept_raw), int(self.fx.original_mask.sum()))
        self.assertGreater(len(self.kept_raw), 0)

    def _convert_env(self):
        env = dict(os.environ)
        for name in ("OBS_INPUT_FILE", "CLEAR_SKY_MASK_FILE", "ADAPTIVE_ACTIVE", "ADAPTIVE_LACC_DIR"):
            env.pop(name, None)
        env.update(
            {
                "domain": "d01",
                "sensor": "AMSUA",
                "assim_channel": "4",
                "NOBS": str(self.fx.nobs),
                "ENS_SIZE": str(self.members),
                "USE_CLEAR_SKY_MASK": "1",
                "current_day": "10",
                "current_hour": "00",
                "current_min": "00",
                "current_time": CENTER_TAG,
                "EXPECTED_LACC_COUNT": "3",
                "OBS_ERR_STD": "0.5",
                "HGT_OBS": "85000",
                "OBS_BT_DIR": str(self.fx.obs_bt_dir),
                "OBS_BT_SUBDIR": f"BT_LACC_{CENTER_TAG}",
                "PROFILE_DIR": str(self.fx.hx_root / "profile"),
                "PROFILE_SUBDIR": f"profile_d01_LACC_{CENTER_TAG}",
                "PARA_FILE": str(self.fx.profile_file),
                "LACC_TIMES_FILE": str(
                    self.fx.obs_bt_dir / f"BT_LACC_{CENTER_TAG}" / "LACC_times.txt"
                ),
                "ENS_BT_DIR": str(self.fx.ens_bt_dir),
                "ENS_BT_SUBDIR": f"BT_LACC_{CENTER_TAG}",
                "CLEAR_SKY_MASK_FILE": self.payload["outputs"]["mask"],
                "OBS_INPUT_FILE": self.payload["inputs"]["observation_file"],
                "OBS_DART_INPUT_FILE": str(self.root / "obs_input.txt"),
                "MERGED_FO_FILE": str(self.root / "merged_FO_data.txt"),
                "CONVERT_TOP_DIR": str(self.root / "convert_top"),
                "MPLCONFIGDIR": os.environ["MPLCONFIGDIR"],
            }
        )
        return env

    def test_both_converters_use_the_merged_mask_row_for_row(self):
        env = self._convert_env()
        for script in ("obs2DART_LACC.py", "merge_FO_in_one_file.py"):
            finished = subprocess.run(
                [sys.executable, str(CONVERT_DIR / script)],
                env=env,
                capture_output=True,
                text=True,
                cwd=str(self.root),
            )
            self.assertEqual(finished.returncode, 0, finished.stderr + finished.stdout)

        # ---- observations: one row per kept raw observation, ascending ---------
        rows = [line.split() for line in Path(env["OBS_DART_INPUT_FILE"]).read_text().splitlines()]
        self.assertEqual(len(rows), len(self.kept_raw))
        for position, raw in enumerate(self.kept_raw):
            self.assertAlmostEqual(float(rows[position][10]), float(self.fx.obs_value[raw - 1]), places=4)
            self.assertAlmostEqual(float(rows[position][1]), float(self.fx.latitude[raw - 1]), places=4)
            self.assertAlmostEqual(float(rows[position][2]), float(self.fx.longitude[raw - 1]), places=4)
        expected_obs = [float(self.fx.obs_value[raw - 1]) for raw in self.kept_raw]
        self.assertEqual(len(set(expected_obs)), len(expected_obs), "values must be unique")
        np.testing.assert_allclose([float(row[10]) for row in rows], expected_obs, atol=1.0e-4)

        # ---- member Hx: same rows, same order, every member column ------------
        merged = np.loadtxt(env["MERGED_FO_FILE"])
        self.assertEqual(merged.shape, (len(self.kept_raw), self.members))
        for position, raw in enumerate(self.kept_raw):
            np.testing.assert_allclose(merged[position], self.fx.hx_members(raw), atol=1.0e-7)
        # No excluded row leaked into either file.
        excluded_values = [
            float(self.fx.obs_value[raw - 1])
            for raw in range(1, self.fx.nobs + 1)
            if not self.fx.expected_keep(raw)
        ]
        first_column = [float(row[10]) for row in rows]
        for value in excluded_values:
            self.assertFalse(
                any(abs(value - present) < 1.0e-5 for present in first_column),
                "an observation the merged mask excludes must not appear",
            )

    def test_the_inputs_are_left_alone_and_the_mask_is_the_only_switch(self):
        mask_path = self.fx.obs_bt_dir / f"BT_LACC_{CENTER_TAG}" / "clear_sky_mask.txt"
        hx_path = (
            self.fx.ens_bt_dir / "mem001" / "AMSUA" / f"BT_LACC_{CENTER_TAG}" / "obs_d01_ch4_totalline.txt"
        )
        before_mask, before_hx = mask_path.read_bytes(), hx_path.read_bytes()
        env = self._convert_env()
        subprocess.run(
            [sys.executable, str(CONVERT_DIR / "merge_FO_in_one_file.py")],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(self.root),
            check=True,
        )
        self.assertEqual(mask_path.read_bytes(), before_mask)
        self.assertEqual(hx_path.read_bytes(), before_hx)

        # With the ORIGINAL mask the merge keeps the unfiltered row count: the
        # merged mask is what narrows the set, not the converter.
        env["CLEAR_SKY_MASK_FILE"] = str(mask_path)
        env["MERGED_FO_FILE"] = str(self.root / "merged_original.txt")
        subprocess.run(
            [sys.executable, str(CONVERT_DIR / "merge_FO_in_one_file.py")],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(self.root),
            check=True,
        )
        self.assertEqual(
            np.loadtxt(env["MERGED_FO_FILE"]).shape,
            (int(self.fx.original_mask.sum()), self.members),
        )


# =============================================================================
# 16-18: the shell driver
# =============================================================================
class DriverIntegrationTests(unittest.TestCase):
    members = 50

    @classmethod
    def setUpClass(cls):
        cls._template = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._template.cleanup)
        cls.template_root = Path(cls._template.name) / "server_root"
        Fixture(cls.template_root, members=cls.members, write_per_lag=False).write()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.base = self.root / "server_root"
        shutil.copytree(self.template_root, self.base, symlinks=True)
        self.fx = Fixture(self.base, members=self.members, write_per_lag=False)

        self.convert_dir = self.base / "4assimilation" / "1convert_obs"
        self.run_dir = self.convert_dir / "run_dir"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "obs2DART_LACC.py",
            "merge_FO_in_one_file.py",
            "filter_LACC_sst_direction.py",
        ):
            (self.convert_dir / name).symlink_to(CONVERT_DIR / name)
        (self.run_dir / "rttov_sensor_db.csv").write_text("dummy sensor database\n")
        (self.run_dir / "input.nml").write_text(
            " &obs_sequence_nml\n"
            "    text_input_file = 'placeholder.txt',\n"
            "    FO_input_file   = 'placeholder_FO.txt',\n"
            "    obs_out_file    = 'placeholder_out.txt',\n"
            " /\n"
        )
        # A local stub for the Fortran text_to_obs: it records every call, so a
        # test can prove the driver never reached it.
        self.call_log = self.root / "text_to_obs_calls.txt"
        stub = self.run_dir / "text_to_obs"
        stub.write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            f"echo called >> {self.call_log}\n"
            "ti=$(awk -F\"'\" '/text_input_file/ {print $2}' input.nml)\n"
            "oo=$(awk -F\"'\" '/obs_out_file/ {print $2}' input.nml)\n"
            'n=$(wc -l < "$ti")\n'
            "printf ' obs_sequence\\nnum_obs: %s\\n' \"$n\" > \"$oo\"\n"
        )
        stub.chmod(0o755)

        self.filter_dir = (
            self.convert_dir / "single_assim_test" / f"LACC_{CENTER_TAG}_ch4_nr_sst_direction"
            / "nr_sst_direction_filter"
        )
        self.work_dir = self.filter_dir.parent
        self.baseline_work_dir = (
            self.convert_dir / "single_assim_test" / f"LACC_{CENTER_TAG}_ch4"
        )
        self.obs_seq_on = self.run_dir / "obs_seq.out_driver_test_nr_sst_direction"
        self.obs_seq_off = self.run_dir / "obs_seq.out_driver_test"

    def driver_env(self, **overrides) -> dict:
        env = dict(os.environ)
        for name in (
            "OBS_INPUT_FILE",
            "CLEAR_SKY_MASK_FILE",
            "ADAPTIVE_ACTIVE",
            "ADAPTIVE_LACC_DIR",
            "EXPERIMENT_MODE",
            "USE_NR_SST_DIRECTION_FILTER",
            "OptimizWeight",
        ):
            env.pop(name, None)
        env.update(
            {
                "BASE_DIR": str(self.base),
                "PYTHON_BIN": sys.executable,
                "EXPERIMENT_MODE": "LACC",
                "USE_CLEAR_SKY_MASK": "1",
                "OptimizWeight": "0",
                "RUN_TRUTH": "0",
                "RUN_ENSEMBLE": "0",
                "RUN_OBS_CONVERT": "1",
                "RUN_HX_MERGE": "1",
                "RUN_TEXT_TO_OBS": "1",
                "RUN_VALIDATION": "1",
                "OVERWRITE_OBS_SEQ": "1",
                "OBS_SEQ_OUT_NAME": "obs_seq.out_driver_test",
                "CONVERT_TOP_DIR": str(self.convert_dir),
                "MPLCONFIGDIR": os.environ["MPLCONFIGDIR"],
            }
        )
        env.update(overrides)
        return env

    def run_driver(self, **overrides):
        return subprocess.run(
            ["bash", str(DRIVER_PATH)],
            env=self.driver_env(**overrides),
            capture_output=True,
            text=True,
            cwd=str(self.root),
        )

    def calls(self) -> int:
        return len(self.call_log.read_text().splitlines()) if self.call_log.exists() else 0

    def log_of(self, finished) -> str:
        return finished.stdout + finished.stderr

    # ------------------------------------------------------------ tests
    def test_switch_off_keeps_the_baseline_paths_and_outputs(self):
        finished = self.run_driver(
            USE_NR_SST_DIRECTION_FILTER="0",
            RUN_OBS_CONVERT="0",
            RUN_HX_MERGE="0",
            RUN_TEXT_TO_OBS="0",
            RUN_VALIDATION="0",
        )
        self.assertEqual(finished.returncode, 0, self.log_of(finished))
        self.assertIn(
            f"Final obs_seq.out path:      {self.run_dir}/obs_seq.out_driver_test\n",
            finished.stdout,
        )
        self.assertNotIn("_nr_sst_direction", finished.stdout)
        self.assertFalse(self.work_dir.exists())
        self.assertFalse(self.filter_dir.exists())
        # The baseline work directory holds nothing but its log directory: no
        # intermediate of any kind was produced or consumed.
        self.assertEqual(
            sorted(path.name for path in self.baseline_work_dir.iterdir()), ["logs"]
        )
        self.assertEqual(list((self.baseline_work_dir / "logs").glob("nr_sst*")), [])

    def test_switch_on_isolates_the_work_directory_and_the_obs_seq_name(self):
        finished = self.run_driver(USE_NR_SST_DIRECTION_FILTER="1")
        self.assertEqual(finished.returncode, 0, self.log_of(finished))
        self.assertIn(
            f"Final obs_seq.out path:      {self.obs_seq_on}\n", finished.stdout
        )
        self.assertTrue(self.obs_seq_on.is_file())
        self.assertFalse(self.obs_seq_off.exists())
        for name in (filt.MASK_FILENAME, filt.AUDIT_FILENAME, filt.SUMMARY_FILENAME):
            self.assertTrue((self.filter_dir / name).is_file(), name)
        # The baseline work directory was never created: no shared intermediates.
        self.assertFalse(self.baseline_work_dir.exists())

    def test_the_full_driver_run_screens_then_converts_and_is_repeatable(self):
        finished = self.run_driver(USE_NR_SST_DIRECTION_FILTER="1")
        self.assertEqual(finished.returncode, 0, self.log_of(finished))
        self.assertEqual(self.calls(), 1)
        # The filter log lists the counts and the three input paths.
        log = self.log_of(finished)
        for fragment in (
            "observation file:",
            "original clear-sky mask:",
            "background directory:",
            "NR file:",
            "rejected (reverse):",
            "final kept:",
        ):
            self.assertIn(fragment, log)

        summary_text = (self.filter_dir / filt.SUMMARY_FILENAME).read_text()
        summary = json.loads(summary_text)
        kept = summary["counts"]["final_kept"]
        self.assertGreater(summary["counts"]["rejected_reverse"], 0)
        self.assertLess(kept, summary["counts"]["n_original_clear_sky"])
        self.assertNotIn("NaN", summary_text)

        obs_rows = (self.work_dir / "obs_input.txt").read_text().splitlines()
        merged = np.loadtxt(self.work_dir / "merged_FO_data.txt")
        self.assertEqual(len(obs_rows), kept)
        self.assertEqual(merged.shape, (kept, self.members))
        self.assertEqual(
            int(re.search(r"num_obs:\s*(\d+)", self.obs_seq_on.read_text()).group(1)), kept
        )

        # Row correspondence through the whole chain, in raw order.
        kept_raw = [raw for raw in range(1, self.fx.nobs + 1) if self.fx.expected_keep(raw)]
        self.assertAlmostEqual(
            float(obs_rows[0].split()[10]), float(self.fx.obs_value[kept_raw[0] - 1]), places=4
        )
        self.assertAlmostEqual(
            float(obs_rows[-1].split()[10]), float(self.fx.obs_value[kept_raw[-1] - 1]), places=4
        )
        np.testing.assert_allclose(merged[0], self.fx.hx_members(kept_raw[0]), atol=1.0e-7)
        np.testing.assert_allclose(merged[-1], self.fx.hx_members(kept_raw[-1]), atol=1.0e-7)

        # Every run starts from the ORIGINAL clear-sky mask: a second run
        # reproduces the merged mask byte for byte instead of compounding.
        mask_path = self.filter_dir / filt.MASK_FILENAME
        audit_path = self.filter_dir / filt.AUDIT_FILENAME
        first_mask, first_audit = mask_path.read_bytes(), audit_path.read_bytes()
        again = self.run_driver(USE_NR_SST_DIRECTION_FILTER="1")
        self.assertEqual(again.returncode, 0, self.log_of(again))
        self.assertEqual(mask_path.read_bytes(), first_mask)
        self.assertEqual(audit_path.read_bytes(), first_audit)
        np.testing.assert_array_equal(
            np.loadtxt(self.fx.obs_bt_dir / f"BT_LACC_{CENTER_TAG}" / "clear_sky_mask.txt").astype(int),
            self.fx.original_mask,
        )
        # An already-suffixed name is not suffixed twice.
        twice = self.run_driver(
            USE_NR_SST_DIRECTION_FILTER="1",
            OBS_SEQ_OUT_NAME="obs_seq.out_already_nr_sst_direction",
        )
        self.assertEqual(twice.returncode, 0, self.log_of(twice))
        self.assertTrue((self.run_dir / "obs_seq.out_already_nr_sst_direction").is_file())
        self.assertFalse((self.run_dir / "obs_seq.out_already_nr_sst_direction_nr_sst_direction").exists())

    def test_an_externally_supplied_obs_input_file_is_used_by_both_sides(self):
        """OBS_INPUT_FILE is resolved once; the screen and the converter share it."""
        centre = self.fx.obs_bt_dir / f"BT_LACC_{CENTER_TAG}"
        alternative = centre / "obs_d01_ch4_totalline_withpert_alt.txt"
        alt_values = self.fx.obs_value + 0.05
        np.savetxt(alternative, alt_values, fmt="%.10f")
        self.fx.obs_value = alt_values  # analytics only; the tree is untouched
        finished = self.run_driver(
            USE_NR_SST_DIRECTION_FILTER="1", OBS_INPUT_FILE=str(alternative)
        )
        self.assertEqual(finished.returncode, 0, self.log_of(finished))
        summary = json.loads((self.filter_dir / filt.SUMMARY_FILENAME).read_text())
        self.assertEqual(summary["inputs"]["observation_file"], str(alternative))
        kept = [raw for raw in range(1, self.fx.nobs + 1) if self.fx.expected_keep(raw)]
        rows = (self.work_dir / "obs_input.txt").read_text().splitlines()
        self.assertEqual(len(rows), len(kept))
        self.assertEqual(len(kept), summary["counts"]["final_kept"])
        self.assertAlmostEqual(
            float(rows[0].split()[10]), float(alt_values[kept[0] - 1]), places=4
        )
        self.assertAlmostEqual(
            float(rows[-1].split()[10]), float(alt_values[kept[-1] - 1]), places=4
        )
        audit = (self.filter_dir / filt.AUDIT_FILENAME).read_text()
        self.assertIn(format(float(alt_values[kept[0] - 1]), ".12g").rstrip("0").rstrip("."), audit)

    def test_the_driver_refuses_incompatible_switches_before_any_stage(self):
        cases = (
            ({"EXPERIMENT_MODE": "STANDARD"}, "EXPERIMENT_MODE"),
            ({"OptimizWeight": "1"}, "OptimizWeight"),
            ({"USE_CLEAR_SKY_MASK": "0"}, "USE_CLEAR_SKY_MASK"),
            ({"RUN_OBS_CONVERT": "0"}, "RUN_OBS_CONVERT"),
            ({"RUN_HX_MERGE": "0"}, "RUN_HX_MERGE"),
            ({"RUN_VALIDATION": "0"}, "RUN_VALIDATION"),
            ({"USE_NR_SST_DIRECTION_FILTER": "2"}, "USE_NR_SST_DIRECTION_FILTER"),
        )
        for overrides, fragment in cases:
            with self.subTest(**overrides):
                env = self.driver_env(USE_NR_SST_DIRECTION_FILTER="1")
                env.update(overrides)
                finished = subprocess.run(
                    ["bash", str(DRIVER_PATH)],
                    env=env,
                    capture_output=True,
                    text=True,
                    cwd=str(self.root),
                )
                self.assertNotEqual(finished.returncode, 0)
                self.assertIn(fragment, self.log_of(finished))
                self.assertFalse(self.work_dir.exists(), "no stage may run")
                self.assertFalse(self.obs_seq_on.exists())
                self.assertEqual(self.calls(), 0)

    def test_a_filter_failure_stops_the_driver_before_text_to_obs(self):
        finished = self.run_driver(USE_NR_SST_DIRECTION_FILTER="1")
        self.assertEqual(finished.returncode, 0, self.log_of(finished))
        self.assertEqual(self.calls(), 1)
        mask_path = self.filter_dir / filt.MASK_FILENAME
        good_mask = mask_path.read_bytes()
        good_obs = (self.work_dir / "obs_input.txt").read_bytes()
        good_merged = (self.work_dir / "merged_FO_data.txt").read_bytes()

        # 1) The filter refuses on a broken input.
        set_times(self.fx.nr_file, "2018-09-09_00:00:00")
        failed = self.run_driver(USE_NR_SST_DIRECTION_FILTER="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("NR", self.log_of(failed))
        self.assertEqual(self.calls(), 1, "text_to_obs must not be called again")
        self.assertEqual(mask_path.read_bytes(), good_mask, "no new mask may be published")
        self.assertEqual((self.work_dir / "obs_input.txt").read_bytes(), good_obs)
        self.assertEqual((self.work_dir / "merged_FO_data.txt").read_bytes(), good_merged)

        # 2) The screen deletes everything: also a stop, never an empty obs set.
        set_times(self.fx.nr_file, CENTER_TIME)
        self.fx.nr_offset = -5.0
        self.fx.slope[:] = -1.0
        self.fx.obs_value = self.fx.hx_base - 0.4
        self.fx.write()
        deleted = self.run_driver(USE_NR_SST_DIRECTION_FILTER="1")
        self.assertNotEqual(deleted.returncode, 0)
        self.assertIn("deleted every observation", self.log_of(deleted))
        self.assertEqual(self.calls(), 1)
        self.assertEqual((self.work_dir / "obs_input.txt").read_bytes(), good_obs)
        # The screen publishes the truthful (all-zero) mask and stops there: the
        # driver never reaches text_to_obs, so no empty observation set is
        # converted and the stale products above are not reused.
        self.assertEqual(set(mask_path.read_text().split()), {"0"})

        # 3) A repaired input runs again and publishes the new (still identical)
        #    mask, so the stop was a stop and not a poisoned cache.
        self.fx.nr_offset = 0.0
        self.fx.slope = np.where(self.fx.row_of < self.fx.side / 2.0, -1.0, 1.0)
        self.fx.obs_value = self.fx.hx_base + self.fx.d_target
        self.fx.write()
        recovered = self.run_driver(USE_NR_SST_DIRECTION_FILTER="1")
        self.assertEqual(recovered.returncode, 0, self.log_of(recovered))
        self.assertEqual(self.calls(), 2)
        self.assertEqual(mask_path.read_bytes(), good_mask)


if __name__ == "__main__":
    unittest.main(verbosity=2)
