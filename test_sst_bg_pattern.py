"""Verify the parameterized SST background (member name pattern + ocean mask)."""
import sys
import tempfile
from pathlib import Path

import netCDF4 as nc
import numpy as np

sys.path.insert(0, "3create_obs/hx_rttov")
from compute_adaptive_LACC import Config, load_sst_target

RNG = np.random.default_rng(7)
NY, NX = 8, 8
LAT = np.linspace(13.0, 15.5, NY)
LON = np.linspace(147.0, 149.5, NX)
GLAT, GLON = np.meshgrid(LAT, LON, indexing="ij")
REGION = (13.2, 15.0, 147.2, 149.1)


def write_nc(path, land2d, mem=1):
    """land2d: 1=land, 0=water (LANDMASK convention).  Deterministic SST so two
    file sets (legacy vs new name) hold the SAME field and are comparable."""
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("bottom_top", 30)
        ds.createDimension("south_north", NY)
        ds.createDimension("west_east", NX)
        v_lat = ds.createVariable("XLAT", "f4", ("south_north", "west_east"))
        v_lon = ds.createVariable("XLONG", "f4", ("south_north", "west_east"))
        v_land = ds.createVariable("LANDMASK", "f4", ("south_north", "west_east"))
        v_xland = ds.createVariable("XLAND", "f4", ("south_north", "west_east"))
        v_sst = ds.createVariable("OM_TMP", "f4", ("Time", "bottom_top", "south_north", "west_east"))
        v_lat[:] = GLAT
        v_lon[:] = GLON
        v_land[:] = land2d
        v_xland[:] = 2.0 - land2d          # XLAND: 1=land, 2=water
        sst2d = 300.0 + 0.5 * GLAT + 0.2 * mem
        v_sst[0, 0, :, :] = sst2d
        v_sst[0, 1:, :, :] = np.nan        # other levels unused


def member_field(mem):
    f = np.zeros((NY, NX))
    # a few land cells (LANDMASK=1) inside the region box, rest water
    f[2, 3] = 1.0
    f[3, 4] = 1.0
    f[4, 5] = 1.0
    return f


def base_cfg(**kw):
    kw.setdefault("domain", "d01")
    kw.setdefault("ens_size", 2)
    kw.setdefault("sst_region", REGION)
    kw.setdefault("sst_var", "OM_TMP")
    kw.setdefault("sst_level", 0)
    kw.setdefault("sst_lat_var", "XLAT")
    kw.setdefault("sst_lon_var", "XLONG")
    kw.setdefault("sst_land_mask_var", "LANDMASK")
    kw.setdefault("sst_member_pattern", "firstguess_{domain}.mem{mem:03d}")
    kw.setdefault("sst_land_mask_water_above", None)
    return Config(**kw)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="sstbg_test_"))
    # legacy naming
    for m in (1, 2):
        write_nc(tmp / f"firstguess_d01.mem{m:03d}", member_field(m))
    cfg_legacy = base_cfg(sst_bg_dir=tmp)
    s_legacy = load_sst_target(cfg_legacy)
    print("legacy s:", s_legacy)

    # new DART post-assim naming (4-digit) + XLAND with water>=1.5
    for m in (1, 2):
        write_nc(tmp / f"preassim_member_{m:04d}_d01.nc", member_field(m))
    cfg_new = base_cfg(
        sst_bg_dir=tmp,
        sst_member_pattern="preassim_member_{mem:04d}_{domain}.nc",
        sst_land_mask_var="XLAND",
        sst_land_mask_water_above=1.5,
    )
    s_new = load_sst_target(cfg_new)
    print("new s:", s_new)

    # both target ESTIMATES must be identical: same land cells, same SST grid.
    assert np.allclose(s_legacy, s_new), (s_legacy, s_new)
    # cross-check ocean cells: 3 land cells marked LANDMASK=1 -> water cells = all others
    land_total = int(sum(member_field(m).sum() for m in (1, 2)))
    print("land cells per member inside grid:", member_field(1).sum())
    # counts should be equal for both conventions
    print("PASS: legacy and new-pattern/XLAND agree,", s_legacy, s_new)


if __name__ == "__main__":
    main()