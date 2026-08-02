# WRF 4.1 Initial SFCLAYREV State Design

## Goal

Reconstruct the latent heat flux at `wrfout` integration time zero by reproducing one WRF 4.1 Revised-MM5 surface-layer call, without introducing an artificial fixed-point solve.

## Evidence and constraints

- The supplied file identifies WRF 4.1, `SF_SFCLAY_PHYSICS=1`, `ISFTCFLX=0`, and `XTIME=0`.
- Its stored `QFX`, `LH`, and `HFX` are zero. `UST` is present and is uniformly `1.0e-4 m s-1`; `ZNT`, `MOL`, `RMOL`, `ZOL`, and `FLQC` are absent.
- WRF 4.1 `module_physics_init.F` initializes `UST=1.0e-4` and `MOL=0` for a non-restart run.
- The file uses `MMINLU=MODIFIED_IGBP_MODIS_NOAH`; its water class is 17. `LANDUSE.TBL` gives water `SFZ0=0.01 cm`, which `landuse_init` converts to `ZNT=1.0e-4 m`.
- `module_sf_sfclayrev.F` evaluates stability and scalar roughness from the incoming `UST/ZNT`, applies one under-relaxed `UST` update, updates ocean `ZNT` once, and then evaluates `FLQC`, `QFX`, and `LH`. It does not iterate this call to a fixed point.

## Design

`SurfaceState` will carry incoming friction velocity and momentum roughness as explicit two-dimensional arrays. `read_surface_state` will read `UST` from the wrfout and use `ZNT` when available. At integration time zero, when `ZNT` is absent, it will reconstruct the WRF ocean initialization value `1.0e-4 m`; a missing `ZNT` away from time zero will be rejected instead of guessed.

`revised_mm5_ocean_flux` will use the incoming state to calculate bulk Richardson number, `z/L`, momentum and moisture transfer denominators, then perform exactly one WRF under-relaxed `UST` update. It will calculate the returned ocean roughness from the updated `UST`, while retaining the incoming-state scalar denominator used by that WRF call. The artificial outer convergence loop and its iteration controls will be removed; the result will report one scheme update for compatibility.

## Error handling

- Reject missing `UST` because the first-call result depends on it.
- Reconstruct absent `ZNT` only at `XTIME=0`; otherwise raise a clear error.
- Continue validating finite, shape-compatible two-dimensional state arrays and supported `ISFTCFLX` values.

## Verification

- A literal neutral-case regression test will distinguish a single under-relaxed update from repeated fixed-point iteration.
- Reader tests will verify that stored `UST` is propagated and that absent `ZNT` is reconstructed only at time zero.
- Existing SFCLAYREV and initial LH/OHC tests will be run.
- The supplied 224 MB wrfout will be used for an end-to-end selected-ocean-point LH calculation, verifying finite output and absence of convergence errors.
