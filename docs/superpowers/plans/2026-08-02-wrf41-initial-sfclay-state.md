# WRF 4.1 Initial SFCLAYREV State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the artificial Revised-MM5 fixed-point solve with the exact one-call WRF 4.1 initial-state update.

**Architecture:** Carry incoming `UST/ZNT` in `SurfaceState`, obtain them from wrfout or the documented time-zero initialization, and evaluate one SFCLAYREV update. Keep NetCDF reading separate from the numerical kernel.

**Tech Stack:** Python, NumPy, xarray, unittest, WRF 4.1 NetCDF output.

## Global Constraints

- Run Python with `/Users/kcfu/miniforge3/envs/wrf/bin/python`.
- Do not add CLI arguments; retain the existing in-script configuration.
- Do not change TC-center, 150 km mask, OHC, plotting, or experiment traversal behavior.
- Implement directly on `master`, preserve unrelated files, and push the verified commit to `origin/master`.

---

### Task 1: Single-call SFCLAYREV numerical kernel

**Files:**
- Modify: `plot_scripts/wrf41_sfclayrev.py`
- Test: `plot_scripts/tests/test_wrf41_sfclayrev.py`

**Interfaces:**
- Consumes: `SurfaceState` arrays and `SfclayOptions.isftcflx`.
- Produces: `revised_mm5_ocean_flux(state, options) -> SurfaceFluxResult` after exactly one WRF update.

- [x] Add a neutral-case test with literal expected updated `UST`, demonstrating one under-relaxed update.
- [x] Run that test and confirm it fails because the current code lacks the incoming WRF state interface.
- [x] Add incoming `UST/ZNT` fields, remove the outer loop and iteration options, and preserve WRF's update order.
- [x] Run the focused numerical tests and confirm they pass.

### Task 2: Time-zero wrfout state reconstruction

**Files:**
- Modify: `plot_scripts/plot_initial_tc_lh_ohc.py`
- Test: `plot_scripts/tests/test_plot_initial_tc_lh_ohc.py`

**Interfaces:**
- Consumes: wrfout `UST`, optional `ZNT`, `XTIME`, and the existing thermodynamic fields.
- Produces: `SurfaceState` with incoming friction velocity and roughness arrays.

- [x] Add reader tests proving stored `UST` propagation and time-zero `ZNT=1.0e-4 m` fallback.
- [x] Run the reader tests and confirm they fail before implementation.
- [x] Read `UST`; read `ZNT` when present; otherwise allow the documented fallback only at `XTIME=0`.
- [x] Run both focused test modules and confirm they pass.

### Task 3: Integration verification and publication

**Files:**
- Modify only if a test exposes a defect in the two files above.

**Interfaces:**
- Consumes: `/Users/kcfu/Downloads/wrfout_d01_2018-09-10_00:00:00`.
- Produces: finite initial LH diagnostics without a surface convergence exception.

- [x] Run the complete relevant unittest suite with the configured WRF Python interpreter.
- [x] Run a selected-ocean-point LH calculation against the supplied wrfout and record finite count and range.
- [ ] Review `git diff`, confirm no unrelated changes or large generated files, then commit the implementation and documentation.
- [ ] Push `master` to `origin/master` and verify the remote branch commit.
