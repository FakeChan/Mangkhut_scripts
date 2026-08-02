# Initial TC-Region LH and OHC Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested WRF 4.1 initial-state diagnostic that reconstructs TC-region latent heat flux for three experiments, calculates OHC26 for the two ocean-running experiments, and produces two separate ensemble-comparison figures.

**Architecture:** Put the WRF 4.1 surface-layer mathematics in `plot_scripts/wrf41_sfclayrev.py`. Put configuration, NetCDF I/O, TC masking, OHC, statistics, CSV output, and plotting in `plot_scripts/plot_initial_tc_lh_ohc.py`. Numerical functions accept NumPy arrays so synthetic tests exercise real calculations without mounted cluster files.

**Tech Stack:** Python 3, NumPy, pandas, xarray, netCDF4, wrf-python, Matplotlib, pytest.

## Global Constraints

- Use `/Users/kcfu/miniforge3/envs/wrf/bin/python` for Python verification.
- Keep user settings inside the script; do not add CLI arguments.
- Reconstruct LH from state variables; never use stored initial `QFX`, `LH`, or accumulated flux.
- Stop on unsupported or unidentified surface-layer physics.
- Use the diagnosed NR minimum-SLP center and a 150 km Haversine ocean mask.
- Omit no-DA from OHC rather than fabricating a value.
- Produce independent LH and OHC PNGs plus member-level CSV files.

---

### Task 1: WRF 4.1 Revised MM5 Kernel

**Files:**
- Create: `plot_scripts/wrf41_sfclayrev.py`
- Create: `plot_scripts/tests/test_wrf41_sfclayrev.py`

**Interfaces:**
- `saturation_mixing_ratio(temperature_k, pressure_pa) -> ndarray`
- `revised_mm5_ocean_flux(state: SurfaceState, options: SfclayOptions) -> SurfaceFluxResult`
- Frozen dataclasses `SurfaceState`, `SfclayOptions`, `SurfaceFluxResult`

- [ ] **Step 1: Write failing thermodynamic and neutral-flux tests**

```python
def test_saturation_mixing_ratio_matches_wrf_expression():
    got = saturation_mixing_ratio(np.array([300.0]), np.array([100000.0]))
    assert got[0] == pytest.approx(0.0227902378, rel=2e-6)

def test_warm_ocean_reconstructs_lh_when_stored_flux_is_zero():
    state = uniform_surface_state(
        air_temperature_k=299.0, surface_temperature_k=301.0,
        vapor_mixing_ratio=0.018, u_ms=8.0,
    )
    result = revised_mm5_ocean_flux(state, SfclayOptions(isftcflx=0))
    assert result.qfx[0, 0] > 0.0
    assert result.lh[0, 0] == pytest.approx(2.5e6 * result.qfx[0, 0])
```

- [ ] **Step 2: Verify RED**

Run: `/Users/kcfu/miniforge3/envs/wrf/bin/python -m pytest plot_scripts/tests/test_wrf41_sfclayrev.py -v`

Expected: import failure because the kernel does not exist.

- [ ] **Step 3: Implement minimal neutral kernel and verify GREEN**

Implement WRF saturation constants, virtual temperature, density, Charnock
momentum roughness, Fairall moisture roughness, friction velocity, `FLQC`,
nonnegative `QFX`, and `LH=XLV*QFX`. Validate shapes, pressure, and height.
Run the Task 1 pytest command until the two tests pass.

- [ ] **Step 4: Write failing stability and hurricane-option tests**

```python
@pytest.mark.parametrize(("ts", "sign"), [(297.0, 1), (303.0, -1)])
def test_stability_solver_returns_expected_richardson_sign(ts, sign):
    result = revised_mm5_ocean_flux(
        uniform_surface_state(surface_temperature_k=ts),
        SfclayOptions(isftcflx=0),
    )
    assert np.sign(result.bulk_richardson[0, 0]) == sign
    assert np.isfinite(result.inverse_obukhov_length).all()

def test_isftcflx_one_uses_fixed_moisture_roughness():
    result = revised_mm5_ocean_flux(
        uniform_surface_state(), SfclayOptions(isftcflx=1)
    )
    assert result.moisture_roughness_m[0, 0] == pytest.approx(1.0e-4)

def test_condensation_is_clipped_to_zero():
    state = uniform_surface_state(
        surface_temperature_k=296.0, vapor_mixing_ratio=0.024
    )
    result = revised_mm5_ocean_flux(state, SfclayOptions(isftcflx=0))
    assert result.qfx[0, 0] == 0.0
```

- [ ] **Step 5: Verify RED, implement remaining WRF branches, verify GREEN**

Port stable Cheng-Brutsaert and unstable similarity functions, the bulk-
Richardson `z/L` solve, bounded friction-velocity iteration, default Fairall
`z0q`, `ISFTCFLX=1`, and `ISFTCFLX=2` Garratt roughness. Raise with an
unconverged-point count. Run all Task 1 tests.

- [ ] **Step 6: Commit Task 1**

```bash
git add plot_scripts/wrf41_sfclayrev.py plot_scripts/tests/test_wrf41_sfclayrev.py
git commit -m "feat: reconstruct WRF 4.1 initial latent heat flux"
```

### Task 2: TC Mask and OHC26

**Files:**
- Create: `plot_scripts/plot_initial_tc_lh_ohc.py`
- Create: `plot_scripts/tests/test_plot_initial_tc_lh_ohc.py`

**Interfaces:**
- `haversine_distance_km(...) -> ndarray`
- `tc_ocean_mask(...) -> ndarray[bool]`
- `ohc26_profile(temp_c, depth_m, rho, cp) -> float`
- `ohc26_field(temperature_3d, depth_m, rho, cp) -> ndarray`

- [ ] **Step 1: Write failing 150 km mask test**

```python
def test_tc_mask_includes_boundary_and_excludes_land():
    lats = np.array([[0.0, 0.0, 0.0]])
    lons = np.array([[0.0, 150.0 / 111.195, 2.0]])
    ocean = np.array([[True, True, False]])
    mask = tc_ocean_mask(lats, lons, 0.0, 0.0, 150.0, ocean)
    assert mask.tolist() == [[True, True, False]]
```

- [ ] **Step 2: Verify RED, implement mask, verify GREEN**

Run: `/Users/kcfu/miniforge3/envs/wrf/bin/python -m pytest plot_scripts/tests/test_plot_initial_tc_lh_ohc.py -v`

Implement Earth-radius-6371-km Haversine distance, inclusive radius, shape
validation, ocean intersection, and an empty-mask error. Re-run the test.

- [ ] **Step 3: Write failing hand-integrated OHC tests**

```python
def test_ohc26_integrates_partial_crossing_interval():
    got = ohc26_profile(
        np.array([29.0, 27.0, 25.0]),
        np.array([0.0, 10.0, 20.0]), 1025.0, 3985.0,
    )
    assert got == pytest.approx(1025.0 * 3985.0 * 22.5)

def test_ohc26_is_zero_for_cool_surface():
    got = ohc26_profile(
        np.array([25.5, 24.0]), np.array([0.0, 10.0]), 1025.0, 3985.0
    )
    assert got == 0.0

def test_ohc26_rejects_missing_crossing():
    with pytest.raises(ValueError, match="26"):
        ohc26_profile(
            np.array([29.0, 28.0]), np.array([0.0, 10.0]), 1025.0, 3985.0
        )
```

- [ ] **Step 4: Verify RED, implement OHC26, verify GREEN**

Implement monotonic-depth validation, first-crossing interpolation, trapezoidal
integration, cool-surface zero, and field traversal. Run Task 2 tests.

- [ ] **Step 5: Commit Task 2**

```bash
git add plot_scripts/plot_initial_tc_lh_ohc.py plot_scripts/tests/test_plot_initial_tc_lh_ohc.py
git commit -m "feat: add TC mask and OHC26 diagnostics"
```

### Task 3: NetCDF Readers and Member Workflow

**Files:**
- Modify: `plot_scripts/plot_initial_tc_lh_ohc.py`
- Modify: `plot_scripts/tests/test_plot_initial_tc_lh_ohc.py`

**Interfaces:**
- `find_unique_wrfout(...) -> Path`
- `read_tc_center(nr_path, slp_reader=None) -> tuple[float, float, float]`
- `read_surface_state(ds) -> tuple[SurfaceState, ndarray, ndarray, ndarray]`
- `read_ohc_inputs(ds) -> tuple[ndarray, ndarray]`
- `calculate_member_records(config) -> tuple[pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Write failing synthetic-NetCDF reader tests**

Create a 2-by-3 mass grid with staggered U/V, zero stored flux fields, required
thermodynamic variables, physics attributes, `OM_TMP`, and `OM_DEPTH`. Assert
mass-grid winds and positive reconstructed LH despite zero stored fluxes.

- [ ] **Step 2: Verify RED, implement discovery/readers, verify GREEN**

Implement the frozen in-script configuration, unique recursive file discovery,
time selection, destaggering, WRF temperature/geopotential reconstruction,
physics-attribute validation, water-mask inference, and ocean unit
normalization. Run Task 2 tests.

- [ ] **Step 3: Write failing workflow test**

Use temporary experiment/filter/member trees and an injected SLP reader.
Assert LH contains all three experiments, OHC contains only weak/strong,
and group statistics use literal mean and sample standard deviation (`ddof=1`).

- [ ] **Step 4: Implement member traversal and auditable records**

Require every configured member, compute member means independently, attach
group statistics, and record paths, center, physics settings, and mask counts.
Do not downgrade missing-data errors. Run Task 2 tests.

- [ ] **Step 5: Commit Task 3**

```bash
git add plot_scripts/plot_initial_tc_lh_ohc.py plot_scripts/tests/test_plot_initial_tc_lh_ohc.py
git commit -m "feat: calculate initial LH and OHC member statistics"
```

### Task 4: Two Figures, CSVs, and Main

**Files:**
- Modify: `plot_scripts/plot_initial_tc_lh_ohc.py`
- Modify: `plot_scripts/tests/test_plot_initial_tc_lh_ohc.py`

**Interfaces:**
- `plot_member_comparison(df, metric, output_path, config) -> None`
- `write_outputs(lh_df, ohc_df, config) -> tuple[Path, Path, Path, Path]`
- `main() -> None`

- [ ] **Step 1: Write failing output test**

```python
def test_write_outputs_creates_two_csvs_and_two_pngs(tmp_path):
    paths = write_outputs(*literal_output_frames(), output_config(tmp_path))
    assert [p.name for p in paths] == [
        "initial_tc150_lh_members.csv",
        "initial_tc150_ohc_members.csv",
        "initial_tc150_lh.png",
        "initial_tc150_ohc.png",
    ]
    assert all(p.stat().st_size > 0 for p in paths)
```

- [ ] **Step 2: Verify RED, implement outputs, verify GREEN**

Use deterministic six-member jitter, existing experiment colors, distinct
filter markers, group mean and sample-standard-deviation error bars, and the
specified units. Create and close two independent figures. Write CSVs without
an index. Add `main`, run Task 2 tests, and require all outputs nonempty.

- [ ] **Step 3: Commit Task 4**

```bash
git add plot_scripts/plot_initial_tc_lh_ohc.py plot_scripts/tests/test_plot_initial_tc_lh_ohc.py
git commit -m "feat: plot separate initial LH and OHC figures"
```

### Task 5: Full Verification and Scientific Audit

**Files:**
- Modify only if a verification test first reproduces a defect.

- [ ] **Step 1: Run focused and regression tests**

```bash
/Users/kcfu/miniforge3/envs/wrf/bin/python -m pytest \
  plot_scripts/tests/test_wrf41_sfclayrev.py \
  plot_scripts/tests/test_plot_initial_tc_lh_ohc.py -v
/Users/kcfu/miniforge3/envs/wrf/bin/python -m pytest plot_scripts/tests -v
```

- [ ] **Step 2: Compile production modules**

```bash
/Users/kcfu/miniforge3/envs/wrf/bin/python -m py_compile \
  plot_scripts/wrf41_sfclayrev.py \
  plot_scripts/plot_initial_tc_lh_ohc.py
```

- [ ] **Step 3: Audit specification coverage**

Confirm no stored initial flux is read as the answer, no-DA is absent from OHC,
both filters and all members are configured, outputs are separate, units are
correct, unsupported physics raises, and every result is traceable in CSV.

- [ ] **Step 4: Check final state**

```bash
git diff --check
git status --short
git log --oneline -6
```

- [ ] **Step 5: Commit only tested verification fixes**

If verification reveals a defect, add a failing regression test before the fix
and commit that isolated change. Otherwise do not make an empty commit.
