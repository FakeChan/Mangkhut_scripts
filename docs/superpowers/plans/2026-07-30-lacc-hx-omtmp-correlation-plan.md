# LACC Hx-OM_TMP Correlation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy TK-TSK time-series diagnostic with a tested ensemble-space LACC diagnostic between historical satellite Hx and current OM_TMP level 0.

**Architecture:** Keep the workflow in `plot_LACC_leadlag_corr.py`, with small pure functions for time generation, profile parsing, Hx loading, interpolation, correlations, summaries, and plotting. Use an in-script configuration dataclass and validate all input shapes before computing statistics.

**Tech Stack:** Python 3, NumPy, pandas, netCDF4, SciPy, Matplotlib, standard-library unittest.

## Global Constraints

- Current time is `2018-09-10_00:00:00`.
- Initial lead times are generated from maximum lag 12 hours and interval 3 hours.
- All five valid times `t`, `t-3`, `t-6`, `t-9`, and `t-12` have 50-member Hx files.
- Hx files contain exactly 676 values in profile order.
- Target state is `OM_TMP[0, 0, :, :]` from current `firstguess_d01.memNNN`.
- Pearson correlations are calculated across members at each observation point.
- Averaged windows include the current time and grow cumulatively toward earlier times.
- Paths and scientific parameters remain editable inside the script; no CLI is added.

---

### Task 1: Core parsing, time, and statistics

**Files:**
- Create: `plot_scripts/tests/test_plot_LACC_leadlag_corr.py`
- Modify: `plot_scripts/plot_LACC_leadlag_corr.py`

**Interfaces:**
- Produces: `Config`, `validate_config`, `build_lag_hours`, `hx_valid_time`, `hx_path`, `read_profile_coordinates`, `read_hx_file`, `pointwise_correlations`, `build_averaged_hx`, and `summarize_pointwise`.

- [ ] **Step 1: Write failing tests**

Add unittest cases with hand-derived expectations for `[0, 3, 6, 9, 12]`,
cross-day Hx paths, marker-based profile parsing, 676-value validation,
pointwise member correlations, cumulative Hx means, signed spatial summaries,
and zero ensemble variance.

- [ ] **Step 2: Run tests and verify the expected import/API failures**

Run:

```bash
MPLCONFIGDIR=/private/tmp/codex-mpl /Users/kcfu/miniforge3/envs/wrf/bin/python -m unittest plot_scripts.tests.test_plot_LACC_leadlag_corr -v
```

Expected: failures because the new interfaces do not yet exist.

- [ ] **Step 3: Implement the minimal pure-function core**

Replace the legacy time-series helpers with the planned configuration,
validation, parsing, array construction, correlation, and summary functions.

- [ ] **Step 4: Run the focused tests**

Run the command from Step 2.

Expected: all Task 1 cases pass.

### Task 2: OM_TMP interpolation and complete workflow

**Files:**
- Modify: `plot_scripts/tests/test_plot_LACC_leadlag_corr.py`
- Modify: `plot_scripts/plot_LACC_leadlag_corr.py`

**Interfaces:**
- Consumes: Task 1 configuration and statistical helpers.
- Produces: `interpolate_to_observations`, `load_current_omtmp`, `calculate_lacc_correlations`, `plot_lacc_correlations`, and `main`.

- [ ] **Step 1: Write failing interpolation and workflow tests**

Create small real NetCDF member fixtures on a rectangular lat-lon grid and Hx
text fixtures. Assert exact linear interpolation at interior points, output
array dimensions, cumulative windows, output table creation, and two-panel PNG
creation.

- [ ] **Step 2: Run tests and verify the expected missing-interface failures**

Run:

```bash
MPLCONFIGDIR=/private/tmp/codex-mpl /Users/kcfu/miniforge3/envs/wrf/bin/python -m unittest plot_scripts.tests.test_plot_LACC_leadlag_corr -v
```

Expected: new Task 2 tests fail because interpolation/workflow interfaces are
not implemented.

- [ ] **Step 3: Implement interpolation, orchestration, tables, and plotting**

Use SciPy Delaunay triangulation and barycentric linear weights to interpolate
each member's `OM_TMP` level 0 at the profile coordinates. Write pointwise,
summary, interpolated-state CSV files and a two-panel PNG.

- [ ] **Step 4: Run focused and integration tests**

Run the command from Step 2.

Expected: all tests pass.

### Task 3: Real-profile and repository verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-30-lacc-hx-omtmp-correlation-design.md`
- Modify: `docs/superpowers/plans/2026-07-30-lacc-hx-omtmp-correlation-plan.md`

**Interfaces:**
- Consumes: completed script and supplied profile.
- Produces: verification evidence only.

- [ ] **Step 1: Parse the supplied profile**

Run:

```bash
MPLCONFIGDIR=/private/tmp/codex-mpl /Users/kcfu/miniforge3/envs/wrf/bin/python -c "from pathlib import Path; from plot_scripts.plot_LACC_leadlag_corr import read_profile_coordinates; print(len(read_profile_coordinates(Path('/Users/kcfu/Downloads/prof09_12:00.dat'), 676)))"
```

Expected: `676`.

- [ ] **Step 2: Compile the refactored script**

Run:

```bash
/Users/kcfu/miniforge3/envs/wrf/bin/python -m py_compile plot_scripts/plot_LACC_leadlag_corr.py plot_scripts/tests/test_plot_LACC_leadlag_corr.py
```

Expected: exit code 0.

- [ ] **Step 3: Run the complete tests and diff checks**

Run:

```bash
MPLCONFIGDIR=/private/tmp/codex-mpl /Users/kcfu/miniforge3/envs/wrf/bin/python -m unittest plot_scripts.tests.test_plot_LACC_leadlag_corr -v
git diff --check
```

Expected: all tests pass and `git diff --check` reports no errors.
