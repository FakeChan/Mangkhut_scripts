# Initial TC-Region LH and OHC Diagnostic Design

## Objective

Create a tested Python diagnostic for the three cycling experiments at
`2018-09-10_00:00:00`. The diagnostic reconstructs latent heat flux (LH) from
the initial atmospheric and surface state because all explicitly stored fluxes
are zero at the integration start. It also calculates upper-ocean heat content
relative to 26 degrees Celsius (OHC26) for the two experiments that run the
ocean model.

All quantities are summarized over ocean grid points within 150 km of the
nature-run typhoon center. The final deliverables are two independent figures:
one for LH and one for OHC.

## Input Layout and User Configuration

All editable settings are grouped near the top of the script; no command-line
arguments or required environment variables are introduced.

Initial settings are:

- experiment root:
  `/scratch/lililei1/kcfu/tc_mangkhut/cycle_test`;
- nature-run root:
  `/share/home/lililei1/kcfu/tc_mangkhut/NR_wrfout`;
- valid time: `2018-09-10_00:00:00`;
- member domain: `d02`;
- nature-run center domain: `d03`;
- filters: `EAKF` and `QCF_RHF`;
- members: `006`, `015`, `029`, `037`, `043`, and `044`;
- radius: 150 km.

The experiments are:

- `6mem_oceanAssim0Run0`: no DA/no ocean integration;
- `6mem_oceanAssim0Run1`: weak-couple DA;
- `6mem_oceanAssim1Run1`: strong-couple DA.

A member file is found recursively below:

```text
{experiment_root}/{experiment}/{filter}/{member}/
```

using the filename pattern:

```text
wrfout_d02_2018-09-10_00:00:00*
```

The nature-run file is found recursively below the nature-run root using:

```text
wrfout_d03_2018-09-10_00:00:00*
```

Missing or duplicate matches are errors rather than silently selecting an
ambiguous input.

## Typhoon Center and 150 km Region

The center follows `plot_EAKFvsQCF_anal.py` and `kc_functions.getTClocation`:
diagnose sea-level pressure with `wrf.getvar(nc, "slp", timeidx=0)` and select
the first grid point containing the finite global minimum.

The resulting nature-run center latitude and longitude are mapped to each
member grid through coordinates, not through shared integer indices. Great-
circle distance is calculated with the Haversine formula and Earth radius
6371 km. A point belongs to the TC region when its distance is no greater than
150 km.

Only ocean points are retained. `XLAND > 1.5` is the primary WRF water test;
`LANDMASK < 0.5` is accepted when `XLAND` is unavailable. No fallback to land
points is allowed.

## Initial Latent Heat Flux Reconstruction

The script never reads `QFX`, `LH`, `LHFLX`, or an accumulated surface flux as
the initial LH value. It reconstructs moisture exchange from state variables.

The implementation reads the WRF physics metadata, including
`SF_SFCLAY_PHYSICS`, `ISFTCFLX`, and `ISFFLX`. The first supported algorithm is
the WRF 4.1 Revised MM5 Monin-Obukhov surface-layer scheme
(`SF_SFCLAY_PHYSICS == 1`). A different or absent surface-layer option is a
hard error so another scheme is never evaluated with Revised MM5 formulas.

For every ocean point, the reconstructed upward moisture flux is:

\[
QFX = \max\left[
\rho_a M\frac{\kappa u_*}{D_q}(q_s-q_a),0
\right],
\]

where:

\[
D_q = \ln\left(\frac{z_a+z_{0q}}{z_{0q}}\right)-\psi_q.
\]

The latent heat flux is:

\[
LH=L_v QFX,
\]

with `L_v` taken from the WRF 4.1 physical constant used by the scheme. Output
LH units are watts per square metre and positive values denote upward ocean-
to-atmosphere transfer.

Inputs are reconstructed from `U`, `V`, `T`, `P`, `PB`, `QVAPOR`, `PH`, `PHB`,
`HGT`, `PSFC`, `TSK`, `XLAND`/`LANDMASK`, and the available surface-layer state.
For ocean-running experiments, surface temperature uses `OM_TMP` level 0;
otherwise it uses `TSK`. WRF staggered winds are destaggered to mass points.
Perturbation potential temperature is converted consistently with WRF pressure.

Sea-surface saturated mixing ratio follows the WRF 4.1 saturation-vapor-
pressure expression. Friction velocity, bulk Richardson number, Monin-
Obukhov length, momentum/moisture stability functions, Charnock momentum
roughness, and moisture roughness are solved using the WRF 4.1 Revised MM5
branches. `ISFTCFLX` selects the matching default, hurricane, or Garratt
roughness branch. Initial flux-dependent inputs are initialized to zero, as in
the integration-start state, and the offline surface-layer solve iterates until
friction velocity and stability converge.

The implementation preserves the WRF 4.1 nonnegative ocean evaporation limit
`QFX = max(QFX, 0)`.

## XTIME-Aware Latent Heat Flux Source

The same LH-source decision applies to every ensemble member and the NR. The
script requires one finite scalar `XTIME`, interpreted in WRF minutes since the
simulation start, and uses `np.isclose(XTIME, 0)` to identify an initial
history output. Stored-flux readiness is evaluated only over the selected
150-km ocean mask, never over land or the full domain.

At `XTIME=0`, missing stored `LH`/`QFX` or values that are all numerically zero
indicate an uninitialized initial flux, so LH is reconstructed from the current
WRF 4.1 Revised-MM5 state algorithm. If either stored field already contains
finite nonzero selected values, the stored diagnostics are used instead: a
nonzero `LH` has priority, while nonzero `QFX` is converted with
`LH=2.5e6*QFX` when `LH` is absent or still zero.

At `XTIME>0`, stored output is required. `LH` is preferred when finite and
nonzero; `QFX` is the fallback when `LH` is missing or zero. If all available
stored flux values are zero, the script warns and preserves the stored zero
rather than silently reconstructing an integrated output. Missing or nonfinite
stored flux at `XTIME>0` is a hard error.

Whenever both fields provide nonzero information, `LH` is checked against
`2.5e6*QFX`; a material mismatch warns while stored `LH` remains authoritative.
Zero tests use `rtol=0`, `atol=1e-6 W m-2` for LH, and
`atol=1e-12 kg m-2 s-1` for QFX. Every output row records `xtime_minutes`,
`lh_source` (`reconstructed_initial`, `stored_LH`, or `derived_QFX`), and the
corresponding `stored_flux_used` boolean.

OHC26 never branches on `XTIME`; it is always calculated from `OM_TMP` and
`OM_DEPTH` with the common OHC integration.

## OHC26 Calculation

OHC is computed only for:

- `6mem_oceanAssim0Run1`;
- `6mem_oceanAssim1Run1`.

The no-DA/no-ocean experiment is excluded and is not represented by a
fabricated OHC value.

At every ocean point:

\[
OHC_{26}=\rho_w c_{p,w}\int_0^{D_{26}}[T(z)-26],dz,
\]

where `rho_w = 1025 kg m-3`, `c_p,w = 3985 J kg-1 K-1`, temperature is read
from `OM_TMP`, and depth is read from `OM_DEPTH`. Kelvin temperature is
converted to degrees Celsius when indicated by units or physically consistent
values. Depth is normalized to metres positive downward after checking its
units and monotonicity.

`D26` is the first downward crossing of 26 degrees Celsius. Temperature is
linearly interpolated to the crossing and the integral is evaluated by the
trapezoidal rule, including the partial final interval. A profile whose surface
is at or below 26 degrees Celsius has zero OHC26. A profile that remains above
26 degrees Celsius through the deepest available level is invalid by default,
because the complete warm-water column is not represented.

OHC is stored in both joules per square metre and kilojoules per square
centimetre using:

\[
1\ \mathrm{kJ\,cm^{-2}}=10^7\ \mathrm{J\,m^{-2}}.
\]

## Spatial and Ensemble Statistics

Each member is processed independently. For diagnostic field `X`, the member
TC-region mean is:

\[
\overline{X}_m=\frac{\sum_{i,j}M_{m,i,j}X_{m,i,j}}
{\sum_{i,j}M_{m,i,j}},
\]

where `M` is the member-specific 150 km ocean mask. WRF d02 has constant map-
plane spacing, so equal-gridpoint averaging is used. The CSV records the number
of selected grid points and finite values so coverage is auditable.

For each experiment/filter combination, ensemble mean and sample standard
deviation (`ddof=1`) are calculated from the six member regional means. All six
members are required; missing or nonfinite member means stop the calculation.

## Outputs

The script writes five files below a configurable output directory:

- `initial_tc150_lh_members.csv`;
- `initial_tc150_ohc_members.csv`;
- `initial_tc150_nr_reference.csv`;
- `initial_tc150_lh.png`;
- `initial_tc150_ohc.png`.

The two member CSVs contain experiment, display label, filter, member, center
coordinates, selected gridpoint counts, the member regional mean, and the
corresponding experiment/filter ensemble mean and sample standard deviation.
The separate NR CSV contains one truth-reference row as specified below.

The LH figure contains all three experiments. The OHC figure contains only the
two ocean-running experiments. Within each experiment, EAKF and QCF_RHF are
offset horizontally. Six member values are shown as jittered points, with an
overlaid ensemble mean and standard-deviation error bar. Experiment colors use
the existing colorblind-safe palette; filters use distinct marker shapes. The
figures use independent scientifically meaningful y-axis units:
`W m-2` for LH and `kJ cm-2` for OHC.

## Nature-Run Reference Values

The diagnostic also calculates one nature-run (NR) reference value for LH and
one for OHC26 at the configured valid time. The NR file and minimum-SLP center
are the same file and center already used to define every experiment's 150-km
region. Experiment values continue to map this common NR center onto each
member grid; NR values apply the same 150-km ocean mask directly on the NR
native grid.

NR latent heat flux follows the same XTIME-aware rule as ensemble members. A
continuous NR output (`XTIME>0`) uses its stored `LH`, or derives LH from
stored `QFX` when necessary. An NR initial output (`XTIME=0`) is reconstructed
only when the selected stored fields are missing or all zero. When both fields
provide nonzero information, the diagnostic compares `LH` against
`2.5e6*QFX`; a material mismatch warns while stored `LH` remains authoritative.
Incompatible grid shapes and nonfinite selected values are hard errors.

NR OHC26 uses the same `OM_TMP`, `OM_DEPTH`, density, heat capacity,
first-26-C crossing, and integration implementation as the ocean-running
experiments. The NR result is kept separate from member and ensemble
statistics because it is a single truth value, not an
experiment/filter/member sample.

The one-row `initial_tc150_nr_reference.csv` records the NR input path, center,
selected ocean-point counts, XTIME, audited LH source, whether a stored flux
was used, LH, and OHC26.
Both figures draw the matching NR value as a solid horizontal line spanning
all experiments. The line uses the established NR red `#d73027`, is labelled
`NR`, and participates in automatic y-axis limits.

## Optional CSV Cache Input

Two independent booleans in the in-script `Config` control reuse of existing
CSV outputs:

- `read_nr_from_csv=False` recalculates the NR reference; `True` reads
  `initial_tc150_nr_reference.csv`;
- `read_members_from_csv=False` recalculates all member values; `True` reads
  both `initial_tc150_lh_members.csv` and `initial_tc150_ohc_members.csv`.

The cache paths are derived from `output_dir`; no command-line arguments or
additional path settings are required. Cached NR and member results can be
selected independently. When both switches are true, plotting requires no
NetCDF calculation. When only the member cache is disabled, member calculation
still reads the NR file to diagnose the common TC center even if the NR metric
itself comes from CSV.

Cache mode is strict: missing, empty, malformed, nonfinite, duplicated, or
configuration-incompatible CSV content stops with a path-specific error. The
member cache must contain exactly the configured experiment/filter/member
combinations for LH and exactly the configured ocean-enabled combinations for
OHC. No partial cache/recalculation mixture is allowed within the member
dataset. A CSV used as input is not rewritten; figures are regenerated from
the loaded values, while newly calculated data continue to be written to CSV.
Cached NR and member rows must include finite `xtime_minutes`, a valid
`lh_source`, and a consistent `stored_flux_used` value, so pre-feature caches
require one recalculation before reuse.

## Error Handling and Auditability

Execution stops with a path-specific message when:

- an input directory or required member/NR file is missing;
- a recursive file match is ambiguous;
- a required state variable or physics attribute is absent;
- the configured surface-layer scheme is unsupported;
- staggered and mass-grid dimensions are incompatible;
- the typhoon center cannot be diagnosed;
- no ocean points fall within 150 km;
- ocean depth is nonmonotonic or has unknown units;
- an OHC profile does not contain the first 26-degree crossing;
- a member regional mean is nonfinite.

The console log reports each input file, detected physics options, center,
mask count, and member result. The CSV files retain the same identifying
metadata.

## Verification

Automated tests use small synthetic arrays and NetCDF fixtures to verify:

- Haversine distance and the inclusive 150 km boundary;
- nature-run minimum-SLP center selection through an injectable SLP reader;
- U/V destaggering and lowest-level thermodynamic reconstruction;
- saturation mixing ratio and neutral/stable/unstable Revised MM5 branches;
- `ISFTCFLX` roughness selection and the nonnegative QFX limit;
- a nonzero LH reconstructed when stored flux variables are zero;
- OHC26 for a complete warm layer and a partial 26-degree crossing interval;
- zero OHC26 for a cool surface and rejection of a missing crossing;
- explicit OHC exclusion for the no-ocean experiment;
- member, ensemble-mean, and sample-standard-deviation table contents;
- direct NR `LH` precedence and `QFX` fallback;
- NR OHC26 on the NR-native 150-km ocean mask;
- independent NR and member CSV-cache switches;
- unified member/NR `XTIME` and stored-flux source selection;
- initial-zero reconstruction and integrated-output stored-flux behavior;
- strict cache schema, finiteness, uniqueness, and configuration validation;
- no rewrite of CSV files selected as cache inputs;
- creation of the NR reference CSV and two separate nonempty PNG figures;
- a solid NR reference line on both figures.

Verification uses `/Users/kcfu/miniforge3/envs/wrf/bin/python` consistently,
following the project execution preference. Because the cluster data paths are
not mounted locally, local verification uses synthetic data; the delivered
script also includes a read-only input-validation stage for execution on the
cluster.
