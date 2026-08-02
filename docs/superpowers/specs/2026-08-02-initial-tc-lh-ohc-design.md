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

The script writes four files below a configurable output directory:

- `initial_tc150_lh_members.csv`;
- `initial_tc150_ohc_members.csv`;
- `initial_tc150_lh.png`;
- `initial_tc150_ohc.png`.

Each CSV contains experiment, display label, filter, member, center coordinates,
selected gridpoint counts, the member regional mean, and the corresponding
experiment/filter ensemble mean and sample standard deviation.

The LH figure contains all three experiments. The OHC figure contains only the
two ocean-running experiments. Within each experiment, EAKF and QCF_RHF are
offset horizontally. Six member values are shown as jittered points, with an
overlaid ensemble mean and standard-deviation error bar. Experiment colors use
the existing colorblind-safe palette; filters use distinct marker shapes. The
figures use independent scientifically meaningful y-axis units:
`W m-2` for LH and `kJ cm-2` for OHC.

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
- creation of two separate nonempty PNG figures.

Verification uses `/Users/kcfu/miniforge3/envs/wrf/bin/python` consistently,
following the project execution preference. Because the cluster data paths are
not mounted locally, local verification uses synthetic data; the delivered
script also includes a read-only input-validation stage for execution on the
cluster.
