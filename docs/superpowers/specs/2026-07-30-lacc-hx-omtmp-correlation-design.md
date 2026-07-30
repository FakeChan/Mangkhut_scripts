# LACC Hx-OM_TMP Correlation Design

## Objective

Refactor `plot_scripts/plot_LACC_leadlag_corr.py` from a WRF time-series
`TK-TSK` diagnostic into an ensemble-space LACC diagnostic for satellite
brightness-temperature priors (`Hx`) and the current ocean surface state
(`OM_TMP` level 0).

The diagnostic must reproduce the ensemble-correlation logic used by Lu et al.
(2015): correlate the current slow-variable forecast ensemble with leading
fast-variable forecast ensembles, and separately correlate it with leading
time-averaged forecast ensembles.

## User Configuration

All editable settings remain in a configuration block near the top of the
script. The script will not require command-line arguments or environment
variables.

Required settings:

- `HX_DIR`: root containing member Hx directories.
- `MEM_DIR`: directory containing `firstguess_d01.mem001` through
  `firstguess_d01.mem050`.
- `PROFILE_PATH`: one center-time RTTOV profile containing the 676 observation
  coordinates.
- `CURRENT_TIME`: full reference time, initially
  `2018-09-10_00:00:00`.
- `MAX_LAG_HOURS`: maximum lead time, initially 12.
- `LAG_INTERVAL_HOURS`: lead-time interval, initially 3.
- `MEMBER_START` and `MEMBER_END`: initially 1 and 50.
- `SENSOR`: initially `AMSUA`.
- `CHANNEL`: initially 4.
- `DOMAIN`: initially `d01`.
- `OUTPUT_DIR`: output directory for tables and the figure.

For a member number and valid time, the Hx path is:

```text
{HX_DIR}/mem{nnn}/{SENSOR}/BT_{dd}_{hh}_{mm}/obs_{DOMAIN}_ch{CHANNEL}_totalline.txt
```

The current member-state path is:

```text
{MEM_DIR}/firstguess_{DOMAIN}.mem{nnn}
```

The script derives `LAG_HOURS = [0, 3, 6, 9, 12]` from the initial maximum
lag and interval, then derives each historical valid time from `CURRENT_TIME`.
Changing `MAX_LAG_HOURS` or `LAG_INTERVAL_HOURS` is sufficient to diagnose a
different regular set of lead times.

## Input Parsing

### Profile Coordinates

The RTTOV profile is parsed by searching for:

```text
! Elevation (km), latitude and longitude (degrees)
```

The following line contains elevation, latitude, and longitude. Exactly 676
coordinate pairs are required, in the same order as the 676 Hx lines.

### Hx Ensembles

Each Hx text file must contain exactly 676 finite numeric values. Files are
loaded into an array with dimensions:

```text
(number_of_lags, number_of_members, 676)
```

The member ordering is identical for Hx and the current member-state files.

### Current OM_TMP Ensemble

For every current member file, read:

```text
OM_TMP[0, 0, :, :]
```

after validating the dimensions. Read the member's `XLAT` and `XLONG` fields
and linearly interpolate the surface ocean temperature to the 676 profile
coordinates. Interpolation is performed independently for every member so that
member-specific coordinate fields remain valid.

Observation locations outside a member domain are rejected with an error rather
than silently extrapolated.

## LACC Statistics

All correlations are Pearson correlations calculated along the ensemble-member
dimension. Spatial points are never pooled with members.

For observation point \(k\), member \(m\), and nonnegative lead time \(\tau\):

\[
r_k(\tau) =
\operatorname{corr}_m
\left[
OM\_TMP_k^{f,(m)}(t),
Hx_k^{f,(m)}(t-\tau)
\right].
\]

This produces the single-time lead-lag diagnostic corresponding to Fig. 2a of
Lu et al. (2015).

The averaged windows always include the current time. With the initial
configuration, the windows are:

```text
Ave1 = [t]
Ave2 = [t - 3 h, t]
Ave3 = [t - 6 h, t - 3 h, t]
Ave4 = [t - 9 h, t - 6 h, t - 3 h, t]
Ave5 = [t - 12 h, t - 9 h, t - 6 h, t - 3 h, t]
```

For the first \(L\) configured lead times after sorting them in ascending order:

\[
\overline{Hx}_{k,L}^{f,(m)}
=
\frac{1}{L}
\sum_{\tau \in W_L} Hx_k^{f,(m)}(t-\tau),
\]

\[
r_{k,L} =
\operatorname{corr}_m
\left[
OM\_TMP_k^{f,(m)}(t),
\overline{Hx}_{k,L}^{f,(m)}
\right].
\]

This produces the leading-averaged diagnostic corresponding to Fig. 2b of Lu
et al. (2015).

For each lag or window, the plotted central curve is the arithmetic mean of all
finite, signed pointwise correlations. The 25th-75th percentile range is shown
as spatial spread. The number of valid points is reported.

If either ensemble vector at a point has zero variance or fewer than three
finite member pairs, that point's correlation is `NaN` and is excluded from the
spatial summary.

## Outputs

The output basename contains the domain, channel, and current time. The script
writes:

- `single_lag_pointwise_*.csv`: correlation for every point and lag.
- `single_lag_summary_*.csv`: mean, median, quartiles, minimum, maximum, and
  valid-point count for every lag.
- `averaged_window_pointwise_*.csv`: correlation for every point and window.
- `averaged_window_summary_*.csv`: spatial summary for every window.
- `omtmp_interpolated_*.csv`: coordinates and interpolated current `OM_TMP`
  values for every member and point.
- `lacc_hx_omtmp_corr_*.png`: two panels showing the single-lag and
  leading-averaged correlation curves.

The figure labels positive lag as `Hx leads current OM_TMP`. The averaged panel
labels each point with its included lead times so irregular choices such as
`[0, 3, 6, 9, 12]` are explicit in the result.

## Error Handling

Execution stops with an explicit message when:

- the maximum lag is negative;
- the lag interval is not positive;
- the maximum lag is not an integer multiple of the lag interval;
- a profile does not contain exactly 676 coordinates;
- an Hx file is missing or does not contain exactly 676 values;
- a current member file or required NetCDF variable is missing;
- array dimensions are incompatible;
- an observation coordinate lies outside the interpolation domain.

Zero ensemble variance is a valid diagnostic outcome and produces `NaN` rather
than terminating the run.

## Verification

Automated tests will use small synthetic profile, Hx, and NetCDF fixtures to
verify:

- marker-based profile coordinate parsing and count validation;
- Hx path/time formatting across the day boundary;
- linear interpolation of `OM_TMP` level 0;
- pointwise ensemble correlations;
- cumulative windows that contain time \(t\);
- signed spatial summaries and zero-variance handling;
- creation of both result tables and the two-panel plot.

The supplied `/Users/kcfu/Downloads/prof09_12:00.dat` will also be parsed during
local verification to confirm that it contains 676 coordinates.
