function [xloc, yloc] = latlon_to_fractional_ij(xlat, xlon, obs_lat, obs_lon)
% LATLON_TO_FRACTIONAL_IJ  map observation (lat, lon) onto continuous
% floating-point grid coordinates of the d02 ensemble grid
%
%   Index convention follows interp_prof.m / interp_point.m, which treat a
%   field as (nz, ny, nx) (or (ny, nx) for 2-D) with
%       xloc : fractional index along the 3rd dimension (west-east)
%       yloc : fractional index along the 2nd dimension (south-north)
%   interp_prof/interp_point access floor(idx) and floor(idx)+1 through
%   toGrid/getCorners, so the executable range is [1, nx) x [1, ny).
%
%   The mapping is built with scatteredInterpolant from the (longitude,
%   latitude) of the d02 grid to its grid indices (linear, no extrapolation).
%   A forward closure check re-interpolates XLAT/XLONG at the obtained
%   fractional coordinates with the same bilinear scheme used for the model
%   variables and compares against the target observation locations.
%
%   Input:
%       xlat, xlon : 2-D XLAT/XLONG fields of the d02 ensemble grid (ny x nx)
%       obs_lat    : target observation latitudes  (npoint x 1)
%       obs_lon    : target observation longitudes (npoint x 1)
%   Output:
%       xloc, yloc : continuous grid coordinates, same order and shape as
%                    the input observation locations

% WRF XLAT/XLONG are commonly stored as NetCDF float and therefore read as
% MATLAB single arrays. scatteredInterpolant requires double input points.
xlat = double(xlat);
xlon = double(xlon);

[ny, nx] = size(xlat);
if ~isequal(size(xlon), [ny nx])
    error('XLAT and XLONG must have identical 2-D sizes, got %s and %s.', ...
          mat2str(size(xlat)), mat2str(size(xlon)));
end

% ---- bring observed longitudes to the longitude convention of the d02 ----
% (e.g. 0-360 vs -180..180 grids)
lon_flat = xlon(:);
if max(lon_flat) > 180
    % grid stored in 0..360
    obs_lon(obs_lon < 0) = obs_lon(obs_lon < 0) + 360;
elseif min(lon_flat) < 0
    % grid stored in -180..180
    obs_lon(obs_lon > 180) = obs_lon(obs_lon > 180) - 360;
end

% ---- sanity: all target locations must be finite ----
if ~all(isfinite(obs_lat)) || ~all(isfinite(obs_lon))
    bad = ~isfinite(obs_lat) | ~isfinite(obs_lon);
    idx = find(bad, 1);
    error('Observation %d has non-finite location: lat=%.6f, lon=%.6f.', ...
          idx, obs_lat(idx), obs_lon(idx));
end

% ---- index fields: Ig along west-east (2nd dim), Jg along south-north ----
[Jg, Ig] = ndgrid(1:ny, 1:nx);
F_x = scatteredInterpolant(xlon(:), xlat(:), Ig(:), 'linear', 'none');
F_y = scatteredInterpolant(xlon(:), xlat(:), Jg(:), 'linear', 'none');
xloc = reshape(F_x(obs_lon(:), obs_lat(:)), size(obs_lat));
yloc = reshape(F_y(obs_lon(:), obs_lat(:)), size(obs_lat));

% ---- bounds: floor(idx) and floor(idx)+1 must stay inside the array ----
% A tiny index-space tolerance absorbs floating-point round-off at the
% domain boundary only; anything truly outside errors (no extrapolation,
% no nearest-neighbour fallback).
loc_tol = 1e-8;
out_of_range = ~isfinite(xloc) | ~isfinite(yloc) | ...
               xloc < 1 - loc_tol | xloc >= nx + loc_tol | ...
               yloc < 1 - loc_tol | yloc >= ny + loc_tol;
if any(out_of_range)
    idx = find(out_of_range, 1);
    error('Observation %d (lat=%.6f, lon=%.6f) maps to out-of-range d02 grid coordinate xloc=%.9f, yloc=%.9f; valid range is [1, %d) x [1, %d) and extrapolation beyond the d02 domain is not allowed.', ...
          idx, obs_lat(idx), obs_lon(idx), xloc(idx), yloc(idx), nx, ny);
end
% clamp round-off-level excursions back onto the executable interval
xloc = min(max(xloc, 1), nx - eps(nx));
yloc = min(max(yloc, 1), ny - eps(ny));

% ---- forward closure check: interpolate XLAT/XLONG back at (xloc, yloc) ----
xlat_chk = bilinear_grid(xlat, xloc, yloc);
xlon_chk = bilinear_grid(xlon, xloc, yloc);
% d02 grid spacing is 1.5 km (~0.0135 deg in latitude), so 1e-3 deg is a
% loose sanity bound (< 10% of a grid cell) far above the ~1e-10 level
% expected from the inverse mapping; it never trips on float round-off.
close_tol = 1e-3;
bad_close = ~isfinite(xlat_chk) | ~isfinite(xlon_chk) | ...
            abs(xlat_chk - obs_lat) > close_tol | ...
            abs(xlon_chk - obs_lon) > close_tol;
if any(bad_close)
    idx = find(bad_close, 1);
    error('Forward closure check failed for observation %d: target (lat=%.6f, lon=%.6f) -> grid (xloc=%.9f, yloc=%.9f) -> interpolated (lat=%.6f, lon=%.6f), |dLat|=%.3e deg, |dLon|=%.3e deg.', ...
          idx, obs_lat(idx), obs_lon(idx), xloc(idx), yloc(idx), ...
          xlat_chk(idx), xlon_chk(idx), ...
          abs(xlat_chk(idx) - obs_lat(idx)), abs(xlon_chk(idx) - obs_lon(idx)));
end

end

function v = bilinear_grid(x2d, xloc, yloc)
% bilinear weighting identical to interp_prof/interp_point (mass-point
% branch): dim 1 indexed by yloc, dim 2 indexed by xloc
[ny, nx] = size(x2d);
i = floor(xloc);
j = floor(yloc);
dx = xloc - i;
dxm = 1 - dx;
dy = yloc - j;
dym = 1 - dy;
v = dym .* (dxm .* x2d(sub2ind([ny nx], j,     i))     + dx .* x2d(sub2ind([ny nx], j,     i + 1))) + ...
    dy  .* (dxm .* x2d(sub2ind([ny nx], j + 1, i))     + dx .* x2d(sub2ind([ny nx], j + 1, i + 1)));
end
