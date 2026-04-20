function [xloc,yloc]=latlon_to_ij(xlat,xlon,lat1,lon1)
% code adpated from DART wrf map utils
% input: xlat - ob latitudes
%        xlon - ob longitudes
%        lat1 - wrf%latitude(1,1)
%        lon1 - wrf%longitude(1,1)

% general parameters
EARTH_RADIUS_M = 6370000.0;
rad_per_deg = pi/180.0;

% model-related parameters
wrfd01_dx_m = 18000.0;

% tc WRF use MAP_PORJ=1 Lambert Conformal projection
% proj paramters: 
CEN_LAT = 25.0; CEN_LON = 135.0;
TRUELAT1 = 30.0; TRUELAT2 = 10.0;
STAND_LON = 135.0;

knowni = 1.0; knownj = 1.0;
rebydx = EARTH_RADIUS_M/wrfd01_dx_m;

% compute cone factor of a Lambert Conformal projection
if ( abs(TRUELAT1-TRUELAT2) > 0.1 )
   cone = log10(cos(TRUELAT1*rad_per_deg)) - log10(cos(TRUELAT2*rad_per_deg));
   cone = cone/(log10(tan((45.0-abs(TRUELAT1)/2.0)*rad_per_deg))-log10(tan((45.0-abs(TRUELAT2)/2.0)*rad_per_deg)));
else
   cone = sin(abs(TRUELAT1)*rad_per_deg);
end

% compute polei,polej
deltalon1 = lon1-STAND_LON;
if (deltalon1 > 180.0)
   deltalon1 = deltalon1 - 360.0;
end
if (deltalon1 < -180.0)
   deltalon1 = deltalon1 + 360.0;
end
tl1r = TRUELAT1 * rad_per_deg;
ctl1r = cos(tl1r);
rsw = rebydx*ctl1r/cone*(tan((90.0-lat1)*rad_per_deg/2.0)/tan((90.0-TRUELAT1)*rad_per_deg/2.0))^cone;
arg = cone*(deltalon1*rad_per_deg);
polei = knowni - rsw*sin(arg);
polej = knownj + rsw*cos(arg);


% compute i,j
deltalon = xlon-STAND_LON;
if (deltalon > 180.0) 
   deltalon = deltalon - 360.0;
end
if (deltalon < -180.0) 
   deltalon = deltalon + 360.0;
end
tl1r = TRUELAT1 * rad_per_deg;
ctl1r = cos(tl1r);
rm = rebydx*ctl1r/cone*(tan((90.0-xlat)*rad_per_deg/2.0)/tan((90.0-TRUELAT1)*rad_per_deg/2.0))^cone;
arg = cone*(deltalon*rad_per_deg);
xloc = polei - rm*sin(arg);  % i - distance index in west-east
yloc = polej + rm*cos(arg);  % j - distance index in south-north

end
