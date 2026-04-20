function tk=wrf_tk(t,pres)
% convert potential temperature to temperature
% adapted from NCL 
% https://github.com/yyr/ncl/blob/master/ni/src/lib/nfp/wrfW.c wrapper wrf_tk -> DCCOMPUTETK
% https://github.com/yyr/ncl/blob/master/ni/src/lib/nfpfort/wrf_user.f DCCOMPUTETK

P1000MB = 100000.0;
R_D = 287.0;
CP = 7.0*R_D/2.0;

PI = (pres/P1000MB)^(R_D/CP);
tk = PI*t;

end
