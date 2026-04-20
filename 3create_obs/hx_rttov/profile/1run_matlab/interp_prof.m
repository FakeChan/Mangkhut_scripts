function xprof = interp_prof(x3d,xloc,yloc,type)
% 'type': U,V,or others on mass point

[nz ny nx]=size(x3d);

if ( strcmp(type,'U') )
   xloc_u = xloc + 0.5;
   % find the lower index for grid, and distances
   [i,dx,dxm]=toGrid(xloc_u);
   [j,dy,dym]=toGrid(yloc);
   % get grid cell corners surrounding ob location
   [ll,ul,lr,ur]=getCorners(i,j);
   % interpolate profiles
   for kk=1:nz
       xprof(kk)=dym*(dxm*x3d(kk,ll(2),ll(1))+dx*x3d(kk,lr(2),lr(1)))+...
                 dy*(dxm*x3d(kk,ul(2),ul(1))+dx*x3d(kk,ur(2),ur(1))); 
   end
elseif ( strcmp(type,'V') )
   yloc_v = yloc + 0.5;
   % find the lower index for grid, and distances
   [i,dx,dxm]=toGrid(xloc);
   [j,dy,dym]=toGrid(yloc_v);
   % get grid cell corners surrounding ob location
   [ll,ul,lr,ur]=getCorners(i,j);
   % interpolate profiles
   for kk=1:nz
       xprof(kk)=dym*(dxm*x3d(kk,ll(2),ll(1))+dx*x3d(kk,lr(2),lr(1)))+...
                 dy*(dxm*x3d(kk,ul(2),ul(1))+dx*x3d(kk,ur(2),ur(1)));
   end
else
   % find the lower index for grid, and distances
   [i,dx,dxm]=toGrid(xloc);
   [j,dy,dym]=toGrid(yloc);
   % get grid cell corners surrounding ob location
   [ll,ul,lr,ur]=getCorners(i,j);
   % interpolate profiles
   for kk=1:nz
       xprof(kk)=dym*(dxm*x3d(kk,ll(2),ll(1))+dx*x3d(kk,lr(2),lr(1)))+...
                 dy*(dxm*x3d(kk,ul(2),ul(1))+dx*x3d(kk,ur(2),ur(1)));
   end

end


end



