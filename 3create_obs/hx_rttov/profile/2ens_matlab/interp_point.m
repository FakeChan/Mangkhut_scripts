function xpoint = interp_point(x2d,xloc,yloc,type)
% 'type': U,V,or others on mass point

[ny nx]=size(x2d);

if ( strcmp(type,'U') || strcmp(type,'V') )
   xloc_u = xloc + 0.5;
   % find the lower index for grid, and distances
   [i,dx,dxm]=toGrid(xloc_u);
   [j,dy,dym]=toGrid(yloc);
   % get grid cell corners surrounding ob location
   [ll,ul,lr,ur]=getCorners(i,j);
   % interpolate points
   xpoint=dym*(dxm*x2d(ll(2),ll(1))+dx*x2d(lr(2),lr(1)))+...
          dy*(dxm*x2d(ul(2),ul(1))+dx*x2d(ur(2),ur(1))); 
else if ( strcmp(type,'V') )
   yloc_v = yloc + 0.5;
   % find the lower index for grid, and distances
   [i,dx,dxm]=toGrid(xloc);
   [j,dy,dym]=toGrid(yloc_v);
   % get grid cell corners surrounding ob location
   [ll,ul,lr,ur]=getCorners(i,j);
   % interpolate points
   xpoint=dym*(dxm*x2d(ll(2),ll(1))+dx*x2d(lr(2),lr(1)))+...
          dy*(dxm*x2d(ul(2),ul(1))+dx*x2d(ur(2),ur(1)));
else
   % find the lower index for grid, and distances
   [i,dx,dxm]=toGrid(xloc);
   [j,dy,dym]=toGrid(yloc);
   % get grid cell corners surrounding ob location
   [ll,ul,lr,ur]=getCorners(i,j);
   % interpolate points
   xpoint=dym*(dxm*x2d(ll(2),ll(1))+dx*x2d(lr(2),lr(1)))+...
          dy*(dxm*x2d(ul(2),ul(1))+dx*x2d(ur(2),ur(1)));

end


end



