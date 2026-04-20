function [j,dx,dxm]=toGrid(x)
% transfer obs. x to grid j and calculate its distance to grid j and j+1

j=floor(x);
dx = x - j;
dxm = 1.0-dx;

end
