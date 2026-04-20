function [ll,ul,lr,ur]=getCorners(i,j)

% lower left
ll(1) = i;
ll(2) = j;

% lower right
lr(1) = ll(1) + 1;
lr(2) = ll(2);

% upper left
ul(1) = ll(1);
ul(2) = ll(2) + 1;

% upper right
ur(1) = lr(1);
ur(2) = ul(2);

end
