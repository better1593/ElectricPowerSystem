function [L] = Cal_L_OHL(OHL_Para)

mu0 = 4*pi*1e-7;
h = OHL_Para(1:end, 3);
r = OHL_Para(1:end, 8);
%  Mur = OHL_Para(1:end, 12);
 

%  Pos = [0    0  10.5 ;         % SW
%        -0.5  0  10.0 ;         % Phase A
%        +0.1  0  10.0 ;         % Phase B
%        +0.6  0  10.0 ] ;        % Phase C

km = mu0/(2*pi);
out = km .* log(2 * h./r);
L = diag(out);

% for i = 1:size(h,1)    
%     hi = h(i);
%     ri = r(i);
%     L(i,i) = mu0 * Mur(i) /2 /pi * log(2 * hi /ri);
% end   
end

