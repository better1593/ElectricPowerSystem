function [Mij] = Cal_M_OHL(Pos)
 Mij = zeros(size(Pos,1),size(Pos,1));
 mu0=4*pi*1e-7;

%  Pos = [0    0  10.5 ;         % SW
%        -0.5  0  10.0 ;         % Phase A
%        +0.1  0  10.0 ;         % Phase B
%        +0.6  0  10.0 ] ;        % Phase C

for i1 = 1:size(Pos,1)-1
    for i2 = i1+1:size(Pos,1)
        d = abs(Pos(i1,7) - Pos(i2,7));
        h1 = Pos(i1,3);
        h2 = Pos(i2,3);
        Mij(i1,i2) = mu0 / 2 / pi * log((d^2 + (h1+h2)^2) / (d^2 + (h1 - h2)^2));
        Mij(i2,i1) = Mij(i1,i2);
    end
end
   
   
end

