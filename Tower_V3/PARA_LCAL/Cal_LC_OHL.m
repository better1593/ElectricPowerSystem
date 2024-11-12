function [L,C]=Cal_LC_OHL(High,Dist,r0)
% Calculate OHL Parameters (L and C per unit) with Height and hori. Dist

Vair = 3e8;                                     % Velocity in free space
mu0 = 4*pi*1e-7;
km = mu0/(2*pi);
Ncon = size(High,1);

out = log(2 * High./r0);
L = diag(out);

for i1 = 1:Ncon-1
    for i2 = i1+1:Ncon
        d = abs(Dist(i1) - Dist(i2));
        h1 = High(i1);
        h2 = High(i2);
        L(i1,i2) = 0.5*log((d^2+(h1+h2)^2)/(d^2+(h1-h2)^2));
        L(i2,i1) = L(i1,i2);
    end
end
L = km*L;
C = inv(L)/Vair^2;
end
