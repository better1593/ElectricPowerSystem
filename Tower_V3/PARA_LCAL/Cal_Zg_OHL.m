function Zg = Cal_Zg_OHL(High,Dist,r0,fre,GND)

ep0=8.854187818e-12;
mu0=4*pi*1e-7;

Sig_g = GND.sig;
Mur_g = GND.mur * mu0;
Eps_g = GND.epr * ep0;
Ncon = size(r0,1);
Nfre = length(fre);
Zg = zeros(Ncon,Ncon,Nfre);

omega = 2 * pi * fre;                          
gamma = sqrt(1i * Mur_g * omega .* (Sig_g + 1i * omega * Eps_g));
km = 1i*omega*Mur_g/4/pi;

for i1 = 1:Ncon
    for i2 = i1:Ncon
        d = abs(Dist(i1) - Dist(i2));
        h1 = High(i1);
        h2 = High(i2);
        Zg(i1,i2,:) = km.*log(((1 +gamma.*(h1 + h2)/2).^2+(d.*gamma./2).^2)./((gamma.*(h1+h2)./2).^2 +(d .*gamma./2).^2));
        Zg(i2,i1,:) = Zg(i1,i2,:);
    end
end
for i = 1:Ncon
    h = High(i);
    Zgg(i,i,:) = km.*log(((1+gamma*h).^2)./((gamma*h).^2));
end
end

