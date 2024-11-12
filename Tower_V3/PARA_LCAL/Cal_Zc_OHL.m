function Zc = Cal_Zc_OHL(ri,sig,mur,epr,fre)

% Physical constant
E = 200;
mu0 = 4*pi*1e-7;
ep0 = 8.854187818e-12;
Ncon = size(ri,1);
Nfre = length(fre);
Zc = zeros(Ncon,Ncon,Nfre);

omega = 2 * pi * fre;                          
for i = 1: Nfre    
    gamma = sqrt(1i*mu0*mur*omega(i).*(sig+1i*omega(i)*ep0*epr));   
    Ri = ri.*gamma;    
    I0i = besseli(0,Ri);
    I1i = besseli(1,Ri);  
    maxR = max(abs(Ri));
    if maxR < E
        out = 1i*mu0*mur*omega(i).*I0i./(2*pi*Ri.*I1i);  
    else
        out = gamma./(2*pi*ri.*sig);
    end
    Zc(:,:,i) = diag(out);
end
end

