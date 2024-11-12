function Para = OHL_Para_Cal(Line,VFmod,VFIT,GND)
% Perfirm the calculation of OHL Parameters (Z and Y) with VFIF
%        Para.Tinv/Cov
%        Para.R/L/C/Ht
% Line = [x1 y1 z1 x2 y2 z2 oft r0 R L              % 1-10
%         sig mur epr mode1 mode1 b0 n1 n2 ]       % 11-18, 19 (comments)
% VFmod =[m1 m2]: m1 (cond), m2 (gnd), 1=yes, 0=no
% Vector fitting parameters
%       [d h r a]=[R0 L0 Residual Pole]=[[R0 L0 b -a]
%
% (1) Initilization 
Vair = 3e8;                             % velocity in free space
High = 0.5*(Line(:,3)+Line(:,6));       % height
Dist = Line(:,7);                       % Horizontal offset
r0 = Line(:,8);                         % conductor radius
Rin = Line(:,9);                        % fixed inner resistance
Lin = Line(:,10);                       % fixed inner rinductance
sig = Line(:,11);
mur = Line(:,12);
epr = Line(:,13);

Ncon = size(Line,1);                    % # of total conductors

% (1a) VFIT initlization
nc_fit = 5;
fre = [1:10:90 1e2:1e2:9e2 1e3:1e3:9e3 1e4:1e4:9e4 1e5:1e5:6e5];

% (2) domain transfer matrix T and T_inv
Tcov = ones(Ncon,Ncon);
for i = 2:Ncon
    Tcov(i,i) = 1-Ncon;
end
Tinv = -diag(ones(1,Ncon-1));
Tinv = [ones(1,Ncon-1);Tinv];
Tinv = [ones(Ncon,1),Tinv]/Ncon;

Para.Tcov = Tcov;
Para.Tinv = Tinv;

% (3) per-unit R, L and C 
[L,C] = Cal_LC_OHL(High,Dist,r0);
Ht = [];

if VFmod(2)==0               % Gnd = 0: Perfect ground impedance
    if VFmod(1)==0           % Cond = 0: perfect ground/perfect conductor
        Para.Imp.R = diag(Rin);         % constant Rin in Wire (:,9)
        Para.Imp.L = L+diag(Lin);       % constant Lin in Wire (:,10)    
        Para.Imp.C = C;
    else                     % Cond =1: VF for conductor internal imped.           
        Zc = Cal_Zc_OHL(r0,sig,mur,epr,fre); 
        Ht=Parameter_VF_Matrix_v2(Zc, fre, nc_fit); 
        close all;
        Para.Imp.R = Ht.d;              % Vfit.d (add DC components)
        Para.Imp.L = L + Ht.h;          % Vfit.h (add DC components)
        Para.Imp.C = C;
    end
else                        % Gnd = 1: Lossyt ground impedance (VFIT)  
    if VFmod(1)==0          % Cond =0: perfect conductor and lossy ground        
        R = diag(Rin);      % constant Rin in Wire (:,9)
        L = L + diag(Lin);  % constant Lin in Wire (:,10)
        
        R = diag(diag(Tinv * R * Tcov));% phase -> mode
        L = diag(diag(Tinv * L * Tcov));
        C = diag(diag(Tinv * C * Tcov));        
        
        Zg = Cal_Zg_OHL(High,Dist,r0,fre,GND);              % Vfit
        for i = 1:numel(fre)           
            Zg(:,:,i) = diag(diag(Tinv*Zg(:,:,i)*Tcov));    % Modal domain       
        end        
        Ht=Parameter_VF_Matrix_v2(Zg, fre, nc_fit);     % Vector fitting
        close all;

        Para.Imp.R = R + Ht.d;          % Vfit.d (add DC components)    
        Para.Imp.L = L + Ht.h;          % Vfit.h (add DC components)
        Para.Imp.C = C;
    else                    % Cond =1: VF for gnd and cond impedance        
        L = diag(diag(Tinv * L * Tcov));
        C = diag(diag(Tinv * C * Tcov));      
                
        Zc = Cal_Zc_OHL(r0,sig,mur,epr,fre);
        Zg = Cal_Zg_OHL(High,Dist,r0,fre,GND);        
        for i = 1:numel(fre)           
            Zg(:,:,i) = diag(diag(Tinv * Zg(:,:,i) * Tcov));
            Zc(:,:,i) = diag(diag(Tinv * Zc(:,:,i) * Tcov));           
        end        
        Ht=Parameter_VF_Matrix_v2(Zc+Zg, fre, nc_fit);  
        close all;

        Para.Imp.R = Ht.d;              % Vfit.d (add DC components)                  % Vfit.d
        Para.Imp.L = L + Ht.h;          % Vfit.h (add DC components) 
        Para.Imp.C = C;
    end      
end
Para.Ht = Ht;
end