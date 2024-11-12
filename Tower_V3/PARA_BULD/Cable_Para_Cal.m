function Para = Cable_Para_Cal(Line,VFIT,GND)
% Perfirm the calculation of OHL Parameters (Z and Y) with VFIF
%        Para.Tinv/Cov
%        Para.R/L/C/Ht
% Line = [x1 y1 z1 x2 y2 z2 oft r0 R L              % 1-10
%         sig mur epr mode1 mode1  b0 n1 n2 ]       % 11-18, 19 (comments)
% VFmod =[m1 m2]: m1 (cond), m2 (gnd), 1=yes, 0=no
% Seg.num = [tot# pha# sw# slen seg# depth]

% (1) Initilization 
Npha = Line.con(2);

% (1a) VFIT initlization
nc_fit = 5;
% fre = [1:10:90 1e2:1e2:9e2 1e3:1e3:9e3 1e4:1e4:9e4 1e5:1e5:6e5];
fre = [1:10:90 1e2:1e2:9e2 1e3:1e3:9e3 1e4:1e4:5e4];
Ht = [];

% (2) domain transfer matrix T and T_inv
Tcov = ones(Npha,Npha);
for ik = 2:Npha
    Tcov(ik,ik) = 1-Npha;
end
Tinv = -diag(ones(1,Npha-1));
Tinv = [ones(1,Npha-1);Tinv];
Tinv = [ones(Npha,1),Tinv]/Npha;

Para.Tcov = Tcov;
Para.Tinv = Tinv;

% (3) per-unit R, L and C           
[L,C]= Cal_LC_Cable(Line);
Cc = C(1:Npha,1:Npha);
Lc_modal = diag(diag(Tinv * L(1:Npha, 1:Npha) * Tcov));     % modal domain
Cc_modal = diag(diag(Tinv * Cc * Tcov));                    % modal domain

La = L(end,end);                                            % phase domain
Ca = C(end,end);                                            % phase domain

Z = Cal_ZY_Cable(Line, GND, fre);
for i = 1:numel(fre)                                        % Z_modal
    Zc_modal(:,:,i) = diag(diag(Tinv * Z(1:end-1,1:end-1,i) * Tcov));
    Zca_modal(:,:,i) = Tinv * Z(1:end-1,end,i);             % modal domain
    Zac_modal(:,:,i) = Z(end, 1:end-1, i) * Tcov;           % phase domain
end
Ht.ord=nc_fit; % (*) x (*) x Nord
Ht.c=Parameter_VF_Matrix_v2(Zc_modal, fre, nc_fit);         % modal domain
Ht.ca=Parameter_VF_Matrix_v2(Zca_modal, fre, nc_fit);       % modal domain
Ht.ac=Parameter_VF_Matrix_v2(Zac_modal, fre, nc_fit);       % phase domain 
Ht.a=Parameter_VF_Matrix_v2(Z(end,end,:), fre, nc_fit);     % phase domain 

Para.Rc = Ht.c.d;                                       % Npha x Npha xNord
Para.Rca = Ht.ca.d;                                     % Npha x 1 x Nord
Para.Rac = Ht.ac.d;                                     % 1 x Npha x Nord
Para.Ra = Ht.a.d;

Para.Lc = Ht.c.h + Lc_modal;                            % Npha x NphaxNord
Para.Lca = Ht.ca.h;                                     % Npha x 1 x Nord
Para.Lac = Ht.ac.h;                                         % phase domain
Para.La = Ht.a.h + La;                                      % phase domain 

Para.Cc = Cc_modal;
Para.Ca = Ca;                                               % phase domain 
Para.Ht = Ht;
Cw.C0 = blkdiag(Ca,Cc);                               % connected Tower Cw
Para.Cw = Cw;
end