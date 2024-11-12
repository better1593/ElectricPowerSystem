function [Hist, Ifdtd, MeasR] = Cable_Circuit_Sol(Cable, Hist, Vpeec)
% Perform one-step (time domain) simulation of cable using the FDTD method
%      (1) either in phase domain (Info{9}=0) or modal doamin (Info{9}=1)
%      (2) either with VF ([Info{8:9}]>0) or without VF ([Info{8:9}]=0)
% Input:  Span, Hist.Vn/Ib/Phi (modal), Vpeec (phase), SRC.ISF/VSF (phase)
% Output:       Hist.Vn/Ib/Phi (modal), Ifdtd (phase), Meas (phase)
%         Vpeec.head/tail: node votlage from connected towers
%         Ifdtd.head/tail: meas current to connected towers
%
% Options: VF:                  Cpp = 0 (no), Cpp ~=0 (yes)
%          Modal/phase doamin:  Tinv/Tcov = 1 (phase), ~= 1 (modal)
%          Dir/Ind source:      SRC.pos = -1 (Ind), = 0 (Dir) >1 (span)
% Line.num = Seg.num = [tot# pha# sw# slen seg# depth]
%

% (1) Initialization
Smod= [Cable.Info{1,8:9}];              % smod=VF of [wire, GND_imp]
Cal = Cable.Cal;
SRC = Cable.SRC;
MeasPosiV = Cable.Meas.Vn;              % [Cond_id Seg_id]
MeasPosiI = Cable.Meas.Ib;              % [Cond_id Seg_id]

Tinv = 1;
Tcov = 1;
if Smod(2) == 1                         % modal domain
    Tinv = Cal.Tinv;
    Tcov = Cal.Tcov;
end

Nord = Cal.ord;                         % VF index
Nseg = Cable.Line.con(5);               % # of seg  
Npha = Cable.Line.con(2);               % Core #
srng = 1:Nseg;
prng = 1:Npha;

Ic_m = Hist.Ic_m;                       % Ic(k-0.5), Ic(k+0.5)
Ic_m2nd = Ic_m;                         % Ic(k-0.5)
Vc_m = Hist.Vc_m;
Ia1_p = Hist.Ia1_p;                     % Ia(k-0.5), Ia(k-1.5)
Ia2_p = Hist.Ia2_p;                     % Ia(k-0.5), Ia(k-1.5)
Va_p = Hist.Va_p;                       % Va(k-1)
Phi_c = Hist.Phi_c;                     % phi_c for indv order
Phi_ca = Hist.Phi_ca;                   % phi_ca for indv order
Phi_ac = Hist.Phi_ac;                   % phi_ac for indv order
Phi_a = Hist.Phi_a;                     % phi_a for indv order

Va_p(1,1)   = Vpeec.head(1);
Va_p(1,end) = Vpeec.tail(1);
Vc_m(:,1)   = Tinv * Vpeec.head(2:end); % terminal votlage
Vn_m(:,end) = Tinv * Vpeec.tail(2:end); % terminal votlage

% (2) get the hist. value of VF terms: Cpp=sum((B+1)/2)phi_i in I eqn
% Ic solver
Cpp_c  = 0;                             % ind. phi_c  (Npha X Nseg x Nord)
Cpp_ca = 0;                             % ind. phi_ca (Npha X Nseg x Nord)
for i = 1:Nord
    Cpp_c  = Cpp_c + Cal.Phic(prng,prng,i)*Phi_c(prng,srng,i);% Npha x Nseg
    Cpp_ca = Cpp_ca + Cal.Phica(1,i)*Phi_ca(prng,srng,i);     % Npha x Nseg 
end

% Ia solver
Cpp_ac = 0;                             % ind. phi_ac (Npha X Nseg x Nord)
Cpp_a  = 0;                             % ind. phi_a  (1 X Nseg x Nord)
for i = 1:Nord
    Cpp_a  = Cpp_a  + Cal.Phia(1,i)*Phi_a(1,srng,i);        % 1 x Nseg !!!!
    Cpp_ac = Cpp_ac + Cal.Phiac(1,i)*Phi_ac(1,srng,i);      % 1 x Nseg !!!!
end

% (3)Update both Vn and Ib
% Ic solver
Ic_m = real(Cal.AIc*Ic_m+Cal.AVc*(Vc_mo(prng,2:end)-Vc_m(prng,1:end-1))...
                                                                  + Cpp_c);
Ic_m(1,srng) = Ic_m(1,srng)+ ...
            real(Cal.AIca1(1)*Ia1_p + Cal.AIca2(1)*Ia2_p + Cpp_ca(1,:)); 
Ia2_p = Ia1_p;                           % 1 x Nseg !!!! 

% Ia solver
tmp1=real(Cal.AIa*Ia1_p + Cal.AVa*(Va_p(:,2:end)-Va_p(:,1:end-1)) + Cpp_a);                                     
tmp2=real(Cal.AIac1(1)*Ic_m(1,:) + Cal.AIac2(1)*Ic_m2nd(1,:) + Cpp_ac);  
Ia1_p = tmp1+tmp2;

if SRC.pos == 0                           % Indirec strike
    Ia1_p = Ia1_p + real(Cal.ASRC*SRC.VSF);
end

% Vc and Va sover
Vc_m(:,2:end-1) =Vc_m(:,2:end-1)+Cal.BIc*(Ic_m(:,2:end)- Ic_m(:,1:end-1));                                              
Va_p(:,2:end-1) =Va_p(:,2:end-1)+Cal.BIa*(Ia1_p(:,2:end)-Ia1_p(:,1:end-1));
                                                   
% (4) Update convolution term phi_i
for i = 1:Nord
    Phi_c(:,:,i) =Cal.Bc(:,:,i)*Ic_m + Cal.Ec(:,:,i)*Phi_c(prng,srng,i);   % Npha x Nseg x Nord
    Phi_ca(1,:,i)=Cal.Bca(1,i)*Ia1_p(1,srng)+Cal.Eca(1,i)*Phi_ca(1,srng,i);% Npha x Nseg x Nord

    Phi_ac(:,:,i)=Cal.Bac(1,i)*Ic_m(1,:)+Cal.Eac(1,i)*Phi_ac(:,srng,i);    % Npha x Nseg x Nord                       
    Phi_a(:,:,i) =Cal.Ba(1,i)*Ia1_p(1,:)+Cal.Ea(1,i)*Phi_a(:,srng,i);  
end                                     % 1 x Nseg x Nord      !!!!

% (5) Update measurement terms 
for i = 1:size(MeasPosiV,1)             % get measurement results
    V_phase = [Va_p(MeasPosiI(i,2)); Tcov*Vc_m(:,MeasPosiV(i,2))];  % m->p
    MeasR.Vn(i) = V_phase(MeasPosiV(i,1));
end                
for i = 1:size(MeasPosiI,1)
    I_phase = [Ia1_p(MeasPosiI(i,2)); Tcov*Ic_m(:,MeasPosiI(i,2))]; % m->p
    MeasR.Ib(i) = I_phase(MeasPosiI(i,1));
end

% (6) Update current source to the tower
Iftdt.head = [Ia1_p(1); Tcov*Ic_m(:,1)];
Iftdt.tail = [Ia1_p(end); Tcov*Ic_m(:,end)];

% (7) summary of the outputs
clear Hist;
Hist.Ic_m = Ic_m;
Hist.Vc_m = Vc_m;
Hist.Ia1_p = Ia1_p;
Hist.Ia2_p = Ia2_p;
Hist.Va_p = Va_p;
Hist.Phi_c = Phi_c;
Hist.Phi_ca = Phi_ca;
Hist.Phi_ac = Phi_ac;
Hist.Phi_a = Phi_a;
end