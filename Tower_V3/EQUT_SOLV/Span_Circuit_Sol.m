function [Hist, Ifdtd, MeasR] = Span_Circuit_Sol(Span, Hist, Vpeec)
% Perform one-step (time domain) simulation of a span using the FDTD method
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
%

% (1) Initialization
Smod= [Span.Info{1,8:9}];                   % smod=VF of [wire, GND_imp]
Cal = Span.Cal;
SRC = Span.SRC;
MeasPosiV = Span.Meas.Vn;                   % [Cond_id Seg_id]
MeasPosiI = Span.Meas.Ib;                   % [Cond_id Seg_id]

Ib = Hist.Ib;
Vn = Hist.Vn;
Phi = Hist.Phi;

Tinv = 1;
Tcov = 1;
if Smod(2) == 1                         % modal domain
    Tinv = Cal.Tinv;
    Tcov = Cal.Tcov;
end
Vn(:,1)   = Tinv * Vpeec.head;          % terminal votlage
Vn(:,end) = Tinv * Vpeec.tail;          % terminal votlage

% (2) get the hist. value of VF terms: Cpp=sum((B+1)/2)phi_i in I eqn
Cpp = 0;                                % VF term
for i = 1:Cal.ord
    Cpp = Cpp+Cal.Cphi(:,:,i)*Phi(:,:,i);
end

% (3)Update both Vn and Ib
Ib = real(Cal.AI*Ib + Cal.AV*(Vn(:,2:end)-Vn(:,1:end-1)) + Cpp);
Vn(:,2:end-1)=Vn(:,2:end-1) + Cal.BI*(Ib(:,2:end)-Ib(:,1:end-1));

if SRC.pos >= 1                          % dir. lightning to the span
    Is = Tinv * SRC.ISF;
    Vn(:,SRC.pos)=Vn(:,SRC.pos)+Cal.BSRC*Is;
elseif SRC.pos == 0                      % ind. lightning 
    Vs = Tinv * SRC.VSF; 
    Ib = real(Ib + Cal.ASRC*Vs);
end
              
% (4) Update convolution term phi_i
for i = 1:Cal.ord
     Phi(:,:,i)=Cal.B(:,:,i)*Ib + Cal.E(:,:,i)*Phi(:,:,i);
end

% (5) Update measurement terms (modal->phase domain)
for i = 1:size(MeasPosiV,1)                 % get measurement results
    Vtmp = Tcov * Vn(:,MeasPosiV(i,2));     % getting column of Vn
    MeasR.Vn(i) =  Vtmp(MeasPosiV(i,1));    % getting row of Vn
end
                
for i = 1:size(MeasPosiI,1)
    Itmp = Tcov * Ib(:,MeasPosiI(i,2));     % modal->phase domain
    MeasR.Ib(i) =  Itmp(MeasPosiI(i,1));    % modal->phase domain
end

% (6) Update current source to the tower
Iftdt.head = Tcov * Ib(:,1);
Iftdt.tail = Tcov * Ib(:,end);

% (7) summary of the outputs
clear Hist;
Hist.Ib = Ib;                   % mode doamin
Hist.Vn = Vn;                   % mode doamin
Hist.Phi = Phi;                 % mode doamin
end

