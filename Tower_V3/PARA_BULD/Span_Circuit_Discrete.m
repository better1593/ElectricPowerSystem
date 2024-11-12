function [ Span ] = Span_Circuit_Discrete( Span, GLB )
% Generating circuit parameters and config. info. for a span in either 
%   phase or mdoal domain with/without VF (VF applied to diagonal only)
%
%   OHL_Para=Span.line=
%    [x1 y1 z2 x2 y2 z2 0  0  0    0    0   0;   % pole-pole position
%     x1 y1 z2 x2 y2 z2 0  rc Rint Lint mur epr; % SW
%     x1 y1 z2 x2 y2 z2 dh rc Rint Lint mur epr; % HV->MV->LV, PH A->B->C 
%              ....                           ]; % dh: offset (hori)
%   Bran.list=[name list];  Bran.listdex=[b1 n1 n2];
%   Bran.num=[total #, air #, gnd  #, dist #, lump #, Speical #] 1-6
%   Node.list=[name list];  Node.listdex=[n1]; 
%   Node.num=[total #, air #, surf #, gnd  #] 1-4
%   Node.com=[name list];   Node.comdex=[int: n1,  ext: n2]; 
%   Meas.node=[n1]
%   Meas.bran=[b1]
%
%   GND.sig, GND.epi, GND.gnd (0=no, 1= perfect and (2) lossy)
%   VFIT: [r0+d0*s+sum(ri/s+di)]
%   VIFT.r(:,1+ordr +1),  VIFT.d(:,1+order +1) for gnd and/or conductor
%       [d h r a]=[R0 L0 Residual Pole]=[[R0 L0 b -a]

%   July 20, 2023

% (0) Constants
% (0a) Global constants
dT = GLB.dT;
dZ = GLB.slg;

% (0b) Span input
% Info = Span.Info;
% Smod = [Info{1,8:9}];                   % Vector fitting flag

Para = Span.Para;
Ht = Span.Ht; 

Cal.ord = Ht.ord;
Cal.Tcov = Para.Tcov;
Cal.Tinv = Para.Tinv;

Ncon = Span.Seg.Ncon;
% (1) Output---------------------------------------------------------------

% (2) Calculate waveequation parameters (Z and Y)
R = Para.Imp.R;
L = Para.Imp.L;
C = Para.Imp.C;
if Smod(1)+Smod(2)==0               % perfect ground/perfect conductor       
    D0 = inv(L/dT + R/2);
    Cal.AI = +D0*(L/dT - R/2);      % current wave eqn
    Cal.AV = -D0/dZ;
    Cal.ASRC = -D0;                 % induced voltage source
    Cal.Cphi = [];                  % VF term
                       
    D0 = inv(C);
    Cal.BI = -dT*D0/dZ;             % voltage wave eqn
    Cal.BSRC = +D0*dT/dZ;           % Cal.BSRC = -D0*dT/dZ;               
else                                % With VF               
    for i = 1:Ncon                  % diagnoal terms only
        Cal.E(i,i,:) = exp( Ht.a(i,i,:) * dT );   
        Cal.B(i,i,:) = Ht.r(i,i,:) ./ Ht.a(i,i,:) .* ( Cal.E(i,i,:) -1 );
    end
                
    D0 = inv(R/2 + L/dT + 0.5*sum(Cal.B,3));    % summation of all orders
    Cal.AI = -D0*(R/2 - L/dT);
    Cal.AV = -D0/dZ;
    Cal.ASRC = -D0;
        
    for i = 1:Cal.ord
        Cal.Cphi(:,:,i) = -1/2*D0*(Cal.E(:,:,i)+eye(Ncon));
    end
                
    D0 = inv(C);
    Cal.BI = -dT*D0/dZ;
    Cal.BSRC = +D0*dT/dZ;           % Cal.BSRC = -D0*dT/dZ;               
end 
Span.Cal = Cal;
end