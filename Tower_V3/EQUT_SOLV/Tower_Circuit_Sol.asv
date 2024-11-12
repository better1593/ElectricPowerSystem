function [Hist, MeasR] = Tower_Circuit_Sol(Tower,Hist)
%   VSF.dat: =[t,wires] indirect lightning voltage source
%   VSF.pos: =[wire index]
%   ISF.dat: =[t,wire] direct lightning current source
%   ISF.pos: =[node index]
%   Bran.num=[total #, wair #, wgnd  #, lair #, gnf #, other]
%   Node.num=[total #, air #, surf #, gnd  #]
%
%   VFIT: [r0+d0*s+sum(ri/s+di)]
%   VIFT.rc(:,1+ordr +1),  VIFT.dc(:,1+order +1) for int imp of conductor
%   VIFT.rc(:,1+ordr +1),  VIFT.dc(:,1+order +1) for ground impedance

% (1) initialization
Cal = Tower.Cal;
SRC = Tower.SRC;
Meas = Tower.Meas;
Kt = Cal.Kt ;

Ib1 = Hist.Ib1;                     % hist data of bran current
Ib2 = Hist.Ib2;                     % hist data of span meas current
Ib3 = Hist.Ib3;                     % hist data of cable meas current
Ib = [Ib1; Ib2; Ib3];
Vn = Hist.Vn;                       % hist data of node votlage
Veqf = Hist.Veqf;                   % vector fitting

Nn =  length(Vn);                   % total Node #
Nb1 =  length(Ib1);                 % tower Bran #
Nb2 =  length(Ib2);                 % span-conencted Bran #
Nb3 =  length(Ib3);                 % cable-connected Bran #
Nb = Nb1 + Nb2 + Nb3;
vsc = zeros(Nb,1);

% (2) Source terms 
% (a) bran equations -A*Vn -(R+L/dt)Ib = Us - L/dt*Ib(n-1)
vsa = SRC.VSF - Cal.X*Ib1;
% (b) node equations (G+C/dt)Vn -A'*Ib -Cw'*Im = Is - C/dt*Vn(n-1)
is1 = SRC.ISF1 + Cal.B1*Vn;
% (c) span node equ C0/dt*Vn + E*Im = +/-2*Ifdtd - Im(n-1) - C0/dt*Vn(n-1)
is2 = [];
if Nb2 ~= 0
    is2 = SRC.ISF2 - Cal.Ib2 + Cal.B2*Vn;
end
% (d) cable node equ C0/dt*Vn + E*Im = +/-2*Ifdtd - Im(n-1) - C0/dt*Vn(n-1)
is3 = [];
if Nb3 ~= 0
    is3 = SRC.ISF3 - Cal.Ib3 + Cal.B3*Vn;
end

% (3) vsc contri. by vector fitting circuits (VF) discrete convolution
if Cal.ord>0
    vsc(Kt.id) = sum(Kt.b.*Veqf(Kt.id,:),2);                % dum(B*phi,2)
end

% (4) construct right side of matrix equation 
RIGHT=[vsa + vsc;
       is1;
       is2;
       is3];

% (5) update branch current node votlage/Vnt Ibn Is for next step
out=LEFT*RIGHT;
Vnt=out(1:Nn);                 % Vnt(1) = current step
Ibt=out(Nn+1:end);             % Ibt(1) = current step
Ibt1=Ibt(1:Nb1,1);
Ibt2=Ibt(Nb1+1:Nb1+Nb);
Ibt3=Ibt(Nb1+Nb2+1:end);

% (6) updating the hist. of discrte convilution
if Cal.ord>0
    Veqf(Kt.id,:)=Kt.a.*repmat(Ibt(Kt.id,1),1,Cal.ord)+Kt.b.*Veqf(Kt.id,:);        
end

% (7) Output parameters
Hist.Vn=Vnt;
Hist.Ib1=Ibt1;
Hist.Ib2=Ibt2;
Hist.Ib3=Ibt3;
Hist.Veqf=Veqf;

% Mreasurment data
MeasR.Ib = Ibt(Meas.Ib);                            % I measurement
MeasR.Vn = Volt_Meas_Obtain(Vnt, Meas.Vn);          % V measurement

tmp.Ib = Ibt(Meas.Pw(:,1));                         % P-I measurement
tmp.Vn = Volt_Meas_Obtain(Vnt, Meas.Pw(:,2:end));   % P-V measurement
MeasR.Pw = tmp.Ib.*tmp.Vn;

tmp.Ib = Ibt(Meas.En(:,1));                         % P-I measurement
tmp.Vn = Volt_Meas_Obtain(Vnt, Meas.En(:,2:end));   % P-V measurement
MeasR.En= tmp.Ib.*tmp.Vn;
end