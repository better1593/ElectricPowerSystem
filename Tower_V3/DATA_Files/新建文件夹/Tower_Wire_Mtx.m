function [A0,R0,L0,P0,Cw,Ht,Bran,Node,Mwir] = Tower_Wire_Mtx(WireP,Node,Bran,VFIT,GND)
%   Return (1) PEEC parameters of wires of tower (TD-Cal) inclduing VFITing
%          (2) TML parameters of the last half segment as the hybrid method
%   A,L,Ht   Matrix: bran x (node, bran, 1)
%   P        Matrix: node x node
%   C        Matrix: [:, n1 n2 Cv]
%   Cw       [node Cw_value], Cap/m of OHL/CAB for connecting a tower  
%
%   Wire_Para=[:,x1,y1,z1,x2,y2,z2,oft,r0,Ri,Li                     /1 -10
%              sig,mur,epr,mode1,mode2,bran0,node1,node2,comment]   /11-19
%   Mode 1 (int empedance) 0 (DC)   x (VFIT index in its database)
%   Mode 2 (VFIT ID) or (Hz) x (VFIT index in its database) 
%   Bran.list=[name list];  Bran.listdex=[b1 n1 n2];
%   Node.list=[name list];  Node.listdex=[n1];
%   Node.num=[total #, air #, surf #, gnd  #]
%   Node.com=[name list];   Node.comdex=[int: n1,  ext: n2]; 
%   Meas.node=[n1]
%   Meas.bran=[b1]
%
%   GND.sig, GND.epi, GND.dex (0=no, 1= perfect and (2) lossy)
%   VIFT.r0(:), *.d0(:), *.ri(:) amd *.di(:) [order=5, r0+d0*s+sum(ri/s+di)
%   VIFT.ord=3/fra;                     % VFIT order/freq. range
%   Ht.r=VFIT.rc(:,2:order);            % resiual value
%   Ht.d=VFIT.dc(:,2:order);            % pole value
%   Ht.id=[index of cond in the set of tower wires (air + gnd)];
%   Ht.ord/fra

r0=0.01;                    % radius of OhL conductors
ep0=8.854187818e-12;
ke=4*pi*ep0\1;              % coef for potential

Nbran=Bran.num(1);
Nnode=Node.num(1);
Wara = WireP(:,1:8);        % wire diemensions
Impe = WireP(:,9:10);       % wire internal impedance (Re, Le)
Mate = WireP(:,11:13);      % wire material (sig, mur, epr)
Mode = WireP(:,14:15);      % Wire modes
brannode = WireP(:,16:18);  % [b0 n1 n2]

% (1) Obtain the incidence matrix
A0=zeros(Nbran,Nnode);
for ik=1:Nbran
    A0(ik,brannode(ik,2))=-1;  % in  = -1
    A0(ik,brannode(ik,3))=+1;  % out = +1
end
%--------------------------------------------------------------------------

% (2) Obtain L and P matrices 
[L0,P0] = Wire_Mtx(Wara, Bran.num, Node.num, GND);

% (3) Updating Ri and Li with Const. impedance or VF results
leg = size(Mode,1);
odc = VFIT.odc; 
Ht.r = zeros(leg, odc);
Ht.d = zeros(leg, odc);
Ht.id = [];

% (3a) constant internal Ri and Li for condcutors
J0 = ~boolean(Mode(:,1));               % convert 0/1 -> 1/0
dR = Impe(1:leg,1).*J0;                 % int resistance (const.)
dL = Impe(1:leg,2).*J0;                 % int. inductance (const.)

% (3b) VFIT results of internal Ri and Li for condcutors
idex=1;
for ik=1:leg                            % Using VFIT data
    I1=Mode(ik,1);                      % Select model in VFIT database
    I2=Mode(ik,2);                      % Select model in VFIT database
    if I1 ~= 0
        dR(ik) =  VFIT.rc(I2,1);
        dL(ik) =  VFIT.dc(I2,1);
        Ht.r(idex,1:odc) = VFIT.rc(I2,2:end);
        Ht.d(idex,1:odc) = VFIT.dc(I2,2:end);
        Ht.id(idex,1)=ik;              % index of wires in the wire set
        idex=idex+1;
    end
end

R0 = dR;
L0 = L0 + diag(dL);

% (4) per-unit length capacitance of OHL/CAB for connection to the Tower
Cw = [];
if ~isempty(Node.com)
    leg=length(Node.comdex(:,2));       % nodes connected to OHL/CABLE
    Cw=zeros(leg,2);                    % capacitance vector
    Hw=zeros(leg,1);                    % Height of the wire 
    for ik=1:leg
        idx=Node.comdex(ik,2);          % common node id
        Cw(ik,1)=idx;
        idy=find(brannode(:,2)==idx);       % find out the node posi in a wire set
        if ~isempty(idy)
            Hw(ik,1)=Wara(idy(1),3);
        else
            idy=find(brannode(:,3)==idx);   % find out the node posi in a wire set
            Hw(ik,1)=Wara(idy(1),3);
        end
    end
    tmp=2*ke*log(2*Hw./r0);             % potential wrt the ground
    Cw(:,2)=tmp.\1;                     % capacitance
end

Mwir.node=[1;12];
Mwir.bran=[7;8];
end

