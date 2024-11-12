function [CK_Para] = Wire_Model_Para3(WireP,Node,Bran,VFmod,VFIT,GND)
%   Return PEEC parameters (TD-Cal) of wires and Circuit Model Parameters
%   Para1: L and P of wires in free space (T-model, P-cell division)
%   Para2: L and P of wires considering the effect of ground
%   Para3: L and P of wires considering the VFIT
%          TML parameters of the last half segment as the hybrid method
%   A       Matrix: bran x node, bran
%   R,L     Matrix: bran x bran
%   Ht      Matrix: bran x order
%   P       Matrix: node x node
%   Cw      [node Cw_value], Cap/m of OHL/CAB for connecting a tower  
%
%   Wire_Para=[:,x1,y1,z1,x2,y2,z2,oft,r0,Ri,Li                     /1 -10
%              sig,mur,epr,mode1,mode2,bran0,node1,node2,comment]   /11-19
%   Mode 1 (int empedance) 0 (DC)   x (VFIT index in its database)
%   Mode 2 (VFIT ID) or (Hz) x (VFIT index in its database) 
%   Bran.list=[name list];  Bran.listdex=[b1 n1 n2];
%   Node.list=[name list];  Node.listdex=[n1];
%   Node.num=[total #, air #, gnd  #, air_off #]
%   Node.com=[name list];   Node.comdex=[int: n1,  ext: n2]; 
%   Meas.node=[n1]
%   Meas.bran=[b1]
%
%   GND.sig, GND.epi, GND.dex (0=no, 1= perfect and (2) lossy)
%   VIFT.r0(:), *.d0(:), *.ri(:) amd *.di(:) [order=5, r0+d0*s+sum(ri/s+di)
%   VIFT.ord=3/fra;                     % VFIT order/freq. range
%   Ht.r=VFIT.rc(:,2:order+1);          % resiual value
%   Ht.d=VFIT.dc(:,2:order+1);          % pole value
%   Ht.id=[index of cond in the set of tower wires (air + gnd)];
%   Ht.ord/fra
%   VFmod = [VF_cond, VF_gnd] 1= yes, 0 = no

% (0) Initial constants
ep0=8.854187818e-12;
ke=4*pi*ep0\1;              % coef for potential

Nbran=Bran.num(2)+Bran.num(3);  % total # of brans excluding surf bran 
Nnode=Node.num(1);          % total # of nodes
Ncom =size(Node.com,1);     % total # of common nodes
imp = WireP(:,9:10);        % wire internal impedance (Re, Le)
mod = WireP(:,14:15);       % Wire modes [cond_VFid, Gnd_VFid]
nodebran = WireP(:,16:18);  % [b0 n1 n2]

% (1) Obtaining the incidence matrix A
A0=zeros(Nbran,Nnode);
for ik=1:Nbran
    A0 = AssignAValue(A0,ik,nodebran(ik,2),-1);  % in  = -1
    A0 = AssignAValue(A0,ik,nodebran(ik,3),+1);  % out = +1
end
%--------------------------------------------------------------------------

% (2) Obtaining L and P matrices of wires considering ground effect
[L0,P0] = Wire_Model_Para2(WireP, Bran.num, Node.num, GND);

% (3) Updating Ri and Li with Const. impedance or VF results
odc = VFIT.odc; 
Ht.r = zeros(Nbran, odc);
Ht.d = zeros(Nbran, odc);
Ht.id = [];

if VFmod(1)== 0                        
% (3a) Constant internal Ri and Li for condcutors
    dR = imp(1:Nbran,1);                % int resistance (const.)
    dL = imp(1:Nbran,2);                % int. inductance (const.)
else
% (3b) VFIT results of internal Ri and Li for condcutors
    for ik = 1:Nbran
        VFid = WireP(ik,14);                  % id of C-VF data in a table
        dR(ik) =  VFIT.rc(VFid,1);            % getting dc res
        dL(ik) =  VFIT.dc(VFid,1);            % getting dc ind
        Ht.r(ik,1:odc) = VFIT.rc(VFid,2:end); % getting residual values
        Ht.d(ik,1:odc) = VFIT.dc(VFid,2:end); % getting pole values 
        Ht.id(ik,1)=ik;                   % index of wires in the wire set
    end
end
R0 = diag(dR);
L0 = L0+diag(dL);

% (4) per-unit length capacitance of OHL/CAB for connection to the Tower
Cw = [];
if Ncom~=0
    Hw=zeros(Ncom,1);                   % Height of the wire 
    Cw=zeros(Ncom,2);                   % capacitance vector
    Cw(:,1)=Node.comdex(:,2);           % id of com_node
    for ik=1:Ncom
        idx=find(nodebran(:,2)==Cw(ik,1));   % find node posi in a wire set
        idy=find(nodebran(:,3)==Cw(ik,1));   % find node posi in a wire set
        if ~isempty(idx)
            Hw(ik,1)=WireP(idx(1),3);
            rw(ik,1)=WireP(idx(1),8);
        else
            Hw(ik,1)=WireP(idy(1),6);
            rw(ik,1)=WireP(idy(1),8);
        end
    end
    tmp0=2*ke*log(2*Hw./rw);             % potential wrt the ground
    Cw(:,2)=tmp0.\1;                     % capacitance
end

CK_Para.A = A0;
CK_Para.R = R0;
CK_Para.L = L0;
CK_Para.C = zeros(Nnode,Nnode);
CK_Para.G = zeros(Nnode,Nnode);
CK_Para.P = P0;
CK_Para.Cw = Cw;
CK_Para.Ht = Ht;

CK_Para.Vs = [];
CK_Para.Is = [];
CK_Para.Nle = [];
CK_Para.Swh = [];

end

