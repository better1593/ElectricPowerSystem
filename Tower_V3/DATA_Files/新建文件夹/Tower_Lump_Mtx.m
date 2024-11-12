function [A,R,L,C,Bran,Node,Mlup]=Tower_Lump_Mtx(LumpP,Node)
%   Given: LumpP (mpdel), Node.com and Node.comdex
%   Find:  R (vector), L=(vector), dim(R)=dim(L), BranR,NodeR,Meas.ins
%          C=[n1 n2 value], C is applied in P, not in Z
%
%   Bran.list=[name list];  Bran.listdex=[b1 n1 n2];
%   Bran.num=[total #, air #, gnd  #, dist #, lump #, swh #]
%   Bran.swh=[b1 n1 n2]   Bran.swhdex = [1 (Ys) = 0 (No)]
%   Bran.nle=[b1 n1 n2]   Bran.nledex = [1 (Ys) = 0 (No)]
%   Node.list=[name list];  Node.listdex=[n1];
%   Node.num=[total #, air #, surf #, gnd  #]
%   Node.com=[name list];   Node.comdex=[int: n1,  ext: n2]; 
%   Meas.ins=[b1,n1,n2,val] val = status of flashover (0 = No, 1 = Yes)
%   Meas.spd=[b1,n1,n2,val] val = nle R, 0=not updated,1=to be updated 
%   Meas.xfm=[b1,n1,n2,val] val = "not used"
%   Meas.node=[n1]
%   Meas.bran=[b1]

Mlup.node=[];
Mlup.bran=[];
if isempty(LumpP)
    A=[];R=[];L=[];C=[]; Meas.ins=[]; 
    Bran.list=[]; Bran.listdex=[]; Bran.num=zeros(1,6); 
    Bran.swh=[];  Bran.swhdex=[];
    Node.list=[]; Node.listdex=[]; Node.num=zeros(1,4); 
    return;
end

Rval=1e-6;                          % Resistance
I1=ones(2,1);
I0 = (1:2)';
R = [I0 I0+2 Rval*I1];              % n1 n2 value (local nodes)
L = I0*0;
C = [];                             % No capacitor

Node.list=Node.comdex;              % air + gnd nodes
Node.listdex=[1;2;3;4];             % local node index
Node.num=[4 0 2 2];
Node.comdex =[Node.comdex Node.listdex];    % [ext int]

    [nr n0]=size(R);
Bran.list={'PO_GB';'TX_GB'};
Bran.listdex=[];
A=zeros(nr,Node.num(1));
for ik=1:nr                         % Resistance branch
    ix=R0(ik,1);    
    iy=R0(ik,2);
    A(ik,ix)=-1;                    % leaving
    A(ik,iy)=+1;                    % going
    Bran.listdex=[Bran.listdex; [ik ix iy]];
end

Bran.num=[nr,0,0,0,0,nr];            % cross-boundary bran
Mlup.node=[];
Mlup.bran=[];
end




