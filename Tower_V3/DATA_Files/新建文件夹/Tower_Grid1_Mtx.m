function [A,R,L,C,Bran,Node,Mgnd]=Tower_Grid1_Mtx(GridP,Node)
%   Given: InsuP (mpdel), Node.com and Node.comdex for commons in Tower
%   Find:  R (vector), L=[], C=[n1 n2 value], BranR,NodeR,Meas.ins
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
%   Meas.gnd=[b1,n1,n2,val] val = "not used"
%   Meas.node=[n1]
%   Meas.bran=[b1]
Mgnd.node=[];
Mgnd.bran=[];

if isempty(GridP)
    A=[];R=[];L=[];C=[]; Meas.gnd=[]; 
    Bran.list=[]; Bran.listdex=[]; Bran.num=zeros(1,6); 
    Node.list=[]; Node.listdex=[]; Node.num=zeros(1,4); 
    return;
end

Rval=10;                   % Resistance
Lval=1e-5;                 % inductance

R0 = [1 2   Rval];        % n1 n2 value (local nodes)
L0 = [2 0   Lval];        % 0 = int node
C = [];
R = [R0(1,3);0];
L = [0;L0(1,3)];

% (3) Construct (common) nodes: Name of gnd nodes in Air 
str=[];
if GridP(1)>0
    for ik=1:GridP(1)
        str=[str;"PO_GD"+num2str(ik)+"_0"];
    end
end
if GridP(2)>0
    for ik=1:GridP(2)
        str=[str;"SA_GD"+num2str(ik)+"_0"];
    end
end
if GridP(3)>0
    for ik=1:GridP(3)
        str=[str;"TX_GD"+num2str(ik)+"_0"];
    end
end
len=length(str);

for ik=1:len
    tp=find(Node.com(:,1)==str(ik));
    if ~isempty(tp)
        Node.com(tp,2)=str(ik);
        Node.comdex(tp,2)=ik;               % node index in the gnd group
    else
        disp('Error in Creating a Model for Surf  Network - Com Node');
        return;
    end
end

% Complete node/bran info.
Node.list=[Node.com(:,2); "GD_PO1_01"];  % no additional nodes
Node.listdex=(1:2)';    % local node index
Node.num=[2 2 0 0];

    [nr n0]=size(R0);
    [nl n0]=size(L0);
Bran.list=["GD_PO1_R1";"GD_PO1_L1"];

% Build A matrix
Bran.listdex=[];
A=zeros(nr+nl,Node.num(1));
for ik=1:nr                         % Resistance branch
    ix=R0(ik,1);    
    iy=R0(ik,2);
    if ix~=0
        A(ik,ix)=-1;                % leaving
    end
    if iy~=0
        A(ik,iy)=+1;                % going
    end
    Bran.listdex=[Bran.listdex; [ik ix iy]];
end

for ik=1:nl                         % Inductance branch
    ix=L0(ik,1);    
    iy=L0(ik,2);
    if ix~=0
        A(ik+nr,ix)=-1;             % leaving
    end
    if iy~=0
        A(ik+nr,iy)=+1;             % going
    end
    Bran.listdex=[Bran.listdex; [ik+nr ix iy]];
end

% Build measurement info.
Bran.num=[2,0,0,2,0,0];
Mgnd.node=[1];
Mgnd.bran=[1];

end




