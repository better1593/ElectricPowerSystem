function [A,R,L,C,Bran,Node,Msuf]=Tower_Surf_Mtx(SurfP,Node)
%   Cross-boundary connection network with R=1e-6
%   Given: SurfP (mpdel), Node.com and Node.comdex
%          SurfP type: [Pole#, SA#, Tx#, other#]
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
Msuf.node=[];
Msuf.bran=[];

if isempty(SurfP)
    A=[];R=[];L=[];C=[];  
    Bran.list=[]; Bran.listdex=[]; Bran.num=zeros(1,6); 
    Node.list=[]; Node.listdex=[]; Node.num=zeros(1,4); 
    return;
end

Nb=sum(SurfP);                          % total # branches
Rval=1e-6;                              % Resistance
I1=ones(Nb,1);
I0 = (1:Nb)';
R0 = [I0 I0+Nb Rval*I1];             	% n1 n2 value (local nodes)
R = R0(1:end,3);
L = I0*0;
C = [];                                 % No capacitor

% (3) Construct (common) nodes: Name of gnd nodes in Air/Gnd   
str1=[];str2=[];str3=[];
if SurfP(1)>0
    for ik=1:SurfP(1)
        str1=[str1;"PO_GD"+num2str(ik)+"_0"];
        str2=[str2;"GD_PO"+num2str(ik)+"_0"];
        str3=[str3;"PO_GB"+num2str(ik)+"_0"];
    end
end
if SurfP(2)>0
    for ik=1:SurfP(2)
        str1=[str1;"SA_GD"+num2str(ik)+"_0"];
        str2=[str2;"GD_SA"+num2str(ik)+"_0"];
        str3=[str3;"SA_GB"+num2str(ik)+"_0"];
    end
end
if SurfP(3)>0
    for ik=1:SurfP(3)
        str1=[str1;"TX_GD"+num2str(ik)+"_0"];
        str2=[str2;"GD_TX"+num2str(ik)+"_0"];
        str3=[str3;"TX_GB"+num2str(ik)+"_0"];
    end
end
len1=length(str1);
len2=length(str2);
len3=length(str3);
str=[str1;str2];
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

Node.list=Node.com(:,2);         % air + gnd nodes
Node.listdex=(1:len)';              % local node index
Node.num=[len len1 len2 0];

    [nr n0]=size(R0);
Bran.list=str3;
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
Msuf.node=[1];
Msuf.bran=[1];

end




