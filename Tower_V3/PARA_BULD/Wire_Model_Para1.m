function [L,P] = Wire_Model_Para1(ps1, ps2, ls, rs, pf1, pf2, lf, rf, At, Nnode)
%   Return PEEC parameters (TD-Cal) of wires and Circuit Model Parameters
%   Para1: L and P of wires in free space (T-model, P-cell division)
%   Para2: L and P of wires considering the effect of ground
%   Para3: L and P of wires considering the VFIT
%          TML parameters of the last half segment as the hybrid method
%   R,L     Matrix: bran x bran
%   P       Matrix: node x node
%
%   Wire_Para=[:,x1,y1,z1,x2,y2,z2,oft,r0,Ri,Li                     /1 -10
%              sig,mur,epr,mode1,mode2,bran0,node1,node2,comment]   /11-19

% (0) Initialization
ELIM=1e-3;
if isempty(ps1)
    L=[]; P=[];
    return;
end

PROD_MOD=2;                         % matrix product
COEF_MOD=1;                         % double integral

% (1) Inductance calculation of branches
L = INT_SLAN_2D(ps1, ps2, rs, pf1, pf2, rf, PROD_MOD, COEF_MOD);

% (2) Generating coordinates of node segments (half of bran segments)
ps0=0.5*(ps1+ps2);
pf0=0.5*(pf1+pf2);

ofs=0;
for ik=1:Nnode                      % size of node segments for source
    pt1=find(At(:,1)==ik);          % pos of ith node in branch
    pt2=find(At(:,2)==ik);        	% pos of ith node in branch
    d1=length(pt1);                 % total # of common nodes for ith node
    d2=length(pt2);                 % total # of common nodes for ith node
 
% (2a) 1st half segment (field segment = source segment)
    if d1~=0
        nrs(ofs+(1:d1),1)=rs(pt1);          % radius (n1)-source
        nls(ofs+(1:d1),1)=ls(pt1)/2;        % length (n1)
        nps1(ofs+(1:d1),1:3)=ps1(pt1,1:3);  % str pts
        nps2(ofs+(1:d1),1:3)=ps0(pt1,1:3);  % end pts

        nrf(ofs+(1:d1),1)=rf(pt1);          % radius(n1)-field
        nlf(ofs+(1:d1),1)=lf(pt1)/2;        % length(n1)
        npf1(ofs+(1:d1),1:3)=pf1(pt1,1:3);  % str pts
        npf2(ofs+(1:d1),1:3)=pf0(pt1,1:3);  % endpts
    end
    ofs=ofs+d1;
    
% (2b) 2nd half segment
    if d2~=0
        nrs(ofs+(1:d2),1)=rs(pt2);          % radius (n2)
        nls(ofs+(1:d2),1)=ls(pt2)/2;        % length (n2)   
        nps1(ofs+(1:d2),1:3)=ps0(pt2,1:3);  % str pts
        nps2(ofs+(1:d2),1:3)=ps2(pt2,1:3);  % endpts

        nrf(ofs+(1:d2),1)=rf(pt2);          % radius (n2)
        nlf(ofs+(1:d2),1)=lf(pt2)/2;        % length (n2)   
        npf1(ofs+(1:d2),1:3)=pf0(pt2,1:3);  % str pts
        npf2(ofs+(1:d2),1:3)=pf2(pt2,1:3);  % endpts
    end
    ofs=ofs+d2;    
    ncom(ik,1)=d1+d2;                       % # of segments for each node    
end    

% (4) Calculating potnetial matrix
PROD_MOD=2;                         % matrix product
COEF_MOD=1;                         % integration only

int = INT_SLAN_2D(nps1,nps2,nrs,npf1,npf2,nrf,PROD_MOD,COEF_MOD);

% (5) merging common nodes
P=int;

ofs=0;
Idel=[];
for ik=1:Nnode
    nc=ncom(ik);                    % # of common nodes for ith node
    if nc>=1
    	Idel=[Idel ofs+(2:nc)];     % collecting deleted row/col
    end
    nlns(ik,1)=sum(nls(ofs+(1:nc)));% total length of ith node (source)
    nlnf(ik,1)=sum(nlf(ofs+(1:nc)));% total length of ith node (field)
    
    tmp=P(ofs+(1:nc),:);          % collecting rows of common nodes
    P(ofs+1,:)=sum(tmp,1);        % sum of all rows
    tmp=P(:,ofs+(1:nc));          % collecting cols of common nodes
    P(:,ofs+1)=sum(tmp,2);        % sum of all cols
    ofs=ofs+nc;           
end  
   
P(Idel,:)=[];
P(:,Idel)=[];
P=P./(nlns*nlnf');  
end