function [L0, P0] = Wire_Model_Para2(WireP, Bnum, Nnum, GND)
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
%   Bran.list=[name list];  Bran.listdex=[b1 n1 n2];
%   Node.list=[name list];  Node.listdex=[n1];
%   Node.num=[total #, air #, gnd  #, air_off #]
%   Node.com=[name list];   Node.comdex=[int: n1,  ext: n2]; 
%   GND.sig, GND.epi, GND.dex (0=no, 1= perfect and (2) lossy)

% Bnum(3)=0 --> there is no gnd wire

% (0) Intial constants
ep0=8.854187818e-12;
mu0=4*pi*1e-7;
ke=4*pi*ep0\1;                      % coef for potential
km=mu0/(4*pi);                      % coef for inductance

Nb=Bnum(2)+Bnum(3);                 % all bran # excluding surf bran
Nn=Nnum(1);                         % all node #
Nba=Bnum(2);                        % air bran #
% Nbg=Bnum(3);                        % gnd bran #
Nna=Nnum(2);                        % air node #
Nng=Nnum(3);                        % gnd node #
rb1=1:Nba;                          % air bran
rb2=1+Nba:Nb;                       % gnd bran
rn1=1:Nna;                          % air node
rn2=1+Nna:Nn;                       % gnd node

% (1) Obtaining L and P matrices without considering the ground effect
% (1a) L and P matrices for all: aa, ag, ga, gg
ps1=WireP(:,1:3);                   % starting points
ps2=WireP(:,4:6);                   % ending points
rs =WireP(:,8);                     % radius
At =WireP(:,17:18);                 % leaving and entering nodes

ds = ps2-ps1;
ls = sqrt(sum(ds.*ds,2));           % segment length
cosa = ds(:,1)./ls;
cosb = ds(:,2)./ls;
cosc = ds(:,3)./ls;

% for gnd and air segments
[Lout, Pout] = Wire_Model_Para1(ps1, ps2, ls, rs, ps1, ps2, ls, rs, At, Nn); 
%--------------------------------------------------------------------------
 
% (2) Constructing L and P by considering the image effect
% no ground (0), perfect ground (1), lossy ground model (2)
% (2a) without ground
L0=Lout.*(cosa*cosa'+cosb*cosb'+cosc*cosc');% free-space inductance
P0=Pout;                                    % free-space potnetial

% (2b)with ground
% L and P matrices for aai,ggi
if GND.gnd~=0
    pf1 = ps1; pf1(:,3) = -pf1(:,3);    % image for air segments
    pf2 = ps2; pf2(:,3) = -pf2(:,3);    % image for gnd segments

[Lai,Pai]=Wire_Model_Para1(ps1(rb1,:),ps2(rb1,:),ls(rb1,1),rs(rb1,1), ...
pf1(rb1,:),pf2(rb1,:),ls(rb1,1),rs(rb1,1),At(rb1,:),Nna);  % image of air 

[Lgi,Pgi]=Wire_Model_Para1(ps1(rb2,:),ps2(rb2,:),ls(rb2,1),rs(rb2,1), ...
pf1(rb2,:),pf2(rb2,:),ls(rb2,1),rs(rb2,1),At(rb2,:)-Nna,Nng);% image of gnd 
end

% (2bi) perfect ground
if GND.gnd == 1                          
    L0 = L0 - Lai.*(cosa*cosa'+cosb*cosb'-cosc*cosc');
    P0 = P0 - Pai;
end

% (2bii)lossy gnd model
if GND.gnd == 2  
    Lag=Lout(rb1,rb2);   Lga=Lout(rb2,rb1); % without x cos(a)
    Pag=Pout(rn1,rn2);   Pga=Pout(rn2,rn1);  

    % image effect
% (i) f./s. wire in air 
%     cosaA=cosa(rb1);                     % air wire: direction numbers
%     cosbA=cosb(rb1);
    coscA=cosc(rb1);
    L0(rb1,rb1) = L0(rb1,rb1) + Lai.*(coscA*coscA'); % vertical contri     
    P0(rn1,rn1) = P0(rn1,rn1) - Pai;

    if Bnum(3)~=0
%         cosaG=cosa(rb2);                 % gnd wire: direction numbers
%         cosbG=cosb(rb2);
        coscG=cosc(rb2);
        
% (ii) f. wire in gnd s. wire in air
        L0(rb2,rb1) = L0(rb2,rb1) + Lga.*(coscG*coscA'); % vertical 
        P0(rn2,rn1) = P0(rn2,rn1) - Pga;

% (iii) f./s. wire in gnd 
        L0(rb2,rb2) = L0(rb2,rb2) - Lgi.*(coscG*coscG'); % vertical 
        P0(rn2,rn2) = P0(rn2,rn2) + Pgi;

% (iv) f. wire in air s. wire in gnd
        L0(rb1,rb2) = L0(rb1,rb2) - Lag.*(coscA*coscG'); % vertical contribution (-imag)             
        P0(rn1,rn2) = P0(rn1,rn2) + Pag;
    end
end
L0=km*L0;
P0=ke*P0;      
end

