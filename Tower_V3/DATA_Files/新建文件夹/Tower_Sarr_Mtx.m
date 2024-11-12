function [A,R,L,C,Bran,Node,Mspd]=Tower_Sarr_Mtx(SarrP,Node)
%   Three_phase
%   Given: SArrP (mpdel), Node.com and Node.comdex
%   Find:  R (vector), L=(vector), C=[n1 n2 value], BranR,NodeR,Meas.spd
%
%   Bran.list=[name list];  Bran.listdex=[b1 n1 n2];
%   Bran.num = [total #, wair #, wgnd  #, lair #, lgnd #, Speical #] % 1-6
%   Bran.swh=[b1 n1 n2]   Bran.swhdex = [>0 (model ID) = 0 (No)]
%   Bran.nle=[b1 n1 n2]   Bran.nledex = [>0 (model ID) = 0 (No)]
%   Node.list=[name list];  Node.listdex=[n1];
%   Node.num = [total #, air #, gnd  #, gnd node offest=wire air node]% 1-4  
%   Node.com=[name list];   Node.comdex=[int: n1,  ext: n2]; 
%   Meas.ins=[b1,n1,n2,val] val = status of flashover (0 = No, 1 = Yes)
%   Meas.spd=[b1,n1,n2,val] val = nle R, 0=not updated,1=to be updated 
%   Meas.xfm=[b1,n1,n2,val] val = "not used"
Mspd.node=[];
Mspd.bran=[];

if isempty(SarrP)
    A=[];R=[];L=[];C=[]; Meas.spd=[]; 
    Bran.list=[]; Bran.listdex=[]; Bran.num=zeros(1,6); 
    Bran.nle=[];  Bran.nledex=[]; Bran.nlemea=[];
    Node.list=[]; Node.listdex=[]; Node.num=zeros(1,4); 
    return;
end

Cir=SarrP.Cir;
Ser=SarrP.Ser;
Nph=SarrP.Nph;                      % # of phase conductors
Pha=SarrP.Pha;                      % phase ID (string)

N=3;
R_dtc=1e-6;  % R for current detection,dtc=detect current
R0 = [1 7 R_dtc
      7 8 19.5
      8 2 10
      8 9 30
      9 2 10
      3 7+N R_dtc
      7+N 8+N 19.5
      8+N 4 10
      8+N 9+N 30
      9+N 4 10
      5 7+2*N R_dtc
      7+2*N 8+2*N 19.5
      8+2*N 6 10
      8+2*N 9+2*N 30
      9+2*N 6 10];
L0 = [7 8 600e-9
      8 9 4.5e-6
      7+N 8+N 600e-9
      8+N 9+N 4.5e-6
      7+2*N 8+2*N 600e-9
      8+2*N 9+2*N 4.5e-6];
C0 = [8  2 333.3e-12
      8+N  4 333.3e-12
      8+2*N  6 333.3e-12]; 

C = C0;
R = [R0(1:15,3);ones(6,1)*0];
L = [ones(15,1)*0;L0(1:6,3)];

% (3) Construct (common) nodes 
Cir0=num2str(Cir);
Ser0=num2str(Ser);
str=[];
tmp=['Cir',Cir0,'_A',Ser0,'_SA1'];  str=[str;string(tmp)];
tmp=['Cir',Cir0,'_A',Ser0,'_SA2'];  str=[str;string(tmp)];
tmp=['Cir',Cir0,'_B',Ser0,'_SA1'];  str=[str;string(tmp)];
tmp=['Cir',Cir0,'_B',Ser0,'_SA2'];  str=[str;string(tmp)];
tmp=['Cir',Cir0,'_C',Ser0,'_SA1'];  str=[str;string(tmp)];
tmp=['Cir',Cir0,'_C',Ser0,'_SA2'];  str=[str;string(tmp)];

for ik=1:6
    tp=find(Node.com(:,1)==str(ik));
    if ~isempty(tp)
        Node.com(tp,2)=str(ik);
        Node.comdex(tp,2)=ik;
    else
        disp('Error in Creating a Model for Surge Arrester - Com Node');
        return;
    end
end

pha=['A',num2str(Ser),'_SA']; % A1_SA
phb=['B',num2str(Ser),'_SA'];
phc=['C',num2str(Ser),'_SA'];

loc=[pha+"03"; pha+"04"; pha+"05";
          phb+"03"; phb+"04"; phb+"05";
          phc+"03"; phc+"04"; phc+"05";];
Node.list=[Node.com(:,2);loc];
len=length(Node.list);
Node.listdex=(1:len)';        % local node index   Node.listdex=(1:42)'; 
Node.num=[len len 0 0];       %  Node.num=[42 42 0 0];

    [nr n0]=size(R0);
    [nl n0]=size(L0);
Bran.list=[pha+"_R1";pha+"_R2";pha+"_R3";pha+"_R4";pha+"_R5";
           phb+"_R1";phb+"_R2";phb+"_R3";phb+"_R4";phb+"_R5";                 
           phc+"_R1";phc+"_R2";phc+"_R3";pha+"_R4";phc+"_R5";         
           pha+"_L1";pha+"_L2";phb+"_L1";phb+"_L2";phc+"_L1";phc+"_L2"];

Bran.listdex=[];
A=zeros(nr+nl,Node.num(1));
for ik=1:nr                         % Resistance branch
    ix=R0(ik,1);    
    iy=R0(ik,2);
    A(ik,ix)=-1;                    % leaving
    A(ik,iy)=+1;                    % going
    Bran.listdex=[Bran.listdex; [ik ix iy]];
end

for ik=1:nl                         % Inductance branch
    ix=L0(ik,1);    
    iy=L0(ik,2);
    A(ik+nr,ix)=-1;                 % leaving
    A(ik+nr,iy)=+1;                 % going
    Bran.listdex=[Bran.listdex; [ik+nr ix iy]];
end

Bran.num=[21, 0, 0, 21, 0, 0];
Bran.nle=[3 8 2                     % [b1 n1 n2]
          5 9 2
          8 11 4
          10 12 4
          13 14 6
          15 15 6];       % considering nonlinear R 
                  
Bran.nledex = [0; 0; 0; 0; 0; 0;];   % index in the nle table (0=fixed R)  

Bran.nlemea=[1  1 2;                % b1 for current, n1-n2 fo voltage
             6  3 4;
             11 5 6];

Mspd.node=[1; 7; 3; 10; 5; 13];        %val = nle R, 0=not updated,1=to be updated 
Mspd.bran=[1; 6; 11];       
end
