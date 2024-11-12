function [A,R,L,C,Bran,Node,Mxfm]=Tower_Txfm_Mtx(TxfmP,Node)
%   Given: SArrP (mpdel), Node.com and Node.comdex
%   Find:  R (vector), L=(vector), C=[n1 n2 value], BranR,NodeR,Meas
%
%   Bran.list=[name list];  Bran.listdex=[b1 n1 n2];
%   Bran.num=[total #, air #, gnd  #, dist #, lump #, special #]
%   Bran.swh=[b1 n1 n2]   Bran.swhdex = [1 (Ys) = 0 (No)]
%   Bran.nle=[b1 n1 n2]   Bran.nledex = [1 (Ys) = 0 (No)]
%   Node.list=[name list];  Node.listdex=[n1];
%   Node.num=[total #, air #, surf #, gnd  #]
%   Node.com=[name list];   Node.comdex=[int: n1,  ext: n2]; 
%   Meas.ins=[b1,n1,n2,val] val = status of flashover (0 = No, 1 = Yes)
%   Meas.spd=[b1,n1,n2,val] val = nle R, 0=not updated,1=to be updated 
%   Meas.xfm=[b1,n1,n2,val] val = "not used"
%   Meas.gnd+=[b1,n1,n2,val] val = "not used"
%   Meas.node=[n1]
%   Meas.bran=[b1]
Mxfm.node=[];
Mxfm.bran=[];
if isempty(TxfmP)
    A=[];R=[];L=[];C=[]; Meas.xfm=[]; 
    Bran.list=[]; Bran.listdex=[]; Bran.num=zeros(1,6); 
    Node.list=[]; Node.listdex=[]; Node.num=zeros(1,4); 
    return;
end

Cir=TxfmP.Cir;
Ser=TxfmP.Ser;
Nph=TxfmP.Nph;                      % # of phase conductors
Pha=TxfmP.Pha;                      % phase ID (string)

R1=0.8231;
R2=0.8235;
R3=0.8236;
R4=0.001053;
R5=0.001053;
R6=0.001053;
R7=1;
L1=39.69;
L2=58.34;
L3=38.19;
L4=0.257;
L5=0.351;
L6=0.366;
C1=3.069e-11;
C2=2.088e-11;
C3=3.190e-11;
C4=3.57e-9;
C5=2.61e-9;
C6=2.508e-9;
C7=7.008e-8;
C8=7.008e-8;
C9=7.008e-8;
C10=7.83e-9;
C11=7.83e-9;
C12=7.83e-9;

R0 = [1 8 R4
      2 9 R5
      3 10 R6
      4 11 R1
      5 12 R2
      6 13 R3];
 L0 = [8 2 L4
       9 3 L5
       10 1 L6
       11 7 L1
       12 7 L2
       7 13 L3]; 
 C0 = [1 2 C4
       2 3 C5
       1 3 C6
       1 4 C7
       2 5 C8
       3 6 C9
       1 7 C10
       2 7 C11
       3 7 C12
       4 7 C1
       5 7 C2
       6 7 C3]; 
  
C = C0;
R = [R0(1:6,3);ones(6,1)*0];
L = [ones(6,1)*0;L0(1:6,3)];

% (3) Construct (common) nodes 
Cir0=num2str(Cir);
Ser0=num2str(Ser);
str=[];
tmp=['Cir',Cir0,'_A',Ser0,'_TX1'];  str=[str;string(tmp)];
tmp=['Cir',Cir0,'_B',Ser0,'_TX1'];  str=[str;string(tmp)];
tmp=['Cir',Cir0,'_C',Ser0,'_TX1'];  str=[str;string(tmp)];
tmp=['Cir',Cir0,'_a',Ser0,'_TX1'];  str=[str;string(tmp)];
tmp=['Cir',Cir0,'_b',Ser0,'_TX1'];  str=[str;string(tmp)];
tmp=['Cir',Cir0,'_c',Ser0,'_TX1'];  str=[str;string(tmp)];
tmp=['Cir',Cir0,'_N',Ser0,'_TX1'];  str=[str;string(tmp)];

for ik=1:7
    tp=find(Node.com(:,1)==str(ik));
    if ~isempty(tp)
        Node.com(tp,2)=str(ik);
        Node.comdex(tp,2)=ik;
    else
        disp('Error in Creating a Model for Transformer - Com Node');
        return;
    end
end

tmp=Ser0+"_TX_m";  
loc=["A"+tmp;"B"+tmp;"C"+tmp;"a"+tmp;"b"+tmp;"c"+tmp];
Node.list=[Node.com(:,2);loc];
len=length(Node.list);
Node.listdex=(1:len)';        % local node index
Node.num=[len len 0 0];

    [nr n0]=size(R0);
    [nl n0]=size(L0);
tmp=Ser0+"_TX_";  
Bran.list=["A"+tmp+"R"; "B"+tmp+"R"; "C"+tmp+"R";
           "a"+tmp+"R"; "b"+tmp+"R"; "c"+tmp+"R";
           "A"+tmp+"L"; "B"+tmp+"L"; "C"+tmp+"L";
           "a"+tmp+"L"; "b"+tmp+"L"; "c"+tmp+"L";];

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

Bran.num=[12,0,0,12,0,0];

Mxfm.node =[4 ; 11; 5; 12; 6; 13];             
Mxfm.bran =[1; 2; 3];             
end
