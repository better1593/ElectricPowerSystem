function [A,R,L,C,Bran,Node,Mins]=Tower_Insu_Mtx(InsuP,Node)
%   Given: InsuP (mpdel), Node.com and Node.comdex
%   Find:  R (vector), L=[], C=[n1 n2 value], BranR,NodeR,Meas.ins
%          C is applied oin P, not in Z
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

% (1) Checking the presence of insulators 
Mins.node=[];
Mins.bran=[];
if isempty(InsuP)
    A=[];R=[];L=[];C=[]; Meas.ins=[]; 
    Bran.list=[]; Bran.listdex=[]; Bran.num=zeros(1,6); 
    Bran.swh=[];  Bran.swhdex=[];
    Node.list=[]; Node.listdex=[]; Node.num=zeros(1,4); 
    return;
end

% (2) Construct R, L and C vectors, Bran.list
Cir=InsuP.Cir;
Ser=InsuP.Ser;
Nph=InsuP.Nph;                      % # of phase conductors
Pha=InsuP.Pha;                      % phase ID (string)

Rval=10^8;                          % Resistance
Cval=30e-12;                        % Capacitance

Cir0=num2str(Cir);
Ser0=num2str(Ser);
str=[];
switch Nph
    case 1
        tmp=['Cir',Cir0,'_',Pha,Ser0,'IN1'];  str=[str;string(tmp)];
        tmp=['Cir',Cir0,'_',Pha,Ser0,'IN2'];  str=[str;string(tmp)];
        len=1;
        
        L = 0;
        R = Rval;
        R0 = [1 2 Rval];             % n1 n2 value (local nodes)
        C  = [1 2 Cval];        
        Bran.list=string(['In_',Pha,'R']);    
    case 3
        tmp=['Cir',Cir0,'_A',Ser0,'_IN1'];  str=[str;string(tmp)];
        tmp=['Cir',Cir0,'_A',Ser0,'_IN2'];  str=[str;string(tmp)];
        tmp=['Cir',Cir0,'_B',Ser0,'_IN1'];  str=[str;string(tmp)];
        tmp=['Cir',Cir0,'_B',Ser0,'_IN2'];  str=[str;string(tmp)];
        tmp=['Cir',Cir0,'_C',Ser0,'_IN1'];  str=[str;string(tmp)];
        tmp=['Cir',Cir0,'_C',Ser0,'_IN2'];  str=[str;string(tmp)];
        len=3;
        
        I0=zeros(3,1);
        I1=ones(3,1);
        I2 = (1:2:5)';
        L = I0;
        R = I1*Rval;
        R0 = [I2 I2+1 Rval*I1];             % n1 n2 value (local nodes)
        C  = [I2 I2+1 Cval*I1];        
                    
        Bran.list=["IN_AR";"IN_BR";"IN_CR"];
    otherwise
        disp('Error in Creating a Model for Insulator - Phase #');
        return;
end

% (3) Construct (common) nodes 
% Ncom = convertCharsToStrings(Node.com(:,1))
for ik=1:len*2
    tp=find(Node.com(:,1)==str(ik));
    if ~isempty(tp)
        Node.com(tp,2)=str(ik);
        Node.comdex(tp,2)=ik;
    else
        disp('Error in Creating a Model for Insulator - Com Node');
    end
end
Node.list=str;
Nn=length(str);
Node.listdex=(1:Nn)';
Node.num=[Nn Nn 0 0];

% (4) Construct A matrix and Bran.listdex
A=zeros(len,Node.num(1));
Bran.listdex=[];
for ik=1:len                        % Resistance branch
    ix=R0(ik,1);    
    iy=R0(ik,2);
    A(ik,ix)=-1;                    % leaving
    A(ik,iy)=+1;                    % going
    Bran.listdex=[Bran.listdex; [ik ix iy]];
end

% (4) Construct others
Nb=len;
Bran.num=[Nb,0,0,Nb,0,0];
Bran.swh=Bran.listdex;              % [b1 n1 n2]
Bran.swhdex = [1; 1; 1];            % model of insulator flashover

Mins.node=[1; 2; 3; 4; 5; 6];           
Mins.bran=[1; 2; 3];                 
end