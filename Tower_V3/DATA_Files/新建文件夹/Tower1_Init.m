function Tower0=Tower1_Init(GLB)

% Define TOW/OHL/CABLE Type parameters
OvhdP = 1;                          % OHL type:
CablP = 1;                          % CAB type:
InsuP.Cir = 3001;                 % INS type: Circuit ID (10kV)
InsuP.Ser = 1;                   % INS type: Ser No. (Eq. ID) 
InsuP.Nph = 3;                   % INS type: # of phases 
InsuP.Pha = [];                  % INS type: phase ID
SarrP=[];
% SarrP.Cir = 3001;                 % SAR type: Circuit ID (10kV)
% SarrP.Ser = 1;                   % SAR type: Ser No. (Eq. ID) 
% SarrP.Nph = 3;                   % SAR type: # of phases 
% SarrP.Pha = [];                  % INS type: phase ID
TxfmP=[];
% TxfmP.Cir = 3001;                 % SAR type: Circuit ID (10kV)
% TxfmP.Ser = 1;                   % XFM type: Ser No. (Eq. ID) 
% TxfmP.Nph = 3;                   % XFM type: # of phases 
% TxfmP.Pha = [];                  % INS type: phase ID
SurfP = [1 0 0 0];             % SUR type: [Pole#, SA#, Tx#, other#]
GridP = [];                    % GND type: lump ground IMP. (1=yes)
% SurfP = [];                      % SUR type: [Pole#, SA#, Tx#, other#]
% GridP = [1 0 0 0];               % GND type: [Pole#, SA#, Tx#, other#]
LumpP = [];                      % lump element in the future

if (~isempty(SurfP))&(~isempty(GridP))
    disp('Problem of Ground Setting');
    return;
end

%----------------(2) Tower parameters -------------------------------
% (2a) Node/Bran in Tower in the air
Ndata  =[{1  0 +0.0 11} {'Cir1001_S1_T0001'}; %POLE(Common node: Tower_Pole)    
         {2  0 -0.4 10} {'Cir3001_A1_T0001'}; %H1_PA1 (Common node: 10kV_A1)
         {3  0 +0.1 10} {'Cir3001_B1_T0001'}; %H1_PB1 (Common node: 10kV_B1)
         {4  0 +0.6 10} {'Cir3001_C1_T0001'}; %H1_PB1 (Common node: 10kV_C1)
         {5  0 -0.4 9.8} {'Cir3001_A1_IN1'}; %IN_PA1 (Common node: IN_A top)
         {6  0 +0.1 9.8} {'Cir3001_B1_IN1'}; %IN_PB1 (Common node: IN_B top)
         {7  0 +0.6 9.8} {'Cir3001_C1_IN1'}; %IN_PB1 (Common node: IN_C top)
         {8  0 -0.4 9.8} {'Cir3001_A1_IN2'}; %IN_PA2 (Common node: IN_A bot)
         {9  0 +0.1 9.8} {'Cir3001_B1_IN2'}; %IN_PA2 (Common node: IN_B bot)
         {10 0 +0.6 9.8} {'Cir3001_C1_IN2'}; %IN_PA2 (Common node: IN_C bot)
         {11 0 +0.0 9.8} {'PO_P01'};     % Pole middle point  
         {12 0 +0.0 0}   {'PO_GD1_0'};       % Pole ground surf
         {13 0 +0.1 1.0} {'Cir5001_A1_T0001'}; %Cab_PA1 (Common node: Cable A)
         {14 0 +0.1 1.0} {'Cir5001_B1_T0001'}; %Cab_Pb1 (Common node: Cable B)
         {15 0 +0.1 1.0} {'Cir5001_C1_T0001'}; %Cab_PC2 (Common node: Cable C)
         {16 0 +0.1 1.0} {'Cir5001_M1_T0001'}];%Cab_PM2 (Common node: Cable M) 

list=string(Ndata(:,5));
tmp=Ndata(:,1:4); Npos=cell2mat(tmp);
Wire =[{2   5 Npos( 2,2:4) Npos( 5,2:4)} {'IN_A'};      
       {3   6 Npos( 3,2:4) Npos( 6,2:4)} {'IN_B'};       
       {4   7 Npos( 4,2:4) Npos( 7,2:4)} {'IN_C'};      
       {8  11 Npos( 8,2:4) Npos(11,2:4)} {'HD_AP'};      
       {9  11 Npos( 9,2:4) Npos(11,2:4)} {'HD_BP'};       
       {10  9 Npos(10,2:4) Npos( 9,2:4)} {'HD_CB'};      
       {1  11 Npos( 1,2:4) Npos(11,2:4)} {'PO_1'};
       {11 12 Npos(11,2:4) Npos(12,2:4)} {'PO_2'};
       {2  13 Npos( 2,2:4) Npos(13,2:4)} {'Cab_A'};       
       {3  14 Npos( 3,2:4) Npos(14,2:4)} {'Cab_B'};      
       {4  15 Npos( 4,2:4) Npos(15,2:4)} {'Cab_C'};
       {11 16 Npos(11,2:4) Npos(16,2:4)} {'Cab_M'}];
   
NodeW.list=list; 
NodeW.listdex = Npos(1:end,1);
nlen=length(list);
NodeW.num = [nlen nlen 0 0]; 
NodeW.com=NodeW.list([1:4 13:16]);       % connected to OHL/cable (SW+3XHV)
NodeW.comdex=NodeW.listdex([1:4 13:16],1);
[blen, n0]=size(Wire);
BranW.num = [blen blen 0 0 0 0];

NodeR.com=list(5:10);               % insulator
NodeR.comdex=Npos(5:10,1);

NodeS.com=[];              % spd
NodeS.comdex=[];

NodeX.com=[];              % transformer
NodeX.comdex=[];

NodeF.com=[];                       % lump elements for wire gnd onnection
NodeF.comdex=[];

NodeG.com=list(12:12);              % lump gnd elements
NodeG.comdex=Npos(12:12,1);

NodeM.com=[];                       % other lump elements
NodeM.comdex=[];

% (2b) Node and bran under the ground 
if isempty(GridP)
    Ndatag=[{17 0 +0.0 0}  {'GD_PO1_0'};    % Pole gnd electrode (middel)
            {18 0 +0.0 -1} {'GD_PO1_1'};    % Pole gnd electrode (middel)
            {19 0 -2.0 -1} {'GD_PO1_2'};    % Pole gnd electrode (left)
            {20 0 +2.0 -1} {'GD_PO1_3'}];   % Pole gnd electrode (right)
    Ndata=[Ndata; Ndatag];
    tmp=Ndata(:,1:4); Npos=cell2mat(tmp);
    list=string(Ndata(:,5));
    NodeW.list=list;
    NodeW.listdex = Npos(1:end,1);
    n0len=length(list);
    NodeW.num = [n0len nlen n0len-nlen nlen]; % Nodew.num(4)=offset for gnd node dex
% Nodes 12 and 17 are surf. nodes given in the air/gnd group
    Wireg= [{17 18 Npos(17,2:4) Npos(18,2:4)} {'PO_G1'};
            {18 19 Npos(18,2:4) Npos(19,2:4)} {'PO_G2'};
            {18 20 Npos(19,2:4) Npos(20,2:4)} {'PO_G3'}];
    Wire=[Wire; Wireg];
    [b0len, n0]=size(Wire);
    BranW.num = [b0len blen b0len-blen 0 0 0];   
    
    NodeF.com=list([12 17]);          % lump elements for wire gnd onnection
    NodeF.comdex=Npos([12 17],1);     % half of air nodes and half gnd nodes

    NodeG.com=[];                   % lump gnd elements
    NodeG.comdex=[];
end
Wire0=Wire;
Wire=Wire0(:,1:4);
Wire=cell2mat(Wire);

%--------------------------------------------------------------------------

% (2c) Update other parameters
x1=Wire(:,3);y1=Wire(:,4);z1=Wire(:,5);
x2=Wire(:,6);y2=Wire(:,7);z2=Wire(:,8);
dx=x2-x1; dy=y2-y1; dz=z2-z1;
r0=0.002;
l0=sqrt(dx.*dx+dy.*dy+dz.*dz);
cosa=dx./l0;
cosb=dy./l0;
cosc=dz./l0;
Ri = 1e-3;                          % int. resistance
Li = 1e-7;                          % int. inductance
model1=0;                           % con impedance (1=yes)
model2=0;                           % gns impedance (1=yes)
node=Wire(:,1:2);
Wire(:,1:2)=[];
sig=5.7e8;
mur=1;

[Nbw n0]=size(Wire); 
tmp=(1:Nbw)';
unit=ones(Nbw,1);
WireP=[Wire cosa cosb cosc unit*r0 l0 unit*Ri unit*Li ...
    unit*model1 unit*model2 node unit*sig unit*mur];
BranW.list=string(Wire0(:,5));
BranW.listdex=[tmp node];  
% -------------------------------------------------------------------------

% (3) DRAW LINE DIAGRAM
TowerLinePlot(WireP,NodeW,NodeR,NodeS,NodeX,NodeG);
%--------------------------------------------------------------------------

% (5) Required (initial) input info. for building a tower
head={'Name','Type','ID','Vcls','Pos x-y','Posz','Angl','Updated'};
data={'Tower No. 1' 0  1 0 [0 0] 0 0 1};
Tower0.Info.head = head;
Tower0.Info.data = data;
Tower0.ID=[];
Tower0.Ats=[];                          % '+' leaving, '+' going
Tower0.Atscab=[];                       % '+' leaving, '+' going
Cir.dat = [1001, 1 1;   % SW:10x; 35kV: 20x; 10kv: 30x; 0.4kv: 40x
           3001  3 1;   % Cir ID + # of each circuit + Ser # on the tower
           5001  4 1];  % cable
Cir.num = [2,  1,  0, 1, 0, 1;
           4,  1,  0, 3, 0, 4];
Tower0.Cir=Cir;
Tower0.GND=GLB.GND;
Tower0.VFIT=GLB.VFIT;
Tower0.Meas.node = [11; 12];       % # of Vm
Tower0.Meas.bran = [1; 2];         % # of Im

Tower0.WireP = WireP;
Tower0.BranW = BranW;
Tower0.NodeW = NodeW;

Tower0.InsuP = InsuP;
Tower0.NodeR = NodeR;

Tower0.SarrP = SarrP;
Tower0.NodeS = NodeS;

Tower0.TxfmP = TxfmP;
Tower0.NodeX = NodeX;

Tower0.SurfP = SurfP;
Tower0.NodeF = NodeF;

Tower0.GridP = GridP;
Tower0.NodeG = NodeG;

Tower0.LumpP = LumpP;
Tower0.NodeM = NodeM;
end

