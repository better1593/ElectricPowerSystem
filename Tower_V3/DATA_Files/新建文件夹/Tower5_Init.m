function Tower0=Tower5_Init(GLB)

% Define TOW/OHL/CABLE Type parameters
OvhdP = 1;                          % OHL type:
CablP = 1;                          % CAB type:
InsuP.Cir = 3001;                 % INS type: Circuit ID (10kV)
InsuP.Ser = 1;                   % INS type: Ser No. (Eq. ID) 
InsuP.Nph = 3;                   % INS type: # of phases 
InsuP.Pha = [];                  % INS type: phase ID
SarrP.Cir = 3001;                 % SAR type: Circuit ID (10kV)
SarrP.Ser = 1;                   % SAR type: Ser No. (Eq. ID) 
SarrP.Nph = 3;                   % SAR type: # of phases 
SarrP.Pha = [];                  % INS type: phase IDTxfmP(1).Cir = 301;                 % XFM type: Circuit ID (10kV)
TxfmP.Cir = 3001;                 % SAR type: Circuit ID (10kV)
TxfmP.Ser = 1;                   % XFM type: Ser No. (Eq. ID) 
TxfmP.Nph = 3;                   % XFM type: # of phases 
TxfmP.Pha = [];                  % INS type: phase ID
% SurfP = [1 0 1 0];             % SUR type: [Pole#, SA#, Tx#, other#]
% GridP = [];                    % GND type: lump ground IMP. (1=yes)
SurfP = [];                      % SUR type: [Pole#, SA#, Tx#, other#]
GridP = [1 0 1 0];               % GND type: [Pole#, SA#, Tx#, other#]
LumpP = [];                      % lump element in the future

if (~isempty(SurfP))&(~isempty(GridP))
    disp('Problem of Ground Setting');
    return;
end

%----------------(2) Tower parameters -------------------------------
% (2a) Node/Bran in Tower in the air
Ndata  =[{1  0 +0.0 11} {'Cir1001_S1_T0005'}; %POLE(Common node: Tower_Pole)    
         {2  0 -0.4 10} {'Cir3001_A1_T0005'}; %H1_PA1 (Common node: 10kV_A1)
         {3  0 +0.1 10} {'Cir3001_B1_T0005'}; %H1_PB1 (Common node: 10kV_B1)
         {4  0 +0.6 10} {'Cir3001_C1_T0005'}; %H1_PB1 (Common node: 10kV_C1)
         {5  0 -0.5 8}  {'Cir4001_A1_T0005'}; %L1_PA1 (Common node: 0.4kV_A1)
         {6  0 -0.2 8}  {'Cir4001_B1_T0005'}; %L1_PB1 (Common node: 0.4kV_B1)
         {7  0 +0.1 8}  {'Cir4001_C1_T0005'}; %L1_PB1 (Common node: 0.4kV_C1)
         {8  0 +0.4 8}  {'Cir4001_N1_T0005'}; %L1_PB1 (Common node: 0.4kV_N1)
         {9  0 -0.4 9.8} {'Cir3001_A1_IN1'}; %IN_PA1 (Common node: IN_A top)
         {10 0 +0.1 9.8} {'Cir3001_B1_IN1'}; %IN_PB1 (Common node: IN_B top)
         {11 0 +0.6 9.8} {'Cir3001_C1_IN1'}; %IN_PB1 (Common node: IN_C top)
         {12 0 -0.4 9.8} {'Cir3001_A1_IN2'}; %IN_PA2 (Common node: IN_A bot)
         {13 0 +0.1 9.8} {'Cir3001_B1_IN2'}; %IN_PA2 (Common node: IN_B bot)
         {14 0 +0.6 9.8} {'Cir3001_C1_IN2'}; %IN_PA2 (Common node: IN_C bot)
         {15 0 -0.3 10} {'Cir3001_A1_SA0'};  % SA_PA1 (Common node: SA_A0)
         {16 0 +0.2 10} {'Cir3001_B1_SA0'};  % SA_PB1 (Common node: SA_B0)
         {17 0 +0.5 10} {'Cir3001_C1_SA0'};  % SA_PC1 (Common node: SA_C0)
         {18 0 -0.3 9.8} {'Cir3001_A1_SA1'}; % SA_PA1 (Common node: SA_1_A)
         {19 0 +0.2 9.8} {'Cir3001_B1_SA1'}; % SA_PB1 (Common node: SA_2_B)
         {20 0 +0.5 9.8} {'Cir3001_C1_SA1'}; % SA_PC1 (Common node: SA_3_C)
         {21 0 -0.3 9.8} {'Cir3001_A1_SA2'}; % SA_PA2 (Common node: SA_4_A)
         {22 0 +0.2 9.8} {'Cir3001_B1_SA2'}; % SA_PB2 (Common node: SA_5_B)
         {23 0 +0.5 9.8} {'Cir3001_C1_SA2'}; % SA_PB2 (Common node: SA_6_C)
         {24 2 +0.1 2} {'Cir3001_A1_TX1'};   % TX_A (Common node: TX_1)
         {25 2 +0.1 2} {'Cir3001_B1_TX1'};   % TX_B (Common node: TX_2)
         {26 2 +0.1 2} {'Cir3001_C1_TX1'};   % TX_C (Common node: TX_3)
         {27 2 +0.1 2} {'Cir3001_a1_TX1'};   % TX_a (Common node: TX_4)
         {28 2 +0.1 2} {'Cir3001_b1_TX1'};   % TX_b (Common node: TX_5)
         {29 2 +0.1 2} {'Cir3001_c1_TX1'};   % TX_c (Common node: TX_6)
         {30 2 +0.1 2} {'Cir3001_N1_TX1'};   % TX_Neutral (Common node: TX_7
         {31 0 +0.0 9.8} {'PO_P01'};     % Pole middle point  
         {32 0 +0.0 0} {'PO_GD1_0'};       % Pole ground surf
         {33 2 +0.1 0} {'TX_GD1_0'};       % TX gnd surface
         {34 0 +0.1 1.0} {'Cir5001_A1_T0005'}; %Cab_PA1 (Common node: Cable A)
         {35 0 +0.1 1.0} {'Cir5001_B1_T0005'}; %Cab_Pb1 (Common node: Cable B)
         {36 0 +0.1 1.0} {'Cir5001_C1_T0005'}; %Cab_PC2 (Common node: Cable C)
         {37 0 +0.1 1.0} {'Cir5001_M1_T0005'}];%Cab_PM2 (Common node: Cable M) 

list=string(Ndata(:,5));
tmp=Ndata(:,1:4); Npos=cell2mat(tmp);
Wire =[{2   9 Npos( 2,2:4) Npos( 9,2:4)} {'IN_A'};      
       {3  10 Npos( 3,2:4) Npos(10,2:4)} {'IN_B'};       
       {4  11 Npos( 4,2:4) Npos(11,2:4)} {'IN_C'};      
       {15 18 Npos(15,2:4) Npos(18,2:4)} {'SA_A'};      
       {16 19 Npos(16,2:4) Npos(19,2:4)} {'SA_B'};       
       {17 20 Npos(17,2:4) Npos(20,2:4)} {'SA_C'};      
       {2  15 Npos( 2,2:4) Npos(15,2:4)} {'SA_Aup'};      
       {3  16 Npos( 3,2:4) Npos(16,2:4)} {'SA_Bup'};       
       {4  17 Npos( 4,2:4) Npos(17,2:4)} {'SA_Cup'};      
       {12 21 Npos(12,2:4) Npos(21,2:4)} {'SA_Adn'};      
       {13 22 Npos(13,2:4) Npos(22,2:4)} {'SA_Bdn'};       
       {14 23 Npos(14,2:4) Npos(23,2:4)} {'SA_Cdn'};      
       {21 31 Npos(21,2:4) Npos(31,2:4)} {'HD_AB'};      
       {13 31 Npos(13,2:4) Npos(31,2:4)} {'HD_BP'};       
       {23 22 Npos(23,2:4) Npos(22,2:4)} {'HD_CB'};      
       {1  31 Npos( 1,2:4) Npos(31,2:4)} {'PO_1'};
       {31 32 Npos(31,2:4) Npos(32,2:4)} {'PO_2'};
       {2  24 Npos( 2,2:4) Npos(24,2:4)} {'TX_A'};     
       {3  25 Npos( 3,2:4) Npos(25,2:4)} {'TX_B'};    
       {4  26 Npos( 4,2:4) Npos(26,2:4)} {'TX_C'};     
       {5  27 Npos( 5,2:4) Npos(27,2:4)} {'TX_a'};     
       {6  28 Npos( 6,2:4) Npos(28,2:4)} {'TX_b'};    
       {7  29 Npos( 7,2:4) Npos(29,2:4)} {'TX_c'};            
       {8  30 Npos( 8,2:4) Npos(30,2:4)} {'TX_n'};            
       {30 33 Npos(30,2:4) Npos(33,2:4)} {'TX_Nu'};            
       {2  34 Npos( 2,2:4) Npos(34,2:4)} {'Cab_A'};       
       {3  35 Npos( 3,2:4) Npos(35,2:4)} {'Cab_B'};      
       {4  36 Npos( 4,2:4) Npos(36,2:4)} {'Cab_C'};
       {31 37 Npos(31,2:4) Npos(37,2:4)} {'Cab_M'}];
   
NodeW.list=list; 
NodeW.listdex = Npos(1:end,1);
nlen=length(list);
NodeW.num = [nlen nlen 0 0]; 
NodeW.com=list([1:8 34:37]);                 % connected to OHL (SW+3XHV+4xLV)
NodeW.comdex=Npos([1:8 34:37],1);
[blen, n0]=size(Wire);
BranW.num = [blen blen 0 0 0 0];

NodeR.com=list(9:14);               % insulator
NodeR.comdex=Npos(9:14,1);

NodeS.com=list(18:23);              % spd
NodeS.comdex=Npos(18:23,1);

NodeX.com=list(24:30);              % transformer
NodeX.comdex=Npos(24:30,1);

NodeF.com=[];                       % lump elements for wire gnd onnection
NodeF.comdex=[];

NodeG.com=list(32:33);              % lump gnd elements
NodeG.comdex=Npos(32:33,1);

NodeM.com=[];                       % other lump elements
NodeM.comdex=[];

% (2b) Node and bran under the ground 
if isempty(GridP)
    Ndatag= [{38 0 +0.0 0} {'GD_PO1_0'};    % Pole gnd electrode (middel)
            {39 2 +0.1  0} {'GD_TX1_0'};    % Pole gnd electrode (middel)
            {40 0 +0.0 -1} {'GD_PO1_1'};    % Pole gnd electrode (middel)
            {41 0 -2.0 -1} {'GD_PO1_2'};    % Pole gnd electrode (left)
            {42 0 +2.0 -1} {'GD_PO1_3'};    % Pole gnd electrode (right)
            {43 2 +0.1 -2} {'GD_TX1_1'}];   % TX ground electrode
    Ndata=[Ndata; Ndatag];
    tmp=Ndata(:,1:4); Npos=cell2mat(tmp);
    list=string(Ndata(:,5));
    NodeW.list=list;
    NodeW.listdex = Npos(1:end,1);
    n0len=length(list);
    NodeW.num = [n0len nlen n0len-nlen nlen]; % Nodew.num(4)=offset for gnd node dex
% Nodes 31 and 32 are surf. nodes given in the air group
% Nodes 38 and 39 are surf. nodes given in the gnd group
    Wireg= [{38 40 Npos(38,2:4) Npos(40,2:4)} {'PO_G1'};
            {40 41 Npos(40,2:4) Npos(41,2:4)} {'PO_G2'};
            {40 42 Npos(40,2:4) Npos(42,2:4)} {'PO_G3'};
            {39 43 Npos(39,2:4) Npos(43,2:4)} {'TX_G1'}];
    Wire=[Wire; Wireg];
    [b0len, n0]=size(Wire);
    BranW.num = [b0len blen b0len-blen 0 0 0];   
    
    NodeF.com=list(32:35);          % lump elements for wire gnd onnection
    NodeF.comdex=Npos(32:35,1);     % half of air nodes and half gnd nodes

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
data={'Tower 5' 0  1 0 [1020 -150] 0 90 1};
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
Tower0.VFIT = GLB.VFIT;
Tower0.GND = GLB.GND;
Tower0.Meas.node = [1; 2];         % # of Vm
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

