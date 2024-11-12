function [Span, S2Tmap]= Span_Circuit_Build...
    (SpanModelName,TCom,TH_Info,TH_Node,TT_Info,TT_Node,VFIT,GLB)
% Generating circuit parameters and config. info. for a span
%   OHL_Para=Span.line=
%    [x1 y1 z2 x2 y2 z2 0  0  0    0    0   0   0;   % pole-pole position
%     x1 y1 z2 x2 y2 z2 0  rc Rint Lint sig mur epr; % SW
%     x1 y1 z2 x2 y2 z2 dh rc Rint Lint sig mur epr; % HV->MV->LV, PH A->B->C 
%              ....                           ]; % dh: offset (hori)
%   Bran.list=[name list];  Bran.listdex=[b1 n1 n2];
%   Bran.num=[total #, air #, gnd  #, dist #, lump #, Speical #] 1-6
%   Node.list=[name list];  Node.listdex=[n1]; 
%   Node.num=[total #, air #, surf #, gnd  #] 1-4
%   Node.com=[name list];   Node.comdex=[int: n1,  ext: n2]; 
%
%   GND.sig, GND.epi, GND.dex (0=no, 1= perfect and (2) lossy)
%
%   NodeCom.list/listdex [T_head T_tail C_head C_tail]
%   NodeCom.[coordinate [Thead(X/Y/Z) Ttail(X/Y/Z)]
%   Seg.Nseg/Ncon/Nphas/Ngdw/Lseg, Seg.num=[Ntot Npha Ngdw]= Line.con

% (1) 初始化变量，Read a complete table 
Vair = 3e8;                                     % light speed
[num,txt,raw_data] = xlsread(SpanModelName);
% data0 = readtable(SpanModelName);      
%--------------------------------------------------------------------------

% (2a) Read general info. of a Cable, T_head, T_tail
[data,Blok,Info,Nwir,GND] = GeneInfoRead_V2(raw_data);
Span.Info = Info;
Span.ID = Info{1,10};
Span.Atn = GLB.A(Span.ID,:);
if GND.glb==1
   Span.GND = GLB.GND;
end
Blok.Seg.Lseg = GLB.slg;                        % length of a segment
Blok.Seg.num(4) = GLB.slg;                      % length of a segment

% (2b) Read relative info. of T_head, T_tail
Span.Info(1,3) = TH_Info(1,1);                  % name of head tower
Span.Info(1,4) = TT_Info(1,1);                  % name of tail tower
Pole(1,1:3) = [TH_Info{1,5:7}];                 % pos of head tower
Pole(1,4:6) = [TT_Info{1,5:7}];                 % pos of tead tower
Span.Info(1,5) = TH_Info(1,10);                 % id of head tower
Span.Info(1,6) = TT_Info(1,10);                 % id of tead tower
rng = 1;
TC_head = TCom.head;                            % Connected tower nodes
TC_tail = TCom.tail;                            % Connected tower nodes
TC_head= AssignElemID(TH_Node,TC_head,rng);     % ID and local pos
TC_tail= AssignElemID(TT_Node,TC_tail,rng);     % ID and local pos
%-------------------------------------------------------------------------

% (3) Getting wire parameters
% (3a) Initilization of Span Model
[Node,Bran,Meas,Cir,Seg]=NodeBranIndex_Line(data,Blok,Pole);
Node.com = [Node.com(:,1) TC_head.list Node.com(:,2) TC_tail.list];
Node.comdex = [Node.comdex(:,1) TC_head.listdex ...
               Node.comdex(:,2) TC_tail.listdex]; 
Ncom = size(Node.com,1);                        % # of common nodes
for i = 1:Ncom
    Node.pos(i,:) = [TC_head.pos(i,:) TC_tail.pos(i,:)] + Pole;
end

% (3b) Read wire parameters of a span
Ncon = Seg.Ncon;                                % # of line/conductors
OHLP = cell2mat(data(1:Ncon,6:20));             % wire para with pole data
OHLP(:,1:6) = Node.pos;                         % update (x/y/z) of OHLP

% (3c) mapping table (hspan/hsid/Thead tspan/tsid/Ttail)
S2Tmap.head = [[Span.ID TH_Info{1,10}]; Node.comdex(:,1:2)];
S2Tmap.tail = [[Span.ID TT_Info{1,10}]; Node.comdex(:,3:4)];

%--------------------------------------------------------------------------

% (4) Calculate wave equation parameters (Z and Y)
VFmod = [Info{1,8:9}];                       % 
Para = OHL_Para_Cal(OHLP,VFmod,VFIT,GND);
% Para = Span_Circuit_Para(Info,OHLP,GND);

% (5) Output---------------------------------------------------------------
Span.Cir = Cir;
Span.Seg = Seg;
Span.Pole = Pole;
Span.OHLP = OHLP;
Span.Para = Para;
Span.Node = Node;
Span.Bran = Bran;
Span.Meas = Meas;
Span.S2Tmap = S2Tmap; 
Span.Soc = [];
end