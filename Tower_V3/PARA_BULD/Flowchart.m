Tower_Circuit_Build ->
    GeneInfoRead_V2 -> ComNodeRead
    NodeBranIndex_Wire -> AssignElemID
    Wire_Model_Para3 -> AssignAValue 
        Wire_Model_Para2 -> 
            Wire_Model_Para1 ->
    Lump_Model_Intepret -> AssignAValue -> AssignSuffix 
        GeneInfoRead_V2 -> ComNodeRead
        NodeBranIndex_Lump -> AssignElemID
    Tower_A2G_Bridge
    Tower_CK_Update -> AssignCValue -> AssignGValue -> InfNodeUpdate -> Lump_Souce_Update
    
Cable_Circuit_Build ->
    GeneInfoRead_V2 -> ComNodeRead
    NodeBranIndex_Line
    Cable_Para_Cal -> Cal_ZY_Cable 
        Parameter_VF_Matrix_v2 ->
        
Span_Circuit_Build ->
    GeneInfoRead_V2 -> ComNodeRead
    NodeBranIndex_Line
    OHL_Para_Cal -> Cal_LC_OHL -> Cal_Zc_OHL -> Cal_Zg_OHL
        Parameter_VF_Matrix_v2 ->

%-------------------------------------------------------------------------- 
% Part A: General Globla data
GLB.NTower = 4;                 % # of towers = 7
GLB.NSpan  = 3;                 % # of spans = 5
GLB.NCable = 0;                 % # of cables = 1

GLB.dT = 1e-8;                  % t increment
GLB.Ns = 2000;                  % calculation steps of each stroke (default
GLB.Nt = GLB.Ns;                % calculation steps of each stroke
GLB.T0 = 1e-3;                  % time interval of adjecent strokes
GLB.Tmax = 2e-3;                % max time of cacl. for multiple strokes
GLB.slg= 20;

% derived simulation constants 
GLB.Nmax = floor(GLB.Tmax/GLB.dT);
GLB.N0   = floor(GLB.T0/GLB.dT);
if GLB.Ns > GLB.N0
    GLB.Ns = GLB.N0;
end

% Ground soil parameters
GND.gnd = 2;                    % GND mode;: 0 free-space, 1 PGD, 2 LSG
GND.gndcha = 2;                 % GND mode;: 0 free-space, 1 PGD, 2 LSG
GND.mur = 1;
GND.epr = 4;
GND.sig = 1e-3;
GND.glb = 1;                    % the same gnd data for whole sys.(0/1=N/Y)
GLB.GND=GND;

Cir.dat = [1001, 1 1;   % SW:10x; 35kV: 20x; 10kv: 30x; 0.4kv: 40x
           3001  3 1;   % Cir ID + # of each circuit + Ser # on the tower
           6001  4 1];  % underground cable
Cir.num = [2,  1,  0, 1, 0, 1;      % gnd cable (not counted in num(1)
           4,  1,  0, 3, 0, 4];     % does not include cablea
GLB.Cir=Cir;
GLB.VFIT;
GLB.SSdy = SSdy;

%-------------------------------------------------------------------------- 
% Part B: Source data
% (1) Channel Parameters in LGT.Init.m
Lch.dT =GLB.dT;
Lch.Nt =GLB.Nt;
Lch.mur = GND.mur;
Lch.sig = GND.sig;
Lch.eps = GND.epr;
Lch.gnd = GND.gnd;              % GND mode;: 0 free-space, 1 PGD, 2 LSG
Lch.gndcha = GND.gndcha;        % 
Lch.flg = channel_model_id;     % 1=TL model,2=MTLL model(H),3=MTLE model
Lch.H =   channel_data(1);      % channel attenuation coef
Lch.lam = channel_data(2);      % load from a file
Lch.vcf = channel_data(3);
Lch.pos = Soc.pos(2:3);             % LGT channel position
Lch.pos = [StrPosi{Ipos(1),8:9}];   % MCLGT channel position
LGT.Lch = Lch;
% (2) Flash parameters (type, location, current, multiple strokes) in
% (2a) Simulation_LGT->Sim_LGT_Init 
% (i) Soc.typ/pos/dat/flash in LGT and GLB
        typ = 1 (dir), 2 (ind)
        pos: [x, y, 0, 0, 0, 0) (ind)
               [0/T/S/C, ID, CirID, PhaseID, CondID, Seg(node)] (dir)
        dat = icurr
        flash.head = [flash id, stroke #, CIGRE/Heilder, datainputflag]
        flash.flag/time/type (0=MC, 0=para input, 1=0.25/100us,
                                2=8/20us; 3=2.6/50us; 4=10/350us: Heidler
        flash.para =[1st stroke: CIGRE/Heilder Ip tf ...
                       2nd stroke: CIGRE/Heilder Ip tf ...]
        flash.wave =[1st stroke: CIGRE/Heilder parameter (kA/us)
                       2nd stroke: CIGRE/Heilder parameter,(kA/us)]
% (ii)SSdy.flag=[0 0 0 0 0 0 0]; one nonzero, "0"=diable, ">0"=enable
%     SSdy.Nca/dat = case#/data_para(numerical or string data)
        flag = [0 0 0 0 0 4 0];
        SSdy.flag = flag;    % 0="diable", >0="enable", Sig=1(tower
        SSdy.Nca = 2;                       % # of cases
        SSdy.dat = [0.02 10; 0.002 4];      % soil conductivity
% (iii) Summary\LGT.Soc = Soc;
        LGT.SSdy = SSdy;
        GLB.SSdy = SSdy;
        LGT.Soc = Soc;
        GLB.Soc = Soc;
        
% (2b) Simulation_LGT->Sim_MCLGT_Init
% (i) Getting sensitivity study table (SSdy)
% [NoSSdy, FO(l/2) NonR(1/2/..),SA file name, Gnd(1/2), Sig(1/2/3/4
flag = [1 0 0 0 0 0 0];             % always the single-event analysis
SSdy.flag = flag;    % 0="diable", >0="enable", Sig=1(tower
SSdy.Nca = 1;                       % # of cases
SSdy.dat = [];                      % empty
% (ii) Getting current waveform considering a multiple-stroke flash
Soc.typ = FSdist(id,3);             % 1= direct stroke, 2= indirect
if Soc.typ == 1
    Soc.pos = [StrPosi{Ipos(1),2:7}]; % 1st span, SW1, Seg3 
elseif Soc.typ == 2
    Soc.pos = [0 StrPosi{Ipos(1),8:9}]; % [x y] 
end
NumStr = FSdist(id,2);              % # of strokes
WavMod = 1;                         % Model of waveform (CIGRE/Heidler)
flash.head = [FSdist(id,1:2) WavMod 0]; % strokes with CIGRE model/CurPara
flash.flag = ones(1,NumStr);        % strokes dis/enabled in simulation
flash.time = (0: NumStr-1)*T0;      % time interval fixted to 1ms
Ipos = find(Icurr(:,1)==id);        % position of strokes in data table
flash.para = Icurr(Ipos,:);         % Current parameters of a stroke
flash.wave = Iwave(Ipos,:);         % Wavwform parameters of a stroke
% (iii) Summary\LGT.Soc = Soc;
        LGT.SSdy = SSdy;
        GLB.SSdy = SSdy;
        LGT.Soc = Soc;
        GLB.Soc = Soc;
 
% (3) flash type: Dir /Ind utilization (1/2) --> (1/0)
Tower=Tower_Circuit_Source(Tower, GLB, LGT, index, Tfold)
Cable=Cable_Circuit_Source(Cable, GLB, LGT, index, Tfold)
Span =Span_Circuit_Source(Span, GLB, LGT, index, Tfold)
[output] = LGTMC_Solu(Tower, Span, Cable, GLB, LGT, MCLGT, ...
    Icurr, Iwave, StrPosi, FSdist, PoleXY)

%-------------------------------------------------------------------------- 
% Part C: Lightning Current Data
icur = Source_Current_Generator(flash,T0,N0,dT);  % for all strokes
flash.flag;                         % cal. flag for individual strokes
% (a) Simulation_LGT->Sim_LGT_Init
-- setup by user
% (b) Simulation_LGT->Sim_MCLGT_Init
-- setup by using the heuristic approach 
% (i) Initilization in Simulation_MCLGT.m
MCLGT.Num = MC_lgtn.fixn;           % totla flash #
MCLGT.flag = 1;                     % Huristic approach                
MCLGT.huri = [];                    % Huristic dataset
MCLGT.radi = 60;                    % radius of heuristic dataset
% (ii) judgement by the heuristic approach
result_huri = Huri_Method(MCLGT, datanew)--> update flash.flag
% (iii) updating heuristic dataset
PoleApp=[PoleXY(dex(1),2:3),PoleXY(dex(2),2:3),dmin(1),dmin(2)];
MCLGT.huri = [MCLGT.huri; [icur(jd,:), PoleApp]];
MCLGT.huri = [MCLGT.huri; [icur(dex,:), repmat(PoleApp,length(dex))]];

%--------------------------------------------------------------------------
% Part D: Utilization of lightning data
LGT_Source_Build
    Tower_Circuit_Source
    Span_Circuit_Source
    Cable_Circuit_Source
    LGT.Soc -> T/S/C.Soc
    
    [Er_T, Ez_T] = E_Cal(LGT, Lne)
    [U_TL]= Cor_Lossy_ground(GLB, LGT, GND, Lne,Er_T,Ez_T); 
    Lch.curr, T/S/C.Soc.pos -> T/S/C.Soc.dat

Soc_pos_gen
    Lch.curr, LGT.Soc.typ/pos -> ???
    
    
convertCharsToStrings