function [ GLB ] = Global_Init(FDIR)

%*********** 4 digits for Key Cir, Node and Bran ID and names *************
%--------------------(1) Global parameters --------------------------------
GLB.FDIR = FDIR;
GLB.IDformat='%04d';

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

% Other constants

% GLB.SSdy.flag = 0;              % defualt = no sensitivity study

% collected from a preset conductor table (database)
VFIT.fr=[1 100 1000];
VFIT.rc=[1e-3 1e-4 1e-5 1e-5 1e-5 1e-5];  % internal imepdance of conductor 
VFIT.dc=[2e-7 3e-6 4e-6 5e-6 1e-5 1e-5];  % order=3,r0+d0*s+sum(ri/s+di)
VFIT.odc=5;                         % VFIT order of conductors
% calculated within a SPAN module
VFIT.rg=[1e-3 1e-4 1e-5 1e-5];      % ground impedance
VFIT.dg=[2e-7 3e-6 4e-6 5e-6];      % order=3,r0+d0*s+sum(ri/s+di)
VFIT.odg=5;                         % VFIT order of conductors
VFIT.fr = [1:10:90 1e2:1e2:9e2 1e3:1e3:9e3 1e4:1e4:9e4 1e5:1e5:7e5]; 
GLB.VFIT=VFIT;

% Define TOW/OHL/CABLE Type parameters
% GLB.A =  [-1 1 0 0 0 0 0;           % incidence matrix btw. span and tower
%           0 -1 1 0 0 0 0;
%           0 0 -1 1 0 0 0;
%           0 0 -1 0 1 0 0;
%           -1 0 0 0 0 1 0];
% GLB.A =  [-1 1];
GLB.A =  [-1 1 0 0 ;           % incidence matrix btw. span and tower
          0 -1 1 0 ;
          0 0 -1 1 ;];
         
% GLB.Acab = [0 0 0 0 -1 0 1];        % underground cable btw T5 and T6 (LV)    
GLB.Acab = [ ];        % underground cable btw T1 and T5    
end
