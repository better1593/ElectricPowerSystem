function LGT = LGT_Init(GLB)

% (1) General parameters
Lch.dT =GLB.dT;
Lch.Nt =GLB.Nt;

GND = GLB.GND;
Lch.mur = GND.mur;
Lch.sig = GND.sig;
Lch.eps = GND.epr;
Lch.gnd = GND.gnd;              % GND mode;: 0 free-space, 1 PGD, 2 LSG
Lch.gndcha = GND.gndcha;        % GND mode;: 0 free-space, 1 PGD, 2 LSG

% (2) Channel model---- 1=Nc,2=dH,3=H0,4=flag_type,5=H,6=lamda,7=vco
channel_model = 'MTLE';  % Chan model (MTLE model(lamda,vcof))
channel_filename = "Channel Model Table.xlsx";

[num,txt,raw_data] = xlsread(channel_filename);
Nrow = size(raw_data,1);    % cell stru
for i = 1:Nrow
    str = raw_data{i,1};
    if strcmp(str,channel_model)
        channel_data = [raw_data{i,2:end-1}];
        channel_model_id = raw_data{i,end};
        break
    end
end

Lch.flg = channel_model_id;     % 1=TL model,2=MTLL model(H),3=MTLE model
Lch.H =   channel_data(1);      % channel attenuation coef
Lch.lam = channel_data(2);
Lch.vcf = channel_data(3);

Lch.H0 = 1000;              % Channel height
Lch.dH = 10;                % Length of channel segments
Lch.Nc = Lch.H0/Lch.dH;     % Number of channel segments
Lch.pos = [];       % ind: (x,y,0), dir: (S/C/T ID, seg ID, phase, seg)

% (3) Generate LGT data
% LGT.Flash.head = [];
% LGT.Stroke.head = [];
LGT.Lch = Lch;
LGT.Soc = [];
end
