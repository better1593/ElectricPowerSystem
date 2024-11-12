function [GLB,LGT] =Soc_Init(GLB,lgt_id)   
% Consider a single stroke with lgt_id [flash_id stroke_id]
%
% FlashSM = [DIR/IND, FlashID, StrokeNo, ChanModel, Pos(1:5)];   
%    DIR/IND: = 1(dir),=2(ind)
%    ChanModel: 1=TL model,2=MTLL model(H),3=MTLE model(lamda,vcof)
%    Pos: [x, y, 0, 0, 0) for IND
%         [T/S, ID, CirID, PhaseID, Seg(node)] for DIR
% StrokeSM = [FlashID, StrokeID, WaveformID, other], other=ip, dt, MC_ID
%   Waveform: -1=input; 0=MC table; 1=8/20us; 2=2.6/50us; 3=10/350us; 4=..
%      other: =Ip (WaveformID~=0); MC_ID in the table (WaveformID=0)

% (0) Initial Constants
dT=GLB.dT;
Nt=GLB.Nt;
t0=(0:Nt-1)*dT;

Chan_Model=3;
flash_id=lgt_id(1);
strok_id=lgt_id(2);

% (1) Finding out flash/stroke parameters and update Flash and Stroke
FilenameStat='Statistics Summary.txt';
FilenameDist='Flash_Stroke Dist.xlsx';
FilenamePara='b_Current Waveform_CIGRE.xlsx';
FilenamePosi='b_Stroke Position.xlsx';

Lgtntmp1=readtable(FilenameStat);
LgtnStat=Lgtntmp1.Var2(:);          % Statistic data of lightning strokes
Lgtntmp2=readtable(FilenameDist);   
LgtnDist=Lgtntmp2{:,:};             % Pair of flash id and stroke #

Nflash=LgtnStat(1);                 % Total flash #
if flash_id>Nflash
    disp('Error in Flash ID (>Max ID)');
    return;
end
Mstrok=LgtnDist(flash_id,2);       % Max stroke # of a flash
if strok_id>Mstrok
    disp('Error in Stroke ID (>Max ID)');
    return;
end    

if flash_id>1                      % the postion of stroke in the table
    strok_index=sum(LgtnDist(1:flash_id-1,2))+strok_id;
else
    strok_index=strok_id;
end
str0=num2str(strok_index+1);        % including headline
str1=['A' str0 ':J' str0];          % Range of Excel sheet
out1 = readtable(FilenamePara,'Range',str1,'ReadVariableNames',false);
wave_para=out1{:,:};                % convert table to double
out2 = readtable(FilenamePosi,'Range',str1,'ReadVariableNames',false);
tmp = out2{1,1}{1};
         
strok_pos = zeros(1,6);    % pos(6) = flag (1=updating for tower)
if strcmp(tmp,'Direct') 
    lgt_typ = 1;                        % Direct stroke
    strok_obj = out2{1,2:3};            % object (T/S/C/G) and object #
    strok_cir = str2num(out2{1,4}{1});  % circuit id
    strok_pha = str2num(out2{1,5}{1});  % phase id
    strok_seg = out2{1,6};              % segment id
    strok_pos = [strok_obj strok_cir strok_pha strok_seg 0];   
else
    lgt_typ = 2;                        % Indirect stroke
    strok_pos(2:3) = out2{1,7:8};
end
FlashSM=[lgt_typ flash_id, Mstrok, Chan_Model, strok_pos];
StrokeSM= [flash_id, strok_id, 0, strok_index];

% (2) Channel model---- 1=Nc,2=dH,3=H0,4=flag_type,5=H,6=lamda,7=vco
Lch.dT =GLB.dT;
Lch.Nt =GLB.Nt;
Lch.eps=GLB.GND.epr;
Lch.sig=GLB.GND.sig;

chan_model=FlashSM(4);  % Chan model = 3 (MTLE model(lamda,vcof))
chan_data=importdata('Channel Model Table.txt');
Lch.flg = chan_model; % 1=TL model,2=MTLL model(H),3=MTLE model(lamda,vcof)
Lch.H = chan_data.data(chan_model,1);       % channel attenuation coef
Lch.lam = chan_data.data(chan_model,2);
Lch.vcf = chan_data.data(chan_model,3);

Lch.H0 = 1000;              % Channel height
Lch.dH = 10;                % Length of channel segments
Lch.Nc = Lch.H0/Lch.dH;     % Number of channel segments
Lch.pos = FlashSM(5:10); % ind: (x,y,0), dir: (S/C/T ID, seg ID, phase, seg)

% (3) Current model------
current_waveform = StrokeSM(3);  % 1=10/350,2=1/200,3=0.25/100
if ismember(current_waveform, [1, 2, 3])
    curdata=importdata('Current Model Table.txt');
    Ip = StrokeSM(4);
    k = curdata.data(current_waveform,2);
    tau1 = curdata.data(current_waveform,3)*1e-6;  % unit: us
    tau2 = curdata.data(current_waveform,4)*1e-6;  % unit: us

    Icurr = (Ip /k)*((t0/tau1).^10)./(1 +(t0/tau1).^10).* exp(-t0/tau2); 
elseif current_waveform==0 %  looking MC table, find out current wave form
    Icurr = CIGRE_Waveform_Generator(wave_para,t0);
elseif current_waveform==-1
%     input current wave;
else    
    error('Error in Waveform ID');
end

% (3) Updating Flash/stroke data
Soc.typ = FlashSM(1);    % Source: 1=dir,2=ind,11=Vs,ac,12=Is,ac
Soc.pos = FlashSM(5:10);  % ind:[x,y,0,0,0 0]
                            % dir:[T/S, ID, CirID, PhaseID, Seg(node) flag]
Soc.dat = Icurr;            % current data
Lch.curr = Icurr;           % current data

LGT.Flash.head = FlashSM;
LGT.Stroke.head = StrokeSM;
LGT.Lch = Lch;
LGT.Soc = Soc;
GLB.Soc = Soc;              % Source: 1=dir, 2=ind, 3=Vs,ac, 4 =Is,ac

figure(2);
plot(t0,Icurr);
end
