function [ Cable ] = Cable1_Init(GLB)
% Cable.Seg.slg = GLB.slg               % segment length in free space
% Cable.Seg.Lseg = GLB.slg              % segment length in dialectric
% Cable.Seg.Nseg =                      % segment number
% Cable.Seg.Ncon =                      % number of conductors per segment
% 
% Cable.Line.pos=[x1 y1 z1 x2 y2 z2];     % pole-pole position
%       Line.rad=[rc,rd,ra1,ra2,rs];      % core, core posi, armor1/2,shearth
%       Line.mat=[sigc,siga,murc,mura,epri] % core=c, armor=a, insulation=i
%       Line.con=total #, core #, arm #] % number of conductors
%       Line.typ=1 (air) -1 (gnd)

V0=3e8;                                 % light speed
% (1) Required (initial) input info. for building a tower
head={'Name','Type','ID','Vcls','Pstr (x-y)','Pend (x-y)','Height','Updated'};
data={'Cable No. 1' 1  1 [0 0] [1020 -150] -1 1};
Cable.Info.head = head;
Cable.Info.data = data;
Cable.ID=1;
Cable.Atn=[1 5];                      % 1st= head, 2nd=tail
Cable.Cir.num=[0,  0,  0, 0, 0, 1];
Cable.Cir.dat=[5001 4 1 1];
Cable.Seg=[];
Cable.Line=[];
Cable.Node=[];
Cable.Bran=[];
Cable.Soc=[];
Cable.mode=[1 1 1];     % [CondImpVF GndImpVF LossyGndCha];  % 1=yes, 0=no
Cable.GND=GLB.GND;
Cable.VFIT=GLB.VFIT;
Cable.Meas.node = [4, 2];       % [Ncon, Nseg]
Cable.Meas.bran = [4, 1];       % [Ncon, Nseg]

height=data{6};
rc=0.01;rd=0.02;ra1=0.04;ra2=0.042;rs=0.045;% core,core,posi,armor1/2,shearth
sigc=5.8e7;siga=1e6;murc=1;mura=40;epri=4;  % core=c, armor=a, insulation=i
Line.pos=[data{4} height data{5} height];   % pole-pole position
Line.con=[4 3 1];           % total #, core #, arm #: number of conductors
Line.rad=[rc,rd,ra1,ra2,rs,height];
sigc=5.8e7;siga=1e6;murc=1;mura=40;epri=4; % core=c, armor=a, insulation=i
Line.mat=[sigc,siga,murc,mura,epri];% core=c, armor=a, insulation=i
if height>0
    Line.typ=1;
else
    Line.typ=-1;
end
 
dis=Line.pos(1:2)-Line.pos(3:4);
dis=sqrt(sum(dis.*dis,2));          % distance of the cable
Seg.slg = GLB.slg;
Seg.Lseg = Seg.slg;                 % segment length in dialectric
Seg.Nseg = round(dis/Seg.Lseg);     % segment number
Seg.Ncon = Line.con(1);             % number of conductors per segment
Seg.Npha = Line.con(2);             % # of phase conductors
Seg.velc = V0/sqrt(Line.mat(5));
Seg.vela = Seg.velc;

Nseg = Seg.Nseg;
Ncon = Seg.Ncon;
Ncom = Seg.Ncon;                    % # of com nodes per port
Nn = (Nseg+1)*Ncon;                 % total # of nodes
tmp = (1:Nn)';
Node.list=num2str(tmp); % ordering: v (cond) x h (segment) from str to end
Node.listdex=tmp;       % ordering: v (cond) x h (segment) from str to end
Node.num=[Nn,Nn,0,Nn,Ncom];                 % 4 com nodes per port
Node.comdex=[Node.listdex(1:Ncom) Node.listdex(end-Ncom+1:end)]; 

str0=num2str(Cable.Cir.dat(1));
str1="Cir"+str0+"_";
strh=num2str(Cable.Cir.dat(3));
strt=num2str(Cable.Cir.dat(4));
strh=strh+"_T"+num2str(Cable.Atn(1),GLB.IDformat);
strt=strt+"_T"+num2str(Cable.Atn(2),GLB.IDformat);
tmph=[str1+"A"+strh;str1+"B"+strh;str1+"C"+strh;str1+"M"+strh;];
tmpt=[str1+"A"+strt;str1+"B"+strt;str1+"C"+strt;str1+"M"+strt;];
Node.com=[tmph tmpt];

Nb=Nseg*Ncon; 
tmp = (1:Nb)';
Bran.list = num2str(tmp);% ordering: v (cond) x h (segment) from str to end
Bran.listdex=tmp;        % ordering: v (cond) x h (segment) from str to end
Bran.num=[Nb,Nb,0,0,0,0];               % 3 brans per port

Cable.Node=Node;
Cable.Bran=Bran;

Cable_Para=zeros(Line.con(1),6);
Cable_Para(1:end-1,1)=Line.rad(1);
Cable_Para(1:end-1,2)=Line.rad(1);
Cable_Para(1:end-1,3)=Line.rad(2);
Cable_Para(end,1:3)=Line.rad(3:5);
theta=2*pi/Line.con(2);
arrng=(0:Line.con(2)-1)*theta;
for ik=1:Line.con(2)
    Cable_Para(ik,4:end)=arrng;
    arrng=circshift(arrng,1);
end
    
Cable.Seg=Seg;
Cable.Line=Line;
Cable.Cable_Para=Cable_Para;
end