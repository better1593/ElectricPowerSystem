function [L, C] = Cal_LC_Cable(Line)
%   Calculate frequnecy-variant circuit parameters of piped cables
%   Cable.Line.pos=[x1 y1 z1 x2 y2 z2]; % pole-pole position
%       Line.rad=[rc,rd,ra1,ra2,rs];    % core, core posi, armor1/2,shearth
%       Line.mat=[sigc,siga,murc,mura,epri] % core=c, armor=a, insulation=i
%       Line.con=[tot# pha# sw# slen seg# depth] % number of conductors
%       Line.typ=1 (air) -1 (gnd)
%   Cable_Para=[
%     rc1_inner, rc1_outer, d1, 0, 2/3*pi, 2/3*pi; % core 1 para
%     rc2_inner, rc2_outer, d2, 2/3*pi, 0, 2/3*pi; % core 2 para
%     rc3_inner, rc3_outer, d3, 2/3*pi, 2/3*pi, 0; % core 3 para
%     rc4_inner, rc4_outer, d4; % core 4 para (optional)
%     ra_inner,   ra_outer, over,  0,    0,   0]   % armor para

mu0 = 4*pi*1e-7;
mur_c = Line.mat(3);
mur_a = Line.mat(4);
Mu_c = mu0 * mur_c;
Mu_a = mu0 * mur_a;
Mu_i = mu0 * 1;

V0 = 3e8;                       % light speed
Vc = V0/sqrt(Line.mat(5));      % speed in cable with directric material

Ncon = Line.con(1);             % number of conductors per segment
Npha = Line.con(2);             % # of phase conductors

rc = Line.rad(1);               % core radius
rd = Line.rad(2);
rsa = Line.rad(3);              % inner radus of armor
rsb = Line.rad(4);              % outer radus of armor
rsc = Line.rad(5);              % overal radius of a cable
hit = Line.rad(6);              % height of the cable above/below gnd
theta = 2*pi/Npha;              % angle between two cores
thrng=(0:Npha-1)*theta;         % theta range

tds = rd/rsa;                   % ratio of d/rsa
out=Mu_c/(2*pi)*log(tds*sqrt((1+1/tds^4-2/tds^2*cos(thrng))./(2-2*cos(thrng))));
out(1)=Mu_c/(2*pi)*log((rsa/rc)*(1-(tds)^2));       % self inductance

L = zeros(Ncon, Ncon);
for ik=1:Npha
    L(ik,1:Npha)=out;
    out=circshift(out,1);
end
if hit >0
    L(end,end) = mu_0/(2*pi)*log(2*hit/rsb);        % above the ground     !!!!
    C(1:Ncon, 1:Ncon)=inv(L(1:Ncon, 1:Ncon))/Vc^2;  %                      !!!!                
    C(end,end)=inv(L(end,end))/V0^2;                %                      !!!!    
else
    L(end,end) = Mu_i/(2*pi)*log(rsc/rsb);          % under the ground     !!!!
    C(1:Ncon, 1:Ncon)=inv(L(1:Ncon, 1:Ncon))/Vc^2;  %                      !!!!    
    C(end, end) = inv(L(end, end))/V0^2;            %                      !!!!    
end 
end

