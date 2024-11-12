function [Z] = Cal_ZY_Cable(Line, GND, fre)
%   Calculate frequnecy-variant circuit parameters of piped cables
%   Cable.Line.pos=[x1 y1 z1 x2 y2 z2]; % pole-pole position
%       Line.rad=[rc,rd,ra1,ra2,rs];    % core, core posi, armor1/2,shearth
%       Line.mat=[sigc,siga,murc,mura,epri] % core=c, armor=a, insulation=i
%       Line.con=[total #, core #, arm #] % number of conductors
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
Mu_g = mu0 * GND.mur; 

ep0=8.854187818e-12;
epr_i = Line.mat(5);
Eps_i = ep0 * epr_i;            % insulation epsilon
Eps_c = ep0 * 1;                % air/core epsilon
Eps_g = ep0 * GND.epr;          % ground epsilon

Sig_c = Line.mat(1);            % core sigma
Sig_a = Line.mat(2);            % amore sigma
Sig_g = GND.sig;                % gnd sigma

Ncon = Line.con(1);             % number of conductors per segment
Npha = Line.con(2);             % # of phase conductors
Nf = numel(fre);

rc = Line.rad(1);               % core radius
rd = Line.rad(2);
rsa = Line.rad(3);              % inner radus of armor
rsb = Line.rad(4);              % outer radus of armor
rsc = Line.rad(5);              % overal radius of a cable
hit = Line.rad(6);              % height of the cable above/below gnd
theta = 2*pi/Npha;              % angle between two cores
thrng=((0:Npha-1)*theta)';      % theta range (Npha x 1)

omega    = 2 * pi * fre;
gamma_c = sqrt(1i * Mu_c * omega .* (Sig_c + 1i * omega * Eps_c));
gamma_a = sqrt(1i * Mu_a * omega .* (Sig_a + 1i * omega * Eps_c));
gamma_g = sqrt(1i * Mu_g * omega .* (Sig_g + 1i * omega * Eps_g));
Rs_a = rsa * gamma_a;
Rs_b = rsb * gamma_a;
Hit_g = abs(hit) * gamma_g;     % gamma of the ground

K0_in_pipe = besselk(0,Rs_a);
K1_in_pipe = besselk(1,Rs_a);
I0_in_pipe = besseli(0,Rs_a);
I1_in_pipe = besseli(1,Rs_a);

K0_out_pipe = besselk(0,Rs_b);
K1_out_pipe = besselk(1,Rs_b);
I0_out_pipe = besseli(0,Rs_b);
I1_out_pipe = besseli(1,Rs_b);

Rc = rc * gamma_c;
I0c =  besseli(0,Rc);
I1c =  besseli(1,Rc);

Zco = 1i*omega*Mu_c.*I0c./(2*pi*Rc.*I1c);               % core inner imp
out = 1i*omega*Mu_a/(2*pi).*K0_in_pipe./(Rs_a.*K1_in_pipe); % Nf vector
out = repmat(out, Npha,1);                              % Npha x Nf matrix
for x = 1:15
    out = out+cos(x*thrng)*(1i*omega*Mu_a/(2*pi).*(rd/rsa)^(2*x)*2./...
          (x*(1+ mur_a)+Rs_a.*besselk(x-1, Rs_a)./besselk(x,Rs_a)));
end
out(1,:) = out(1,:)+ Zco;                               % self impedance

Z = zeros(Ncon, Ncon, Nf);                              % Npha x Npha x Nf
for ik=1:Npha                                           % rotation
    Z(ik,1:Npha,1:Nf) = out;                            % 1 x Npha x Nf
    out=circshift(out,1,1);
end

Zsa = 1i*omega*Mu_a/(2*pi)./(Rs_a.* Rs_b)./(I1_in_pipe .* K1_out_pipe- ...
                                            I1_out_pipe.* K1_in_pipe);

tmp1 = I0_out_pipe.*K1_in_pipe + I1_in_pipe.* K0_out_pipe;
tmp2 = I1_out_pipe.*K1_in_pipe - I1_in_pipe.* K1_out_pipe;
Zpo = 1i*omega*Mu_a./(2*pi*Rs_b).*tmp1./tmp2;
Zpg = 1i*omega*Mu_a/(2*pi).*(log((1+Hit_g)./Hit_g));
Zaa = Zpo + Zpg;

for ik = 1:Npha
    Z(ik,Ncon,:) = Zsa;
    Z(Ncon,ik,:) = Zsa.';
end

Z(Ncon,Ncon,:) = Zaa;
end

