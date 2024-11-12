function [Z_11,Z_22,Z_33,Z_12,Z_21,Z_23,Z_32,C_1,C_2,C_3] = ZS_Coaxil(r_o_c,...
    r_i_s,r_o_s,r_i_a,r_o_a,sig_c,mur_c,sig_s,mur_s,sig_a,mur_a,fre,eps_cs,...
    mur_cs,eps_sa,mur_sa,height)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate surface impedance of circular conductors using bessel function. 
% 
% Output: 
%               Observation location              Source location
% Zc_o_s1      outer surface of core               core conductor
% Zs_i_s1      inner surface of shell              core conductor
% Zs_i_s2      inner surface of shell              shell conductor
% Zs_o_s2      outer surface of shell              shell conductor
% Za_i_s12     inner surface of armor              core and shell conductor
% Za_i_s3      inner surface of armor              armor conductor
% Za_o_s3      outer surface of armor              armor conductor
% 
% Input:
% r_o_c :  Radius of outer surface of core conductor
% r_i_s :  Radius of inner surface of shell conductor
% r_o_s :  Radius of outer surface of shell conductor
% r_i_a :  Radius of inner surface of armor conductor
% r_o_a :  Radius of outer surface of armor conductor
% sig   :  Conductivity
% mur   :  Permeability
% fre   :  Interested frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Physical constant
mu0    = 4*pi*1e-7;
Mur_core  = mu0 * mur_c;
Mur_shell = mu0 * mur_s;
Mur_armor = mu0 * mur_a;
Mur_CS = mu0 * mur_cs;
Mur_SA = mu0 * mur_sa;
eps0   = 8.854187818e-12;
Eps    = eps0 * 1;
Eps_CS = eps0 * eps_cs;
Eps_SA = eps0 * eps_sa;
Sig_core  = sig_c;
Sig_shell = sig_s;
Sig_armor = sig_a;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C and L of the armed cable
C_1 = 2 * pi * Eps_CS / log(r_i_s / r_o_c);
C_2 = 2 * pi * Eps_SA / log(r_i_a / r_o_s);
C_3 = 2 * pi * Eps_SA / log(2 * height / r_o_a);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bessel initialization
omega    = 2 * pi * fre;                          
gamma_core  = sqrt(1i * Mur_core * omega .* (Sig_core + 1i * omega * Eps));
gamma_shell = sqrt(1i * Mur_shell * omega .* (Sig_shell + 1i * omega * Eps));
gamma_armor = sqrt(1i * Mur_armor * omega .* (Sig_armor + 1i * omega * Eps));
Rc = r_o_c * gamma_core;
Ra = r_i_s * gamma_shell;
Rb = r_o_s * gamma_shell;
Rd = r_i_a * gamma_armor;
Re = r_o_a * gamma_armor;

I0c = besseli(0,Rc);
I1c = besseli(1,Rc);
I0a = besseli(0,Ra);
I1a = besseli(1,Ra);
I0b = besseli(0,Rb);
I1b = besseli(1,Rb);
I0d = besseli(0,Rd);
I1d = besseli(1,Rd);
I0e = besseli(0,Re);
I1e = besseli(1,Re);
K0a = besselk(0,Ra);
K1a = besselk(1,Ra);
K0b = besselk(0,Rb);
K1b = besselk(1,Rb);
K0d = besselk(0,Rd);
K1d = besselk(1,Rd);
K0e = besselk(0,Re);
K1e = besselk(1,Re);
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Impedance calculation
Zcore_out = 1i * omega * Mur_core .* I0c ./ (2 * pi * Rc .* I1c);

% DD_shell = 1 ./ (2 * pi * Sig_shell * (K1a .* I1b - I1a .* K1b));
% Zsheath_in = gamma_shell .* DD_shell / r_i_s .* (I0a .* K1b + K0a .* I1b);
% Zsheath_out = gamma_shell .* DD_shell / r_o_s .* (I0b .* K1a + K0b .* I1a);
DD_shell = 1i * omega * Mur_shell ./ (2 * pi * Ra .* Rb .* (K1a .* I1b - I1a .* K1b));
Zsheath_in = DD_shell .* ((K1a .* Ra - K1b .* Rb) .* I0a - (I1b .* Rb - I1a .* Ra) .* K0a);
Zsheath_mutual = DD_shell;
Zsheath_out = DD_shell .* Ra .* (K1a .* I0b + I1a .* K0b);

% Zsheath_mutual = DD_shell / r_o_s / r_i_s;
% Zsheath_mutual = 1i * omega * Mur_shell ./ (2 * pi * Ra .* Rb .* (K1a .* I1b - I1a .* K1b));

% DD_armor = 1i * omega * Mur_shell ./ (2 * pi * Ra .* Rb .* (K1a .* I1b - I1a .* K1b));
% Zsheath_mutual2 = DD_armor .* ((K1a .* Ra - K1b .* Rb) .* I0b - (I1b .* Rb - I1a .* Ra) .* K0b);
DD_armor = 1i * omega * Mur_armor ./ (2 * pi * Rd .* Re .* (K1d .* I1e - I1d .* K1e));
Zsheath_mutual2 = DD_armor .* ((K1d .* Rd - K1e .* Re) .* I0d - (I1e .* Re - I1d .* Rd) .* K0d);

% DD_armor = 1 ./ (2 * pi * Sig_armor * (K1d .* I1e - I1d .* K1e));
% Zarmor_in = gamma_armor .* DD_armor / r_i_a .* (I0d .* K1e + K0d .* I1e);
% Zarmor_out = gamma_armor .* DD_armor / r_o_a .* (I0e .* K1d + K0e .* I1d);

DD_armor = 1i * omega * Mur_armor ./ (2 * pi * Rd .* Re .* (K1d .* I1e - I1d .* K1e));
Zarmor_in = DD_armor .* ((K1d .* Rd - K1e .* Re) .* I0d - (I1e .* Re - I1d .* Rd) .* K0d);
Zarmor_mutual = DD_armor;
Zarmor_out = DD_armor .* Rd .* (K1d .* I0e + I1d .* K0e);

% Zarmor_mutual = DD_armor / r_o_a / r_i_a;
% Zarmor_mutual = 1i * omega * Mur_armor ./ (2 * pi * Rd .* Re .* (K1d .* I1e - I1d .* K1e));

Z_core_sheath_insulation = 1i * omega .* Mur_CS / 2 / pi * log(r_i_s / r_o_c);

Z_sheath_armor_insulation = 1i * omega .* Mur_SA / 2 / pi * log(r_i_a / r_o_s);

Z_armor_earth_insulation = 1i * omega .* Mur_SA / 2 / pi .* log(2*(height+gamma_armor) / r_o_a);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Impedance matrix
Z_11 = Zcore_out + Zsheath_in + Z_core_sheath_insulation;
Z_22 = Zsheath_out + Zarmor_in + Z_sheath_armor_insulation;
Z_33 = Zarmor_out + Z_armor_earth_insulation;
Z_12 = - Zsheath_mutual;
Z_21 = 0;
Z_23 = - Zarmor_mutual;
Z_32 = 0;






end