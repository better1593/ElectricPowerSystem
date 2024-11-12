%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Materials of conductors 
Sig_A = 1e6;
Sig_P = 1e6;
mur_A = 1;
mur_P = 1;
% Materials of insulators
eps_CC = 1;
mur_CC = 1;
% Geometry size of conductors
d_i = 0.003;
d_j = 0.003;
r_P = 0.005;
r_d = 0.0009;
r_e = 0.001;
% Angle between two cables
Theta = 180;
% Frequency points of VFT
fre = [1:10:90 1e2:1e2:9e2 1e3:1e3:9e3 1e4:1e4:9e4 1e5:1e5:9e5 1e6:1e6:9e6 1e7:1e7:9e7 1e8:1e8:9e8];
n_fit = 5;
% Time step
dt = 1.9245e-11;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bessel initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu0   = 4*pi*1e-7;
Mur_P = mu0 * mur_P;
Mur_A = mu0 * mur_A;
eps0  = 8.854187818e-12;
Eps   = eps0 * 1;
omega = 2 * pi * fre;                          
gamma_P = sqrt(1i * Mur_P * omega .* (Sig_P + 1i * omega * Eps));
gamma_A = sqrt(1i * Mur_A * omega .* (Sig_A + 1i * omega * Eps));
R_P = r_P * gamma_P;
R_e = r_e * gamma_A;
R_d = r_d * gamma_A;
I0e = besseli(0,R_e);
I1e = besseli(1,R_e);
I1d = besseli(1,R_d);
K0e = besselk(0,R_e);
K1d = besselk(1,R_d);
K1e = besselk(1,R_e);
K0P = besselk(0,R_P);
K1P = besselk(1,R_P);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outer surface impedance of cable
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DD_armor = 1i * omega * Mur_A ./ (2 * pi * R_d .* R_e .* (K1d .* I1e - I1d .* K1e));
Zc = DD_armor .* R_d .* (K1d .* I0e + I1d .* K0e);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mutual impedance between pipe and cables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Zpc = 1i * omega * mu0 / 2 / pi * log(r_P/r_e*(1-(d_i/r_P)^2));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inner surface impedance of pipe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Zp = 1i * omega * Mur_P / 2 / pi .* K0P ./ gamma_P/r_P./K1P;
Zp_0 = Zp;
for N = 1:15
    Zp = Zp + 1i * omega * Mur_P/pi .* (d_i*d_i/r_P/r_P)^N .* besselk(N,R_P)...
        ./ (N*mur_P*besselk(N,R_P)-gamma_P*r_P.*(-(N*besselk(N,R_P))/R_P - besselk(N-1,R_P)));
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mutual impedance between pipes 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Theta = Theta / 180 * pi;
Zcc = 1i * omega * mu0 / 2 / pi .* (log( r_P / sqrt(d_i^2 + d_j^2 - 2*d_i*d_i*cos(Theta)) ) + mur_P * K0P ./ gamma_P/r_P/K1P);
Zcc_0 = Zcc;
for N = 1:15
    Zcc = Zcc + 1i * omega * mu0 / 2 / pi .* (d_i*d_j/r_P/r_P)^N * cos(N*Theta) ...
        .* (2*mur_P * besselk(N,R_P) ./ (N*mur_P*besselk(N,R_P)-gamma_P*r_P.*(- N*besselk(N,R_P)/R_P - besselk(N-1,R_P))) - 1/N);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Total impedance between pipe and cables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Zpc = Zc + Zpc + Zp;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Potential coefficient matrix within pipe cable
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P_pc = 1 / 2 / pi / Eps * log(r_P/r_e*(1-(d_i/r_P)^2));
P_cc = 1 / 2 / pi / Eps .* log( r_P / sqrt(d_i^2 + d_j^2 - 2*d_i*d_i*cos(Theta)) );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot impedances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(1)
% semilogx(fre, real(Zcc./omega), '-r'); hold on
% semilogx(fre, real(Zcc_0./omega), '-*k'); hold off
% legend('15th order','0 order');
% xlabel('Frequency(Hz)');
% ylabel('Impedance(ohm)');
% title('Real part of Z_c_c');
% 
% figure(2)
% semilogx(fre, imag(Zcc./omega), '-r'); hold on
% semilogx(fre, imag(Zcc_0./omega), '-*k'); hold off
% legend('15th order','0 order');
% xlabel('Frequency(Hz)');
% ylabel('Impedance(ohm)');
% title('Imaginary part of Z_c_c');
% 
% figure(3)
% semilogx(fre, real(Zp), '-r'); hold on
% semilogx(fre, real(Zp_0), '-*k'); hold off
% legend('15th order','0 order');
% xlabel('Frequency(Hz)');
% ylabel('Impedance(ohm)');
% title('Real part of Z_p');
% 
% figure(4)
% semilogx(fre, imag(Zp), '-r'); hold on
% semilogx(fre, imag(Zp_0), '-*k'); hold off
% legend('15th order','0 order');
% xlabel('Frequency(Hz)');
% ylabel('Impedance(ohm)');
% title('Imaginary part of Z_p');
% 
% figure(5)
% semilogx(fre, real(Zpc), '-k'); 
% xlabel('Frequency(Hz)');
% ylabel('Impedance(ohm)');
% title('Real part of Z_p_c');
% 
% figure(6)
% semilogx(fre, imag(Zpc), '-k'); 
% xlabel('Frequency(Hz)');
% ylabel('Impedance(ohm)');
% title('Imaginary part of Z_p_c');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Approximate frequency dependent impedances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[d_pc,h_pc,E_pc,B_pc,d_cc,h_cc,E_cc,B_cc]=Parameter_VF(Zpc,Zcc,fre,n_fit,dt);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = zeros(2,2);
H = zeros(2,2);
P = zeros(2,2);
B = zeros(2,2,n_fit);
E = zeros(2,2,n_fit);
D(:,:) = [d_pc,d_cc;
          d_cc,d_pc];
H(:,:) = [h_pc,h_cc;
          h_cc,h_pc];
P(:,:) = [P_pc,P_cc;
          P_cc,P_pc];
for n = 1:n_fit
    B(:,:,n) = [B_pc(n),B_cc(n);
                B_cc(n),B_pc(n)];
    E(:,:,n) = [E_pc(n),E_cc(n);
                E_cc(n),E_pc(n)];
end
save('VF_data_2phases.mat','D','H','P','B','E','d_i','d_j','r_P','r_d','r_e',...
    'Sig_P','mur_P','Sig_A','mur_A','Theta','fre','fre','n_fit','dt');


