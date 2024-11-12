function [i_sr]=I_sr_stan_wave(GLB,current_standard_wave,curdata)
dt=GLB.dT;
Nt=GLB.Nt;
% %%%%%% I source %%%%%%%%%%%%%%%%%%%%%%%%%%
Imax=1e4;
k0=1;
% % tau1=0.454e-6;
% % tau2=143e-6;
% tau1=1e-6;
% tau2=68e-6;
% % tau1=19e-6;
% % tau2=485e-6;
tau1 = curdata.data(current_standard_wave,3)*1e-6;
tau2 = curdata.data(current_standard_wave,4)*1e-6;

n=5;

for i=1:Nt
        I_temp(i)=(i*dt/tau1)^n/(1+(i*dt/tau1)^n);
        i_sr(i)=Imax/k0*(I_temp(i))*exp(-i*dt/tau2);
end
end