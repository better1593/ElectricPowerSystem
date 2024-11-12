function [d_pc,h_pc,E_pc,B_pc,d_cc,h_cc,E_cc,B_cc]=Parameter_VF(Zpc,Zcc,fre,n_fit,dt)

% Evaluate number of frequency points
n_fre = length(fre);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Approximate surface impedance within a cable

% Zpc
if any(real(Zpc)<0)
    Zpc_3 = reshape(Zpc-10*min(real(Zpc)),[1,1,n_fre]);
    [d_pc,h_pc,r_pc,a_pc] = VF_PolesResidues(Zpc_3, fre, n_fit);
    d_pc = d_pc + 10*min(real(Zpc));
else
    Zpc_3 = reshape(Zpc,[1,1,n_fre]);
    [d_pc,h_pc,r_pc,a_pc] = VF_PolesResidues(Zpc_3, fre, n_fit);
end
E_pc = exp( a_pc * dt );
B_pc = r_pc ./ a_pc .* ( E_pc -1 );
[Zpc_fit] = PR_VF_Test(d_pc,h_pc,r_pc,a_pc,fre);
figure(3);
subplot(2,1,1);
semilogx(fre,real(Zpc),'-k');hold on
semilogx(fre,real(Zpc_fit),'-r');hold off
grid on
xlabel('Frequency(Hz)');
ylabel('Real(Z_p_c)');
legend('Origin','Vecfit');
title('Real part of (Z_p_c)');
subplot(2,1,2);
semilogx(fre,imag(Zpc),'-k');hold on
semilogx(fre,imag(Zpc_fit),'-r');hold off
grid on
xlabel('Frequency(Hz)');
ylabel('Imag(Z_p_c)');
legend('Origin','Vecfit');
title('Imagine part of (Z_p_c)');
saveas(gcf, 'Zpc', 'jpg');
saveas(gcf, 'Zpc', 'fig');


% Zcc
if any(real(Zcc)<0)
    Zcc_3 = reshape(Zcc-10*min(real(Zcc)),[1,1,n_fre]);
    [d_cc,h_cc,r_cc,a_cc] = VF_PolesResidues(Zcc_3, fre, n_fit);
    d_cc = d_cc + 10*min(real(Zcc));
else
    Zcc_3 = reshape(Zcc,[1,1,n_fre]);
    [d_cc,h_cc,r_cc,a_cc] = VF_PolesResidues(Zcc_3, fre, n_fit);
end
E_cc = exp( a_cc * dt );
B_cc = r_cc ./ a_cc .* ( E_cc -1 );
[Zcc_fit] = PR_VF_Test(d_cc,h_cc,r_cc,a_cc,fre);
figure(4);
subplot(2,1,1);
semilogx(fre,real(Zcc),'-k');hold on
semilogx(fre,real(Zcc_fit),'-r');hold on
grid on
xlabel('Frequency(Hz)');
ylabel('Real(Z_c_c)');
legend('Origin','Vecfit');
title('Real part of (Z_c_c)');
subplot(2,1,2);
semilogx(fre,imag(Zcc),'-k');hold on
semilogx(fre,imag(Zcc_fit),'-r');hold on
grid on
xlabel('Frequency(Hz)');
ylabel('Imag(Z_c_c)');
legend('Origin','Vecfit');
title('Imagine part of (Z_c_c)');
saveas(gcf, 'Zcc', 'jpg');
saveas(gcf, 'Zcc', 'fig');



end