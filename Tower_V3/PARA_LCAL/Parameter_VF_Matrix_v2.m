function VFIT=Parameter_VF_Matrix_v2(Z,fre,n_fit)
% N x N x Nf matrix impedance matrix : Z
N1 = size(Z,1);
N2 = size(Z,2);
n_fre = numel(fre);
VFIT.d = zeros(N1,N2);
VFIT.h = zeros(N1,N2);
VFIT.r = zeros(N1,N2,n_fit);
VFIT.a = zeros(N1,N2,n_fit);
VFIT.ord = n_fit;                  %  # of VFIT order

for i1 = 1:N1
    for i2 = 1:N2
        Ztmp = squeeze(Z(i1,i2,:));
        
        %    if (i1==1 && i2==3) || ((i2==1 && i1==3))
        if ~all(any(Ztmp,1))
            
        else
            if any(real(Ztmp)<0)
                Zpc_3 = reshape(Ztmp-10*min(real(Ztmp)),[1,1,n_fre]);
                [d_pc,h_pc,r_pc,a_pc] = VF_PolesResidues(Zpc_3, fre, n_fit);
                d_pc = d_pc + 10*min(real(Ztmp));
            else
                Zpc_3 = reshape(Ztmp,[1,1,n_fre]);
                [d_pc,h_pc,r_pc,a_pc] = VF_PolesResidues(Zpc_3, fre, n_fit);
            end
 
% plot --------------------------------------------------------------------
                [Zpc_fit] = PR_VF_Test(d_pc,h_pc,r_pc,a_pc,fre);
                figure();
                subplot(2,1,1);
                loglog(fre,real(Ztmp),'-k');hold on
                loglog(fre,real(Zpc_fit),'--r');hold off
                grid on
                xlabel('Frequency(Hz)');
                ylabel('Real(Z_p_c)');
                legend('Origin','Vecfit');
                title('Real part of (Z_p_c)');
                subplot(2,1,2);
                loglog(fre,imag(Ztmp),'-k');hold on
                loglog(fre,imag(Zpc_fit),'--r');hold off
                grid on
                xlabel('Frequency(Hz)');
                ylabel('Imag(Z_p_c)');
                legend('Origin','Vecfit');
                title('Imagine part of (Z_p_c)');
% plot --------------------------------------------------------------------
            
            VFIT.d(i1,i2) = d_pc;
            VFIT.h(i1,i2) = h_pc;
            VFIT.r(i1,i2,:) = r_pc;
            VFIT.a(i1,i2,:) = a_pc;
            
            %     E_pc = exp( a_pc * dt );
            %     B_pc = r_pc ./ a_pc .* ( E_pc -1 );
            %     d(i1,i2) = d_pc;
            %     h(i1,i2) = h_pc;
            %     E(i1,i2,:) = E_pc;
            %     B(i1,i2,:) = B_pc;
            %
        end
    end
end
