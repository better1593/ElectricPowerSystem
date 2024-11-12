function [R0, L0, Rn, Ln, Zfit] = vecfit_kernel_Z_Ding(Zi, f0, Nfit, vf_mod)
%% Using Vector Fitting Toolkit
% H : (n,n,Ns) 3D matrix holding the H-samples (Ycab)
% s : (1,Ns) vector holding the frequency samples, s= jw? [rad/sec]. 
% poles: (1,N)  vector holding the initial poles (manual specification of 
%         initial poles). Use optsfor automated specification. 
% SERis a structure with the model on pole-residue form. For a model
%      with nports and N residue matrices, the dimensions are
% SER.poles: (1,N) 
% SER.R: (n,n,N)  (residue matrices)
% SER.D: (n,n) 
% SER.E: (n,n) 

if length(f0) < 2
    disp('Parameters for Vector Fitting MUST be Multi-Freqency', ...
        'Vector Fitting Error');
    R0 = real(Zi);
    L0 = imag(Zi)/(2*pi*f0);
    return;
end

if nargin == 4
    VFopts.asymp = vf_mod;
else
    VFopts.asymp = 3;
end


s = 1j*2*pi*f0;

VFopts.plot = 0;            % disable plotting (0) enable plotting (1)

VFopts.N = Nfit;
%opts.asymp = 3;
%opts.poletype = 'logcmplx';
VFopts.Niter1 = 10;
VFopts.Niter2 = 5;
%opts.weightparam = 4;

poles = []; %[] initial poles are automatically generated as defined by opts.startpoleflag
SER = VFdriver_Ding(Zi,s,poles,VFopts);

% SER :The perturbed model, on pole residue form and on state space form.
% Yfit : (n,n,Ns)3D matrix holding the Y-samples (or S-samples) of the 
%         perturbed model (at freq. s) 

RFopts.Niter_in = 5;
[SER, Zfit] = RPdriver_Ding(SER,s,RFopts); 

R0 = SER.D;
L0 = SER.E;

Nc = size(Zi,1);
% if Nc == 1
%     Ln = zeros(1,VFopts.N);
%     Rn = zeros(1,VFopts.N);
%     for ik = 1:VFopts.N
%         Rn(ik) = SER.R(:,:,ik)/SER.poles(ik);
%         Ln(ik) = -1/SER.poles(ik)*Rn(ik);
%     end
% elseif Nc >1
    Ln = zeros(Nc,Nc,VFopts.N);
    Rn = zeros(Nc,Nc,VFopts.N);
    for ik = 1:VFopts.N
        Rn(:,:,ik) = SER.R(:,:,ik)/SER.poles(ik);
        Ln(:,:,ik) = -1/SER.poles(ik)*Rn(:,:,ik);
    end
% end

    % SER  Structure holding the rational model, as produced by VFdriveror RPdriver
    
    
end


