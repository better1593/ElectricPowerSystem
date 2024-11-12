function [d, h, r, a] = VF_PolesResidues(Zi, f0, Nfit)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Using Vector Fitting Toolkit
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if length(f0) < 2
    disp('Parameters for Vector Fitting MUST be Multi-Freqency', ...
        'Vector Fitting Error');
    return;
end


VFopts.asymp = 3;
s = 1j*2*pi*f0;
VFopts.plot = 0;
VFopts.N = Nfit;
%opts.asymp = 3;
%opts.poletype = 'logcmplx';
VFopts.Niter1 = 10;
VFopts.Niter2 = 5;
%opts.weightparam = 4;

poles = []; %[] initial poles are automatically generated as defined by opts.startpoleflag
SER = VFdriver(Zi,s,poles,VFopts);

% SER :The perturbed model, on pole residue form and on state space form.
% Yfit : (n,n,Ns)3D matrix holding the Y-samples (or S-samples) of the 
%         perturbed model (at freq. s) 

RFopts.Niter_in = 5;
[SER, ~] = RPdriver(SER,s,RFopts); 

r = zeros(1,Nfit);
a = zeros(1,Nfit);

d = SER.D;
h = SER.E;
for ik = 1:Nfit
    r(ik) = SER.R(:,:,ik);
    a(ik) = SER.poles(ik);
end
    
    
end


