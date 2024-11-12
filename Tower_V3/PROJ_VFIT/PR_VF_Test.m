function [Z_fit]=PR_VF_Test(d,h,r,a,fre)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test the approximation performance by Vector fitting tool
% d : DC component
% h : Differential component
% r : Residues
% a : Poles
% Z_fit : Approximated data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_fre = length(fre);
N_fit = length(r);

s = 1j * 2 * pi * fre;
Z_fit = ones(1,N_fre) * d + s * h;

for i = 1:N_fit
    Z_fit = Z_fit + repmat(r(i),1,N_fre)./(s-a(i));
end

end