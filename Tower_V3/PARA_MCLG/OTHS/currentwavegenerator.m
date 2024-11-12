function out = currentwavegenerator(para,t)
% t = 0:1e-6:1e-4;  

tn=para(1);A=para(2);B=para(3);n=para(4);I1=para(5);t1=para(6);
I2=para(7);t2=para(8);Ipi=para(9);Ipc=para(10);

out=((t<= tn).* (A.*t+B.*(t.^n)) + (t>tn) .* (I1.*exp(-(t-tn)./t1) - I2.*exp(-(t-tn)./t2))).*(Ipi./Ipc);
end