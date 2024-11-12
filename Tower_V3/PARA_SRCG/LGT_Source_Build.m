 function [Tower, Span, Cable, GLB, LGT]=LGT_Source_Build...
    (Tower, Span, Cable, GLB, LGT, Tfold)
% Generating LGT source data for Tower, Span and Cable
%   Soc.ISF (dir)
%   Soc.VSF (ind) for MNA equation
%   Soc.typ = 0/1/2 (no/dir/ind);
%   Soc.pos = [T/S, id, cir_id, phase_id, cond_id, seg(node)_id]]

% (1) Read one stroke data from MC_lightning_table
index = 1;       % 2= load induced source data from TowerX/SpanX/CableX.mat

% (2) Generating source current or induced-E source on Tower/Span/Cable
              
% load i_sr.src          
load i_sr.src       
temp=1e3;
LGT.Lch.Nt=temp;
GLB.Nt=temp;
LGT.Soc.dat=i_sr(1:1e3);
LGT.Lch.curr=i_sr(1:1e3);
LGT.Soc.pos(1)=1000/2;
LGT.Soc.pos(2)=50;
LGT.Lch.pos(1)=1000/2;
LGT.Lch.pos(2)=50;

% LGT.Soc.pos(1)=0;
% LGT.Soc.pos(2)=0;
% LGT.Lch.pos(1)=0;
% LGT.Lch.pos(2)=0;
% LGT.Lch.dH=10;
% LGT.Lch.Nc=100;
% LGT.Lch.dT=1e-9;
% LGT.Lch.Nt=1e4;

for ik=1:GLB.NSpan
    Span(ik) = Span_Circuit_Source( Span(ik), GLB, LGT, index, Tfold);
end

for ik=1:GLB.NCable
    Cable(ik) = Cable_Circuit_Source(Cable(ik), GLB, LGT, index, Tfold);
end

for ik=1:GLB.NTower
    Tower(ik) = Tower_Circuit_Source( Tower(ik), GLB, LGT, index, Tfold);
end
end
