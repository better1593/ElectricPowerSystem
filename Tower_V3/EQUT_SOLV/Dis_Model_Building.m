function [TowerData, SpanData, CableData]=Dis_Model_Building...
    (Tower,Span,Cable,T2Smap,GLB)
% Building discrete models of major sub-parts
% (1) Tower
% (2) Span
% (3) Cable

% (3) Generate discrete equation ofr MNA solution
NTower=GLB.NTower;                 % # of towers = 5
NSpan=GLB.NSpan;                   % # of spans = 4
NCable=GLB.NCable;                 % # of cables = 4

for ik=1:NTower
    [ TowerData(ik)] = Tower_Circuit_Discete( Tower(ik), T2Smap(ik), GLB );
end

for ik=1:NSpan
     [ SpanData(ik)] = Span_Circuit_Discrete( Span(ik), GLB );
end

for ik=1:NCable
     [ CableData(ik)] = Cable_Circuit_Discrete( Cable(ik), GLB );
end

end