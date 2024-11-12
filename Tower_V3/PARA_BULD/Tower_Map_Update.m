function [TH_T2Xmap,TT_T2Xmap]=Tower_Map_Update(TH_T2Xmap,TT_T2Xmap,X2Tmap)
% Update TH_T2Xmap and TT_T2Xmap with Span or Cable mapping table X2Tmap

% (1) Head mapping table update
TH_T2Xmap.head = [TH_T2Xmap.head, {X2Tmap.head}];
TH_T2Xmap.hspn = TH_T2Xmap.hspn+1;
TH_T2Xmap.hsid = [TH_T2Xmap.hsid, X2Tmap.head(1,1)];

[tp, Itp]=sort(TH_T2Xmap.hsid);      % Arrange in the ascend order of sid
TH_T2Xmap.head=TH_T2Xmap.head(Itp);

% (2) Tail mapping table update
TT_T2Xmap.tail = [TT_T2Xmap.tail, {X2Tmap.tail}];
TT_T2Xmap.tspn = TT_T2Xmap.tspn+1;
TT_T2Xmap.tsid = [TT_T2Xmap.tsid, X2Tmap.tail(1,1)];

[tp, Itp]=sort(TT_T2Xmap.tsid);      % Arrange in the ascend order of sid
TT_T2Xmap.tail=TT_T2Xmap.tail(Itp);
end
