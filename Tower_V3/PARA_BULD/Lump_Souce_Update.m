function [CK_Te] = Lump_Souce_Update(CK_Te,CK_Xe,NodeX,oftn,oftb)
% Update CK_element (Vs, Is, Nle and Swh) of Tower by appending Sub_CK data
%        with offset No.: oftn and oftb
%   CK_T.Vs.dat/pos: Tower-CK element to be updated with CK_X
%   CK_X.Vs.dat/pos: Sub-CK element
%   Ncol = 1 for Is (n1), and 
%   Ncol = 3 for Vs, Nle and Swh [b0 n1 n2]

if isempty(CK_Xe.dat)
    return;
end

Ncom = size(NodeX.com,1);
Ncol = size(CK_Xe.pos,2);
if Ncol == 1
    Xref = CK_Xe.pos;
elseif Ncol == 3
    Xref = CK_Xe.pos(:,2:3);
end
Locelem= [] ; 
Valelem = [];

% (1) Record the postion of common nodes in Sub_CK list
for i = 1:Ncom
    num1 = NodeX.comdex(i,1);                   % Global id
    num2 = NodeX.comdex(i,2);                   % Local id
    tmp0 = find(Xref==num2);                    % Vs
    if ~isempty(tmp0)
        Locelem = [Locelem; tmp0];              % store position for update
        Valelem = [Valelem; tmp0*0+num1];    % store value for update
    end
end

% (2)  Update node id in CK_Xe.pos
Xref = Xref + oftn;
Xref(Locelem) = Valelem;            % Replace Com_node id with GLB_node id

% (3) Update GLB CK elements (CK_Te.dat and pos)
if Ncol ==1
    if isempty(CK_Te)
        CK_Te.pos = Xref;
        CK_Te.dat = CK_Xe.dat;
    else
        CK_Te.pos = [CK_Te.pos; Xref];          % Is
        CK_Te.dat = [CK_Te.dat; CK_Xe.dat];
    end
elseif Ncol == 3
    Xban = CK_Xe.pos(:,1) + oftb;
    if isempty(CK_Te)
        CK_Te.pos = [Xban Xref];                % Vs Nle and Swh
        CK_Te.dat = CK_Xe.dat;
    else        
        CK_Te.pos = [CK_Te.pos; [Xban Xref]];   % Vs Nle and Swh
        CK_Te.dat = [CK_Te.dat; CK_Xe.dat];
    end
end  
end