function out = AssignSuffix(NameList,app)
% Attach suffix to the namelist with the size of (Nrow x Mcol)

if isempty(NameList)
    out = NameList;
    return;
end

% NameList = rmmissing(NameList0);
[Nrow, Ncol] = size(NameList);
out = strings(Nrow, Ncol);
for i = 1:Nrow
    for j = 1:Ncol
        tmp=string(NameList(i,j));                    % !!! modified
        if ~ismissing(tmp)
            if tmp~="" && tmp~=" " && tmp~="0" && tmp~="NaN"
                out(i,j) = tmp + app;
            end
        end
    end
end
end