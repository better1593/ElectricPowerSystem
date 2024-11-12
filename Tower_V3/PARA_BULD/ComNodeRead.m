function [rowdel,nodecom,filename] = ComNodeRead(datatable,row,col)
% Return (1) common node.list of tower wrf to each sub-CK (???) or
%            common node.list/listdex of each sub-CK  (local name)
%        (2) row id for delection from the table
%        (3) filename of sub-CK if Tower Block only
%
% datatable (cell array): the complete Block Input File Data
% row: current row for extracting the info
% col: posi of the filename

list = [];
listdex = [];
oft = 0;
rownum = datatable{row, 6};                 % # of line for input vari
for i = 1:rownum
    rowid = row+(i-1);
    for j=2:5
        tmp0=string(datatable{rowid,j});    %  node name
        if ~ismissing(tmp0) & ~strcmp(tmp0," ")
            oft = oft + 1;
            list = [list; tmp0];
            listdex = [listdex; oft];            
        end
    end
end
rowdel = row+(0:rownum-1);                  % row id to be deleted

% read filename of sub_CK
filename = [];
if col~=0
    filename = string(datatable{row,col});
end
% common node list/listdex
nodecom.list = list;
nodecom.listdex = listdex;                  % not used for Tower block
end