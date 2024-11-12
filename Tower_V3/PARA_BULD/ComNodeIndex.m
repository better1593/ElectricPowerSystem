function [Data,Node,NodeCom]=ComNodeIndex(Data0,Node,Mode)
% Return (1) Data0 with main body of the Wire/Lump elements
%        (2) Common node info. (Node.com,Node.comdex, tower info if any)
%        (3) Ground node info. (Node.gnd,Node.gndex)
%        (4) Name of the model as the suffix of node/bran names
%        Note: "/" '0' means this is an empty node (to be deleted)
% Mode = 1 (1) and (2),
% Mode = 2 (1), (2) and (3)

% (0) Initializtion
Data = Data0;
str0 ='TOWER';
if Mode == 1
    str0 = 'LUMP';
end
NodeCom.lista = string([]);
NodeCom.listdexa = [];
NodeCom.listg = string([]);
NodeCom.listdexg = [];
Node.info = [];
NodeCom.blok = zeros(1,6);                      % block flag

% (1) Common node, datatable updating, and tower info. if any
oft = 1;                                        % ID of common nodes
for i = 1:size(Data, 1)                         % 循环遍历数据表   
    firstColumnValue = string(Data{i, 1});        % 提取第一列的值
    switch firstColumnValue
        case str0
            n1 = Data{i, 6};               % # of line for input vari
            for k=1:n1
                for j=2:5
                    tmp2=string(Data{i + k - 1, j});%  node name
                    if ~strcmp(tmp2,"")
                        NodeCom.lista = [NodeCom.lista; tmp2];
                        NodeCom.listdexa = [NodeCom.listdexa; oft];
                        oft = oft + 1;
                    end
                end
            end
        case 'END'
            n3 = i;
            NodeCom.app = string(Data0{i,2});       % node/bran name appended
            Node.app = string(Data0{i,2});          % Updating node.app
        case 'INFO'
            Node.info = Data(i,2:10);               % Updating Node.info
            NodeCom.info = Data(i,2:10);           % Updating Node.info
            n2 = i;
        case 'INSU'
            NodeCom.blok(2)=1;
            n0 = Data{i, 6};               % # of line for input vari
            for k=1:n0
                for j=2:5
                    tmp2=string(Data{i + k - 1, j});%  node name
                    if ~strcmp(tmp2,"")
                        NodeCom.lista = [NodeCom.lista; tmp2];
                        NodeCom.listdexa = [NodeCom.listdexa; oft];
                        oft = oft + 1;
                    end
                end
            end
        case 'SARR'
            NodeCom.blok(3)=1;
        case 'TXFM'
            NodeCom.blok(4)=1;
       case 'GRID'
            NodeCom.blok(5)=1;
       case 'OTHS'
            NodeCom.blok(6)=1;
    end
end
if ~isempty(Node.info)
    Data([1:n1 n2 n3],:) = [];                  % delete common node line
else
    Data([1:n1 n3],:) = [];                     % delete common node line    
end
Ncom = size(NodeCom.lista,1);                   % # of common nodes
Node.com(1:Ncom,2) = NodeCom.lista;
Node.comdex(1:Ncom,2) = NodeCom.listdexa;
clear n1 n2 n3 tmp1 tmp2

if Mode == 1
    return;
end
%--------------------------------------------------------------------------

% (2) Ground connection info. (Air-Gnd pair)
oftg = 1;                                   % gund node
for i = 1:size(Data, 1)                         % 循环遍历数据表    
    firstColumnValue = Data{i, 1};              % 提取第一列的值
    if strcmp(firstColumnValue{1},'AGpair')
        n2 = Data{i, 6};                        % # of line for input vari
        for k=1:2:n2                            % every two lines
            for j=2:5
                tmp2=string(Data{i+k-1,j});     % node name in air               
                if ~strcmp(tmp2,"")
                    tmp1=string(Data{i+k,j});   % node name in gnd
                    Node.gnd= [Node.gnd; [tmp2 tmp1]];
                    tmp3 = find(Node.com(:,2)==tmp2);
                    oftg0 = 0;
                    if ~strcmp(tmp1,"")         % gnd node
                        oftg0 = oftg;
                        oftg = oftg + 1;
                    end                                               
                    if isempty(tmp3)            % air node
                        Node.gnddex = [Node.gnddex; [oft oftg0]];  %[n1 n2]
                        oft = oft + 1;
                    else
                        Node.gnddex = [Node.gnddex; [tmp3 oftg0]]; %[n1 n2]
                    end                       
                end
            end
        end
    end
end
if ~isempty(Node.gnd)
    NodeCom.lista = [NodeCom.lista; Node.gnd(:,1)];
    NodeCom.listdexa = [NodeCom.listdexa; Node.gnddex(:,1)];
    NodeCom.lista = unique(NodeCom.lista,'rows','stable');
    NodeCom.listdexa = unique(NodeCom.listdexa,'rows','stable');
    NodeCom.listg = Node.gnd(:,2);                  % gnd node
    NodeCom.listdexg = Node.gnddex(:,2);
    
    Data(1:n2,:)=[];       % delete gnd_node linesne
end                         
end