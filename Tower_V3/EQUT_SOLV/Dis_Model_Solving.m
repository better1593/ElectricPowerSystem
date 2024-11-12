function [Tower, Span, Cable]=Dis_Model_Solving(Tower, Span, Cable, GLB)
% Sovle the dis-model for transient analysis (phase domain quantities)
% (1) Tower
% (2) Span
% (3) Cable
%
% Note: (1) All quanttites except HistX are in phase domain, modal domain
%           quantities  are handled in individual modules
%       (2) All SRC.ISF(dir)/VSF(ind)/pos are defined here according to 
%           X.Soc, pos = 0 (ind), >0 (dir to T/S), <0 (dir to others)
%       (3) VF are handled in individual modules via hist.Phi

NTower=GLB.NTower;                 % # of towers = 5
NSpan=GLB.NSpan;                   % # of spans = 4
NCable=GLB.NCable;                 % # of cables = 4

Nt=GLB.Nt;
dT=GLB.dT;
%--------------------------------------------------------------------------
% (1) Initialization of vectors
for ik=1:NSpan
    Nfit = Span(ik).Cal.ord;
    Ncon = Span(ik).Seg.Ncon;
    Nseg = Span(ik).Seg.Nseg;
    VpeecS(ik).head=zeros(Ncon,1);
    VpeecS(ik).tail=zeros(Ncon,1);
    IfdtdS(ik).head=zeros(Ncon,1);
    IfdtdS(ik).tail=zeros(Ncon,1);
    HistS(ik).Vn =zeros(Ncon, Nseg+1);
    HistS(ik).Ib =zeros(Ncon, Nseg);
    HistS(ik).Phi=zeros(Ncon, Nseg, Nfit);
    Span(ik).MeasR.Ib = [];
    Span(ik).MeasR.Vn = [];
end

for ik=1:NCable
    Nfit = Cable(ik).Cal.ord;
    Ncon = Cable(ik).Seg.Ncon;
    Npha = Cable(ik).Seg.Npha;
    Nseg = Cable(ik).Seg.Nseg;
    Lseg = Cable(ik).Seg.Lseg;
    VpeecC(ik).head=zeros(Ncon,1);          % [Va;Vc1;Vc2 ...]
    VpeecC(ik).tail=zeros(Ncon,1);
    IfdtdC(ik).head=zeros(Ncon,1);          % [Ia;Ic1;Ic2 ...]
    IfdtdC(ik).tail=zeros(Ncon,1);
    HistC(ik).Ic_m = zeros(Npha, Nseg);
    HistC(ik).Vc_m = zeros(Npha, Nseg+1);
    HistC(ik).Ia1_p = zeros(1, Nseg);
    HistC(ik).Ia2_p = zeros(1, Nseg);
    HistC(ik).Va_p = zeros(1, Nseg+1);
    HistC(ik).Phi_c = zeros(Npha, Nseg, Nfit);
    HistC(ik).Phi_ca = zeros(Npha, Nseg, Nfit);                
    HistC(ik).Phi_ac = zeros(1, Nseg, Nfit); 
    HistC(ik).Phi_a = zeros(1, Nseg, Nfit);    
    Cable(ik).MeasR.Ib = [];
    Cable(ik).MeasR.Vn = [];
end

for ik=1:NTower
    Nfit = Tower(ik).Cal.ord;
    Nb   = Tower(ik).Bran.num(1);                  % wire branch #
    Nn   = Tower(ik).Node.num(1);                  % total Node #
    Nbw  = Tower(ik).Bran.num(2)+Tower(ik).Bran.num(3);
    Nbms  = length(Tower(ik).Cal.Isef); % measurment branch # (OHL con. #)
    Nbmc  = length(Tower(ik).Cal.Isefc);% measurment branch # (CAB con. #)

    HistT(ik).Vn =zeros(Nn,1);
    HistT(ik).Ib1=zeros(Nb,1);
    HistT(ik).Ib2=zeros(Nbm,1);
    HistT(ik).Ib3=zeros(Nbc,1);
    HistT(ik).Veqf=zeros(Nbw,Nfit);
    Tower(ik).MeasR.Ib = [];
    Tower(ik).MeasR.Vn = [];
    Tower(ik).MeasR.Pw = [];
    Tower(ik).MeasR.En = [];
end

%--------------------------------------------------------------------------
% (Part B) Calculate fdtd results per time step  
for ik=1:Nt  
% (B1) Cables: Input (node.Vn -> bran.Vn), Output (bran.Ib -> node.Ib)
    for jk=1:NCable
        % Assign spatial vectors: Vs or Is in the span 
        Soc = Cable(jk).Soc;
        SRC.pos = -1;                           % direct to others
        if Soc.pos(1)==0                        % indirect
            SRC.VSF = Soc.dat(ik,:);
            SRC.pos = 0;
        end                
        Cable(jk).SRC = SRC;
        
        % Convert node voltage into bran voltage for cores, not armor
        VpeecC.head(2:end) = VpeecC(2:end).head - VpeecC(1).head;        
        VpeecC.tail(2:end) = VpeecC(2:end).tail - VpeecC(1).tail;        

        [HistC(jk),IfdtdC(jk), MeasR] = Cable_Circuit_Sol(Cable(jk), ...
                                                  HistC(jk), VpeecC(jk));        
        Cable(jk).MeasR.Ib = [Cable(jk).MeasR.Ib; MeasR.Ib ];
        Cable(jk).MeasR.Vn = [Cable(jk).MeasR.Vn; MeasR.Vn ];
        
        % Convert bran current into node current for cores, not armor
        IfdtdC(jk).head(1) =IfdtdC(jk).head(1)-sum(IfdtdC(jk).head(2:end));
        IfdtdC(jk).tail(1) =IfdtdC(jk).tail(1)-sum(IfdtdC(jk).tail(2:end));        
    end
    
% (B2) Spans: Inout (Node votlage), Outout (Node current), 
    for jk=1:NSpan
        % Assign spacital vectors: Vs or Is in the span 
        Ncon = Cable(ik).Seg.Ncon;
        Nseg = Cable(ik).Seg.Nseg;
        Soc = Span(jk).Soc;
        SRC.pos = -1;                           % direct to others
        if Soc.pos(1)==2                        % direct to a span
            if (Soc.pos(2)==Span(jk).ID)
                SRC.pos = Soc.pos(6);           % Seg id  
                SRC.ISF = zeros(Ncon,1);
                SRC.ISF(Soc.pos(5))=Soc.dat(ik);% Con id 
            end
        elseif Soc.pos(1)==0                    % indirect 
            SRC.pos = 0;                        % indirect 
            SRC.VSF=Soc.dat(ik,:);
            SRC.VSF = reshape(SRC.VSF,[Ncon,Nseg]);           
        end
        Span(jk).SRC = SRC;

        % Perform one interation
        [HistS(jk),IfdtdS(jk), MeasR] = Span_Circuit_Sol(Span(jk),...
                                               HistS(jk),  VpeecS(jk));                
        Span(jk).MeasR.Ib = [Span(jk).MeasR.Ib; MeasR.Ib ];
        Span(jk).MeasR.Vn = [Span(jk).MeasR.Vn; MeasR.Vn ];
    end
    
% (B3)  Tower: Input (node current) and Output (node current)
    for jk=1:NTower
        SRC.ISF1 = 0;
        SRC.ISF2 = 0;
        SRC.ISF3 = 0;
        SRC.VSF = 0;
       
% (3a1) Provide spatial source vectors: ISF and VSF (lightning source)     
        Nb=Tower(ik).Bran.num(1);               % wire branch #
        Soc=Tower(jk).Soc;
        SRC.pos = -1;                           % direct to others
        if Soc.pos(1)==1                        % direct to tower
            if (Soc.pos(2)==Tower(jk).ID)
                SRC.ISF1 = zeros(Nb,1);
                SRC.pos = Soc.pos(6);           % injected node id                     
                SRC.ISF1(SRC.pos) = Soc.dat(1,ik);
            end
        elseif Soc.pos(1)==0                    % indirect 
            Nbw = size(Soc.dat,1);              % wire branch # (air + gnd)        
            SRC.VSF(1:Nbw,1)=Soc.dat(:,ik);
            SRC.pos = 0;                        
        end
        
% (3a2) Convert span data into tower data
        T2Smap = Tower(jk).T2Smap;
        tspn=T2Smap.tspn;
        Ipeec.tail=[];
        for kk=1:tspn
            sid=T2Smap.tsid(kk);
            Ipeec.tail=[Ipeec.tail; IfdtdS(sid).tail];
        end
        
        hspn = T2Smap.hspn;
        Ipeec.head = [];                      % source currnet to tower
        for kk=1:hspn
            sid=T2Smap.hsid(kk);        % span id
            Ipeec.head=[Ipeec.head; IfdtdS(sid).head];
        end
               
        SRC.ISF2 = [];
        if ~isempty(Cal.Isef)
            SRC.ISF2=Cal.Isef.*[Ipeec.tail; Ipeec.head];
        end
        
% (3a3) Convert cable data into tower data
        T2Cmap = Tower(jk).T2Cmap;
        tspn=T2Cmap.tspn;
        Ipeec.tail=[];
        for kk=1:tspn
            sid=T2Cmap.tsid(kk);
            Ipeec.tail=[Ipeec.tail; IfdtdC(sid).tail];
        end
        
        hspn = T2Cmap.hspn;
        Ipeec.head = [];                      % source currnet to tower
        for kk=1:hspn
            sid=T2Cmap.hsid(kk);        % span id
            Ipeec.head=[Ipeec.head; IfdtdC(sid).head];
        end
        
        SRC.ISF3 = [];
        if ~isempty(Cal.Isef)
            SRC.ISF3=Cal.Isef.*[Ipeec.tail; Ipeec.head];
        end     
        
% (3b) Perform the iteration
        Tower(jk).SRC = SRC;
        [HistT(jk), MeasR] = Tower_Circuit_Sol(Tower(jk), HistT(jk));
        Tower(jk).MeasR.Ib = [Tower(jk).MeasR.Ib; MeasR.Ib1];
        Tower(jk).MeasR.Vn = [Tower(jk).MeasR.Vn; MeasR.Vn];        
        Tower(jk).MeasR.Pw = [Tower(jk).MeasR.Vn; MeasR.Pw];        
        Tower(jk).MeasR.En = Tower(jk).MeasR.Vn + MeasR.En;        
       
% (3c1) Convert Tower node voltages into Span terminal voltages     
        hspn = T2Smap.hspn;
        for kk=1:hspn
            sid=T2Smap.hsid(kk);                % Span id (head)
            map=T2Smap.head{kk};
            if hspn>1
                map=map{1};
            end
            tmp=map(2:end,2);                   % V node in Tower
            VpeecS(sid).head=HistT(jk).Vn(tmp);
        end       
        tspn = T2Smap.tspn;
        for kk=1:tspn
            sid=T2Smap.tsid(kk);            % Span id (tail)
            map=T2Smap.tail{kk};
            if tspn>1
                map=map{1};
            end
            tmp=map(2:end,2);                   % V node in Tower
            VpeecS(sid).tail=HistT(jk).Vn(tmp);
        end 
        
% (3c2) Convert Tower node voltages into Cable terminal voltages     
        hspn = T2Cmap.hspn;
        for kk=1:hspn
            sid=T2Cmap.hsid(kk);                % Span id (head)
            map=T2Cmap.head{kk};
            if hspn>1
                map=map{1};
            end
            tmp=map(2:end,2);                   % V node in Tower
            VpeecC(sid).head=HistT(jk).Vn(tmp);
        end       
        tspn = T2Cmap.tspn;
        for kk=1:tspn
            sid=T2Cmap.tsid(kk);            % Span id (tail)
            map=T2Cmap.tail{kk};
            if tspn>1
                map=map{1};
            end
            tmp=map(2:end,2);                   % V node in Tower
            VpeecC(sid).tail=HistT(jk).Vn(tmp);
        end               
    end   
end
end