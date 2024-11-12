function Span = Span_Circuit_Source(Span, GLB, LGT, index, Tfold) 
% Generate lightning source parameters due to direct/indirect lightning
%   Update Span.Soc.typ/dat/pos, excluding flash
%          
%   Initilize Span.Soc.ISF/VSF
%   Pos: [x, y, 0, 0, 0) for IND
%        [T/S/0, ID, CirID, PhaseID, CondID, Seg(node)] for DIR

ID = Span.ID;
OHLP = Span.OHLP;

LSoc = LGT.Soc;
Soc.typ=LSoc.typ;
Soc.pos=LSoc.pos;
Soc.dat=[];                 % store all source dat
icur = LGT.Lch.curr;        % lightning stroke current

Seg = Span.Seg;
Nseg = Seg.Nseg;
Ncon = Seg.Ncon;           % # of total conductors

str=Tfold + "\Span"+num2str(ID)+".mat";   
% load('data.mat'); % include (a) data_SRC.direct (b) data_SRC.induced

if LSoc.typ==1
    if (LSoc.pos(1)~=2) || (LSoc.pos(2)~=Span.ID)
        Soc.pos=[];
        Span.Soc=Soc;
        return;
    else
        Soc.dat = icur;
    end
elseif LSoc.typ == 0   
    if index==2
        dat=load(str);
        Soc.dat=dat.dat;
        Span.Soc=Soc;
        return;
    end

    Pstr=OHLP(1:end,1:3);               % position of insulator on a tower
    Pend=OHLP(1:end,4:6);               % position of insulator on a tower
        
    delta=(Pend-Pstr)/Nseg;
    x0=zeros(Ncon,Nseg+1);
    y0=zeros(Ncon,Nseg+1);
    z0=zeros(Ncon,Nseg+1);
    for ik=1:Ncon
        x0(ik,:)=Pstr(ik,1)+(0:Nseg)*delta(ik,1);
        y0(ik,:)=Pstr(ik,2)+(0:Nseg)*delta(ik,2);
        z0(ik,:)=Pstr(ik,3)+(0:Nseg)*delta(ik,3);
    end
    Lne.x1=x0(:,1:end-1);   Lne.x2=x0(:,2:end);
    Lne.y1=y0(:,1:end-1);   Lne.y2=y0(:,2:end);
    Lne.z1=z0(:,1:end-1);   Lne.z2=z0(:,2:end);
    Lne.x1=reshape(Lne.x1,[],1);
    Lne.x2=reshape(Lne.x2,[],1);
    Lne.y1=reshape(Lne.y1,[],1);
    Lne.y2=reshape(Lne.y2,[],1);
    Lne.z1=reshape(Lne.z1,[],1);
    Lne.z2=reshape(Lne.z2,[],1);
    
    Lne.tran.pt_start=[Lne.x1 Lne.y1 Lne.z1];
    Lne.tran.pt_end  =[Lne.x2 Lne.y2 Lne.z2];
    slen=sqrt(sum(delta.*delta,2));
    Lne.tran.L=repmat(slen,Nseg,1);
 
    [Er_T, Ez_T] = E_Cal(LGT, Lne);     
    
    GND = Span.GND;
    [U_TL]= Cor_Lossy_ground(GLB, LGT, GND, Lne,Er_T,Ez_T);
    dat=U_TL;                               % format=[(t1 t2..) (p1 p2..)]
    save(str,'dat');
    Soc.dat = dat;
end
Span.Soc=Soc;   
end