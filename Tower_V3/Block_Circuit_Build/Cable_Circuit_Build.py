import pandas as pd
import numpy as np
import math
import Vector_Fitting

from scipy.special import jv, iv, kv

np.seterr(divide="ignore",invalid="ignore")

class Cable_Circuit_Build():
    def __init__(self,path):
        self.path = path

    def Cable_Circuit_Build(self, CableModelName, TCom, TH_Info, TH_Node, TT_Info, TT_Node, VFIT, GLB):
        # (1) Read a complete table
        # path for excel table
        excel_path = self.path + '/' + CableModelName

        # # path for excel table
        # path = "H:\Ph.D. Python\StrongEMPPlatform\Tower\Main Program\DATA_Files"
        # excel_path = path + '/' + TowerModelName

        # num, txt, raw_data = pd.read_excel(TowerModelName)
        # reading table
        raw_data = pd.read_excel(excel_path, header=None).values

        # --------------------------------------------------------------------------

        # (2) Read general info. of a Cable
        data, Blok, Info, Nwir, GND = self.Gene_Info_Read_v2(raw_data)
        Cable = {}
        Cable["Info"] = Info
        Cable["ID"] = int(Info[9])
        Cable['Acab'] = GLB['Acab'][Cable['ID'] -1, :]
        if GND["glb"] == 1:
            Cable["GND"] = GLB["GND"]
        Blok["Seg"]["Lseg"] = GLB["slg"]  # length of a segment

        if len(Blok['Seg']['num']) < 4:
            Blok['Seg']['num'].append(GLB['slg'])  # 添加第四个元素并设置为 GLB['slg']
        else:
            Blok['Seg']['num'][3] = GLB['slg']  # 如果已存在第四个元素，更新其值为 GLB['slg']

        # (2b) Read relative info. of T_head, T_tail
        Cable["Info"][2] = TH_Info[0]  # name of head tower
        Cable["Info"][3] = TT_Info[0]  # name of tail tower
        Pole = np.array([TH_Info[4], TH_Info[5], TH_Info[6], TT_Info[4], TT_Info[5], TT_Info[6]])  # pos of towers
        Cable["Info"][4] = TH_Info[9]  # id of head tower
        Cable["Info"][5] = TT_Info[9]  # id of tail tower
        rng = 1
        TC_head = TCom["head"]
        TC_tail = TCom["tail"]
        TC_head = self.Assign_Elem_id(TH_Node, TC_head, rng)  # ID and local pos
        TC_tail = self.Assign_Elem_id(TT_Node, TC_tail, rng)  # ID and local pos

        # (3) Cable Parameters
        # (3a) Initilization of Cable parameters
        Node, Bran, Meas, Cir, Seg = self.Node_Bran_Index_Line(data, Blok, Pole)
        TC_head['list'] = TC_head['list'].reshape(-1, 1)  # 重塑为列向量
        TC_tail['list'] = TC_tail['list'].reshape(-1, 1)  # 重塑为列向量
        TC_head['listdex'] = TC_head['listdex'].reshape(-1, 1)  # 重塑为列向量
        TC_tail['listdex'] = TC_tail['listdex'].reshape(-1, 1)  # 重塑为列向量
        # 连接数组
        Node['com'] = np.hstack([Node['com'][:, [0]], TC_head['list'], Node['com'][:, [1]], TC_tail['list']])
        Node['comdex'] = np.hstack([Node['comdex'][:, [0]], TC_head['listdex'], Node['comdex'][:, [1]], TC_tail['listdex']])
        # 计算共同节点的数量
        Ncom = Node["com"].shape[0]  # # of common nodes
        pos = np.zeros((Ncom, 6))
        for i in range(Ncom):
            concatenated_row = np.concatenate((TC_head['pos'][i, :], TC_tail['pos'][i, :]))
            pos[i, :] = concatenated_row + Pole
        Node['pos'] = pos

        # (3b) Read wire parameters of a cable
        Ncab = int(Cir["num"][0, 5])  # # of line/conductors
        Seg["Ncab"] = Ncab
        OHLP = data[0:Ncab, 5:19]  # Cell: Cable data only

        # Getting cooridnates for Pole and lines
        Line = {}
        Line["pos"] = Pole  # Pole start/end positions
        Line["rad"] = OHLP[0, :6]  # wire radius and height
        Line["mat"] = OHLP[0, 8:13]  # wire mat: sigc/siga/murc/mura/epri
        Line["con"] = Blok["Seg"]["num"]  # total# core# arm# seg#

        # (3d) Mapping table (hspan/hsid/Thead tspan/tsid/Ttail)
        C2Tmap = {}
        C2Tmap["head"] = np.vstack((np.array([Cable["ID"], TH_Info[9]]), Node["comdex"][:, 0:2]))
        C2Tmap["tail"] = np.vstack((np.array([Cable["ID"], TT_Info[9]]), Node["comdex"][:, 2:4]))

        # --------------------------------------------------------------------------

        # (4) Calculate wave equation parameters (Z and Y)
        VFmod = [Info[7], Info[8]]
        Para = self.Cable_Para_Cal(Line, VFIT, GND)

        # (5) Output
        Cable["Cir"] = Cir
        Cable["Seg"] = Seg
        Cable["Pole"] = Pole
        Cable["Line"] = Line
        Cable["Para"] = Para
        # Cable["Cal"] = Cal
        # Cable['Ht'] =  Ht
        Cable["Node"] = Node
        Cable["Bran"] = Bran
        Cable["Meas"] = Meas
        Cable["C2Tmap"] = C2Tmap
        Cable["Soc"] = []

        return Cable, C2Tmap


    def Gene_Info_Read_v2(self,data):
        # (1) Remove comment lines starting with
        # cell stru
        row_len = len(data)         # change from Nrow = len(data)
        row_del = []
        for i in range(row_len):
            if isinstance(data[i][0], float):
                pass
            else:
                str_val = data[i][0]
                if str_val[0] == '%':
                    row_del.append(i)

        # Create a copy of the data without comment lines
        data_copy = data.copy()     # cell array

        # for i in reversed(row_del):
        data_copy = np.delete(data_copy, row_del, axis = 0)     # delete raw
        DATA = data_copy

        BTYPE = DATA[0][0]  # Block Type (TOWER/LUMP)
        COL = 0             # col for sub_CK file name

        if BTYPE == 'TOWER':
            COL = 24

        # (2) Read  general info. about T/S/C and L about com_node (name and/or id)
        Info = []
        GND = {}
        Cir = {}
        blokflag = [0] * 7
        blokname = []
        bloktow = []
        blokins = []
        bloksar = []
        bloktxf = []
        blokgrd = []
        blokoth = []
        bloka2g = {"list": [], "listdex": []}       # air-2-gnd bridge
        bloklup = []
        sys_name = ""

        nair = 0
        ngnd = 0
        nmea = 0

        row_del = []
        Row_len = len(DATA)                         # of lines with [b0, n1 n2] names

        for i in range(Row_len):  # read data_para table
            first_column_value = str(DATA[i][0])  # check the key word

            if first_column_value == "TOWER":
                row_del.append(i)
                blokflag[0:10] = [float(x) for x in DATA[i][1:11]]  # T: sub_CK flag (char)
                blokname = ["WIRE", "", "", "", "", "", "", "", "", ""]  # with a air-gnd bridge
            elif first_column_value == "SPAN":
                row_del.extend([i, i + 1])
                Cir['num'] = np.array(DATA[i:i + 2, 1:7], dtype=float)  # S: Cir.dat (2 lines)
            elif first_column_value == "CABLE":
                row_del.extend([i, i + 1])
                Cir['num'] = np.array(DATA[i:i + 2, 1:7], dtype=float)  # C: Cir.dat (2 lines)
            elif first_column_value == "INFO":
                row_del.append(i)
                Info = DATA[i][1:13]  # T/S/C Info (cell array)
            elif first_column_value == "GND":
                row_del.append(i)  # T/S/C soil model
                GND["glb"] = float(DATA[i][1])  # 1 = GLB_GND_data
                GND["sig"] = float(DATA[i][5])
                GND["mur"] = float(DATA[i][6])
                GND["epr"] = float(DATA[i][7])
                GND["gnd"] = float(DATA[i][8])  # gnd model: 0 or 1 or 2
            elif first_column_value == "A2GB":  # T: air-2-gnd conn pair
                blokflag[0] = 2  # + 1 = AirW only, 2 = AirW+GndW
                blokname[0] = "Wire+A2G"
                bloka2g["list"] = np.zeros((0, 3))
                row_num = int(DATA[i][5])  # line # for input vari
                oft = 0
                for j in range(1, row_num + 1, 2):
                    row_id = i + (j - 1)
                    for k in range(1, 6):
                        tmp1 = str(DATA[row_id][k])  # Air node name
                        tmp2 = str(DATA[row_id + 1][k])  # Gnd node name
                        if tmp1.strip() != "" and tmp1 != " " and tmp1 != 'nan' and tmp2 != 'nan':
                            oft += 1
                            tmp0 = "S" + f"{oft:02d}"
                            bloka2g["list"] = np.vstack([bloka2g['list'], [tmp0, tmp1, tmp2]])
                bloka2g["listdex"] = []
                row_del.extend(range(i, i + row_num))
            elif first_column_value == "INSU":
                row_ins, blokins, blokname[1] = self.Com_Node_Read(DATA, i, COL)
                row_del.extend(row_ins)
            elif first_column_value == "SARR":
                row_sar, bloksar, blokname[2] = self.Com_Node_Read(DATA, i, COL)
                row_del.extend(row_sar)
            elif first_column_value == "TXFM":
                row_txf, bloktxf, blokname[3] = self.Com_Node_Read(DATA, i, COL)
                row_del.extend(row_txf)
            elif first_column_value == "GRID":
                row_grd, blokgrd, blokname[4] = self.Com_Node_Read(DATA, i, COL)
                row_del.extend(row_grd)
            elif first_column_value == "INVT":
                row_int, blokint, blokname[5] = self.Com_Node_Read(DATA, i, COL)
                row_del.extend(row_int)
            elif first_column_value == "INVF":
                row_inf, blokinf, blokname[6] = self.Com_Node_Read(DATA, i, COL)
                row_del.extend(row_inf)
            elif first_column_value == "MTCK":
                row_mck, blokmck, blokname[7] = self.Com_Node_Read(DATA, i, COL)
                row_del.extend(row_mck)
            elif first_column_value == "OTH1":
                row_oth1, blokoth1, blokname[8] = self.Com_Node_Read(DATA, i, COL)
                row_del.extend(row_oth1)
            elif first_column_value == "OTH2":
                row_oth2, blokoth2, blokname[9] = self.Com_Node_Read(DATA, i, COL)
                row_del.extend(row_oth2)
            elif first_column_value == "END":  # Used for loading lump model
                row_del.append(i)
                sys_name = str(DATA[i][1])  # Lump sub-CK name
            elif first_column_value == "AirW":
                nair += 1  # record air wire #
            elif first_column_value == "GndW":
                ngnd += 1  # record gnd wire #
            elif first_column_value == "Meas":  # record the meas line #
                nmea = int(DATA[i][5])
            elif first_column_value == "LUMP":  # read com_node in lump CK file
                row_lup, bloklup, not_used = self.Com_Node_Read(DATA, i, COL)
                row_del.extend(row_lup)

        # (0) Initialization
        Blok = {
            "sysname": sys_name,
            "Cir": Cir,
            "Seg": {},
            "flag": blokflag,
            "name": blokname,
            "tow": bloktow,
            "ins": blokins,
            "sar": bloksar,
            "txf": bloktxf,
            "grd": blokgrd,
            "oth": blokoth,
            "a2g": bloka2g,
            "lup": bloklup,
            "pos": {}
        }

        # Create a copy of the data without comment lines
        DATA_copy = DATA.copy()     # cell array
        # for i in reversed(row_del):
        DATA_copy = np.delete(DATA_copy, row_del, axis = 0)     # delete raw
        DATA = DATA_copy
        nrow = DATA.shape[0]                                    # of lines with [b0, n1 n2] names\

        if Cir:
            Blok["Seg"]["Ncon"] = Cir["num"][1,0]
            Blok["Seg"]["Npha"] = Cir["num"][1,0] - Cir["num"][1,1]
            Blok["Seg"]["Ngdw"] = Cir["num"][1,1]
            Blok["Seg"]["num"] = [Blok["Seg"]["Ncon"], Blok["Seg"]["Npha"], Blok["Seg"]["Ngdw"]]

        # of all wires or lump components
        Nwir = {"nair": nair, "ngnd": ngnd, "nmea": nmea, "comp": nrow - nmea}

        return DATA, Blok, Info, Nwir, GND

    def Com_Node_Read(self,datatable, row, col):
        # Return (1) common node list of tower wrf to each sub-CK (???) or common node list/listdex of each sub-CK (local name)
        # (2) row id for deletion from the table
        # (3) filename of sub-CK if Tower Block only
        # datatable (cell array): the complete Block Input File Data
        # row: current row for extracting the info
        # col: posi of the filename

        lists = []
        listdex = []
        oft = 0
        rownum = int(datatable[row][5])  # # of line for input vari
        for i in range(1, rownum + 1):
            rowid = row + (i - 1)
            for j in range(1, 5):
                tmp0 = str(datatable[rowid][j])  # node name
                if tmp0.strip() != "" and tmp0 != "nan":
                    oft += 1
                    lists.append(tmp0)
                    listdex.append(oft)
        rowdel = list(range(row, row + rownum))  # row id to be deleted

        # read filename of sub_CK
        filename = ""
        if col != 0:
            filename = str(datatable[row][col - 1])

        # common node list/listdex
        nodecom = {"list": lists, "listdex": listdex}  # not used for Tower block

        return rowdel, nodecom, filename

    def Assign_Elem_id(self,ref_elem, elem, rng):
        # rng在现在的程序中仅为1，因此酌情修改，带＃的为原来的方程
        # Update Elem.listdex[:, rng] according to Elem.list with Ref_Elem.list
        # Elem = Note or Bran, rng = col # to be updated
        # Ref_Elem = Note.list[:, 1] or Bran.list[:, 1]

        elem["list"] = np.array(elem["list"])
        Nrow = elem["list"].shape[0]

        if "listdex" in elem:
            elem["listdex"] = np.array(elem["listdex"])
        else:
            elem["listdex"] = np.empty((1, Nrow))

        elem["pos"] = np.zeros((Nrow, 3))
        if isinstance(rng, int):
            Ncol = 1
        else:
            Ncol = len(rng)


        if elem["list"].size > 0:
            if elem["list"].ndim == 1:
                elem["list"] = elem["list"].reshape((1, Nrow))
        if elem["listdex"].size > 0:
            if elem["listdex"].ndim == 1:
                elem["listdex"] = elem["listdex"].reshape((1, Nrow))

        for i in range(Nrow):
            for j in range(Ncol):
                if isinstance(rng, int):
                    id = rng - 1
                    elem_name = elem["list"][id, i]
                    row_id = np.where(ref_elem["list"] == elem_name)[0]
                else:
                    id = rng[j]
                    elem_name = elem["list"][i, id]
                    row_id = np.where(ref_elem["list"] == elem_name)[0]

                if row_id.size > 0:
                    # num_elem = elem["pos"].shape[0]
                    if ref_elem["listdex"].shape[0] > 0:
                        if elem['listdex'].shape[0] > 1:
                            elem["listdex"][i, id] = ref_elem["listdex"][row_id[0]]
                        else:
                            elem["listdex"][id, i] = ref_elem["listdex"][row_id[0], id]

                    if ref_elem["pos"] is not None and ref_elem["pos"].size > 0:

                        # elem["pos"] = elem["pos"].reshape((num_ref_elem, 3))
                        elem["pos"][i, 0:3] = ref_elem["pos"][row_id[0], 0:3]

        return elem

    def Node_Bran_Index_Line(self,data, Blok, Pole):
        # (0) Initialization
        Seg = Blok['Seg']
        Cir = Blok['Cir']
        Lseg = Seg['Lseg']  # length of a segment
        Nrow = data.shape[0]
        Ncon = Seg['num'][0]  # total conductors of the line
        Ngdw = Seg['num'][2]  # total ground wire or armor

        # Node = {"list": np.array([], dtype=str), "listdex": np.array([], dtype=int),
        #         "com": np.array([], dtype=float), "comdex":np.array([],dtype=float),
        #         "num":np.array([],dtype=float)}  # cooridnates of nodes
        #
        # Bran = {"list": np.array([], dtype=str), "listdex": np.array([], dtype=int),
        #         "pos": np.array([], dtype=float),"num":np.array([],dtype=float)}

        # (1) Read pole data, obtain Nseg
        pos1 = np.array(Pole[0:3])
        pos2 = np.array(Pole[3:6])
        pos1 = pos1.reshape(1,3)
        pos2 = pos2.reshape(1,3)
        ds = pos2 - pos1
        ds2 = ds * ds
        ds2 = np.array(ds2, dtype=float)
        dis = np.sqrt(np.sum(ds2, axis=1))
        Nseg = int(np.round(dis / Lseg).astype(int))

        ## length of a segment
        if len(Blok['Seg']['num']) < 6:
            Seg['num'].append(Nseg) # Seg['num'][4] = Nseg
            Seg['num'].append(0)  # depth # Seg['num'][5] = 0  # depth
        else:
            Blok['Seg']['num'][3] = GLB['slg']  # 如果已存在第四个元素，更新其值为 GLB['slg']

        if "Nseg" not in Seg:
            Seg["Nseg"] = np.array(Nseg)


        # (2) Read OHL/Cable data
        STR = 'ABCN'
        Nlist = []
        Blist = []
        Cir['dat'] = []

        for i in range(Nrow):
            STR0 = 'ABCN'
            str0 = str(data[i, 0])  # type name
            if not str0.strip():    # nan/ ''
                continue
            ckid = data[i, 2]  # circuit id
            if str(ckid) == 'nan':
                continue
            npha = data[i, 4]  # phase #
            str2 = "CIR{:04d}".format(ckid)
            if str0 == "CIRC":  # one cable only
                for j in range(npha):
                    Blist.append("Y" + str2 + STR0[j])  # Bran
                    Nlist.append("X" + str2 + STR0[j])  # Head common node
                if Ngdw == 1:
                    Blist.append("Y" + str2 + 'M')
                    Nlist.append("X" + str2 + 'M')  # Head common node
                Seg['num'][5] = data[i, 10]  # depth of cable
                Cir['dat'].append([ckid, Ncon, 0, 0])  # AM
            elif str0 == "SW":
                Blist.append("Y" + str2 + 'S')  # Bran
                Nlist.append("X" + str2 + 'S')  # Head common node
                Cir['dat'].append([ckid, 1, 0, 0])  # SW
            elif str0 == "CIRO":
                for j in range(npha):
                    Blist.append("Y" + str2 + STR0[j])  # Bran
                    Nlist.append("X" + str2 + STR0[j])  # Head common node
                    Cir['dat'].append([ckid, j + 1, 0, 0])  # SWend
            elif str0 == "Meas":
                nmea = data[i, 5]  # # of measurment lines
                Meas = np.array(data[i:i + nmea, 2:5])  # pha (V) (I)

        # 确保 Nseg 和 Ncon 是整数
        Nseg_int = int(Nseg)
        Ncon_int = int(Ncon)

        Node = {'list': np.zeros((len(Nlist), Nseg_int), dtype='<U20'),
                'listdex': np.zeros((Ncon_int, Nseg_int), dtype=int),
                'pos':np.array([])}
        Bran = {'list': np.zeros((len(Blist), Nseg_int), dtype='<U20'),
                'listdex': np.zeros((Ncon_int, Nseg_int), dtype=int)}

        for j in range(Nseg_int):
            Node['list'][:, j] = [str(n) + f"{j + 1:03d}"for n in Nlist]
            Bran['list'][:, j] = [str(b) + f"{j + 1:03d}" for b in Blist]
            Node['listdex'][:, j] = np.arange(1, Ncon_int + 1) + j * Ncon_int
            Bran['listdex'][:, j] = np.arange(1, Ncon_int + 1) + j * Ncon_int

        # Adding new columns for Tail_node
        tail_node_column = [str(n) + f"{Nseg + 1:03d}" for n in Nlist]
        Node['list'] = np.column_stack((Node['list'], tail_node_column))
        Node['listdex'] = np.column_stack((Node['listdex'], np.arange(1, Ncon_int + 1) + Nseg * Ncon_int))

        # Creating com and comdex arrays
        Node['com'] = Node['list'][:, ::Nseg_int]
        Node['comdex'] = Node['listdex'][:, ::Nseg_int]

        # Setting up num arrays
        Nn = Ncon_int * (Nseg_int + 1)
        Nb = Ncon_int * Nseg_int
        Node['num'] = [Nn, Nn, 0, 0]
        Bran['num'] = [Nb, 0, 0, Nb, 0, 0]
        Cir['dat'] = np.array([Cir['dat']])
        Cir['dat'] = np.squeeze(Cir['dat'])

        return Node, Bran, Meas, Cir, Seg
    

    def Cable_Para_Cal(self,Line, VFIT, GND):
        def circshift(u,shiftnum1,shiftnum2):
            h,w = u.shape
            if shiftnum1 < 0:
                u = np.vstack((u[-shiftnum1:,:],u[:-shiftnum1,:]))
            else:
                u = np.vstack((u[(h-shiftnum1):,:],u[:(h-shiftnum1),:]))
            if shiftnum2 > 0:
                u = np.hstack((u[:, (w - shiftnum2):], u[:, :(w - shiftnum2)]))
            else:
                u = np.hstack((u[:,-shiftnum2:],u[:,:-shiftnum2]))
            return u
        
        def Cal_LC_Cable(Line):

            mu0 = 4*math.pi*1e-7
            mur_c = Line['mat'][2]
            mur_a = Line['mat'][3]
            Mur_c = mu0 * mur_c
            Mur_a = mu0 * mur_a
            Mur_i = mu0 * 1

            V0 = 3e8                            # light speed
            Vc = V0/math.sqrt(Line['mat'][4])   # speed in cable with directric material

            Ncon = int(Line['con'][0])          # number of conductors per segment
            Npha = int(Line['con'][1])          # of phase conductors


            L = np.zeros([Ncon, Ncon])
            C = np.zeros([Ncon, Ncon])
            
            rc = Line['rad'][0]                 # core radius
            rd = Line['rad'][1]
            rsa = Line['rad'][2]                # inner radus of armor
            rsb = Line['rad'][3]                # outer radus of armor
            rsc = Line['rad'][4]                # overal radius of a cable
            hit = Line['rad'][5]                # height of the cable above/below gnd
            # hit = 0.5 * (Line['pos'][2] + Line['pos'][5])   # height of the cable above/below gnd
            theta = 2 * math.pi / Npha          # angle between two cores
            thrng = np.arange(Npha) * theta     # theta range

            tds = rd/rsa                        # ratio of d/rsa
            out = Mur_c / (2*math.pi) * np.log(tds * np.power((1 + 1/np.power(tds,4) - 2/tds**2 * np.cos(thrng)) / (2 - 2*np.cos(thrng)), 0.5))
            out[0] = Mur_c/(2*math.pi) * math.log((rsa/rc) * (1-(tds)**2))  # self inductance
        
            for i in range(Npha):
                L[i,0:Ncon-1] = out.ravel()
                out = out.reshape(1,Ncon-1)
                out = circshift(out,0,1)
            if hit>0:
                L[-1][-1] = mu0/(2*math.pi)*math.log(2*hit/rsb)             # above the ground
                C[0:Npha, 0:Npha] = np.linalg.inv(L[0:Npha, 0:Npha])/Vc**2
                C[-1][-1] = 1/(L[-1][-1])/V0**2
            else:
                L[-1][-1] = Mur_i/(2*math.pi)*math.log(rsc/rsb)             # under the ground
                C[0:Npha, 0:Npha] = np.linalg.inv(L[0:Npha, 0:Npha])/Vc**2
                C[-1][-1] = 1/(L[-1][-1])/V0**2

            return L,C
    
        def Cal_ZY_Cable(Line, fre, GLB):
            fre = fre.reshape(1,np.size(fre))
            # fre = VFIT['fr'].reshape(1,np.size(VFIT['fr']))
            mu0 = 4 * math.pi * 1e-7
            mur_c = Line['mat'][2]
            mur_a = Line['mat'][3]
            Mur_c = mu0 * mur_c
            Mur_a = mu0 * mur_a
            Mur_g = mu0 * GLB['mur']

            ep0 = 8.854187818e-12
            epr_i = Line['mat'][4]
            Eps_i = ep0 * epr_i         # ground epsilon
            Eps_c = ep0 * 1             # air/core epsilon
            Eps_g = ep0 * GLB['epr']    # ground epsilon

            Sig_c = Line['mat'][0]      # core sigma
            Sig_a = Line['mat'][1]      # amore sigma
            Sig_g = GLB['sig']          # gnd sigma

            Ncon = int(Line['con'][0])  # number of conductors per segment
            Npha = int(Line['con'][1])  # of phase conductors
            # Nf = VFIT['fr'].size
            Nf = fre.size

            rc = Line['rad'][0]         # core radius
            rd = Line['rad'][1]
            rsa = Line['rad'][2]        # inner radus of armor
            rsb = Line['rad'][3]        # outer radus of armor
            rsc = Line['rad'][4]        # overal radius of a cable
            hit = Line['rad'][5]        # height of the cable above/below gnd
            # hit = 0.5 * (Line['pos'][2] + Line['pos'][5])      ####### hit ~= 0 !!!!!!!!!
            # hit = 1
            theta = 2 * math.pi / Npha  # angle between two cores
            thrng = np.arange(Npha).reshape(Npha,1) * theta # theta range (Npha x 1)

            omega = 2 * math.pi * fre
            gamma_c = np.power(1j * Mur_c * omega * (Sig_c + 1j * omega * Eps_c),0.5)
            gamma_a = np.power(1j * Mur_a * omega * (Sig_a + 1j * omega * Eps_c),0.5)
            gamma_g = np.power(1j * Mur_g * omega * (Sig_g + 1j * omega * Eps_g),0.5)

            Rs_a = rsa * gamma_a
            Rs_b = rsb * gamma_a 
            Hit_g = np.abs(hit) * gamma_g       # gamma of the ground

            K0_in_pipe = kv(0, Rs_a)
            K1_in_pipe = kv(1, Rs_a)
            I0_in_pipe = iv(0, Rs_a)
            I1_in_pipe = iv(1, Rs_a)

            K0_out_pipe = kv(0, Rs_b)
            K1_out_pipe = kv(1, Rs_b)
            I0_out_pipe = iv(0, Rs_b)
            I1_out_pipe = iv(1, Rs_b)

            Rc = rc * gamma_c 
            I0c =  iv(0, Rc)
            I1c =  iv(1, Rc)

            Zco = 1j * omega * Mur_c * I0c/ (2*math.pi*Rc*I1c)  # core inner imp
            out = 1j * omega * Mur_a / (2 * math.pi) * K0_in_pipe / (Rs_a * K1_in_pipe) # Nf vector
            out = np.tile(out, (Npha, 1))   # Npha x Nf matrix

            for x in range(1,16):
                out = out + np.dot(np.cos(x*thrng), ( 1j * omega * Mur_a / (2*math.pi) * np.power((rd/rsa), 2*x) * 2 / (x * (1+mur_a) + Rs_a * kv(x-1, Rs_a)/kv(x, Rs_a) ) ))
            

            out[0][:] = out[0][:] + Zco     # self impedance
            Z = np.zeros((Ncon, Ncon, Nf), dtype='complex_')    # Npha x Npha x Nf

            for i in range(Npha):   # rotation
                Z[i][0:Npha] = out  # 1 x Npha x Nf
                out = circshift(out, 1, 0)

            Zsa = 1j * omega * Mur_a / (2 * math.pi) / (Rs_a * Rs_b) / (I1_in_pipe * K1_out_pipe - I1_out_pipe * K1_in_pipe)

            tmp1 = I0_out_pipe * K1_in_pipe + I1_in_pipe * K0_out_pipe
            tmp2 = I1_out_pipe * K1_in_pipe - I1_in_pipe * K1_out_pipe
            Zpo = 1j * omega * Mur_a / (2*math.pi*Rs_b) * tmp1/tmp2
            Zpg = 1j * omega * Mur_a / (2*math.pi) * np.log ( (1 + Hit_g)/Hit_g)
            Zaa = Zpo + Zpg

            for i in range(Npha):
                Z[i][Npha][:] = Zsa.ravel()
                Z[Npha][i][:] = Zsa.ravel()

            Z[Npha][Npha][:] = Zaa

            return Z

        # (1a) VFIT initialization
        nc_fit = 5
        fre = np.concatenate([np.arange(1, 100, 10), np.arange(100, 1000, 100), np.arange(1000, 10000, 1000),
                              np.arange(10000, 50000, 10000)])
        Ht = []

        Para = {}
        Ht = {}

        Npha = int(Line['con'][1])
        Tcov = np.ones([Npha, Npha])

        for i in range(1,Npha):
            Tcov[i][i] = 1-Npha

        Tinv = -np.diag(np.ones(Npha-1))
        Tinv = np.vstack((np.ones(Npha-1), Tinv))
        Tinv = np.hstack((np.ones([Npha,1]), Tinv))/Npha

        L,C = Cal_LC_Cable(Line)

        Lc_modal = np.diag(np.diag(np.dot(np.dot(Tinv, L[0:Npha, 0:Npha]), Tcov)))  # modal domain
        Cc_modal = np.diag(np.diag(np.dot(np.dot(Tinv, C[0:Npha, 0:Npha]), Tcov)))  # modal domain

        La = L[-1,-1]   # phase domain
        Ca = C[-1,-1]   # phase domain

        Z = Cal_ZY_Cable(Line, fre, GND)

        Zc_modal = np.zeros([Npha, Npha, np.size(fre)])
        Zca_modal = np.zeros([Npha, 1, np.size(fre)])
        Zac_modal = np.zeros([1,Npha, np.size(fre)])

        for i in range(np.size(fre)):            # Z_modal
            Zc_modal[:,:,i] = np.diag(np.diag(np.dot(np.dot(Tinv, Z[:-1,:-1,i]), Tcov)))
            Zca_modal[:,:,i] = np.dot(Tinv, Z[:-1, -1,i].reshape(Npha,1))   # modal domain
            Zac_modal[:,:,i] = np.dot(Z[-1,:-1,i].reshape(1, Npha), Tcov)   # phase domain

        ord = VFIT['odg']   # % (*) x (*) x Nord

        htmp = np.zeros((Npha,Npha))
        dtmp = np.zeros((Npha,Npha))
        rtmp = np.zeros((Npha,Npha, ord*2))
        atmp = np.zeros((Npha,Npha, ord*2))
        for i in range(Npha):
            atmp[i,i,:], rtmp[i,i,:], dtmp[i,i], htmp[i,i] = Vector_Fitting.Vector_Fitting_auto(Zc_modal[i,i,:], fre, n_poles=ord)

        Ht['c_a'] = atmp
        Ht['c_r'] = rtmp
        Ht['c_d'] = dtmp 
        Ht['c_h'] = htmp

        htmp = np.zeros((Npha,1))
        dtmp = np.zeros((Npha,1))
        rtmp = np.zeros((Npha,1, ord*2))
        atmp = np.zeros((Npha,1, ord*2))

        atmp[0,0,:], rtmp[0,0,:], dtmp[0,0], htmp[0,0] = Vector_Fitting.Vector_Fitting_auto(Zca_modal[0,0,:], fre, n_poles=ord)

        Ht['ca_a'] = atmp
        Ht['ca_r'] = rtmp
        Ht['ca_d'] = dtmp 
        Ht['ca_h'] = htmp

        htmp = np.zeros((1,Npha))
        dtmp = np.zeros((1,Npha))
        rtmp = np.zeros((1,Npha, ord*2))
        atmp = np.zeros((1,Npha, ord*2))

        atmp[0,0,:], rtmp[0,0,:], dtmp[0,0], htmp[0,0] = Vector_Fitting.Vector_Fitting_auto(Zac_modal[0,0,:], fre, n_poles=ord)
        
        Ht['ac_a'] = atmp
        Ht['ac_r'] = rtmp
        Ht['ac_d'] = dtmp 
        Ht['ac_h'] = htmp

        Ht['a_a'], Ht['a_r'], Ht['a_d'], Ht['a_h'] = Vector_Fitting.Vector_Fitting_auto(Z[-1,-1,:], fre, n_poles=ord)

        Para['L'] = L
        Para['C'] = C
        Para['Z'] = Z
        Para['Rc'] = Ht['c_d']
        Para['Rca'] = Ht['ca_d'] 
        Para['Rac'] = Ht['ac_d']
        Para['Ra'] = Ht['a_d']
        Para['Lc'] = Ht['c_h'] + Lc_modal 
        Para['Lca'] = Ht['ca_h']
        Para['Lac'] = Ht['ac_h']
        Para['La'] = Ht['a_h'] + La 
        Para['Cc'] = Cc_modal 
        Para['Ca'] = Ca 
        Para['Tcov'] = Tcov
        Para['Tinv'] = Tinv
        Para['Ht'] = Ht

        return Para

