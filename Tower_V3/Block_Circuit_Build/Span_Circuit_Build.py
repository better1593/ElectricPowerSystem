import pandas as pd
import numpy as np
import math
import Vector_Fitting

from scipy.special import jv, iv

np.seterr(divide="ignore",invalid="ignore")

class Span_Circuit_Build():
    def __init__(self,path):
        self.path = path

    def Span_Circuit_Build(self, SpanModelName, TCom, TH_Info, TH_Node, TT_Info, TT_Node, VFIT, GLB):
        # (1) Read a complete table
        # path for excel table
        excel_path = self.path + '/' + SpanModelName
        # reading table
        raw_data = pd.read_excel(excel_path, header=None).values

        # (2a) Read general info. of a Cable, T_head, T_tail
        data, Blok, Info, Nwir, GND = self.Gene_Info_Read_v2(raw_data)
        Span = {}
        Span['Info'] = Info
        Span['ID'] = int(Info[9])
        Span['Atn'] = GLB['A'][Span['ID'] - 1, :]

        if GND['glb'] == 1:
            Span['GND'] = GLB['GND']
        Blok['Seg']['Lseg'] = GLB['slg']  # length of a segment

        if len(Blok['Seg']['num']) < 4:
            Blok['Seg']['num'].append(GLB['slg'])  # 添加第四个元素并设置为 GLB['slg']
        else:
            Blok['Seg']['num'][3] = GLB['slg']  # 如果已存在第四个元素，更新其值为 GLB['slg']
        # (2b) Read relative info. of T_head, T_tail
        Span['Info'][2] = TH_Info[0]  # name of head tower
        Span['Info'][3] = TT_Info[0]  # name of tail tower
        Pole = np.array([TH_Info[4], TH_Info[5], TH_Info[6], TT_Info[4], TT_Info[5], TT_Info[6]])  # pos of towers
        Span['Info'][4] = TH_Info[9]  # id of head tower
        Span['Info'][5] = TT_Info[9]  # id of tail tower
        rng = 1
        TC_head = TCom['head'] # Connected tower nodes
        TC_tail = TCom['tail'] # Connected tower nodes
        TC_head = self.Assign_Elem_id(TH_Node, TC_head, rng)  # ID and local pos
        TC_tail = self.Assign_Elem_id(TT_Node, TC_tail, rng)  # ID and local pos


        # (3) Getting wire parameters
        # (3a) Initilization of Span Model
        Node, Bran, Meas, Cir, Seg = self.Node_Bran_Index_Line(data, Blok, Pole)
        TC_head['list'] = TC_head['list'].reshape(-1, 1)  # 重塑为列向量
        TC_tail['list'] = TC_tail['list'].reshape(-1, 1)  # 重塑为列向量
        TC_head['listdex'] = TC_head['listdex'].reshape(-1, 1)  # 重塑为列向量
        TC_tail['listdex'] = TC_tail['listdex'].reshape(-1, 1)  # 重塑为列向量
        # 连接数组
        Node['com'] = np.hstack([Node['com'][:, [0]], TC_head['list'], Node['com'][:, [1]], TC_tail['list']])
        Node['comdex'] = np.hstack([Node['comdex'][:, [0]], TC_head['listdex'], Node['comdex'][:, [1]], TC_tail['listdex']])

        # 计算共同节点的数量
        Ncom = Node['com'].shape[0]
        pos = np.zeros((Ncom, 6))
        # 更新 Node 的位置信息
        for i in range(Ncom):
            concatenated_row = np.concatenate((TC_head['pos'][i, :], TC_tail['pos'][i, :]))
            pos[i, :] = concatenated_row + Pole
        Node['pos'] = pos

        # (3b) Read wire parameters of a span
        Ncon = Seg['Ncon']  # # of line/conductors
        # 从 data 中读取 wire 参数
        OHLP_data = [data[i][5:20] for i in range(int(Ncon))]  # 以 0-based 索引为例，根据实际情况调整索引范围
        OHLP = np.array(OHLP_data, dtype=float)  # 转换为 NumPy 数组 # wire para with pole data
        OHLP[:, 0:6] = Node['pos']  # update (x/y/z) of OHLP

        # (3c) mapping table (hspan/hsid/Thead tspan/tsid/Ttail)
        S2Tmap = {}
        a = Span['ID']
        b = TH_Info[9]
        S2Tmap['head'] = np.vstack((np.array([Span['ID'], TH_Info[9]]), Node['comdex'][:, 0:2]))
        S2Tmap['tail'] = np.vstack((np.array([Span['ID'], TT_Info[9]]), Node['comdex'][:, 2:4]))

        # (4) Calculate wave equation parameters (Z and Y)
        VFmod = [Info[7], Info[8]]
        Para = self.OHL_Para_Cal(OHLP, VFmod, VFIT, GND)


        # (5) Output---------------------------------------------------------------
        Span['Cir'] = Cir
        Span['Seg'] = Seg
        Span['Pole'] = Pole
        Span['OHLP'] = OHLP
        Span['Para'] = Para
        Span['Node'] = Node
        Span['Bran'] = Bran
        Span['Meas'] = Meas
        Span['S2Tmap'] = S2Tmap
        Span['Soc'] = []
        # Span['Ht'] = Ht
        # Span['Cal'] = Cal

        return Span, S2Tmap

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
        blokint = []        # + inveter (simple
        blokinf = []        # + inveter (full)
        blokmck = []        # + matching circuit
        blokoth1 = []       # + other 1
        blokoth2 = []       # + other 2
        bloka2g = {}        # air-2-gnd bridge
        bloklup = []
        sys_name = ''

        nair = 0
        ngnd = 0
        nmea = 0

        row_del = []
        Row_len = len(DATA)                         # of lines with [b0, n1 n2] names
        for i in range(Row_len):                    # read data_para table
            first_column_value = str(DATA[i][0])    # check the key word

            if first_column_value == "TOWER":
                row_del.append(i)
                blokflag[0:10] = [float(x) for x in DATA[i][1:11]]    # T: sub_CK flag (char)
                blokname = ["WIRE", "", "", "", "", "", "", "", "", ""]         # with a air-gnd bridge
            elif first_column_value == "SPAN":
                row_del.extend([i, i + 1])
                Cir['num'] = np.array(DATA[i:i + 2, 1:7], dtype=float)  # S: Cir.dat (2 lines)
            elif first_column_value == "CABLE":
                row_del.extend([i, i + 1])
                Cir['num'] = np.array(DATA[i:i + 2, 1:7], dtype=float)  # C: Cir.dat (2 lines)
            elif first_column_value == "INFO":
                row_del.append(i)
                Info = DATA[i][1:13]                                # T/S/C Info (cell array)
            elif first_column_value == "GND":
                row_del.append(i)                                   # T/S/C soil model
                GND["glb"] = float(DATA[i][1])                      # 1 = GLB_GND_data
                GND["sig"] = float(DATA[i][5])
                GND["mur"] = float(DATA[i][6])
                GND["epr"] = float(DATA[i][7])
                GND["gnd"] = float(DATA[i][8])                      # gnd model: 0 or 1 or 2
            elif first_column_value == "A2GB":                      # T: air-2-gnd conn pair
                blokflag[0] = 2                                     # + 1 = AirW only, 2 = AirW+GndW
                blokname[0] = "Wire+A2G"
                bloka2g["list"] = np.zeros((0, 3))
                row_num = int(DATA[i][5])                           # line # for input vari
                oft = 0
                for j in range(1, row_num + 1, 2):
                    row_id = i + (j - 1)
                    for k in range(1, 6):
                        tmp1 = str(DATA[row_id][k])                 # Air node name
                        tmp2 = str(DATA[row_id + 1][k])             # Gnd node name
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
            elif first_column_value == "END":                       # Used for loading lump model
                row_del.append(i)
                sys_name = str(DATA[i][1])                          # Lump sub-CK name
            elif first_column_value == "AirW":
                nair += 1                                           # record air wire #
            elif first_column_value == "GndW":
                ngnd += 1                                           # record gnd wire #
            elif first_column_value == "Meas":                      # record the meas line #
                nmea = int(DATA[i][5])
            elif first_column_value == "LUMP":                      # read com_node in lump CK file
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
            "int": blokint,
            "inf": blokinf,
            "mck": blokmck,
            "oth1": blokoth1,
            "oth2": blokoth2,
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
            Blok["Seg"]["Ncon"] = Cir["num"][1][0]
            Blok["Seg"]["Npha"] = Cir["num"][1][0] - Cir["num"][1][1]
            Blok["Seg"]["Ngdw"] = Cir["num"][1][1]
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
        # Ncol = len(rng)
        Ncol = 1
        if elem["list"].size > 0:
            if elem["list"].ndim == 1:
                elem["list"] = elem["list"].reshape((1, Nrow))
        if elem["listdex"].size > 0:
            if elem["listdex"].ndim == 1:
                elem["listdex"] = elem["listdex"].reshape((1, Nrow))

        for i in range(Nrow):
            # for j in range(Ncol):
            # id = rng[j]
            # elem_name = elem["list"][i, id]

            id = rng - 1
            elem_name = elem["list"][id, i]
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
        Meas = {}

        for i in range(Nrow):
            STR0 = 'ABCN'
            str0 = str(data[i, 0])  # type name
            if not str0.strip():
                continue
            ckid = data[i, 2]  # circuit id
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
                Meas['list'] = np.array(data[i:i+nmea, 2:5])  # CK, pha, Seg
                Meas['flag'] = np.array(data[i:i + nmea, 1])  # 1=I, 2=V
                Itmp = np.where(Meas['list'][:, 0] > 5000)[0]  # cable only

                if len(Itmp) == 0:  # span
                    Meas['listdex'] = np.empty((0, 2), dtype=int)  # [cond_id seg_id]
                    for ik in range(nmea):
                        pos = [0, 0, *Meas['list'][ik, :2]]
                        cond_id = self.Cond_ID_Read(Cir, pos)
                        Meas['listdex'] = np.vstack((Meas['listdex'], [cond_id, Meas['list'][ik, 2]]))
                else:  # cable only
                    Meas['listdex'] = Meas['list'][:, 1:3]  # [cond_id seg_id]

                Meas['Ib'] = Meas['listdex'][Meas['flag'] == 1]
                Meas['Vn'] = Meas['listdex'][Meas['flag'] == 2]

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
    

    def OHL_Para_Cal(self, OHLP, VFmod, VFIT, GND):

        # Define output
        Para = {}
        Ht = {}
        Imp = {}

        ord = VFIT['odc'] * 2
        Smod = VFmod
        High = 0.5 * (OHLP[:, 2] + OHLP[:, 5]) # High
        Dist = OHLP[:, 6]
        r0 = OHLP[:, 7] # r0
        Vair = 3e8  # Velocity in free space
        Ncon = np.size(OHLP,0)
        Tcov = np.ones([Ncon,Ncon])
        for i in range(1,Ncon):
            Tcov[i,i] = 1-Ncon

        Tinv = -np.diag(np.ones([Ncon-1]))
        Tinv = np.concatenate((np.ones([1,Ncon-1]), Tinv), axis=0)
        Tinv = np.concatenate((np.ones([Ncon, 1]), Tinv), axis=1)/Ncon


        # L = self.Cal_L_OHL(OHLP) + self.Cal_M_OHL(OHLP)
        # C = np.linalg.inv(L)/(3e8**2)

        L, C = self.Cal_LC_OHL(High, Dist, r0)

        if VFmod[1] == 0:
            if VFmod[0]==0:
                R = np.diag(OHLP[:,8])
                L = L + np.diag(OHLP[:,9])
                C = np.linalg.inv(L)/3e8/3e8
            else:
                Zc = self.Cal_Zc_OHL(OHLP, VFIT['fr'])
                atmp = np.zeros([Ncon, Ncon, ord])
                rtmp = np.zeros([Ncon,Ncon,ord])
                dtmp = np.zeros([Ncon, Ncon])
                htmp= np.zeros([Ncon, Ncon])    
                for i in range(Ncon):
                    atmp[i,i,:], rtmp[i,i,:], dtmp[i,i], htmp[i,i] = Vector_Fitting.Vector_Fitting_auto(Zc[i,i,:], GLB.frq, n_poles=int(ord/2))

                R = dtmp 
                L = L + htmp
                C = C 

                Ht['a'] = atmp
                Ht['r'] = rtmp
                Ht['d'] = dtmp
                Ht['h'] = htmp

        else:
            if VFmod[0] == 0:
                R = np.diag(OHLP[:,8])
                L = L + np.diag(OHLP[:,9])
                C = np.linalg.inv(L)/3e8/3e8

                R = np.diag(np.diag(np.dot(np.dot(Tinv, R), Tcov)))
                L = np.diag(np.diag(np.dot(np.dot(Tinv, L), Tcov)))
                C = np.diag(np.diag(np.dot(np.dot(Tinv, C), Tcov)))

                Zg = self.Cal_Zg_OHL(OHLP, VFIT['fr'], GND)
                for i in range(len(VFIT['fr'])):
                    Zg[:,:,i] = np.diag(np.diag(np.dot(np.dot(Tinv, Zg[:,:,i]), Tcov)))

                atmp = np.zeros([Ncon, Ncon, ord])
                rtmp = np.zeros([Ncon,Ncon,ord])
                dtmp = np.zeros([Ncon, Ncon])
                htmp= np.zeros([Ncon, Ncon])

                for i in range(Ncon):
                    atmp[i,i,:], rtmp[i,i,:], dtmp[i,i], htmp[i,i] = Vector_Fitting.Vector_Fitting_auto(Zg[i,i,:], GLB.frq, n_poles=int(ord/2))

                R = R + dtmp
                L = L + htmp
                C = C 
                Ht['a'] = atmp
                Ht['r'] = rtmp
                Ht['d'] = dtmp
                Ht['h'] = htmp

            else:
                L = np.diag(np.diag(np.dot(np.dot(Tinv, L), Tcov)))
                C = np.diag(np.diag(np.dot(np.dot(Tinv, C), Tcov)))

                Zc = self.Cal_Zc_OHL(OHLP, VFIT['fr'])
                Zg = self.Cal_Zg_OHL(OHLP, VFIT['fr'], GND)
                for i in range(len(VFIT['fr'])):
                    Zg[:,:,i] = np.diag(np.diag(np.dot(np.dot(Tinv, Zg[:,:,i]), Tcov)))
                    Zc[:,:,i] = np.diag(np.diag(np.dot(np.dot(Tinv, Zc[:,:,i]), Tcov)))

                atmp = np.zeros([Ncon, Ncon, ord])
                rtmp = np.zeros([Ncon,Ncon,ord])
                dtmp = np.zeros([Ncon, Ncon])
                htmp = np.zeros([Ncon, Ncon])
                
                for i in range(Ncon):
                    atmp[i,i,:], rtmp[i,i,:], dtmp[i,i], htmp[i,i] = Vector_Fitting.Vector_Fitting_auto(Zg[i,i,:]+Zc[i,i,:], VFIT['fr'], n_poles=int(ord/2))

                R = dtmp
                L = L + htmp
                C = C 

                Ht['a'] = atmp
                Ht['r'] = rtmp
                Ht['d'] = dtmp
                Ht['h'] = htmp                

        Imp['R'] = R
        Imp['L'] = L
        Imp['C'] = C

        Para['Imp'] = Imp
        Para['Tcov'] = Tcov
        Para['TinV'] = Tinv
        Para['Ht'] = Ht

        return Para

    def Cal_L_OHL(self, OHL_Para):
        mu0 = 4 * math.pi * 1e-7
        h = OHL_Para[:, 2] # High
        r = OHL_Para[:, 7] # r0
        km = mu0 / (2 * math.pi)
        out = km * np.log(2 * h / r)
        L = np.diag(out)
        return L

    def Cal_LC_OHL(self, High, Dist, r0):
        # Calculate OHL Parameters (L and C per unit) with Height and hori. Dist
        Vair = 3e8  # Velocity in free space
        mu0 = 4 * math.pi * 1e-7
        km = mu0 / (2 * math.pi)
        Ncon = High.shape[0]

        out = np.log(2 * High / r0)
        L = np.diag(out)

        for i1 in range(Ncon - 1):
            for i2 in range(i1 + 1, Ncon):
                d = np.abs(Dist[i1] - Dist[i2])
                h1 = High[i1]
                h2 = High[i2]
                L[i1, i2] = 0.5 * np.log((d ** 2 + (h1 + h2) ** 2) / (d ** 2 + (h1 - h2) ** 2))
                L[i2, i1] = L[i1, i2]

        L = km * L
        C = np.linalg.inv(L) / Vair ** 2

        return L, C

    def Cal_M_OHL(self, OHL_Para):
        mu0 = 4 * math.pi * 1e-7
        M = np.zeros((np.size(OHL_Para, 0), np.size(OHL_Para, 0)), dtype='complex_')

        for i1 in range(0, np.size(OHL_Para, 0) - 1):
            for i2 in range(i1 + 1, np.size(OHL_Para, 0)):
                d = abs(OHL_Para[i1][6] - OHL_Para[i2][6])
                h1 = OHL_Para[i1][2]
                h2 = OHL_Para[i2][2]
                M[i1][i2] = mu0 / 2 / math.pi * np.log(
                    (np.square(d) + np.square(h1 + h2)) / (np.square(d) + np.square(h1 - h2)))
                M[i2][i1] = M[i1][i2]

        return M

    def Cal_Zc_OHL(self, OHL_Para, fre):
        ep0 = 8.854187818e-12
        mu0 = 4 * math.pi * 1e-7
        ri = OHL_Para[:, 7]
        sig = OHL_Para[:, 10]
        mur = OHL_Para[:, 11]
        epr = OHL_Para[:, 12]
        Ncon = np.size(OHL_Para, 0)
        omega = 2 * math.pi * fre
        Zc = np.zeros((np.size(OHL_Para, 0), np.size(OHL_Para, 0), np.size(fre, 0)), dtype='complex_')
        for i in range(0, Ncon):
            gamma = (1j * mu0 * mur[i] * omega * (sig[i] + 1j * omega * ep0 * epr[i])) ** 0.5
            Ri = ri[i] * gamma
            I0i = iv(0, Ri)
            I1i = iv(1, Ri)
            Zc[i, i, :] = 1j * mu0 * mur[i] * omega * I0i / (2 * math.pi * Ri * I1i)
        return Zc

    def Cal_Zg_OHL(self, OHL_Para, fre, GND):
        ep0 = 8.854187818e-12
        mu0 = 4 * np.pi * 1e-7
        Sig_g = GND.sig
        Mur_g = GND.mur * mu0
        Eps_g = GND.epr * ep0
        Ncon = OHL_Para.shape[0]

        omega = 2 * np.pi * fre
        gamma = np.sqrt(1j * Mur_g * omega * (Sig_g + 1j * omega * Eps_g))

        Zg = np.zeros((Ncon, Ncon, len(fre)), dtype=np.complex128)

        for i1 in range(Ncon - 1):
            for i2 in range(i1 + 1, Ncon):
                d = np.abs(OHL_Para[i1, 6] - OHL_Para[i2, 6])
                h1 = OHL_Para[i1, 2]
                h2 = OHL_Para[i2, 2]
                Zg[i1, i2, :] = 1j * omega * mu0 / (2 * np.pi) * np.log(
                    ((1 + gamma * (h1 + h2) / 2) ** 2 + (d * gamma / 2) ** 2) / (
                                (gamma * (h1 + h2) / 2) ** 2 + (d * gamma / 2) ** 2))
                Zg[i2, i1, :] = Zg[i1, i2, :]

        for i in range(Ncon):
            h = OHL_Para[i, 2]
            Zg[i, i, :] = 1j * omega * Mur_g / (2 * np.pi) * np.log(((1 + gamma * h) ** 2) / ((gamma * h) ** 2))

        return Zg

    def Cond_ID_Read(self, Cir, Pos):
        # Return Conductor ID of a Span given by Pos
        # Pos = [T/S/C, id, cir_id, phas_id, cond_id, seg_id]

        cir_id = Pos[2]  # cir_id
        pha_id = Pos[3]  # pha_id

        if cir_id in [row[0] for row in Cir['dat']]:
            idx = [row[0] for row in Cir['dat']].index(cir_id)
            if idx == 0:
                con_id = pha_id  # 电路组中的ID
            else:
                con_id = sum([row[1] for row in Cir['dat'][:idx]], 1) + pha_id - 1
        else:
            raise ValueError('ID in Global Data and Tower/Span Data does not match')

        return con_id