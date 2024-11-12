import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import numpy.matlib

np.seterr(divide="ignore",invalid="ignore")


class Tower_Circuit_Build():
    def __init__(self,path):
        self.path = path

    # Tower_Circuit_Build
    def Tower_Circuit_Build(self,TowerModelName, VFIT, GLB):
        # (1) Read a complete table
        excel_path = self.path + '/' + TowerModelName # path for excel table
        raw_data = pd.read_excel(excel_path, header=None).values # reading table

        # (2) Read general info. of tower inc. Blok (file name, com_node name)
        data, Blok, Info, Nwir, GND = self.Gene_Info_Read_v2(raw_data) # common/gnd node and data
        Tower = {"Info": Info, "ID": Info[9]} # define the Tower
        if GND["glb"] == 1:
            Tower["GND"] = GLB["GND"]
        Tower["Ats"] = GLB["A"][:, Tower["ID"]-1]
        if GLB['Acab']:
            Tower['Acabts'] = GLB['Acab'][:, Tower["ID"]-1]
        else:
            Tower['Acabts'] = []

        # (3) Build Tower Model
        # (3a) Wire: Node/Bran/Meas (list/listdex/num) and Blok: sys (listdex)
        Node, Bran, Meas, Blok, nodebran = self.Node_Bran_Index_Wire(data, Blok, Nwir)

        if Blok['flag'][1] >= 1:
            # (3b) Read wire parameters of a tower
            WireP = data[:, 5:20]  # MATLAB中的6:20在Python中是5:19
            WireP = np.hstack([WireP, nodebran])

            # delete measurement lines
            dmea = Nwir['comp']
            WireP = WireP[:dmea, :]

            # Rotate the pole according to the angle in Info(4)
            theta = Info[3]  # 假设 Info 已经是一个二维数组或列表

            # update coordinates
            WireP[:, 0:2] = self.RotPos(WireP[:, 0:2], theta)
            WireP[:, 3:5] = self.RotPos(WireP[:, 3:5], theta)
            Node['pos'][:, 0:2] = self.RotPos(Node['pos'][:, 0:2], theta)  # 假设 Node 是一个字典

            # plot the figure
            # self.Wire_Plot(WireP, 11, Bran)  # 需要相应的绘图函数

            # close the figure
            plt.close()

            # (3b) Cal. wire-Model-Parameters
            VFmod = Info[7:9]
            CK_Para = self.Wire_Model_Para3(WireP, Node, Bran, VFmod, VFIT, GND)
        else:
            Nn = Node['num'][0]
            WireP = []
            CK_Para = {
                'A': np.zeros((1, Nn)),  # Add pseudo bran to retain node struct
                'R': np.array([]),
                'L': np.array([]),
                'C': np.zeros((Nn, Nn)),
                'G': np.zeros((Nn, Nn)),
                'P': np.array([]),
                'Cw': np.array([]),
                'Ht': np.array([]),
                'Vs': np.array([]),
                'Is': np.array([]),
                'Nle': np.array([]),
                'Swh': np.array([])
            }

        # (4) Update CK_Para, Node, Bran, Meas with Aie2Gnd Bridge
        CK_Para, Bran, Blok = self.Tower_A2G_Bridge(CK_Para, Node, Bran, Blok)

        # (5) Build the tower
        Tower['CK_Para'] = CK_Para
        Tower['Blok'] = Blok
        Tower['Bran'] = Bran
        Tower['Node'] = Node
        Tower['Meas'] = Meas
        Tower['Tower0'] = Tower.copy()                       # for updating CK model only
        Tower['WireP'] = WireP
        Tower['T2Smap'] = self.Tower_Map_Init()     # Mapping table initialization
        Tower['T2Cmap'] = self.Tower_Map_Init()     # Mapping table initialization
        Tower['Soc'] = []

        # (6) Update the tower with lump CK+
        Tower = self.Tower_Circuit_Update(Tower)

        return Tower

    # Gene_Info_Read_v2
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
            Blok["Seg"]["Ncon"] = Cir["num"][1]
            Blok["Seg"]["Npha"] = Cir["num"][1] - Cir["num"][2]
            Blok["Seg"]["Ngdw"] = Cir["num"][2]
            Blok["Seg"]["num"] = [Blok["Seg"]["Ncon"], Blok["Seg"]["Npha"], Blok["Seg"]["Ngdw"]]

        # of all wires or lump components
        Nwir = {"nair": nair, "ngnd": ngnd, "nmea": nmea, "comp": nrow - nmea}

        return DATA, Blok, Info, Nwir, GND

    # Com_Node_Read
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

    # Node_Bran_Index_Wire
    def Node_Bran_Index_Wire(self,data, Blok, Nwir):
        # Return Name and ID for Node and Bran, and update id in Blok
        # Node.list/listdex/num
        # Bran.list/listdex/num
        # Meas.list/listdex
        # With data (list of lists), common node/bran, Nwir.nair/ngnd/nmea/comp

        flag = Blok['flag']
        # (0) Initialization
        Node = {"list": np.array([], dtype=str), "listdex": np.array([], dtype=int),
                "pos": np.array([], dtype=float),"com": np.array([],dtype=str,),
                "num": np.zeros([1,4])}  # cooridnates of nodes

        Bran = {"list": np.array([], dtype=str), "listdex": np.array([], dtype=int),
                "pos": np.array([], dtype=float),"num": np.zeros([1,6])}

        Meas = {"list": np.array([], dtype=str), "listdex": np.array([], dtype=int),
                "flag": np.array([], dtype=int)}

        oftn = 0
        Blist = np.array([])
        Blistdex = np.empty((0,3))
        nodebran = np.empty((0,3))


        Nrow = len(data)
        Ncop = Nwir["comp"]  # # of componenets
        Nmea = Nwir["nmea"]  # # of measurment lines
        Nair = Nwir["nair"]  # air wire #
        Ngnd = Nwir["ngnd"]  # gnd wire #

        if Nrow != 0:
            # (1) bran indexing
            Nnum = np.zeros(4, dtype=int)
            namelist = np.array(data)[:, 2:5]  # string array [b0 n1 n2]
            for i in range(Nrow):  # Assuming num_rows is the number of rows in namelist
                tmp0 = np.zeros(3,dtype=int) # [b0 n1 n2]
                tmp0[0] = i + 1
            # (2) node indexing
                for j in range(0,2):
                    str_val = namelist[i, 1 + j]
                    # Check if the string is not empty, not a single space, and not missing
                    if str_val != "" and str_val != " " and str_val is not np.nan:
                        # Try to find the index of the string in Node['list']
                        tmp1 = np.where(Node['list'] == str_val)[0]

                        if len(tmp1) == 0:  # If not found, add the node to the list
                            oftn += 1
                            Node['list'] = np.concatenate([Node['list'], [str_val]])
                            Node['listdex'] = np.concatenate([Node['listdex'], [oftn]])
                            pos = data[i, (j) * 3 + 5:(j) * 3 + 8]
                            Node['pos'] = np.concatenate([Node['pos'], pos])
                            tmp0[1 + j] = oftn
                        else:  # If found, use the existing index
                            tmp0[1 + j] = tmp1[0] + 1

                Bran["listdex"] = np.append(Bran["listdex"], tmp0)
                Blistdex = np.vstack([Blistdex, tmp0])
                nodebran = np.vstack([nodebran, tmp0])

                if i == Nair - 1:  # for wire case
                    Nnum[1] = oftn
                    Nnum[3] = oftn
                if i == Nair + Ngnd:  # for wire case
                    Nnum[2] = oftn - Nnum[1]

            Nnum[0] = oftn
            # define Node
            Node["num"] = Nnum
            # Node["com"] = np.array([], dtype=int)
            # Node["condex"] = np.array([], dtype=int)
            # 重新定义Node的行列
            num_elements = Node["list"].size
            Node["list"] = Node["list"].reshape((num_elements,1))
            Node["listdex"] = Node["listdex"].reshape((num_elements,1))
            Node["pos"] = Node["pos"].reshape((num_elements,3))
            # 定义Bran
            Bran["list"] = namelist[:Ncop]
            Bran["listdex"] = nodebran[:Ncop]
            # 重新定义Bran
            Bran["num"] = np.array([Ncop, Nair, Ngnd, 0, 0, 0], dtype=int)  # wire parameters

            # 定义Meas
            tmp = data[:, 1]  # 获取数据中的第二列
            Itmp = [i for i, val in enumerate(tmp) if val > 0]  # 找到大于0的数值的索引
            Meas["list"] = np.array([namelist[i] for i in Itmp])  # 根据索引获取 namelist 中的对应项
            Meas["listdex"] = np.array([nodebran[i, :3] for i in Itmp])  # 根据索引获取 nodebran 中的对应项
            Meas['flag'] = np.array([tmp[i] for i in Itmp])  # 根据索引获取 tmp 中的对应项
            Meas = self.Assign_Elem_id(Bran, Meas, 1) # not necessary！
        else:
            oft = 0
            Node = {}  # 初始化 Node 字典

            lump_elements = [Blok.ins, Blok.sar, Blok.txf, Blok.grd, Blok.int, Blok.inf, Blok.mck, Blok.oth1, Blok.oth2]

            for lump_elem in lump_elements:
                Node, oft = self.Assign_Elem_Lump(Node, lump_elem, oft)


        # (3)  update blok.sys.listdex (Com_Node id in WIRE for all sub-cir)
        flag = Blok["flag"]
        if flag[1] == 1:
            Blok["ins"] = self.Assign_Elem_id(Node, Blok["ins"], 1)
        if flag[2] == 1:
            Blok["sar"] = self.Assign_Elem_id(Node, Blok["sar"], 1)
        if flag[3] == 1:
            Blok["txf"] = self.Assign_Elem_id(Node, Blok["txf"], 1)
        if flag[4] == 1:
            Blok["grd"] = self.Assign_Elem_id(Node, Blok["grd"], 1)
        if flag[5] == 1:
            Blok["int"] = self.Assign_Elem_id(Node, Blok["int"], 1)
        if flag[6] == 1:
            Blok["inf"] = self.Assign_Elem_id(Node, Blok["inf"], 1)
        if flag[7] == 1:
            Blok["mck"] = self.Assign_Elem_id(Node, Blok["mck"], 1)
        if flag[8] == 1:
            Blok["oth1"] = self.Assign_Elem_id(Node, Blok["oth1"], 1)
        if flag[9] == 1:
            Blok["oth2"] = self.Assign_Elem_id(Node, Blok["oth2"], 1)

        return Node, Bran, Meas, Blok, nodebran

    # Assign_Elem_id
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
                            a = row_id[0]
                            if ref_elem["listdex"].shape[1] > 1:
                                elem["listdex"][i, id] = ref_elem["listdex"][row_id[0], row_id[0]]
                            else:
                                elem["listdex"][i, id] = ref_elem["listdex"][row_id[0]]
                        else:
                            if ref_elem["listdex"].shape[0] > 1:
                                elem["listdex"][id, i] = ref_elem["listdex"][row_id[0], id]
                            else:
                                elem["listdex"][i, id] = ref_elem["listdex"][row_id[0]]

                    if ref_elem["pos"] is not None and ref_elem["pos"].size > 0:

                        # elem["pos"] = elem["pos"].reshape((num_ref_elem, 3))
                        elem["pos"][i, 0:3] = ref_elem["pos"][row_id[0], 0:3]

        return elem

    # Wire_Plot
    def Wire_Plot(self,WireP,NodeW,BranW):
        # DRAW LINE DIAGRAM
        scale = 0.1
        wireplot = 1  # 如果为1则绘制，为0则不绘制
        WireP = np.nan_to_num(WireP, nan=0)
        n1 = WireP[:, 14].astype(int)
        n2 = WireP[:, 15].astype(int)
        str1 = BranW["list"][:, 1]
        str2 = BranW["list"][:, 2]

        x1, y1, z1 = WireP[:, 0], WireP[:, 1], WireP[:, 2]
        x2, y2, z2 = WireP[:, 3], WireP[:, 4], WireP[:, 5]
        px, py, pz = np.column_stack([x1, x2]), np.column_stack([y1, y2]), np.column_stack([z1, z2])
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        num_dx = dx.size
        # dx = dx.reshape(num_dx,1)
        # dy = dy.reshape(num_dx,1)
        # dz = dz.reshape(num_dx,1)
        ds2 = dx ** 2 + dy ** 2 + dz ** 2
        ds2 = np.array(ds2, dtype=float)
        # ds = np.sqrt(dx * dx + dy * dy + dz * dz)
        # ds = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        ds = np.sqrt(ds2)
        cosa, cosb, cosc = dx / ds, dy / ds, dz / ds

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if wireplot:
            for i in range(len(px)):
                ax.plot([px[i, 0], px[i, 1]], [py[i, 0], py[i, 1]], [pz[i, 0], pz[i, 1]], '-k', linewidth=2)
            # ax.plot(px, py, pz, '-k', linewidth=2)
            # ax.plot通常输入一维数组


        ax.quiver(x1, y1, z1, cosa, cosb, cosc, length=scale, color='r', arrow_length_ratio=0.1)

        for i in range(len(n1)):
            ax.text(x1[i], y1[i], z1[i], str1[i], fontsize=12, color='red')
            ax.text(x2[i], y2[i], z2[i], str2[i], fontsize=12, color='red')

        # 设置视角
        ax.view_init(elev=20, azim=45)  # elev 是仰角，azim 是方位角
        # 设置坐标
        # plt.xlabel(' ')
        # plt.ylabel(' ')
        plt.title('3D Tower Plot')
        plt.show() # 画图
        # 保存图形
        # plt.savefig("3d_plot.png", dpi=300)  # 保存为PNG格式，分辨率为300 DPI

    # RotPos
    def RotPos(self, Pos, theta):
        # Perform rotation of coordinates (x y) of elements using theta (anti-clockwise)
        # Pos: numpy array of shape (n, 2)
        # theta: angle in degrees

        # Convert theta to radians
        theta_rad = np.radians(theta)

        # Rotation matrix
        rotM = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                         [np.sin(theta_rad), np.cos(theta_rad)]])

        # Perform rotation
        PosNew = rotM @ Pos.T  # 矩阵乘法
        PosNew = PosNew.T  # 转置回原始形状

        return PosNew

    # INT_LINE_D2P_D
    def INT_LINE_D2P_D(self, U1a, U1b, V1, W1, r1, U2a, U2b, V2, W2, r2):

        # (0) initialization
        ELIM = 1e-9  # limit for changing formula
        a2 = np.maximum(r2, r1)  # avoiding log(0) for negative uij
        a2 = a2 * a2

        no = len(U1a)
        ns = len(U2a)

        if no != ns:
            out = []
            return out

        u13 = U1a - U2a
        u14 = U1a - U2b
        u23 = U1b - U2a
        u24 = U1b - U2b

        u13s = u13 * u13
        u23s = u23 * u23
        u14s = u14 * u14
        u24s = u24 * u24

        As = (V2 - V1) * (V2 - V1) + (W2 - W1) * (W2 - W1)
        As = np.maximum(As, a2)
        t132 = np.array(As + u13s, dtype=float)
        t232 = np.array(As + u23s, dtype=float)
        t142 = np.array(As + u14s, dtype=float)
        t242 = np.array(As + u24s, dtype=float)

        t13 = np.sqrt(t132)
        t23 = np.sqrt(t232)
        t14 = np.sqrt(t142)
        t24 = np.sqrt(t242)

        # using the exact formulas for calculation
        a = -u24 * np.log(u24 + t24)
        b = - u13 * np.log(u13 + t13)
        c = u23 * np.log(u23 + t23)
        d = u14 * np.log(u14 + t14)
        I1 = -u24 * np.log(u24 + t24) - u13 * np.log(u13 + t13) + u23 * np.log(u23 + t23) + u14 * np.log(u14 + t14)

        s = u24 + t24
        Idex = s < ELIM
        s = u13 + t13
        s = u23 + t23
        s = u14 + t14

        if np.sum(Idex) != 0:
            I1a = u24 * np.log(t24 - u24) + u13 * np.log(t13 - u13) - u23 * np.log(t23 - u23) - u14 * np.log(
                t14 - u14)
            I1[Idex] = I1a[Idex]

        I2 = t24 + t13 - t23 - t14
        out = I1 + I2

        return out

    # Int_Slan_2D
    def INT_SLAN_2D(self, ps1, ps2, rs, pf1, pf2, rf, PROD_MOD, COEF_MOD):
        # (0) initialization
        g0 = 1e-5
        d0 = 1e-6
        r0 = 1e-10

        # get the size of matrix
        Ns = len(ps1[:, 0])  # report error if len directly
        Nf = len(pf1[:, 0])
        # 定义rs和rf
        rs = rs.reshape(Ns, 1)
        rf = rf.reshape(Nf, 1)
        # a = ps1 - ps2
        ls2 = np.sum((ps1 - ps2) * (ps1 - ps2), axis=1)
        lf2 = np.sum((pf1 - pf2) * (pf1 - pf2), axis=1)
        # ls2 = [round(num, 2) for num in ls2]
        # lf2 = [round(num, 2) for num in lf2]
        # 无法使用np.round四舍五入，强制保留小数，其中  只能现在这么操作了 ，其中的2代表保留几位小数（已解决）
        ls2_elements = ls2.size
        lf2_elements = lf2.size
        if ls2_elements == 0:
            pass
        else:
            ls2_row = int(ls2_elements / Ns)
            ls2 = ls2.reshape(Ns, ls2_row)
        if lf2_elements == 0:
            pass
        else:
            lf2_row = int(lf2_elements / Nf)
            lf2 = lf2.reshape(Nf, lf2_row)

        ls2 = np.array(ls2, dtype=float)
        lf2 = np.array(lf2, dtype=float)
        ls = np.sqrt(ls2)
        lf = np.sqrt(lf2)

        # (1) determine the distance of 4 points
        if PROD_MOD == 1:  # dot product
            # case 1
            OMG = np.zeros((Nf, Ns))

            # ensure matrix has elements in all positions

            ls = np.matlib.repmat(np.transpose(ls), Nf, 1)
            lf = np.matlib.repmat(lf, 1, Ns)
            ls2 = np.matlib.repmat(np.transpose(ls2), Nf, 1)
            lf2 = np.matlib.repmat(lf2, 1, Ns)

            R12 = (ps2[:, 0] - pf2[:, 0]) ** 2 + (ps2[:, 1] - pf2[:, 1]) ** 2 + (
                    ps2[:, 2] - pf2[:, 2]) ** 2  # Decrease in columns
            R22 = (ps2[:, 0] - pf1[:, 0]) ** 2 + (ps2[:, 1] - pf1[:, 1]) ** 2 + (ps2[:, 2] - pf1[:, 2]) ** 2
            R32 = (ps1[:, 0] - pf1[:, 0]) ** 2 + (ps1[:, 1] - pf1[:, 1]) ** 2 + (ps1[:, 2] - pf1[:, 2]) ** 2
            R42 = (ps1[:, 0] - pf2[:, 0]) ** 2 + (ps1[:, 1] - pf2[:, 1]) ** 2 + (ps1[:, 2] - pf2[:, 2]) ** 2

            R12 = np.array([R12], dtype=float)
            R22 = np.array([R22], dtype=float)
            R32 = np.array([R32], dtype=float)
            R42 = np.array([R42], dtype=float)
        elif PROD_MOD == 2:  # vector product
            # case 2
            OMG = np.zeros((Nf, Ns))

            # ensure matrix has elements in all positions
            ls = np.matlib.repmat(np.transpose(ls), Nf, 1)
            lf = np.matlib.repmat(lf, 1, Ns)
            ls2 = np.matlib.repmat(np.transpose(ls2), Nf, 1)
            lf2 = np.matlib.repmat(lf2, 1, Ns)

            # 如果你需要保持二维形状，即形状为 (m, 1) 而不是 (m,)，你需要显式地进行重塑操作：
            # column_2d = my_array[:, j:j+1]  # 保持二维形状
            a = np.tile(pf2[:, 0: 1], (1, Ns))
            b = np.tile(np.transpose(ps2[:, 0: 1]), (Nf, 1))
            dx = np.tile(pf2[:, 0: 1], (1, Ns)) - np.tile(np.transpose(ps2[:, 0: 1]), (Nf, 1))  # transpose matrix
            dy = np.tile(pf2[:, 1: 2], (1, Ns)) - np.tile(np.transpose(ps2[:, 1: 2]), (Nf, 1))
            dz = np.tile(pf2[:, 2: 3], (1, Ns)) - np.tile(np.transpose(ps2[:, 2: 3]), (Nf, 1))
            R12 = dx ** 2 + dy ** 2 + dz ** 2

            dx = np.tile(pf1[:, 0: 1], (1, Ns)) - np.tile(np.transpose(ps2[:, 0: 1]), (Nf, 1))
            dy = np.tile(pf1[:, 1: 2], (1, Ns)) - np.tile(np.transpose(ps2[:, 1: 2]), (Nf, 1))
            dz = np.tile(pf1[:, 2: 3], (1, Ns)) - np.tile(np.transpose(ps2[:, 2: 3]), (Nf, 1))
            R22 = dx ** 2 + dy ** 2 + dz ** 2

            dx = np.tile(pf1[:, 0: 1], (1, Ns)) - np.tile(np.transpose(ps1[:, 0: 1]), (Nf, 1))
            dy = np.tile(pf1[:, 1: 2], (1, Ns)) - np.tile(np.transpose(ps1[:, 1: 2]), (Nf, 1))
            dz = np.tile(pf1[:, 2: 3], (1, Ns)) - np.tile(np.transpose(ps1[:, 2: 3]), (Nf, 1))
            R32 = dx ** 2 + dy ** 2 + dz ** 2

            dx = np.tile(pf2[:, 0: 1], (1, Ns)) - np.tile(np.transpose(ps1[:, 0: 1]), (Nf, 1))
            dy = np.tile(pf2[:, 1: 2], (1, Ns)) - np.tile(np.transpose(ps1[:, 1: 2]), (Nf, 1))
            dz = np.tile(pf2[:, 2: 3], (1, Ns)) - np.tile(np.transpose(ps1[:, 2: 3]), (Nf, 1))
            R42 = dx ** 2 + dy ** 2 + dz ** 2

        else:
            print('No such case in INT_ARBI_2D')
            exit()

        R12 = np.array(R12, dtype=float)
        R22 = np.array(R22, dtype=float)
        R32 = np.array(R32, dtype=float)
        R42 = np.array(R42, dtype=float)
        # get the distance between each point
        R1 = np.sqrt(R12)  # pf2-ps2
        R2 = np.sqrt(R22)  # pf1-ps2
        R3 = np.sqrt(R32)  # pf1-ps1
        R4 = np.sqrt(R42)  # pf2-ps1

        # (2) find the cos and sin
        a2 = (R42 - R32 + R22 - R12)
        cose = a2 / (2 * ls * lf)
        sine2 = 1 - cose ** 2
        # sine = np.sqrt(sine2)
        # 存在负数的情况
        sine = np.sqrt(sine2.astype(complex))
        # (2a) update u (alpha) and v (beta)
        DIS = 4 * ls2 * lf2 - a2 * a2

        par1 = cose > 1 - g0  # parallel lines 1
        par2 = cose < g0 - 1  # parallel lines 2
        # para = par1 | par2
        # sign = par1 ^ par2  # change into ^
        # 存在负号的形式
        para = np.logical_or(par1, par2)
        sign = np.where(par1, 1, 0) - np.where(par2, 1, 0)
        sign = sign[para]  # sign for lf
        # sign = sign.reshape(38, 1)

        u = np.array(ls * ((2 * lf2 * (R22 - R32 - ls2) + a2 * (R42 - R32 - lf2)) / DIS))
        v = (lf * ((2 * ls2 * (R42 - R32 - lf2) + a2 * (R22 - R32 - ls2)) / DIS))
        u[para] = 0
        v[para] = -(R22[para] - R32[para] - ls2[para]) / (2 * ls[para])  # parallel lines

        d2 = abs(R32 - u ** 2 - v ** 2 + 2 * u * v * cose)
        d = (np.sqrt(d2))

        copn = d < d0  # co-plane
        Itmp = para | copn  # both co-plane and parallel lines
        id = ~Itmp  # lines other than Itmp
        # print(id)

        R1 = np.maximum(r0, R1)  # avoid zero distance
        R2 = np.maximum(r0, R2)
        R3 = np.maximum(r0, R3)
        R4 = np.maximum(r0, R4)

        # (3) lines in different planes and non-parallel lines
        OMG[id] = (np.arctan(np.real((d2[id] * cose[id] + (u[id] + ls[id]) * (v[id] + lf[id]) * sine2[id]) / (d[id] * R1[id] * sine[id])))
                   - np.arctan(np.real((d2[id] * cose[id] + (u[id] + ls[id]) * v[id] * sine2[id]) / (d[id] * R2[id] * sine[id])))
                   + np.arctan(np.real((d2[id] * cose[id] + (u[id] * v[id]) * sine2[id]) / (d[id] * R3[id] * sine[id])))
                   - np.arctan(np.real((d2[id] * cose[id] + u[id] * (v[id] + lf[id]) * sine2[id]) / (d[id] * R4[id] * sine[id]))))

        # (4) main item (end pt positioned on the another line)
        INT = 0
        tp0 = lf / (R1 + R2)
        tp1 = (u + ls) * np.arctanh(tp0)
        tp1[abs(tp0 - 1) < r0] = 0
        INT = INT + tp1

        tp0 = ls / (R1 + R4)
        tp1 = (v + lf) * np.arctanh(tp0)
        tp1[abs(tp0 - 1) < r0] = 0
        INT = INT + tp1

        tp0 = lf / (R3 + R4)
        tp1 = u * np.arctanh(tp0)
        tp1[abs(tp0 - 1) < r0] = 0
        INT = INT - tp1

        tp0 = ls / (R2 + R3)
        tp1 = v * np.arctanh(tp0)
        tp1[abs(tp0 - 1) < r0] = 0
        INT = INT - tp1

        tp0 = OMG * d / sine
        tp0[abs(sine) < g0] = 0
        INT = 2 * INT - tp0
        INT = np.real(INT)  # 输出将只包含实数部分
        # print(INT)

        # (5) update integral with parallel line results
        if para.any():
            tp = np.zeros((Nf, Ns))
            if len(rf) == 1:  # rf是个数 取不了长度到时候可以再改改
                Rf = np.tile(rf, (Nf, Ns))
            else:
                Rf = np.tile(rf, (1, Ns))
            if len(rs) == 1:  #
                Rs = np.tile(rs, (Nf, Ns))
            else:
                Rs = np.tile(np.transpose(rs), (Nf, 1))  #

        d = np.transpose(d)
        para = np.transpose(para)
        # tp = np.transpose(tp)
        # ls = np.transpose(ls)
        # Rs = np.transpose(Rs)
        # v = np.transpose(v)
        # lf = np.transpose(lf)
        # Rf = np.transpose(Rf)

        out = self.INT_LINE_D2P_D(tp[para], ls[para], tp[para], 0, \
                                  Rs[para], v[para], v[para] + sign * lf[para], d[para], 0, Rf[para])
        INT[para] = np.abs(out)

        # (6) check whether it is the integral for inductance or potential
        if COEF_MOD == 2:  # inductance
            INT = cose * INT

        # return result
        return INT

    # Wire_Model_Para1
    def Wire_Model_Para1(self, ps1, ps2, ls, rs, pf1, pf2, lf, rf, At, Nnode):
        # (0) Initialization
        ELIM = 1e-3  # 没用到
        if ps1.size == 0:
            L = np.array([])
            P = np.array([])
            return L, P

        PROD_MOD = 2  # matrix product
        COEF_MOD = 1  # double integral

        # (1) Inductance calculation of branches
        L = self.INT_SLAN_2D(ps1, ps2, rs, pf1, pf2, rf, PROD_MOD, COEF_MOD)

        # (2) Generating coordinates of node segments (half of bran segments)
        ps0 = 0.5 * (ps1 + ps2)
        pf0 = 0.5 * (pf1 + pf2)
        ofs = 0

        # init
        ps1_len = len(ps1)
        rs = rs.reshape(ps1_len, 1)
        rf = rf.reshape(ps1_len, 1)
        ls = ls.reshape(ps1_len, 1)
        lf = lf.reshape(ps1_len, 1)
        N = ps1_len * 2
        nrs = np.zeros((N, 1))
        nls = np.zeros((N, 1))
        nps1 = np.zeros((N, 3))
        nps2 = np.zeros((N, 3))
        nrf = np.zeros((N, 1))
        nlf = np.zeros((N, 1))
        npf1 = np.zeros((N, 3))
        npf2 = np.zeros((N, 3))
        ncom = np.zeros((Nnode, 1))

        for ik in range(1, Nnode + 1):  # size of node segments for source
            pt1 = np.where(At[:, 0] == ik)[0]  # pos of ith node in branch
            pt2 = np.where(At[:, 1] == ik)[0]  # pos of ith node in branch
            d1 = len(pt1)  # total # of common nodes for ith node
            d2 = len(pt2)  # total # of common nodes for ith node

            # (2a) 1st half segment
            if d1 != 0:
                indices = slice(ofs, ofs + d1)
                nrs[indices, 0:1] = rs[pt1]  # radius (n1)-source
                nls[indices, 0:1] = ls[pt1] / 2  # length (n1)
                nps1[indices, 0:3] = ps1[pt1, 0:3]  # start points
                nps2[indices, 0:3] = ps0[pt1, 0:3]  # end points

                nrf[indices, 0:1] = rf[pt1]  # radius(n1)-field
                nlf[indices, 0:1] = lf[pt1] / 2  # length(n1)
                npf1[indices, 0:3] = pf1[pt1, 0:3]  # start points
                npf2[indices, 0:3] = pf0[pt1, 0:3]  # end points

            ofs += d1
            # (2b) 2nd half segment
            if d2 != 0:
                indices = slice(ofs, ofs + d2)
                nrs[indices, 0:1] = rs[pt2]  # radius (n2)
                nls[indices, 0:1] = ls[pt2] / 2  # length (n2)
                nps1[indices, 0:3] = ps0[pt2, 0:3]  # start points
                nps2[indices, 0:3] = ps2[pt2, 0:3]  # end points

                nrf[indices, 0:1] = rf[pt2]  # radius(n1)
                nlf[indices, 0:1] = lf[pt2] / 2  # length(n2)
                npf1[indices, 0:3] = pf0[pt2, 0:3]  # start points
                npf2[indices, 0:3] = pf2[pt2, 0:3]  # end points

            ofs += d2
            ncom[ik - 1] = d1 + d2  # of segments for each node

        # (4) Calculating potential matrix
        PROD_MOD = 2  # matrix product
        COEF_MOD = 1  # integration only

        INT = self.INT_SLAN_2D(nps1, nps2, nrs, npf1, npf2, nrf, PROD_MOD, COEF_MOD)

        # (5) merging common nodes
        P = INT

        Idel = []  # 假设 Idel 是一个已经初始化的列表
        ofs = 0  # 假设 ofs 是一个初始偏移量
        nlns = np.zeros((Nnode, 1))  # 初始化 nlns
        nlnf = np.zeros((Nnode, 1))  # 初始化 nlnf

        for ik in range(Nnode):
            nc = ncom[ik][0]  # of common nodes for ith node
            nc = int(nc)
            if nc >= 1:
                Idel.extend(range(ofs + 1, ofs + nc))  # collecting deleted row/col
            # 使用切片来求和
            nlns[ik] = np.sum(nls[ofs:ofs + nc])  # total length of ith node (source)
            nlnf[ik] = np.sum(nlf[ofs:ofs + nc])  # total length of ith node (field)

            # 收集共同节点的行和列
            tmp = P[ofs:ofs + nc, :]  # collecting rows of common nodes
            P[ofs, :] = np.sum(tmp, axis=0)  # sum of all rows
            tmp = P[:, ofs:ofs + nc]  # collecting cols of common nodes
            P[:, ofs] = np.sum(tmp, axis=1)  # sum of all cols

            ofs += nc

        P = np.delete(P, Idel, axis=0)
        P = np.delete(P, Idel, axis=1)
        P = P / (nlns * np.transpose(nlnf))

        return L, P

    # Wire_Model_Para2
    def Wire_Model_Para2(self, Wire_Para, Bnum, Nnum, GND):
        # （0) Intial constants
        ep0 = 8.854187818e-12
        mu0 = 4 * math.pi * 1e-7
        ke = 1 / (4 * math.pi * ep0)
        km = mu0 / (4 * math.pi)  # coef for inductance

        Nb = Bnum[1] + Bnum[2]  # all bran # excluding surf bran
        Nn = Nnum[0]  # all node #
        Nba = Bnum[1]  # air bran #
        # Nbg = Bnum[2]               # gnd bran #
        Nna = Nnum[1]  # air node #
        Nng = Nnum[2]  # gnd node #

        rb1 = np.arange(0, Nba)  # air bran
        rb1 = rb1.flatten()

        rb2 = np.arange(Nba, Nb)  # gnd bran
        rb2 = rb2.flatten()

        rn1 = np.arange(0, Nna)  # air node
        rn1 = rn1.flatten()

        rn2 = np.arange(Nna, Nn)  # gnd node
        rn2 = rn2.flatten()

        # (1) Obtain L and P matrices without considering the ground effect
        # (1a) L and P matrices for all: aa, ag, ga, gg
        ps1 = Wire_Para[:, 0:3]  # starting points
        ps2 = Wire_Para[:, 3:6]  # ending points
        rs = Wire_Para[:, 7: 8]  # radius
        At = Wire_Para[:, 16:18]  # leaving and entering nodes

        # get segment length
        ds = ps2 - ps1
        ds2 = ds * ds
        sum_ds2 = np.sum(ds2, axis=1)
        sum0 = len(sum_ds2)
        sum_ds2 = sum_ds2.reshape(sum0, 1)
        sum_ds2 = np.array(sum_ds2, dtype=float)
        ls = np.sqrt(sum_ds2)  # segment length

        # get cosa, cosb and cosc
        cosa = ds[:, 0] / ls
        cosb = ds[:, 1] / ls
        cosc = ds[:, 2] / ls
        cosa = np.diag(cosa)
        cosa = cosa.reshape(Nb, 1)
        cosb = np.diag(cosb).reshape(Nb, 1)
        cosb = cosb.reshape(Nb, 1)
        cosc = np.diag(cosc).reshape(Nb, 1)
        cosc = cosc.reshape(Nb, 1)

        # for gnd and air segments
        Lout, Pout = self.Wire_Model_Para1(ps1, ps2, ls, rs, ps1, ps2, ls, rs, At, Nn)

        # (2) Constructing L and P by considering the image effect
        # no ground (0), perfect ground (1), lossy ground model (2)
        # (2a) without ground
        # free-space inductance
        L0 = Lout * (cosa * np.transpose(cosa) + cosb * np.transpose(cosb) + cosc * np.transpose(cosc))
        # free-space potential
        P0 = Pout

        # (2b) with ground
        # L and P matrices for aai,ggi
        if GND['gnd'] != 0:
            pf1 = ps1.copy()
            pf1[:, 2] = -pf1[:, 2]  # image for air segments
            pf2 = ps2.copy()
            pf2[:, 2] = -pf2[:, 2]  # image for gnd segments

            Lai, Pai = self.Wire_Model_Para1(ps1[rb1, :], ps2[rb1, :], ls[rb1, 0], rs[rb1, 0], \
                                             pf1[rb1, :], pf2[rb1, :], ls[rb1, 0], rs[rb1, 0], At[rb1, :],
                                             Nna)  # image of air

            a = ps1[[rb2], :]
            b = ps2[rb2, :]
            c = At[rb2, :] - Nna

            Lgi, Pgi = self.Wire_Model_Para1(ps1[rb2, :], ps2[rb2, :], ls[rb2, 0], rs[rb2, 0], \
                                             pf1[rb2, :], pf2[rb2, :], ls[rb2, 0], rs[rb2, 0], At[rb2, :] - Nna,
                                             Nng)  # image of gnd
        # (2bi) perfect ground
        if GND['gnd'] == 1:
            L0 = L0 - Lai * (cosa * np.transpose(cosa) + cosb * np.transpose(cosb) + cosc * np.transpose(cosc))
            P0 = P0 - Pai

        # (2bii) lossy ground model
        if GND['gnd'] == 2:
            if not rb1.size or not rb2.size:
                Lag = np.array([])
                Lga = np.array([])
                Pag = np.array([])
                Pga = np.array([])
            else:
                Lag = Lout[rb1[:, np.newaxis], rb2]
                Lga = Lout[rb2[:, np.newaxis], rb1]
                Pag = Pout[rn1[:, np.newaxis], rn2]
                Pga = Pout[rn2[:, np.newaxis], rn1]

            # image effect
            # (i) f./s. wire in air
            # cosaA = cosa[rb1]       # air wire: direction numbers
            # cosbA = cosb[rb1]
            coscA = cosc[rb1]
            L0[rb1[:, np.newaxis], rb1] = L0[rb1[:, np.newaxis], rb1] + Lai * coscA * np.transpose(
                coscA)  # vertical contribution
            P0[rn1[:, np.newaxis], rn1] = P0[rn1[:, np.newaxis], rn1] - Pai

            # 还没调试好
            if Bnum[2] != 0:
                # cosaG = cosa[rb2]  # gnd wire: direction numbers
                # cosbG = cosb[rb2]
                coscG = cosc[rb2]

                # (ii) f. wire in gnd s. wire in air
                L0[rb2[:, np.newaxis], rb1] = L0[rb2[:, np.newaxis], rb1] + Lga * coscG * np.transpose(
                    coscA)  # vertical contribution
                P0[rn2[:, np.newaxis], rn1] = P0[rn2[:, np.newaxis], rn1] - Pga

                # (iii) f./s. wire in gnd
                L0[rb2[:, np.newaxis], rb2] = L0[rb2[:, np.newaxis], rb2] - Lgi * coscG * np.transpose(
                    coscG)  # vertical contribution
                P0[rn2[:, np.newaxis], rn2] = P0[rn2[:, np.newaxis], rn2] + Pgi

                # (iv) f. wire in air s. wire in gnd
                L0[rb1[:, np.newaxis], rb2] = L0[rb1[:, np.newaxis], rb2] - Lag * coscA * np.transpose(
                    coscG)  # vertical contribution
                P0[rn1[:, np.newaxis], rn2] = P0[rn1[:, np.newaxis], rn2] + Pag

        L0 = km * L0
        P0 = ke * P0
        return L0, P0

    # Wire_Model_Para3
    def Wire_Model_Para3(self, WireP, Node, Bran, VFmod, VFIT, GND):
        # (0) Initial constants
        ep0 = 8.854187818e-12
        # r0 = 0.01                         # radius of OhL conductors # former
        ke = 1 / (4 * math.pi * ep0)  # coef for potential

        Nbran = Bran['num'][1] + Bran['num'][2]  # total # of brans excluding surf bran
        Nnode = Node['num'][0]  # total # of nodes
        Ncom = Node['com'].shape[0]  # total # of common nodes
        imp = WireP[:, 8:10]  # wire internal impedance (Re, Le)
        mod = WireP[:, 13:15]  # Wire modes [cond_VFid, Gnd_VFid]
        nodebran = WireP[:, 15:18]  # [b0 n1 n2]
        nodebran = nodebran.astype(int)

        # (1) Obtaining the incidence matrix A
        A0 = np.zeros((Nbran, Nnode))
        for ik in range(Nbran):
            A0 = self.Wire_AssignAValue(A0, ik, nodebran[ik, 1] - 1, -1)  # in  = -1
            A0 = self.Wire_AssignAValue(A0, ik, nodebran[ik, 2] - 1, +1)  # out = +1

        # (2) Obtaining L and P matrices of wires considering ground effect
        L0, P0 = self.Wire_Model_Para2(WireP, Bran['num'], Node['num'], GND)

        # (3) Updating Ri and Li with Const. impedance or VF results
        odc = VFIT['odc']
        Ht = {'r': np.zeros((Nbran, odc)), 'd': np.zeros((Nbran, odc)), 'id': []}

        if VFmod[0] == 0:
            # (3a) Constant internal Ri and Li for condcutors
            dR = imp[:Nbran, 0]  # int resistance (const.)
            dL = imp[:Nbran, 1]  # int. inductance (const.)
        else:
            # (3b) VFIT results of internal Ri and Li for condcutors
            for ik in range(Nbran):
                VFid = WireP[ik, 13]  # id of C-VF data in a table
                dR[ik] = VFIT['rc'][VFid, 0]  # getting dc res
                dL[ik] = VFIT['dc'][VFid, 0]  # getting dc ind
                Ht['r'][ik, :odc] = VFIT['rc'][VFid, 1:]  # getting residual values
                Ht['d'][ik, :odc] = VFIT['dc'][VFid, 1:]  # getting pole values
                Ht['id'].append(ik)  # index of wires in the wire set

        R0 = np.diag(dR)
        L0 = L0 + np.diag(dL)

        # (4) per-unit length capacitance of OHL/CAB for connection to the Tower
        Cw = []
        if Ncom != 0:
            Hw = np.zeros((Ncom, 1))  # Height of the wire
            Cw = np.zeros((Ncom, 2))  # capacitance vector
            Cw[:, 0] = Node['comdex'][:, 1]  # id of com_node
            for ik in range(Ncom):
                idx = np.where(nodebran[:, 1] == Cw[ik, 0])[0]  # find node posi in a wire set
                idy = np.where(nodebran[:, 2] == Cw[ik, 0])[0]  # find node posi in a wire set
                if len(idx) != 0:
                    Hw[ik, 0] = WireP[idx[0], 2]
                    rw[ik, 0] = WireP[idx[0], 7]
                else:
                    Hw[ik, 0] = WireP[idy[0], 5]
                    rw[ik, 0] = WireP[idy[0], 7]
            tmp0 = 2 * ke * np.log(2 * Hw / rw)  # potential wrt the ground
            Cw[:, 1] = tmp0 ** -1  # capacitance

        CK_Para = {'A': A0, 'R': R0, 'L': L0, 'C': np.zeros((Nnode, Nnode)), 'G': np.zeros((Nnode, Nnode)),
                   'P': P0, 'Cw': Cw, 'Ht': Ht, 'Vs': [], 'Is': [], 'Nle': [], 'Swh': []}
        return CK_Para

    def Lump_Model_Intepret(self, Flag, Name, ID, TCOM):
        # Build Model of a lump sub-system with input from an excel file
        # Input: Flag = 1/0 enable/disable
        #        Name = input file name （存储了电气元件表格名）
        #        ID =   sub-system ID (2=INS, 3=SAR, 4=TXF, 5=GRD, 6=OTH)
        #        TXF=tranformer SAR=surge arrester
        #        BCOM = Nodecom.list/listdex (external/local)

        # (1) 初始化变量，用于存储提取的数据
        if Flag[ID] == 0:
            CK_Para = {}
            Bran = {}
            Node = {}
            Meas = {}
            return CK_Para, Node,Bran, Meas

        # (2) Read a complete table
        # 直接找到对应的电气元件表格文件名
        LumpModelName = Name[ID]
        # path for excel table
        # path = "H:\Ph.D. Python\StrongEMPPlatform\Tower\Main Program\DATA_Files"
        # excel_path = path + '/' + LumpModelName

        excel_path = self.path + '/' + LumpModelName
        # num, txt, raw_data = pd.read_excel(TowerModelName)
        # reading table
        raw_data = pd.read_excel(excel_path, header=None).values
        # # data0 = pd.read_excel(LumpModelName)     # read a complete table
        # raw_data = pd.read_excel(LumpModelName, header=None)

        # (2a) Read common node (node.com/comdex/suffix=sysname)
        data, Blok, Info, Nwir, notused = self.Gene_Info_Read_v2(raw_data)
        NodeCom_int = Blok['lup']  # com_node in sub_CK，获得节点名和索引
        Ncom = len(Blok['lup']['list'])

        # (2b) Read bran/node/meas/num, nodebran and assign them with ID
        Node, Bran, Meas, nodebran = self.Node_Bran_Index_Lump(data, NodeCom_int, Nwir)

        # (2c) Update node.com/comdex including both int and ext info.
        Node['com'] = np.vstack((TCOM['list'], np.transpose(NodeCom_int['list'])),dtype='U20').transpose()  # [ext int]: Node.com
        Node['com'] = Node['com'].astype(str)  # 定义元素类型为字符串

        Node['comdex'] = np.vstack((TCOM['listdex'], np.transpose(NodeCom_int['listdex']))).transpose()  # [ext int]: Node.comdex
        Node['comdex'] = Node['comdex'].astype(int)  # 定义元素类型为整数

        # (2d) Add the suffix into the name of nodes/brans
        app = "_" + Blok['sysname']
        Node['list'] = self.Assign_Suffix(Node['list'], app)
        Node_com_element = Node['com'][:, 1]
        Node_com_element = self.Assign_Suffix(Node_com_element, app)
        Node_com_element = Node_com_element.reshape(-1,1)

        Node['com'][:, 1] = Node_com_element[:, 0]
        Bran['list'] = self.Assign_Suffix(Bran['list'], app)
        Meas['list'] = self.Assign_Suffix(Meas['list'], app)

        # (3) Build circuit models by groups
        Nn = Node['num'][0]  # node #
        Nb = Bran['num'][0]  # bran #
        R = np.zeros([Nb, Nb])
        L = np.zeros([Nb, Nb])
        C = {'list': np.empty((0, 2)), 'listdex': np.empty((0, 3), dtype=int)}
        G = {'list': np.empty((0, 2)), 'listdex': np.empty((0, 3), dtype=int)}
        A = np.zeros([Nb, Nn])
        Vs = {'dat': np.array([]), 'pos': np.array([])}  # pos 为 [bran id node_1 node_2]
        Is = {'dat': np.empty((0, 3)), 'pos': np.empty((0, 1))}  # pos 为 node id
        Nle = {'dat': np.empty((0, 1)), 'pos': np.empty((0, 3))}  # dat为model id，pos为[bran id node_1 node_2]
        Swh = {'dat': np.empty((0, 1)), 'pos': np.empty((0, 3))}  # model id

        Nline = Nwir['comp']  # # of components
        for i in range(Nline):  # 循环遍历数据表
            firstColumnValue = str(data[i, 0])  # 提取第一列的值
            if firstColumnValue == 'RL':
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 2] - 1, +1)
                R[nodebran[i, 0] - 1, nodebran[i, 0] - 1] = data[i, 5]
                L[nodebran[i, 0] - 1, nodebran[i, 0] - 1] = data[i, 6]
            elif firstColumnValue == 'R':
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 2] - 1, +1)
                R[nodebran[i, 0] - 1, nodebran[i, 0] - 1] = data[i, 5]
            elif firstColumnValue == 'L':
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 2] - 1, +1)
                L[nodebran[i, 0] - 1, nodebran[i, 0] - 1] = data[i, 5]
            elif firstColumnValue == 'C':
                C['list'] = np.vstack((C['list'], (data[i, 3:5])))
                new_row = np.array([nodebran[i, 1], nodebran[i, 2], data[i, 5]])
                C['listdex'] = np.vstack((C['listdex'], new_row))
            elif firstColumnValue == 'G':
                G['list'] = np.vstack((G['list'], (data[i, 3:5])))
                G['listdex'] = np.vstack((G['listdex'], (nodebran[i, 1], nodebran[i, 2], str(data[i, 5]))))
            elif firstColumnValue == 'M2':
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 2] - 1, +1)
                A = self.Assign_A_Value(A, nodebran[i + 2, 0] - 1, nodebran[i + 2, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i + 2, 0] - 1, nodebran[i + 2, 2] - 1, +1)
                R[nodebran[i, 0] - 1, nodebran[i, 0] - 1] = data[i, 5]
                R[nodebran[i, 0] - 1, nodebran[i + 2, 0] - 1] = data[i + 1, 5]
                R[nodebran[i + 2, 0] - 1, nodebran[i, 0] - 1] = data[i + 1, 5]
                R[nodebran[i + 2, 0] - 1, nodebran[i + 2, 0] - 1] = data[i + 2, 5]
                L[nodebran[i, 0] - 1, nodebran[i, 0] - 1] = data[i, 6]
                L[nodebran[i, 0] - 1, nodebran[i + 2, 0] - 1] = data[i + 1, 6]
                L[nodebran[i + 2, 0] - 1, nodebran[i, 0] - 1] = data[i + 1, 6]
                L[nodebran[i + 2, 0] - 1, nodebran[i + 2, 0] - 1] = data[i + 2, 6]
            elif firstColumnValue == 'M3':
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 2] - 1, +1)
                A = self.Assign_A_Value(A, nodebran[i + 3, 0] - 1, nodebran[i + 3, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i + 3, 0] - 1, nodebran[i + 3, 2] - 1, +1)
                A = self.Assign_A_Value(A, nodebran[i + 5, 0] - 1, nodebran[i + 5, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i + 5, 0] - 1, nodebran[i + 5, 2] - 1, +1)

                R[nodebran[i, 0] - 1, nodebran[i, 0] - 1] = data[i, 5]
                R[nodebran[i + 3, 0] - 1, nodebran[i + 3, 0] - 1] = data[i + 3, 5]
                R[nodebran[i + 5, 0] - 1, nodebran[i + 5, 0] - 1] = data[i + 5, 5]

                R[nodebran[i, 0] - 1, nodebran[i + 3, 0] - 1] = data[i + 1, 5]
                R[nodebran[i + 3, 0] - 1, nodebran[i, 0] - 1] = data[i + 1, 5]
                R[nodebran[i, 0] - 1, nodebran[i + 5, 0] - 1] = data[i + 2, 5]
                R[nodebran[i + 5, 0] - 1, nodebran[i, 0] - 1] = data[i + 2, 5]
                R[nodebran[i + 3, 0] - 1, nodebran[i + 5, 0] - 1] = data[i + 4, 5]
                R[nodebran[i + 5, 0] - 1, nodebran[i + 3, 0] - 1] = data[i + 4, 5]

                L[nodebran[i, 0] - 1, nodebran[i, 0] - 1] = data[i, 6]
                L[nodebran[i + 3, 0] - 1, nodebran[i + 3, 0] - 1] = data[i + 3, 6]
                L[nodebran[i + 5, 0] - 1, nodebran[i + 5, 0] - 1] = data[i + 5, 6]

                L[nodebran[i, 0] - 1, nodebran[i + 3, 0] - 1] = data[i + 1, 6]
                L[nodebran[i + 3, 0] - 1, nodebran[i, 0] - 1] = data[i + 1, 6]
                L[nodebran[i, 0] - 1, nodebran[i + 5, 0] - 1] = data[i + 2, 6]
                L[nodebran[i + 5, 0] - 1, nodebran[i, 0] - 1] = data[i + 2, 6]
                L[nodebran[i + 3, 0] - 1, nodebran[i + 5, 0] - 1] = data[i + 4, 6]
                L[nodebran[i + 5, 0] - 1, nodebran[i + 3, 0] - 1] = data[i + 4, 6]
            elif firstColumnValue == 'nle':
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 2] - 1, +1)
                R[nodebran[i, 0] - 1, nodebran[i, 0] - 1]= data[i, 5]
                new_row = np.array([nodebran[i, 0: 3]]).ravel()
                Nle['pos'] = np.vstack((Nle['pos'],new_row))
                new_row = np.array([data[i, 8]]).ravel()
                Nle['dat'] = np.vstack((Nle['dat'],new_row))
            elif firstColumnValue == 'swh':
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 2] - 1, +1)
                R[nodebran[i, 0] - 1, nodebran[i, 0] - 1] = data[i, 5]
                new_row = np.array([nodebran[i, 0: 3]]).ravel()
                Swh['pos'] = np.vstack((Swh['pos'],new_row))
                new_row = np.array([data[i, 8]]).ravel()
                Swh['dat'] = np.vstack((Swh['dat'],new_row))
            elif firstColumnValue == 'Is':
                tmp2 = str(data[i, 8])  # file name
                if tmp2 != '':
                    Is['dat'] = np.array(pd.read_excel(tmp2, header=None, engine='openpyxl'))  # Cell array
                else:
                    ispeak = data[i, 6]
                    isfreq = data[i, 7]
                    new_row = np.array([np.nan, isfreq, ispeak])
                    Is['dat'] = np.vstack((Is['dat'],new_row))
                new_row = np.array([nodebran[i, 1]])
                Is['pos'] = np.vstack((Is['pos'], new_row))  # 本应是纵向
            elif firstColumnValue == 'Vs':
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 2] - 1, +1)
                R[nodebran[i, 0] - 1, nodebran[i, 0] - 1] = data[i, 5]

                tmp2 = str(data[i, 8])
                if tmp2 != '':
                    with open(tmp2, 'r') as file:
                        tmp3 = np.loadtxt(file, dtype=float)
                    Vs['dat'] = np.append(Vs['dat'], tmp3)
                else:
                    vspeak = data[i, 6]
                    vsfreq = data[i, 7]
                    new_row = np.array([np.nan, vsfreq, vspeak])
                    Vs['dat'] = np.vstack((Vs['dat'], new_row))
                new_row = np.array([nodebran[i, 0:3]])
                Vs['pos'] = np.vstack((Vs['dat'], new_row)) # 纵向
            elif firstColumnValue == 'vcis':
                G['list'] = np.vstack((G['list'],data[i, 3:5]))  # Node name
                G['listdex'] = np.vstack((G['listdex'], [nodebran[i, 1], nodebran[i, 2], data[i, 5], 0]))
            elif firstColumnValue == 'icvs':
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i, 0] - 1, nodebran[i, 2], +1)
                A = self.Assign_A_Value(A, nodebran[i + 1, 0] - 1, nodebran[i + 1, 1] - 1, -1)
                A = self.Assign_A_Value(A, nodebran[i + 1, 0] - 1, nodebran[i + 1, 2] - 1, +1)

                firstColumnValue = data[i + 1, 0]
                if firstColumnValue == 'RR':
                    R[nodebran[i, 0] - 1, nodebran[i + 1, 0] - 1] = data[i, 5]
                    R[nodebran[i + 1, 0] - 1, nodebran[i + 1, 0] - 1] = data[i + 1, 5]
                elif firstColumnValue == 'RL':
                    R[nodebran[i, 0] - 1, nodebran[i + 1, 0] - 1] = data[i, 5]
                    L[nodebran[i + 1, 0] - 1, nodebran[i + 1, 0] - 1] = data[i + 1, 5]
                elif firstColumnValue == 'LR':
                    L[nodebran[i, 0] - 1, nodebran[i + 1, 0] - 1] = data[i, 5]
                    R[nodebran[i + 1, 0] - 1, nodebran[i + 1, 0] - 1] = data[i + 1, 5]
                elif firstColumnValue == 'LL':
                    L[nodebran[i, 0] - 1, nodebran[i + 1, 0] - 1] = data[i, 5]
                    L[nodebran[i + 1, 0] - 1, nodebran[i + 1, 0] - 1] = data[i + 1, 5]

            elif firstColumnValue == 'ML':
                Vair = 3e8
                Dist = data[i:i + Ncom - 1, 5]
                High = data[i:i + Ncom - 1, 6]
                r0 = np.mean(data[i:i + Ncom - 1, 7])
                L0, C0 = self.Cal_LC_OHL(High, Dist, r0)            # 调用函数 Cal_LC_OHL，并将距离、高度和平均半径作为参数传递给它，以计算线路的电感（L0）和电容（C0）
                R = L0 * Vair            # 使用计算得到的值 L0 和 Vair 计算搜杆阻抗（R）
                A = np.eye(Ncom)            # 初始化矩阵 A 为单位矩阵，大小为 Ncom

        C['list'] = self.Assign_Suffix(C['list'], app)
        G['list'] = self.Assign_Suffix(G['list'], app)
        C['list'] = np.array(C['list'], dtype=str)
        C['listdex'] = np.array(C['listdex'])
        G['list'] = np.array(G['list'], dtype=str)
        G['listdex'] = np.array(G['listdex'])
        Vs['dat'] = np.array(Vs['dat'])
        Vs['pos'] = np.array(Vs['pos'])
        Is['dat'] = np.array(Is['dat'])
        Is['pos'] = np.array(Is['pos'])
        Nle['dat'] = np.array(Nle['dat'])
        Nle['pos'] = np.array(Nle['pos'])
        Swh['dat'] = np.array(Swh['dat'])
        Swh['pos'] = np.array(Swh['pos'])


        CK_Para = {'Info': Info,
                   'app': Blok['sysname'],
                   'A': A,  # float
                   'R': R,  # list
                   'L': L,
                   'C': C,
                   'G': G,
                   'Vs': Vs,
                   'Is': Is,
                   'Nle': Nle,
                   'Swh': Swh}

        return CK_Para, Node, Bran, Meas

    def Tower_A2G_Bridge(self,CK_Para, Node, Bran, Blok):
        # Building GRound surface model connecting air-node/gnd-node blocks

        if Blok["flag"][0] != 2:
            return CK_Para, Bran, Blok

        # (1) Initialization
        Rmin = 1e-6
        Nn = Node["num"][0]  # total # of Wire Node
        Nb = Bran["num"][0]  # total # of Wire Bran
        a2g = Blok["a2g"]  # Air-2-Gnd Bridge:
        list_a2g = a2g["list"]  # [b0 air_node gnd_node]
        listdex = []
        Np = len(list_a2g)   # Number of A2G Bridges

        A = CK_Para["A"]
        R = CK_Para["R"]
        L = CK_Para["L"]

        # (2) Updating Bran.list/listdex/num and Blok.agb
        for i in range(Np):  # In Python, range starts from 0 and goes to Np-1
            # Find indices in Node.list that match the elements in list_a2g
            # Assuming Node['list'] is a list or array
            tmp1 = [j for j, x in enumerate(Node['list']) if x == list_a2g[i][1]]
            tmp2 = [j for j, x in enumerate(Node['list']) if x == list_a2g[i][2]]

            # In Python, lists are zero-indexed, so we add Nb (assuming it's adjusted for Python's 0-indexing)
            # We append the found indices to listdex
            listdex.append([i + Nb] + tmp1 + tmp2)

        Blok["a2g"]["listdex"] = np.array(listdex)
        Bran["a2g"] = Blok["a2g"]

        # Update Bran.num
        tmp0 = np.array([Np, 0, 0, 0, 0, Np])
        Bran["num"] = Bran["num"] + tmp0

        # Update Bran.list
        # Convert list_a2g to a NumPy array and concatenate it with Bran['list']
        list_a2g_array = np.array(list_a2g)
        Bran['list'] = np.concatenate((Bran['list'], list_a2g_array))

        # Update Bran.listdex
        # Assuming listdex is a list of lists, convert it to a NumPy array first
        listdex_array = np.array(listdex)
        Bran['listdex'] = np.concatenate((Bran['listdex'], listdex_array))

        # (3) Update A, R, and L
        Asf = np.zeros((Np, Nn))
        Rsf = np.eye(Np) * Rmin
        Lsf = np.zeros((Np, Np))

        for i in range(Np):
            # Update Asf matrix based on listdex
            # -1 for air to ground, and +1 for air to ground
            Asf[i, listdex[i][1]] = -1  # air -> gnd
            Asf[i, listdex[i][2]] = 1   # air -> gnd

        # Concatenating A with Asf
        A = np.vstack((A, Asf))

        # Creating block diagonal matrices for R and L
        R = np.block([[R, np.zeros((R.shape[0], Rsf.shape[1]))],
                      [np.zeros((Rsf.shape[0], R.shape[1])), Rsf]])
        L = np.block([[L, np.zeros((L.shape[0], Lsf.shape[1]))],
                      [np.zeros((Lsf.shape[0], L.shape[1])), Lsf]])

        # (4) CK-Para Updating
        CK_Para["A"] = A
        CK_Para["R"] = R
        CK_Para["L"] = L

        return CK_Para, Bran, Blok

    def Assign_A_Value(self, A, b1, n1, val):
        if n1 >= 0:
            val = int(val)
            A[b1, n1] = val
        return A.astype(int)

    def Assign_Suffix(self, NameList, app):
        # Attach suffix to the namelist with the size of (Nrow x Mcol)
        NameList = NameList.astype(np.dtype(str))
        if len(NameList) == 0:
            out = NameList
        else:
            condition = (NameList != "") & (NameList != " ") & (NameList != "0")
            NameList = np.where(condition, np.char.add(NameList, app), NameList)

            out = NameList
        # if isinstance(NameList, list):
        #
        #     if not NameList:
        #         return NameList
        #     else:
        #         out = [i + app for i in NameList]
        # else:
        #     out = np.char.add(NameList, app)

        return out

    def Assign_C_Value(self, C, Clist):
        """
        return C = val (+/-val) at pos (n1, n2) if n1/2~=0, and
        = no change if n1/2 = 0
        """

        if len(Clist) == 0:
            return C

        Nrow = Clist.shape[0]
        N1, N2, VAL = np.array(Clist[:, 0], dtype=int), np.array(Clist[:, 1],dtype=int), Clist[:, 2]

        for i in range(Nrow):
            n1, n2, val = N1[i] - 1, N2[i] - 1, VAL[i]
            if (n1 + 1) != 0:
                C[n1, n1] = C[n1, n1] + val
            if (n2 + 1) != 0:
                C[n2, n2] = C[n2, n2] + val
            if (n1 + 1) * (n2 + 1) != 0:
                C[n1, n2] = C[n1, n2] - val
                C[n2, n1] = C[n2, n1] - val

        return C

    def Assign_G_Value(self, G, Glist):
        """
        return G = val (+/-val) at pos (n1, n2) if n1/2~=0, and
        = no change if n1/2 = 0
        """
        if len(Glist) == 0:
            return G

        Nrow = Glist.shape[0]
        N1, N2, VAL, MOD = Glist[:, 0], Glist[:, 1], Glist[:, 2], Glist[:, 3]

        for i in range(Nrow):
            n1, n2, val, mod = N1[i] - 1, N2[i] - 1, VAL[i], MOD[i]
            if mod == 2:
                if (n1 + 1) != 0:
                    G[n1, n1] = G[n1, n1] + val
                if (n2 + 1) != 0:
                    G[n2, n2] = G[n2, n2] + val
                if (n1 + 1) * (n2 + 1) != 0:
                    G[n1, n2] = G[n1, n2] - val
                    G[n2, n1] = G[n2, n1] - val
            elif mod == 1:
                G[n1, n2] = G[n1, n2] + val  # controlled Is source

        return G

    def Inf_Node_Update(self, listdex, oftn):
        # Reset id of 0 (oftn) to be 0 after merging
        id_tmp = np.where(listdex == oftn)
        listdex[id_tmp] = 0
        return listdex

    def Tower_Map_Init(self):
    # Initialization of T2Xmap
        T2Xmap = {
            'head': [],  # Mapping table cell
            'hspn': 0,   # # of table cells
            'hsid': [],  # id of spans/cables
            'tail': [],
            'tspn': 0,
            'tsid': []
        }
        return T2Xmap

    def Node_Bran_Index_Lump(self, data, NodeCom, Nwir):
        """
        :param data: 除节点信息和END行的其他关键信息
        :param NodeCom:公共节点
        :param Nwir:
        :return:返回所有Node和所有Bran的name与index
        """
        # (1) Initialization
        # 将字典中的列表转换为NumPy数组，并指定行数和列数
        Meas = {"list": np.array([], dtype=str), "listdex": np.array([], dtype=int),
                "flag": np.array([], dtype=int)}
        num_rows = len(NodeCom['list'])  # 行数与列表长度一致
        num_columns = 1  # 列数为1
        NodeCom['list'] = np.array(NodeCom['list']).reshape(num_rows, num_columns)
        NodeCom['listdex'] = np.array(NodeCom['listdex']).reshape(num_rows, num_columns)
        if not NodeCom:
            Node = {'list': np.array([]), 'listdex':  np.array([])}
            oftn = 0
        else:
            Node = {'list': np.array([]), 'listdex':  np.array([])}
            oftn = len(NodeCom['listdex'])  # 节点个数
            Node['list'] = NodeCom['list'][:]
            Node['listdex'] = NodeCom['listdex'][:]

        Node['pos'] = np.array([])
        Bran = {'pos': [], 'list': [], 'num': np.zeros(6, dtype=int), 'listdex': None}

        # (2) Assign node id and bran id for every line
        namelist = data[:, 2:5]
        namelist = np.array(namelist, dtype=str)    # string array of [b0 n1 n2]
        # namelist = namelist.reset_index(drop=True)  # string array of [b0 n1 n2]
        Nrow = data.shape[0]

        # (2a) find out bran.list/listdex/num
        elist = ["RL", "R", "L", "nle", "swh", "Vs"]  # single line for Bran counting
        Bran['list'] = pd.DataFrame()
        oftb = 0

        for i in range(Nrow):
            first_str = data[i, 0]  # 1st field = char
            if first_str in elist:
                iter_df = np.array(namelist[i, 0:3], dtype=str)
                Bran['list'][i] = iter_df
                oftb += 1
            if first_str == "icvs":  # "icvs" (1+2 lines)
                Bran['list'].append(namelist[i:i + 2, 0:3])
                oftb += 2
            if first_str == "M2":  # "M2" (1+3 lines)
                Bran['list'].append(namelist[i:i + 3:2, 0:3])
                oftb += 2
            if first_str == "M3":  # "M3" (1+4+6 lines)
                Bran['list'].append(namelist[i:i + 6:2, 0:3])
                oftb += 3
            if first_str == "ML": # Get matching pole impedance network (a)
                Bran['list'].append(namelist[i:i + oftn,0:3])
                oftb = oftb + oftn


        Bran['list'] = np.array(Bran['list'], dtype=str)
        Bran['list'] = np.transpose(Bran['list'])
        Bran["num"] = np.array([oftb, 0, 0, oftb, 0, 0], dtype=int)  # wire parameters

        # (2a) find out Node.list/listdex/num and bran.list/listdex
        # tmp0第一个数返回支路的索引，第二和第三个数返回节点的索引，感觉做一下mapping就行了？
        nodebran = np.zeros((Nrow, 3), dtype=int)  # 初始化nodebran
        for i in range(Nrow):
            tmp0 = np.zeros(3, dtype=int)  # [b0 n1 n2]
            # (1) bran indexing
            first_str = namelist[i, 0]  # bran name
            if first_str != " " and first_str != "" and not pd.isna(first_str):
                index = np.where(np.array(Bran['list'])[:, 0] == first_str)[0]
                if len(index) > 0:
                    tmp0[0] = index[0] + 1

            # (2) node indexing
            for j in range(2):  # for one node of two
                first_str = namelist[i, j + 1]  # node name in con. 4  & 5
                if first_str != " " and first_str != "" and not pd.isna(first_str):
                    tmp1 = np.where(np.array(Node['list']) == first_str)[0]  # string array
                    tmp1 += 1
                    if len(tmp1) == 0:
                        Node['list'] = np.append(Node['list'], first_str)
                        Node['listdex'] = np.append(Node['listdex'],oftn + 1)
                        tmp0[j + 1] = oftn + 1
                        oftn += 1
                    else:
                        tmp0[j + 1] = tmp1[0]
            nodebran[i, :] = tmp0  # [b0 n1 n2] for all lines

        Node['num'] = [oftn, oftn, 0, 0]
        Node['list'] = Node['list'].reshape(-1, 1)  #将其强行转换为列
        Node['listdex'] = Node['listdex'].reshape(-1, 1)  #将其强行转换为列

        Bran['listdex'] = np.zeros((oftb, 3), dtype=int)
        # Bran['listdex'][:, 0] = np.arange(0, oftb)
        Bran['listdex'][:, 0] = np.arange(1, oftb + 1) # 与matlab一致
        Bran = self.Assign_Elem_id(Node, Bran, [1, 2])

        # (3) Measurement list
        Ncop = Nwir['comp']  # Nwir.comp = nrow - nmea;nmea是mea行的value
        # 定义Meas
        tmp = data[:, 1]  # 获取数据中的第二列
        Itmp = [i for i, val in enumerate(tmp) if val > 0]  # 找到大于0的数值的索引
        Meas["list"] = np.array([namelist[i] for i in Itmp])  # 根据索引获取 namelist 中的对应项
        Meas["listdex"] = np.array([nodebran[i, :3] for i in Itmp] ) # 根据索引获取 nodebran 中的对应项
        Meas['flag'] = np.array([tmp[i] for i in Itmp])  # 根据索引获取 tmp 中的对应项
        # Meas = self.Assign_Elem_id(Bran, Meas, 1) # not necessary！

        return Node, Bran, Meas, nodebran

    def Lump_Souce_Update(self, CK_Te, CK_Xe, NodeX, oftn, oftb):
        """
        Update CK_element (Vs, Is, Nle, and Swh) of Tower by appending Sub_CK data
        with offset No.: oftn and oftb
        CK_T['Vs']['dat'/'pos']: Tower-CK element to be updated with CK_X
        CK_X['Vs']['dat'/'pos']: Sub-CK element
        Ncol = 1 for Is (n1), and
        Ncol = 3 for Vs, Nle, and Swh [b0 n1 n2]
        """

        if CK_Xe['dat'].size == 0:
            return CK_Te

        Ncom = NodeX['com'].shape[0]
        Ncol = CK_Xe['pos'].shape[1]

        if Ncol == 1:
            Xref = CK_Xe['pos'].copy()
        elif Ncol == 3:
            Xref = CK_Xe['pos'][:, 1:3]

        Locelem = []
        Valelem = []
        Xref_rownum = Xref.shape[0]
        Xref_colnum = Xref.shape[1]
        # (1) Record the position of common nodes in Sub_CK list
        for i in range(Ncom):
            num1, num2 = NodeX['comdex'][i, :2]
            tmp0 = []
            for m in range(Xref_colnum):
                for n in range(Xref_rownum):
                    if Xref[n, m] == num2:
                        tmp0.append([n, m])  # Vs
            # tmp0 = np.where(Xref == num2)  # Vs
            if len(tmp0) != 0:
                Locelem.extend(tmp0)  # store position for update
                Valelem.extend(np.full(len(tmp0), num1))  # store value for update
        Locelem = np.array(Locelem)
        Valelem = np.array(Valelem)

        # (2) Update node id in CK_Xe.pos
        Xref = Xref + oftn
        Xref[Locelem[:, 0], Locelem[:, 1]] = Valelem  # Replace Com_node id with GLB_node id

        # (3) Update GLB CK elements (CK_Te.dat and pos)
        if Ncol == 1:
            if len(CK_Te) == 0:
                CK_Te = {'pos': [], 'dat': []}
                CK_Te['pos'] = Xref.copy()
                CK_Te['dat'] = CK_Xe['dat'][:]
            else:
                CK_Te['pos'] = np.concatenate((CK_Te['pos'], Xref))  # Is
                CK_Te['dat'] = np.concatenate((CK_Te['dat'], CK_Xe['dat']))
        elif Ncol == 3:
            Xban = CK_Xe['pos'][:, 0] + oftb
            if len(CK_Te) == 0:
                CK_Te = {'pos': np.column_stack((Xban, Xref)), 'dat': []}
                CK_Te['dat'] = CK_Xe['dat'][:]
            else:
                CK_Te['pos'] = np.concatenate((CK_Te['pos'], np.column_stack((Xban, Xref))))  # Vs Nle and Swh
                CK_Te['dat'] = np.concatenate((CK_Te['dat'], CK_Xe['dat']))

        return CK_Te

    def Tower_CK_Update(self, CK_Para, Node, Bran, Meas, CK_X, NodeX, BranX, MeasX, Flag):
        """
        Update CK_Para(A, R, L, C, G, Vs, Is, Nle and Swh), Node, Bran, Meas
        with CK_X, NodeX, BranX, MeasX using NodeX.com / comdex
        CK_X['C']['list'] = [n1 n2] / listdex = [n1 n2 val]
        CK_X['G']['list'] = [n1 n2] / listdex = [n1 n2 val mod], mod = 1(vcis), 2 = G
        Special issue: (1) 0 in n1 and n2 retains 0 after megering
        (2) duplicate common nodes should be deleted after merg.
        """

        if Flag == 0:
            return CK_Para, Node, Bran, Meas

        # (0) Update Node/Bran.num
        Ncom = len(NodeX['com'])
        Node['num'] = Node['num'] + NodeX['num'] - [Ncom, Ncom, 0, 0]
        Bran['num'] = Bran['num'] + BranX['num']

        # Part A:
        # (1) update node/bran id in NodeX/BranX/MeasX (Sub-CK)
        oftn = len(Node['list']) - Ncom
        oftb = len(Bran['list'])
        NodeX['listdex'] = NodeX['listdex'] + oftn
        BranX['listdex'][:, 0] = BranX['listdex'][:, 0] + oftb
        BranX['listdex'][:, 1:3] = BranX['listdex'][:, 1:3] + oftn
        MeasX['listdex'][:, 0] = MeasX['listdex'][:, 0] + oftb
        MeasX['listdex'][:, 1:3] = MeasX['listdex'][:, 1:3] + oftn

        if len(CK_X['C']['list']) != 0:
            CK_X['C']['listdex'][:, :2] = CK_X['C']['listdex'][:, :2] + oftn
            CK_X['C']['listdex'][:, :2] = self.Inf_Node_Update(CK_X['C']['listdex'][:, :2], oftn)

        if len(CK_X['G']['list']) != 0:
            CK_X['G']['listdex'][:, :2] = CK_X.G.listdex[:, :2] + oftn
            CK_X['G']['listdex'][:, :2] = self.Inf_Node_Update(CK_X['G']['listdex'][:, :2], oftn)

        BranX['listdex'][:, 1:3] = self.Inf_Node_Update(BranX['listdex'][:, 1:3], oftn)
        MeasX['listdex'][:, 1:3] = self.Inf_Node_Update(MeasX['listdex'][:, 1:3], oftn)

        # (2) Replace common nodes of NoteX/BranX/MeasX/CK_X.C/G with GLB node
        for i in range(Ncom):
            str1, str2 = NodeX['com'][i, :2]
            num1 = NodeX['comdex'][i, 0]

            # Node updating
            tmp0 = np.where(np.array(NodeX['list'], dtype=str) == str2)
            NodeX['list'][tmp0] = str1
            NodeX['listdex'][tmp0] = num1

            # Node (C.list/listdex) updating
            if len(CK_X['C']['list']) != 0:
                tmp0 = np.where(np.array(CK_X['C']['list'], dtype=str) == str2)
                CK_X['C']['list'][tmp0] = str1
                CK_X['C']['listdex'][tmp0] = num1

            # Node (G.list/listdex) updating
            if len(CK_X['G']['list']) != 0:
                tmp0 = np.where(np.array(CK_X['G']['list'], dtype=str) == str2)
                CK_X['G']['list'][tmp0] = str1
                CK_X['G']['listdex'][tmp0] = num1

            # Bran updating
            tmp0 = np.where(np.array(BranX['list'], dtype=str) == str2)
            BranX['list'][tmp0] = str1
            BranX['listdex'][tmp0] = num1

            # Meas updating
            tmp0 = np.where(np.array(MeasX['list'], dtype=str) == str2)
            MeasX['list'][tmp0] = str1
            MeasX['listdex'][tmp0] = num1

        # Part B:
        # (1) Updating overall Note/Bran/Meas
        Node['list'] = np.concatenate((Node['list'], NodeX['list']), axis=0)
        Node_list_unique, original_indices1 = np.unique(Node['list'], axis=0, return_index=True)
        original_indices1 = np.sort(original_indices1)
        Node['list'] = Node['list'][original_indices1]  # 使用unique且不改变原顺序

        Bran['list'] = np.concatenate((Bran['list'], BranX['list']), axis=0)
        Meas['list'] = np.concatenate((Meas['list'], MeasX['list']), axis=0)

        Node['listdex'] = np.concatenate((Node['listdex'], NodeX['listdex']), axis=0)
        Node_listdex_unique, original_indices2 = np.unique(Node['listdex'], axis=0, return_index=True)
        original_indices2 = np.sort(original_indices2)
        Node['listdex'] = Node['listdex'][original_indices2]


        Bran['listdex'] = np.concatenate((Bran['listdex'], BranX['listdex']), axis=0)
        Meas['listdex'] = np.concatenate((Meas['listdex'], MeasX['listdex']), axis=0)
        Meas['flag'] = np.concatenate((Meas['flag'], MeasX['flag']), axis=0)

        # (2) Updating Vs Is Nle and Swh
        CK_Para['Vs'] = self.Lump_Souce_Update(CK_Para['Vs'], CK_X['Vs'], NodeX, oftn, oftb)
        CK_Para['Is'] = self.Lump_Souce_Update(CK_Para['Is'], CK_X['Is'], NodeX, oftn, oftb)
        CK_Para['Nle'] = self.Lump_Souce_Update(CK_Para['Nle'], CK_X['Nle'], NodeX, oftn, oftb)
        CK_Para['Swh'] = self.Lump_Souce_Update(CK_Para['Swh'], CK_X['Swh'], NodeX, oftn, oftb)

        # (3) Update Circuit Parameters
        CK_Para['R'] = np.block([[CK_Para['R'], np.zeros((CK_Para['R'].shape[0], CK_X['R'].shape[1]))],
                                 [np.zeros((CK_X['R'].shape[0], CK_Para['R'].shape[1])), CK_X['R']]])

        CK_Para['L'] = np.block([[CK_Para['L'], np.zeros((CK_Para['L'].shape[0], CK_X['L'].shape[1]))],
                                 [np.zeros((CK_X['L'].shape[0], CK_Para['L'].shape[1])), CK_X['L']]])

        # A = np.block([[CK_Para['A'], CK_X['A']]])
        A = np.block([[CK_Para['A'], np.zeros((CK_Para['A'].shape[0], CK_X['A'].shape[1]))],
                      [np.zeros((CK_X['A'].shape[0], CK_Para['A'].shape[1])), CK_X['A']]])
        for i in range(Ncom):
            num1, num2 = NodeX['comdex'][i, :2]
            A[:, num1-1] = A[:, num1-1] + A[:, oftn + Ncom + num2-1]

        del_col = NodeX['comdex'][:, 1] + oftn + Ncom -1  #要删掉的列
        A = np.delete(A, del_col, axis=1)
        CK_Para['A'] = A

        Noft = Node['num'][0] - (oftn + Ncom)
        Ca = np.zeros((Noft, Noft))
        Cx = np.block([[CK_Para['C'], np.zeros((CK_Para['C'].shape[0], Ca.shape[1]))],
               [np.zeros((Ca.shape[0], CK_Para['C'].shape[1])), Ca]])
        CK_Para['C'] = self.Assign_C_Value(Cx, CK_X['C']['listdex'])

        Ga = np.zeros((Noft, Noft))
        Gx = np.block([[CK_Para['G'], np.zeros((CK_Para['G'].shape[0], Ga.shape[1]))],
               [np.zeros((Ga.shape[0], CK_Para['G'].shape[1])), Ga]])
        CK_Para['G'] = self.Assign_G_Value(Gx, CK_X['G']['listdex'])

        return CK_Para, Node, Bran, Meas

    def Tower_Map_Update(self, TH_T2Xmap, TT_T2Xmap, X2Tmap):
        # (1) Head mapping table update
        TH_T2Xmap['head'].append(X2Tmap['head'])
        TH_T2Xmap['hspn'] += 1
        TH_T2Xmap['hsid'].append(X2Tmap['head'][0][0])

        # Arrange in the ascend order of sid
        tp, Itp = zip(*sorted(zip(TH_T2Xmap['hsid'], range(len(TH_T2Xmap['hsid'])))))
        TH_T2Xmap['head'] = [TH_T2Xmap['head'][i] for i in Itp]
        # zip创建了一个元组的迭代器，其中每个元组包含了TH_T2Xmap['hsid']列表中的一个元素和相应的索引。
        # sorted对这些元组进行排序，排序的依据是元组中第一个元素，即TH_T2Xmap['hsid']列表的元素值。
        # zip将排序后的元组进行解压缩，得到两个独立的元组，分别包含了排序后的TH_T2Xmap['hsid']元素和相应的排序后的索引位置。

        # (2) Tail mapping table update
        TT_T2Xmap['tail'].append(X2Tmap['tail'])
        TT_T2Xmap['tspn'] += 1
        TT_T2Xmap['tsid'].append(X2Tmap['tail'][0][0])

        # Arrange in the ascend order of sid
        tp, Itp = zip(*sorted(zip(TT_T2Xmap['tsid'], range(len(TT_T2Xmap['tsid'])))))
        TT_T2Xmap['tail'] = [TT_T2Xmap['tail'][i] for i in Itp]

        return TH_T2Xmap, TT_T2Xmap
    def Wire_AssignAValue(self, A, b1, n1, val):
        if n1 >= 0:
            val = int(val)
            A[b1, n1] = val
        return A.astype(int)

    def Tower_Circuit_Update(self, Tower):
        CK_Para = Tower['CK_Para']
        GND = Tower['GND']
        Blok = Tower['Blok']
        Bran = Tower['Bran']
        Node = Tower['Node']
        Meas = Tower['Meas']

        # (5a) Read Circuit Modules
        Bflag = Blok["flag"]
        Bname = Blok["name"]
        CKins, Nins, Bins, Mins = self.Lump_Model_Intepret(Bflag, Bname, 2 - 1, Blok["ins"])
        CKsar, Nsar, Bsar, Msar = self.Lump_Model_Intepret(Bflag, Bname, 3 - 1, Blok["sar"])
        CKtxf, Ntxf, Btxf, Mtxf = self.Lump_Model_Intepret(Bflag, Bname, 4 - 1, Blok["txf"])
        CKgrd, Ngrd, Bgrd, Mgrd = self.Lump_Model_Intepret(Bflag, Bname, 5 - 1, Blok["grd"])
        CKint, Nint, Bint, Mint = self.Lump_Model_Intepret(Bflag, Bname, 6 - 1, Blok['int'])
        CKinf, Ninf, Binf, Minf = self.Lump_Model_Intepret(Bflag, Bname, 7 - 1, Blok['inf'])
        CKmck, Nmck, Bmck, Mmck = self.Lump_Model_Intepret(Bflag, Bname, 8 - 1, Blok['mck'])
        CKoth1, Noth1, Both1, Moth1 = self.Lump_Model_Intepret(Bflag, Bname, 9 - 1, Blok['oth1'])
        CKoth2, Noth2, Both2, Moth2 = self.Lump_Model_Intepret(Bflag, Bname, 10 - 1, Blok['oth2'])

        # (5b) Update Tower_CK_Block
        CK_Para, Node, Bran, Meas = self.Tower_CK_Update(CK_Para, Node, Bran, Meas, CKins, Nins, Bins, Mins, Bflag[1])
        CK_Para, Node, Bran, Meas = self.Tower_CK_Update(CK_Para, Node, Bran, Meas, CKsar, Nsar, Bsar, Msar, Bflag[2])
        CK_Para, Node, Bran, Meas = self.Tower_CK_Update(CK_Para, Node, Bran, Meas, CKtxf, Ntxf, Btxf, Mtxf, Bflag[3])
        CK_Para, Node, Bran, Meas = self.Tower_CK_Update(CK_Para, Node, Bran, Meas, CKgrd, Ngrd, Bgrd, Mgrd, Bflag[4])
        CK_Para, Node, Bran, Meas = self.Tower_CK_Update(CK_Para, Node, Bran, Meas, CKint, Nint, Bint, Mint, Bflag[5])
        CK_Para, Node, Bran, Meas = self.Tower_CK_Update(CK_Para, Node, Bran, Meas, CKinf, Ninf, Binf, Minf, Bflag[6])
        CK_Para, Node, Bran, Meas = self.Tower_CK_Update(CK_Para, Node, Bran, Meas, CKmck, Nmck, Bmck, Mmck, Bflag[7])
        CK_Para, Node, Bran, Meas = self.Tower_CK_Update(CK_Para, Node, Bran, Meas, CKoth1, Noth1, Both1, Moth1, Bflag[8])
        CK_Para, Node, Bran, Meas = self.Tower_CK_Update(CK_Para, Node, Bran, Meas, CKoth2, Noth2, Both2, Moth2, Bflag[9])

        # % fixing bugs in  C matrix
        # Find diagonal is 0 and assign 1e-12
        array_C = CK_Para.get('C')          # 获取 'C' 对应的数组
        if array_C is not None:             # 检查 'C' 是否存在且为二维数组
            num_rows = len(array_C)         # 获取数组的行数和列数
            num_cols = len(array_C[0])
            for i in range(min(num_rows, num_cols)):            # 遍历对角线上的元素，如果为零，则赋值为 1e-12
                if array_C[i][i] == 0:
                    array_C[i][i] = 1e-12
        CK_Para['C'] = array_C

        # Update Ground Conductance G
        CK_Para['G'], CK_Para['P'] = self.Ground_Conductance(CK_Para, Node, GND)

        # Delete psuedo bran in A (all-zero row)
        if Blok['flag'][0] == 0:
            row_all_zeros = np.where(np.all(CK_Para['A'] == 0, axis=1))[0]
            CK_Para['A'] = np.delete(CK_Para['A'], row_all_zeros, axis=0)

        # group meas according to its flag=1(I), 2(V), 3(I/V), 4(P), 5(V/I/P) 11(E)
        mrow = Meas['flag'].shape[0]
        Meas['Ib'] = np.empty((0,1), dtype=int)
        Meas['Vn'] = np.empty((0,2), dtype=int)
        Meas['Pw'] = np.empty((0,3), dtype=int)
        Meas['En'] = np.empty((0,3), dtype=int)
        Meas['IbList'] = np.empty((0,1), dtype=str)
        Meas['VnList'] = np.empty((0,2), dtype=str)
        Meas['PwList'] = np.empty((0,3), dtype=str)
        Meas['EnList'] = np.empty((0,3), dtype=str)

        for ik in range(mrow):
            flag = Meas['flag'][ik]
            if flag == 1:  # current
                Meas['Ib'] = np.vstack((Meas['Ib'], Meas['listdex'][ik, 0]))
                Meas['IbList'] = np.vstack((Meas['IbList'], Meas['list'][ik, 0]))
            elif flag == 2:  # voltage
                Meas['Vn'] = np.vstack((Meas['Vn'], Meas['listdex'][ik, 1:3]))
                Meas['VnList'] = np.vstack((Meas['VnList'], Meas['list'][ik, 1:3]))
            elif flag == 3:  # power
                Meas['Pw'] = np.vstack((Meas['Pw'], Meas['listdex'][ik]))
                Meas['PwList'] = np.vstack((Meas['PwList'], Meas['list'][ik]))
            elif flag == 4:  # power, current, voltage
                Meas['Ib'] = np.vstack((Meas['Ib'], Meas['listdex'][ik, 0]))
                Meas['IbList'] = np.vstack((Meas['IbList'], Meas['list'][ik, 0]))
                Meas['Vn'] = np.vstack((Meas['Vn'], Meas['listdex'][ik, 1:3]))
                Meas['VnList'] = np.vstack((Meas['VnList'], Meas['list'][ik, 1:3]))
                Meas['Pw'] = np.vstack((Meas['Pw'], Meas['listdex'][ik]))
                Meas['PwList'] = np.vstack((Meas['PwList'], Meas['list'][ik]))
            elif flag == 11:  # Energy
                Meas['En'] = np.vstack((Meas['En'], Meas['listdex'][ik]))
                Meas['EnList'] = np.vstack((Meas['EnList'], Meas['list'][ik]))
            else:
                print('!!! Not used in Tower Circuit Update (Meas)')

        Tower['CK_Para'] = CK_Para
        Tower['Bran'] = Bran
        Tower['Node'] = Node

        return  Tower

    def Ground_Conductance(self, CK_Para, Node, GND):
        ep0 = 8.854187818e-12
        epr = GND['epr']
        k = ep0 / GND['sig']

        P = CK_Para['P'].copy()
        G = CK_Para['G'].copy()

        ns = Node['num'][3] + 1             # starting node id
        ne = Node['num'][3] + Node['num'][2]  # ending node id
        nn = Node['num'][2]                # ground node #

        Pg = P[ns - 1:ne, ns - 1:ne]            # potential coefficience of ground node

        G[ns - 1:ne, ns - 1:ne] = k * Pg        # ground conductance
        P[ns - 1:ne, ns - 1:ne] = Pg / epr      # ground potential

        return G, P

    def Assign_Elem_Lump(self, node_tower, node_lump, oft):
        if not node_lump:
            return node_tower, oft

        nt = node_tower
        tmp = [0, 0, 0]
        for ik in range(len(node_lump['listdex'])):
            if node_lump['list'][ik] not in nt['list']:
                nt['list'].append(node_lump['list'][ik])
                nt['listdex'].append(node_lump['listdex'][ik] + oft)
                nt['pos'].append(tmp)
                nt['num'][0] += 1
                nt['num'][1] += 1

        oft += len(node_lump['listdex'])
        return nt, oft

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

    def Tower_Line_Connect(self, Tower, Span, Cable):
        # Build a model of connecting line (Span/Cable) to a Tower using meas. bran
        #       meas bran = Line node -> Tower node
        #       Cwx.id =[S/C, Span_id, CK_id, in/out, tower_nid],
        #       Cwx.C0;
        #       with T2S or T2C mapping table (X=[]: span, X=C: cable)

        # Part A: Span only
        # (A0) Init
        T2Smap = Tower['T2Smap']
        Cw = {}
        Cw_id = []
        Cw_C0 = []
        # (A1) Obtain Cw for OHL
        hspn = T2Smap['hspn']
        hsid = T2Smap['hsid']
        tspn = T2Smap['tspn']
        tsid = T2Smap['tsid']
        for jk in range(tspn):
            map = T2Smap['tail'][jk][1:]
            nr = len(map)
            Itmp = np.ones((nr, 1))
            tmp = np.column_stack((Itmp, Itmp * tsid[jk], Itmp * 0, Itmp, map[:, 1]))
            Cw_id.extend(tmp)

            sid = int(tsid[jk]) - 1
            Line = Span[sid]['OHLP']
            High = Line[:, 5]  # height
            Dist = Line[:, 6]  # Horizontal offset
            r0 = Line[:, 7]  # conductor radius
            L, C = self.Cal_LC_OHL(High, Dist, r0)
            Cw_C0.extend(np.diag(C).tolist())  # diagnal elements only

        for jk in range(hspn):
            map = T2Smap['head'][jk][1:]
            nr = len(map)
            Itmp = np.ones((nr, 1))
            tmp = np.column_stack((Itmp, Itmp * hsid[jk], Itmp * 0, -Itmp, map[:, 1]))
            Cw_id.extend(tmp)

            sid = int(hsid[jk]) - 1
            Line = Span[sid]['OHLP']
            High = Line[:, 2]  # height
            Dist = Line[:, 6]  # Horizontal offset
            r0 = Line[:, 7]  # conductor radius
            L, C = self.Cal_LC_OHL(High, Dist, r0)
            Cw_C0.extend(np.diag(C).tolist())  # diagnal elements only

        # Part B: CABLE only
        # (B0) Init
        T2Cmap = Tower['T2Cmap']
        Cwc = {}
        Cwc_C0 = []
        Cwc_id = []

        # (B2) Obtain Cw for CABLE
        hspn = T2Cmap['hspn']
        hsid = T2Cmap['hsid']
        tspn = T2Cmap['tspn']
        tsid = T2Cmap['tsid']

        for jk in range(tspn):
            map = T2Cmap['tail'][jk][1:]
            nr = len(map)
            Itmp = [1] * nr
            tmp = [[Itmp[i], Itmp[i] * tsid[jk], 0, Itmp[i], map[i]] for i in range(nr)]
            Cwc_id.extend(tmp)
            tsid = int(tsid)
            Para = Cable[tsid]['Para']
            Cwc_C0.extend(Para['Cw']['C0'])

        for jk in range(hspn):
            map = T2Cmap['head'][jk][1:]
            nr = len(map)
            Itmp = [1] * nr
            tmp = [[Itmp[i], Itmp[i] * hsid[jk], 0, -Itmp[i], map[i]] for i in range(nr)]
            Cwc_id.extend(tmp)
            hsid = int(hsid)
            Para = Cable[hsid]['Para']
            Cwc_C0.extend(Para['Cw']['C0'])

        Cw = {"Cw_id": Cw_id, 'Cw_C0': Cw_C0}
        Cwc = {"Cwc_id": Cwc_id, "Cwc_C0": Cwc_C0}

        return Cw, Cwc