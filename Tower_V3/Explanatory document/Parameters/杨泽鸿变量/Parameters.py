class Gloabl_Init:
    # 共有的变量，全局都可以使用的
    # 在程序中以 GLB的形式存在，以下均为GLB中的变量
    # GLB在全局中的数据类型为字典

    IDformat = '%04d'   # 数据类型为字符串
    NTower = 5          # 数据类型为int
    NSpan = 4           # 数据类型为int
    NCable = 1          # 数据类型为int
    dT = 1e-8           # 数据类型为int
    Nt = 1000           # 数据类型为int
    slg = 30            # 数据类型为int


    Cir = {'dat': Cir_dat,
           'num': Cir_num}
    # Cir的数据类型为字典
    # Cir_dat的数据类型为np.array，Cir_num的数据类型为np.array


    Gnd = {
            'gnd': 2,       # GND mode;: 0 free-space, 1 PGD, 2 LSG
            'mur': 1,
            'epr': 4,
            'sig': 1e-3,
            'gndcha': 2     # GND mode;: 0 free-space, 1 PGD, 2 LSG
    }
    # GND的数据类型为字典
    # gnd的数据类型为int，mur的数据类型为int，epr的数据类型为int，sig的数据类型为int，gndcha的数据类型为int


    VFIT = {
        'fr':,
        'rc':,  # 内部导体阻抗
        'dc':,  # order=3, r0+d0*s+sum(ri/s+di)
        'odc':,  # 导体的 VFIT 阶数
        'rg':,  # 接地阻抗
        'dg':,  # order=3, r0+d0*s+sum(ri/s+di)
        'odg':
    }
    # VFIT的数据类型为字典
    # fr的数据类型为np.array，rc的数据类型为np.array，dc的数据类型为np.array
    # odc的数据类型为int，rh的数据类型为np.array，dg的数据类型为np.array，odg的数据类型为int


    A = np.array([
        [-1, 1, 0, 0, 0],          # incidence matrix btw. span and tower
         [0, -1, 1, 0, 0],
         [0, 0, -1, 1, 0],
         [0, 0, -1, 0, 1]
    ])
    # A的数据类型为np.array


    Ats = np.array([-1, 0, 0, 0, 1])    # underground cable btw T1 and T5
    # Acab的数据类型为np.array


    # 最后得到的是GLB，数据类型为字典，其中包括上述所有部分
    GLB = {
        'IDformat':,
        'NTower':,
        'NSpan':,
        'NCable':,
        'dT':,
        'Nt':,
        'slg':,
        'Cir':,
        'GND':,
        'VFIT':,
        'A':,
        'Ats':,
    }

class Tower_Data(Global_Init):
    # Tower_Data调用父类Global_Init构造函数，初始化全局变量
    # 用户自己定义的数据，其中部分如接地的数据继承于GLB

    TOWER = np.array([])
    # Tower的数据类型为np.array，里面内容的数据类型均为int


    INFO = np.array([])
    # INFO的数据类型为np.array
    INFO = np.array([Tower No. 1,Type-01,10kV,0,0,0,0,0,0,1])
    # 以上述Info为例子，np.array中的数据如 Tower No.1、Type-01、10kV的数据类型字符串，其余数字的数据类型均为int


    TOWER_ID = 1
    # Tower_ID的数据类型为int


    Gnd = {
        'gnd': 2,  # GND mode;: 0 free-space, 1 PGD, 2 LSG
        'mur': 1,
        'epr': 4,
        'sig': 1e-3,
        'gndcha': 2  # GND mode;: 0 free-space, 1 PGD, 2 LSG
    }
    # 继承于Global_Init
    # GND的数据类型为字典
    # gnd的数据类型为int，mur的数据类型为int，epr的数据类型为int，sig的数据类型为int，gndcha的数据类型为int


    Ats = np.array([-1,0,0,0])
    # 继承于Global_Init
    # Atsc的数据类型为np.array


    A2GB = np.array([])
    # A2GB的数据类型np.array


    INSU = np.array([])
    # Insu的数据类型np.array
    INSU = np.array(['X09', 'X12', 'X10', 'X13', 3, nan, nan, nan, nan, nan, nan, nan,
                     nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,'Model_Input_INSU2.xlsx'])
    # 以上述INSU为例子，np.array中的数据如 X09、Model_Input_INSU2.xlsx、X13的数据类型字符串，其余数字的数据类型均为int


    AirW = np.array([])
    # AirW的数据类型为np.array
    AirW = np.array(['I', 'Y01', 'X02', 'X09', 0, -0.4, 10, 0, -0.4, 9.8, nan, 0.1, 0,
       0, 58000000, 1, 1, 0, 20000, nan, nan, nan, 'InsA1'])
    # 以上述AirW为例子，np.array中的数据如 I、Y01、InsA1的数据类型字符串，其余数字的数据类型均为int


    GndW = np.array([])
    # GndW的数据类型为np.array
    GndW = np.array([nan, 'Y19', 'X34', 'X36', 0, 0, 0, 0, 0, -1, nan, 0.1, 0, 0,
       58000000, 1, 1, 0, 20000, nan, nan, nan, 'Gnd-Pole2')
    # 以上述GndW为例子，np.array中的数据如 I、Y01、InsA1的数据类型字符串，其余数字的数据类型均为int


    Meas = np.array([])
    # Meas的数据类型为np.array
    Meas = np.array(['Meas', nan, 'Y01', 'X01', 'X02', 1, nan, nan, nan, nan, nan, nan,
       nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
    # 以上述Meas为例子，np.array中的数据如 Meas、Y01的数据类型字符串，其余数字的数据类型均为int


    End = np.array([])
    # End的数据类型为np.array


    # 最后得到的是Tower_Data，数据类型为字典，其中包括上述所有部分
    Tower_Data = {
        'TOWER':,
        'INFO':,
        'TOWER_ID':,
        'GND':,
        'A2GB':,
        'INSU':,
        'AirW':,
        'GndW':,
        'Meas':,
        'END':,
        'Gnd':,
        'Ats':,
    }


class Tower_Model(Tower_Data):
    # 调用父类Global_Init 和 Tower_Data构造函数，初始化全局变量和数据
    # 经过计算得到的Model变量


    INFO = np.array([])
    # INFO的数据类型为np.array
    # 继承于Tower_Data
    INFO = np.array([Tower No. 1,Type-01,10kV,0,0,0,0,0,0,1])
    # 以上述Info为例子，np.array中的数据如 Tower No.1、Type-01、10kV的数据类型字符串，其余数字的数据类型均为int


    TOWER_ID = 1
    # Tower_ID的数据类型为int
    # 继承于Tower_Data

    TOWER_Gnd = {
        'gnd': 2,  # GND mode;: 0 free-space, 1 PGD, 2 LSG
        'mur': 1,
        'epr': 4,
        'sig': 1e-3,
        'gndcha': 2  # GND mode;: 0 free-space, 1 PGD, 2 LSG
    }
    # 继承于Global_Init
    # GND的数据类型为字典
    # gnd的数据类型为int，mur的数据类型为int，epr的数据类型为int，sig的数据类型为int，gndcha的数据类型为int

    CK_Para = {
        'A',
        'R',  # 内部导体阻抗
        'L',  # order=3, r0+d0*s+sum(ri/s+di)
        'C',  # 导体的 VFIT 阶数
        'G',  # 接地阻抗
        'P',  # order=3, r0+d0*s+sum(ri/s+di)
        'Cw'
        'Ht'
        'Vs'
        'ls'
        'Nle'
        'Swh'
    }
    # CK_Para的数据类型为字典
    # A的数据类型为np.array，R的数据类型为np.array，L的数据类型为np.array，C的数据类型为np.array，G的数据类型为np.array，P的数据类型为np.array
    # Cw的数据类型为字典，Ht的数据类型为字典，Vs的数据类型为字典，Is的数据类型为字典，Nle的数据类型为字典，Swh的数据类型为字典

    WireP = np.array([
    [0.0,-0.4,10,0.0,-0.4,9.8,nan,0.1,0,0,58000000,1,1,0,20000,1.0,1.0,2.0]
    ])
    # WireP的数据类型为np.array，以上述WireP为例子。

    Bran= {
        'list',
        'listdex',
        'pos'
        'num'
        'a2g'
    }
    # Bran的数据类型为字典
    # list的数据类型为np.array，listdex的数据类型为np.array，pos的数据类型为np.array，num的数据类型为np.array
    # a2g的数据类型为字典，包括list和listdex两个np.array

    Node = {
        'list',
        'listdex',
        'pos'
        'num'
        'com'
        'condex'
    }
    # Node的数据类型为字典
    # list的数据类型为np.array，listdex的数据类型为np.array，pos的数据类型为np.array，num的数据类型为np.array
    # com的数据类型为np.array，condex的数据类型为np.array

    Meas= {
        'list',
        'listdex',
        'pos'
    }
    # Meas的数据类型为字典
    # list的数据类型为np.array，listdex的数据类型为np.array，pos的数据类型为np.array

    T2Smap={
        'head'
        'hspn'
        'hsid'
        'tail'
        'tspn'
        'tsid'
    }
    # T2Smap的数据类型为字典

    T2Cmap={
        'head'
        'hspn'
        'hsid'
        'tail'
        'tspn'
        'tsid'
    }
    # T2Cmap的数据类型为字典

    Soc = {}
    # Soc的数据类型为字典

    # 最后得到的是Tower_Model，数据类型为字典，其中包括上述所有部分
    Tower_Model = {
        'INFO':,
        'TOWER_ID':,
        'TOWER_Gnd':,
        'Ats':,
        'CK_Para':,
        'WireP':,
        'Bran':,
        'Node':,
        'Meas':,
        'T2Smap':,
        'T2Cmap':,
        'Soc':,
    }