class Global_Init:
    # 共有的变量，全局都可以使用的，GLB在全局中的数据类型为字典
    # 最后得到的是GLB，数据类型为字典，其中包括上述所有部分
    GLB = {
        'IDformat':,        # 数据类型为字符串
        'NTower':,          # 数据类型为int
        'NSpan':,           # 数据类型为int
        'NCable':,          # 数据类型为int
        'dT':,              # 数据类型为int
        'Nt':,              # 数据类型为int
        'Lseg':,            # 数据类型为int
        'Cir':,             # Cir的数据类型为字典，里面的数据类型np.array
        'Gnd':,             # Gnd的数据类型为字典，里面的数据类型为int
        'VFIT':,            # VFIT的数据类型为字典，里面的数据类型为np.array
        'A':,               # A的数据类型为np.array
        'Acab':,            # Acab的数据类型为np.array
        '':,                # MC的变量1，尚未定义
        '':,                # MC的变量2，尚未定义
    }

class MC(Global_Init):
    # MC 调用父类Global_Init，初始化全局变量
    # Monte Carlo Method的变量，该部分可能需要冯师姐填充。
    MC = {

    }

class LGT(MC):
    # LGT调用父类MC，初始化全局变量
    # LGT的变量，该部分可能需要杜师兄填充。
    LGT = {

    }


class Tower_Data(LGT):
    # Tower_Data调用父类LGT，初始化全局变量
    # 用户自己定义的数据

    Tower_Data = {
        'INFO':,            # INFO的数据类型为np.array
        'GND':,             # GND为接地节点，其数据类型为字典，里面的数据类型为int
        'A2GB':,            # A2GB的数据类型np.array
        'INSU':,            # Insu的数据类型np.array
        'SARR':,            # SARR的数据类型np.array
        'TXFM':,            # TXFM的数据类型np.array
        'Grid':,            # Grid的数据类型np.array
        'Others':,          # Others的数据类型np.array
        'AirW':,            # AirW的数据类型为np.array
        'GndW':,            # GndW的数据类型为np.array
        'Meas':,            # Meas的数据类型为np.array
        'END':,             # End的数据类型为np.array，主要记录文件名
        'Gnd':,             # Gnd为接地参数，其数据类型为字典，里面的数据类型为int
        'Acab':,            # Acab的数据类型为np.array
    }

class Tower_Model(Tower_Data):
    # 调用父类Tower_Data构造函数，初始化全局变量和数据
    # 经过计算得到的Model变量

    Tower_Model = {
        'TOWER_Gnd':, 	# TOWER_Gnd的数据类型为字典
        'Ats' :,          # Ats的数据类型为np.array
        'Atscab':,		# Atscab的数据类型为np.array
        'WireP':,		# WireP的数据类型为np.array
        'Bran':,		# Bran的数据类型为字典，里面内容为np.array
        'Node':,		# Node的数据类型为字典，里面内容为np.array
        'Meas':,		# Meas的数据类型为字典，
        'T2Smap':,		# T2Smap的数据类型为字典
        'T2Cmap':,		# T2Cmap的数据类型为字典
        'Soc':,		    # Soc的数据类型为字典
        'CK_Para':,		# CK_Para的数据类型为字典
                        # A的数据类型为np.array，R的数据类型为np.array，L的数据类型为np.array，C的数据类型为np.array，G的数据类型为np.array，P的数据类型为np.array
                        # Cw的数据类型为字典，Ht的数据类型为字典，Vs的数据类型为字典，Is的数据类型为字典，Nle的数据类型为字典，Swh的数据类型为字典
        'CK_Clas':,     # CK_Clas的数据类型为字典
    }

class Span_Data(LGT):
    # Span_Data调用父类LGT，初始化全局变量
    # 用户自己定义的数据

    Span_Data = {
        'SPAN_Cir'      # Span_Cir的数据类型为字典，里面的数据类型为np.array
        'INFO':,        # INFO的数据类型为np.array
        'GND':,         # GND为接地节点，其数据类型为字典，里面的数据类型为int
        'Cir_SW':,      # Cir_SW的数据类型为np.array
        'Cir_OHLs':,    # Cir_OHLs的数据类型为np.array
        'Meas':,        # Meas的数据类型为np.array
        'END':,         # End的数据类型为np.array，主要记录文件名
        'Gnd':,         # Gnd为接地参数，其数据类型为字典，里面的数据类型为int
        'Atn':,         # Atn的数据类型为np.array
    }

class Span_Model(Span_Data):
    # 调用父类Span_Data构造函数，初始化全局变量和数据
    # 经过计算得到的Model变量，计算部分可能需要吕师兄填充。

    Span_Model = {
        需要吕师兄填充

        'Bran':,        # Bran的数据类型为字典，里面内容为np.array
        'Node':,        # Node的数据类型为字典，里面内容为np.array
        'Meas':,        # Meas的数据类型为字典，
        'S2Tmap':,      # S2Tmap的数据类型为字典
        'Soc':,         # Soc的数据类型为字典
    }

class Cable_Data(LGT):
    # Cable_Data调用父类LGT，初始化全局变量
    # 用户自己定义的数据

    Cable_Data = {
        'Cable_Cir'     # Cable_Cir的数据类型为np.array，暂定
        'INFO':,        # INFO的数据类型为np.array
        'GND':,         # GND为接地节点，其数据类型为字典，里面的数据类型为int
        'Cir_Cable':,   # Cir_Cable的数据类型为np.array
        'Meas':,        # Meas的数据类型为np.array
        'END':,         # End的数据类型为np.array，主要记录文件名
        'Gnd':,         # Gnd为接地参数，其数据类型为字典，里面的数据类型为int
        'Atn':,         # Atn的数据类型为np.array
    }


class Cable_Model(Cable_Data):
    # 调用父类Cable_Data构造函数，初始化全局变量和数据
    # 经过计算得到的Model变量，计算部分可能需要吕师兄填充。

    Cable_Model = {
        需要吕师兄填充

        'Bran':,        # Bran的数据类型为字典，里面内容为np.array
        'Node':,        # Node的数据类型为字典，里面内容为np.array
        'Meas':,        # Meas的数据类型为字典，
        'C2Tmap':,      # C2Tmap的数据类型为字典
        'Soc':,         # Soc的数据类型为字典
    }
