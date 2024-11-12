class GlbInit:
    # 共有的变量，全局都可以使用的，GLB在全局中的数据类型为字典
    # 最后得到的是GLB，数据类型为字典，其中包括上述所有部分
    golbal_init = {
        'MU0':,
        'EP0':,
        'LLIM':,
        'HLIM':,            #
        'LRES':,            # 1Mohm
        'HRES':,            # 1uohm

        'IDformat':,        # 数据类型为字符串
        'NTower':,          # 数据类型为int
        'NSpan':,           # 数据类型为int
        'NCable':,          # 数据类型为int
        'dT':,              # 数据类型为int
        'Nt':,              # 数据类型为int
        'Lseg':,            # 数据类型为int (segment length)
        'Cir':,             # Cir的数据类型为字典，里面的数据类型np.array (Cir.num only) (total number)
        'Gnd':,             # Gnd的数据类型为字典，里面的数据类型为float (sig, mur, epr, gnd_model, gnd_model_chan, flag)
        'VFIT':,            # VFIT的数据类型为字典，里面的数据类型为np.array
        'A':,               # A的数据类型为np.array (span-tower incident matrix)
        'Acab':,            # Acab的数据类型为np.array (cable-tower incident matrix)
    }

class McData(GlbInit):
    # MC 调用父类Global_Init，初始化全局变量
    # Monte Carlo Method的变量，该部分可能需要冯师姐填充。
    latdis_max              # 数据类型为float

    stroke_dist= {
        'Mode':,            # 数据类型为int (mode=1: num=fixed total flash no; 2 :num=fixed total stroke no)
        'MaxN':,            # 数据类型为int
        'SNmu':,            # 数据类型为float: mean of no of strokes per flash
        'SNsig':,           # 数据类型为float: std. deviation of no of strokes per flash
        'FSmu':,            # 数据类型为np.array: mean of 1st stroke (Im, tf, Sm, th)
        'FSsig':,           # 数据类型为np.array: std. deviation of  1st stroke  (Im, tf, Sm, th)
        'SSmu':,            # 数据类型为np.array: mean of sub stroke  (Im, tf, Sm, th)
        'SSsig':,           # 数据类型为np.array: std. deviation of  sub stroke  (Im, tf, Sm, th)
        'Rhoi':,            # 数据类型为np.array: correlation matrix of  (Im, tf, Sm, th)
    }

    stroke_posi= {
        'Flash_ID':,        # 数据类型为int
        'Stroke_ID':,       # 数据类型为int
        'Stroke_Type':,     # 数据类型为int  0 = indirect, 1 = direct
        'Stroke_Object':,   # 数据类型为int  0 = ground, 1 = tower, 2 = span
        'Tower_ID':,        # 数据类型为int  struck tower id (0 = other)
        'Span_ID':,         # 数据类型为int  struck span id (0 = other)
        'Circle_ID':,       # 数据类型为int  struck circuit id (0 = other)
        'Phase_ID:',        # 数据类型为int  struck phase id (0 = other)
        'Conductor_ID':,    # 数据类型为int  struck conductor id (0 = other)
        'Segment_ID':,      # 数据类型为int  struck segment id (0 = other)
        'TP_XY_1':,         # 数据类型为int  attachment position (x)
        'TP_XY_2':,         # 数据类型为int  attachment position (y)
        'SP_XY_1':,         # 数据类型为int  stroke position (x)
        'SP_XY_2':,         # 数据类型为int  stroke position (y)
    }

    stroke_wave_model       # 数据类型为int 1 = CIGRE, 2 = Heidler
    stroke_curr_para        # 数据类型为np.array [int init float x 4]  [flash stroke Im tf Sm th]]
    stroke_wave_para        # 数据类型为np.array [float x 10]  [tn A B n I1 t1 I2 t2 Ipi Ipc]
    flash_stroke_dist       # 数据类型为np.array [int int] [flash id, stroke no.]
    stroke_summary          # 数据类型为np.array [char int/float x 16] [.....]

class LgtSource(MCData):
    # LGT调用父类MC，初始化全局变量
    # LGT的变量，该部分可能需要杜师兄填充。
    # Input Tables: Channel Table, Waveform Table,
    lgt_soc = {
        'ID':,          # np.array (Dir/Ind/orthers, Flash_id, Stroke_id, ID), ID=0 (single stroke sim)
                        # ID[1]: 0=ind,1=dir,11=Vs(ac),12=Is(ac)
        'Model':,       # np.array (Waveform_id, channel_id), Waveform_id=0 (ac), =1(CIGRE), =2(Heidle)
        'Wform':,       # int (0/1/2/3) -1=input; 0=MC table; 1=8/20us; 2=2.6/50us; 3=10/350us; 4=..
        'LGTSfile':,    # string (input source file) 需定义文件数据格式？？？
        'Posi':,        # np.array (T/S/ind, T_id (x), S_id(y), Cir_id, Phase_id, Cond_id, Seg_id)
        'Icur':,        # 数据类型np.array (current data)
    } 
   
    lgt_chan = {
        'H':,       # H的数据类型为字典，里面内容为int
        'lam':,     # lam的数据类型为字典，里面内容为int
        'vcf':,     # lam的数据类型为字典，里面内容为int
        'H0':,      # 数据类型为int
        'dH':,      # 数据类型为int
        'Nc':,      # 数据类型为int
        'pos':,     # pos的数据类型np.array
        'curr':,    # curr的数据类型np.array
        'flg':,     # 数据类型为int
    }

class TowerData(LgtSource):
    # Tower_Data调用父类LGT，初始化全局变量
    # 用户自己定义的数据

    tower_data = {
        'INFO':,            # INFO的数据类型为np.array
        'GND':,             # GND为接地节点，其数据类型为字典，里面的数据类型为int
        'Ats':,             # Ats的数据类型为np.array
        'Atscab':,          # Acab的数据类型为np.array
        'AirW':,            # AirW的数据类型为np.array
        'GndW':,            # GndW的数据类型为np.array
        'A2GB':,            # A2GB的数据类型np.array
        'Meas':,            # Meas的数据类型为np.array
        'INSU':,            # Insu的数据类型np.array,  # INSU_Data
        'SARR':,            # SARR的数据类型np.array,  # SARR_Data
        'TXFM':,            # TXFM的数据类型np.array,  # TXFM_Data
        'GRID':,            # TXFM的数据类型np.array,  # GRID_Data
        'OTHS':,            # TXFM的数据类型np.array,  # OTHS_Data
     }

class TowerModel(TowerData):
    # 调用父类Tower_Data构造函数，初始化全局变量和数据
    # 经过计算得到的Model变量

    tower_model = {
        'WireP':,		# WireP的数据类型为np.array
        'Node':,		# Node的数据类型为字典，里面内容为np.array
        'Bran':,		# Bran的数据类型为字典，里面内容为np.array
        'Meas':,		# Meas的数据类型为字典，
        'T2Smap':,		# T2Smap的数据类型为字典
        'T2Cmap':,		# T2Cmap的数据类型为字典
        'CK_Para':,		# CK_Para的数据类型为字典
                        # A的数据类型为np.array，R的数据类型为np.array，L的数据类型为np.array，C的数据类型为np.array，G的数据类型为np.array，P的数据类型为np.array
                        # Cw的数据类型为字典，Ht的数据类型为字典，Vs的数据类型为字典，Is的数据类型为字典，Nle的数据类型为字典，Swh的数据类型为字典
        'CK_Cals':,		# CK_Para的数据类型为字典
        'Soc':,		    # Soc的数据类型为字典
   }

class SpanData(LgtSource):
    # Span_Data调用父类LGT，初始化全局变量
    # 用户自己定义的数据

    span_data = {
        'INFO':,            # INFO的数据类型为np.array
        'GND':,             # GND为接地节点，其数据类型为字典，里面的数据类型为int
        'Atn':,             # Atn的数据类型为np.array
        'Atncab':,          # Atncab的数据类型为np.array
        'Cir'               # Span_Cir的数据类型为字典，里面的数据类型为np.array
        'Seg':,             # Cir_SW的数据类型为np.array
        'Pole':,            # Cir_OHLs的数据类型为np.array
        'OHLP':,            # Cir_OHLs的数据类型为np.array
        'Meas':,        # Meas的数据类型为np.array
    }

class SpanModel(SpanData):
    # 调用父类Span_Data构造函数，初始化全局变量和数据
    # 经过计算得到的Model变量，计算部分可能需要吕师兄填充。

    span_model = {
        需要吕师兄填充

        'Bran':,        # Bran的数据类型为字典，里面内容为np.array
        'Node':,        # Node的数据类型为字典，里面内容为np.array
        'Meas':,        # Meas的数据类型为字典，
        'S2Tmap':,      # S2Tmap的数据类型为字典
        'CK_Para':,		# CK_Para的数据类型为字典
                        # A的数据类型为np.array，R的数据类型为np.array，L的数据类型为np.array，C的数据类型为np.array，G的数据类型为np.array，P的数据类型为np.array
                        # Cw的数据类型为字典，Ht的数据类型为字典，Vs的数据类型为字典，Is的数据类型为字典，Nle的数据类型为字典，Swh的数据类型为字典
        'CK_Cals':,		# CK_Para的数据类型为字典
        'Soc':,		    # Soc的数据类型为字典
    }

class CableData(LgtTSource):
    # Cable_Data调用父类LGT，初始化全局变量
    # 用户自己定义的数据

    cable_data = {
        'INFO':,            # INFO的数据类型为np.array
        'GND':,             # GND为接地节点，其数据类型为字典，里面的数据类型为int
        'Atn':,             # Atn的数据类型为np.array
        'Atncab':,          # Atncab的数据类型为np.array
        'Cir':,             # Span_Cir的数据类型为字典，里面的数据类型为np.array
        'Seg':,             # Cir_SW的数据类型为np.array
        'Pole':,            # Cir_OHLs的数据类型为np.array
        'Line':,            # Cir_OHLs的数据类型为np.array
        'Meas':,            # Meas的数据类型为np.array
    }


class CableModel(CableData):
    # 调用父类Cable_Data构造函数，初始化全局变量和数据
    # 经过计算得到的Model变量，计算部分可能需要吕师兄填充。

    cable_model = {
        需要吕师兄填充

        'Bran':,        # Bran的数据类型为字典，里面内容为np.array
        'Node':,        # Node的数据类型为字典，里面内容为np.array
        'Meas':,        # Meas的数据类型为字典，
        'C2Tmap':,      # S2Tmap的数据类型为字典
        'CK_Para':,		# CK_Para的数据类型为字典
                        # A的数据类型为np.array，R的数据类型为np.array，L的数据类型为np.array，C的数据类型为np.array，G的数据类型为np.array，P的数据类型为np.array
                        # Cw的数据类型为字典，Ht的数据类型为字典，Vs的数据类型为字典，Is的数据类型为字典，Nle的数据类型为字典，Swh的数据类型为字典
        'CK_Cals':,		# CK_Para的数据类型为字典
        'Soc':,		    # Soc的数据类型为字典
    }
