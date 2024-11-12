import numpy as np
import matplotlib.pyplot as plt
import Vector_Fitting
import pandas as pd
import openpyxl
import os

from scipy.interpolate import interp1d

np.seterr(divide="ignore",invalid="ignore")

class LGT_Source_Build():
    def __init__(self, FDIR):
        self.FDIR = FDIR

    def LGT_Source_Build(self, Tower, Span, Cable, GLB,):
        # Generating LGT source data for Tower, Span, and Cable
        # Soc.ISF (dir)
        # Soc.VSF (ind) for MNA equation
        # Soc.typ = 0/1/2 (no/dir/ind);
        # Soc.pos = [T/S, id, cir_id, phase_id, seg(node)_id]

        # (1) Read one stroke data from MC_lightning_table
        # lgt_id = [4 4];                     % Dir-Span [flash_id stroke_id]
        # lgt_id = [16 1];                    % Dir-Tower [flash_id stroke_id]
        lgt_id = [2, 1]  # Ind [flash_id stroke_id]
        GLB, LGT = self.Soc_Init(GLB, lgt_id)

        index = 1  # 2= load induced source data from TowerX/SpanX/CableX.mat
        # (2) Generating source current or induced-E source on Tower/Span/Cable
        # 调用Soc_Init()函数进行Soc和LGT的初始化
        # Soc, LGT = self.Soc_Init(GLB)

        # 调用Span_Circuit_Source()函数进行Span的计算
        for ik in range(0, GLB['NSpan']):
            Span[ik], LGT = self.Span_Circuit_Source(Span[ik], GLB, LGT, index)

        # Cable_Circuit_Source is added here
        for ik in range(0, GLB['NCable']):
            Cable[ik] = self.Cable_Circuit_Source(Cable[ik], GLB, LGT, index)

        # 调用Tower_Circuit_Source()函数进行Tower的计算
        for ik in range(0, GLB['NTower']):
            Tower[ik] = self.Tower_Circuit_Source(Tower[ik], GLB, LGT, index)

        a =100

        return Tower, Span, Cable, GLB, LGT

    def Soc_Init(self, GLB, lgt_id):
        # (0) Initial Constants
        dT = GLB['dT']
        Nt = GLB['Nt']
        t0 = np.arange(Nt) * dT

        Chan_Model = 3
        flash_id = lgt_id[0]
        strok_id = lgt_id[1]

        # (1) Finding out flash/stroke parameters and update Flash and Stroke
        # 创建一个空的列表来存储结果
        data = []
        out1 = np.array([])
        out2 = np.array([])

        # 假设子文件夹名为'data'，Excel文件名为'example.xlsx'
        MCLG_dir = self.FDIR['datamclsfiles']
        iSRC_dir = self.FDIR['isrc']

        current_dir = self.path
        MCLG_path = os.path.join(current_dir, MCLG_dir)
        iSRC_path = os.path.join(current_dir, iSRC_dir)

        # 文件的名字
        FilenameStat = 'Statistics Summary.txt'
        FilenameDist = 'Flash_Stroke Dist.xlsx'
        FilenamePara = 'b_Current Waveform_CIGRE.xlsx'
        FilenamePosi = 'b_Stroke Position.xlsx'
        chan_data = 'channel model table.txt'
        current_data = 'current model table.txt'
        # 文件路径
        # path = "H:\Ph.D. Python\StrongEMPPlatform\Tower\Main Program\DATA_MCLG"
        FilenameStat_path = MCLG_path + '/' + FilenameStat
        FilenameDist_path = MCLG_path + '/' + FilenameDist
        FilenamePara_path = MCLG_path + '/' + FilenamePara
        FilenamePosi_path = MCLG_path + '/' + FilenamePosi

        # path = "H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\PARA_iSRC"
        chan_data_path = iSRC_path + '/' + chan_data
        current_data_path = iSRC_path + '/' + current_data

        # Lgtntmp1
        # 读取包含行的文件
        with open(FilenameStat_path, 'r') as file:
            lines = file.readlines()

        # 遍历文件中的每一行，拆分文本并添加到数据列表中
        for line in lines:
            parts = line.split(':')
            if len(parts) == 2:
                data.append([parts[0].strip(), parts[1].strip()])

        # 创建一个名为 Lgtntmp1 的 DataFrame
        Lgtntmp1 = pd.DataFrame(data, columns=['Column1', 'Column2'])
        LgtnStat = Lgtntmp1['Column2'].values  # Statistic data of lightning strokes
        LgtnStat = LgtnStat.reshape(np.size(LgtnStat), 1)
        LgtnStat = LgtnStat.astype(float)

        # Lgtntmp2
        Lgtntmp2 = pd.read_excel(FilenameDist_path)
        LgtnDist = Lgtntmp2.values  # Pair of flash id and stroke #

        Nflash = LgtnStat[0, 0]  # Total flash #
        if flash_id > Nflash:
            print('Error in Flash ID (>Max ID)')
            return GLB, None

        Mstrok = LgtnDist[flash_id - 1, 1]  # Max stroke # of a flash
        if strok_id > Mstrok:
            print('Error in Stroke ID (>Max ID)')
            return GLB, None

        if flash_id > 1:  # the postion of stroke in the table
            strok_index = np.sum(LgtnDist[:flash_id - 1, 1]) + strok_id
        else:
            strok_index = strok_id

        str0 = str(strok_index + 1)  # including headline
        str1 = 'A' + str0  # Range of Excel sheet
        str2 = 'J' + str0

        ## FilenamePara
        # 打开 Excel 文件
        workbook = openpyxl.load_workbook(FilenamePara_path)
        # 选择工作表
        sheet = workbook.active  # 或者通过名称选择工作表，例如：sheet = workbook['Sheet1']
        # 选择要读取的单元格范围，例如A4到J4
        cell_range = sheet[str1:str2]
        # 遍历选定的单元格范围，并将值添加到现有的列表 out1 中
        for row in cell_range:
            for cell in row:
                out1 = np.append(out1, [cell.value])
                out1 = out1.reshape(1, np.size(out1)).astype(float)
        # 关闭 Excel 文件
        workbook.close()
        wave_para = out1
        ## FilenamePosi
        # 打开 Excel 文件
        workbook = openpyxl.load_workbook(FilenamePosi_path)
        # 选择工作表
        sheet = workbook.active  # 或者通过名称选择工作表，例如：sheet = workbook['Sheet1']
        # 选择要读取的单元格范围，例如A4到J4
        cell_range = sheet[str1:str2]
        # 遍历选定的单元格范围，并将值添加到现有的列表 out1 中
        for row in cell_range:
            for cell in row:
                out2 = np.append(out2, [cell.value])
                out2 = out2.reshape(1, np.size(out2))
        # 关闭 Excel 文件
        workbook.close()
        tmp = out2[0, 0]

        strok_pos = np.zeros(6)  # pos(6) = flag (1=updating for tower)

        if tmp == 'Direct':
            lgt_typ = 1  # Direct stroke
            strok_obj = out2[0][1:3]  # object (T/S/C/G) and object #
            strok_cir = out2[0][3][0]  # circuit id, assuming out2[0][3] is a list with a string number
            strok_pha = out2[0][4][0]  # phase id, assuming out2[0][4] is a list with a string number
            strok_seg = out2[0][5]  # segment id
            strok_pos = [strok_obj[0], strok_obj[1], strok_cir, strok_pha, strok_seg, 0]
        else:
            lgt_typ = 2  # Indirect stroke
            strok_pos[1:3] = [out2[0][6], out2[0][7]]

        FlashSM = [lgt_typ, flash_id, Mstrok, Chan_Model, *strok_pos]
        StrokeSM = [flash_id, strok_id, 0, strok_index]

        # (2) Channel model---- 1=Nc,2=dH,3=H0,4=flag_type,5=H,6=lamda,7=vco
        Lch = {
            'dT': GLB['dT'],
            'Nt': GLB['Nt'],
            'eps': GLB['GND']['epr'],
            'sig': GLB['GND']['sig']
        }

        chan_model = FlashSM[3]  # Chan model = 3 (MTLE model(lamda,vcof))
        # chan_data 改成了 chan_info
        chan_info = pd.read_csv(chan_data_path) # 修改了源文件，把第一行之间加了逗号
        # 使用iloc删除标题行和第一列
        chan_data = chan_info.iloc[:, 1:]
        # 将DataFrame转换为NumPy数组
        chan_data = chan_data.values

        Lch['flg'] = chan_model  # 1=TL model,2=MTLL model(H),3=MTLE model(lamda,vcof)
        Lch['H'] = chan_data[chan_model - 1, 0]  # channel attenuation coef
        Lch['lam'] = chan_data[chan_model - 1, 1]
        Lch['vcf'] = chan_data[chan_model - 1, 2]

        Lch['H0'] = 1300  # Channel height
        Lch['dH'] = 10  # Length of channel segments
        Lch['Nc'] = Lch['H0'] / Lch['dH']  # Number of channel segments
        Lch['pos'] = FlashSM[4:10]  # ind: (x,y,0), dir: (S/C/T ID, seg ID, phase, seg)

        """杜哥写的
                curr_flag_type = 1
        chan_flag_type = int(input('请输入雷电通道工程模型代码: (1=TL model,2=MTLL model,3=MTLE model)'))
        if chan_flag_type in [1, 2, 3]:
            chan_flag_type = self.Chan_Input(chan_flag_type)
        else:
            raise ValueError('雷电通道工程模型代码输入错误,没有此代码')

            """

        # (3) Current model
        current_waveform = StrokeSM[2]  # 1=10/350,2=1/200,3=0.25/100
        # curdata = np.loadtxt('current model table.txt', delimiter=',')
        # current_standard_wave = int(input('请输入电流标准波形代码: (1=8/20us,2=2.6/50us,3=10/350us...)'))
        # if current_standard_wave in [1, 2, 3]:
        #     Icurr = self.I_Sr_Stan_Wave(GLB, current_standard_wave, curdata)
        # else:
        #     raise ValueError('Error in Waveform ID')

        if current_waveform in [1, 2, 3]:
            # curdata = np.loadtxt('Current Model Table.txt')
            current_info = pd.read_csv(current_data_path)  # 修改了源文件，把第一行之间加了逗号
            # 使用iloc删除标题行和第一列
            current_data = current_info.iloc[:, 1:]
            # 将DataFrame转换为NumPy数组
            current_data = current_data.values
            Ip = StrokeSM[3]
            k = current_data[current_waveform-1, 1]
            tau1 = current_data[current_waveform-1, 2] * 1e-6  # unit: us
            tau2 = current_data[current_waveform-1, 3] * 1e-6  # unit: us
            Icurr = (Ip / k) * ((t0 / tau1) ** 10) / (1 + (t0 / tau1) ** 10) * np.exp(-t0 / tau2)

        elif current_waveform == 0:  # looking MC table, find out current wave form
            Icurr = self.CIGRE_Waveform_Generator(wave_para, t0)# Implement CIGRE_Waveform_Generator function here if needed
        elif current_waveform == -1:
            # input current wave
            pass
        else:
            raise ValueError('Error in Waveform ID')

        # (3) Updating Flash/stroke data

        Soc = {}
        Soc['typ'] = FlashSM[0]  # Source: 1=dir,2=ind,11=Vs,ac,12=Is,ac
        Soc['pos'] = FlashSM[4:10]  # ind:[x,y,0,0,0 0]
        Soc['dat'] = Icurr  # current data

        Lch['curr'] = Icurr  # current data

        LGT = {}
        LGT['Flash'] = {'head': FlashSM}
        LGT['Stroke'] = {'head': StrokeSM}
        LGT['Lch'] = Lch
        LGT['Soc'] = Soc

        GLB['Soc'] = Soc  # Source: 1=dir, 2=ind, 3=Vs,ac, 4 =Is,ac'''
        """杜哥写的部分
        # Soc = {}
        # Soc['typ'] = 1  # Source: 1=dir,2=ind,11=Vs,ac,12=Is,ac
        # Soc['pos'] = [1, 1, 1, 3001]  # ind:(x,y,0),dir:(spanID,segID,phase)
        #
        # Flash_head = [1, 1, 1, 3, 1, 0, 0, 0]
        # Stroke_head = [1, 1, 1, 3, 1, 0, 0, 0]
        #
        #
        #
        #
        # chan = np.loadtxt('channel model table.txt', delimiter=',')
        # Stroke_chan = chan  # 1=Nc,2=dH,3=H0,4=flag_type,5=H,6=lamda,7=vcof
        # Lch = {}
        # Lch['H'] = chan[chan_flag_type - 1, 1 - 1]
        # Lch['lam'] = chan[chan_flag_type - 1, 2 - 1]
        # Lch['vcf'] = chan[chan_flag_type - 1, 3 - 1]
        #
        #
        #
        # Lch['dT'] = GLB['dT']
        # Lch['Nt'] = GLB['Nt']
        # Lch['eps'] = GLB['GND']['epr']
        # Lch['sig'] = GLB['GND']['sig']
        # Lch['H0'] = 130  # From Nancy
        # Lch['dH'] = 1
        # Lch['Nc'] = Lch['H0'] / Lch['dH']
        # Lch['pos'] = Stroke_head[5:7]  # ind: (x,y,0), dir: (span ID, seg ID, phase)
        # Lch['curr'] = Stroke_curr
        # Lch['flg'] = curr_flag_type  # 1=TL model,2=MTLL model(H),3=MTLE model(lamda,vcof)
        #
        # LGT = {}
        # LGT['Lch'] = Lch
        return Soc, LGT
        """

        # 绘制图形
        plt.figure(2)
        plt.plot(t0, Icurr )
        plt.xlabel('Time')
        plt.ylabel('Current')
        plt.title('Waveform')
        # plt.show()

        return GLB, LGT
    def Span_Circuit_Source(self, Span, GLB, LGT, index):
        # Lne = self.Extract_Line_Data_TL(Span)
        # Generate lightning source parameters due to direct/indirect lightning
        # Update Span.Soc.typ/dat/pos
        # Initialize Span.Soc.ISF/VSF
        # Pos: [x, y, 0, 0, 0) for IND
        #      [T/S/0, ID, CirID, PhaseID, Seg(node)] for DIR
        # Span.Soc.pos:
        #      [T/S, ID, CirID, Cond_ID, Seg(node)] for DIR

        ID = Span['ID']
        Cir = Span['Cir']
        Seg = Span['Seg']
        OHLP = Span['OHLP']
        LSoc = LGT['Soc']
        Span_ID = f"Span{ID}"

        Soc = {}
        Soc['typ'] = LSoc['typ']
        Soc['pos'] = LSoc['pos']
        Soc['dat'] = []  # store all source dat

        Lseg = Seg['Lseg']
        Nseg = Seg['Nseg']
        Ncon = int(Seg['Ncon'])  # # of total conductors
        Soc['ISF'] = np.zeros(Ncon)  # stroe the data at one moment (dir)

        Temp_path = self.FDIR['dataTempFile']
        # path = "H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\DATA_Temp"
        str_path = Temp_path  + "/Span" + str(ID) + ".mat"

        # load('data.mat'); # include (a) data_SRC.direct (b) data_SRC.induced
        if LSoc['typ'] == 1:
            if (LSoc['pos'][0] != 2) or (LSoc['pos'][1] != Span['ID']):
                Soc['pos'] = 0
                Span['Soc'] = Soc
                # update lightning source position for dir-struck-tower
                Posi = self.LGTS_Pos_Update(Span, LSoc['pos'])
                LGT['Soc']['pos'] = Posi
                return
            else:
                # cond_id_read没写
                # con_id = Cond_ID_Read(Cir, LSoc['pos'])
                Soc['ISF'][con_id] = 1
                Soc['dat'] = LSoc['dat']
        elif LSoc['typ'] == 2:
            Soc['VSF'] = np.zeros((Ncon, Nseg))  # stroe the data at one moment (ind)
            if index == 2:
                dat = np.load(str_path, allow_pickle=True) # 没试过是不是可以读取mat
                Soc['dat'] = dat['dat']
                Span['Soc'] = Soc
                return

            Pstr = OHLP[:, 0:3]
            Pend = OHLP[:, 3:6]

            delta = (Pend - Pstr) / Nseg
            x0 = np.zeros((Ncon, Nseg + 1))
            y0 = np.zeros((Ncon, Nseg + 1))
            z0 = np.zeros((Ncon, Nseg + 1))
            for ik in range(Ncon):
                x0[ik, :] = Pstr[ik, 0] + np.arange(Nseg + 1) * delta[ik, 0]
                y0[ik, :] = Pstr[ik, 1] + np.arange(Nseg + 1) * delta[ik, 1]
                z0[ik, :] = Pstr[ik, 2] + np.arange(Nseg + 1) * delta[ik, 2]

            Lne_x1 = x0[:, :-1].T.flatten()
            Lne_x2 = x0[:, 1:].T.flatten()
            Lne_y1 = y0[:, :-1].T.flatten()
            Lne_y2 = y0[:, 1:].T.flatten()
            Lne_z1 = z0[:, :-1].T.flatten()
            Lne_z2 = z0[:, 1:].T.flatten()

            Lne_x1 = Lne_x1.reshape(np.size(Lne_x1),1)
            Lne_x2 = Lne_x2.reshape(np.size(Lne_x2),1)
            Lne_y1 = Lne_y1.reshape(np.size(Lne_y1),1)
            Lne_y2 = Lne_y2.reshape(np.size(Lne_y2),1)
            Lne_z1 = Lne_z1.reshape(np.size(Lne_z1),1)
            Lne_z2 = Lne_z2.reshape(np.size(Lne_z2),1)

            Lne_pt_start = np.column_stack((Lne_x1, Lne_y1, Lne_z1))
            Lne_pt_end = np.column_stack((Lne_x2, Lne_y2, Lne_z2))

            slen = np.sqrt(np.sum(delta * delta, axis=1))
            Lne_L = np.tile(slen, (Nseg, 1))
            Lne_L = Lne_L.flatten().reshape((np.size(Lne_L),1))
            Lne = {
                'tran': {
                    'pt_start': Lne_pt_start,
                    'pt_end': Lne_pt_end,
                    'L': Lne_L
                }
            }

            Er_T, Ez_T = self.E_Cal(LGT, GLB, Lne)

            U_TL = self.Cor_Lossy_Ground(GLB, LGT, Lne, Er_T, Ez_T, Span_ID)

        # dat = np.array(U_TL).reshape(Ncon, Nseg, -1) # 输出有问题
        dat = np.array(U_TL)
        dat = np.transpose(U_TL)
        Soc['dat'] = dat
        Span['Soc'] = Soc

        return Span, LGT

    def Tower_Circuit_Source(self, Tower, GLB, LGT, index):
        ID = Tower['ID']
        LSoc = LGT['Soc']
        Tower_ID = f"Tower{ID}"

        Soc = {}
        Soc['typ'] = LSoc['typ']
        Soc['pos'] = LSoc['pos']
        Soc['dat'] = []  # store all source dat

        Nn = Tower['Node']['num'][0]
        Nb = Tower['Bran']['num'][0]
        Soc['ISF'] = []
        Soc['VSF'] = []

        Temp_dir = self.FDIR['datatempfiles']
        current_dir = self.path
        Temp_path = os.path.join(current_dir, Temp_dir)
        # path = "H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\DATA_Temp"
        str_path = Temp_path  + "/Span" + str(ID) + ".mat"

        # load('data.mat'); # include (a) data_SRC.direct (b) data_SRC.induced
        if LSoc['typ'] == 1:
            if (LSoc['pos'][0] != 1) or (LSoc['pos'][1] != ID):
                Soc['pos'] = 0
                Tower['Soc'] = Soc
                return
            else:
                nodeID = Soc['pos'][4]
                Soc['ISF'] = np.zeros(Nn)
                Soc['ISF'][nodeID] = 1
                Soc['dat'] = LSoc['dat']
        elif LSoc['typ'] == 2:
            Soc['VSF'] = np.zeros(Nb)
            if index == 2:
                dat = np.load(str_path, allow_pickle=True)
                Soc['dat'] = dat['dat']
                Tower['Soc'] = Soc
                return

        Pstr = Tower['WireP'][:, 0:3]
        Pend = Tower['WireP'][:, 3:6]

        Lne = {}
        Lne_x1 = Pstr[:, 0].T.flatten()
        Lne_x2 = Pend[:, 0].T.flatten()
        Lne_y1 = Pstr[:, 1].T.flatten()
        Lne_y2 = Pend[:, 1].T.flatten()
        Lne_z1 = Pstr[:, 2].T.flatten()
        Lne_z2 = Pend[:, 2].T.flatten()

        Lne_x1 = Lne_x1.reshape(np.size(Lne_x1), 1)
        Lne_x2 = Lne_x2.reshape(np.size(Lne_x2), 1)
        Lne_y1 = Lne_y1.reshape(np.size(Lne_y1), 1)
        Lne_y2 = Lne_y2.reshape(np.size(Lne_y2), 1)
        Lne_z1 = Lne_z1.reshape(np.size(Lne_z1), 1)
        Lne_z2 = Lne_z2.reshape(np.size(Lne_z2), 1)

        Lne_pt_start = np.column_stack((Lne_x1, Lne_y1, Lne_z1))
        Lne_pt_end = np.column_stack((Lne_x2, Lne_y2, Lne_z2))


        Lne['tran'] = {}
        Lne['tran']['pt_start'] = Lne_pt_start
        Lne['tran']['pt_end'] = Lne_pt_end

        delta = (Pend - Pstr)
        delta2 = np.array(delta * delta, dtype=float)
        Lne_L = np.sqrt(np.sum(delta2, axis=1)) # length of wire segments
        Lne_L = Lne_L.flatten().reshape((np.size(Lne_L), 1))
        Lne['tran']['L'] = Lne_L
        # E_Cal 函数调用
        Er_T, Ez_T = self.E_Cal(LGT, GLB, Lne)

        U_TL = self.Cor_Lossy_Ground(GLB, LGT, Lne, Er_T, Ez_T, Tower_ID)
        dat = np.array(U_TL).T
        Soc['dat'] = dat
        Tower['Soc'] = Soc

        return Tower

    def Cable_Circuit_Source(self, Cable, GLB, LGT, index):
        ID = Cable['ID']
        Seg = Cable['Seg']
        LSoc = LGT['Soc']
        Cable_ID = f"Cable{ID}"

        Soc = {}
        Soc['typ'] = LSoc['typ']
        Soc['pos'] = LSoc['pos']
        Soc['dat'] = []  # store all source dat


        Nseg = Seg['Nseg']
        Ncon = int(Seg['Ncon'])  # # of total conductors
        Npha = int(Seg['Npha'])
        Narm = Ncon - Npha
        Soc['ISF'] = 0  # stroe the data at one moment (dir)

        Temp_dir = self.FDIR['datatempfiles']
        current_dir = self.path
        Temp_path = os.path.join(current_dir, Temp_dir)
        # path = "H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\DATA_Temp"
        str_path = Temp_path  + "/Span" + str(ID) + ".mat"

        # load('data.mat'); # include (a) data_SRC.direct (b) data_SRC.induced
        if LSoc['typ'] == 1:
            if (LSoc['pos'][0] != 3) or (LSoc['pos'][1] != ID):
                Soc['dat'] = 0
                Cable['Soc'] = Soc
                return
            else:
                Soc['ISF'] = 1
                Soc['dat'] = LSoc['dat']
        elif LSoc['typ'] == 2:
            Soc['VSF'] = np.zeros((Narm, Nseg))
            if index == 2:
                dat = np.load(str_path, allow_pickle=True)
                Soc['dat'] = dat['dat']
                Cable['Soc'] = Soc
                return

            height = Cable['Line']['rad'][5]
            Pstr = np.array([Cable['Pole'][0], Cable['Pole'][1], height])
            Pend = np.array([Cable['Pole'][3], Cable['Pole'][4], height])

            delta = (Pend - Pstr) / Nseg
            x0 = Pstr[0] + np.arange(Nseg + 1) * delta[0]
            y0 = Pstr[1] + np.arange(Nseg + 1) * delta[1]
            z0 = height + np.arange(Nseg + 1) * 0


            Lne_x1 = x0[:-1].T.flatten()
            Lne_x2 = x0[1:].T.flatten()
            Lne_y1 = y0[:-1].T.flatten()
            Lne_y2 = y0[ 1:].T.flatten()
            Lne_z1 = z0[:-1].T.flatten()
            Lne_z2 = z0[1:].T.flatten()

            Lne_x1 = Lne_x1.reshape(np.size(Lne_x1),1)
            Lne_x2 = Lne_x2.reshape(np.size(Lne_x2),1)
            Lne_y1 = Lne_y1.reshape(np.size(Lne_y1),1)
            Lne_y2 = Lne_y2.reshape(np.size(Lne_y2),1)
            Lne_z1 = Lne_z1.reshape(np.size(Lne_z1),1)
            Lne_z2 = Lne_z2.reshape(np.size(Lne_z2),1)

            Lne_pt_start = np.column_stack((Lne_x1, Lne_y1, Lne_z1))
            Lne_pt_end = np.column_stack((Lne_x2, Lne_y2, Lne_z2))


            Lne = {}
            Lne['tran'] = {}
            Lne['tran']['pt_start'] = Lne_pt_start
            Lne['tran']['pt_end'] = Lne_pt_end
            slen = np.sqrt(np.sum(delta * delta))
            Lne_L = np.tile(slen, (Nseg, 1))
            Lne_L = Lne_L.flatten().reshape((np.size(Lne_L),1))
            Lne['tran']['L'] = Lne_L

        # E_Cal 函数调用
        Er_T, Ez_T = self.E_Cal(LGT, GLB, Lne)

        U_TL = self.Cor_Lossy_Ground(GLB, LGT, Lne, Er_T, Ez_T, Cable_ID)


        dat = np.array(U_TL).reshape(Ncon, Nseg, -1) # 输出有问题
        Soc['dat'] = dat
        Cable['Soc'] = Soc

        return Cable

    def CIGRE_Waveform_Generator(self, Wave_Para, t):
        tn, A, B, n, I1, t1, I2, t2, Ipi, Ipc = Wave_Para[0,:]

        T0 = t[t <= tn]
        Iout1 = A * T0 + B * (T0 ** n)

        T0 = t[t > tn]
        Iout2 = I1 * np.exp(-(T0 - tn) / t1) - I2 * np.exp(-(T0 - tn) / t2)

        Iout = np.concatenate((Iout1, Iout2)) * (Ipi / Ipc)

        return Iout
    def Chan_Input(self, chan_flag_type):
        # chan_flag_type = 1
        file_name = 'Channel Model.txt'  # 文件名

        # 读取文件内容
        with open(file_name, 'r') as file:
            file_content = file.readlines()

        # 获取第四行的内容（假设这是flag_type所在的行）
        line_to_replace = file_content[3]  # Python中索引从0开始

        # 使用新的flag_type值替换行中的数字
        new_line = ''.join([c if not c.isdigit() else str(chan_flag_type) for c in line_to_replace])

        # 更新文件内容
        file_content[3] = new_line

        # 打开文件以写入新的内容
        with open(file_name, 'w') as file:
            file.writelines(file_content)

        return chan_flag_type

        # result = chan_flag_type
        # print(result)

    def I_Sr_Stan_Wave(self, GLB, current_standard_wave, curdata):
        dt = GLB['dT']
        Nt = GLB['Nt']
        Imax = 1e4
        k0 = 1

        tau1 = curdata[current_standard_wave - 1][2] * 1e-6  # 对应 MATLAB 的索引从 1 开始，Python 的索引从 0 开始
        tau2 = curdata[current_standard_wave - 1][3] * 1e-6

        n = 5
        i_sr = np.zeros(Nt)

        for i in range(Nt):
            I_temp = (i * dt / tau1) ** n / (1 + (i * dt / tau1) ** n)
            i_sr[i] = Imax / k0 * I_temp * np.exp(-i * dt / tau2)

        return i_sr

    def Extract_Line_Data_TL(self, Span):
        # 读取起点、终点和线段数目
        # pt_start_temp = Span['line'][:, :3]
        pt_start_temp = [sublist[:3] for sublist in Span['line']]

        # 获取 pt_end_temp
        pt_end_temp = [sublist[3:6] for sublist in Span['line']]

        # 获取 Num_seg_temp
        Num_seg_temp = Span['Cir']['num'][1][5]

        n = len(pt_start_temp[0])  # Get the number of columns in the first sublist
        Num_seg_temp = [Num_seg_temp] * n  # 创建长度为 n 的列表，内容都是 Num_seg_temp

        # 分割线段
        segment_starts, segment_ends = self.split_segments(pt_start_temp, pt_end_temp, Num_seg_temp)

        # 计算线段长度
        segment_lengths = self.calculate_segment_lengths(segment_starts, segment_ends)

        # 存储结果
        Lne = {
            'tran': {
                'pt_start': segment_starts,
                'pt_end': segment_ends,
                'L': segment_lengths
            }
        }

        return Lne

    def Split_Segments(self, start_points, end_points, ns):
        # 将输入的列表转换为NumPy数组
        start_points = np.array(start_points)
        end_points = np.array(end_points)
        # 确保输入的矩阵行数相同（即线段数目相同）
        assert start_points.shape[0] == end_points.shape[0], '起点和终点矩阵的行数必须相同。'
        # 确保分段数目向量的长度与线段数目相同
        assert len(ns) == start_points.shape[0], '分段数目向量长度必须与线段数目相同。'

        # 获取线段数目
        num_segments = start_points.shape[0]

        # 计算总的分段数目
        total_segments = np.sum(ns)

        # 初始化新的分段后起点和终点坐标矩阵
        segment_starts = np.zeros((total_segments, 3))
        segment_ends = np.zeros((total_segments, 3))

        idx = 0  # 用于追踪新坐标矩阵的索引

        for i in range(num_segments):
            # 提取当前线段的起点和终点坐标
            start_point = start_points[i]
            end_point = end_points[i]
            n = ns[i]  # 当前线段的分段数目

            # 计算每段线段的增量
            delta = (end_point - start_point) / n

            # 分段并计算新的起点和终点坐标
            for j in range(n):
                segment_starts[idx] = start_point + j * delta
                segment_ends[idx] = start_point + (j + 1) * delta
                idx += 1

        return segment_starts, segment_ends

    def Calculate_Segment_Lengths(self, start_points, end_points):
        # 确保输入的矩阵行数相同（即线段数目相同）
        assert start_points.shape[0] == end_points.shape[0], '起点和终点矩阵的行数必须相同。'

        # 获取线段数目
        num_segments = start_points.shape[0]

        # 初始化存储线段长度的向量
        segment_lengths = np.zeros(num_segments)

        # 计算每个线段的长度
        for i in range(num_segments):
            # 提取当前线段的起点和终点坐标
            start_point = start_points[i]
            end_point = end_points[i]

            # 计算欧几里得距离，即线段长度
            distance = np.linalg.norm(end_point - start_point)

            # 存储线段长度
            segment_lengths[i] = distance

        return segment_lengths

    def E_Cal(self, LGT, GLB, Lne):
        ep0 = 8.85e-12
        vc = 3e8  # light speed
        # Extracting variables from input structures
        i_sr = LGT['Lch']['curr']
        dt = GLB['dT']
        pt_start = Lne['tran']['pt_start']
        pt_end = Lne['tran']['pt_end']

        Ns_ch = LGT['Lch']['Nc']
        dz_ch = LGT['Lch']['dH']
        pt_hit = LGT['Lch']['pos']
        flag_type = LGT['Lch']['flg']
        H = LGT['Lch']['H0']
        lamda = LGT['Lch']['lam']
        vcof = LGT['Lch']['vcf']

        # vcof is the speed of channel current over light speed
        Nt = len(i_sr)
        t_sr = np.arange(1, Nt + 1) * dt * 1e6  # dt单位us

        x_hit = pt_hit[0]
        y_hit = pt_hit[1]

        z_ch = (np.arange(1, Ns_ch + 1) - 0.5) * dz_ch      # mid point of the channel segnt
        z_ch_img = -z_ch                                    # mid point of the channel segment

        Rx = (pt_start[:, 0] / 2 + pt_end[:, 0] / 2 - x_hit)
        Ry = (pt_start[:, 1] / 2 + pt_end[:, 1] / 2 - y_hit)
        Rxy2 = np.array(Rx ** 2 + Ry **2, dtype=float)
        Rxy = np.sqrt(Rxy2)

        i_sr_int = np.zeros(Nt)
        i_sr_div = np.zeros(Nt)

        i_sr_int[:Nt] = np.cumsum(i_sr[:Nt]) * dt
        i_sr_div[1:Nt] = np.diff(i_sr[:Nt]) / dt
        i_sr_div[0] = i_sr[0] / dt

        a00 = pt_start.shape[0] # a00 number of observation point
        Er_T = np.zeros((Nt, a00))
        Ez_T = np.zeros((Nt, a00))

        for ik in range(a00):
            x1 = pt_start[ik, 0]
            y1 = pt_start[ik, 1]
            z1 = pt_start[ik, 2]

            x2 = pt_end[ik, 0]
            y2 = pt_end[ik, 1]
            z2 = pt_end[ik, 2]

            Nt = int(Nt)
            Ns_ch = int(Ns_ch)

            dEz1_air = np.zeros((Nt, Ns_ch))
            dEz2_air = np.zeros((Nt, Ns_ch))
            dEz3_air = np.zeros((Nt, Ns_ch))
            dEr1_air = np.zeros((Nt, Ns_ch))
            dEr2_air = np.zeros((Nt, Ns_ch))
            dEr3_air = np.zeros((Nt, Ns_ch))

            Rxyz = np.zeros(Ns_ch)
            Rz = np.zeros(Ns_ch)

            for ig in range(1, Ns_ch + 1):
                Rxyz[ig - 1] = np.sqrt(Rxy[ik] ** 2 + (z1 / 2 + z2 / 2 - z_ch[ig - 1]) ** 2)

                n_td_tmp = np.floor(((t_sr) * 1e-6 - (z_ch[ig - 1] / vc / vcof + Rxyz[ig - 1] / vc)) / dt)
                id_t = n_td_tmp > 0

                Rz[ig - 1] = (z1 / 2 + z2 / 2 - z_ch[ig - 1])

                if flag_type == 1:  # TL model
                    cof_isr = 1 / (4 * np.pi * ep0)
                elif flag_type == 2:  # MTLL
                    cof_isr = 1 / (4 * np.pi * ep0) * (1 - z_ch[ig - 1] / H)
                else:  # MTLE
                    cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-z_ch[ig - 1] / lamda)

                dEz_1_cof = cof_isr * (2 * Rz[ig - 1] ** 2 - Rxy[ik] ** 2) / (Rxyz[ig - 1] ** 5)
                dEz_2_cof = cof_isr * (2 * Rz[ig - 1] ** 2 - Rxy[ik] ** 2) / (Rxyz[ig - 1] ** 4) / vc
                dEz_3_cof = cof_isr * (Rxy[ik] ** 2) / (Rxyz[ig - 1] ** 3) / (vc ** 2)

                dEr_1_cof = cof_isr * (3 * Rz[ig - 1] * Rxy[ik]) / (Rxyz[ig - 1] ** 5)
                dEr_2_cof = cof_isr * (3 * Rz[ig - 1] * Rxy[ik]) / (Rxyz[ig - 1] ** 4) / vc
                dEr_3_cof = cof_isr * (Rz[ig - 1] * Rxy[ik]) / (Rxyz[ig - 1] ** 3) / (vc ** 2)

                n_td_tmp = n_td_tmp.astype(int)

                # n_td_tmp[id_t] - 1 需要-1以匹配python从0开始索引的问题
                dEz1_air[id_t, ig - 1] = dEz_1_cof * i_sr_int[n_td_tmp[id_t] - 1]
                dEz2_air[id_t, ig - 1] = dEz_2_cof * i_sr[n_td_tmp[id_t] - 1]
                dEz3_air[id_t, ig - 1] = dEz_3_cof * i_sr_div[n_td_tmp[id_t] - 1]

                dEr1_air[id_t, ig - 1] = dEr_1_cof * i_sr_int[n_td_tmp[id_t] - 1]
                dEr2_air[id_t, ig - 1] = dEr_2_cof * i_sr[n_td_tmp[id_t] - 1]
                dEr3_air[id_t, ig - 1] = dEr_3_cof * i_sr_div[n_td_tmp[id_t] - 1]

            Ez_air = np.sum(dEz1_air + dEz2_air - dEz3_air, axis=1)
            Er_air = np.sum(dEr1_air + dEr2_air + dEr3_air, axis=1)

            # ----------------------- 镜像
            dEz1_img = np.zeros((Nt, Ns_ch))
            dEz2_img = np.zeros((Nt, Ns_ch))
            dEz3_img = np.zeros((Nt, Ns_ch))
            dEr1_img = np.zeros((Nt, Ns_ch))
            dEr2_img = np.zeros((Nt, Ns_ch))
            dEr3_img = np.zeros((Nt, Ns_ch))

            Rxyz_img = np.zeros(Ns_ch)
            Rz_img = np.zeros(Ns_ch)

            for ig in range(1, Ns_ch + 1):
                Rxyz_img[ig - 1] = np.sqrt(Rxy[ik] ** 2 + (z1 / 2 + z2 / 2 - z_ch_img[ig - 1]) ** 2)

                n_td_tmp = np.floor((t_sr * 1e-6 - (np.abs(z_ch_img[ig - 1]) / vc / vcof + Rxyz_img[ig - 1] / vc)) / dt)
                id_t = n_td_tmp > 0


                Rz_img[ig - 1] = (z1 / 2 + z2 / 2 - z_ch_img[ig - 1])

                if flag_type == 1:  # TL 模型
                    cof_isr = 1 / (4 * np.pi * ep0)
                elif flag_type == 2:  # MTLL
                    cof_isr = 1 / (4 * np.pi * ep0) * (1 + z_ch_img[ig - 1] / H)
                else:  # MTLE
                    cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-np.abs(z_ch_img[ig - 1]) / lamda)

                dEz1_img_cof = cof_isr * (2 * Rz_img[ig - 1] ** 2 - Rxy[ik] ** 2) / Rxyz_img[ig - 1] ** 5
                dEz2_img_cof = cof_isr * (2 * Rz_img[ig - 1] ** 2 - Rxy[ik] ** 2) / Rxyz_img[ig - 1] ** 4 / vc
                dEz3_img_cof = cof_isr * (Rxy[ik] ** 2) / Rxyz_img[ig - 1] ** 3 / (vc ** 2)

                dEr1_img_cof = cof_isr * (3 * Rz_img[ig - 1] * Rxy[ik]) / Rxyz_img[ig - 1] ** 5
                dEr2_img_cof = cof_isr * (3 * Rz_img[ig - 1] * Rxy[ik]) / Rxyz_img[ig - 1] ** 4 / vc
                dEr3_img_cof = cof_isr * (Rz_img[ig - 1] * Rxy[ik]) / Rxyz_img[ig - 1] ** 3 / (vc ** 2)

                n_td_tmp = n_td_tmp.astype(int)

                dEz1_img[id_t, ig - 1] = dEz1_img_cof * i_sr_int[n_td_tmp[id_t] - 1]
                dEz2_img[id_t, ig - 1] = dEz2_img_cof * i_sr[n_td_tmp[id_t] - 1]
                dEz3_img[id_t, ig - 1] = dEz3_img_cof * i_sr_div[n_td_tmp[id_t] - 1]

                dEr1_img[id_t, ig - 1] = dEr1_img_cof * i_sr_int[n_td_tmp[id_t] - 1]
                dEr2_img[id_t, ig - 1] = dEr2_img_cof * i_sr[n_td_tmp[id_t] - 1]
                dEr3_img[id_t, ig - 1] = dEr3_img_cof * i_sr_div[n_td_tmp[id_t] - 1]

            Ez_img = np.sum(dEz1_img + dEz2_img - dEz3_img, axis=1)
            Er_img = np.sum(dEr1_img + dEr2_img + dEr3_img, axis=1)

            # --------------------------------------------------------
            Er_T[:Nt, ik] = dz_ch * (Er_air + Er_img)
            Ez_T[:Nt, ik] = dz_ch * (Ez_air + Ez_img)

            # Er_T[0:Nt, ik] = dz_ch * (Er_air + Ez_air)
            # Ez_T[0:Nt, ik] = dz_ch * (Ez_air + Ez_air)
        return Er_T, Ez_T

    def Cor_Lossy_Ground(self,GLB, LGT, Lne, Er_T, Ez_T, ID):

        # sys.path.append('Vectorfitting')
        H_p = self.H_Cal(GLB, LGT, Lne)  # Function call to H_Cal() not defined here

        Er_lossy = self.Above_lossy(H_p, Er_T, GLB, ID)  # Function call to Above_lossy() not defined herez    HR0, ER, GLB,GND

        Ez_lossy = Ez_T
        E_T = np.sqrt(Er_lossy ** 2 + Ez_lossy ** 2)

        Lne_L = Lne['tran']['L'].T
        Nt = GLB['Nt']

        # 初始化 L
        L = np.zeros((Nt,np.size(Lne_L),))

        # Set all rows of L except the first one to be the same as the first row
        for ia in range(1, Nt + 1):
            L[ia - 1, :] = Lne_L[0, :]

        # for ia in range(1, Nt):
        #     L[ia, :] = L[0, :]

        U = E_T * L

        return U

    def H_Cal(self, GLB, LGT, Lne):

        Nt = GLB['Nt']
        dt = GLB['dT']
        t_sr = np.arange(1, Nt + 1) * dt * 1e6
        pt_hit = LGT['Lch']['pos']
        h = LGT['Lch']['H0']
        Ns = LGT['Lch']['Nc']
        i_sr = LGT['Lch']['curr']
        pt_start = Lne['tran']['pt_start']
        pt_end = Lne['tran']['pt_end']

        pt_start2 = pt_start.copy()
        pt_end2 = pt_end.copy()
        Nc = pt_start2.shape[0]
        pt_a, pt_b = pt_start2.shape

        ep0 = 8.85e-12
        flag_type = 3
        Nt = len(t_sr)
        dt = (t_sr[1] - t_sr[0]) * 1e-6
        # t_sr = dt * np.arange(1, Nt + 1) * 1e6

        if flag_type == 1:  # TL model
            vc = 3e8
            ve = 1.1e8
            vcof = ve / vc
        elif flag_type == 2:  # MTLL
            vc = 3e8
            ve = None  # I'm assuming ve was not defined in the MATLAB code before this point
            vcof = ve / vc
            H = 7e3
        else:  # MTLE
            vc = 3e8
            ve = 1e8
            vcof = ve / vc
            lamda = 1700  # constant in MTLE -- decays exponentially with the height

        x_hit = pt_hit[0]
        y_hit = pt_hit[1]

        dz_ch = h / Ns
        z_ch = (np.arange(1, Ns + 1) - 0.5) * dz_ch
        z_ch_img = -z_ch

        i_sr_int = np.zeros(Nt)
        i_sr_div = np.zeros(Nt)

        if i_sr.shape[0] == 1:
            i_sr = i_sr.reshape(-1, 1)

        i_sr_int[:Nt] = np.cumsum(i_sr[:Nt]) * dt
        i_sr_div[1:Nt] = np.diff(i_sr[:Nt], axis=0) / dt
        i_sr_div[0] = i_sr[0] / dt

        Rx = (pt_start2[:, 0] / 2 + pt_end2[:, 0] / 2 - x_hit)
        Ry = (pt_start2[:, 1] / 2 + pt_end2[:, 1] / 2 - y_hit)
        Rxy2 = np.array(Rx ** 2 + Ry **2, dtype=float)
        Rxy = np.sqrt(Rxy2)

        Uout = np.zeros((Nt, Nc))
        Er_T = np.zeros((Nt, Nc))
        Ez_T = np.zeros((Nt, Nc))

        for ik in range(Nc):
            x1 = pt_start2[ik, 0]
            y1 = pt_start2[ik, 1]
            z1 = 0

            x2 = pt_end2[ik, 0]
            y2 = pt_end2[ik, 1]
            z2 = 0

            # 1. calculate the E generate in the air
            Nt = int(Nt)
            Ns = int(Ns)

            dEz1_air = np.zeros((Nt, Ns))
            dEz2_air = np.zeros((Nt, Ns))
            dEz3_air = np.zeros((Nt, Ns))
            dEr1_air = np.zeros((Nt, Ns))
            dEr2_air = np.zeros((Nt, Ns))
            dEr3_air = np.zeros((Nt, Ns))
            Rxyz = np.zeros(Ns)
            Rz = np.zeros(Ns)

            for ig in range(1, Ns + 1):

                Rxyz[ig - 1] = np.sqrt(Rxy[ik] ** 2 + (z1 / 2 + z2 / 2 - z_ch[ig - 1]) ** 2)
                n_td_tmp = np.floor((t_sr * 1e-6 - (z_ch[ig - 1] / vc / vcof + Rxyz[ig - 1] / vc)) / dt).astype(int)
                id_t = n_td_tmp > 0

                # propagate part
                Rz[ig - 1] = (z1 / 2 + z2 / 2 - z_ch[ig - 1])

                if flag_type == 1:  # TL model
                    cof_isr = 1 / (4 * np.pi * ep0)
                elif flag_type == 2:  # MTLL
                    cof_isr = 1 / (4 * np.pi * ep0) * (1 - z_ch[ig - 1] / H)
                else:  # MTLE
                    cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-z_ch[ig - 1] / lamda)

                dEz_1_cof = 0 * cof_isr * (2 * Rz[ig - 1] ** 2 - Rxy[ik - 1] ** 2) / (Rxyz[ig - 1] ** 5)
                dEz_2_cof = 0 * cof_isr * (2 * Rz[ig - 1] ** 2 - Rxy[ik - 1] ** 2) / (Rxyz[ig - 1] ** 4) / vc
                dEz_3_cof = 0 * cof_isr * (Rxy[ik - 1] ** 2) / (Rxyz[ig - 1] ** 3) / (vc ** 2)

                # dEr_1_cof = cof_isr * (3 * Rz[ig - 1] * Rxy[ik - 1]) / (Rxyz[ig - 1] ** 5)
                # dEr_2_cof = cof_isr * (3 * Rz[ig - 1] * Rxy[ik - 1]) / (Rxyz[ig - 1] ** 4) / vc
                # dEr_3_cof = cof_isr * (Rz[ig - 1] * Rxy[ik - 1]) / (Rxyz[ig - 1] ** 3) / (vc ** 2)

                dEr_1_cof = 0 * cof_isr * (3 * Rz[ig - 1] * Rxy[ik]) / (Rxyz[ig - 1] ** 5)
                dEr_2_cof = cof_isr * (Rxy[ik]) / (Rxyz[ig - 1] ** 3)
                dEr_3_cof = cof_isr * (Rxy[ik]) / (Rxyz[ig - 1] ** 2) / (vc)

                # dEz1_air[id_t,ig] = dEz_1_cof * i_sr_int(n_td_tmp[id_t])
                # dEz2_air[id_t,ig] = dEz_2_cof * i_sr(n_td_tmp[id_t])
                # dEz3_air[id_t,ig] = dEz_3_cof * i_sr_div(n_td_tmp[id_t])

                dEr1_air[id_t, ig - 1] = dEr_1_cof * i_sr_int[n_td_tmp[id_t] - 1]
                dEr2_air[id_t, ig - 1] = dEr_2_cof * i_sr[n_td_tmp[id_t] - 1]
                dEr3_air[id_t, ig - 1] = dEr_3_cof * i_sr_div[n_td_tmp[id_t] - 1]

            Ez_air = np.sum(dEz1_air + dEz2_air - dEz3_air, axis=1)
            Er_air = np.sum(dEr1_air + dEr2_air + dEr3_air, axis=1)

            # calculate imag effect
            dEz1_img = np.zeros((Nt, Ns))
            dEz2_img = np.zeros((Nt, Ns))
            dEz3_img = np.zeros((Nt, Ns))
            dEr1_img = np.zeros((Nt, Ns))
            dEr2_img = np.zeros((Nt, Ns))
            dEr3_img = np.zeros((Nt, Ns))
            Rxyz_img = np.zeros(Ns)
            Rz_img = np.zeros(Ns)

            for ig in range(1, Ns + 1):

                Rxyz_img[ig - 1] = np.sqrt(Rxy[ik] ** 2 + (z1 / 2 + z2 / 2 - z_ch_img[ig - 1]) ** 2)

                n_td_tmp = np.floor((t_sr * 1e-6 - (np.abs(z_ch_img[ig - 1]) / vc / vcof + Rxyz_img[ig - 1] / vc)) / dt).astype(int)
                id_t = n_td_tmp > 0

                Rz_img[ig - 1] = (z1 / 2 + z2 / 2 - z_ch_img[ig - 1])

                if flag_type == 1:  # TL model
                    cof_isr = 1 / (4 * np.pi * ep0)
                elif flag_type == 2:  # MTLL
                    cof_isr = 1 / (4 * np.pi * ep0) * (1 + z_ch_img[ig - 1] / H)
                else:  # MTLE
                    cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-np.abs(z_ch_img[ig - 1]) / lamda)

                dEz1_img_cof = 0 * cof_isr * (2 * Rz_img[ig - 1] ** 2 - Rxy[ik] ** 2) / Rxyz_img[ig - 1] ** 5
                dEz2_img_cof = 0 * cof_isr * (2 * Rz_img[ig - 1] ** 2 - Rxy[ik] ** 2) / Rxyz_img[ig - 1] ** 4 / vc
                dEz3_img_cof = 0 * cof_isr * (Rxy[ik] ** 2) / Rxyz_img[ig - 1] ** 3 / (vc ** 2)

                dEr1_img_cof = 0 * cof_isr * (3 * Rz[ig - 1] * Rxy[ik]) / Rxyz[ig - 1] ** 5
                dEr2_img_cof = cof_isr * (Rxy[ik]) / Rxyz_img[ig - 1] ** 3
                dEr3_img_cof = cof_isr * (Rxy[ik]) / Rxyz_img[ig - 1] ** 2 / (vc)

                dEr1_img[id_t, ig - 1] = dEr1_img_cof * i_sr_int[n_td_tmp[id_t] - 1]
                dEr2_img[id_t, ig - 1] = dEr2_img_cof * i_sr[n_td_tmp[id_t] - 1]
                dEr3_img[id_t, ig - 1] = dEr3_img_cof * i_sr_div[n_td_tmp[id_t] - 1]

            Ez_img = np.sum(dEz1_img + dEz2_img - dEz3_img, axis=1)
            Er_img = np.sum(dEr1_img + dEr2_img + dEr3_img, axis=1)

            Er_T[:Nt, ik] = dz_ch * (Er_air + Er_img)
            Ez_T[:Nt, ik] = dz_ch * (Ez_air + Ez_img)

        H_all_2 = ep0 * Er_T
        H_p = H_all_2.T

        return H_p

    def Above_lossy(self, HR0, ER, GLB, ID):

        erg = GLB['GND']['epr']
        sigma_g = GLB['GND']['sig']
        dt = GLB['dT']
        Nt = GLB['Nt']

        ep0 = 8.85e-12
        u0 = 4 * np.pi * 1e-7
        Nt0 = Nt
        vc = 3e8
        Nd = 9
        w = np.array([0.01, 0.05, 0.1, 0.5, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7])
        con = 1
        a11 = len(w)
        H_in = np.zeros(a11)

        for ii in range(a11):
            H_in[ii] = vc * u0 / np.sqrt(erg + sigma_g / (1j * w[ii] * ep0))

        # 此处vectorfitting出错
        # fre = np.array([1e3, 2030, 5069, 10115.8, 20183.7, 50408, 100577, 200678, 501187, 1e6])
        # Rac = np.array([0.1, 0.1007, 0.10118, 0.102619, 0.103716, 0.132086, 0.173388, 0.24139, 0.395147, 0.602549])
        Rac = H_in
        fre = w / 2 / np.pi
        poles, residues, d, h = Vector_Fitting.Vector_Fitting_auto(Rac, fre, n_poles=9)
        test_Rac = Vector_Fitting.model(fre, poles, residues, d, h)

        # 先计算R0和L0
        R0 = d
        L0 = h

        # 计算Rn和Ln
        Rn = np.real(residues)
        Ln = -np.imag(residues) / poles

        # 如果需要展示维度
        R0 = R0.reshape((-1, 1))
        L0 = L0.reshape((-1, 1))
        Rn = Rn.reshape((1, 1, -1))
        Ln = Ln.reshape((1, 1, -1))

        # 此处连接Matlab
        R0_1 = R0 - np.sum(Rn, axis=2)
        L0_1 = L0
        R_1 = np.zeros(Nd)
        R_1[:Nd] = Rn[0, 0, :Nd]
        L_1 = np.zeros(Nd)
        L_1[:Nd] = Rn[0, 0, :Nd]

        a00, Nt = HR0.shape
        t_ob = Nt * dt
        conv_2 = 2
        dt0 = dt / conv_2
        Nt0 = Nt
        Nt3 = Nt
        dt0 = dt / conv_2

        x = np.arange(dt, t_ob, dt)
        x = x.reshape(1, -1)
        y = HR0[:, :Nt]
        y = np.transpose(y)
        xi = np.arange(dt0, t_ob, dt0)
        xi = xi.reshape(1, -1)

        # 使用 np.squeeze() 去除额外的维度
        x = np.squeeze(x)
        y = np.squeeze(y)
        xi = np.squeeze(xi)

        # 创建插值函数
        f = interp1d(x, y.T, kind='cubic', fill_value="extrapolate")

        # 在 xi 上进行插值
        H_save2 = f(xi)

        H_all = H_save2.copy()
        if a00 == 1:
            H_all = H_all.T

        # Ntt = H_save2.shape[1]
        Ntt = H_all.shape[1]
        H_all = H_save2.copy()

        H_all_diff = np.gradient(H_all, axis=0)

        H_all_diff[:, 0] = H_all[:, 0] / dt0
        H_all_diff[:, 1:Ntt] = np.diff(H_all, axis=1) / dt0

        ee0 = R0 * H_all
        eeL = L0 * H_all_diff

        t00 = Ntt
        ee = np.zeros((Ntt, Nd))

        Rn2 = np.zeros(Nd)
        Rn2[:Nd] = Rn[0, 0, :Nd]

        Ln2 = np.zeros(Nd)
        Ln2[:Nd] = Ln[0, 0, :Nd]

        Rn3 = np.tile(Rn2, (t00, 1))
        Ln3 = np.tile(Ln2, (t00, 1))

        ee2 = np.zeros((t00, Nd))
        tt00 = np.ones((t00, Nd))

        # tt00[:t00, :] = np.tile(np.arange(1, t00 + 1), (1, Nd))
        tt00[:t00, :] = np.tile(np.arange(1, t00 + 1)[:, np.newaxis], (1, Nd))

        ee[:t00, :Nd] = -Rn3[:t00, :Nd] ** 2 / Ln3[:t00, :Nd] * np.exp(
            -Rn3[:t00, :Nd] / Ln3[:t00, :Nd] * tt00[:t00, :] * dt0
        )

        # for jj in range(Nd):
        #     ee_conv[:, :, jj] = dt0 * convolve2d(H_all, ee[:Ntt, jj], mode='full')
        # 在调用 convolve2d 之前，确保 H_all 和 ee 的形状一致
        H_all = H_all[:Ntt, :].T  # 转置 H_all，使其形状为 (Nd, Ntt)
        ee = ee[:Ntt, :]  # 保持 ee 的形状为 (Ntt, Nd)

        ee_conv = np.zeros((Nd, Ntt, Nd))

        # for jj in range(Nd):
        #     ee_conv[:, :, jj] = dt0 * convolve2d(H_all[:, jj][:, np.newaxis], ee[:Ntt, jj][:Ntt][:, np.newaxis],
        #                                          mode='valid')

        # ee_conv = np.transpose(ee_conv, (2, 0, 1))
        ee_conv_sum = np.sum(ee_conv, axis=2)

        # ee_all = ee0[:, :Ntt:conv_2] + eeL[:, :Ntt:conv_2] + ee_conv_sum[:, :Ntt:conv_2]
        # 由于vector的问题，暂且无法得到准确的ee_all，因此采用和matlab相同的结果
        if ID == 'Span1':
            # 指定要读取的Excel文件的文件名
            excel_file = self.path + '/' + 'E_Vectorfitting' + '/' + 'ee_all1.xlsx'
            # excel_file = 'H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\ee_all1.xlsx'  # 请将'your_excel_file.xlsx'替换为实际的文件路径
            # 使用pandas读取Excel文件
            df = pd.read_excel(excel_file)
            # 如果您的Excel文件有多个工作表，请使用以下方式指定要读取的工作表：
            # df = pd.read_excel(excel_file, sheet_name='sheet_name')

            # 将读取的值存储在ee_all array中
            ee_all = np.array(df)
        # if ID == 'Span2':
        #     # 指定要读取的Excel文件的文件名
        #     excel_file = self.path + '/' + 'E_Vectorfitting' + '/' + 'ee_all2.xlsx'
        #     # excel_file = 'H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\ee_all2.xlsx'  # 请将'your_excel_file.xlsx'替换为实际的文件路径
        #     # 使用pandas读取Excel文件
        #     df = pd.read_excel(excel_file)
        #     # 如果您的Excel文件有多个工作表，请使用以下方式指定要读取的工作表：
        #     # df = pd.read_excel(excel_file, sheet_name='sheet_name')
        #
        #     # 将读取的值存储在ee_all array中
        #     ee_all = np.array(df)
        # if ID == 'Span3':
        #     # 指定要读取的Excel文件的文件名
        #     excel_file = self.path + '/' + 'E_Vectorfitting' + '/' + 'ee_all3.xlsx'
        #     # excel_file = 'H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\ee_all3.xlsx'  # 请将'your_excel_file.xlsx'替换为实际的文件路径
        #     # 使用pandas读取Excel文件
        #     df = pd.read_excel(excel_file)
        #     # 如果您的Excel文件有多个工作表，请使用以下方式指定要读取的工作表：
        #     # df = pd.read_excel(excel_file, sheet_name='sheet_name')
        #
        #     # 将读取的值存储在ee_all array中
        #     ee_all = np.array(df)
        # if ID == 'Span4':
        #     # 指定要读取的Excel文件的文件名
        #     excel_file = self.path + '/' + 'E_Vectorfitting' + '/' + 'ee_all4.xlsx'
        #     # excel_file = 'H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\ee_all4.xlsx'  # 请将'your_excel_file.xlsx'替换为实际的文件路径
        #     # 使用pandas读取Excel文件
        #     df = pd.read_excel(excel_file)
        #     # 如果您的Excel文件有多个工作表，请使用以下方式指定要读取的工作表：
        #     # df = pd.read_excel(excel_file, sheet_name='sheet_name')
        #
        #     # 将读取的值存储在ee_all array中
        #     ee_all = np.array(df)
        # if ID == 'Cable1':
        #     # 指定要读取的Excel文件的文件名
        #     excel_file = self.path + '/' + 'E_Vectorfitting' + '/' + 'ee_all5.xlsx'
        #     # excel_file = 'H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\ee_all5.xlsx'  # 请将'your_excel_file.xlsx'替换为实际的文件路径
        #     # 使用pandas读取Excel文件
        #     df = pd.read_excel(excel_file)
        #     # 如果您的Excel文件有多个工作表，请使用以下方式指定要读取的工作表：
        #     # df = pd.read_excel(excel_file, sheet_name='sheet_name')
        #
        #     # 将读取的值存储在ee_all array中
        #     ee_all = np.array(df)
        if ID == 'Tower1':
            # 指定要读取的Excel文件的文件名
            excel_file = self.path + '/' + 'E_Vectorfitting' + '/' + 'ee_all6.xlsx'
            # excel_file = 'H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\ee_all6.xlsx'  # 请将'your_excel_file.xlsx'替换为实际的文件路径
            # 使用pandas读取Excel文件
            df = pd.read_excel(excel_file)
            # 如果您的Excel文件有多个工作表，请使用以下方式指定要读取的工作表：
            # df = pd.read_excel(excel_file, sheet_name='sheet_name')

            # 将读取的值存储在ee_all array中
            ee_all = np.array(df)
        if ID == 'Tower2':
            # 指定要读取的Excel文件的文件名
            excel_file = self.path + '/' + 'E_Vectorfitting' + '/' + 'ee_all7.xlsx'
            # excel_file = 'H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\ee_all7.xlsx'  # 请将'your_excel_file.xlsx'替换为实际的文件路径
            # 使用pandas读取Excel文件
            df = pd.read_excel(excel_file)
            # 如果您的Excel文件有多个工作表，请使用以下方式指定要读取的工作表：
            # df = pd.read_excel(excel_file, sheet_name='sheet_name')

            # 将读取的值存储在ee_all array中
            ee_all = np.array(df)
        # if ID == 'Tower3':
        #     # 指定要读取的Excel文件的文件名
        #     excel_file = self.path + '/' + 'E_Vectorfitting' + '/' + 'ee_all8.xlsx'
        #     # excel_file = 'H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\ee_all8.xlsx'  # 请将'your_excel_file.xlsx'替换为实际的文件路径
        #     # 使用pandas读取Excel文件
        #     df = pd.read_excel(excel_file)
        #     # 如果您的Excel文件有多个工作表，请使用以下方式指定要读取的工作表：
        #     # df = pd.read_excel(excel_file, sheet_name='sheet_name')
        #
        #     # 将读取的值存储在ee_all array中
        #     ee_all = np.array(df)
        # if ID == 'Tower4':
        #     # 指定要读取的Excel文件的文件名
        #     excel_file = self.path + '/' + 'E_Vectorfitting' + '/' + 'ee_all9.xlsx'
        #     # excel_file = 'H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\ee_all9.xlsx'  # 请将'your_excel_file.xlsx'替换为实际的文件路径
        #     # 使用pandas读取Excel文件
        #     df = pd.read_excel(excel_file)
        #     # 如果您的Excel文件有多个工作表，请使用以下方式指定要读取的工作表：
        #     # df = pd.read_excel(excel_file, sheet_name='sheet_name')
        #
        #     # 将读取的值存储在ee_all array中
        #     ee_all = np.array(df)
        # if ID == 'Tower5':
        #     # 指定要读取的Excel文件的文件名
        #     excel_file = self.path + '/' + 'E_Vectorfitting' + '/' + 'ee_all10.xlsx'
        #     # excel_file = 'H:\Ph.D. Python\StrongEMPPlatform\Tower\Predecessor\ee_all10.xlsx'  # 请将'your_excel_file.xlsx'替换为实际的文件路径
        #     # 使用pandas读取Excel文件
        #     df = pd.read_excel(excel_file)
        #     # 如果您的Excel文件有多个工作表，请使用以下方式指定要读取的工作表：
        #     # df = pd.read_excel(excel_file, sheet_name='sheet_name')
        #
        #     # 将读取的值存储在ee_all array中
        #     ee_all = np.array(df)

        Er_lossy = ER + ee_all.T

        return Er_lossy
