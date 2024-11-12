import numpy as np

np.seterr(divide="ignore",invalid="ignore")

class Circuit_Generate():
    def __init__(self):
        pass

    def Circuit_Generate(self, Tower, Span, Cable, GLB):
        NTower = GLB['NTower']
        NSpan = GLB['NSpan']
        Node = np.zeros(NTower + NSpan, dtype=int)
        Branch = np.zeros(NTower + NSpan, dtype=int)

        for i in range(NTower):
            Node[i] = Tower[i]['Node']['num'][0]
            Branch[i] = Tower[i]['Bran']['num'][0]

        for i in range(NSpan):
            Node[NTower + i] = Span[i]['Node']['num'][0]
            Branch[NTower + i] = Span[i]['Bran']['num'][0]

        # Generate Span Circuit (As,Rs,Ls,Cs)
        Span_circuit = {}

        for i in range(NSpan):
            Span_circuit[i] = {}
            Span_circuit[i]['Ncon'] = int(Span[i]['Seg']['Ncon'])
            Span_circuit[i]['Lseg'] = int(Span[i]['Seg']['Lseg'])
            Span_circuit[i]['Nseg'] = int(Span[i]['Seg']['Nseg'])

            Len_Rs = int(Span_circuit[i]['Ncon'] * Span_circuit[i]['Nseg'])
            Len_Ls = int(Span_circuit[i]['Ncon'] * Span_circuit[i]['Nseg'])
            Len_Cs = int(Span_circuit[i]['Ncon'] * (Span_circuit[i]['Nseg'] + 1))
            Len_As_row = int(Span_circuit[i]['Ncon'] * Span_circuit[i]['Nseg'])
            Len_As_columns = int(Span_circuit[i]['Ncon'] * (Span_circuit[i]['Nseg'] + 1))

            # get Rs, Ls, Cs, As
            Rs = np.zeros((Len_Rs, Len_Rs))
            Ls = np.zeros((Len_Ls, Len_Ls))
            Cs = np.zeros((Len_Cs, Len_Cs))
            As = np.zeros((Len_As_row, Len_As_columns))

            for y in range(1, Span_circuit[i]['Nseg'] + 1):
                Rs[(y - 1) * Span_circuit[i]['Ncon']:(y * Span_circuit[i]['Ncon']),
                (y - 1) * Span_circuit[i]['Ncon']:(y * Span_circuit[i]['Ncon'])] = Span_circuit[i]['Lseg'] * \
                                                                             Span[i]['Para']['Imp']['R']
                Span_circuit[i]['R'] = Rs

                Ls[(y - 1) * Span_circuit[i]['Ncon']:(y * Span_circuit[i]['Ncon']),
                (y - 1) * Span_circuit[i]['Ncon']:(y * Span_circuit[i]['Ncon'])] = Span_circuit[i]['Lseg'] * \
                                                                             Span[i]['Para']['Imp']['L']
                Span_circuit[i]['L'] = Ls

            for y in range(1, Span_circuit[i]['Nseg'] + 2):
                if y == 1 or y == 31:
                    Cs[(y - 1) * Span_circuit[i]['Ncon']:(y * Span_circuit[i]['Ncon']),
                    (y - 1) * Span_circuit[i]['Ncon']:(y * Span_circuit[i]['Ncon'])] = Span_circuit[i]['Lseg'] * \
                                                                                 Span[i]['Para']['Imp']['C'] / 2
                else:
                    Cs[(y - 1) * Span_circuit[i]['Ncon']:(y * Span_circuit[i]['Ncon']),
                    (y - 1) * Span_circuit[i]['Ncon']:(y * Span_circuit[i]['Ncon'])] = Span_circuit[i]['Lseg'] * \
                                                                                 Span[i]['Para']['Imp']['C']
                Span_circuit[i]['C'] = Cs

            for y in range(1, Span_circuit[i]['Nseg'] * Span_circuit[i]['Ncon'] + 1):
                As[y - 1, y - 1] = -1
                As[y - 1, y - 1 + int(Span[i]['Seg']['Ncon'])] = 1
                Span_circuit[i]['A'] = As

        # Generate Tower&Span total Circuit (A,R,L,C)
        temp_Nb = 0
        temp_Nn = 0

        # Initilize total circuit parameters
        Nb_total = sum(Branch)
        Nn_total = sum(Node)

        R_total = np.zeros((Nb_total, Nb_total))
        L_total = np.zeros((Nb_total, Nb_total))
        C_total = np.zeros((Nn_total, Nn_total))
        A_total = np.zeros((Nb_total, Nn_total))

        for i in range(NTower):
            Nb = Tower[i]['Bran']['num'][0]
            Nn = Tower[i]['Node']['num'][0]

            idx_b = slice(temp_Nb, temp_Nb + Nb)
            idx_n = slice(temp_Nn, temp_Nn + Nn)

            R_total[idx_b, idx_b] = Tower[i]['CK_Para']['R']
            L_total[idx_b, idx_b] = Tower[i]['CK_Para']['L']

            C_total1 = np.linalg.inv(Tower[i]['CK_Para']['P'])
            C_total1[-1, -1] = 0  # Assuming last index for Nn
            # 扩充两个数组，以确保两个数组的行列大小一致
            # 获取两个数组的形状
            C_Tower = Tower[i]['CK_Para']['C']
            shape_C_total1 = C_total1.shape
            shape_C_Tower = C_Tower.shape

            # 确定目标形状
            target_shape = (max(shape_C_total1[0], shape_C_Tower[0]), max(shape_C_total1[1], shape_C_Tower[1]))

            # 扩展 C_total1 到目标形状
            if shape_C_total1 != target_shape:
                C_total1_expend = np.zeros(target_shape)
                C_total1_expend[:shape_C_total1[0], :shape_C_total1[1]] = C_total1
            else:
                C_total1_expend = C_total1

            # 扩展 C_Tower 到目标形状
            if shape_C_Tower != target_shape:
                C_Tower_expend = np.zeros(target_shape)
                C_Tower_expend[:C_Tower[0], :C_Tower[1]] = Tower[i]['CK_Para']['C']
            else:
                C_Tower_expend = Tower[i]['CK_Para']['C']

            C_total1 = C_total1_expend
            C_Tower = C_Tower_expend
            

            C_total3 = C_total1 + C_Tower
            C_total[idx_n, idx_n] = C_total3
            
            A_total[idx_b, idx_n] = Tower[i]['CK_Para']['A']
            temp_Nb += Nb
            temp_Nn += Nn

        for i in range(NSpan):

            Nb = Span[i]['Bran']['num'][0]
            Nn = Span[i]['Node']['num'][0]

            idx_b = slice(temp_Nb, temp_Nb + Nb)
            idx_n = slice(temp_Nn, temp_Nn + Nn)
            
            R_total[idx_b, idx_b] = Span_circuit[i]['R']
            L_total[idx_b, idx_b] = Span_circuit[i]['L']
            C_total[idx_n, idx_n] = Span_circuit[i]['C']
            A_total[idx_b, idx_n] = Span_circuit[i]['A']
            
            temp_Nb += Nb
            temp_Nn += Nn

        # Connecting wire between Tower and Span
        Tower_nodes = np.sum(Node[0:NTower])
        Tower_nodes_0 = np.sum(Node[0:NTower])
        Nb_total = R_total.shape[0]
        Nn_total = C_total.shape[0]
        len = Span[i]['S2Tmap']['head'].shape[0]

        for i in range(0, NSpan):
            for y in range(1, len):
                Span_num = int(Span[i]['S2Tmap']['head'][0, 0])
                Tower_num = int(Span[i]['S2Tmap']['head'][0, 1])
                Tower_nodes = np.sum(Node[0:(Tower_num - 1)])
                Span_nodes = np.sum(Node[0:(NTower + Span_num - 1)])

                # Nb_total += 1

                # 为A_total增加一行
                new_row_A = np.zeros((1, A_total.shape[1]))
                new_row_A[0, Span_nodes + int(Span[i]['S2Tmap']['head'][y, 0]) - 1] = -1
                new_row_A[0, Tower_nodes + int(Span[i]['S2Tmap']['head'][y, 1]) - 1] = 1
                A_total = np.vstack((A_total, new_row_A))

                # 为L_total和R_total增加一行一列
                L_total = np.pad(L_total, ((0, 1), (0, 1)), mode='constant', constant_values=0)
                R_total = np.pad(R_total, ((0, 1), (0, 1)), mode='constant', constant_values=0)
                L_total[-1, -1] = 1e-12  # 短路
                R_total[-1, -1] = 1e-9  # 短路

                # # Python中数组索引是基于0的，因此需要相应地调整
                # Span_head = int(Span[i]['S2Tmap']['head'][y, 0])
                # A_total_col = Span_nodes + Span_head - 1
                # A_total[Nb_total - 2, int(A_total_col)] = -1
                # A_total[Nb_total - 2, Tower_nodes + int(Span[i]['S2Tmap']['head'][y, 1]) - 1] = 1
                # L_total[Nb_total - 2, Nb_total - 2] = 1e-12  # short circuit
                # R_total[Nb_total - 2, Nb_total - 2] = 1e-9  # short circuit

                # 处理tail部分
                # 为A_total增加一行
                new_row_A = np.zeros((1, A_total.shape[1]))
                new_row_A[0, Span_nodes + int(Span[i]['S2Tmap']['head'][y, 0]) - 1] = -1
                new_row_A[0, Tower_nodes + int(Span[i]['S2Tmap']['head'][y, 1]) - 1] = 1
                A_total = np.vstack((A_total, new_row_A))

                # 为L_total和R_total增加一行一列
                L_total = np.pad(L_total, ((0, 1), (0, 1)), mode='constant', constant_values=0)
                R_total = np.pad(R_total, ((0, 1), (0, 1)), mode='constant', constant_values=0)
                L_total[-1, -1] = 1e-12  # 短路
                R_total[-1, -1] = 1e-9  # 短路
                # Span_num = int(Span[i]['S2Tmap']['head'][0, 0])
                # Tower_num = int(Span[i]['S2Tmap']['head'][0, 1])
                # Tower_nodes = np.sum(Node[0:(Tower_num - 1)])
                # Span_nodes = np.sum(Node[0:(NTower + Span_num - 1)])
                # Nb_total += 1
                #
                # # 调整索引以适应Python
                # A_total[Nb_total - 3, Span_nodes + int(Span[i]['S2Tmap']['head'][y, 0]) - 1] = -1
                # A_total[Nb_total - 3, Tower_nodes + int(Span[i]['S2Tmap']['head'][y, 1]) - 1] = 1
                # L_total[Nb_total - 3, Nb_total - 3] = 1e-12  # short circuit
                # R_total[Nb_total - 3, Nb_total - 3] = 1e-9  # short circuit

        return A_total, R_total, L_total, C_total, Tower_nodes_0