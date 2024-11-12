import numpy as np

np.seterr(divide="ignore",invalid="ignore")

class Source_Position_Generate():
    def __init__(self):
        pass
    def Source_Position_Generate(self, Tower, Span, GLB, A_total):
        NTower = GLB['NTower']
        NSpan = GLB['NSpan']

        # Initialize nodes_tower, bran_tower, bran_soc_tower arrays, nodes_span and bran_span arrays
        nodes_tower = np.zeros(NTower, dtype=int)
        nodes_span = np.zeros(NSpan, dtype=int)
        bran_tower = np.zeros(NTower, dtype=int)
        bran_soc_tower = np.zeros(NTower, dtype=int)
        nodes_span = np.zeros(NSpan, dtype=int)
        bran_span = np.zeros(NSpan, dtype=int)

        for i in range(NTower):
            nodes_tower[i] = Tower[i]['Node']['num'][0]
            bran_tower[i] = Tower[i]['Bran']['num'][0]
            bran_soc_tower[i] = len(Tower[i]['Soc']['dat'])

        for i in range(NSpan):
            nodes_span[i] = Span[i]['Node']['num'][0]
            bran_span[i] = Span[i]['Bran']['num'][0]

        Tower_nodes0 = np.sum(nodes_tower)
        Tower_bran0 = np.sum(bran_tower)

        if GLB['Soc']['typ'] == 1:
            if GLB['Soc']['pos'][0] == 1:
                Soc_pos = sum(nodes_tower[: (GLB['Soc']['pos'][1] - 1)]) + GLB['Soc']['pos'][4]

            elif GLB['Soc']['pos'][0] == 2:
                temp1 = GLB['Soc']['pos'][1]
                temp2 = GLB['Soc']['pos'][3]
                temp3 = GLB['Soc']['pos'][4]
                Soc_pos = (
                        Tower_nodes0
                        + sum(nodes_span[: (temp1 - 1)])
                        + (temp3 - 1) * Span[temp2].Seg.Ncon
                        + temp2
                )
                # GLB['Soc']['pos'][4]
            else:
                Soc_pos = {}
        else:
            Soc_pos = {}

        Soc_data = GLB['Soc']['dat']
        bran_total = sum(bran_span) + sum(bran_tower)
        Nb = A_total.shape[0]
        Nt = len(Soc_data)
        Soc_data = Soc_data.reshape((Nt,1))
        temp = 0

        if not Soc_pos:
            Soc_data = np.zeros((Nb, Nt))

            # for i in range(NTower):
            #     rows_range = list(range(temp, temp + bran_soc_tower[i]))
            #     # rows_range = np.array(rows_range).reshape(1,len(rows_range))
            #     cols_range = list(range(Nt))
            #     # cols_range = np.array(cols_range).reshape(1,len(cols_range))
            #     # Soc_data[rows_range, cols_range] = Tower[i]['Soc']['dat']

            for i in range(NTower):
                rows_range = range(temp, temp + bran_soc_tower[i])
                for ia, row in enumerate(rows_range):
                    for col in range(Nt):
                        Soc_data[row, col] = Tower[i]['Soc']['dat'][ia, col]
                temp += bran_tower[i]

            for i in range(NSpan):
                rows_range = range(temp, temp + bran_span[i])
                for ia, row in enumerate(rows_range):
                    for col in range(Nt):
                        Soc_data[row, col] = Span[i]['Soc']['dat'][ia, col]
                temp += bran_span[i]


        return Soc_pos, Soc_data