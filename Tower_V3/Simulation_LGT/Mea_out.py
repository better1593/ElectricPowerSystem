def Mea_out(GLB, Tower, Span, Tower_nodes0, Tower_brans0, Vnt_total2, Ibt_total2, Ins_on, Ins_pos):
    Nt = GLB['Nt']
    NTower = GLB['NTower']
    NSpan = GLB['NSpan']
    nodes_span = [0] * (NSpan + 1)
    bran_span = [0] * (NSpan + 1)

    for ia in range(NSpan):
        nodes_span[ia + 1] = Span[ia].Node['num'][0]

    for ia in range(NSpan):
        bran_span[ia + 1] = Span[ia].Bran['num'][0]

    for ia in range(NTower):
        tempa = Tower[ia].Meas['Ib'] + Tower_brans0[ia]
        Tower[ia].out['Ib'] = Tower[ia].Meas['Ib']
        Tower[ia].out['Idat'] = Ibt_total2[tempa, :Nt]

        tempb = Tower[ia].Meas['Vn'] + Tower_nodes0[ia]
        for ib in range(tempb.shape[0]):
            Tower[ia].out['Vn'] = Tower[ia].Meas['Vn']
            if tempb[ib, 1] == 0:
                Tower[ia].out['Vdat'][ib, :Nt] = Vnt_total2[tempb[ib, 0], :Nt]
            else:
                Tower[ia].out['Vdat'][ib, :Nt] = Vnt_total2[tempb[ib, 0], :Nt] - Vnt_total2[tempb[ib, 1], :Nt]

    for ia in range(NSpan):
        tempa = Span[ia].Meas['Ib'] + Tower_brans0[ia] + Tower_brans0[-1] + bran_span[ia]
        Span[ia].out['Ib'] = Span[ia].Meas['Ib']
        Span[ia].out['Idat'] = Ibt_total2[tempa, :Nt]

        tempb = Span[ia].Meas['Vn'] + Tower_nodes0[-1] + nodes_span[ia]
        for ib in range(tempb.shape[0]):
            Span[ia].out['Vn'] = Span[ia].Meas['Vn']
            if tempb[ib, 1] == 0:
                Span[ia].out['Vdat'][ib, :Nt] = Vnt_total2[tempb[ib, 0], :Nt]
            else:
                Span[ia].out['Vdat'][ib, :Nt] = Vnt_total2[tempb[ib, 0], :Nt] - Vnt_total2[tempb[ib, 1], :Nt]

    for ia in range(NTower):
        SA_mea = [[sum(Tower_brans0[:ia]) + 35, sum(Tower_brans0[:ia]) + 8, sum(Tower_brans0[:ia]) + 14],
                  [sum(Tower_brans0[:ia]) + 42, sum(Tower_brans0[:ia]) + 10, sum(Tower_brans0[:ia]) + 16],
                  [sum(Tower_brans0[:ia]) + 49, sum(Tower_brans0[:ia]) + 12, sum(Tower_brans0[:ia]) + 18]]
        SA_I = Ibt_total2[SA_mea[:, 0], :]
        SA_V = Vnt_total2[SA_mea[:, 1], :] - Vnt_total2[SA_mea[:, 2], :]
        SA_Energy = np.cumsum(np.abs(SA_I * SA_V), axis=1) * GLB['dT']
        Tower[ia].out['SA_Energy'] = SA_Energy

    for ia in range(NTower):
        temp = 1
        for ib in range(len(Ins_pos)):
            if Ins_pos[ib] > Tower_brans0[ia] and Ins_pos[ib] <= Tower_brans0[ia + 1]:
                Tower[ia].out['Ins_pos'].append(Ins_pos[ib] - Tower_brans0[ia])
                Tower[ia].out['Ins_on'].append(Ins_on[ib])
                temp += 1

    return Tower, Span