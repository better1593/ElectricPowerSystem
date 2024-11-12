def Nle_gen(Tower, Tower_brans0):

    Ins_pos = []  # Branch position of SA
    Ins_type = []

    for i in range(len(Tower_brans0) - 1):
        if not Tower[i]['CK_Para']['Swh']:
            temp = Tower_brans0[i] + Tower[i]['CK_Para']['Swh']['pos'][:, 0]
            Ins_pos.extend(temp)

            temp2 = Tower[i]['CK_Para']['Swh']['dat']
            Ins_type.extend(temp2)

    SA_pos = []  # Branch position of SA
    SA_type = []

    for i in range(len(Tower_brans0) - 1):
        if not Tower[i]['CK_Para']['Nle']:
            temp = Tower_brans0[i] + Tower[i]['CK_Para']['Nle']['pos'][:, 0]
            SA_pos.extend(temp)

            temp2 = Tower[i]['CK_Para']['Nle']['dat']
            SA_type.extend(temp2)

    return Ins_pos, Ins_type, SA_pos, SA_type