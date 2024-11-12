import numpy as np


def Huri_Method (MCLGT, datanew):

    dataset = MCLGT['huri']
    R = MCLGT['radi']

    result = 0
    if not dataset:
        result = 1
        return result

    for i in range(dataset.shape[0]):
        sample = dataset[i]
        Dki = np.sqrt((datanew[4] - sample[4]) ** 2 + (datanew[5] - sample[5]) ** 2) #新落雷点和过去的距离
        if Dki <= R:
            if datanew[0] <= sample[0]:
                if datanew[1] >= sample[1]:
                    ds1 = np.sqrt((datanew[4] - sample[6]) ** 2 + (datanew[5] - sample[7]) ** 2)#新落雷点和两端的距离
                    ds2 = np.sqrt((datanew[4] - sample[8]) ** 2 + (datanew[5] - sample[9]) ** 2)
                    if ds1 >= sample[10] and ds2 >= sample[11]:
                        result = 1
                        break
                    else:
                        continue
                else:
                    continue
            else:
                continue
        else:
            continue

    return result
