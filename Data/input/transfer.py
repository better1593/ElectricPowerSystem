import pandas as pd
import random

#df = pd.read_excel('Tower_Input.xlsx')
sheet=pd.read_excel('UI输出数据格式V4.xlsx',sheet_name=None)


def read_tower(dic):
    print(type(dic))

for k,v in sheet.items():
    #v = v.to_dict()
    if k =="Tower":
        read_tower(v)


