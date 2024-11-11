import pandas as pd
from sympy import false
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ast
# df = pd.read_csv("ROD.csv", index_col=0)
# df2 = pd.read_csv("ROD_modified.csv", index_col=0)
#
# for y,x,y2,x2 in zip(df.iloc[0,:].to_list(),df.columns.to_list(),df2.iloc[0,:].to_list(),df2.columns.to_list()):
#     fig, ax = plt.subplots()  # 创建图实例
#     ax.plot(np.arange(0,2001,1), ast.literal_eval(y), label='Origin Parameter')  # 作y1 = x 图，并标记此线名为linear
#     ax.plot(np.arange(0,2001,1), ast.literal_eval(y2), label='Change ROD')  # 作y2 = x^2 图，并标记此线名为quadratic
#     ax.legend() #自动检测要在图例中显示的元素，并且显示
#     ax.set_title(x)
#     plt.show()
#


df = pd.read_csv("Data/output/01_8_output.csv", index_col=0)
print(df.columns)