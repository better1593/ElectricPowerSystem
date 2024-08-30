import pandas as pd
import numpy as np
from functools import reduce
a = [1,2,3]
b = [5,6,7]
s=[]
for x in a+b:
    s.append(x)

print(a+b)