import pandas as pd
import numpy as np
a = {"1":[0,0],"2":[1,2],"3":[1,1],"4":[2,1]}
b = {"1":[1,1],"2":[2,1]}
a_df = pd.DataFrame(a)
b_df = pd.DataFrame(b)
c = pd.concat([a_df, b_df], axis=1)

H_invert = pd.DataFrame(np.linalg.pinv(a_df.values), columns=a_df.index, index=a_df.columns)
identity = np.dot(a_df, H_invert)
print(identity)
print(H_invert)
print(a_df)