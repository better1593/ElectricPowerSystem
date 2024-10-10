import pandas as pd
import pickle
import numpy as np


df = pd.read_pickle('2022-2024年.pkl')
df['涨跌幅']= df['收盘价']/df['开盘价']-1
test_df = pd.read_pickle('2024年.pkl')
test_df['涨跌幅']= test_df['收盘价']/test_df['开盘价']-1
true_set = pd.read_excel('真集.xlsx',nrows=204)
#A
df_high = df[(df['涨跌幅']>0.09)]

def code(x):
    return f"{x[2:]}.{x[:2]}"
true_set['股票代码'] = true_set['股票代码'].apply(code)

#B
true_set['日期'] = pd.to_datetime(true_set['日期'], format='%Y%m%d')
df['日期'] = pd.to_datetime(df['日期'])

# C
df = df.merge(true_set[['股票代码', '日期', 'label']], how='left', left_on=['股票代码', '日期'], right_on=['股票代码', '日期'])
df['label'] = df['label'].fillna(0)  # 未在真集中的股票标记为 0


# 生成输入数据的函数
def prepare_timeseries_data(A_group, C_group, window_size=180):
    X = []
    y = []

    for index, row in A_group.iterrows():
        stock_code = row['股票代码']
        date = row['日期']

        # 获取股票的历史数据 (C组) 并生成时间序列特征
        stock_history = C_group[(C_group['股票代码'] == stock_code) & (C_group['日期'] < date)].tail(window_size)

        #if len(stock_history) == window_size:  # 确保有足够的历史数据
        # 提取特征（如：开盘价，收盘价，最高价，最低价，成交量等）
        features = stock_history[[ '收盘价', '成交量(股)']].values
        X.append(features)
        # 从 B 组中获取标签（如果在 B 组中，则有标签，否则为 0）
        label = C_group[(C_group['股票代码'] == stock_code) & (C_group['日期'] == date)]['label'].values[0]
        y.append(label)

    return np.array(X), np.array(y)

# 假设我们有A、B、C组数据，并已加载到 A_df, B_df, C_df
window_size = 180
X, y = prepare_timeseries_data(df_high, df, window_size)

# 确保输入数据的形状 (样本数, 时间序列长度, 特征数)
X = np.expand_dims(X, axis=-1)  # 添加最后一维特征数
