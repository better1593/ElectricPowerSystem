import numpy as np
import pandas as pd
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# 定义股票交易环境
class StockEnv:
    def __init__(self, stock_data):
        self.stock_data = stock_data.reset_index(drop=True)
        self.current_step = 0
        self.done = False
        self.action_space = [0, 1, 2, 3]  # 0: 不买入, 1: 买入等级1, 2: 买入等级2, 3: 买入等级3

    def reset(self):
        self.current_step = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        if self.current_step < len(self.stock_data):
            return self.stock_data.iloc[self.current_step].values  # 包含收盘价和成交量
        else:
            return np.zeros(self.stock_data.shape[1])

    def step(self, action):
        current_price = self.stock_data.iloc[self.current_step]['close']
        current_volume = self.stock_data.iloc[self.current_step]['volume']
        next_state = self.get_state()
        reward = 0

        # 奖励机制：如果买入，计算未来价格变化的奖励
        if action > 0:  # 买入
            future_prices = self.stock_data.iloc[self.current_step + 1:self.current_step + 4]['close'].values
            reward = np.mean(future_prices) - current_price  # 计算奖励

        self.current_step += 1

        if self.current_step >= len(self.stock_data) - 1:
            self.done = True

        return next_state, reward, self.done


# 定义DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state.reshape(1, -1), verbose=0)[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

def read():
    df = pd.read_pickle('2022-2024年.pkl')
    df['涨跌幅'] = df['收盘价'] / df['开盘价'] - 1
    test_df = pd.read_pickle('2024年.pkl')
    test_df['涨跌幅'] = test_df['收盘价'] / test_df['开盘价'] - 1
    true_set = pd.read_excel('真集.xlsx', nrows=204)
    # A
    df_high = df[(df['涨跌幅'] > 0.09)]

    def code(x):
        return f"{x[2:]}.{x[:2]}"

    true_set['股票代码'] = true_set['股票代码'].apply(code)

    # B
    true_set['日期'] = pd.to_datetime(true_set['日期'], format='%Y%m%d')
    df['日期'] = pd.to_datetime(df['日期'])

    # C
    df = df.merge(true_set[['股票代码', '日期', 'label']], how='left', left_on=['股票代码', '日期'],
                  right_on=['股票代码', '日期'])
    df['label'] = df['label'].fillna(0)  # 未在真集中的股票标记为 0
    return df_high,true_set,df


# 主程序
if __name__ == "__main__":
    # 假设你有A组数据的收盘价和成交量
    # A组数据
    A_stock_data, B_buy_data  ,C_historical_data = read()

    # 使用A组数据创建环境
    env = StockEnv(A_stock_data)
    agent = DQNAgent(state_size=A_stock_data.shape[1], action_size=4)

    episodes = 1000
    batch_size = 32
    for e in range(episodes):
        state = env.reset()
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # Decay exploration rate
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    # 在这里，你可以使用D组数据进行模型验证
    print("Training complete. You can now validate using D data.")
