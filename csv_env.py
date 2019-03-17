import pandas as pd
import numpy as np

class CsvEnv:
    def __init__(self, path, features, win_size=1):
        self.path = path
        self.features  = features
        self.observation_space = len(features)
        self.data = pd.read_csv(path,encoding='utf-16',sep='\t')
        self.n = len(self.data)
        self.win_size = win_size
        self.action_space = 3

    def reset(self):
        self.i = self.win_size -1
        self.total_profit = 0.0
        self.profitable = 0
        self.the_end = False
        state = self.data.loc[self.i-self.win_size+1:self.i][self.features]
        state  = np.array(state)
        return state

    def step(self, action):
        self.i += 1
        next_state = self.data.loc[self.i-self.win_size+1:self.i][self.features]
        next_state = np.array(next_state)

        # BUY
        if action == 1:
            reward = 1e5*(self.data.at[self.i,'Close'] - self.data.at[self.i,'Open']) 
             #  - self.data.at[self.i,'Spread']
        # SELL
        elif action == 2:
            reward = 1e5*(self.data.at[self.i,'Open'] - self.data.at[self.i,'Close']) 
            #    - self.data.at[self.i,'Spread']
        # DO NOTHING
        else:
            reward = 0.0
        if reward > 0:
            self.profitable += 1
        self.total_profit += reward
        if self.i > self.n - 2:
            self.the_end = True

        return next_state, reward


