"""
This is a very simple training environment
"""

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
        self.use_spread = False

    def reset(self, position = 0):
        self.total_score = 0.0
        if (position > self.win_size):
            self.i = position
        else:
            self.i = self.win_size - 1
        self.the_end = False
        if (self.win_size == 1):
            state = self.data.loc[self.i][self.features]
        else:
            state = self.data.loc[self.i-self.win_size+1:self.i][self.features]
        state  = np.array(state)
        state = np.reshape(state, [1,self.win_size, self.observation_space])
        return state

    def step(self, action):
        self.i += 1
        next_state = self.data.loc[self.i-self.win_size+1:self.i][self.features]
        next_state = np.array(next_state)

        # BUY
        if action == 1:
            reward = 1e5*(self.data.at[self.i,'Close'] - self.data.at[self.i,'Open']) 
        # SELL
        elif action == 2:
            reward = 1e5*(self.data.at[self.i,'Open'] - self.data.at[self.i,'Close']) 
        # DO NOTHING
        else:
            reward = 0

        # 
        if(self.use_spread and action != 0):
            reward -= self.data.at[self.i,'Spread']

        self.total_score += reward

        if self.i > self.n - 2:
            self.the_end = True

        return next_state, reward


