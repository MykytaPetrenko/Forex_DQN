from csv_env import CsvEnv
from collections import deque
import random
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


#INPUT BLOCK
DATA_PATH = 'toCSV.csv'
# features names must be the same as CSV column names
features = ['RelativeHLC B#0', 'RelativeHLC B#1','RelativeHLC B#2', 'MinMax(16) B#0',
            'MinMax(16) B#1','MinMax(16) B#2','MinMax(16) B#3']

BATCH_SIZE = 24
GAMMA  = 0.95       # Discount rate

class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = 1.0
        self.memory = deque(maxlen = 256)
        self.action_space = action_space
        self.model  = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if(np.random.rand() < self.exploration_rate):
            return random.randrange(self.action_space)
        q = self.model.predict(state)
        return np.argmax(q[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, terminal in batch:
            q_upd =  (reward + GAMMA * np.amax(self.model.predict(next_state)[0]))
            q_val = self.model.predict(state)
            q_val[0][action] = q_upd
            self.model.fit(state, q_val, verbose=0)
        self.exploration_rate *= 0.999
        self.exploration_rate = max(0.025, self.exploration_rate)


env = CsvEnv(DATA_PATH, features)
state = env.reset()
solver = DQNSolver(env.observation_space, env.action_space)
total_profit = 0.0

log = pd.DataFrame()
j = 0
while not env.the_end:
    action = solver.act(state)
    next_state, reward = env.step(action)

    terminal = False
    if (solver.exploration_rate <= 0.05):
        total_profit += reward
    solver.remember(state, action, reward, next_state, terminal)
    state = next_state
    solver.experience_replay()
    print('bar {}/{}\texp_rate: {:.2f},\taction:{},\treward: {:.0f},\tprofit = {:.0f}%'.format(env.i,env.n,solver.exploration_rate,action,reward,total_profit))
    log.at[j,'bar'] = env.i
    log.at[j,'action'] = action
    log.at[j,'reward'] = reward
    log.at[j,'profit'] = total_profit
    j += 1

    # Save csv log each 1000 step
    if(j % 1000 == 0):
        log.to_csv('log.csv',sep='\t', encoding='utf-16')