from csv_env import CsvEnv
from collections import deque
import random
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Flatten,Input
from keras.optimizers import Adam


#INPUT BLOCK
DATA_PATH = 'data/GBPUSD_M15_2018.csv'
WEIGHTS_PATH = 'model/weights.h5'
# features names must be the same as CSV column names
FEATURES = ['RelativeHLC(16) B#0', 'RelativeHLC(16) B#1',
            'RelativeHLC(16) B#3','RelativeHLC(16) B#4'] 
           #'MinMax(16) B#0','MinMax(16) B#1','MinMax(16) B#2','MinMax(16) B#3']

GAMMA  = 0.95       # Discount rate
START_POSITION = 0
MIN_EXPLORATION = 0.025
START_EXPLORATION = 1.0
EXPLORATION_DECAY = 0.999
MEMOTY_LIMIT = 512
BATCH_SIZE = 64
win_size = 1

class DQNAgent:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = START_EXPLORATION
        self.memory = deque(maxlen = MEMOTY_LIMIT)
        self.action_space = action_space
        self.observation_space = observation_space
        self.batch_size = BATCH_SIZE

        inputs = Input(shape=(win_size, observation_space))
        x = Flatten()(inputs)
        # can use LSTM layer instead of Flatten
        #x = LSTM(100, input_shape=(win_size,observation_space), activation='relu',dropout_W=0.2, dropout_U=0.2)(inputs)
        x = Dense(256, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        out = Dense(self.action_space, activation="linear")(x)
        self.model = Model(inputs, out)
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1,win_size, self.observation_space])
        if(np.random.rand() < self.exploration_rate):
            return random.randrange(self.action_space)
        q = self.model.predict(state)
        return np.argmax(q[0])

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)

        # Create containers for batch S-A-R-S
        S = np.zeros([self.batch_size,win_size, self.observation_space])
        A,R = [],[]
        Sn = np.zeros([self.batch_size,win_size, self.observation_space])
        for i in range(self.batch_size):
            S[i] = batch[i][0]
            A.append(batch[i][1])
            R.append(batch[i][2])
            Sn[i] = batch[i][3]

        S = np.array(S)
        Sn = np.array(Sn)

        Q = self.model.predict(S)
        Qn = self.model.predict(Sn)
        for i in range(self.batch_size):
            q_upd =  (R[i] + GAMMA * np.amax(Qn[i]))
            Q[i][A[i]] = q_upd
        self.model.fit(S,Q,verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(MIN_EXPLORATION, self.exploration_rate)

        #for state, action, reward, next_state, terminal in batch:
        #    q_upd =  (reward + GAMMA * np.amax(self.model.predict(next_state)[0]))
        #    q_val = self.model.predict(state)
        #    q_val[0][action] = q_upd
        #    self.model.fit(state, q_val, verbose=0)



if __name__ =="__main__":

    # Load csv to environment
    # # and identify features set
    env = CsvEnv(DATA_PATH, FEATURES, win_size)
    inaction_penalty = 1.0
    state = env.reset(START_POSITION)
    agent = DQNAgent(env.observation_space, env.action_space)

    log = pd.DataFrame()
    j = 0
    #np.random.seed(49)
    while not env.the_end:
        action = agent.act(state)
        next_state, reward = env.step(action)
        terminal = False
        if action == 0:
            reward = -inaction_penalty
        agent.remember(state, action, reward, next_state, terminal)
        state = next_state
        agent.experience_replay()
        print('bar {}/{}\texp_rate: {:.3f},\taction:{},\treward: {:.0f},\ttotal score = {:.0f} points'
              .format(env.i,env.n,agent.exploration_rate,action,reward,env.total_score))
        log.at[j,'bar'] = env.i
        log.at[j,'action'] = action
        log.at[j,'reward'] = reward
        log.at[j,'profit'] = env.total_score
        j += 1

        # Save csv log  and trained model each 1000 step
        if(j % 1000 == 0):
            log.to_csv('log.csv',sep='\t', encoding='utf-16')
            agent.model.save_weights(WEIGHTS_PATH)

        log.to_csv('log.csv',sep='\t', encoding='utf-16')
        agent.model.save_weights(WEIGHTS_PATH)    
