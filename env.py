import pandas as pd
import numpy as np

class Env:

    def __init__(self, data=None, period=9, init_state=np.array([0, 0, 0])):
        self._DATA = data
        self._PERIOD = period
        self.last_state = init_state

    def set_state(self, state):
        self.last_state = state

    def load_data(self, data):
        self._DATA = data

    def data_len(self):
        return len(self._DATA['open'])

    def get_env(self, today):

        plus = lambda value: value if value > 0 else 0
        total = 0
        for j in [self._DATA['open'][plus(today - i)] for i in range(self._PERIOD)]:
            total += j
        moving_average = (total / self._PERIOD) - self._DATA['open'][0]
        moving_average_diff = moving_average - self.last_state[0]

        if today == 0:
            actual_trend = 0
        else:
            actual_trend = (self._DATA['open'][today] - self._DATA['open'][today - 1]) / self._DATA['open'][today - 1]

        if moving_average_diff > 1.5:
            action_real = 4
        elif moving_average_diff > 0.04:
            action_real = 3
        elif moving_average_diff < -0.19:
            action_real = 0
        elif moving_average_diff < -0.04:
            action_real = 1
        else:
            action_real = 2
        return np.array([moving_average, moving_average_diff, action_real])

    def step(self, today, action):

        if abs(self.last_state[2] - action)> 1:
            reward = -100
        elif self.last_state[2] == action:
            reward = 20
        elif action == 0 :
            reward = -50
        else:
            reward = -30

        n_state = self.get_env(today)
        self.set_state(n_state)
        return (reward, n_state)
