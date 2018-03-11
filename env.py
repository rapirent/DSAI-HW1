import pandas as pd
import numpy as np

class Env:

    def __init__(self):
        self._DATA = None
        self._last_state = np.array([0, 0, 0])

    def load_data(self, data):
        self._DATA = data

    def data_len(self):
        return len(self._DATA['open'])

    def _get_env(self, today):
        if today < 10:
            today_ = 10
        else:
            today_ = today
        data_tenday = list(self._DATA['open'][today_-10:today+1])
        # print(list(data_tenday))
        moving_average = np.mean(data_tenday) - data_tenday[0]
        moving_average_diff = moving_average - self._last_state[0]
        # std = np.std(data_tenday)
        if today == 0:
            actual_trend = 0
        else:
            actual_trend = (self._DATA['open'][today] - self._DATA['open'][today - 1]) / self._DATA['open'][today - 1]

        if actual_trend > 0.01:
            action_real = 4
        elif actual_trend > 0.005:
            action_real = 3
        elif actual_trend < -0.01:
            action_real = 2
        elif actual_trend < -0.005:
            action_real = 1
        else:
            action_real = 0
        # print('adctual', action_real)
        return np.array([moving_average, moving_average_diff, action_real])

    def step(self, today, action):
        n_state = self._get_env(today)
        # print(n_state[2])
        if abs(n_state[2] - action)> 2:
            reward = -100
        elif n_state[2] == action:
            reward = 50
        elif n_state[2] > 0 and n_state[2] < 2 :
            reward = -70
        else:
            reward = -30
        return n_state, reward
