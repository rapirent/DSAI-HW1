import pandas as pd
import numpy as np


class Env:

    def __init__(self, data=None, period=10):
        self._DATA = data
        self._PERIOD = period
        self.train_mean = None
        self.train_quantile = None
        self.error_count = [0,0,0]

    def load_data(self, data):
        self._DATA = data

    def data_len(self):
        return len(self._DATA['open'])

    def training_preprocess(self):
        self.train_quantile =  (np.percentile(self._DATA['open'], 25), np.percentile(self._DATA['open'], 40),
                            np.percentile(self._DATA['open'], 60), np.percentile(self._DATA['open'], 75))
        self.train_mean = np.mean(self._DATA['open'])
        print('percentile 25 - mean = {}'.format(self.train_quantile[0] - self.train_mean))
        print('percentile 40 - mean = {}'.format(self.train_quantile[1] - self.train_mean))
        print('percentile 60 - mean = {}'.format(self.train_quantile[2] - self.train_mean))
        print('percentile 75 - mean= {}'.format(self.train_quantile[3] - self.train_mean))


    def get_env(self, today, last_average):

        plus = lambda value: value if value > 0 else 0
        total = 0
        tenday_list = [self._DATA['open'][plus(today - i)] for i in range(self._PERIOD)]
        # print(tenday_list)
        for j in tenday_list:
            total += j
        moving_average = (total / self._PERIOD) - self._DATA['open'][today]
        moving_average_diff = moving_average - last_average
        # day_diff = self._DATA['close'][today] - self._DATA['open'][today]

        if today == 0:
            actual_trend = 0
        else:
            actual_trend = (self._DATA['open'][today] - self._DATA['open'][today - 1]) / self._DATA['open'][today - 1]

        if moving_average_diff > 1:
            action_real = 4
        elif moving_average_diff > 0.3:
            action_real = 3
        elif moving_average_diff < -1:
            action_real = 0
        elif moving_average_diff < -0.4:
            action_real = 1
        else:
            action_real = 2
        return [moving_average, moving_average_diff, action_real]

    def step(self, today, action, action_real, last_average):

        n_state = np.array(self.get_env(today, last_average))
        # self.set_state(n_state)
        if action_real == action:
            reward = 20
            self.error_count[1] += 1
        elif action_real - action > 1 or action_real - action < -1:
            reward = -100
            self.error_count[0] += 1
        else:
            reward = -50
            self.error_count[2] += 1

        return (reward, n_state)
