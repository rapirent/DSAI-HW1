import pandas as pd
import numpy as np


class Env:

    def __init__(self, data=None, short_period=10, long_period=5):
        self._DATA = data
        self._SHORT_PERIOD = short_period
        self._LONG_PERIOD = long_period
        self.train_mean = None
        self.train_quantile = None
        self.error_count = [0,0,0]

    def load_data(self, data):
        self._DATA = data

    def data_len(self):
        return len(self._DATA['open'])

    def training_preprocess(self):
        self.train_quantile =  (np.percentile(self._DATA['close'], 10), np.percentile(self._DATA['close'], 40),
                                np.percentile(self._DATA['close'], 60), np.percentile(self._DATA['close'], 80))
        plus = lambda value: value if value > 0 else 0
        short_long_total = []
        for index in range(self.data_len()):
            long_total = np.mean([self._DATA['close'][plus(index - i)] for i in range(self._LONG_PERIOD)])
            short_total = np.mean([self._DATA['close'][plus(index - i)] for i in range(self._SHORT_PERIOD)])
            short_long_total.append(long_total - short_total)
        self.short_long_total_mean = np.mean(short_long_total)
        self.train_open_close_diff_mean = np.mean([ (i - j) for (i, j) in zip(self._DATA['open'], self._DATA['close']) ])

    def get_env(self, today, last_average):

        plus = lambda value: value if value > 0 else 0
        long_total = sum([self._DATA['close'][plus(today - i)] for i in range(self._LONG_PERIOD)])
        short_total = sum([self._DATA['close'][plus(today - i)] for i in range(self._SHORT_PERIOD)])
        moving_short_average = (short_total / self._SHORT_PERIOD) - self._DATA['open'][today]
        moving_long_average = (long_total / self._LONG_PERIOD) - self._DATA['open'][today]
        open_close_diff = self._DATA['open'][today] - self._DATA['close'][today]
        moving_average_diff = moving_short_average - moving_long_average

        if abs(moving_average_diff - self.short_long_total_mean) > 0.3:

            if (moving_average_diff > 0) and (last_average > 0):
                if self._DATA['close'][today] > self.train_quantile[2]:
                    action_real = 4
                else:
                    action_real = 3
            elif (moving_average_diff < 0) and (last_average < 0):
                if self._DATA['close'][today] < self.train_quantile[1]:
                    action_real = 0
                else:
                    action_real = 1
            else:
                action_real = 2
        else:
            action_real = 2

        return [moving_average_diff, moving_short_average, moving_long_average, action_real]

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
            reward = -20
            self.error_count[2] += 1

        return (reward, n_state)
