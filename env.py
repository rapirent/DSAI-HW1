import pandas as pd
import numpy as np

plus = lambda value: value if value > 0 else 0

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
        all_moving_average = []
        all_moving_average_diff = []
        for index in range(self.data_len()):
            all_moving_average.append(np.mean([ self._DATA['open'][plus(index - i)] for i in range(self._PERIOD) ] ) - self._DATA['open'][0])
        for index in range(1,self.data_len()):
            all_moving_average_diff.append(all_moving_average[index] - all_moving_average[index - 1])

        self.train_quantile =  (np.percentile(all_moving_average_diff, 30), np.percentile(all_moving_average_diff, 40),
                            np.percentile(all_moving_average_diff, 60), np.percentile(all_moving_average_diff, 70))
        self.train_mean = np.mean(all_moving_average)
        self.avg_diff = np.mean(all_moving_average_diff)
        error = []
        for index in all_moving_average_diff:
            error.append(index- self.avg_diff)
        squaredError = []
        for index in error:
            squaredError.append(index ** 2)

        self.avg_diff_mse = np.mean(squaredError) ** 0.5
        self.avg_diff_std = np.std(all_moving_average_diff)
        self.avg_diff_quantile = (self.avg_diff_std,
                                  self.avg_diff_std/4 - self.avg_diff,
                                  (-self.avg_diff_std/4) + self.avg_diff,
                                  -self.avg_diff_std)
        print('mse {}'.format(self.avg_diff_mse))
        print('mse/4 {}'.format(self.avg_diff_mse/4))
        print('std {}'.format(self.avg_diff_std))
        print('std/4 {}'.format(self.avg_diff_std/4))
        print('avg {}'.format(self.avg_diff))
        print('avg_diff_quantile 25 {}'.format(self.avg_diff_quantile[3]))
        print('avg_diff_quantile 40  {}'.format(self.avg_diff_quantile[2]))
        print('avg_diff_quantile 60  {}'.format(self.avg_diff_quantile[1]))
        print('avg_diff_quantile 75  {}'.format(self.avg_diff_quantile[0]))


    def get_env(self, today, last_average):

        moving_average = np.mean( [self._DATA['open'][plus(today - i)] for i in range(self._PERIOD)] ) - self._DATA['open'][0]
        moving_average_diff = moving_average - last_average
        # day_diff = self._DATA['close'][today] - self._DATA['open'][today]

        if today == 0:
            actual_trend = 0
        else:
            actual_trend = (self._DATA['open'][today] - self._DATA['open'][today - 1]) / self._DATA['open'][today - 1]

        if moving_average_diff > self.avg_diff_quantile[0]:
            action_real = 4
        elif moving_average_diff > self.avg_diff_quantile[1]:
            action_real = 3
        elif moving_average_diff < self.avg_diff_quantile[3]:
            action_real = 0
        elif moving_average_diff < self.avg_diff_quantile[3]:
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
