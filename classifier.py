from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np

plus = lambda value: value if value > 0 else 0
fill = lambda value: value if value > 9 else 9

class Classifier():

    def __init__(self, data=None):
        self._DATA = data
        self._PERIOD = 10

    def load_data(self, data):
        self._DATA = data

    def featuring(self, data):
        data_feature = []
        last_moving_average = 0
        moving_average_all = []
        moving_average_diff = []
        for index in range(len(data['open'])):
            if index < 9:
                moving_average_all.append(np.mean([ data['open'][0] for i in range(self._PERIOD) ]))
            else:
                moving_average_all.append(np.mean([ data['open'][index - i] for i in range(self._PERIOD)]))
        for index in range(len(data['open'])):
            moving_average_diff.append((moving_average_all[index] - moving_average_all[plus(index-1)])/moving_average_all[plus(index-1)])
        print(moving_average_all)
        for index in range(0, len(data['open'])):
            data_feature.append([moving_average_all[plus(index-2)],
                                moving_average_all[plus(index-1)],
                                moving_average_all[index],
                                moving_average_diff[index],
                                moving_average_diff[index-1]])

        return (data_feature, moving_average_all, moving_average_diff)

    def train_feature(self, moving_average_all, moving_average_diff):
        self.moving_average_all = moving_average_all
        self.moving_average_diff = moving_average_diff

    def labeling(self, data):
        data_label = []
        for index in range(0, len(data['open'])):
            diff = self.moving_average_diff[index]
            if diff > 0.01:
                data_label.append(5)
            elif diff > 0.005:
                data_label.append(4)
            elif diff < -0.01:
                data_label.append(0)
            elif diff < -0.005:
                data_label.append(1)
            else:
                data_label.append(2)

        return data_label

    def classify(self, feature, label):
        # self.bdt = AdaBoostRegressor(DecisionTreeRegressor(max_features='sqrt', splitter='random', min_samples_split=4, max_depth=5),
        #                         loss="square",
        #                         n_estimators=1000,
        #                         learning_rate=0.5)
        bdt = AdaBoostClassifier(n_estimators=500)
        self.bdt = bdt.fit(feature, label)

    def predict(self, data):
        return self.bdt.predict(data)
