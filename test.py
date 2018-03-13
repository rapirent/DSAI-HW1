from classifier import Classifier
from stocktrader import (StockTrader, ACTION_LIST)
import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    training_data = pd.read_csv(args.training, names=['open', 'high', 'low', 'close'])
    c = Classifier()
    c.load_data(training_data)
    train_f, train_avg_all, train_avg_diff= c.featuring(training_data)
    c.train_feature(moving_average_all=train_avg_all, moving_average_diff=train_avg_diff)
    train_l = c.labeling(training_data)
    c.classify(feature=train_f, label=train_l)

    testing_data = pd.read_csv(args.testing, names=['open', 'high', 'low', 'close'])
    testing_data_feature, testing_avg_all, testing_avg_diff = c.featuring(testing_data)
    trend = c.predict(testing_data_feature)
    # print(trend)
    trader = StockTrader()
    trader.load_data(testing_data)
    output_file = open(args.output, 'w')
    for i in range(0, len(trend)-1):
        predict_action = trader.predict_action(trend[i], i)
        trader.reaction(i, predict_action)
        output_file.write(str(predict_action) + '\n')

        print('Day {}: your money is {} | open: {} | action : {}'.format(i, trader.get_money(), trader.get_today_price(i), predict_action))
    # for i in range(len(testing_data['open'])):
    #     if i > 0:
    #         trader.reaction(i, predict_action)
    #         output_file.write(str(predict_action) + '\n')
    #         print('Day {}: your money is {} | open: {} | action : {}'.format(i, trader.get_money(), trader.get_today_price(i), predict_action))

    #     trend = c.predict(testing_data_array[i])
    #     predict_action = trader.predict_action(trend, i)

    output_file.close()

    if trader.current_state == -1:
        final_line = 'your final money is {}, the last close price is {}'.format(
            trader.get_money() - testing_data['close'][trader.data_len() - 1],
            testing_data['close'][trader.data_len() - 1])
    elif trader.current_state == 1:
        final_line = 'your final money is {}, the last close price is {}'.format(
            trader.get_money() + testing_data['close'][trader.data_len() - 1],
            testing_data['close'][trader.data_len() - 1])
    else:
        final_line = 'your final money is {}, the last close price is {}'.format(
            trader.get_money(),
            testing_data['close'][trader.data_len() - 1])
    print(final_line)
    print('the origin buy-and-hold is {}'.format((testing_data['close'][trader.data_len() - 1] - testing_data['open'][1])))

