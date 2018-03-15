from env import Env
from stocktrader import (StockTrader, ACTION_LIST)
from ac import (Actor, Critic)
import pandas as pd
import numpy as np
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

MAX_EPISODE = 25
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic
N_F = 5 # 3 feature as state
N_A = 5 # 5 action

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
    env = Env()
    env.load_data(training_data)
    env.training_preprocess() # compute the training data quantile

    sess = tf.Session()

    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

    sess.run(tf.global_variables_initializer())

    # start training
    for i_episode in range(MAX_EPISODE):
        (last_long_avg, last_short_avg, state) = env.reset()
        # reset training_env

        track_r = []
        total_action = [0,0,0,0,0]
        error_count = [0, 0, 0]
        env.error_count = [0,0,0]
        error_index = [0,0,0,0,0]
        error_index_2 = [0,0,0,0,0]
        error_index_cumulator = []
        for i in range(env.data_len()):

            action = actor.choose_action(state)
            (last_long_avg_, last_short_avg_, reward, state_) = env.step(i, action, state[2], state[0], last_long_avg, last_short_avg)

            track_r.append(reward)

            # gradient = grad[r + gamma * V(s_) - V(s)]
            td_error = critic.learn(state, reward, state_)
            # true_gradient = grad[logPi(s,a) * td_error]
            actor.learn(state, action, td_error)
            state = state_
            last_long_avg = last_long_avg_
            last_short_avg = last_short_avg_


        ep_rs_sum = sum(track_r)
        print("episode:", i_episode, "  reward:", ep_rs_sum)

    # Testing
    testing_data = pd.read_csv(args.testing, names=['open', 'high', 'low', 'close'])
    output_file = open(args.output, 'w')
    trader = StockTrader()
    trader.load_data(testing_data)
    env.load_data(testing_data)
    (last_long_avg, last_short_avg, state) = env.reset()
    error_count = [0,0,0]
    error_index = [0,0,0,0,0]
    error_index_2 = [0,0,0,0,0]
    error_index_cumulator = []
    for i in range(trader.data_len()):
        if i > 0:
            ## according the yesterday action
            ## if i == data_len() - 1(269), it will be the 268's action to create a reaction
            trader.reaction(i, predict_action)
            output_file.write(str(predict_action) + '\n')
            print('Day {}: your money is {} | open: {} | action : {}'.format(i, trader.get_money(), trader.get_today_price(i), predict_action))

        trend = actor.choose_action(state)
        predict_action = trader.predict_action(trend, i)
        (last_long_avg_, last_short_avg_, state_) = env.get_env(i, state[0], last_long_avg, last_short_avg)
        state = state_
        last_long_avg = last_long_avg_
        last_short_avg = last_short_avg_

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
