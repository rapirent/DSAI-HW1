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

MAX_EPISODE = 30
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic
N_F = 4 # 4 feature as state
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
        state = np.array([0, 0, 0, 0])
        # reset training_env

        track_r = []
        total_action = [0,0,0,0,0]
        error_count = [0, 0, 0]
        env.error_count = [0,0,0]
        error_index = [0,0,0,0,0]
        error_index_2 = [0,0,0,0,0]
        error_index_cumulator = []
        error_index_cumulator_2 = []
        for i in range(0, env.data_len()):

            action = actor.choose_action(state)
            (reward, state_) = env.step(i, action, action_real=state[3], last_average=state[0])

            track_r.append(reward)

            # gradient = grad[r + gamma * V(s_) - V(s)]
            td_error = critic.learn(state, reward, state_)
            # true_gradient = grad[logPi(s,a) * td_error]
            actor.learn(state, action, td_error)
            if (state[3] - action > 1) or (state[3] - action < -1):
                error_count[2] += 1
                error_index[action] += 1
                error_index_cumulator.append(action)
            elif state[3] != action:
                error_count[1] += 1
                error_index_2[action] += 1
                error_index_cumulator_2.append(action)
            else:
                error_count[0] += 1
            state = state_
            total_action[action] +=1
            # print(state[3], end=' ')
            # print(action, reward)

        ep_rs_sum = sum(track_r)
        print('the actions are \n 0: {0},1: {1}, 2: {2}, 3: {3}, 4: {4} '.format(total_action[0],total_action[1],total_action[2], total_action[3], total_action[4]))
        print('---\n0: {}, +-<1: {}, +->1: {}\n---'.format(error_count[0], error_count[1], error_count[2]))
        print('reward list:\n -100: {}, 10: {}, -50: {}'.format(env.error_count[0], env.error_count[1], env.error_count[2]))
        print('---')
        print('most error is {}'.format(error_index))
        print('less error is {}'.format(error_index_2))
        print('cumulator of most error: ', error_index_cumulator)
        print('cumulator of less error: ', error_index_cumulator_2)
        print("episode:", i_episode, "  reward:", ep_rs_sum)

    # Testing
    testing_data = pd.read_csv(args.testing, names=['open', 'high', 'low', 'close'])
    output_file = open(args.output, 'w')
    trader = StockTrader()
    trader.load_data(testing_data)
    env.load_data(testing_data)
    state = np.array([0, 0, 0, 0])
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
        state_ = np.array(env.get_env(today=i, last_average=state[0]))
        state = state_
        if (state[3] - trend > 1) or (state[3] - trend < -1):
            error_count[2] += 1
            error_index[trend] += 1
            error_index_cumulator.append(trend)
        elif state[3] - trend != 0:
            error_count[1] += 1
            error_index_2[trend] += 1
        else:
            error_count[0] += 1
    print('0: {}, +-<1: {}, +->1: {}'.format(error_count[0], error_count[1], error_count[2]))
    print('most error is {}'.format(error_index))
    print('less error is {}'.format(error_index_2))
    print('cumulator of most error: ', error_index_cumulator)

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
