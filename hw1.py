from env import Env
from trader import (Trader, ACTION_LIST)
import pandas as pd
import numpy as np
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


MAX_EPISODE = 30
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

N_F = 3 # 3 feature as state
N_A = 5 # 5 action
HEADER = ['open', 'high', 'low', 'close']


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

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

    sess = tf.Session()

    actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

    sess.run(tf.global_variables_initializer())

    # start training
    for i_episode in range(MAX_EPISODE):
        state = np.array([0, 0, 0])
        # reset training_env
        env.set_state(state)

        track_r = []
        for i in range(0, env.data_len()):

            action = actor.choose_action(state)
            # print('i: {}, action: {}？, state: {}'.format(i, action, state))
            (reward, state_) = env.step(i, action)

            track_r.append(reward)

            # gradient = grad[r + gamma * V(s_) - V(s)]
            td_error = critic.learn(state, reward, state_)
            # true_gradient = grad[logPi(s,a) * td_error]
            actor.learn(state, action, td_error)
            state = state_

        ep_rs_sum = sum(track_r)
        print("episode:", i_episode, "  reward:", ep_rs_sum)

    # Testing
    testing_data = pd.read_csv(args.training, names=['open', 'high', 'low', 'close'])
    output_file = open(args.output, 'w')
    trader = Trader()
    trader.load_data(testing_data)
    state = np.array([0, 0, 0])
    env.set_state(state)
    for i in range(trader.data_len()-1):
        if i != 0:
            trader.reaction(i)
        trend = actor.choose_action(state)
        action = trader.predict_action(trend, i)
        output_file.write(str(action) + '\n')
        state = env.get_env(0)
        print('Day {}: your money is {}'.format(i, trader.get_money()))

    output_file.close()

    if trader.current_state == -1:
        print('your final money is {}'.format(trader.get_money() - testing_data['open'][trader.data_len()-1]))
    if trader.current_state == 1:
        print('your final money is {}'.format(trader.get_money() + testing_data['open'][trader.data_len()-1]))
