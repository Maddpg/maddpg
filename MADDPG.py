import tensorflow as tf
import numpy as np
import gym
import time
import math
import sys


#####################  hyper parameters  ####################

MAX_EPISODES = 5000
MAX_EP_STEPS = 1000
LR_A = 0.0001    # learning rate for actor
LR_GF = 0.0003
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount  TODO
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32


RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'CO-v0'  # TODO


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.sess = tf.Session(config=config)

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.St = tf.placeholder(tf.float32, [None, s_dim], 'St')
        self.St_ = tf.placeholder(tf.float32, [None, s_dim], 'St_')
        self.not_terminal = tf.placeholder(tf.float32, 1, 'not_terminal')

        # 建立预测AC网络
        self.a = self._build_a(self.S,)
        self.q = self._build_c(self.S, self.a,)

        # 利用滑动平均建立targetAC网络
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        self.a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, self.a_, reuse=True, custom_getter=ema_getter)

        self.a_loss = - tf.reduce_mean(self.q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_ * self.not_terminal
            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def ini(self, s):
        print(self.sess.run(self.q, {self.S: s[np.newaxis, :]}))

    def choose_action(self, s):
        x = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return x

    def get_q(self, s, a):
        return self.sess.run(self.q, {self.a: a, self.S: s[np.newaxis, :]})

    def learn(self, n_t):                # TODO
        indices = np.random.choice(min(self.pointer, MEMORY_CAPACITY), size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]

        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})

        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.not_terminal: n_t})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, a_dim, activation=tf.nn.sigmoid, name='e', trainable=trainable)
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a, transpose_a=False) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
ddpg = DDPG(a_dim, s_dim, a_bound)


D_list = [1.0]
for d in D_list:

    ddpg.sess.run(tf.global_variables_initializer())
    ddpg.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
    ddpg.pointer = 0

    # if OUTPUT_GRAPH:
    #     tf.summary.FileWriter("logs/", ddpg.sess.graph)

    var = 5  # control exploration TODO
    var_t = 0
    v_t = 0

    k = 1

    test = 0
    t1 = time.time()
    # ddpg.saver.restore(ddpg.sess, './model/all/DDPG-RA-KNN-3-10')

    num_epi = 0
    max_r = 0
    for i in range(MAX_EPISODES):
        s = env.reset(d).copy()
        arri = 0
        ddpg.ini(s)
        ep_reward = 0
        ep_energy = 0
        ep_queue = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            # if test == 1:
            #     x = ddpg.choose_action(s)
            #     a = int(x)
            #     if a == 16807:
            #         a = a - 1
            #     if a not in a_list:
            #         a = np.random.choice(a_list)
            # else:
            if np.random.uniform(0, 5) > var:
                a = ddpg.choose_action(s)
            else:
                # a = np.random.choice(a_list)
                a = np.random.randint(117649)
                times = 0
                while not env.is_excu_a(a):
                    times += 1
                    a = np.random.randint(117649)
                    if times == 10000:
                        a = np.zeros(1)

            s_, r, done, info, e, q = env.step(a, 0)

            # if s_[0] > 1000:
            #     break

            if test != 1:
                ddpg.store_transition(s, a, r, s_)

                if ddpg.pointer > MEMORY_CAPACITY:
                    Not_terminal = [1]
                    if j > MAX_EP_STEPS-100:
                        Not_terminal = [0]
                    ddpg.learn(Not_terminal)

            arri += s[2*env.n+1]
            s = s_.copy()
            ep_reward += r
            ep_energy += e
            ep_queue += q
            # print(r)

            if j == MAX_EP_STEPS-1:
                f = open("DDPG-RA-%0.1f.txt" % d, "a")
                f.write("%0.2f %d \n" % (ep_reward, i))
                f.close()
                if var < 5:
                    f = open("episode-%0.1f.txt" % d, "a")
                    f.write("%0.2f %0.2f %d %d\n" % (ep_reward, ep_energy, ep_queue, arri))
                    f.close()
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, 'test: ', test, ' arriv: ', arri)
                # if ep_reward > -10:
                #     sys.exit(0)
                break

        num_epi += 1

        # if test != 0:
        #     v_t = test
        # test = abs(test - v_t)

        # if var != 0:
        #     var_t = var
        # var = abs(var - var_t)  # decay the action randomness  TODO

        if num_epi >= 15:
            var -= 0.5
            # test = 1
            if var < 0:
                var = 0
            num_epi = 0

        if ep_reward > max_r:
            max_r = ep_reward
            print("-------------------------------------\n-----------------------------------")
            ddpg.saver.save(ddpg.sess, './model/all/DDPG-RA-KNN-3-1-lambda5')
    print('Running time: ', time.time() - t1)

