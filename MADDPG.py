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

    def get_a(self, s):
        return self.sess.run(self.a, {self.S: s})

    def learn(self):                # TODO
        indices = np.random.choice(min(self.pointer, MEMORY_CAPACITY), size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]

        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})

        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.not_terminal: 1})

    def learn_actor(self, s, a):
        self.sess.run(self.atrain, {self.S: s, self.a: a})

    def learn_critic(self, s, a, r, s_):
        self.sess.run(self.ctrain, {self.S: s, self.a: a, self.R: r, self.S_: s_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def get_exp(self, indices):
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        return bs, ba, br, bs_

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
agents = []
for i in range(env.n):
    agents.append(DDPG(a_dim, s_dim, a_bound))
    agents[i].sess.run(tf.global_variables_initializer())
    agents[i].memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
    agents[i].pointer = 0

var = 5  # control exploration TODO
var_t = 0
v_t = 0

k = 1

test = 0
t1 = time.time()
# ddpg.saver.restore(ddpg.sess, './model/all/DDPG-RA-KNN-3-10')


def all_learn(agents):
    indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
    s_n = []
    a_n = []
    r_n = []
    s__n = []

    for agent in agents:
        s, a, r, s_ = agent.get_exp(indices)
        s_n.append(s)
        a_n.append(a)
        r_n.append(r)
        s__n.append(s_)

    for p, agent in enumerate(agents):
        agent.learn_critic(s_n, a_n, r_n[p], s__n)

    actor_a = [agent.get_a(s_n[p]) for p, agent in enumerate(agents)]

    for agent in agents:
        agent.learn_actor(s_n, actor_a)


num_epi = 0
max_r = 0
for i in range(MAX_EPISODES):
    obs_n = env.reset().copy()
    arri = 0
    map(lambda tt, s: tt.ini(s), zip(agents, obs_n))
    ep_reward = 0.0
    agent_reward = [0.0 for _ in range(env.n)]
    ep_energy = 0.0
    ep_queue = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        if np.random.uniform(0, 5) > var:     # 重新改变探索策略 TODO
            action_n = [agent.choose_action(obs) for agent, obs in zip(agents, obs_n)]
        else:
            action_n = np.random.randint(16807, env.n)
            times = 0
            while not env.is_excu_a(action_n):
                times += 1
                action_n = np.random.randint(16807, env.n)
                if times == 10000:
                    action_n = np.zeros(env.n)

        new_obs_n, r_n, done, info, e_n, q_n = env.step(action_n, 0)

        if test != 1:
            for p, agent in enumerate(agents):
                agent.store_transition(obs_n[p], action_n[p], r_n[p], new_obs_n[p])
                arri += obs_n[p][2 * env.n]

            if j % 50 == 0:
                if all(list(map(lambda tt: tt.pointer > MEMORY_CAPACITY, agents))):
                    all_learn(agents)

        obs_n = new_obs_n.copy()

        for p, r, e, q in enumerate(zip(r_n, e_n, q_n)):
            ep_reward += r
            agent_reward[p] += r
            ep_energy += e
            ep_queue += q

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



