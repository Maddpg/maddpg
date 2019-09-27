import tensorflow as tf
import numpy as np
import gym
import time
import random
import math
import sys


#####################  hyper parameters  ####################

MAX_EPISODES = 3000
MAX_EP_STEPS = 1000
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount  TODO
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 16


RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'CO-v0'  # TODO


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, o_dim, a_bound, index):
        self.memory = np.zeros((MEMORY_CAPACITY, o_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.index = index
        # tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.sess = tf.Session(config=config)

        self.a_dim, self.o_dim, self.a_bound = a_dim, o_dim, a_bound,

        self.O = tf.placeholder(tf.float32, [None, o_dim])
        self.a_n = tf.placeholder(tf.float32, [None, env.n-1, a_dim])
        self.S = tf.placeholder(tf.float32, [None, env.n, o_dim], 's')

        self.O_ = tf.placeholder(tf.float32, [None, o_dim])
        self.a_n_ = tf.placeholder(tf.float32, [None, env.n-1, a_dim])
        self.S_ = tf.placeholder(tf.float32, [None, env.n, o_dim], 's_')

        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # 建立预测AC网络
        self.a = self._build_a(self.O,)
        self.q = self._build_c(self.S, self.a_n, self.a[:, np.newaxis, :],)

        # 利用滑动平均建立targetAC网络
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor-%d' % self.index)
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic-%d' % self.index)
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        self.a_ = self._build_a(self.O_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, self.a_n_, self.a_[:, np.newaxis, :], reuse=True, custom_getter=ema_getter)

        self.a_loss = - tf.reduce_mean(self.q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def choose_action(self, s):
        x = self.sess.run(self.a, {self.O: s[np.newaxis, :]})
        return x

    def get_q(self, s, a_n, a):
        return self.sess.run(self.q, {self.S: s, self.a_n: a_n, self.a: a})

    def get_a(self, s):
        return self.sess.run(self.a, {self.O: s})

    def get_a_(self, s):
        return self.sess.run(self.a_, {self.O_: s})

    def learn_actor(self, s, a_n, o):
        self.sess.run(self.atrain, {self.S: s, self.a_n: a_n, self.O: o})

    def learn_critic(self, s, a_n, a, r, s_, a_n_, a_):
        if self.pointer % 500 == 0:
            print(self.sess.run(self.td_error, {self.S: s, self.a_n: a_n, self.a: a, self.R: r, self.S_: s_,
                                                self.a_n_: a_n_, self.a_: a_}))
        self.sess.run(self.ctrain, {self.S: s, self.a_n: a_n, self.a: a, self.R: r,
                                    self.S_: s_, self.a_n_: a_n_, self.a_: a_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def get_exp(self, indices):
        bt = self.memory[indices, :]
        bs = bt[:, :self.o_dim]
        ba = bt[:, self.o_dim: self.o_dim + self.a_dim]
        br = bt[:, -self.o_dim - 1: -self.o_dim]
        bs_ = bt[:, -self.o_dim:]
        return bs, ba, br, bs_

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor-%d' % self.index, reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.swish, name='l1', trainable=trainable)
            a = tf.layers.dense(net, a_dim, activation=tf.nn.sigmoid, name='e', trainable=trainable)
            return a

    def _build_c(self, s, a_n, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic-%d' % self.index, reuse=reuse, custom_getter=custom_getter):
            input_sa = tf.contrib.layers.flatten(tf.concat((s, tf.concat((a_n, a), axis=1)), axis=2))
            net1 = tf.layers.dense(input_sa, 30, activation=tf.nn.swish, name='l21', trainable=trainable)
            net2 = tf.layers.dense(net1, 30, activation=tf.nn.swish, name='l22', trainable=trainable)
            net3 = tf.layers.dense(net2, 30, activation=tf.nn.swish, name='l23', trainable=trainable)
            qsa = tf.layers.dense(net3, 1, name='r', trainable=trainable)
            return qsa


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

o_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.max_m


def get_knn(k, a, a_list):
    ka_list = a_list.copy()
    L = []
    distances = [math.sqrt(np.sum((aa - a) ** 2)) for aa in ka_list]
    nearest = np.argsort(distances)
    for i in nearest[:k]:
        L.append(ka_list[i])
    if L:
        return L[0]
    else:
        return [0.0 for _ in range(env.n)]


def all_learn(agents, j):
    indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
    s_n = np.zeros((env.n, BATCH_SIZE, o_dim))
    a_n = np.zeros((env.n, BATCH_SIZE, a_dim))
    r_n = np.zeros((env.n, BATCH_SIZE, 1))
    s__n = np.zeros((env.n, BATCH_SIZE, o_dim))

    for p, agent in enumerate(agents):
        s, a, r, s_ = agent.get_exp(indices)
        s_n[p] = s.copy()
        a_n[p] = a.copy()
        r_n[p] = r.copy()
        s__n[p] = s_.copy()

    actor_a_ = np.array([agent.get_a_(s__n[p]) for p, agent in enumerate(agents)])

    # for i in range(env.n):
    #     for p in range(BATCH_SIZE):
    #         for q in range(env.n):
    #             actor_a_[i][p][q] = int(actor_a_[i][p][q]*a_bound)/a_bound

    for p, agent in enumerate(agents):
        act = a_n[p].copy()
        act_n = a_n.copy()
        act_n = np.delete(act_n, p, 0)

        act_ = actor_a_[p].copy()
        act_n_ = actor_a_.copy()
        act_n_ = np.delete(act_n_, p, 0)

        agent.learn_critic(s_n.swapaxes(1, 0), act_n.swapaxes(1, 0), act, r_n[p], s__n.swapaxes(1, 0),
                           act_n_.swapaxes(1, 0), act_)

    if j % 1 == 0:
        actor_a = [agent.get_a(s_n[p]) for p, agent in enumerate(agents)]

        # for i in range(env.n):
        #     for p in range(BATCH_SIZE):
        #         for q in range(env.n):
        #             actor_a[i][p][q] = int(actor_a[i][p][q] * a_bound) / a_bound

        for p, agent in enumerate(agents):
            act_n = np.delete(actor_a, p, 0)
            agent.learn_actor(s_n.swapaxes(1, 0), act_n.swapaxes(1, 0), s_n[p])


for choose in range(3, 5):
    agents = []
    for i in range(env.n):
        agents.append(DDPG(a_dim, o_dim, a_bound, i))
        agents[i].memory = np.zeros((MEMORY_CAPACITY, o_dim * 2 + a_dim + 1), dtype=np.float32)
        agents[i].pointer = 0
        # tf.summary.FileWriter("logs/", agents[i].sess.graph)

    var = 0  # control exploration TODO
    exp = 5
    exp_t = 0
    var_t = 0
    v_t = 0

    SIGMA = 0.5
    k = 1
    test = 0
    t1 = time.time()

    # for p, agent in enumerate(agents):
    #     agent.saver.restore(agent.sess, './model/all/%d/agent[%d]-n=5' % (choose, p))

    f = open("./%d/net_parameter.txt" % choose, "a")
    f.write("STEP = %d\nLR_A = %f\nLR_C = %f\nGAMMA = %.3f\nEXP_CAP = %d\nBATCH_SIZE = %d"
            % (MAX_EP_STEPS, LR_A, LR_C, GAMMA, MEMORY_CAPACITY, BATCH_SIZE))
    f.close()

    max_r = -np.inf
    for i in range(MAX_EPISODES):
        obs_n = env.reset(choose).copy()
        if i == 0:
            env.write_para(choose)
        arri = [0.0 for _ in range(env.n)]
        ep_reward = 0.0
        ep_energy = 0.0
        ep_queue = 0
        ep_drop = 0
        agent_reward = [0.0 for _ in range(env.n)]
        agent_energy = [0.0 for _ in range(env.n)]
        agent_queue = [0.0 for _ in range(env.n)]
        agent_drop = [0.0 for _ in range(env.n)]

        SIGMA *= 0.99
        orn = [OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.n), sigma=SIGMA) for _ in range(env.n)]
        ou_n = [np.zeros(env.n) for _ in range(env.n)]

        for j in range(MAX_EP_STEPS):
            if test == 0:
                ou_n = [ou() for ou in orn]
            action_n = [np.reshape(agent.choose_action(obs), env.n)+ou for agent, obs, ou in zip(agents, obs_n, ou_n)]
            action_n_real = [action_n[p].copy() for p in range(env.n)]
            for p in range(env.n):
                for q in range(env.n):
                    action_n[p][q] = int(action_n[p][q] * a_bound)
                    if action_n[p][q] >= a_bound:
                        action_n[p][q] = env.max_m - 1
                    if action_n[p][q] < 0:
                        action_n[p][q] = 0

                if not env.is_excu_a(p, action_n[p]):
                    a_list = env.find_excu_a(p)
                    action_n[p] = np.array(get_knn(k, action_n[p], a_list))

            new_obs_n, r_n, done, info, e_n, q_n, drop = env.step(action_n, exp)

            if test != 1:
                for p, agent in enumerate(agents):
                    agent.store_transition(obs_n[p], action_n_real[p], r_n[p], new_obs_n[p])

                if all(list(map(lambda tt: tt.pointer > MEMORY_CAPACITY, agents))):
                    all_learn(agents, j)

            for p, (r, e, q, d) in enumerate(zip(r_n, e_n, q_n, drop)):
                ep_reward += r
                ep_energy += e
                ep_queue += q
                ep_drop += d
                agent_reward[p] += r
                agent_energy[p] += e
                agent_queue[p] += q
                agent_drop[p] += d
                arri[p] += obs_n[p][2 * env.n] * 15

            obs_n = new_obs_n.copy()

            if j == MAX_EP_STEPS-1:
                if var < 5:
                    tt = 1 if exp == 0 else 0
                    f = open("./%d/episode-%.1f-%.2f.txt" % (choose, env.n, tt), "a")
                    f.write("%0.2f %0.2f %d %d %d\n" % (ep_reward, ep_energy, ep_queue, ep_drop, sum(arri)))
                    f.close()
                    for p in range(env.n):
                        f = open("./%d/agent-%d-%.2f.txt" % (choose, p, tt), "a")
                        f.write("%0.2f %0.2f %d %d %d\n"
                                % (agent_reward[p], agent_energy[p], agent_queue[p], agent_drop[p], arri[p]))
                        f.close()
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % SIGMA, 'test: ', test, ' arriv: ', arri)
                break

        if ep_reward > max_r and test == 1:
            max_r = ep_reward
            print("-------------------------------------\n-----------------------------------")
            for p, agent in enumerate(agents):
                agent.saver.save(agent.sess, './model/all/%d/agent[%d]-n=4' % (choose, p))

        if agents[0].pointer >= MEMORY_CAPACITY-1:
            if exp != 0:
                exp -= 0.05
                if exp < 2:
                    exp = 2
                exp_t = exp
                test = 1
            else:
                test = 0
            exp = abs(exp-exp_t)

        # if var != 0:
        #     var -= 0.02
        #     if var < 0.1:
        #         var = 0.1
        #     var_t = var
        #     test = 1
        # else:
        #     test = 0
        # var = abs(var-var_t)

    print('Running time: ', time.time() - t1)



