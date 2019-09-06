import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import math
from os import path
import scipy.stats as st


mode = 0   # 0为训练，1为测试


class Mec_co_1(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.test = np.zeros(10000)

        self.n = 5  # 基站数（实际考虑action时要考虑md所以是n+1）
        self.max_m = 7   # 最大任务数
        self.lamda = [1, 2, 3, 4, 5]  # slot的平均任务到达
        self.net_speed_min = 3
        self.net_speed_max = 7   # 最大传输速率 TODO 想要缩小维度

        self.NL = [4, 4, 4, 6, 4]
        self.NH = [4, 4, 4, 2, 2]
        self.FL = [1.8, 1.7, 1.7, 1.7, 1.6]
        self.FH = [2.6, 2.9, 2.9, 2, 2.5]
        self.aL = 0.1125
        self.aH = 0.1125

        self.W = 1  # 若改变，注意后续取整问题
        self.D = 1.0
        self.C_delta = 1

        self.P_TX = 0.25
        self.P_RX = 0.1

        self.net = np.zeros((self.n, self.n))

        self.C_max = np.zeros(self.n)
        self.P_max = np.zeros(self.n)

        for i in range(self.n):
            self.C_max[i] = (self.NL[i] * self.FL[i] + self.NH[i] * self.FH[i]) / self.W
            self.P_max[i] = self.aL * self.NL[i] * self.FL[i] ** 3 + self.aH * self.NH[i] * self.FH[i] ** 3

        self.t_step = 0
        self.ls = np.zeros((5002, self.n + 1))
        self.task = np.zeros(self.n+1)

        self.alpha = np.array([1, 1, 0.3, 0.3, 1]) * 50
        self.beta = np.array([0.3, 0.3, 1, 0.3, 0.3])
        self.eta = 20
        self.ddl = 2

        self.V = 0.1

        self.viewer = None
        self.dt = .05

        self.state = np.zeros((self.n, 2*self.n+1))
        self.de_q = 0

        low = np.zeros(2*self.n+1)
        high = np.ones(2*self.n+1)*self.max_m  # TODO 想一下0和上限问题

        self.action_space = spaces.Box(low=np.zeros(self.n), high=np.ones(self.n)*self.max_m, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed()
        f = open("env_parameter.txt", "a")
        f.write("n = %d\nW = %f\nD = %f\n" % (self.n, self.W, self.D))
        f.write("alpha         beta         eta         lambda\n")
        for i in range(self.n):
            f.write("%.2f          %.2f         %.2f         %d\n"
                    % (self.alpha[i], self.beta[i], self.eta, self.lamda[i]))
        f.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def is_excu_a(self, p, a):
        limit = self.state[p][2 * self.n]
        rest = sum(a) - limit
        if (rest - a[p] <= 0) and (rest <= self.state[p][p]):
            return True
        return False

    def find_excu_a(self, p):
        list = []
        limit = self.state[p][2 * self.n]
        for i in range(self.max_m ** self.n):
            action = turn_to_action(i, self.max_m, self.n)
            rest = sum(action)-limit
            if (rest - action[p] <= 0) and (rest <= self.state[p][p]):
                list.append(action)
        if list:
            return list
        else:
            return [0]

    def step(self, action, var):

        C_0, P_0 = choose_epsilon(action, self.n, self.C_max, self.P_max)

        # 需要重新检查理清action
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    if action[i][i] > C_0[i]:
                        action[i][i] = C_0[i] // 1
                else:
                    if action[i][j] > self.state[i][j+self.n]//self.D:
                        action[i][j] = self.state[i][j+self.n]//self.D  # TODO

        # reward的计算
        E = np.zeros(self.n)
        for i in range(self.n):
            E_0 = 0
            if C_0[i] != 0:
                E_0 = action[i][i] / (C_0[i]+0.0) * P_0[i]     # 要确认非整除   TODO 应该是要执行的任务/每个任务消耗的时间
            t_T = 0
            t_R = 0
            for j in range(self.n):
                if j != i:
                    t_T += action[i][j] / (self.state[i][j+self.n]/self.D + 0.0)
                    t_R += action[j][i] / (self.state[i][j+self.n]/self.D + 0.0)
            E[i] = E_0 + t_T * self.P_TX + t_R * self.P_RX

        # state的更新
        # 执行
        for i in range(self.n):
            L = self.state[i][2 * self.n]
            x = sum(action[i])
            self.state[i][i] = self.state[i][i] - x + L  # TODO
            # exe[0] = action[0]

        # 卸载        TODO 直接通过矩阵运算可能会节省很多时间
        s_Q = np.zeros(self.n)
        drop = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.state[i][i] += action[j][i]
            s_Q[i] = self.state[i][i]
            if s_Q[i] > 10:
                drop[i] += s_Q[i] - 10
                s_Q[i] = 10
        for i in range(self.n):
            self.state[i][:self.n] = s_Q.copy()

        # 载入文件 TODO
        if mode == 1:
            temp = self.test[self.t_step].split()
            for k in range(self.n):
                self.net[k] = int(temp[k])
            arr_t = int(temp[self.n])
        else:
            arr_t = np.random.poisson(self.lamda, self.n)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    self.net[i][j] = random.randint(self.net_speed_min, self.net_speed_max + 1)
                    self.net[j][i] = self.net[i][j]

        for i in range(self.n):
            self.state[i][self.n:2*self.n] = self.net[i]
            self.state[i][2*self.n] = arr_t[i]

        Q = s_Q.copy()

        reward = - self.alpha * E - self.beta * Q - self.eta * drop

        if var == 0.5:
            for p in range(self.n):
                f = open("DDPG-RA-%d.txt" % p, "a")
                f.write("%0.2f  %0.2f  %0.2f  " % (E[p], Q[p], reward[p]) + str(action[p])+" "+str(self.state[p])+"\n")
                f.close()

        return self.state, reward, False, {}, E*100, Q, drop

    def reset(self):
        # 载入文件
        self.net = np.zeros((self.n, self.n))
        if mode == 1:
            with open("test-%d.txt" % self.lamda, "r") as f:
                self.test = f.readlines()
                temp = self.test[0].split()
                for x in range(self.n):
                    self.net[x] = int(temp[x])
                arr_t = int(temp[self.n])
        else:
            arr_t = np.random.poisson(self.lamda, self.n)
            for i in range(self.n):
                for j in range(i+1, self.n):
                    self.net[i][j] = random.randint(self.net_speed_min, self.net_speed_max+1)
                    self.net[j][i] = self.net[i][j]

        self.state = np.zeros((self.n, 2*self.n+1))

        for i in range(self.n):
            self.state[i][self.n:2*self.n] = self.net[i]
            self.state[i][2*self.n] = arr_t[i]
        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def turn_to_action(a, m, n):
    action = np.zeros(n)
    for i in range(n):
        action[i] = a % m
        a = a // m
    return action


def choose_epsilon(u, n, c_max, p_max):
    P = np.zeros(n)
    C = np.zeros(n)
    for i in range(n):
        epsilon = math.ceil(u[i][i]*10/c_max[i])/10.0
        if epsilon > 1.0:
            epsilon = 1.0
        P[i] = pow(epsilon, 3) * p_max[i]
        C[i] = c_max[i] * epsilon
    return C, P

