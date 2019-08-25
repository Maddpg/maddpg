import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from os import path
import scipy.stats as st


mode = 0   # 0为训练，1为测试


class Mec_co_1(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.test = np.zeros(10000)

        self.n = 5  # 基站数（实际考虑action时要考虑md所以是n+1）
        self.max_m = 7   # 最大任务数
        self.lamda = 10  # slot的平均任务到达
        self.net_speed_min = 3
        self.net_speed_max = 7   # 最大传输速率 TODO 想要缩小维度

        self.NL, self.NH = 4, 0
        self.FL, self.FH = 2, 2.5
        self.aL = self.aH = 0.15625

        self.W = 2  # 若改变，注意后续取整问题
        self.D = 1.0
        self.C_delta = 1

        # self.C = np.random.randint(2, 5, self.n)      # 动态变换？
        self.C = np.ones(self.n) * 10
        self.C[0] = 8 * self.C_delta
        self.C[1] = 10 * self.C_delta
        self.C[2] = 12 * self.C_delta
        self.C[3] = 9 * self.C_delta
        self.C[4] = 11 * self.C_delta
        self.C = self.C // self.W

        self.P_TX = 0.25

        self.net = np.random.randint(self.net_speed_min, self.net_speed_max+1, self.n)

        self.C_max = self.NL * self.FL + self.NH * self.FH
        self.C_max = self.C_max / self.W
        self.P_max = self.aL * self.NL * pow(self.FL, 3) + self.aH * self.NH * pow(self.FH, 3)
        self.N_max = min(self.C_max, (self.max_m - 1))
        self.E_max = self.P_max * (self.N_max * self.W / self.C_max) + self.P_TX * 1

        self.t_step = 0
        self.ls = np.zeros((5002, self.n+1))
        self.task = np.zeros(self.n+1)

        self.alpha = 1
        self.beta = 1
        self.eta = 0.2
        self.ddl = 2

        self.V = 0.1

        self.viewer = None
        self.dt = .05

        self.state = np.zeros((1, 2 * self.n + 2))
        self.de_q = 0

        low = np.zeros(2*self.n+2)
        high = np.ones(2*self.n+2)*self.max_m  # TODO 想一下0和上限问题

        self.action_space = spaces.Box(low=np.array([0]), high=np.array([np.power(self.max_m, self.n+1)]), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def find_k_excu_a(self, k, a):
        action = turn_to_action(a, self.max_m, self.n)
        list_a = []
        sq_num = [0, 1, 4, 9, 16, 25, 36]
        for x in range(1, self.max_m ** 2 * (self.n+1)):
            list_dis = squ(x, sq_num, self.n+1, [], [])
            for dis in list_dis:
                dis_root = list(map(lambda tt: tt ** 0.5, dis))

                for t1 in [-1, 1]:
                    for t2 in [-1, 1]:
                        for t3 in [-1, 1]:
                            for t4 in [-1, 1]:
                                for t5 in [-1, 1]:
                                    for t6 in [-1, 1]:
                                        act = []
                                        tag = 0
                                        temp = [t1, t2, t3, t4, t5, t6]
                                        for i in range(self.n+1):
                                            act.append(action[i] + dis_root[i] * temp[i])
                                            if act[i] < 0 or act[i] > self.max_m-1:
                                                tag = 1
                                                break
                                        if tag == 0 and self.is_excu_a(act):
                                            list_a.append(turn_to_index(act, self.max_m, self.n))
                                            if len(list_a) == k:
                                                return list_a
        return list_a

    def is_excu_a(self, a):
        limit = self.state[2 * self.n + 1]
        rest = sum(a) - limit
        if (rest - a[0] <= 0) and (rest <= self.state[0]):
            return True
        return False

    def find_excu_a(self, s):
        list = []
        limit = self.state[2 * self.n + 1]
        for i in range(int(self.action_space.high)):
            action = turn_to_action(i, self.max_m, self.n)
            rest = sum(action)-limit
            if (rest - action[0] <= 0) and (rest <= self.state[0]):
                list.append(i)
        if list:
            return list
        else:
            return [0]

    def choose_a(self, x):
        limit = self.state[2*self.n + 1]
        for i in range(10000):
            a = np.array(np.argmax(x))
            action = turn_to_action(a, self.max_m, self.n)
            Q = 0
            for j in range(1, self.n + 1):
                Q = Q + action[j]

            if (Q > limit) or (action[0] > self.state[0] + limit - Q):
                x[a] = 0
            else:
                return a
        return 0      # TODO 若多次无法找到合适的action，则说明这个action分布和State不匹配，使用0来增大惩罚。但是若0的惩罚不够大，则会出现训练不良的情况。

    def step(self, u, var):

        action = turn_to_action(u, self.max_m, self.n)
        C_0, P_0 = choose_epsilon(action[0], self.C_max, self.P_max)
        # 需要重新检查理清action

        if action[0] > C_0:
            action[0] = C_0//1  # TODO

        for i in range(1, self.n+1):
            if action[i] > self.state[i+self.n]//self.D:
                action[i] = self.state[i+self.n]//self.D

        for i in range(1, self.n+1):
            self.ls[self.t_step][i] = action[i]
        self.ls[self.t_step][0] = self.state[2*self.n+1] - sum(action) + action[0]
        self.t_step += 1

        # reward的计算
        E_0 = 0
        if C_0 != 0:
            E_0 = action[0] / (C_0+0.0) * P_0     # 要确认非整除   TODO 应该是要执行的任务/每个任务消耗的时间
        T = 0
        for i in range(1, self.n+1):
            T += action[i] / (self.state[i+self.n]/self.D + 0.0)
        E_off = T * self.P_TX
        E = E_0 + E_off

        # 记录本次执行的数量。
        exe = np.zeros(self.n+1)

        # state的更新
        L = self.state[2 * self.n + 1]
        x = sum(action)
        self.state[0] = self.state[0] - x + L  # TODO
        exe[0] = action[0]

        for i in range(1, self.n+1):
            x = min(self.state[i], self.C[i-1])
            self.state[i] = self.state[i] - x + action[i]
            exe[i] = x

        # 载入文件 TODO
        if mode == 1:
            temp = self.test[self.t_step].split()
            for k in range(self.n):
                self.net[k] = int(temp[k])
            t = int(temp[self.n])
        else:
            t = np.random.poisson(self.lamda)
            self.net = np.random.randint(low=self.net_speed_min, high=self.net_speed_max+1, size=self.n)

        for i in range(self.n):
            self.state[i + self.n + 1] = self.net[i]
            self.state[2 * self.n + 1] = t

        # 时延的计算
        Q = 0
        for i in range(self.n + 1):
            Q = Q + self.state[i]

        # print(Q, '  ', exe, '  ', sum(self.ls), '  ', self.state[2 * self.n + 1])

        # 计算任务完成情况并计算收益
        ut = 0
        for i in range(self.n+1):
            while exe[i] != 0:
                t_temp = self.ls[int(self.task[i])][i]
                if exe[i] >= t_temp:
                    exe[i] -= t_temp
                    ut += calcu_ut(self.task[i], t_temp, self.t_step, self.alpha, self.eta)
                    self.ls[int(self.task[i])][i] = 0
                    self.task[i] += 1
                    # print(exe, '  ', self.task, '  ', t_temp, '  ', self.t_step, '  ', Q)
                else:
                    ut += calcu_ut(self.task[i], exe[i], self.t_step, self.alpha, self.eta)
                    self.ls[int(self.task[i])][i] -= exe[i]
                    exe[i] = 0

        drop = 0
        for i in range(self.n+1):
            while self.task[i] + self.ddl <= self.t_step:
                t_temp = self.ls[int(self.task[i])][i]
                self.ls[int(self.task[i])][i] = 0
                self.state[i] -= t_temp
                drop += t_temp
                self.task[i] += 1

        reward = ut - self.beta * E - self.eta * drop
        if var == 0:
            f = open("DDPG-RA-%0.1f.txt" % self.D, "a")
            if self.n == 1:
                f.write("%0.2f  %0.2f  %0.2f  %d  [%d %d %d %d]\n" %
                        (E, drop, reward, u, self.state[0], self.state[1], self.state[2], self.state[3]))
            if self.n == 2:
                f.write("%0.2f  %0.2f  %0.2f  %d  [%d %d %d %d %d %d]\n" %
                        (E, drop, reward, u, self.state[0], self.state[1], self.state[2], self.state[3], self.state[4],
                         self.state[5]))
            if self.n == 3:
                f.write("%0.2f  %0.2f  %0.2f  %d  [%d %d %d %d %d %d %d %d]\n" %
                        (E, drop, reward, u, self.state[0], self.state[1], self.state[2], self.state[3], self.state[4],
                         self.state[5], self.state[6], self.state[7]))
            if self.n == 4:
                f.write("%0.2f  %0.2f  %0.2f  %d  [%d %d %d %d %d %d %d %d %d %d]\n" %
                        (E, drop, reward, u, self.state[0], self.state[1], self.state[2], self.state[3],
                         self.state[4], self.state[5], self.state[6], self.state[7], self.state[8],
                         self.state[9]))
            if self.n == 5:
                f.write("%0.2f  %0.2f  %0.2f  %d  [%d %d %d %d %d %d %d %d %d %d %d %d]\n" %
                        (E, drop, reward, u, self.state[0], self.state[1], self.state[2], self.state[3],
                         self.state[4], self.state[5], self.state[6], self.state[7], self.state[8],
                         self.state[9], self.state[10], self.state[11]))
            f.close()

        return self.state, reward, False, {}, E, drop

    def reset(self, d):
        # for i in range(1, 6):
        #     C_0, P_0 = choose_epsilon(i, self.C_max, self.P_max)
        #     print("%d   %0.3f   %0.3f" % (i, C_0, i / (C_0 + 0.0) * P_0))
        # 载入文件 TODO

        self.D = d
        if mode == 1:
            with open("test-%d.txt" % self.lamda, "r") as f:
                self.test = f.readlines()
                temp = self.test[0].split()
                for x in range(self.n):
                    self.net[x] = int(temp[x])
                t = int(temp[self.n])
        else:
            t = np.random.poisson(self. lamda)
            self.net = np.random.randint(low=self.net_speed_min, high=self.net_speed_max + 1, size=self.n)

        self.state = np.zeros(2*self.n+2)

        for i in range(self.n):
            self.state[i+self.n+1] = self.net[i]
        self.state[2*self.n+1] = t
        self.ls = np.zeros((5002, self.n+1))
        self.task = np.zeros(self.n+1)
        self.t_step = 0
        return self.state

    # 不需要绘图，暂无
    def render(self, mode='human'):
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# action的编码转换（数-->数组），方便使用
def turn_to_action(a, m, n):
    action = np.zeros(n+1)
    for i in range(n+1):
        action[i] = a % m
        a = a // m
    return action


def turn_to_index(a, m, n):
    d = 0
    for i in range(n, -1, -1):
        d *= m
        d += a[i]
    return int(d)


def choose_epsilon(u, c_max, p_max):
    if u/c_max > 0.75:
        epsilon = 1.0
    else:
        if u/c_max <= 1:
            epsilon = 0.5
        else:
            epsilon = 0.75
    P = pow(epsilon, 3) * p_max
    C = c_max * epsilon
    return C, P


def squ(x, sq, n, l, all_d):
    if x >= 0 and n > 0:
        for i in sq:
            l.append(i)
            temp = x-i
            if temp < 0:
                l.pop(-1)
                return all_d
            all_d = squ(temp, sq, n-1, l, all_d)
            l.pop(-1)
    else:
        if x == 0 and n == 0:
            all_d.append(l.copy())
            return all_d
    return all_d


def calcu_ut(task, a, step, alpha, eta):
    # if task + 2 >= step:
    return alpha * a
    # else:
    #     return alpha * pow(np.e, -eta * (step - task - 2)) * a
