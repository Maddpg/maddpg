import gym
import numpy as np

ENV_NAME = 'CO-v0'
MAX_EPISODES = 10
MAX_EP_STEPS = 1000

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
m_max = env.max_m

D_list = [1]
for choose in D_list:

    v_t = 0

    SIGMA = 0.3
    k = 1
    test = 0

    max_r = -np.inf
    for i in range(MAX_EPISODES):
        obs_n = env.reset(choose).copy()

        arri = [0.0 for _ in range(env.n)]
        ep_reward = 0.0
        ep_energy = 0.0
        ep_queue = 0
        ep_drop = 0
        agent_reward = [0.0 for _ in range(env.n)]
        agent_energy = [0.0 for _ in range(env.n)]
        agent_queue = [0.0 for _ in range(env.n)]
        agent_drop = [0.0 for _ in range(env.n)]

        for j in range(MAX_EP_STEPS):
            action_n = np.random.rand(env.n, env.n) * a_bound
            for p in range(env.n):
                count = 0
                while not env.is_excu_a(p, action_n[p]):
                    action_n[p] = np.random.rand(env.n) * a_bound
                    count += 1
                    if count == 100:
                        action_n[p] = np.zeros(env.n)

            new_obs_n, r_n, done, info, e_n, q_n, drop = env.step(action_n, 1)

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

            if j == MAX_EP_STEPS - 1:
                tt = 2.0
                f = open("./test/random/episode-%.1f-%.2f.txt" % (env.n, tt), "a")
                f.write("%0.2f %0.2f %d %d %d\n" % (ep_reward, ep_energy, ep_queue, ep_drop, sum(arri)))
                f.close()
                for p in range(env.n):
                    f = open("./test/random/agent-%d-%.2f.txt" % (p, tt), "a")
                    f.write("%0.2f %0.2f %d %d %d\n"
                            % (agent_reward[p], agent_energy[p], agent_queue[p], agent_drop[p], arri[p]))
                    f.close()
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % SIGMA, 'test: ', test,
                      ' arriv: ', arri)
                break

