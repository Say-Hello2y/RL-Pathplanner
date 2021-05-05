import numpy as np
from gridworld import RLworld
import matplotlib.pyplot as plt


def get_path_length(path):
    length = 0
    for i in range(len(path) - 1):
        if abs(path[i] - path[i + 1]) == 1:
            length += 1
        elif abs(path[i] - path[i + 1]) == 40:
            length += 1
        else:
            length += 1.4
    return length


# ################ env settings #######################
'''
智能体在环境中每移动一步奖励为-1，注斜向移动奖励为-1.4
到达终点奖励为100

'''
env = RLworld()
env.start = (0, 0)
env.ends = [(35, 15)]
env.refresh_setting()
# #################learning####################
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = .6
gamma = .98
epsilon = .01
num_episodes = 3000
# create lists to contain total rewards and steps per episode
# jList = []
rList = []
path_len = []
# ###################training######################
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    # length = 0
    path = []
    done = False
    j = 0
    # The Q-Table learning algorithm
    while j < 1000:
        j += 1
        # Choose an action by greedily (with noise) picking from Q table
        path.append(s)
        if np.random.random() < epsilon:
            a = np.random.randint(0, env.action_space.n,  dtype='l')
        else:
            a = np.argmax(Q[s, :])
        # a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # Get new state and reward from environment
        s1, r, done, _ = env.step(a)
        # Update Q-Table with new knowledge
        if np.random.random() < epsilon:
            a_ = np.random.randint(0, env.action_space.n,  dtype='l')
        else:
            a_ = np.argmax(Q[s, :])
        Q[s, a] = Q[s, a] + lr * (r + gamma * Q[s1, a_] - Q[s, a])
        rAll += r
        s = s1
        if done == True:
            break
    # jList.append(j)
    rList.append(rAll)
    path.append(env._xy_to_state((35, 15)))
    path_len.append(get_path_length(path))


# #######################data process#############
print("Average Reward : " + str(sum(rList) / num_episodes))
print('final Reward is :', rAll)
# print("Final Q-Table Values")
# print(Q)
print("After Epoch {},path length is : {}".format(num_episodes, get_path_length(path)))
# print('after training path length is :', get_path_length(path))
# ######################routh map #################
env.get_path(path)
env.render()
# ############# reward  ##########################
plt.title('reward Curve')
plt.xlabel('Epoch')
plt.ylabel('reward')
plt.plot(np.arange(num_episodes), rList, label='$Reward$')
plt.legend()


# ########## path ####################################
plt.title('Path_len Curve')
plt.xlabel('Epoch')
plt.ylabel('length')
plt.plot(np.arange(num_episodes), path_len, label='$Length$')
plt.legend()
plt.savefig('sarsa_path_length.pdf', dpi=600, format='pdf')
