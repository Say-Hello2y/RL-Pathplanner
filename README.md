# reforcementLearningPathplanner
An implementation of qlearning and sarsa path planning algorithm 

requirements(需要安装的python库)

python gym

numpy

matplotlib

python版本3.8（python3.6以上都能正常运行）

将强化学习中常见的Q学习和SARSA学习做了路径规划方面的尝试，主要借助了gym的库，建立了一个gym风格的二维栅格地图用于路径规划。
The common Q learning and SARSA learning in reinforcement learning are tried in path planning. The gym library is mainly used to establish a gym-style two-dimensional grid map for path planning.
其中gridword.py文件创建了一个基本的栅格地图，环境给智能体的奖励也由其给出，主要参考了 https://github.com/qqiang00/Reinforce/blob/master/reinforce/gridworld.py的工作，并加了一点改进。
qlearning.py文件与sarsa.py文件 则给出了具体的算法以及可视化结果。
