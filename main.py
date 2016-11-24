from MonteCarlo_Learning import MonteCarlo
from TD_Learning import TDLearning

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time

start_time = time.time()

# Parameters
episodes = 100000
N0 = 100
landa = 0.9
gamma = 1

# TODO epsilon doesn't decrease with time?
Smart_Agent_1 = MonteCarlo(N0=N0, gamma=gamma)
# Smart_Agent_2 = TDLearning(landa=0.9, gamma=gamma, N0=N0)
value, state_decision = Smart_Agent_1.learn2(episodes=episodes)
# value, state_decision = Smart_Agent_2.learn_sarsa(episodes=episodes)
# value, state_decision = Smart_Agent_2.learn_sarsa_landa(episodes=episodes, landa=1)

epsilon_list = Smart_Agent_1.epsilon_list

plt.plot(np.arange(np.size(epsilon_list)), epsilon_list)
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

x = range(np.shape(value)[1])
y = range(np.shape(value)[0])

xs, ys = np.meshgrid(x, y)
zs = value
zs2 = state_decision

fig1 = plt.figure()
ax1 = Axes3D(fig1)
ax1.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='hot')
plt.show()


fig2 = plt.figure()
ax2 = Axes3D(fig2)
ax2.plot_surface(xs, ys, zs2, rstride=1, cstride=1, cmap='hot')
plt.show()

input()
