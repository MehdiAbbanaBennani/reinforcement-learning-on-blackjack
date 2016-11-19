from MonteCarlo_Learning import MonteCarlo
from TD_Learning import TDLearning

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time

start_time = time.time()

# Parameters
episodes = 10000
N0 = 100
landa = 0.5
gamma = 1

Smart_Agent_1 = MonteCarlo(N0=N0, gamma=gamma)
Smart_Agent_2 = TDLearning(landa=0.5, gamma=gamma, N0=N0)
# value, value_action = Smart_Agent_1.learn2(episodes=episodes)
value, value_action = Smart_Agent_2.learn_sarsa(episodes=episodes)

print("--- %s seconds ---" % (time.time() - start_time))

x = range(np.shape(value)[1])
y = range(np.shape(value)[0])

xs, ys = np.meshgrid(x, y)
zs = value

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='hot')
plt.show()