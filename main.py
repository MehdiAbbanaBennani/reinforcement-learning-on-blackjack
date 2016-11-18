from Monte_Carlo_Control import MonteCarlo

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Smart_Agent_1 = MonteCarlo()
value = Smart_Agent_1.learn(episodes=100)
print(value)

x = range(np.shape(value)[1])
y = range(np.shape(value)[0])

xs, ys = np.meshgrid(x, y)
# z = calculate_R(xs, ys)
zs = value

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='hot')
plt.show()