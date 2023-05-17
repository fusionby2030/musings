import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create cylinder
theta = np.linspace(0, 2*np.pi, 100)
z = np.linspace(0, 1, 100)
Z, Theta = np.meshgrid(z, theta)
X = np.cos(Theta)
Y = np.sin(Theta)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5)

# draw circles
for i in [0.2, 0.4, 0.6]:
    ax.plot(np.cos(theta), np.sin(theta), zs=[i]*len(theta), zdir='z', color='black')

# set axis labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1)
plt.show()
