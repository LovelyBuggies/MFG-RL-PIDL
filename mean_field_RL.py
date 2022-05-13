import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from value_iteration import value_iteration, value_iteration_non_separable
from utils import get_rho_from_u

cell = 32
t_num = 4
episode = 300
tolerance = 1

u = np.zeros((cell, cell * t_num), dtype=np.float64)
u_hist = [u]

d = np.zeros((cell * t_num, 1), dtype=np.float64)
d[1] = 1

u_tmp = u
rho = get_rho_from_u(u, d)
for _ in range(episode):
    u, v = value_iteration_non_separable(rho)
    u_hist.append(u)
    if sum(sum(np.power(u - u_tmp, 2))) < tolerance:
        break

    u_tmp = u
    rho = get_rho_from_u(u, d)

fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.linspace(0, 1, cell)
t = np.linspace(0, t_num, cell * t_num)
t_mesh, x_mesh = np.meshgrid(t, x)

surf = ax.plot_surface(x_mesh, t_mesh, rho, cmap=cm.jet,
                       linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()
