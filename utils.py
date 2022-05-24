import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def get_rho_from_u(u, d):
    cell = u.shape[0]
    t_num = int(u.shape[1] / u.shape[0])
    rho = np.zeros((cell, cell * t_num), dtype=np.float64)
    for t in range(cell * t_num):
        for i in range(cell):
            if t == 0:
                continue
            else:
                if i == 0:
                    rho[i][t] = rho[i][t - 1] + d[t] - rho[i][t - 1] * u[i, t - 1]
                else:
                    rho[i][t] = rho[i][t - 1] + rho[i - 1][t - 1] * u[i - 1][t - 1] - rho[i][t - 1] * u[i][t - 1]

    return rho

def plot_rho(cell, t_num, rho, fig_name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.linspace(0, 1, cell)
    t = np.linspace(0, t_num, cell * t_num)
    t_mesh, x_mesh = np.meshgrid(t, x)

    surf = ax.plot_surface(x_mesh, t_mesh, rho, cmap=cm.jet,
                           linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.savefig(f"./fig/{fig_name}.png")

def update_density(density, action):
    cell = density.shape[0]
    granularity = density.shape[1]
    for x in range(cell):
        for t in range(1, granularity):
            density[x][t] = density[x][t - 1] + density[x - 1][t - 1] * action[x - 1][t - 1] - density[x][t - 1] * action[x][t - 1]

    return density