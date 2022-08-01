import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def get_rho_from_u(u, d):
    n_cell = u.shape[0]
    T_terminal = int(u.shape[1] / u.shape[0])
    rho = np.zeros((n_cell, n_cell * T_terminal), dtype=np.float64)
    for t in range(n_cell * T_terminal):
        for i in range(n_cell):
            if t == 0:
                continue
            else:
                if i == 0:
                    rho[i, t] = rho[i, t - 1] + d[t] - rho[i, t - 1] * u[i, t - 1]
                else:
                    rho[i, t] = rho[i][t - 1] + rho[i - 1, t - 1] * u[i - 1, t - 1] - rho[i, t - 1] * u[i, t - 1]

    return rho

def plot_rho(n_cell, T_terminal, rho, fig_name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.linspace(0, 1, n_cell)
    t = np.linspace(0, T_terminal, n_cell * T_terminal)
    t_mesh, x_mesh = np.meshgrid(t, x)
    surf = ax.plot_surface(x_mesh, t_mesh, rho, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.xlim(max(x), min(x))
    if not fig_name:
        plt.show()
    else:
        plt.savefig(fig_name)