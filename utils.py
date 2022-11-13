import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import csv
import torch
from torch import nn


class RhoNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.model(torch.from_numpy(x).float())
        x = torch.tanh(x)
        x = (x + 1) / 2
        return x


def get_rho_from_u(u, d):
    n_cell = u.shape[0]
    T_terminal = int(u.shape[1] / u.shape[0])
    rho = np.zeros((n_cell, n_cell * T_terminal), dtype=np.float64)
    for t in range(n_cell * T_terminal):
        for i in range(n_cell):
            if t == 0:
                rho[i, t] = d[i]
            else:
                if i == 0:
                    rho[i, t] = rho[i, t - 1] + rho[-1, t - 1] * u[-1, t - 1] - rho[i, t - 1] * u[i, t - 1]
                else:
                    rho[i, t] = rho[i][t - 1] + rho[i - 1, t - 1] * u[i - 1, t - 1] - rho[i, t - 1] * u[i, t - 1]

    return rho


def network_loading(model, u, beta, demand, n_cell, T):
    rho = np.zeros((model.n_edge, n_cell, T))
    for l in range(model.n_edge):
        for t in range(1, T):
            for i in range(n_cell):
                if t == 0:
                    rho[l, i, t] == 0
                else:
                    if i >= 1:
                        rho[l, i, t] = rho[l, i, t - 1] + rho[l, i - 1, t - 1] * u[l, i - 1, t - 1] - rho[l, i, t - 1] * u[l, i, t - 1]
                    else:
                        q_in = 0
                        start_node = model.edges[l, 0]
                        if start_node == model.origin:
                            q_in = demand[t - 1]
                        else:
                            for in_node in range(model.n_node):
                                k = model.adjacency[in_node, start_node]
                                if k > -1:
                                    q_in += rho[k, n_cell - 1, t - 1] * u[k, n_cell - 1, t - 1]

                        rho[l, 0, t] = rho[l, 0, t - 1] + beta[l, t - 1] * q_in - rho[l, 0, t - 1] * u[l, 0, t - 1]

    return rho


def plot_3d(n_cell, T_terminal, rho, fig_name):
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


def plot_4d(n_cell, T_terminal, rho, concat, fig_name):
    rho_new = rho[concat[0], :, :]
    for i in range(1, len(concat)):
        rho_new = np.append(rho_new, rho[concat[i], :, :], axis=0)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.linspace(0, len(concat), len(concat) * n_cell)
    t = np.linspace(0, T_terminal, n_cell * T_terminal)
    t_mesh, x_mesh = np.meshgrid(t, x)
    surf = ax.plot_surface(x_mesh, t_mesh, rho_new, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.xlim(max(x), min(x))
    if not fig_name:
        plt.show()
    else:
        plt.savefig(fig_name)


def array2csv(n_cell, T_terminal, array, file_name):
    res = np.append([np.array(range(len(array))) / n_cell], array, axis=0)
    column = np.append([np.array([0])], np.reshape(np.arange(len(array[0])) / n_cell, (len(array[0]), 1)), axis=0)
    res = np.append(column, res, axis=1)
    with open(file_name, "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(res)
