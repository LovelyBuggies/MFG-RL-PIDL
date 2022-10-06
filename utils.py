import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import csv
import torch

def calculate_optimal_costs(u, V):
    n_cell = len(u)
    T_terminal = int(len(u[0]) / n_cell)
    curr_i, curr_t = 0, 0
    costs = V[0, 0]
    while curr_i < n_cell - 1:
        if curr_t > T_terminal:
            return float('inf')

        curr_i += int(u[curr_i, curr_t])
        curr_t += 1
        costs += V[curr_i, curr_t]

    return costs



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


def get_rho_network_from_u(n_cell, T_terminal, u, d, rho_network, rho_optimizer, n_iterations=100):
    states, rho_values = list(), list()
    T = n_cell * T_terminal
    for t in range(T):
        for i in range(n_cell):
            states.append([i / n_cell, t / n_cell])
            if t == 0:
                rho_values.append(d[i])
            else:
                if i == 0:
                    rho_values.append(float(rho_network(np.array([i, t - 1]) / n_cell) + rho_network(np.array([-1, t - 1]) / n_cell) * u[-1, t - 1] - rho_network(np.array([i, t - 1]) / n_cell) * u[i, t - 1]))
                else:
                    rho_values.append(float(rho_network(np.array([i, t - 1]) / n_cell) + rho_network(np.array([i - 1, t - 1]) / n_cell) * u[i - 1, t - 1] - rho_network(np.array([i, t - 1]) / n_cell) * u[i, t - 1]))

    rho_values = torch.tensor(np.array(rho_values))
    for _ in range(n_iterations):
        preds = torch.reshape(rho_network(np.array(states)), (1, len(rho_values)))
        rho_loss = (rho_values - preds).abs().mean()
        rho_optimizer.zero_grad()
        rho_loss.backward()
        rho_optimizer.step()

    return rho_network



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
    # ax.tick_params(axis='both', which='major', labelsize=15)
    # plt.grid(alpha=0.3)
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
