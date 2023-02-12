import numpy as np
import torch
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
                rho[i, t] = d[i]
            else:
                if i == 0:
                    rho[i, t] = rho[i, t - 1] + rho[-1, t - 1] * u[-1, t - 1] - rho[i, t - 1] * u[i, t - 1]
                else:
                    rho[i, t] = rho[i][t - 1] + rho[i - 1, t - 1] * u[i - 1, t - 1] - rho[i, t - 1] * u[i, t - 1]

    return rho


def network_loading(n_cell, T_terminal, model, u, beta, demand):
    T = n_cell * T_terminal
    rho = np.zeros((model.n_edge, n_cell, T))
    link_sequence = [0, 2, 1, 4, 3]
    for l in link_sequence:
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


def network_loading_from_rho_network(n_cell, T_terminal, model, u, beta, demand, rho_network, rho_optimizer):
    T = n_cell * T_terminal
    states, rho_values = list(), list()
    link_sequence = [0, 2, 1, 4, 3]
    for l in link_sequence:
        for t in range(T):
            for i in range(n_cell):
                states.append(np.array([l, i / n_cell, t / n_cell]))
                if t == 0:
                    rho_values.append(0)
                else:
                    if i >= 1:
                        rho_values.append(
                            float(rho_network(np.array([l, i / n_cell, (t - 1) / n_cell])) +
                                  rho_network(np.array([l, (i - 1) / n_cell, (t - 1) / n_cell])) * u[l, i - 1, t - 1] -
                                  rho_network(np.array([l, i / n_cell, (t - 1) / n_cell])) * u[l, i, t - 1])
                        )
                    else:
                        q_in = 0
                        start_node = model.edges[l, 0]
                        if start_node == model.origin:
                            q_in = demand[t - 1]
                        else:
                            for in_node in range(model.n_node):
                                k = model.adjacency[in_node, start_node]
                                if k > -1:
                                    q_in += float(
                                        rho_network(np.array([k, (n_cell - 1) / n_cell, (t - 1) / n_cell])) *
                                        u[k, n_cell - 1, t - 1]
                                    )

                        rho_values.append(
                            float(rho_network(np.array([l, 0, (t - 1) / n_cell]))) +
                            beta[l, t - 1] * q_in -
                            float(rho_network(np.array([l, 0, (t - 1) / n_cell])) * u[l, 0, t - 1])
                        )

    rho_values = torch.tensor(rho_values)
    for _ in range(2000):
        preds = torch.reshape(rho_network(np.array(states)), (1, len(rho_values)))
        rho_loss = (rho_values - preds).abs().mean()
        rho_optimizer.zero_grad()
        rho_loss.backward()
        rho_optimizer.step()

    return rho_network


def get_rho_from_net(n_cell, T_terminal, model, rho_network):
    T = n_cell * T_terminal
    rho = np.zeros((model.n_edge, n_cell, T))
    for l in range(model.n_edge):
        for i in range(n_cell):
            for t in range(T):
                rho[l, i, t] = rho_network(np.array([l, i / n_cell, t / n_cell]))

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
