import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
import torch
import os
from model import Critic


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
                    rho_values.append(float(rho_network(np.array([i, t - 1]) / n_cell) + rho_network(np.array([n_cell - 1, t - 1]) / n_cell) * u[-1, t - 1] - rho_network(np.array([i, t - 1]) / n_cell) * u[i, t - 1]))
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

def get_rho_network_from_u_lwr(n_cell, T_terminal, u, d, rho_network, rho_optimizer, n_iterations=100):
    states, rho_values = list(), list()
    T = n_cell * T_terminal
    for t in range(T):
        for i in range(n_cell):
            states.append([i / n_cell, t / n_cell])
            if t == 0:
                rho_values.append(d[i])
            else:
                if i == 0:
                    rho_values.append(1 - u[i, t])
                else:
                    rho_values.append(1 - u[i, t])

    rho_values = torch.tensor(np.array(rho_values))
    for _ in range(n_iterations):
        preds = torch.reshape(rho_network(np.array(states)), (1, len(rho_values)))
        rho_loss = (rho_values - preds).abs().mean()
        rho_optimizer.zero_grad()
        rho_loss.backward()
        rho_optimizer.step()

    return rho_network


def train_rho_network_n_step(n_cell, T_terminal, rho, rho_network, rho_optimizer, n_iterations=1):
    truths, keys = list(), list()
    for i in range(len(rho)):
        for j in range(len(rho[0])):
            truths.append(rho[i, j])
            keys.append(np.array([i, j]) / n_cell)

    for _ in range(n_iterations):
        truths = torch.tensor(truths, requires_grad=True)
        preds = torch.reshape(rho_network(np.array(keys)), (1, len(truths)))
        loss = (truths - preds).abs().mean()
        rho_optimizer.zero_grad()
        loss.backward()
        rho_optimizer.step()

    return rho_network


def train_fake_critic(n_cell, T_terminal, V_array):
    T = n_cell * T_terminal
    truths = []
    keys = []

    liu = Critic(2)
    liu_optimizer = torch.optim.Adam(liu.parameters(), lr=1e-3)

    for i in range(n_cell + 1):
        for t in range(T + 1):
            truths.append(V_array[i, t])
            keys.append(np.array([i, t]) / n_cell)

    for _ in range(1000):
        truths = torch.tensor(truths, requires_grad=True)
        preds = torch.reshape(liu(np.array(keys)), (1, len(truths)))
        loss = (truths - preds).abs().mean()
        liu_optimizer.zero_grad()
        loss.backward()
        liu_optimizer.step()

    pred_V = np.zeros((n_cell + 1, T + 1))
    for i in range(n_cell + 1):
        for t in range(T + 1):
            pred_V[i, t] = liu(np.array([i, t]) / n_cell)

    return liu


def plot_3d(n_cell, T_terminal, rho, ax_name, fig_name=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
    x = np.linspace(0, 1, n_cell)
    t = np.linspace(0, T_terminal, n_cell * T_terminal)
    t_mesh, x_mesh = np.meshgrid(t, x)
    surf = ax.plot_surface(x_mesh, t_mesh, rho, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=18, pad=10)

    ax.set_xlabel(r"$x$", fontsize=24, labelpad=20)
    ax.set_xlim(min(x), max(x))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.ylabel(r"$t$", fontsize=24, labelpad=20)
    ax.set_ylim(min(t), max(t))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_zlabel(ax_name, fontsize=24, labelpad=20, rotation=90)
    # if ax_names == 'u':
    #     ax.set_zlim(.6, 1.)
    # else:
    #     ax.set_zlim(.2, .5)
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(elev=25, azim=-128)
    if not fig_name:
        plt.show()
    else:
        plt.savefig(fig_name, bbox_inches='tight')


def plot_diff(fig_path=None, smooth=False):
    for reward in ["lwr", "non-sep", "sep"]:
        if os.path.exists(f"./diff/u-{reward}.csv"):
            fig, ax = plt.subplots(figsize=(8, 4))
            u_diff_hist = pd.read_csv(f"./diff/u-{reward}.csv")['0'].values.tolist()
            u_diff_plot = savgol_filter(u_diff_hist, 53, 3) if smooth else u_diff_hist
            plt.plot(u_diff_plot, lw=3, label=r"$|u^{(k)} - u^{(k-1)}|$", c='steelblue', ls='--')
            rho_diff_hist = pd.read_csv(f"./diff/rho-{reward}.csv")['0'].values.tolist()
            rho_diff_plot = savgol_filter(rho_diff_hist, 53, 3) if smooth else rho_diff_hist
            plt.plot(rho_diff_plot, lw=3, label=r"$|\rho^{(k)} - \rho^{(k-1)}|$", c='indianred', alpha=.8)
            plt.xlabel("iterations", fontsize=18, labelpad=6)
            plt.xticks(fontsize=18)
            plt.ylabel("convergence difference", fontsize=18, labelpad=6)
            plt.yticks(fontsize=18)
            plt.ylim(-.01, .11)
            plt.legend()
            if not fig_path:
                plt.show()
            else:
                plt.savefig(f"{fig_path}/diff_{reward}.pdf", bbox_inches='tight')