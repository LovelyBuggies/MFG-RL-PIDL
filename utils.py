import os
import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.signal import savgol_filter

from model import Critic, Actor


def get_rho_from_u(u, d, option="ring"):
    n_cell = u.shape[0]
    T_terminal = int(u.shape[1] / u.shape[0])
    rho = np.zeros((n_cell, n_cell * T_terminal), dtype=np.float64)
    for t in range(n_cell * T_terminal):
        for i in range(n_cell):
            if t == 0:
                rho[i, t] = d[i] if option == "ring" else 0
            else:
                if i == 0:
                    if option == "ring":
                        rho[i, t] = (
                            rho[i, t - 1]
                            + rho[-1, t - 1] * u[-1, t - 1]
                            - rho[i, t - 1] * u[i, t - 1]
                        )
                    else:
                        rho[i][t] = rho[i][t - 1] + d[t] - rho[i][t - 1] * u[i, t - 1]
                else:
                    rho[i, t] = (
                        rho[i][t - 1]
                        + rho[i - 1, t - 1] * u[i - 1, t - 1]
                        - rho[i, t - 1] * u[i, t - 1]
                    )

    return rho


""" Supervised Learning """


def train_critic_from_V(n_cell, T_terminal, V):
    T = n_cell * T_terminal
    truths = []
    keys = []

    critic = Critic(2)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    for i in range(n_cell + 1):
        for t in range(T + 1):
            truths.append(V[i, t])
            keys.append(np.array([i, t]) / n_cell)

    for _ in range(1000):
        truths = torch.tensor(truths, requires_grad=True)
        preds = torch.reshape(critic(np.array(keys)), (1, len(truths)))
        loss = (truths - preds).abs().mean()
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()

    pred_V = np.zeros((n_cell + 1, T + 1))
    for i in range(n_cell + 1):
        for t in range(T + 1):
            pred_V[i, t] = critic(np.array([i, t]) / n_cell)

    return critic


def train_actor_from_u(u):
    actor = Actor(2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    truths = []
    keys = []
    for i in range(8):
        for j in range(8):
            truths.append(u[i, j])
            keys.append(np.array([i, j]) / 8)

    truths = torch.tensor(truths, requires_grad=True)
    for _ in range(5000):
        preds = torch.reshape(actor(np.array(keys)), (1, len(truths)))
        loss = (truths - preds).abs().mean()
        actor_optimizer.zero_grad()
        loss.backward()
        actor_optimizer.step()

    return actor


def train_rho_network_from_rho(
    n_cell, T_terminal, rho, rho_network, rho_optimizer, n_iterations=1
):
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


""" PIDL """


def get_rho_network_from_u(
    n_cell,
    T_terminal,
    u,
    d,
    rho_network,
    rho_optimizer,
    n_iterations=100,
    option="ring",
):
    states, rho_values = list(), list()
    T = n_cell * T_terminal
    for t in range(T):
        for i in range(n_cell):
            states.append([i / n_cell, t / n_cell])
            # ini_rho -> rho
            if t == 0:
                rho_values.append(d[i] if option == "ring" else 0)
            else:
                if i == 0:
                    if option == "ring":
                        rho_values.append(
                            float(
                                rho_network(np.array([i, t - 1]) / n_cell)
                                + rho_network(np.array([n_cell - 1, t - 1]) / n_cell)
                                * u[-1, t - 1]
                                - rho_network(np.array([i, t - 1]) / n_cell)
                                * u[i, t - 1]
                            )
                        )
                else:
                    rho_values.append(
                        float(
                            rho_network(np.array([i, t - 1]) / n_cell)
                            + rho_network(np.array([i - 1, t - 1]) / n_cell)
                            * u[i - 1, t - 1]
                            - rho_network(np.array([i, t - 1]) / n_cell) * u[i, t - 1]
                        )
                    )

    rho_values = torch.tensor(np.array(rho_values))
    for _ in range(n_iterations):
        preds = torch.reshape(rho_network(np.array(states)), (1, len(rho_values)))
        rho_loss = (rho_values - preds).abs().mean()
        rho_optimizer.zero_grad()
        rho_loss.backward()
        rho_optimizer.step()

    return rho_network


def get_rho_network_from_actor(
    n_cell,
    T_terminal,
    actor,
    d,
    rho_network,
    rho_optimizer,
    n_iterations=100,
    physical_step=1,
    option="ring",
):
    states, rho_values = list(), list()
    T = n_cell * T_terminal
    for t in range(T):
        for i in range(n_cell):
            states.append([i / n_cell, t / n_cell])
            if t == 0:
                rho_values.append(d[i] if option == "ring" else 0)
            else:
                if i == 0:
                    if option == "ring":
                        rho_values.append(
                            float(
                                rho_network(np.array([i, t - physical_step]) / n_cell)
                                + rho_network(
                                    np.array(
                                        [n_cell - physical_step, t - physical_step]
                                    )
                                    / n_cell
                                )
                                * actor.forward(
                                    np.array(
                                        [n_cell - physical_step, t - physical_step]
                                    )
                                    / n_cell
                                )
                                - rho_network(np.array([i, t - physical_step]) / n_cell)
                                * actor.forward(
                                    np.array([i, t - physical_step]) / n_cell
                                )
                            )
                        )
                else:
                    rho_values.append(
                        float(
                            rho_network(np.array([i, t - physical_step]) / n_cell)
                            + rho_network(
                                np.array([i - physical_step, t - physical_step])
                                / n_cell
                            )
                            * actor.forward(
                                np.array([i - physical_step, t - physical_step])
                                / n_cell
                            )
                            - rho_network(np.array([i, t - physical_step]) / n_cell)
                            * actor.forward(np.array([i, t - physical_step]) / n_cell)
                        )
                    )

    rho_values = torch.tensor(np.array(rho_values))
    for _ in range(n_iterations):
        preds = torch.reshape(rho_network(np.array(states)), (1, len(rho_values)))
        rho_loss = (rho_values - preds).abs().mean()
        rho_optimizer.zero_grad()
        rho_loss.backward()
        rho_optimizer.step()

    return rho_network


""" Plotting """


def plot_3d(n_cell, T_terminal, rho, ax_name, fig_name=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection="3d")
    x = np.linspace(0, 1, n_cell)
    t = np.linspace(0, T_terminal, n_cell * T_terminal)
    t_mesh, x_mesh = np.meshgrid(t, x)
    surf = ax.plot_surface(
        x_mesh, t_mesh, rho, cmap=cm.jet, linewidth=0, antialiased=False
    )
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=18, pad=10)

    ax.set_xlabel(r"$x$", fontsize=24, labelpad=20)
    ax.set_xlim(min(x), max(x))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    plt.ylabel(r"$t$", fontsize=24, labelpad=20)
    ax.set_ylim(min(t), max(t))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    ax.set_zlabel(ax_name, fontsize=24, labelpad=20, rotation=90)
    # if ax_names == 'u':
    #     ax.set_zlim(.6, 1.)
    # else:
    #     ax.set_zlim(.2, .5)
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.view_init(elev=25, azim=-128)
    if not fig_name:
        plt.show()
    else:
        plt.savefig(fig_name, bbox_inches="tight")


def plot_diff(fig_path=None, smooth=False):
    colors = ["orange", "steelblue", "indianred"]
    labels = ["LWR", "Separable", "Non-separable"]
    lss = ["-", "--", ":"]
    for i, option in enumerate(["lwr", "sep", "non-sep"]):
        if os.path.exists(f"./diff/u-gap-{option}.csv"):
            fig, ax = plt.subplots(figsize=(8, 4))
            u_diff_hist = pd.read_csv(f"./diff/u-gap-{option}.csv")["0"].values.tolist()
            rho_diff_hist = pd.read_csv(f"./diff/rho-gap-{option}.csv")[
                "0"
            ].values.tolist()
            if smooth:
                u_diff_plot = savgol_filter([u for u in u_diff_hist], 13, 3)
                u_diff_plot = [u if u > 0 else 0 for u in u_diff_plot]
                rho_diff_plot = savgol_filter([rho for rho in rho_diff_hist], 13, 3)
                rho_diff_plot = [rho if rho > 0 else 0 for rho in rho_diff_plot]
            else:
                u_diff_plot = u_diff_hist
                rho_diff_plot = rho_diff_hist

            plt.plot(
                u_diff_plot,
                lw=3,
                label=r"$|u^{(i)} - u^{(i-1)}|$",
                c="steelblue",
                ls="--",
            )
            plt.plot(
                rho_diff_plot,
                lw=3,
                label=r"$|\rho^{(i)} - \rho^{(i-1)}|$",
                c="indianred",
                alpha=0.8,
            )
            plt.xlabel("iterations", fontsize=18, labelpad=6)
            plt.xticks(fontsize=18)
            plt.ylabel("convergence gap", fontsize=18, labelpad=6)
            plt.yticks(fontsize=18)
            plt.ylim(-0.01, 0.11)
            plt.legend(prop={"size": 16})
            if not fig_path:
                plt.show()
            else:
                plt.savefig(f"{fig_path}/pidl_gap_{option}.pdf", bbox_inches="tight")

        if os.path.exists(f"./diff/u-loss-{option}.csv"):
            fig, ax = plt.subplots(figsize=(8, 4))
            u_diff_hist = pd.read_csv(f"./diff/u-loss-{option}.csv")[
                "0"
            ].values.tolist()
            rho_diff_hist = pd.read_csv(f"./diff/rho-loss-{option}.csv")[
                "0"
            ].values.tolist()
            if smooth:
                u_diff_plot = savgol_filter([u for u in u_diff_hist], 13, 3)
                u_diff_plot = [u if u > 0 else 0 for u in u_diff_plot]
                rho_diff_plot = savgol_filter([rho for rho in rho_diff_hist], 13, 3)
                rho_diff_plot = [rho if rho > 0 else 0 for rho in rho_diff_plot]
            else:
                u_diff_plot = u_diff_hist
                rho_diff_plot = rho_diff_hist

            plt.plot(
                u_diff_plot, lw=3, label=r"$|u^{(i)} - u^*|$", c="steelblue", ls="--"
            )
            plt.plot(
                rho_diff_plot,
                lw=3,
                label=r"$|\rho^{(i)} - \rho^*|$",
                c="indianred",
                alpha=0.8,
            )
            plt.xlabel("iterations", fontsize=18, labelpad=6)
            plt.xticks(fontsize=18)
            plt.ylabel("loss", fontsize=18, labelpad=6)
            plt.yticks(fontsize=18)
            plt.ylim(-0.01, 0.1)
            plt.legend(prop={"size": 16})
            if not fig_path:
                plt.show()
            else:
                plt.savefig(f"{fig_path}/pidl_loss_{option}.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, option in enumerate(["lwr", "sep", "non-sep"]):
        if os.path.exists(f"./diff/V-exploit-{option}.csv"):
            V_exploit_gap_hist = pd.read_csv(f"./diff/V-exploit-{option}.csv")[
                "0"
            ].values.tolist()
            if smooth:
                V_exploit_gap_hist_plot = savgol_filter(
                    [u for u in V_exploit_gap_hist], 13, 3
                )
                V_exploit_gap_hist_plot = [
                    u if u > 0 else 0 for u in V_exploit_gap_hist_plot
                ]
            else:
                V_exploit_gap_hist_plot = V_exploit_gap_hist

            plt.plot(
                [100 * i for i in range(len(V_exploit_gap_hist_plot))],
                V_exploit_gap_hist_plot,
                lw=3,
                c=colors[i],
                ls=lss[i],
                label=labels[i],
            )

    plt.xlabel("steps", fontsize=18, labelpad=6)
    plt.xticks(fontsize=18)
    plt.ylabel("exploitability", fontsize=18, labelpad=6)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16, loc=4)
    plt.ylim(-0.005, 0.0005)
    if not fig_path:
        plt.show()
    else:
        plt.savefig(f"{fig_path}/rl_pidl_exploit.pdf", bbox_inches="tight")
