import numpy as np
import pandas as pd
import torch
import random
from utils import (
    get_rho_from_u,
    train_critic_from_V,
    train_rho_network_from_rho,
    get_rho_network_from_u,
    get_rho_network_from_actor,
)
from utils import plot_3d, plot_diff
from model import Critic, Actor, RhoNetwork


def train_ddpg(
    alg,
    option,
    n_cell,
    T_terminal,
    d,
    fake_critic,
    pidl_rho_network,
    surf_plot,
    smooth_plot,
    diff_plot,
):
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)

    if alg not in ["pidl", "rl+pidl"]:
        raise NotImplementedError(f"Algorithm {alg} is not implemented.")
    if option not in ["lwr", "non-sep", "sep"]:
        raise NotImplementedError(f"Reward {option} is not implemented.")
    params = {
        "lwr": {
            "n_episode": 200 if alg == "pidl" else 500,
            "n_train_critic": 100,
            "n_train_actor": 1,
            "n_train_rho_net": 100,
            "plot_interval": 20,
            "init_speed": 0.8,
            "reward": lambda speed, rho: 0.5 * (1 - speed - rho) ** 2,
            "optimal_speed": lambda a, b, rho: min(max(1 - rho, 0), 1),
        },
        "non-sep": {
            "n_episode": 300 if alg == "pidl" else 500,
            "n_train_critic": 100,
            "n_train_actor": 1,
            "n_train_rho_net": 100,
            "plot_interval": 20,
            "init_speed": 0.5,
            "reward": lambda speed, rho: 0.5 * speed**2 + rho * speed - speed,
            "optimal_speed": lambda a, b, rho: min(
                max((a - b) / delta_T + 1 - rho, 0), 1
            ),
        },
        "sep": {
            "n_episode": 200 if alg == "pidl" else 500,
            "n_train_critic": 100,
            "n_train_actor": 1,
            "n_train_rho_net": 100,
            "plot_interval": 20,
            "init_speed": 0.3,
            "reward": lambda speed, rho: 0.5 * speed**2 + rho - speed,
            "optimal_speed": lambda a, b, rho: min(max((a - b) / delta_T + 1, 0), 1),
        },
    }

    u_hist = [params[option]["init_speed"] * np.ones((n_cell, T))]
    u = np.zeros((n_cell, T))
    if fake_critic:
        fake_critic = train_critic_from_V(
            n_cell, T_terminal, np.zeros((n_cell + 1, T + 1))
        )
    else:
        fake_critic = Critic(2)

    critic = Critic(2)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    for it in range(params[option]["n_episode"] + 1):
        print(it)
        # train critic
        keys, truths = list(), list()
        for i in range(n_cell + 1):
            for t in range(T + 1):
                keys.append([i / n_cell, t / n_cell])
                if t != T:
                    if i != n_cell:
                        rho = get_rho_from_u(n_cell, T_terminal, u, d)
                        rho_i_t = rho[i, t]
                        speed = float(
                            params[option]["optimal_speed"](
                                critic(np.array([i, t + 1]) / n_cell),
                                critic(np.array([i + 1, t + 1]) / n_cell),
                                rho_i_t,
                            )
                        )
                        truths.append(
                            delta_T * params[option]["reward"](speed, rho_i_t)
                            + fake_critic.forward(np.array([i + speed, t + 1]) / n_cell)
                        )
                        u[i, t] = speed
                    else:
                        rho = get_rho_from_u(n_cell, T_terminal, u, d)
                        rho_i_t = rho[0, t]
                        speed = float(
                            params[option]["optimal_speed"](
                                critic(np.array([i, t + 1]) / n_cell),
                                critic(np.array([0, t + 1]) / n_cell),
                                rho_i_t,
                            )
                        )
                        truths.append(
                            delta_T * params[option]["reward"](speed, rho_i_t)
                            + fake_critic.forward(np.array([speed, t + 1]) / n_cell)
                        )

                else:
                    truths.append(0)

        truths = torch.tensor(truths, requires_grad=True)
        for _ in range(params[option]["n_train_critic"]):
            preds = torch.reshape(critic.forward(np.array(keys)), (1, len(truths)))
            critic_loss = (truths - preds).abs().mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        fake_critic = critic

        # u_hist.append(u)
        # u = np.array(u_hist).mean(axis=0)

        if surf_plot:
            if it % params[option]["plot_interval"] == 0:
                plot_3d(n_cell, T_terminal, u, "u", f"./fig/u/{it}.pdf")
                plot_3d(n_cell, T_terminal, rho, r"$\rho$", f"./fig/rho/{it}.pdf")
