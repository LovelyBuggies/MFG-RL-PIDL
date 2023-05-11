import numpy as np
import pandas as pd
import torch
import random
from utils import (
    get_rho_from_u,
    get_rho_from_u_at_t,
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
            "n_train_critic": 1000,
            "n_train_actor": 500,
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

    actor = Actor(n_cell)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    fake_critic = Critic(n_cell)
    critic = Critic(n_cell)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.5 * 1e-4)
    for it in range(params[option]["n_episode"] + 1):
        print(it)
        rhos, rhos_next = list(), list()
        rho = torch.tensor(d)
        for t in range(T):
            speed = actor.forward(rho)
            rho_next = get_rho_from_u_at_t(n_cell, rho, speed)
            rhos.append(rho.tolist())
            rhos_next.append(rho_next.tolist())
            rho = rho_next.detach().clone()

        # train actor
        rhos, rhos_next = torch.tensor(rhos, requires_grad=True), torch.tensor(
            rhos_next, requires_grad=True
        )
        for _ in range(params[option]["n_train_actor"]):
            speeds = actor.forward(rhos)
            advantages = torch.sum(
                rhos * delta_T * params[option]["reward"](speeds, rhos), 0
            ) + torch.reshape(
                fake_critic.forward(rhos_next) - fake_critic.forward(rhos), (T, 1)
            )
            policy_loss = advantages.mean()
            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()

        # train critic
        for _ in range(params[option]["n_train_critic"]):
            speeds = actor.forward(rhos)
            critic_advantages = torch.sum(
                rhos * delta_T * params[option]["reward"](speeds, rhos), 0
            ) + torch.reshape(
                fake_critic.forward(rhos_next) - fake_critic.forward(rhos), (T, 1)
            )
            critic_loss = critic_advantages.mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        fake_critic = critic

        if surf_plot:
            if it % params[option]["plot_interval"] == 0:
                plot_3d(
                    n_cell,
                    T_terminal,
                    speeds.detach().numpy(),
                    "u",
                    f"./fig/u/{it}.pdf",
                )
                plot_3d(
                    n_cell,
                    T_terminal,
                    rhos.detach().numpy(),
                    r"$\rho$",
                    f"./fig/rho/{it}.pdf",
                )
