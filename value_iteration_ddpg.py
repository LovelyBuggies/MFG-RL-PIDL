import numpy as np
import pandas as pd
import torch
import random
from utils import get_rho_from_u, train_critic_from_V, train_rho_network_from_rho, get_rho_network_from_u, get_rho_network_from_actor
from utils import plot_3d, plot_diff
from model import Critic, Actor, RhoNetwork


def train_ddpg(alg, option, n_cell, T_terminal, d, fake_critic, pidl_rho_network, surf_plot, smooth_plot, diff_plot):
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)

    if alg not in ["pidl", "rl+pidl"]:
        raise NotImplementedError(f'Algorithm {alg} is not implemented.')
    if option not in ["lwr", "non-sep", "sep"]:
        raise NotImplementedError(f'Reward {option} is not implemented.')
    params = {
        "lwr": {
            "n_episode": 200 if alg == "pidl" else 500, "n_train_critic": 100,
            "n_train_actor": 1, "n_train_rho_net": 100,
            "plot_interval": 20, "init_speed": 0.8,
            "reward": lambda speed, rho: 0.5 * (1 - speed - rho) ** 2,
            "optimal_speed": lambda a, b, rho: min(max(1 - rho, 0), 1),
        },
        "non-sep": {
            "n_episode": 300 if alg == "pidl" else 500, "n_train_critic": 100,
            "n_train_actor": 1, "n_train_rho_net": 100,
            "plot_interval": 20, "init_speed": 0.5,
            "reward": lambda speed, rho: 0.5 * speed ** 2 + rho * speed - speed,
            "optimal_speed": lambda a, b, rho: min(max((a - b) / delta_T + 1 - rho, 0), 1),
        },
        "sep": {
            "n_episode": 200 if alg == "pidl" else 500, "n_train_critic": 100,
            "n_train_actor": 1, "n_train_rho_net": 100,
            "plot_interval": 20, "init_speed": 0.3,
            "reward": lambda speed, rho: 0.5 * speed ** 2 + rho - speed,
            "optimal_speed": lambda a, b, rho: min(max((a - b) / delta_T + 1, 0), 1),
        },
    }

    u_hist = [params[option]["init_speed"] * np.ones((n_cell, T))]
    rho_hist = [get_rho_from_u(u_hist[0], d)]
    u_loss_hist, rho_loss_hist = list(), list()
    u_gap_hist, rho_gap_hist = list(), list()
    exploit_gap_hist = list()
    u_res = np.loadtxt(f"data/u-{option}.txt")
    rho_res = np.loadtxt(f"data/rho-{option}.txt")

    if alg != 'pidl':
        actor = Actor(2)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    u = np.zeros((n_cell, T))  # only used for pidl

    if fake_critic:
        fake_critic = train_critic_from_V(n_cell, T_terminal, np.zeros((n_cell + 1, T + 1)))
    else:
        fake_critic = Critic(2)

    critic = Critic(2)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    rho_network = RhoNetwork(2)
    rho_optimizer = torch.optim.Adam(rho_network.parameters(), lr=1e-3)
    rho = rho_hist[0]
    rho_network = train_rho_network_from_rho(n_cell, T_terminal, rho, rho_network, rho_optimizer, n_iterations=1)

    for it in range(params[option]["n_episode"] + 1):
        print(it)
        # train critic
        keys, truths = list(), list()
        for i in range(n_cell + 1):
            for t in range(T + 1):
                keys.append([i / n_cell, t / n_cell])
                if t != T:
                    if i != n_cell:
                        rho_i_t = float(rho_network.forward(np.array([i, t]) / n_cell))
                        if alg == "pidl":
                            speed = float(params[option]["optimal_speed"](
                                critic(np.array([i, t + 1]) / n_cell),
                                critic(np.array([i + 1, t + 1]) / n_cell),
                                rho_i_t,
                            ))
                        else:
                            speed = float(actor.forward(np.array([i, t]) / n_cell))
                        truths.append(
                            delta_T * params[option]["reward"](speed, rho_i_t) + \
                            fake_critic.forward(np.array([i + speed, t + 1]) / n_cell)
                        )
                        u[i, t] = speed
                    else:
                        rho_i_t = float(rho_network.forward(np.array([0, t]) / n_cell))
                        if alg == "pidl":
                            speed = float(params[option]["optimal_speed"](
                                critic(np.array([i, t + 1]) / n_cell),
                                critic(np.array([0, t + 1]) / n_cell),
                                rho_i_t,
                            ))
                        else:
                            speed = float(actor.forward(np.array([i, t]) / n_cell))
                        truths.append(
                            delta_T * params[option]["reward"](speed, rho_i_t) + \
                            fake_critic.forward(np.array([speed, t + 1]) / n_cell)
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

        # train actor
        if alg != 'pidl':
            states, rhos, Vs, Vus = list(), list(), list(), list()
            for i in range(n_cell):
                for t in range(T):
                    rho_i_t = float(rho_network.forward(np.array([i, t]) / n_cell))
                    states.append([i / n_cell, t / n_cell])
                    rhos.append(rho_i_t)

            for i in range(T):
                for t in range(n_cell):
                    Vs.append(
                        float(
                            critic.forward(np.array([i, t + 1]) / n_cell) - critic.forward(np.array([i, t]) / n_cell)))
                    Vus.append(float(
                        critic.forward(np.array([i + 1, t + 1]) / n_cell) - critic.forward(
                            np.array([i, t + 1]) / n_cell)))

            states = np.array(states)
            rhos = torch.tensor(np.reshape(np.array(rhos), (n_cell * T, 1)))
            Vs = torch.tensor(np.reshape(np.array(Vs), (n_cell * T, 1)))
            Vus = torch.tensor(np.reshape(np.array(Vus), (n_cell * T, 1)))

            for _ in range(params[option]["n_train_critic"]):
                speeds = actor.forward(states)
                advantages = delta_T * params[option]["reward"](speeds, rhos) + Vus * speeds + Vs
                policy_loss = advantages.mean()
                actor_optimizer.zero_grad()
                policy_loss.backward()
                actor_optimizer.step()

            for i in range(n_cell):
                for t in range(T):
                    u[i, t] = float(actor.forward(np.array([i, t]) / n_cell))

        # train rho net
        u_hist.append(u)
        rho_hist.append(get_rho_from_u(u, d))
        u = np.array(u_hist).mean(axis=0)
        u_exploit, exploit_gap = np.zeros((n_cell, T)), np.zeros((n_cell, T))
        for i in range(n_cell):
            for t in range(T):
                rho[i, t] = rho_network(np.array([i, t]) / n_cell)
                if diff_plot:
                    u_exploit[i, t] = float(params[option]["optimal_speed"](
                        critic(np.array([i, t + 1]) / n_cell),
                        critic(np.array([i + 1, t + 1]) / n_cell),
                        rho[i, t],
                    ))
                    exploit_cost = delta_T * params[option]["reward"](u_exploit[i, t], rho[i, t]) + critic(
                        np.array([i + u_exploit[i, t], t + 1]) / n_cell)
                    current_cost = delta_T * params[option]["reward"](u[i, t], rho[i, t]) + critic(
                        np.array([i + u[i, t], t + 1]) / n_cell)
                    exploit_gap[i, t] = exploit_cost - current_cost

        if diff_plot:
            u_loss_hist.append(np.mean(abs(u - u_res)))
            rho_loss_hist.append(np.mean(abs(rho - rho_res)))
            u_gap_hist.append(np.mean(abs(u_hist[-1] - u_hist[-2])))
            rho_gap_hist.append(np.mean(abs(rho_hist[-1] - rho_hist[-2])))
            exploit_gap_hist.append(exploit_gap.mean())

        if pidl_rho_network:
            if alg == 'pidl':
                rho_network = get_rho_network_from_u(n_cell, T_terminal, u, d, rho_network, rho_optimizer,
                                                         n_iterations=params[option]["n_train_rho_net"])
            else:
                rho_network = get_rho_network_from_actor(n_cell, T_terminal, actor, d, rho_network, rho_optimizer,
                                                     n_iterations=params[option]["n_train_rho_net"],
                                                     physical_step=random.uniform(0.9, 1))
        else:
            rho_network = train_rho_network_from_rho(n_cell, T_terminal, rho, rho_network, rho_optimizer,
                                                   n_iterations=params[option]["n_train_rho_net"])

        if surf_plot:
            if it % params[option]["plot_interval"] == 0:
                if smooth_plot:
                    u_plot = np.zeros((n_cell * 4, T * 4))
                    rho_plot = np.zeros((n_cell * 4, T * 4))
                    for i in range(n_cell * 4):
                        for t in range(T * 4):
                            u_plot[i, t] = actor.forward(np.array([i, t]) / n_cell / 4)
                            rho_plot[i, t] = rho_network(np.array([i, t]) / n_cell / 4)

                    plot_3d(n_cell * 4, T_terminal, u_plot, "u", f"./fig/u/{it}.pdf")
                    plot_3d(n_cell * 4, T_terminal, rho_plot, r"$\rho$", f"./fig/rho/{it}.pdf")
                else:
                    plot_3d(n_cell, T_terminal, u, "u", f"./fig/u/{it}.pdf")
                    plot_3d(n_cell, T_terminal, rho, r"$\rho$", f"./fig/rho/{it}.pdf")

        elif smooth_plot:
            raise ValueError("Using smooth plot when surf plot is True")

    if diff_plot:
        pd.DataFrame(u_gap_hist).to_csv(f"./diff/u-gap-{option}.csv")
        pd.DataFrame(rho_gap_hist).to_csv(f"./diff/rho-gap-{option}.csv")
        pd.DataFrame(u_loss_hist).to_csv(f"./diff/u-loss-{option}.csv")
        pd.DataFrame(rho_loss_hist).to_csv(f"./diff/rho-loss-{option}.csv")
        pd.DataFrame(exploit_gap_hist).to_csv(f"./diff/V-exploit-{option}.csv")
        plot_diff("./diff/", smooth=False)
