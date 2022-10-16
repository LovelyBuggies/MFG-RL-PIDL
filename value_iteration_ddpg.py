import numpy as np
import pandas as pd
import torch
from utils import train_fake_critic, get_rho_from_u, get_rho_network_from_u, train_rho_network_n_step, plot_3d, plot_diff
from model import Critic, Actor, RhoNetwork


def train_ddpg(reward, n_cell, T_terminal, d, iterations, fake_critic, pidl, surf_plot, smooth_plot, diff_plot):
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    u_hist = [0.5 * np.ones((n_cell, T))]
    rho_hist = [get_rho_from_u(u_hist[0], d)]
    if diff_plot:
        u_diff_hist, rho_diff_hist = list(), list()

    actor = Actor(2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    if fake_critic:
        fake_critic = train_fake_critic(n_cell, T_terminal, np.zeros((n_cell + 1, T + 1)))
    else:
        fake_critic = Critic(2)

    critic = Critic(2)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    rho_network = RhoNetwork(2)
    rho_optimizer = torch.optim.Adam(rho_network.parameters(), lr=1e-3)
    rho = rho_hist[0]
    rho_network = train_rho_network_n_step(n_cell, T_terminal, rho, rho_network, rho_optimizer, n_iterations=1)

    for it in range(iterations):
        print(it)
        # train critic
        keys, truths = list(), list()
        for i in range(n_cell + 1):
            for t in range(T + 1):
                keys.append([i / n_cell, t / n_cell])
                if t != T:
                    if i != n_cell:
                        speed = float(actor.forward(np.array([i, t]) / n_cell))
                        rho_i_t = float(rho_network.forward(np.array([i, t]) / n_cell))
                        if reward == 'lwr':
                            truths.append(delta_T * 0.5 * (1 - speed - rho_i_t) ** 2 + fake_critic(
                                np.array([i + speed, t + 1]) / n_cell))
                        elif reward == 'non-sep':
                            truths.append(delta_T * (0.5 * speed ** 2 + rho_i_t * speed - speed) + fake_critic(
                                np.array([i + speed, t + 1]) / n_cell))
                        else:
                            raise NotImplementedError()
                    else:
                        speed = float(actor.forward(np.array([0, t]) / n_cell))
                        rho_i_t = float(rho_network.forward(np.array([0, t]) / n_cell))
                        if reward == 'lwr':
                            truths.append(delta_T * 0.5 * (1 - speed - rho_i_t) ** 2 + fake_critic(
                                np.array([speed, t + 1]) / n_cell))
                        elif reward == 'non-sep':
                            truths.append(delta_T * (0.5 * speed ** 2 + rho_i_t * speed - speed) + fake_critic(
                                np.array([speed, t + 1]) / n_cell))
                        else:
                            raise NotImplementedError()
                else:
                    truths.append(0)

        truths = torch.tensor(truths, requires_grad=True)
        n_critic_train_loop = 100 if reward == "non-sep" else 3000
        for _ in range(n_critic_train_loop):
            preds = torch.reshape(critic(np.array(keys)), (1, len(truths)))
            critic_loss = (truths - preds).abs().mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        fake_critic = critic

        # train actor
        states, rhos, Vs, Vus = list(), list(), list(), list()
        for i in range(n_cell):
            for t in range(T):
                rho_i_t = float(rho_network.forward(np.array([i, t]) / n_cell))
                states.append([i / n_cell, t / n_cell])
                rhos.append(rho_i_t)

        for i in range(T):
            for t in range(n_cell):
                Vs.append(float(critic(np.array([i, t + 1]) / n_cell) - critic(np.array([i, t]) / n_cell)))
                Vus.append(float(critic(np.array([i + 1, t + 1]) / n_cell) - critic(np.array([i, t + 1]) / n_cell)))

        states = np.array(states)
        rhos = torch.tensor(np.reshape(np.array(rhos), (n_cell * T, 1)))
        Vs = torch.tensor(np.reshape(np.array(Vs), (n_cell * T, 1)))
        Vus = torch.tensor(np.reshape(np.array(Vus), (n_cell * T, 1)))
        n_actor_train_loop = 1 if reward == "non-sep" else 3000
        if reward == 'lwr':
            for _ in range(n_actor_train_loop):
                speeds = actor.forward(states)
                advantages = delta_T * 0.5 * (1 - speeds - rhos) ** 2 + Vus * speeds + Vs
                policy_loss = advantages.mean()
                actor_optimizer.zero_grad()
                policy_loss.backward()
                actor_optimizer.step()
        elif reward == 'non-sep':
            for _ in range(n_actor_train_loop):
                speeds = actor.forward(states)
                advantages = delta_T * (0.5 * speeds ** 2 + rhos * speeds - speeds) + Vus * speeds + Vs
                policy_loss = advantages.mean()
                actor_optimizer.zero_grad()
                policy_loss.backward()
                actor_optimizer.step()
        else:
            raise NotImplementedError()

        # train rho net
        u = np.zeros((n_cell, T))
        for i in range(n_cell):
            for t in range(T):
                u[i, t] = actor(np.array([i, t]) / n_cell)
                rho[i, t] = rho_network(np.array([i, t]) / n_cell)

        u_hist.append(u)
        rho_hist.append(get_rho_from_u(u, d))
        u = np.array(u_hist).mean(axis=0)
        if diff_plot:
            u_diff_hist.append(np.mean(abs(u_hist[-1] - u_hist[-2])))
            rho_diff_hist.append(np.mean(abs(rho_hist[-1] - rho_hist[-2])))

        n_rho_train_loop = 100 if reward == "non-sep" else 3000
        if pidl:
            rho_network = get_rho_network_from_u(n_cell, T_terminal, u, d, rho_network, rho_optimizer, n_iterations=n_rho_train_loop)
        else:
            rho_network = train_rho_network_n_step(n_cell, T_terminal, rho, rho_network, rho_optimizer, n_iterations=3000)

        if surf_plot:
            plot_interval = 20 if reward == "non-sep" else 2
            if it % plot_interval == 0 and it != 0:
                if smooth_plot:
                    u_plot = np.zeros((n_cell * 4, T * 4))
                    rho_plot = np.zeros((n_cell * 4, T * 4))
                    for i in range(n_cell * 4):
                        for t in range(T * 4):
                            u_plot[i, t] = actor(np.array([i, t]) / n_cell / 4)
                            rho_plot[i, t] = rho_network(np.array([i, t]) / n_cell / 4)

                    plot_3d(n_cell * 4, T_terminal, u_plot, "u", f"./fig/u/{it}.pdf")
                    plot_3d(n_cell * 4, T_terminal, rho_plot, r"$\rho$", f"./fig/rho/{it}.pdf")
                else:
                    plot_3d(n_cell, T_terminal, u, "u", f"./fig/u/{it}.pdf")
                    plot_3d(n_cell, T_terminal, rho, r"$\rho$", f"./fig/rho/{it}.pdf")

        elif smooth_plot:
            raise ValueError("Using smooth plot when surf plot is True")

    if diff_plot:
        u_diff_df = pd.DataFrame(u_diff_hist)
        u_diff_df.to_csv(f"./diff/u-{reward}.csv")
        rho_diff_df = pd.DataFrame(rho_diff_hist)
        rho_diff_df.to_csv(f"./diff/rho-{reward}.csv")
        plot_diff()

    return
