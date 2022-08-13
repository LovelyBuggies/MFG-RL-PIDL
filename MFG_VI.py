import numpy as np
import pandas as pd
import copy

from value_iteration import value_iteration
from value_iteration_ddpg import train_ddpg, train_critic_fake, train_rho
from utils import get_rho_from_u, plot_3d



if __name__ == '__main__':
    n_cell = 32
    T_terminal = 1
    d = np.array([
        0.39999,
        0.39998,
        0.39993,
        0.39977,
        0.39936,
        0.39833,
        0.39602,
        0.39131,
        0.38259,
        0.36804,
        0.34619,
        0.31695,
        0.28248,
        0.24752,
        0.21861,
        0.20216,
        0.20216,
        0.21861,
        0.24752,
        0.28248,
        0.31695,
        0.34619,
        0.36804,
        0.38259,
        0.39131,
        0.39602,
        0.39833,
        0.39936,
        0.39977,
        0.39993,
        0.39998,
        0.39999
    ])

    u = 0.5 * np.ones((n_cell, n_cell * T_terminal), dtype=np.float64)
    u_hist = list()
    rho = get_rho_from_u(u, d)
    rho_network = train_rho(n_cell, T_terminal, rho)
    critic = train_critic_fake(n_cell, T_terminal, value_iteration(n_cell, T_terminal, rho_network, fake=True)[1])
    for episode in range(1000):
        print(episode)
        u, V_ddpg, actor, critic = train_ddpg(n_cell, T_terminal, rho_network, copy.deepcopy(critic))
        if episode < 10 or episode % 10 == 0:
            plot_3d(n_cell, T_terminal, rho, f"./fig_rho/ddpg_{episode}.png")
            plot_3d(n_cell, T_terminal, u, f"./fig/ddpg_{episode}.png")

        u_hist.append(u)
        u = np.array(u_hist).mean(axis=0)
        rho = get_rho_from_u(u, d)
        rho_network = train_rho(n_cell, T_terminal, rho)