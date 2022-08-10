import numpy as np

from value_iteration import value_iteration
from value_iteration_ddpg import value_iteration_ddpg
from utils import get_rho_from_u, plot_rho


def MFG(n_cell, T_terminal, u_max, episode, d):
    u = 0.5 * np.ones((n_cell, n_cell * T_terminal), dtype=np.float64)
    rho = get_rho_from_u(u, d)
    u_hist = list()
    u_hist_ddpg = list()
    for loop in range(episode):
        print(loop)
        u, V = value_iteration(rho, u_max)
        u_ddpg, V_ddpg = value_iteration_ddpg(rho, u_max)
        # print(u, V, '\n')
        # print(u_ddpg, V_ddpg, '\n')
        u_hist.append(u)
        u_hist_ddpg.append(u_ddpg)
        u = np.array(u_hist).mean(axis=0)
        u_ddpg = np.array(u_hist_ddpg).mean(axis=0)
        rho = get_rho_from_u(u, d)
        rho_ddpg = get_rho_from_u(u_ddpg, d)
        plot_rho(n_cell, T_terminal, V[:-1, :-1], f"./fig/{loop}.png")
        plot_rho(n_cell, T_terminal, rho, f"./fig_rho/{loop}.png")
        plot_rho(n_cell, T_terminal, V_ddpg[:-1, :-1], f"./fig/{loop}_ddpg.png")
        plot_rho(n_cell, T_terminal, rho_ddpg, f"./fig_rho/{loop}_ddpg.png")


if __name__ == '__main__':
    n_cell = 16
    T_terminal = 1
    u_max = 1
    episode = 100

    d = np.array([
        0.799965565466756,
        0.799608644838254,
        0.796856950383300,
        0.782162470168305,
        0.728464520450162,
        0.597273007215400,
        0.394019692302963,
        0.225484614598068,
        0.225484614598068,
        0.394019692302963,
        0.597273007215400,
        0.728464520450162,
        0.782162470168305,
        0.796856950383300,
        0.799608644838254,
        0.799965565466756,
    ])
    MFG(n_cell, T_terminal, u_max, episode, d)
