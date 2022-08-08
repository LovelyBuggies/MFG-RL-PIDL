import numpy as np
import csv

from value_iteration import value_iteration
from value_iteration_dqn import value_iteration_dqn
from value_iteration_a2c import value_iteration_a2c
from value_iteration_ddpg import value_iteration_ddpg
from utils import get_rho_from_u, plot_rho, calculate_optimal_costs


def MFG(n_cell, n_action, T_terminal, u_max, episode, d):
    u = 0.5 * np.ones((n_cell, n_cell * T_terminal), dtype=np.float64)
    rho = get_rho_from_u(u, d)
    u_hist = list()
    u_hist_dqn = list()
    u_hist_a2c = list()
    u_hist_ddpg = list()
    for loop in range(episode):
        print(loop)
        u, V = value_iteration(rho, u_max, n_action)
        u_dqn, V_dqn = value_iteration_dqn(rho, u_max, n_action)
        u_a2c, V_a2c = value_iteration_a2c(rho, u_max, n_action)
        u_ddpg, V_ddpg = value_iteration_ddpg(rho, u_max, n_action)
        print(u, V, '\n')
        print(u_dqn, V_dqn, '\n')
        print(u_a2c, V_a2c, '\n')
        print(u_ddpg, V_ddpg, '\n')
        u_hist.append(u)
        u_hist_dqn.append(u_dqn)
        u_hist_a2c.append(u_a2c)
        u_hist_ddpg.append(u_ddpg)
        u = np.array(u_hist).mean(axis=0)
        u_dqn = np.array(u_hist_dqn).mean(axis=0)
        u_a2c = np.array(u_hist_a2c).mean(axis=0)
        u_ddpg = np.array(u_hist_ddpg).mean(axis=0)
        rho = get_rho_from_u(u, d)
        rho_dqn = get_rho_from_u(u_dqn, d)
        rho_a2c = get_rho_from_u(u_a2c, d)
        rho_ddpg = get_rho_from_u(u_ddpg, d)
        plot_rho(n_cell, T_terminal, V[:-1, :-1], f"./fig/{loop}.png")
        plot_rho(n_cell, T_terminal, rho, f"./fig_rho/{loop}.png")
        plot_rho(n_cell, T_terminal, V_dqn[:-1, :-1], f"./fig/{loop}_dqn.png")
        plot_rho(n_cell, T_terminal, rho_dqn, f"./fig_rho/{loop}_dqn.png")
        plot_rho(n_cell, T_terminal, V_a2c[:-1, :-1], f"./fig/{loop}_a2c.png")
        plot_rho(n_cell, T_terminal, rho_a2c, f"./fig_rho/{loop}_a2c.png")
        plot_rho(n_cell, T_terminal, V_ddpg[:-1, :-1], f"./fig/{loop}_ddpg.png")
        plot_rho(n_cell, T_terminal, rho_ddpg, f"./fig_rho/{loop}_ddpg.png")

    return calculate_optimal_costs(u_dqn, V_dqn)


if __name__ == '__main__':
    n_cell = 16
    n_action = 4
    T_terminal = 4
    u_max = 1
    episode = 15

    with open('demand.csv', 'w', newline='') as csv_file:
        field_names = [str(s) for s in list(range(n_cell * T_terminal))] + ['label']
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for _ in range(1):
            d = np.zeros((n_cell * T_terminal, 1), dtype=float)
            d[1] = 1
            label = MFG(n_cell, n_action, T_terminal, u_max, episode, d)
            dict_to_write = {'label': label}
            for i in range(n_cell * T_terminal):
                dict_to_write[str(i)] = d[i][0]

            writer.writerow(dict_to_write)
