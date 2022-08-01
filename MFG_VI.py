import numpy as np

from value_iteration import value_iteration
from value_iteration_dqn import value_iteration_dqn
from value_iteration_a2c import value_iteration_a2c
from utils import get_rho_from_u, plot_rho

n_cell = 16
n_action = 4
T_terminal = 4
u_max = 1
episode = 15

u = 0.5 * np.ones((n_cell, n_cell * T_terminal), dtype=np.float64)
d = np.zeros((n_cell * T_terminal, 1), dtype=np.float64)
d[1] = 1

rho = get_rho_from_u(u, d)
u_hist = list()
u_hist_dqn = list()
u_hist_a2c = list()
for loop in range(episode):
    print(loop)
    u, V = value_iteration(rho, u_max, n_action)
    u_dqn, V_dqn = value_iteration_dqn(rho, u_max, n_action)
    u_a2c, V_a2c = value_iteration_a2c(rho, u_max, n_action)
    print(u, V, '\n')
    print(u_dqn, V_dqn, '\n')
    print(u_a2c, V_a2c, '\n')
    u_hist.append(u)
    u_hist_dqn.append(u_dqn)
    u_hist_a2c.append(u_a2c)
    u = np.array(u_hist).mean(axis=0)
    u_dqn = np.array(u_hist_dqn).mean(axis=0)
    u_a2c = np.array(u_hist_a2c).mean(axis=0)
    rho = get_rho_from_u(u, d)
    rho_dqn = get_rho_from_u(u_dqn, d)
    rho_a2c = get_rho_from_u(u_a2c, d)
    plot_rho(n_cell, T_terminal, V[:-1, :-1], f"./fig/{loop}.png")
    plot_rho(n_cell, T_terminal, rho, f"./fig_rho/{loop}.png")
    plot_rho(n_cell, T_terminal, V_dqn[:-1, :-1], f"./fig/{loop}_dqn.png")
    plot_rho(n_cell, T_terminal, rho_dqn, f"./fig_rho/{loop}_dqn.png")
    plot_rho(n_cell, T_terminal, V_a2c[:-1, :-1], f"./fig/{loop}_a2c.png")
    plot_rho(n_cell, T_terminal, rho_a2c, f"./fig_rho/{loop}_a2c.png")