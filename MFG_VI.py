import numpy as np

from value_iteration import value_iteration
from value_iteration_a2c import value_iteration_a2c
from utils import get_rho_from_u, plot_rho

n_cell = 16
n_action = 4
T_terminal = 2
u_max = 1
episode = 15

u = 0.5 * np.ones((n_cell, n_cell * T_terminal), dtype=np.float64)
d = np.zeros((n_cell * T_terminal, 1), dtype=np.float64)
d[1] = 1

rho = get_rho_from_u(u, d)
u_hist = list()
for loop in range(episode):
    print(loop)
    u, V = value_iteration(rho, u_max, n_action)
    u_dqn, V_dqn = value_iteration_a2c(rho, u_max, n_action)
    print(u, V)
    print()
    print(u_dqn, V_dqn)
    u_hist.append(u)
    u = np.array(u_hist).mean(axis=0)
    rho = get_rho_from_u(u, d)
    plot_rho(n_cell, T_terminal, V[:-1, :-1], str(loop))
    plot_rho(n_cell, T_terminal, V_dqn[:-1, :-1], str(loop) + "_dqn")