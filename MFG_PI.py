import numpy as np
from policy_iteration import policy_iteration, policy_iteration_critic
from utils import get_rho_from_u, plot_rho


cell = 32
t_num = 4
episode = 5
tolerance = 1

u = np.zeros((cell, cell * t_num), dtype=np.float64)
u_hist = np.array([u])

d = np.zeros((cell * t_num, 1), dtype=np.float64)
d[1] = 1

rho = get_rho_from_u(u, d)
rho_tmp = np.zeros(u.shape, dtype=np.float64)
for loop in range(episode):
    print(f"********************* RL Loop: {loop} ********************* \n")
    u, v = policy_iteration_critic(rho)
    np.append(u_hist, u)
    if np.sqrt(sum(sum((rho - rho_tmp)**2))) < tolerance:
        break

    rho_tmp = rho
    rho = get_rho_from_u(u, d)
    plot_rho(cell, t_num, rho, str(loop))
