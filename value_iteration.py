import numpy as np

def value_iteration(rho, u_max, n_action):
    iteration = 36
    n_cell = rho.shape[0]
    delta_u = u_max / n_action
    u_action = np.arange(0, u_max + delta_u, delta_u)
    T_terminal = int(rho.shape[1] / rho.shape[0])
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    u = np.zeros((n_cell * n_action, T))
    V = np.zeros((n_cell * n_action + 1, T + 1), dtype=np.float64)
    u_new = np.zeros((n_cell, T))
    V_new = np.zeros((n_cell + 1, T + 1), dtype=np.float64)

    for _ in range(iteration):
        for i in range(n_cell * n_action):
            for t in range(T):
                min_value = np.float('inf')
                for j in np.arange(n_action + 1):
                    speed = u_action[j]
                    new_i = int(i + speed / delta_u)
                    rho_i = int(i / n_action)
                    if new_i <= n_cell * n_action:
                        value = delta_T * (0.5 * speed ** 2 + rho[rho_i, t] + 1) + V[new_i, t + 1]
                    else:
                        time = delta_u * delta_T * (n_cell * n_action - i) / speed
                        value = time * (0.5 * speed ** 2 + rho[rho_i, t] + 1)

                    if min_value > value:
                        min_value = value
                        u[i, t] = speed
                        V[i, t] = min_value

    for i in range(n_cell):
        for t in range(T):
            u_new[i, t] = u[i * n_action, t]
            V_new[i, t] = V[i * n_action, t]

    return u_new, V_new