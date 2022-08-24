import numpy as np

def value_iteration(n_cell, T_terminal, rho, fake=False):
    iteration = 36
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    u = np.zeros((n_cell, T))
    V = np.zeros((n_cell + 1, T + 1), dtype=np.float64)

    for _ in range(iteration):
        for i in range(n_cell):
            for t in range(T):
                rho_i_t = rho[i, t]
                u[i, t] = (V[i, t + 1] - V[i + 1, t + 1]) / delta_T + 1 - rho_i_t
                u[i, t] = 0.5 if fake else min(max(u[i, t], 0), 1)
                V[i, t] = delta_T * (0.5 * u[i, t] ** 2 + rho_i_t * u[i, t] - u[i, t]) + (1 - u[i, t]) * V[i, t + 1] + u[i, t] * V[i + 1, t + 1]

        V[-1, :] = V[0, :].copy()

    return u, V