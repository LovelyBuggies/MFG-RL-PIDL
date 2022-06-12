import numpy as np


def value_iteration(rho):
    episode = 500
    cell = rho.shape[0]
    t_num = int(rho.shape[1] / rho.shape[0])
    t_delta = 1 / cell
    u = np.zeros(rho.shape, dtype=np.float64)
    v = np.zeros((rho.shape[0] + 1, rho.shape[1] + 1), dtype=np.float64)
    v_0 = 2
    for i in range(cell + 1):
        v[i][-1] = v_0 - i * v_0 / cell

    for _ in range(episode):
        for t in range(t_num * cell):
            for i in range(cell):
                u[i][t] = (v[i][t + 1] - v[i + 1][t + 1]) / (2 * t_delta)
                u[i][t] = min(max(u[i][t], 0), 1)
                v[i][t] = t_delta * (0.5 * np.power(u[i][t], 2) + rho[i][t] + 1) + \
                          (1 - u[i][t]) * v[i][t + 1] + u[i][t] * v[i + 1][t + 1]

    return u, v

def value_iteration_non_separable(rho):
    episode = 500
    cell = rho.shape[0]
    t_num = int(rho.shape[1] / rho.shape[0])
    t_delta = 1 / cell
    u = np.zeros(rho.shape, dtype=np.float64)
    v = np.zeros((rho.shape[0] + 1, rho.shape[1] + 1), dtype=np.float64)
    v_0 = 2
    for i in range(cell + 1):
        v[i][-1] = v_0 - i * v_0 / cell

    for _ in range(episode):
        for t in range(t_num * cell):
            for i in range(cell):
                u[i][t] = (v[i][t + 1] - v[i + 1][t + 1] - rho[i][t] * t_delta) / (2 * t_delta)
                u[i][t] = min(max(u[i][t], 0), 1)
                v[i][t] = t_delta * (0.5 * np.power(u[i][t], 2) + rho[i][t] * u[i][t] + 1) + \
                          (1 - u[i][t]) * v[i][t + 1] + u[i][t] * v[i + 1][t + 1]

    return u, v
