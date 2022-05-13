import numpy as np


def get_rho_from_u(u, d):
    cell = u.shape[0]
    t_num = int(u.shape[1] / u.shape[0])
    rho = np.zeros((cell, cell * t_num), dtype=np.float64)
    for t in range(cell * t_num):
        for i in range(cell):
            if t == 0:
                continue
            else:
                if i == 0:
                    rho[i][t] = rho[i][t - 1] + d[t] - rho[i][t - 1] * u[i, t - 1]
                else:
                    rho[i][t] = rho[i][t - 1] + rho[i - 1][t - 1] * u[i - 1][t - 1] - rho[i][t - 1] * u[i][t - 1]

    return rho

