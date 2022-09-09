import numpy as np
from value_iteration_ddpg import train_ddpg
from utils import get_rho_from_u

if __name__ == '__main__':
    n_cell = 32
    T_terminal = 1
    T = n_cell * T_terminal
    u = 0.5 * np.ones((n_cell, T))
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
    rho = get_rho_from_u(u, d)
    u = train_ddpg(rho, d, 100000)


