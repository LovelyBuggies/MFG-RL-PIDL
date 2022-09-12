import numpy as np
from value_iteration_ddpg import train_ddpg
from utils import get_rho_from_u
from utils import plot_3d

if __name__ == '__main__':
    n_cell = 32
    T_terminal = 1
    d = np.array([
        0.8000,
        0.7999,
        0.7998,
        0.7993,
        0.7981,
        0.7950,
        0.7881,
        0.7739,
        0.7478,
        0.7041,
        0.6386,
        0.5509,
        0.4474,
        0.3426,
        0.2558,
        0.2065,
        0.2065,
        0.2558,
        0.3426,
        0.4474,
        0.5509,
        0.6386,
        0.7041,
        0.7478,
        0.7739,
        0.7881,
        0.7950,
        0.7981,
        0.7993,
        0.7998,
        0.7999,
        0.8000
    ])
    u, rho = train_ddpg(n_cell, T_terminal, d, 800)
    plot_3d(32, 1, u, f"./fig/u.png")  # show without fp
    plot_3d(32, 1, rho, f"./fig/rho.png")  # show with fp on rho


