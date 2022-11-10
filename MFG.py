import numpy as np
from value_iteration_ddpg import train_ddpg
from utils import plot_diff


if __name__ == '__main__':
    n_cell = 8
    T_terminal = 1

    algs = ["pidl", "rl+pidl"]
    options = ["lwr", "non-sep", "sep"]
    alg, option = algs[1], options[0]
    d = np.loadtxt(f"data/rho-{option}.txt")[:, 0].flatten('F')
    train_ddpg(
        alg,
        option,
        n_cell,
        T_terminal,
        d,
        fake_critic=True,
        pidl_rho_network=True,
        surf_plot=True,
        smooth_plot=False,
        diff_plot=False,
    )
