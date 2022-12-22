import numpy as np
import torch

from utils import plot_diff
from value_iteration_ddpg import train_ddpg


if __name__ == '__main__':
    n_cell = 8
    T_terminal = 1

    algs = ["pidl", "rl+pidl"]
    options = ["lwr", "non-sep", "sep"]
    alg, option = algs[0], options[0]
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
        diff_plot=True,
    )
