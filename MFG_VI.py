import numpy as np
from value_iteration_ddpg import train_ddpg
import pandas as pd
from utils import plot_diff


if __name__ == '__main__':
    n_cell = 8
    T_terminal = 1
    options = ["lwr", "non-sep", "sep"]
    option = options[0]
    d = np.array(pd.read_csv(f'data/init_rho_{n_cell}.csv').values).flatten('F')
    train_ddpg(option, n_cell, T_terminal, d, fake_critic=True, pidl=True, surf_plot=True, smooth_plot=True, diff_plot=True)

    plot_diff("./diff/", smooth=False)