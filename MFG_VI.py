import numpy as np
from value_iteration_ddpg import train_ddpg
import pandas as pd
from utils import plot_diff


if __name__ == '__main__':
    n_cell = 8
    T_terminal = 1
    data = pd.read_csv('data_rho_8.csv')
    rho = np.array(data.iloc[:, 1:len(data.iloc[0, :])])
    d = rho[:, 0]
    options = ["lwr", "non-sep", "sep"]
    option = options[0]
    train_ddpg(option, n_cell, T_terminal, d, fake_critic=True, pidl=True, surf_plot=True, smooth_plot=False, diff_plot=True)

    # plot_diff("./diff/", smooth=False)