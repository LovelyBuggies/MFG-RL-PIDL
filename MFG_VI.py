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
    reward = "lwr"
    n_episodes = 1500 if reward == "non-sep" else 30
    train_ddpg(reward, n_cell, T_terminal, d, n_episodes, fake_critic=True, pidl=True, surf_plot=True, smooth_plot=False, diff_plot=False)

    # plot_diff()