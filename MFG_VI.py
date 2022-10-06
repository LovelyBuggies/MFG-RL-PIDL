import numpy as np
from value_iteration_ddpg import train_ddpg
import pandas as pd

if __name__ == '__main__':
    n_cell = 8
    T_terminal = 1
    data = pd.read_csv('data_rho_8.csv')
    rho = np.array(data.iloc[:, 1:len(data.iloc[0, :])])
    d = rho[:, 0]
    train_ddpg('lwr', n_cell, T_terminal, d, 5000, fake_start=False)
