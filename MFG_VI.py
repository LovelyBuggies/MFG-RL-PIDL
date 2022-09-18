import numpy as np
from value_iteration_ddpg import train_ddpg
from utils import get_rho_from_u
from utils import plot_3d
import pandas as pd

if __name__ == '__main__':
    n_cell = 32
    T_terminal = 1
    data = pd.read_csv('data_rho_sep.csv')
    rho = np.array(data.iloc[:, 1:len(data.iloc[0, :])])
    d = rho[:, 0]
    u, rho = train_ddpg(n_cell, T_terminal, d, 30000)


