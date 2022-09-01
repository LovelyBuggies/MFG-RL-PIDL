import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt

class LwrLoader():

    def __init__(self):
        '''
        data_param is a dictionary containing param. for data generation
        It is defined as one element in the config.json
        example of data_param:
        {
        ""
        } # currently none
        '''
        self.load_data()

    def FD(self, rho, u_max, rho_max):
        return u_max * (1 - rho / rho_max)

    def load_data(self):
        # note the time in V_x_t is one column more that that in rho_x_t (and u_x_t)
        data = scipy.io.loadmat('raw_data/origin_lwr/rho_bellshape_10grid_DS10_gn_eps005_solver2_ring.mat')
        ## all the V is fake here for running code

        self.train_data = dict()
        # t = data['t'].flatten()[:, None]  # 960 by 1
        # x = data['x'].flatten()[:, None]  # 240 by 1
        t = data['t'].flatten()
        x = data['x'].flatten()
        rho = np.real(data['rho'])
        u = self.FD(rho, 1, 1)
        self.train_data["x_of_rho"] = x
        self.train_data["t_of_rho"] = t
        self.train_data["rho"] = rho
        self.train_data["rho_0"] = rho[:, 0]

        self.train_data["x_of_u"] = x
        self.train_data["t_of_u"] = t
        self.train_data["u"] = u.T
        self.train_data["du_dx"] = (u - np.vstack([u[1:, :], u[0, :]])) / 0.0041841


        self.train_data["x_of_V"] = x
        self.train_data["t_of_V"] = t
        self.train_data["V"] = rho[:, -1]

        self.train_data["mean_x"] = np.mean(x)
        self.train_data["std_x"] = np.std(x)
        self.train_data["mean_t"] = np.mean(t)
        self.train_data["std_t"] = np.std(t)


        self.test_data = dict()
        xx, tt = np.meshgrid(x, t, indexing='ij')
        self.test_data["xdim_of_rho"] = len(x)
        self.test_data["tdim_of_rho"] = len(t)
        self.test_data["x_of_rho"] = xx.flatten()
        self.test_data["t_of_rho"] = tt.flatten()
        self.test_data["rho"] = rho.flatten()

        self.test_data["xdim_of_V"] = len(x)
        self.test_data["tdim_of_V"] = len(t)
        self.test_data["x_of_V"] = xx.flatten()
        self.test_data["t_of_V"] = tt.flatten()
        self.test_data["V"] = rho.flatten()

        self.test_data["xdim_of_u"] = len(x)
        self.test_data["tdim_of_u"] = len(t)
        self.test_data["x_of_u"] = xx.flatten()
        self.test_data["t_of_u"] = tt.flatten()
        self.test_data["u"] = self.train_data["u"].flatten()


    def construct_torch_loader(self):
        ### temporarily no need to use batch, because the data size if merely (33,33); thus we can use them all
        pass

    def get_batch(self):
        ### temporarily no need to use batch, because the data size is merely (33,33); thus we can use them all
        batch = self.train_data
        return batch
    def get_test_data(self):
        return self.test_data

