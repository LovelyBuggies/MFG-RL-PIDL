import numpy as np
import os

class Lwr_f_Loader():

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

    def load_data(self):
        # note the time in V_x_t is one column more that that in rho_x_t (and u_x_t)
        rho_x_t = np.loadtxt(os.path.join("data", "lwr_0.5_0.6", "data_rho_lwr.csv"), delimiter=',')
        V_x_t = np.loadtxt(os.path.join("data", "lwr_0.5_0.6", "data_u_lwr.csv"), delimiter=',')
        u_x_t = np.loadtxt(os.path.join("data", "lwr_0.5_0.6", "data_u_lwr.csv"), delimiter=',')


        x_of_rho = rho_x_t[1:, 0]
        t_of_rho = rho_x_t[0, 1:]
        t_of_rho_obs = np.ones_like(x_of_rho) * rho_x_t[0, 0]
        rho = rho_x_t[1:, 1:33]

        x_of_V = V_x_t[1:, 0] # equals to x_of_rho
        t_of_V = V_x_t[0, 1:]
        t_of_V_obs = np.ones_like(x_of_V) * V_x_t[0, -1]
        V = V_x_t[1:, 1:]

        x_of_u = u_x_t[1:, 0]
        t_of_u = u_x_t[0, 1:]
        u = u_x_t[1:, 1:33]

        ### below is the training data
        self.train_data = dict()
        # initial condition
        self.train_data["x_of_rho"] = x_of_rho
        self.train_data["t_of_rho"] = t_of_rho_obs
        self.train_data["rho_0"] = rho[:, 0]
        self.train_data["rho"] = rho
        # u
        self.train_data["x_of_u"] = x_of_u
        self.train_data["t_of_u"] = t_of_u
        self.train_data["u"] = u
        self.train_data["du_dx"] = (u - np.vstack([u[1:, :], u[0, :]])) / (1/32)

        # terminal cost
        self.train_data["x_of_V"] = x_of_V
        self.train_data["t_of_V"] = t_of_V_obs
        self.train_data["V"] = V[:, -1]

        # hyper
        self.train_data["mean_x"] = np.mean(x_of_rho)
        self.train_data["std_x"] = np.std(x_of_rho)
        self.train_data["mean_t"] = np.mean(t_of_rho)
        self.train_data["std_t"] = np.std(t_of_rho)



        ### below is the test data
        self.test_data = dict()
        xx, tt = np.meshgrid(x_of_rho, t_of_rho, indexing='ij')
        self.test_data["xdim_of_rho"] = len(x_of_rho)
        self.test_data["tdim_of_rho"] = len(t_of_rho)
        self.test_data["x_of_rho"] = xx.flatten()
        self.test_data["t_of_rho"] = tt.flatten()
        self.test_data["rho"] = rho.flatten()

        xx, tt = np.meshgrid(x_of_V, t_of_V, indexing='ij')
        self.test_data["xdim_of_V"] = len(x_of_V)
        self.test_data["tdim_of_V"] = len(t_of_V)
        self.test_data["x_of_V"] = xx.flatten()
        self.test_data["t_of_V"] = tt.flatten()
        self.test_data["V"] = V.flatten()

        xx, tt = np.meshgrid(x_of_u, t_of_u, indexing='ij')
        self.test_data["xdim_of_u"] = len(x_of_u)
        self.test_data["tdim_of_u"] = len(t_of_u)
        self.test_data["x_of_u"] = xx.flatten()
        self.test_data["t_of_u"] = tt.flatten()
        self.test_data["u"] = u.flatten()

    def construct_torch_loader(self):
        ### temporarily no need to use batch, because the data size if merely (33,33); thus we can use them all
        pass

    def get_batch(self):
        ### temporarily no need to use batch, because the data size is merely (33,33); thus we can use them all
        batch = self.train_data
        return batch
    def get_test_data(self):
        return self.test_data

class LwrSepLoaderFullObs():
    ## both rho and V are observed at t=0 and t=T

    def __init__(self, data_param):
        '''
        data_param is a dictionary containing param. for data generation
        It is defined as one element in the config.json
        example of data_param:
        {
        ""
        } # currently none
        '''
        self.data_param = data_param
        self.load_data()

    def load_data(self):
        # function to place loop
        def get_obs(x_1d, t_1d, var_2d, where_x, where_t):
            '''
            :param x_1d: the whole x: np.array([0,1])
            :param t_1d: the whole t: np.array([0,1])
            :param var_2d: the state variable, 32*32 (rho) or 32*33 (V)
            :param where_x: location of x_obs; eg: np.array([0,1,....,end])
            :param where_t: location of t_obs; eg: np.array([0,16, 32])
            :return: x_of_obs, t_of_obs, var_of_obs
            '''
            whole_x_2d, whole_t_2d = np.meshgrid(x_1d, t_1d, indexing='ij')
            obs_xt_idx = np.ix_(where_x, where_t)
            x_of_obs = whole_x_2d[obs_xt_idx]
            t_of_obs = whole_t_2d[obs_xt_idx]
            var_of_obs = var_2d[obs_xt_idx]

            return x_of_obs, t_of_obs, var_of_obs

        # note the time in V_x_t is one column more that that in rho_x_t (and u_x_t)
        rho_x_t = np.loadtxt(os.path.join("data", "lwr_sep", "data_rho_sep.csv"), delimiter=',')
        V_x_t = np.loadtxt(os.path.join("data", "lwr_sep", "data_V_sep.csv"), delimiter=',')
        u_x_t = np.loadtxt(os.path.join("data", "lwr_sep", "data_u_sep.csv"), delimiter=',')

        # below are the whole data, not the observed.
        x_of_rho = rho_x_t[1:, 0]
        t_of_rho = rho_x_t[0, 1:]
        x_of_V = V_x_t[1:, 0]  # equals to x_of_rho
        t_of_V = V_x_t[0, 1:]
        x_of_u = u_x_t[1:, 0]
        t_of_u = u_x_t[0, 1:]
        rho = rho_x_t[1:, 1:]
        V = V_x_t[1:, 1:]
        u = u_x_t[1:, 1:]

        ## ===== hyper for # observation ====
        # change here for other observation types
        x_of_rho_obs_idx = np.arange(len(x_of_rho))
        t_of_rho_obs_idx = np.array([0,-1])
        # t_of_rho_obs_idx = np.arange(len(t_of_rho))
        x_of_V_obs_idx = np.arange(len(x_of_V))
        t_of_V_obs_idx = np.array([0,-1])
        # t_of_V_obs_idx = np.arange(len(t_of_V))
        ## ==================================
        x_of_rho_obs, t_of_rho_obs, rho_obs = get_obs(x_of_rho, t_of_rho, rho,
                                                       x_of_rho_obs_idx,
                                                       t_of_rho_obs_idx)
        x_of_V_obs, t_of_V_obs, V_obs = get_obs(x_of_V, t_of_V, V,
                                                x_of_V_obs_idx,
                                                t_of_V_obs_idx)

        ### below is the training/observed data
        self.train_data = dict()
        # initial condition
        self.train_data["x_of_rho"] = x_of_rho_obs.flatten()
        self.train_data["t_of_rho"] = t_of_rho_obs.flatten()
        self.train_data["rho"] = rho_obs.flatten()

        # terminal cost
        self.train_data["x_of_V"] = x_of_V_obs.flatten()
        self.train_data["t_of_V"] = t_of_V_obs.flatten()
        self.train_data["V"] = V_obs.flatten()

        # hyper
        self.train_data["mean_x"] = np.mean(x_of_rho)
        self.train_data["std_x"] = np.std(x_of_rho)
        self.train_data["mean_t"] = np.mean(t_of_rho)
        self.train_data["std_t"] = np.std(t_of_rho)

        ### below is the test data
        self.test_data = dict()
        xx, tt = np.meshgrid(x_of_rho, t_of_rho, indexing='ij')
        self.test_data["xdim_of_rho"] = len(x_of_rho)
        self.test_data["tdim_of_rho"] = len(t_of_rho)
        self.test_data["x_of_rho"] = xx.flatten()
        self.test_data["t_of_rho"] = tt.flatten()
        self.test_data["rho"] = rho.flatten()

        xx, tt = np.meshgrid(x_of_V, t_of_V, indexing='ij')
        self.test_data["xdim_of_V"] = len(x_of_V)
        self.test_data["tdim_of_V"] = len(t_of_V)
        self.test_data["x_of_V"] = xx.flatten()
        self.test_data["t_of_V"] = tt.flatten()
        self.test_data["V"] = V.flatten()

        xx, tt = np.meshgrid(x_of_u, t_of_u, indexing='ij')
        self.test_data["xdim_of_u"] = len(x_of_u)
        self.test_data["tdim_of_u"] = len(t_of_u)
        self.test_data["x_of_u"] = xx.flatten()
        self.test_data["t_of_u"] = tt.flatten()
        self.test_data["u"] = u.flatten()

    def construct_torch_loader(self):
        ### temporarily no need to use batch, because the data size if merely (33,33); thus we can use them all
        pass

    def get_batch(self):
        ### temporarily no need to use batch, because the data size is merely (33,33); thus we can use them all
        batch = self.train_data
        return batch

    def get_test_data(self):
        return self.test_data
