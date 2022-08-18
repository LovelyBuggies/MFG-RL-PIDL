import numpy as np


class RoutingModel(object):
    def __init__(self, n_node, n_edge):
        self.n_node = n_node
        self.n_edge = n_edge
        self.origin = 0
        self.destination = n_node - 1
        self.edges = np.zeros((n_edge, n_node), dtype=int)
        self.adjacency = -np.ones((n_node, n_node), dtype=int)
        self.terminal = np.zeros((n_node, 1), dtype=float)
        self.length = 1
        self.rho_init = lambda l, x: 0 * x
        self.V_terminal = lambda l, x: 0.5 * ((l <= 2) * (2 - x) + (l > 2) * (1 - x))
        self.q_in = lambda t: 0.5 * (t <= 2)
        self.V_out = lambda t: 0 * t


def load_model_braess():
    model = RoutingModel(4, 5)

    model.terminal[0] = 4
    model.terminal[1] = 2.7
    model.terminal[2] = 1.4

    model.adjacency[0, 1] = 0
    model.adjacency[0, 2] = 1
    model.adjacency[1, 3] = 2
    model.adjacency[2, 3] = 3
    model.adjacency[1, 2] = 4

    model.edges[0, 0] = 0
    model.edges[0, 1] = 1
    model.edges[1, 0] = 0
    model.edges[1, 1] = 2
    model.edges[2, 0] = 1
    model.edges[2, 1] = 3
    model.edges[3, 0] = 2
    model.edges[3, 1] = 3
    model.edges[4, 0] = 1
    model.edges[4, 1] = 2

    return model

