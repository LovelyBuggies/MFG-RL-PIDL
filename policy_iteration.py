import numpy as np
import random
import torch
from torch import nn
from utils import plot_rho

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 1)
        )

    def forward(self, X):
        return self.model(X)


def policy_iteration(rho):
    # set up
    episode = 50
    epsilon = 1
    gamma = 1
    cell = rho.shape[0]
    t_num = int(rho.shape[1] / rho.shape[0])
    delta_t = 1
    delta_x = 1 / cell

    # get all possibles
    all_possible_states = set()
    all_possible_actions = set()
    for t in np.arange(t_num * cell):
        if t == 0:
            all_possible_states.add((0, 0))
            all_possible_actions.add(0.)
            continue

        for x in np.arange(cell):
            all_possible_states.add((x * delta_x, t * delta_t))
            all_possible_actions.add(x * delta_x / delta_t)

    print(len(all_possible_actions))
    print(len(all_possible_states))

    # initialization
    v = dict()
    policy = dict()
    for state in all_possible_states:
        v[state] = 0
        policy[state] = random.sample(all_possible_actions, 1)[0]

    for loop in range(episode):
        print(f"PI Loop: {loop}")
        # policy evaluation
        while True:
            max_delta = 0
            for state in all_possible_states:
                if state[1] < cell * t_num - 1:
                    u = policy[state]
                    r = (0.5 * u * u + rho[int(state[0])][int(state[1] / delta_t)] + 1) * delta_t
                    state_next = (max(min(state[0] + u * delta_t, (cell - 1) / cell), 0), state[1] + delta_t)
                    max_delta = max(max_delta, abs(v[state] - r - gamma * v[state_next]))
                    v[state] = r + gamma * v[state_next]

            if max_delta < epsilon:
                break

        # policy improvement
        converge = True
        for state in all_possible_states:
            if state[1] < cell * t_num - 1:
                u_tmp = policy[state]
                best_u = None
                best_v = float('-inf')
                for u in all_possible_actions:
                    r = (0.5 * u * u + rho[int(state[0])][int(state[1] / delta_t)] + 1) * delta_t
                    state_next = (max(min(state[0] + u * delta_t, (cell - 1) / cell), 0), state[1] + delta_t)
                    value = r + gamma * v[state_next]
                    if value > best_v:
                        best_v = value
                        best_u = u

                if best_u != u_tmp:
                    converge = False

        if converge:
            break

    u_matrix = np.zeros((cell, t_num * cell))
    v_matrix = np.zeros((cell, t_num * cell))
    for t in range(t_num * cell):
        if t == 0:
            for x in range(cell):
                u_matrix[x][0] = policy[(0., 0.)]
                v_matrix[x][0] = v[(0, 0)]

        else:
            for x in range(cell):
                u_matrix[x][t] = policy[(x * delta_x, t * delta_t)]
                v_matrix[x][t] = v[(x * delta_x, t * delta_t)]

    file = open("data/u_matrix.txt", "w")
    for row in u_matrix:
        np.savetxt(file, row)
    file.close()
    plot_rho(cell, t_num, u_matrix, None)
    file = open("data/v_matrix.txt", "w")
    for row in v_matrix:
        np.savetxt(file, row)
    file.close()
    return u_matrix, v_matrix


def policy_iteration_critic(rho):
    # set up
    episode = 10
    epsilon = 1
    gamma = 1
    cell = rho.shape[0]
    t_num = int(rho.shape[1] / rho.shape[0])
    delta_t = 1
    delta_x = 1 / cell

    # get all possibles
    all_possible_states = set()
    all_possible_actions = set()
    for t in np.arange(t_num * cell):
        if t == 0:
            all_possible_states.add((np.double(0.), np.double(0.)))
            all_possible_actions.add(0.)
            continue

        for x in np.arange(cell):
            all_possible_states.add((x * delta_x, t * delta_t))
            all_possible_actions.add(x * delta_x / delta_t)

    print(all_possible_actions)
    print(all_possible_states)

    # initialization
    v = Critic(2).double()
    v_optimizer = torch.optim.Adam(v.parameters(), lr=1e-3)
    policy = dict()
    for state in all_possible_states:
        policy[state] = random.sample(all_possible_actions, 1)[0]

    for loop in range(episode):
        print(f"PI Loop: {loop}")
        # policy evaluation
        while True:
            max_delta = 0
            for state in all_possible_states:
                if state[1] < cell * t_num - 1:
                    u = policy[state]
                    r = (0.5 * u * u + rho[int(state[0])][int(state[1] / delta_t)] + 1) * delta_t
                    state_next = (max(min(state[0] + u * delta_t, np.double((cell - 1) / cell)), 0.), np.double(state[1] + delta_t))
                    max_delta = max(max_delta, abs(v(torch.tensor(state)) - r - gamma * v(torch.tensor(state_next))))
                    loss = v(torch.tensor(state)) - (r + gamma * v(torch.tensor(state_next)))
                    v_loss = loss.pow(2).mean()
                    v_optimizer.zero_grad()
                    v_loss.backward()
                    v_optimizer.step()

            if max_delta < epsilon:
                break

        # policy improvement
        converge = True
        for state in all_possible_states:
            if state[1] < cell * t_num - 1:
                u_tmp = policy[state]
                best_u = None
                best_v = float('-inf')
                for u in all_possible_actions:
                    r = (0.5 * u * u + rho[int(state[0])][int(state[1] / delta_t)] + 1) * delta_t
                    state_next = (max(min(state[0] + u * delta_t, np.double((cell - 1) / cell)), np.double(0)), np.double(state[1] + delta_t))
                    value = r + gamma * v(torch.tensor(state_next))
                    if value > best_v:
                        best_v = value
                        best_u = u

                if best_u != u_tmp:
                    converge = False

        if converge:
            break

    u_matrix = np.zeros((cell, t_num * cell))
    v_matrix = np.zeros((cell, t_num * cell))
    for t in range(t_num * cell):
        if t == 0:
            for x in range(cell):
                u_matrix[x][0] = policy[(0., 0.)]
                v_matrix[x][0] = np.double(v(torch.tensor((np.double(0.), np.double(0.)))))

        else:
            for x in range(cell):
                u_matrix[x][t] = policy[(x * delta_x, t * delta_t)]
                v_matrix[x][t] = np.double(v(torch.tensor((np.double(x * delta_x), np.double(t * delta_t)))))

    file = open("data/u_matrix.txt_critic", "w")
    for row in u_matrix:
        np.savetxt(file, row)
    file.close()

    plot_rho(cell, t_num, u_matrix, None)
    file = open("data/v_matrix.txt_critic", "w")
    for row in v_matrix:
        np.savetxt(file, row)
    file.close()
    return u_matrix, v_matrix