from Game import Game

import numpy as np


class Algorithms():
    def __init__(self, N0, gamma):
        # The rows represent the player state, and the columns the dealer's state

        # The policy maps for each state the probability of hit
        self.Environment = Game()
        self.state_value_shape = (21, 10)
        self.state_action_value_shape = (21, 10, 2)

        self.policy = np.random.rand(21, 10)

        self.value_function = np.zeros(self.state_value_shape)
        self.eligibility_trace = np.zeros(self.state_value_shape)
        self.state_visit_count = np.zeros(self.state_value_shape)

        self.state_action_visit_count = np.zeros(self.state_action_value_shape)
        self.state_action_total_return = np.zeros(self.state_action_value_shape)
        self.state_action_value_estimation = np.zeros(self.state_action_value_shape)

        self.epsilon_list = []

        self.N0 = N0
        self.gamma = gamma

    @staticmethod
    def coord(vector):
        return int(vector[0]) - 1, int(vector[1]) - 1

    @staticmethod
    def coord_3d(vector):
        return int(vector[0]) - 1, int(vector[1]) - 1, int(vector[2])

    @staticmethod
    def coord_3d_2(state, action):
        return int(state[0]) - 1, int(state[1]) - 1, int(action)

    def random_policy(self, state, policy):
        action = round(np.random.binomial(1, policy[self.coord(state)]))
        return action

    def epsilon_greedy(self, state, epsilon):
        pick = round(np.random.binomial(1, epsilon / 2))
        if pick:
            return round(np.random.binomial(1, 1 / 2))
        else:
            return self.state_action_value_estimation[int(state[0]) - 1, int(state[1]) - 1, :].argmax()

    @staticmethod
    def to_value_function(state_value_function):
        # TODO Check if it really works
        return state_value_function.max(axis=2)

    def epsilon_t(self, current_state):
        return self.N0 / (self.N0 + self.state_visit_count[self.coord(current_state)])

    def alpha_t(self, current_state):
        return 1 / (self.state_visit_count[self.coord(current_state)] + 1)

    @staticmethod
    def to_state_action(state, action):
        state2 = np.copy(state)
        state2 = np.append(state2, action)
        return state2
