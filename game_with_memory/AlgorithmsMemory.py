from game_with_memory.Game_with_memory import GameMemory

import numpy as np


class AlgorithmsMemory():
    def __init__(self, N0, gamma, red_count, black_count):
        # The rows represent the player state, and the columns the dealer's state

        # The policy maps for each state the probability of hit
        self.Environment = GameMemory(black_count=black_count,
                                      red_count=red_count)
        self.state_value_shape = (21, 10)
        self.state_action_value_shape = (21, 10, 2)

        self.value_function = np.zeros(self.state_value_shape)
        self.eligibility_trace = np.zeros(self.state_value_shape)

        self.state_action_total_return = np.zeros(self.state_action_value_shape)
        self.state_action_value_estimation = np.zeros(self.state_action_value_shape)

        self.epsilon_list = []

        self.N0 = N0
        self.gamma = gamma

    @staticmethod
    def epsilon_greedy(state, epsilon, approximation_function):
        pick = round(np.random.binomial(1, epsilon / 2))
        choices_values = [approximation_function(state=state, action=0), approximation_function(state=state, action=1)]
        if pick or (choices_values[0] == 0 and choices_values[1] == 0):
            return round(np.random.binomial(1, 1 / 2))
        else:
            # The problem with this is that this always returns 0 if the values are equal, which is a problem
            # in the beginning where the value function is initialized to 0
            return choices_values.argmax()

    @staticmethod
    def to_value_function(state_value_function):
        # TODO Check if it really works
        return state_value_function.max(axis=2)

    @staticmethod
    def to_state_action(state, action):
        state2 = np.copy(state)
        state2 = np.append(state2, action)
        return state2



