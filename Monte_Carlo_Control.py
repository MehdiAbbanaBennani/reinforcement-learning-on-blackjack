import numpy as np
from Game import Game
from Algorithms import Algorithms

class MonteCarlo(Algorithms):

    def __init__(self):
        super().__init__()

    def every_visit_monte_carlo(self, policy, episodes):
        # Variables initialization
        state_visit_count = np.zeros(self.state_value_shape)
        state_total_return = np.zeros(self.state_value_shape)

        for i in range(episodes):
            states_list, reward = self.run_episode(policy=policy)

            for j in range(np.shape(states_list)[0]):
                current_state = states_list[j]
                state_visit_count[self.coord(current_state)] += 1
                state_total_return[self.coord(current_state)] += reward

        # TODO check replacement by 1
        state_visit_count[state_visit_count == 0] = 1
        # value_estimation = state_total_return
        value_estimation = np.divide(state_total_return, state_visit_count)

        return value_estimation

    def glie_monte_carlo(self, policy, episodes, epsilon):
        # Variables initialization
        state_action_visit_count = np.zeros(self.state_action_value_shape)
        state_action_total_return = np.zeros(self.state_action_value_shape)
        state_action_value_estimation = np.zeros(self.state_action_value_shape)

        for i in range(episodes):
            states_actions_list, reward = self.run_episode_state_action_value(policy=policy,
                                                                              state_action_value=state_action_value_estimation,
                                                                              epsilon=epsilon)

            for j in range(np.shape(states_actions_list)[0]):
                current_state_action = states_actions_list[j]
                state_action_visit_count[self.coord_3d(current_state_action)] += 1
                state_action_total_return[self.coord_3d(current_state_action)] += reward

        # TODO check replacement by 1
        state_action_visit_count[state_action_visit_count == 0] = 1
        # value_estimation = state_total_return
        state_action_value_estimation = np.divide(state_action_total_return, state_action_visit_count)

        return state_action_value_estimation

    def learn1(self, episodes):
        value_estimation = self.every_visit_monte_carlo(policy=self.policy, episodes=episodes)
        return value_estimation

    def learn2(self, episodes, epsilon):
        state_action_value_estimation = self.glie_monte_carlo(policy=self.policy, episodes=episodes, epsilon=epsilon)
        state_value_estimation = self.to_value_function(state_value_function=state_action_value_estimation)
        return state_value_estimation, state_action_value_estimation
