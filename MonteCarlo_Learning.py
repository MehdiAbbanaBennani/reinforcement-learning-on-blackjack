import numpy as np
from Algorithms import Algorithms


class MonteCarlo(Algorithms):
    def __init__(self, N0, gamma):
        super().__init__(N0, gamma=gamma)

    def every_visit_monte_carlo(self, policy, episodes):
        # Variables initialization
        state_visit_count = np.zeros(self.state_value_shape)
        state_total_return = np.zeros(self.state_value_shape)

        for i in range(episodes):
            states_list, reward = self.run_episode_state_value(policy=policy)

            for j in range(np.shape(states_list)[0]):
                current_state = states_list[j]
                state_visit_count[self.coord(current_state)] += 1
                state_total_return[self.coord(current_state)] += reward

        # TODO check replacement by 1
        state_visit_count[state_visit_count == 0] = 1
        # value_estimation = state_total_return
        value_estimation = np.divide(state_total_return, state_visit_count)

        return value_estimation

    def glie_monte_carlo(self, episodes):

        for i in range(episodes):
            if i % 40 == 0:
                print(i, '/', episodes)

            states_actions_list, reward = self.run_episode_state_action_value()

            for j in range(np.shape(states_actions_list)[0]):
                current_state_action = states_actions_list[j]
                self.state_action_visit_count[self.coord_3d(current_state_action)] += 1
                self.state_action_total_return[self.coord_3d(current_state_action)] += reward

        # TODO check replacement by 1
        self.state_action_visit_count[self.state_action_visit_count == 0] = 1
        self.state_action_value_estimation = np.divide(self.state_action_total_return, self.state_action_visit_count)

    def run_episode_state_value(self, policy):
        is_terminal = 0
        states_list = []

        current_state = self.Environment.first_step()
        while is_terminal == 0:
            action = self.random_policy(state=current_state, policy=policy)
            new_state, reward, is_terminal = self.Environment.step(do_hit=action, scores=current_state)
            states_list.append(current_state)
            current_state = new_state

        return states_list, reward

    def run_episode_state_action_value(self):
        is_terminal = 0
        states_actions_list = []

        current_state = self.Environment.first_step()
        while is_terminal == 0:
            epsilon = self.epsilon_t(current_state=current_state)
            action = self.epsilon_greedy(state=current_state, epsilon=epsilon)
            new_state, reward, is_terminal = self.Environment.step(do_hit=action,
                                                                   scores=current_state)
            current_state.append(action)
            current_state_action = current_state
            states_actions_list.append(current_state_action)
            current_state = new_state

        return states_actions_list, reward

    def learn1(self, episodes):
        value_estimation = self.every_visit_monte_carlo(policy=self.policy, episodes=episodes)
        return value_estimation

    def learn2(self, episodes):
        self.glie_monte_carlo(episodes=episodes)
        state_value_estimation = self.to_value_function(state_value_function=self.state_action_value_estimation)
        return state_value_estimation, self.state_action_value_estimation
