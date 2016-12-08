import numpy as np
from Algorithms import Algorithms


class MonteCarlo(Algorithms):
    def __init__(self, N0, gamma):
        super().__init__(N0, gamma=gamma)

    def glie_monte_carlo(self, episodes):
        # State value function initialization
        self.state_action_value_estimation = np.zeros(self.state_action_value_shape)

        for i in range(episodes):
            # State action count initialization
            state_action_visit_count = np.zeros(self.state_action_value_shape)

            # Running an episode
            states_actions_list, reward = self.run_episode_state_action_value()

            for j in range(np.shape(states_actions_list)[0]):
                current_state_action = states_actions_list[j]
                # This one is for the epsilon decay
                self.state_action_visit_count[self.coord_3d(current_state_action)] += 1
                # TODO Recheck the 0,2
                self.state_visit_count[self.coord(current_state_action[0:2])] += 1

                state_action_visit_count[self.coord_3d(current_state_action)] += 1

                # Here Gamma = 1 and the reward is only in the terminal episode, therefore the
                # invrement is simplified like this

                N = self.state_action_visit_count[self.coord_3d(current_state_action)]
                self.state_action_value_estimation[self.coord_3d(current_state_action)] += \
                    (reward - self.state_action_value_estimation[self.coord_3d(current_state_action)]) / N

    def run_episode_state_action_value(self):
        is_terminal = 0
        states_actions_list = []

        current_state = self.Environment.first_step()
        while is_terminal == 0:

            epsilon = self.epsilon_t(count=self.state_visit_count[self.coord(current_state)])
            self.epsilon_list.append(epsilon)

            action = self.epsilon_greedy(state=current_state, epsilon=epsilon)

            new_state, reward, is_terminal = self.Environment.step(do_hit=action,
                                                                   scores=current_state)

            current_state_action = self.to_state_action(state=current_state, action=action)
            states_actions_list.append(current_state_action)
            current_state = new_state

        return states_actions_list, reward

    def learn_glie(self, episodes):
        self.glie_monte_carlo(episodes=episodes)
        state_value_estimation = self.to_value_function(state_value_function=self.state_action_value_estimation)
        return state_value_estimation, self.state_action_value_estimation.argmax(axis=2), \
               self.state_action_value_estimation

    def every_visit(self, episodes):
        value_estimation = self.every_visit_monte_carlo(episodes=episodes)
        return value_estimation

    def every_visit_monte_carlo(self, episodes):
        # Variables initialization
        state_visit_count = np.zeros(self.state_value_shape)
        state_total_return = np.zeros(self.state_value_shape)

        for i in range(episodes):
            states_list, reward = self.run_episode_state_value()

            for j in range(np.shape(states_list)[0]):
                current_state = states_list[j]
                state_visit_count[self.coord(current_state)] += 1
                state_total_return[self.coord(current_state)] += reward

        # TODO check replacement by 1
        state_visit_count[state_visit_count == 0] = 1
        # value_estimation = state_total_return
        value_estimation = np.divide(state_total_return, state_visit_count)

        return value_estimation

    def run_episode_state_value(self):
        is_terminal = 0
        states_list = []

        current_state = self.Environment.first_step()
        while is_terminal == 0:
            action = self.epsilon_greedy(state=current_state, epsilon=self.epsilon_t(current_state=current_state))
            new_state, reward, is_terminal = self.Environment.step(do_hit=action, scores=current_state)
            states_list.append(current_state)
            current_state = new_state

        return states_list, reward