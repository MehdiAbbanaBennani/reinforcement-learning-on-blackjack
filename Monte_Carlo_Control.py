import numpy as np
from Game import Game


class MonteCarlo:

    def __init__(self):
        # The rows represent the player state, and the columns the dealer's state
        self.value_function = np.zeros((21, 10))

        # The policy maps for each state the probability of hit
        self.Environment = Game()
        self.state_value_shape = (21, 10)
        self.state_action_value_shape = (21, 10, 2)
        self.policy = np.random.rand(21, 10)

    @staticmethod
    def coord(vector):
        return int(vector[0]) - 1, int(vector[1]) - 1

    def decision(self, state, policy):
        action = round(np.random.binomial(1, policy[self.coord(state)]))
        return action

    def epsilon_greedy(self, state, state_action_value, policy, epsilon):
        pick = round(np.random.binomial(1, epsilon / 2))
        if pick:
            return self.decision(state, policy)
        else:
            array = state_action_value[self.coord(state), :]
            return array.argmax()

    def run_episode_value(self, policy):
        is_terminal = 0
        states_list = []

        current_state = self.Environment.first_step()
        while is_terminal == 0:
            action = self.decision(state=current_state, policy=policy)
            new_state, reward, is_terminal = self.Environment.step(do_hit=action, scores=current_state)
            states_list.append(current_state)
            current_state = new_state

        return states_list, reward

    def run_episode_action_value(self, policy, state_action_value, epsilon):
        is_terminal = 0
        states_actions_list = []

        current_state = self.Environment.first_step()
        while is_terminal == 0:
            action = self.epsilon_greedy(state=current_state, policy=policy, state_action_value=state_action_value, epsilon=epsilon)
            new_state, reward, is_terminal = self.Environment.step(do_hit=action, scores=current_state)
            current_state_action = current_state.append(action)
            states_actions_list.append(current_state_action)
            current_state = new_state

        return states_actions_list, reward

    def every_visit_monte_carlo(self, policy, episodes):
        # Variables initialization
        state_visit_count = np.zeros(self.state_value_shape)
        state_total_return = np.zeros(self.state_value_shape)

        for i in range(episodes):
            states_list, reward = self.run_episode(policy=policy)

            for j in range(np.shape(states_list)[0]):
                current_state = states_list[i]
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
            states_actions_list, reward = self.run_episode_action_value(policy=policy,
                                                                        state_action_value=state_action_value_estimation,
                                                                        epsilon=epsilon)

            for j in range(np.shape(states_actions_list)[0]):
                current_state = states_actions_list[i]
                state_action_visit_count[self.coord(current_state)] += 1
                state_action_total_return[self.coord(current_state)] += reward


        # TODO check replacement by 1
        state_action_visit_count[state_action_visit_count == 0] = 1
        # value_estimation = state_total_return
        state_action_value_estimation = np.divide(state_action_total_return, state_action_visit_count)

        return state_action_value_estimation

    def learn(self, episodes):
        value_estimation = self.every_visit_monte_carlo(policy=self.policy, episodes=episodes)
        return value_estimation
