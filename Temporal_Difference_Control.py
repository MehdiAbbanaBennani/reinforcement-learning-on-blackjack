import numpy as np

from Algorithms import Algorithms


class TD_Learning(Algorithms):

    def __init__(self):
        super().__init__()


    def Sarsa(self, episodes, alpha, gamma, policy):
        state_action_value_estimation = np.random.rand(self.state_action_value_shape)

        for i in range(episodes):
            is_terminal = 0

            current_state = self.Environment.first_step()
            current_action = self.epsilon_greedy(state=current_state, policy=policy)

            while is_terminal == 0:

                new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                       scores=current_state)
                new_action = self.epsilon_greedy(state=new_state, policy=policy)

                state_action_value_estimation[current_state_value] += alpha * (reward + gamma * state_action_value_estimation[] - state_action_value_estimation[])

                current_state = new_state
                current_action = new_action

        return state_action_value_estimation

    def Sarsa_lambda(self, episodes, alpha, gamma, landa, policy ):
        state_action_value_estimation = np.random.rand(self.state_action_value_shape)
        for i in range(episodes):
            is_terminal = 0

            current_state = self.Environment.first_step()
            current_action = self.epsilon_greedy(state=current_state, policy=policy)

            while is_terminal == 0:
                new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                       scores=current_state)
                new_action = self.epsilon_greedy(state=new_state, policy=policy)

                state_action_value_estimation[] += alpha * (
                reward + gamma * state_action_value_estimation[] - state_action_value_estimation[])

                current_state = new_state
                current_action = new_action

        return state_action_value_estimation