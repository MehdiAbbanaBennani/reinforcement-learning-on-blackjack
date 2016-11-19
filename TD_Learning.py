import numpy as np

from Algorithms import Algorithms


class TD_Learning(Algorithms):

    def __init__(self, landa, N0):
        super().__init__(N0=N0)
        self.landa = landa

    def sarsa_initialize(self):
        self.state_action_value_estimation = np.random.rand(self.state_action_value_shape)
    #     TODO zero terminal state

    def sarsa_lambda_initialize(self):
        self.state_action_value_estimation = np.random.rand(self.state_action_value_shape)
    #     TODO zero terminal state

    def sarsa(self, episodes):
        self.sarsa_initialize()

        for i in range(episodes):
            is_terminal = 0

            current_state = self.Environment.first_step()
            current_action = self.epsilon_greedy(state=current_state)

            while is_terminal == 0:

                new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                       scores=current_state)
                new_action = self.epsilon_greedy(state=new_state)
                alpha = self.alpha_t(current_state=current_state)
                self.state_action_value_estimation[current_state[0], current_state[1], current_action] += alpha * (reward +
                        self.gamma * self.state_action_value_estimation[new_state[0], new_state[1], new_action]
                         - self.state_action_value_estimation[current_state[0], current_state[1], current_action])

                current_state = new_state
                current_action = new_action

    def sarsa_lambda(self, episodes, landa):
        self.sarsa_lambda_initialize()

        for i in range(episodes):
            is_terminal = 0

            self.eligibility_trace = np.zeros(self.state_action_value_shape)
            current_state = self.Environment.first_step()
            current_action = self.epsilon_greedy(state=current_state)

            while is_terminal == 0:

                new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                       scores=current_state)
                new_action = self.epsilon_greedy(state=new_state)
                delta = reward + self.gamma * self.state_action_value_estimation[self.coord(new_state), new_action]\
                        - self.state_action_value_estimation[self.coord(current_state), current_action]
                self.eligibility_trace[self.coord(current_state), current_action] += 1

                alpha = self.alpha_t(current_state=current_state)
                self.state_action_value_estimation += alpha * delta * self.eligibility_trace
                self.eligibility_trace = self.gamma * landa * self.eligibility_trace

                current_state = new_state
                current_action = new_action
