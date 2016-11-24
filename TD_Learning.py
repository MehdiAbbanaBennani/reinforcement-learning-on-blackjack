import numpy as np

from Algorithms import Algorithms


class TDLearning(Algorithms):

    def __init__(self, landa, N0, gamma):
        super().__init__(N0=N0, gamma=gamma)
        self.landa = landa

    def sarsa_initialize(self):
        self.state_action_value_estimation = np.zeros((21, 10, 2))
        self.state_action_visit_count = np.zeros(self.state_action_value_shape)
        self.state_visit_count = np.zeros(self.state_value_shape)

    #     TODO zero terminal state

    def sarsa_lambda_initialize(self):
        self.state_action_value_estimation = np.random.rand(21, 10, 2)
    #     TODO zero terminal state

    def sarsa(self, episodes):
        self.sarsa_initialize()

        for i in range(episodes):
            # print(i, '/', episodes)

            current_state = self.Environment.first_step()
            epsilon = self.epsilon_t(count=self.state_visit_count[self.coord(current_state)])

            self.epsilon_list.append(epsilon)

            current_action = self.epsilon_greedy(state=current_state, epsilon=epsilon)
            current_state_action = self.to_state_action(state=current_state, action=current_action)

            self.state_action_visit_count[self.coord_3d(current_state_action)] += 1
            self.state_visit_count[self.coord(current_state)] += 1

            new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                   scores=current_state)
            if is_terminal == 1:
                current_state_action = self.to_state_action(action=current_action, state=current_state)
                alpha = self.alpha_t(current_state_action=current_state_action)

                self.state_action_value_estimation[self.coord_3d_2(current_state, current_action)] += alpha * (reward
                            - self.state_action_value_estimation[self.coord_3d_2(current_state, current_action)])

            while is_terminal == 0:
                epsilon = self.epsilon_t(count=self.state_visit_count[self.coord(new_state)])
                new_action = self.epsilon_greedy(state=new_state, epsilon=epsilon)
                current_state_action = self.to_state_action(action=current_action, state=current_state)
                alpha = self.alpha_t(current_state_action=current_state_action)

                self.state_action_value_estimation[self.coord_3d_2(current_state, current_action)] += alpha * (reward +
                            self.gamma * self.state_action_value_estimation[self.coord_3d_2(new_state, new_action)]
                         - self.state_action_value_estimation[self.coord_3d_2(current_state, current_action)])

                current_state = new_state
                current_action = new_action

                new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                       scores=current_state)

    def sarsa_lambda(self, episodes, landa):
        self.sarsa_lambda_initialize()

        for i in range(episodes):
            print(i, '/', episodes)

            self.eligibility_trace = np.zeros(self.state_action_value_shape)

            current_state = self.Environment.first_step()
            epsilon = self.epsilon_t(current_state=current_state)
            current_action = self.epsilon_greedy(state=current_state, epsilon=epsilon)

            # TODO Check again the algorithm

            new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                   scores=current_state)
            print(is_terminal)

            while is_terminal == 0:
                epsilon = self.epsilon_t(current_state=new_state)
                new_action = self.epsilon_greedy(state=new_state, epsilon=epsilon)
                delta = reward + self.gamma * self.state_action_value_estimation[self.coord_3d_2(new_state, new_action)]\
                        - self.state_action_value_estimation[self.coord_3d_2(current_state, current_action)]
                self.eligibility_trace[self.coord(current_state)] += 1

                alpha = self.alpha_t(current_state=current_state)
                self.state_action_value_estimation += alpha * delta * self.eligibility_trace
                self.eligibility_trace = self.gamma * landa * self.eligibility_trace

                current_state = new_state
                current_action = new_action
                new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                       scores=current_state)

    def learn_sarsa(self, episodes):
        self.sarsa(episodes=episodes)
        state_value_estimation = self.to_value_function(state_value_function=self.state_action_value_estimation)
        return state_value_estimation, self.state_action_value_estimation.argmax(axis=2)

    def learn_sarsa_landa(self, episodes, landa):
        self.sarsa_lambda(episodes=episodes, landa=landa)
        state_value_estimation = self.to_value_function(state_value_function=self.state_action_value_estimation)
        return state_value_estimation, self.state_action_value_estimation.argmax(axis=2)
