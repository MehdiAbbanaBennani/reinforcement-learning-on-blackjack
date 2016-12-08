import numpy as np

from TD_Learning import TDLearning


class FunctionApproximation(TDLearning):

    def __init__(self, landa, N0, gamma, feature_space_size):
        super(FunctionApproximation, self).__init__(landa, N0, gamma)
        self.feature_space_size = feature_space_size
        self.theta = np.zeros(feature_space_size)

    def linear_approximation(self, state, action, theta):
        return np.dot(self.feature_vector(state=state, action=action), theta)

    @staticmethod
    def feature_vector(state, action):
        dealer_feature_vector = [3 * i < state[1] < 5 + 3 * i for i in range(3)]
        player_feature_vector = [3 * i < state[0] < 7 + 3 * i for i in range(6)]
        half_feature_vector = np.outer(dealer_feature_vector, player_feature_vector)
        half_feature_vector = half_feature_vector.flatten()
        features = np.hstack(((1 - action) * half_feature_vector, action * half_feature_vector))
        return features

    def sarsa_lambda_linear_approximation(self, episodes, landa):
        self.sarsa_lambda_initialize()

        for i in range(episodes):

            self.eligibility_trace = np.zeros(self.feature_space_size)

            current_state = self.Environment.first_step()
            epsilon = self.epsilon_t(count=self.state_visit_count[self.coord(current_state)])
            current_action = self.epsilon_greedy(state=current_state, epsilon=epsilon)
            current_state_action = self.to_state_action(action=current_action, state=current_state)

            self.state_action_visit_count[self.coord_3d(current_state_action)] += 1
            self.state_visit_count[self.coord(current_state)] += 1

            new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                   scores=current_state)

            while is_terminal == 0:
                epsilon = self.epsilon_t(count=self.state_visit_count[self.coord(new_state)])
                new_action = self.epsilon_greedy(state=new_state, epsilon=epsilon)
                new_state_action = self.to_state_action(state=new_state, action=new_action)

                delta = reward + self.gamma * self.linear_approximation(state=new_state, action=new_action, theta=self.theta)\
                        - self.linear_approximation(state=current_state, action=current_action, theta=self.theta)
                gradient = self.linear_approximation(state=current_state, action=current_action, theta=self.theta)
                self.eligibility_trace = self.gamma * landa * self.eligibility_trace + gradient

                alpha = self.alpha_t(current_state_action=current_state_action)
                self.theta += delta * np.multiply(alpha, self.eligibility_trace)
                # self.eligibility_trace = self.gamma * landa * self.eligibility_trace

                current_state = new_state.copy()
                current_action = new_action
                current_state_action = self.to_state_action(action=current_action, state=current_state)

                self.state_action_visit_count[self.coord_3d(current_state_action)] += 1
                self.state_visit_count[self.coord(current_state)] += 1
                new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                       scores=current_state)

            alpha = self.alpha_t(current_state_action=current_state_action)

            self.state_action_value_estimation[self.coord_3d(current_state_action)] += \
                alpha * (reward - self.state_action_value_estimation[self.coord_3d(current_state_action)])

    def value_function(self, state, action):
        return np.dot(self.theta, self.feature_vector(state=state, action=action))

    def learn_sarsa_landa_linear_approximation(self, episodes, landa):
        self.sarsa_lambda_linear_approximation(episodes=episodes, landa=landa)
        state_value_estimation = self.to_value_function(state_value_function=self.state_action_value_estimation)
        return state_value_estimation, self.state_action_value_estimation.argmax(axis=2), \
               self.state_action_value_estimation

    # TODO code the general approximation algorithm
    def sarsa_lambda_general_approximation(self, episodes, landa, gradient_function):
        self.sarsa_lambda_initialize()

        for i in range(episodes):

            self.eligibility_trace = np.zeros(self.feature_space_size)

            current_state = self.Environment.first_step()
            epsilon = self.epsilon_t(count=self.state_visit_count[self.coord(current_state)])
            current_action = self.epsilon_greedy(state=current_state, epsilon=epsilon)
            current_state_action = self.to_state_action(action=current_action, state=current_state)

            self.state_action_visit_count[self.coord_3d(current_state_action)] += 1
            self.state_visit_count[self.coord(current_state)] += 1

            new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                   scores=current_state)

            while is_terminal == 0:
                epsilon = self.epsilon_t(count=self.state_visit_count[self.coord(new_state)])
                new_action = self.epsilon_greedy(state=new_state, epsilon=epsilon)
                new_state_action = self.to_state_action(state=new_state, action=new_action)

                delta = reward + self.gamma * self.linear_approximation(state=new_state, action=new_action, theta=self.theta)\
                        - self.linear_approximation(state=current_state, action=current_action, theta=self.theta)

                # gradient = self.linear_approximation(state=current_state, action=current_action, theta=self.theta)
                gradient_value = gradient_function(state=current_state, action=current_action, theta=self.theta)

                self.eligibility_trace = self.gamma * landa * self.eligibility_trace + gradient_value

                alpha = self.alpha_t(current_state_action=current_state_action)
                self.theta += delta * np.multiply(alpha, self.eligibility_trace)
                # self.eligibility_trace = self.gamma * landa * self.eligibility_trace

                current_state = new_state.copy()
                current_action = new_action
                current_state_action = self.to_state_action(action=current_action, state=current_state)

                self.state_action_visit_count[self.coord_3d(current_state_action)] += 1
                self.state_visit_count[self.coord(current_state)] += 1
                new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                       scores=current_state)

            alpha = self.alpha_t(current_state_action=current_state_action)

            self.state_action_value_estimation[self.coord_3d(current_state_action)] += \
                alpha * (reward - self.state_action_value_estimation[self.coord_3d(current_state_action)])

    def linear_gradient(self, theta, state, action):
        return self.linear_approximation(state=state, action=action, theta=theta)

    def quadratic_gradient(self, theta, state, action):
        pass