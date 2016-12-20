import numpy as np

from game_with_memory.AlgorithmsMemory import AlgorithmsMemory


class FunctionApproximation(AlgorithmsMemory):

    def __init__(self, landa, N0, gamma, feature_space_size):
        super(FunctionApproximation, self).__init__(landa, N0, gamma)
        self.feature_space_size = feature_space_size
        self.theta = np.zeros(feature_space_size)
        self.theta2 = np.zeros(feature_space_size)

    @staticmethod
    def feature_vector(state, action):
        dealer_feature_vector = [3 * i < state[1] < 5 + 3 * i for i in range(3)]
        player_feature_vector = [3 * i < state[0] < 7 + 3 * i for i in range(6)]
        half_feature_vector = np.outer(dealer_feature_vector, player_feature_vector)
        half_feature_vector = half_feature_vector.flatten()
        features = np.hstack(((1 - action) * half_feature_vector, action * half_feature_vector))
        return features

    @staticmethod
    def state_features(state):
        scores = state['scores']
        memory = state['full_memory']
        return scores

    def sarsa_lambda_initialize(self):
        self.state_action_value_estimation = np.zeros((21, 10, 2))
        # self.state_action_visit_count = np.zeros(self.state_action_value_shape)
        # self.state_visit_count = np.zeros(self.state_value_shape)

    def sarsa_lambda_general_approximation(self, episodes, landa, gradient_function, approximation_function, theta,
                                           epsilon, alpha):
        self.sarsa_lambda_initialize()

        for i in range(episodes):

            self.eligibility_trace = np.zeros(self.feature_space_size)

            current_state = self.Environment.first_step()
            current_state_features = self.state_features(current_state)

            # TODO variable epsilon
            # We also keep a constant epsilon
            # epsilon = self.epsilon_t(count=self.state_visit_count[self.coord(current_state_features)])
            epsilon = epsilon
            current_action = self.epsilon_greedy(state=current_state_features,
                                                 epsilon=epsilon,
                                                 approximation_function=approximation_function)
            # current_state_action = self.to_state_action(action=current_action, state=current_state_features)

            # TODO integrate state and state action count
            # self.state_action_visit_count[self.coord_3d(current_state_action)] += 1
            # self.state_visit_count[self.coord(current_state_features)] += 1

            new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                   scores=current_state['scores'])
            new_state_features = self.state_features(new_state)
            while is_terminal == 0:
                # epsilon = self.epsilon_t(count=self.state_visit_count[self.coord(new_state)])
                epsilon = epsilon
                new_action = self.epsilon_greedy(state=new_state_features,
                                                 epsilon=epsilon)

                delta = reward + self.gamma * approximation_function(state=new_state_features, action=new_action)\
                        - approximation_function(state=current_state_features, action=current_action)

                gradient_value = gradient_function(state=current_state_features, action=current_action)

                self.eligibility_trace = self.gamma * landa * self.eligibility_trace + gradient_value
                # alpha = self.alpha_t(current_state_action=current_state_action)
                alpha = alpha
                theta += delta * np.multiply(alpha, self.eligibility_trace)

                current_state = new_state.copy()
                current_state_features = self.state_features(current_state)
                current_action = new_action
                # current_state_action = self.to_state_action(action=current_action, state=current_state_features)

                # TODO integrate state and state action count
                # We can make the hypothesis that the state space is so big that we have very little chance to meet a
                # state or state action pair many times, for a first step, we neglect their count incrementation
                # self.state_action_visit_count[self.coord_3d(current_state_action)] += 1
                # self.state_visit_count[self.coord(current_state)] += 1
                new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                       scores=current_state['scores'])

            delta = reward - approximation_function(state=current_state_features, action=current_action)
            gradient_value = gradient_function(state=current_state_features, action=current_action)
            self.eligibility_trace = self.gamma * landa * self.eligibility_trace + gradient_value
            # alpha = self.alpha_t(current_state_action=current_state_action)
            alpha = alpha
            theta += delta * np.multiply(alpha, self.eligibility_trace)

    @staticmethod
    def estimate_state_action_value(approximation_function):
        player_states = np.arange(1, 22)
        dealer_states = np.arange(1, 11)
        # all_states = [(x, y) for x in player_states for y in dealer_states]
        all_actions = [0, 1]
        return np.asarray([[[approximation_function(state=[player_state, dealer_state], action=action) for action in all_actions] for dealer_state in dealer_states] for player_state in player_states])

    def linear_approximation(self, state, action):
        return np.dot(self.feature_vector(state=state, action=action), self.theta)

    def quadratic_approximation(self, state, action):
        feature_vector = self.feature_vector(state=state, action=action)
        return np.dot(self.theta2, np.multiply(feature_vector, feature_vector))

    def linear_gradient(self, state, action):
        return self.feature_vector(state=state, action=action)

    def quadratic_gradient(self, state, action):
        feature_vector = self.feature_vector(state=state, action=action)
        return np.multiply(feature_vector, feature_vector)

    def learn_sarsa_landa_general_approximation(self, episodes, landa, gradient_function, approximation_function, theta):
        self.sarsa_lambda_general_approximation(episodes=episodes,
                                                landa=landa,
                                                gradient_function=gradient_function,
                                                approximation_function=approximation_function,
                                                theta=theta)
        self.state_action_value_estimation = self.estimate_state_action_value(approximation_function=approximation_function)
        state_value_estimation = self.to_value_function(state_value_function=self.state_action_value_estimation)
        output = {'state_value': state_value_estimation,
                  'decision': self.state_action_value_estimation.argmax(axis=2),
                  'state_action_value': self.state_action_value_estimation
                  }
        return output
