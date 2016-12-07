import numpy as np


class FunctionApproximation():
    def __init__(self, feature_space_size):
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

    def sarsa_landa_control(self):
        pass

    def value_function(self, state, action):
        return np.dot(self.theta, self.feature_vector(state=state, action=action))
