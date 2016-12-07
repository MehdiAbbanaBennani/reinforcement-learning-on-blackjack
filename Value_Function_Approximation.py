import numpy as np


class FunctionApproximation():
    def __init__(self):
        pass

    def linear_approximation(self, state, action, theta):
        return np.dot(self.feature_vector(state=state, action=action), theta)

    @staticmethod
    def feature_vector(state, action):
        dealer_feature_vector = [3 * i < state[1] < 5 + 3 * i for i in range(3)]
        player_feature_vector = [3 * i < state[0] < 5 + 3 * i for i in range(6)]
        half_feature_vector = np.outer(dealer_feature_vector, player_feature_vector)
        half_feature_vector = half_feature_vector.flatten()
        features = np.hstack(((1 - action) * half_feature_vector, action * half_feature_vector))
        # TODO Remove assertion
        assert np.size(features) == 36
        return features

    def sarsa_landa_control(self):
        pass
