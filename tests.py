import numpy as np


def feature_vector(state, action):

    dealer_feature_vector = [state[1] < 5, 3 < state[1] < 8, 6 < state[1]]
    player_feature_vector = [3 * i < state[0] < 7 + 3 * i for i in range(5)]
    half_feature_vector = np.outer(dealer_feature_vector,player_feature_vector)
    half_feature_vector = half_feature_vector.flatten()
    features = np.hstack(((1 - action) * half_feature_vector, action * half_feature_vector))
    return features

print(feature_vector([1, 2], 0))

print(feature_vector([4, 5], 0))

print(1)