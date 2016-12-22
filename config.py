parameters = {
    'algorithm': 4,
    # Please select 0 for Monte Carlo GLIE Learning
    #               1 for Sarsa
    #               2 for Sarsa-lambda
    #               3 for linear function approximation
    #               4 for quadratic function approximation, however, since the features are binary,
    #                 it is equivalent to linear approximation
    'episodes': int(1e5),
    'N0': 100,
    'landa': 0.5,  # The sarsa lambda parameter
    'gamma': 1  # The reward decay
}

plot_parameters = {
    'values': 1,
    'rmse': 0,
    'rmse_matrix': 0,
    'measure_step': 1000  # For plotting the rmse over episodes
}

rule_parameters = {
    # Please choose a multiple of 3
    'score_upper_bound': 21
}