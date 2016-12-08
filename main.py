from MonteCarlo_Learning import MonteCarlo
from TD_Learning import TDLearning
from Value_Function_Approximation import FunctionApproximation
from Functions import plot_results
from Functions import rmse_by_episodes
from Functions import plot_rmse_matrix
from Functions import rmse_over_landas

import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Parameters
episodes = int(1e6)
N0 = 100
landa = 0.8
gamma = 1
measure_step = 1000
landa_array = np.arange(0, 1, 0.1)

Smart_Agent_1 = MonteCarlo(N0=N0, gamma=gamma)
Smart_Agent_2 = TDLearning(landa=landa, gamma=gamma, N0=N0)
Smart_Agent_3 = FunctionApproximation(landa=landa, gamma=gamma, N0=N0, feature_space_size=36)


# value_mc, state_decision_mc, state_action_value_mc = Smart_Agent_1.learn_glie(episodes=episodes)
# value_sarsa, state_decision_sarsa, state_action_value_sarsa = Smart_Agent_2.learn_sarsa(episodes=episodes)
# value_sarsa_lambda, state_decision_sarsa_lambda, state_action_value_sarsa_lambda = Smart_Agent_2.learn_sarsa_landa(episodes=episodes, landa = landa)
# value_sarsa_lambda_linear_appr, state_decision_sarsa_lambda_linear_appr, state_action_value_sarsa_lambda_linear_appr = Smart_Agent_3.learn_sarsa_landa_linear_approximation(episodes=episodes, landa=landa)

# plot_results(value=value_sarsa_lambda_linear_appr, state_decision=state_decision_sarsa_lambda_linear_appr)

# rmse_array = rmse_by_episodes(episodes=episodes, landa=landa, gamma=gamma, N0=N0, measure_step=measure_step)
# abs = np.arange(0, episodes, measure_step)
# plt.plot(abs, rmse_array)
# plt.show()

rmse_matrix = rmse_over_landas(episodes=episodes,
                               landa_array=landa_array,
                               gamma=gamma,
                               N0=N0,
                               measure_step=measure_step)
plot_rmse_matrix(rmse_matrix)


print("--- %s seconds ---" % (time.time() - start_time))
