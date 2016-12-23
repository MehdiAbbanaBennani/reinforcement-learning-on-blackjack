from MonteCarlo_Learning import MonteCarlo
from TD_Learning import TDLearning
from Value_Function_Approximation import FunctionApproximation
from Functions import plot_results

from Functions import rmse_by_episodes
from Functions import plot_rmse_matrix
from Functions import rmse_over_landas

import matplotlib.pyplot as plt
import numpy as np


def run(parameters, plot_parameters, rules_parameters):
    # Parameters
    episodes = parameters['episodes']
    N0 = parameters['N0']
    landa = parameters['landa']
    gamma = parameters['gamma']
    measure_step = plot_parameters['measure_step']
    landa_array = np.arange(0, 1, 0.1)
    score_upper_bound = rules_parameters['score_upper_bound']
    feature_space_size = 3 * 2 * (int(score_upper_bound / 3) - 1)

    # Running the algorithm
    print("Computing the optimal value function ... ")
    if parameters['algorithm'] == 0:

        Smart_Agent_1 = MonteCarlo(N0=N0, gamma=gamma, score_upper_bound=score_upper_bound)
        output = Smart_Agent_1.learn_glie(episodes=episodes)

    elif parameters['algorithm'] == 1:
        landa = 0
        Smart_Agent_2 = TDLearning(landa=landa, gamma=gamma, N0=N0, score_upper_bound=score_upper_bound)
        output = Smart_Agent_2.learn_sarsa(episodes=episodes)

    elif parameters['algorithm'] == 2:
        Smart_Agent_2 = TDLearning(landa=landa, gamma=gamma, N0=N0, score_upper_bound=score_upper_bound)
        output = Smart_Agent_2.learn_sarsa_landa(episodes=episodes, landa=landa)

    elif parameters['algorithm'] == 3:
        Smart_Agent_3 = FunctionApproximation(landa=landa,
                                              gamma=gamma,
                                              N0=N0,
                                              feature_space_size=feature_space_size,
                                              score_upper_bound=score_upper_bound)
        output = Smart_Agent_3.learn_sarsa_landa_general_approximation(episodes=episodes,
                                                                       landa=landa,
                                                                       gradient_function=Smart_Agent_3.linear_gradient,
                                                                       approximation_function=Smart_Agent_3.linear_approximation)
    elif parameters['algorithm'] == 4:
        Smart_Agent_4 = FunctionApproximation(landa=landa,
                                              gamma=gamma,
                                              N0=N0,
                                              feature_space_size=feature_space_size,
                                              score_upper_bound=score_upper_bound)

        output = Smart_Agent_4.learn_sarsa_landa_general_approximation(episodes=episodes,
                                                                       landa=landa,
                                                                       gradient_function=Smart_Agent_4.quadratic_gradient,
                                                                       approximation_function=Smart_Agent_4.quadratic_approximation)
    print("Done!")

    # Plots
    if plot_parameters['values'] == 1:
        plot_results(value=output['state_value'], state_decision=output['decision'])

    # RMSE over episodes plot
    if plot_parameters['rmse'] == 1 and parameters['algorithm'] > 0:
        print("Computing the RMSE over episodes ...")
        rmse_array = rmse_by_episodes(episodes=episodes,
                                      landa=landa,
                                      gamma=gamma,
                                      N0=N0,
                                      measure_step=measure_step,
                                      algorithm=parameters['algorithm'],
                                      feature_space_size=feature_space_size,
                                      score_upper_bound=score_upper_bound)
        abs = np.arange(0, episodes, measure_step)
        plt.plot(abs, rmse_array)
        print("Done!")
        plt.show()

    # RMSE matrix plot
    if plot_parameters['rmse_matrix'] == 1 and parameters['algorithm'] > 1 :
        print("Computing the RMSE matrix, this may take a lot of time ...")
        rmse_matrix = rmse_over_landas(episodes=episodes,
                                       landa_array=landa_array,
                                       gamma=gamma,
                                       N0=N0,
                                       measure_step=measure_step,
                                       algorithm=parameters['algorithm'],
                                       score_upper_bound=score_upper_bound,
                                       feature_space_size=feature_space_size)
        print("Done!")
        plot_rmse_matrix(rmse_matrix)
