import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from MonteCarlo_Learning import MonteCarlo
from TD_Learning import TDLearning
from Value_Function_Approximation import FunctionApproximation


def plot_results(value, state_decision):

    x = range(np.shape(value)[1])
    y = range(np.shape(value)[0])

    xs, ys = np.meshgrid(x, y)
    zs = value
    zs2 = state_decision

    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    ax1.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='hot')
    plt.show()

    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    ax2.plot_surface(xs, ys, zs2, rstride=1, cstride=1, cmap='hot')
    plt.show()

def plot_rmse_matrix(rmse_matrix):
    x = range(np.shape(rmse_matrix)[1])
    y = range(np.shape(rmse_matrix)[0])

    xs, ys = np.meshgrid(x, y)
    zs = rmse_matrix

    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    ax1.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='hot')
    plt.show()


def rmse_by_episodes(episodes, landa, gamma, N0, measure_step, algorithm, feature_space_size, score_upper_bound):
    Smart_Agent_1 = MonteCarlo(N0=N0, gamma=gamma, score_upper_bound=score_upper_bound)

    output_mc = Smart_Agent_1.learn_glie(episodes=episodes)
    state_action_value_mc = output_mc['state_action_value']

    if algorithm == 1:
        Smart_Agent_2 = TDLearning(landa=landa, gamma=gamma, N0=N0, score_upper_bound=score_upper_bound)
        rmse_array = Smart_Agent_2.rmse_sarsa(measure_step=measure_step,
                                              episodes=episodes,
                                              state_action_value_mc=state_action_value_mc)
    elif algorithm == 2:
        SmartAgent = TDLearning(landa=landa, gamma=gamma, N0=N0, score_upper_bound=score_upper_bound)
        rmse_array = SmartAgent.rmse_sarsa_landa(landa=landa,
                                                 measure_step=measure_step,
                                                 episodes=episodes,
                                                 state_action_value_mc=state_action_value_mc)
    elif algorithm == 3:
        SmartAgent = FunctionApproximation(landa=landa,
                                           N0=N0,
                                           gamma=gamma,
                                           feature_space_size=feature_space_size,
                                           score_upper_bound=score_upper_bound)
        rmse_array = SmartAgent.rmse_general_approximation(landa=landa,
                                                           measure_step=measure_step,
                                                           episodes=episodes,
                                                           state_action_value_mc=state_action_value_mc,
                                                           approximation_function=SmartAgent.linear_approximation,
                                                           gradient_function=SmartAgent.linear_gradient)
    elif algorithm == 4:
        SmartAgent = FunctionApproximation(landa=landa,
                                           N0=N0,
                                           gamma=gamma,
                                           feature_space_size=feature_space_size,
                                           score_upper_bound=score_upper_bound)
        rmse_array = SmartAgent.rmse_general_approximation(landa=landa,
                                                           measure_step=measure_step,
                                                           episodes=episodes,
                                                           state_action_value_mc=state_action_value_mc,
                                                           approximation_function=SmartAgent.quadratic_approximation,
                                                           gradient_function=SmartAgent.quadratic_gradient)
    return rmse_array


def rmse_over_landas(episodes, landa_array, gamma, N0, measure_step, algorithm, feature_space_size, score_upper_bound):
    size = int(episodes/measure_step)
    rmse_matrix = np.empty((0,size))
    for landa in landa_array:
        rmse_array = rmse_by_episodes(episodes=episodes,
                                      landa=landa,
                                      gamma=gamma,
                                      N0=N0,
                                      measure_step=measure_step,
                                      algorithm=algorithm,
                                      feature_space_size=feature_space_size,
                                      score_upper_bound=score_upper_bound)
        rmse_matrix = np.vstack((rmse_matrix, rmse_array))
    return rmse_matrix

