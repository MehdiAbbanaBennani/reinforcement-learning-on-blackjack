# Reinforcement-Learning
Simplified Blackjack with Reinforcement Learning, using Monte Carlo, SARSA, SARSA-landa and Action Value function approximation.

It corresponds to the Assignement given by Pr. David Silver at UCL. The goal is to apply reinforcement learning methods to a simple card game, similar to the Blackjack example in Sutton and Barto 5.3 – however, the rules of the card game are different and non-standard

The full assignement is available under the file Assignement.pdf

The corresponding lectures are under this link : http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python 3.5 and the numpy library.

### Running the simulations

You can change the config file parameters in order to choose the algorithm (Monte Carlo Glie, sarsa, sarsa-lambda or linear_function_approximation)

Then simply run the main.py

```
python3 main.py
```

## Example output plots

The possible plots are the value function, the decision function and the Root Mean Squared Error in function of the number of episodes.

The following pictures correspond show :

The value function for the SARSA algorithm after 10^6 episodes.

![Sarsa value](https://github.com/MehdiAB161/Reinforcement-Learning/blob/master/Plots/Sarsa_lambda_0.8_value_1e6.png?raw=true)

The decision function for the sarsa-lambda algorithm after 10^6 episodes and lamda=0.5

![SARSA decision](https://github.com/MehdiAB161/Reinforcement-Learning/blob/master/Plots/Sarsa_lambda_0.8_decision_1e6.png?raw=true)

The RMSE for the SARSA algorithm in function of the number of episodes.

![SARSA decision](https://github.com/MehdiAB161/Reinforcement-Learning/blob/master/Plots/Sarsa_decision_1e6.png?raw=true)

## Report

The full report of the assignement is availabe under the Report.pdf file.

### More advanced commands

Quadratic function approximation is also possible, however it is not relevant with the current binary features, you can set your own features under the Value_Function_Approximation.py file.
You only need to change the feature vector_function, and change the feature_space_size under the run.py file.

If you want to add other function approximators, you only have to add a function which returns the value of the approximator, and a function which returns the gradient of the approximator, then you can pass these functions as parameters to the sarsa lamda with function approximation algorithm.

You can also easily change the rules of the game by modifying the Game class.

Please feel free to contact me if you have any questions.

## Versioning

The latest release is v0.2. It corresponds to the Easy21 Assignment instructions, with a few additional features.

Further possible developments are :
Adding memory to the agent and considering a finite card game without resampling.
