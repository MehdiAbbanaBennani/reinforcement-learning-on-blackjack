# Reinforcement-Learning
Simplified Blackjack with Reinforcement Learning, using Monte Carlo, SARSA, SARSA-landa and Action Value function approximation.
The detailed assignment steps can be found in the Assignment file.

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

### More advanced commands

Quadratic function approximation is also possible, however it is not relevant with the current binary features, you can set your own features under the Value_Function_Approximation.py file.
You only need to change the feature vector_function, and change the feature_space_size under the run.py file.

If you want to add other function approximators, you only have to add a function which returns the value of the approximator, and a function which returns the gradient of the approximator, then you can pass these functions as parameters to the sarsa lamda with function approximation algorithm.

You can also easily change the rules of the game by modifying the Game class.

Please feel free to contact me if you have any questions.

## Example output plots

Some output plots are provided under the Plots folder.

## Versioning

The latest release is v0.2. It corresponds to the Easy21 Assignment instructions, with a few additional features.

Further possible developments are :
Adding memory to the agent and considering a finite card game without resampling.