# Reinforcement-Learning
Simplified Blackjack with Reinforcement Learning, using Monte Carlo, SARSA-landa and Action Value function approximation.
The detailed assignement steps can be found in the Assignement file.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python 3.5 and the numpy library.

## Running the simulations

In order to run a simulation, you can run the command :

### Break down into end to end tests

You can change the config file parameters in order to choose the algorithm (Monte Carlo Glie, sarsa, sarsa-lambda or linear_function_approximation)

Then simply run the main.py

```
python3 main.py
```

## More advanced commands

Quadratic function approximation is also possible, however it is not relevant with the current binary features, you can set your own features under the Value_Function_Approximation.py file.
You only need to change the feature vector_function, and change the feature_space_size under the run.py file.

Please feel free to contact me if you have any questions.

### Example output plots

Some output plots are provided under the Plots folder.

## Versioning

The latest release is v0.2. It corresponds to the Easy21 Assignement instructions, and a few additional features.

Further planned developements are :
Adding memory to the agent and considering a finite card game without resampling.
