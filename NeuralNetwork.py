import numpy as np
import tensorflow as tf

from Value_Function_Approximation import FunctionApproximation


class NeuralNetworkApproximation(FunctionApproximation):

    def __init__(self, landa, N0, gamma, feature_space_size):
        super(NeuralNetworkApproximation, self).__init__(landa, N0, gamma)


    def sarsa_lambda_general_approximation(self, episodes, landa, gradient_function, approximation_function, theta):
        self.sarsa_lambda_initialize()

        for i in range(episodes):

            self.eligibility_trace = np.zeros(self.feature_space_size)

            current_state = self.Environment.first_step()
            epsilon = self.epsilon_t(count=self.state_visit_count[self.coord(current_state)])
            current_action = self.epsilon_greedy(state=current_state, epsilon=epsilon)
            current_state_action = self.to_state_action(action=current_action, state=current_state)

            self.state_action_visit_count[self.coord_3d(current_state_action)] += 1
            self.state_visit_count[self.coord(current_state)] += 1

            new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                   scores=current_state)

            while is_terminal == 0:
                epsilon = self.epsilon_t(count=self.state_visit_count[self.coord(new_state)])
                new_action = self.epsilon_greedy(state=new_state, epsilon=epsilon)
                new_state_action = self.to_state_action(state=new_state, action=new_action)

                delta = reward + self.gamma * approximation_function(state=new_state, action=new_action)\
                        - approximation_function(state=current_state, action=current_action)

                # Gradient to compute with tensorflow
                gradient_value = gradient_function(state=current_state, action=current_action)

                self.eligibility_trace = self.gamma * landa * self.eligibility_trace + gradient_value
                alpha = self.alpha_t(current_state_action=current_state_action)
                theta += delta * np.multiply(alpha, self.eligibility_trace)

                current_state = new_state.copy()
                current_action = new_action
                current_state_action = self.to_state_action(action=current_action, state=current_state)

                self.state_action_visit_count[self.coord_3d(current_state_action)] += 1
                self.state_visit_count[self.coord(current_state)] += 1
                new_state, reward, is_terminal = self.Environment.step(do_hit=current_action,
                                                                       scores=current_state)

            delta = reward - approximation_function(state=current_state, action=current_action)
            gradient_value = gradient_function(state=current_state, action=current_action)
            self.eligibility_trace = self.gamma * landa * self.eligibility_trace + gradient_value
            alpha = self.alpha_t(current_state_action=current_state_action)
            theta += delta * np.multiply(alpha, self.eligibility_trace)
