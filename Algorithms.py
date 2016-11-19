class Algorithms():
    def __init__(self):
        # The rows represent the player state, and the columns the dealer's state
        self.value_function = np.zeros((21, 10))

        # The policy maps for each state the probability of hit
        self.Environment = Game()
        self.state_value_shape = (21, 10)
        self.state_action_value_shape = (21, 10, 2)
        self.policy = np.random.rand(21, 10)

    @staticmethod
    def coord(vector):
        return int(vector[0]) - 1, int(vector[1]) - 1

    @staticmethod
    def coord_3d(vector):
        return int(vector[0]) - 1, int(vector[1]) - 1, int(vector[2])

    def decision(self, state, policy):
        action = round(np.random.binomial(1, policy[self.coord(state)]))
        return action

    def epsilon_greedy(self, state, state_action_value, policy, epsilon):
        pick = round(np.random.binomial(1, epsilon / 2))
        if pick:
            return self.decision(state, policy)
        else:
            array = state_action_value[int(state[0]) - 1, int(state[1]) - 1, :]
            return array.argmax()

    @staticmethod
    def to_value_function(state_value_function):
        # TODO Check if it really works
        return state_value_function.max(axis=2)


    def run_episode_state_value(self, policy):
        is_terminal = 0
        states_list = []

        current_state = self.Environment.first_step()
        while is_terminal == 0:
            action = self.decision(state=current_state, policy=policy)
            new_state, reward, is_terminal = self.Environment.step(do_hit=action, scores=current_state)
            states_list.append(current_state)
            current_state = new_state

        return states_list, reward

    def run_episode_state_action_value(self, policy, state_action_value, epsilon):
        is_terminal = 0
        states_actions_list = []

        current_state = self.Environment.first_step()
        while is_terminal == 0:
            action = self.epsilon_greedy(state=current_state,
                                         policy=policy,
                                         state_action_value=state_action_value,
                                         epsilon=epsilon)
            new_state, reward, is_terminal = self.Environment.step(do_hit=action,
                                                                   scores=current_state)
            current_state.append(action)
            current_state_action = current_state
            states_actions_list.append(current_state_action)
            current_state = new_state

        return states_actions_list, reward