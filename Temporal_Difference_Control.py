def every_visit_monte_carlo_bis(self, policy, episodes):
    # Variables initialization
    state_visit_count = np.zeros((21, 10))
    state_total_return = np.zeros((21, 10))
    is_terminal = 0

    # Algorithm
    current_state = self.Environment.first_step()

    for i in range(episodes):
        while is_terminal == 0:
            action = self.decision(state=current_state, policy=policy)
            new_state, reward, is_terminal = self.Environment.step(do_hit=action, scores=current_state)
            state_visit_count[self.coord(current_state)] += 1
            state_total_return[self.coord(current_state)] += reward
            current_state = new_state

    # TODO check replacement by 1
    state_visit_count[state_visit_count == 0] = 1
    value_estimation = np.divide(state_total_return, state_visit_count)

    return value_estimation