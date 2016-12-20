import numpy as np


class GameMemory():
    def __init__(self, black_count, red_count):
        assert black_count % 10 == 0 and red_count % 10 == 0
        self.black_count = black_count
        self.red_count = red_count
        self.deck_size = black_count + red_count
        self.full_deck = np.asarray(self.generate_deck(nb_black=self.black_count,
                                                       nb_red=self.red_count))
        self.sampling_permutation = np.random.permutation(np.arange(self.deck_size))
        self.current_card_number = 0
        self.full_memory = np.empty((0, 2))

    def generate_card(self, player_score):
        card_picked_number = self.sampling_permutation[self.current_card_number]
        card_picked = self.full_deck[card_picked_number]
        card_number = card_picked[0]
        card_color = card_picked[1]
        # The card color 1 is black

        player_score += card_color * card_number
        self.full_memory = np.concatenate((self.full_memory, [card_picked]))
        self.current_card_number += 1
        return player_score

    def first_step(self):
        # The first card for the player and the dealer need to be positive
        self.current_card_number = 0
        self.full_memory = np.empty((0, 2))

        while self.full_deck[self.sampling_permutation[0], 1] < 0 or self.full_deck[self.sampling_permutation[1], 1] < 0:
            self.sampling_permutation = np.random.permutation(np.arange(self.deck_size))
        self.current_card_number = 2
        return {'state': [self.full_deck[self.sampling_permutation[0], 0], self.full_deck[self.sampling_permutation[1], 0]],
                'full_memory': self.full_memory}

    def step(self, do_hit, scores):
        reward = 0
        is_terminal = 0
        player_score = scores[0]
        dealer_score = scores[1]

        if self.current_card_number >= self.deck_size:
            is_terminal = 1

        elif do_hit == 1:
            player_score = self.generate_card(player_score=player_score)
            if player_score > 21 or player_score <= 0:
                reward = -1
                is_terminal = 1

        else:

            while 0 < dealer_score < 17 and self.current_card_number < self.deck_size:
                dealer_score = self.generate_card(player_score=dealer_score)

            if dealer_score > 21 or player_score > dealer_score or dealer_score < 1:
                reward = 1
            elif dealer_score == player_score:
                reward = 0
            else:
                reward = -1

            is_terminal = 1

        scores = [player_score, dealer_score]
        full_memory = self.full_memory
        returned_state = {'scores': scores,
                          'full_memory': full_memory}
        return returned_state, reward, is_terminal

    @staticmethod
    def generate_deck(nb_black, nb_red):
        black_deck = [[i % 10 + 1, 1] for i in range(nb_black)]
        red_deck = [[i % 10 + 1, -1] for i in range(nb_red)]
        full_deck = np.append(black_deck, red_deck, axis=0)
        return full_deck

